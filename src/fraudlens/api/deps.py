"""Dependency wiring and the scoring pipeline for the API (Part 13A/13B).

``Scorer`` loads the **frozen** bundle once at startup, verifies every artifact
hash against ``results/freeze.json`` (refusing to start on a mismatch), and exposes
``score`` / ``score_batch``. All feature logic is reached through
``features.serving`` — nothing is re-implemented here.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import Request

from fraudlens.api.history import (
    HistoryProvider,
    get_provider,
    history_backend_from_env,
    summarize_history,
)
from fraudlens.config import load_config
from fraudlens.evaluation import locked
from fraudlens.evaluation.cost_model import (
    ACTION_NAMES,
    APPROVE,
    CostParams,
    expected_cost,
)
from fraudlens.explain.reasons import reasons_for
from fraudlens.features.serving import compute_features_for_transaction
from fraudlens.models import io as model_io
from fraudlens.models.xgb import encode_features
from fraudlens.paths import artifacts_dir, raw_data_dir, results_dir
from fraudlens.risk.policy import amount_aware_actions, enforce_capacity, expected_saving
from fraudlens.risk.scoring import action_name, lift, risk_level, risk_score
from fraudlens.utils import get_logger

logger = get_logger(__name__)

_CAPACITY_NOTE = (
    "enforce_capacity caps only REVIEW within this batch (per IST day); BLOCK and "
    "STEP_UP are chosen by expected cost and are never bounded by the cap. The cap is "
    "not aware of requests outside the batch unless predictions are persisted."
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _jsonable(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so JSON never emits NaN."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def load_entities(profile: str) -> SimpleNamespace:
    """Static reference data ``build_features`` needs: customers + locations."""
    raw = raw_data_dir(profile)
    return SimpleNamespace(
        customers=pd.read_parquet(raw / "customers.parquet"),
        locations=pd.read_parquet(raw / "locations.parquet"),
    )


def load_base_rate(profile: str) -> float:
    """Dev population fraud rate, read from the simulation report (not hard-coded)."""
    report = json.loads((raw_data_dir(profile) / "simulation_report.json").read_text("utf-8"))
    return float(report["fraud_rate"])


def load_freeze(path: str | Path | None = None) -> dict:
    freeze_path = Path(path) if path else results_dir() / "freeze.json"
    return json.loads(freeze_path.read_text(encoding="utf-8"))


def resolve_model_version(profile: str) -> str:
    versions = json.loads((results_dir() / f"models_{profile}.json").read_text(encoding="utf-8"))
    return versions["xgb"]


class Scorer:
    """Holds the frozen bundle + history provider and scores transactions."""

    def __init__(
        self,
        *,
        profile: str = "dev",
        model_version: str | None = None,
        history_backend: str | None = None,
        freeze_path: str | Path | None = None,
        verify_freeze: bool = True,
        write_predictions: bool | None = None,
    ) -> None:
        self.profile = profile
        self.cfg = load_config(profile)
        self.freeze = load_freeze(freeze_path)
        if verify_freeze:
            # Refuse to start on any frozen-artifact sha256 mismatch.
            locked.verify_artifact_hashes(self.freeze)
            logger.info("freeze hashes verified against %s", freeze_path or "results/freeze.json")

        self.model_version = model_version or resolve_model_version(profile)
        bundle = model_io.load_serving_bundle(self.model_version)
        self.model = bundle["model"]
        self.metadata = bundle["metadata"]
        self.calibrator = bundle["calibrator"]
        self.policy = bundle["policy"]
        if self.calibrator is None or self.policy is None:
            raise RuntimeError(
                f"artifact {self.model_version!r} lacks a calibrator or policy.json"
            )

        self.params = CostParams(**self.policy["costs"])
        self.capacity = int(self.policy["capacity_per_day"])
        self.base_rate = load_base_rate(profile)
        self.entities = load_entities(profile)
        self.provider: HistoryProvider = get_provider(
            profile, history_backend or history_backend_from_env()
        )
        self.write_predictions = (
            _env_flag("WRITE_PREDICTIONS") if write_predictions is None else write_predictions
        )
        self._explainer = None
        self.final_results = json.loads(
            (results_dir() / "final_results.json").read_text(encoding="utf-8")
        )
        self.model_size_bytes = (artifacts_dir(self.model_version) / "model.joblib").stat().st_size
        logger.info(
            "scorer ready: profile=%s model=%s backend=%s base_rate=%.6f provider_load=%.2fs",
            profile,
            self.model_version,
            self.provider.backend,
            self.base_rate,
            getattr(self.provider, "load_seconds", 0.0),
        )

    @property
    def explainer(self):
        """Lazily built TreeSHAP explainer (honours ``best_iteration``)."""
        if self._explainer is None:
            from fraudlens.explain.shap_explainer import ShapExplainer

            self._explainer = ShapExplainer(self.model)
        return self._explainer

    def _tx_row(self, req) -> dict[str, Any]:
        return {
            "transaction_id": req.transaction_id or f"live-{uuid4().hex}",
            "customer_id": req.customer_id,
            "timestamp": req.timestamp,
            "amount": float(req.amount),
            "merchant_id": req.merchant_id,
            "merchant_category": req.merchant_category.value if req.merchant_category else None,
            "device_id": req.device_id,
            "location_id": req.location_id,
            "recipient_id": req.recipient_id,
            "transaction_type": req.transaction_type.value,
            "channel": req.channel.value,
        }

    def score(self, req) -> dict[str, Any]:
        """Score one transaction (no capacity — that is a batch concept)."""
        start = time.perf_counter()
        tx_row = self._tx_row(req)

        t0 = time.perf_counter()
        history = self.provider.history(req.customer_id, req.device_id, req.timestamp)
        t1 = time.perf_counter()
        feat = compute_features_for_transaction(tx_row, history, self.entities, self.cfg)
        t2 = time.perf_counter()
        x = encode_features(pd.DataFrame([feat]))
        raw = float(self.model.predict_proba(x)[:, 1][0])
        t3 = time.perf_counter()
        p = float(self.calibrator.predict(np.array([raw]))[0])
        t4 = time.perf_counter()

        p_arr = np.array([p])
        amount_arr = np.array([float(req.amount)])
        action = int(amount_aware_actions(p_arr, amount_arr, self.params)[0])
        t5 = time.perf_counter()

        reasons: list[str] = []
        mitigations: list[str] = []
        shap_rows: list[dict[str, Any]] = []
        if req.include_explanation:
            contributions = self.explainer.explain_local(x)
            explanation = reasons_for(x.iloc[0], contributions, k=5)
            reasons, mitigations = explanation.raises_risk, explanation.lowers_risk
            shap_rows = [_jsonable(c.to_dict()) for c in contributions]
        t6 = time.perf_counter()

        latency_ms = (t6 - start) * 1000.0
        result = {
            "transaction_id": tx_row["transaction_id"],
            "fraud_probability": p,
            "raw_score": raw,
            "risk_score": float(risk_score(p_arr)[0]),
            "base_rate": self.base_rate,
            "lift_over_base_rate": float(lift(p_arr, self.base_rate)[0]),
            "risk_level": str(risk_level(np.array([action]))[0]),
            "recommended_action": str(action_name(np.array([action]))[0]),
            "reasons": reasons,
            "mitigating_factors": mitigations,
            "contributions": shap_rows,
            "model_version": self.model_version,
            "feature_version": self.metadata["feature_version"],
            "policy_version": self.policy["policy_version"],
            "latency_ms": latency_ms,
            "timing_ms": {
                "history_fetch": (t1 - t0) * 1000.0,
                "features": (t2 - t1) * 1000.0,
                "model": (t3 - t2) * 1000.0,
                "calibration": (t4 - t3) * 1000.0,
                "policy": (t5 - t4) * 1000.0,
                "explanation": (t6 - t5) * 1000.0,
            },
        }
        self._maybe_persist(req, result, action)
        return result

    def score_batch(self, requests: list) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Score a batch, apply the per-day REVIEW cap, and summarise."""
        results = [self.score(req) for req in requests]
        probs = np.array([r["fraud_probability"] for r in results])
        amounts = np.array([float(req.amount) for req in requests])
        dates = np.array([req.timestamp.date() for req in requests], dtype=object)

        actions = amount_aware_actions(probs, amounts, self.params)
        capped = enforce_capacity(dates, probs, amounts, actions, self.params, self.capacity)

        approve_cost = expected_cost(APPROVE, probs, amounts, self.params)
        chosen_cost = expected_cost(capped.actions, probs, amounts, self.params)
        avoided = approve_cost - chosen_cost
        saving = expected_saving(probs, amounts, self.params)

        rows = [
            {
                "transaction_id": results[i]["transaction_id"],
                "amount": float(amounts[i]),
                "fraud_probability": float(probs[i]),
                "risk_score": results[i]["risk_score"],
                "risk_level": str(risk_level(np.array([capped.actions[i]]))[0]),
                "recommended_action": ACTION_NAMES[int(capped.actions[i])],
                "expected_saving": float(saving[i]),
            }
            for i in np.argsort(-saving, kind="stable")
        ]
        counts = {
            name: int((capped.actions == code).sum())
            for code, name in enumerate(ACTION_NAMES)
        }
        summary = {
            "n": len(requests),
            "counts_by_action": counts,
            "review_workload": int(capped.n_review),
            "n_demoted": int(capped.n_demoted),
            "total_expected_loss_avoided": float(avoided.sum()),
            "capacity_note": _CAPACITY_NOTE,
        }
        return rows, summary

    def customer_summary(self, customer_id: str, before: Any = None) -> dict[str, Any]:
        """Strictly-prior customer context (typical amount, usual hours, devices, recent).

        ``before`` defaults to now; on the synthetic data that returns the full
        history. Only the customer's own rows are summarised.
        """
        cutoff = pd.Timestamp(before) if before is not None else pd.Timestamp(datetime.now())
        history = self.provider.history(customer_id, None, cutoff)
        return summarize_history(customer_id, history, before=before)

    def _maybe_persist(self, req, result: dict[str, Any], action: int) -> None:
        """Optionally write the score to ``model.predictions`` (off by default)."""
        if not self.write_predictions or not req.transaction_id:
            return
        try:
            from sqlalchemy import text

            from fraudlens.db.loader import get_engine

            with get_engine().begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO model.predictions (transaction_id, model_version, "
                        "raw_probability, calibrated_probability, risk_score, action, "
                        "scored_at, latency_ms) VALUES (:tid, :mv, :raw, :cal, :rs, :act, "
                        ":at, :lat)"
                    ),
                    {
                        "tid": req.transaction_id,
                        "mv": self.model_version,
                        "raw": result["raw_score"],
                        "cal": result["fraud_probability"],
                        "rs": result["risk_score"],
                        "act": ACTION_NAMES[int(action)],
                        "at": datetime.now(),
                        "lat": result["latency_ms"],
                    },
                )
        except Exception as exc:  # noqa: BLE001 - persistence must never break scoring
            logger.warning("prediction persistence failed: %s", exc)


def build_scorer(**kwargs) -> Scorer:
    """Factory used by the app lifespan and the benchmark script."""
    return Scorer(**kwargs)


def get_scorer(request: Request) -> Scorer:
    """FastAPI dependency: the process-wide scorer on ``app.state``."""
    return request.app.state.scorer
