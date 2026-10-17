"""Build and load the precomputed artifacts the Streamlit app reads (Part 14).

The app must contain **no** feature, scoring or cost logic, so everything it shows
offline is materialised here under ``artifacts/app/<profile>/`` (gitignored):

* ``scored_sample.parquet`` — the ``val_thr`` slice (later half of validation, the
  calibrator-fit slice excluded) with raw/calibrated score, action, risk level,
  archetype, precomputed top reason for the highest-risk rows, and the true label
  (rendered only behind an explicit "show labels" toggle).
* ``curves.json`` — validation PR/ROC/calibration/recall@K points per model, the
  frozen test calibration bins, and the validation score histogram.
* ``psi_by_month.csv`` — per-feature PSI of each validation month against train.
* ``presets.json`` — four fixed-seed example transactions (with history).

Every value comes from data or files; nothing here hard-codes a metric. The test
window is never read (``val_thr`` only).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from fraudlens.config import load_config
from fraudlens.evaluation.cost_model import CostParams, resolve_cost_params
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.metrics import recall_at_top_k
from fraudlens.evaluation.splits import chronological_fraction_split, make_temporal_split
from fraudlens.evaluation.threshold_search import daily_capacity
from fraudlens.explain.reasons import reasons_for
from fraudlens.explain.shap_explainer import Contribution, ShapExplainer
from fraudlens.features.registry import FEATURE_COLUMNS
from fraudlens.models import io as model_io
from fraudlens.models.xgb import encode_features
from fraudlens.monitoring.metrics import psi_by_period
from fraudlens.paths import artifacts_dir, raw_data_dir, results_dir
from fraudlens.risk.policy import amount_aware_actions, enforce_capacity
from fraudlens.risk.scoring import action_name, risk_level, risk_score
from fraudlens.utils import get_logger

logger = get_logger(__name__)

SEED = 42
N_SCORED = 100_000
N_SHAP = 5_000
MODELS_FOR_CURVES = ("xgb", "logreg")

MAKE_SCRIPT = "scripts/make_app_artifacts.py"


@dataclass
class AppArtifacts:
    """Everything an offline page needs, loaded once and cached by the app."""

    directory: Path
    scored_sample: pd.DataFrame
    curves: dict[str, Any]
    psi_by_month: pd.DataFrame
    presets: list[dict[str, Any]]
    manifest: dict[str, Any]


def app_artifacts_dir(profile: str) -> Path:
    """``artifacts/app/<profile>/`` (overridable for smoke tests)."""
    override = os.environ.get("FRAUDLENS_APP_ARTIFACTS_DIR")
    base = Path(override) if override else artifacts_dir("app")
    return base / profile


def _resolve_model_version(profile: str) -> str:
    versions = json.loads((results_dir() / f"models_{profile}.json").read_text("utf-8"))
    return versions["xgb"]


def _sample_positions(y: np.ndarray, n_scored: int, seed: int) -> np.ndarray:
    """All frauds (<= n_scored) plus a seeded legit fill; sorted positions."""
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    room = max(n_scored - len(pos), 0)
    if len(neg) > room:
        neg = np.sort(rng.choice(neg, size=room, replace=False))
    elif len(pos) > n_scored:
        pos = np.sort(rng.choice(pos, size=n_scored, replace=False))
    return np.sort(np.concatenate([pos, neg]))


def _top_raise_reasons(explainer: ShapExplainer, x: pd.DataFrame, k: int = 1) -> list[str]:
    """First risk-raising sentence per row from TreeSHAP margin contributions."""
    values = explainer.shap_values(x)
    out: list[str] = []
    for i in range(len(x)):
        row = x.iloc[i]
        contributions = [
            Contribution(
                feature=feature,
                value=float(row[feature]) if pd.notna(row[feature]) else float("nan"),
                shap_value=float(values[i][j]),
                direction="increases" if values[i][j] >= 0 else "decreases",
            )
            for j, feature in enumerate(explainer.feature_names)
        ]
        contributions.sort(key=lambda c: (-abs(c.shap_value), c.feature))
        reasons = reasons_for(row, contributions, k=k)
        out.append(reasons.raises_risk[0] if reasons.raises_risk else "")
    return out


def _curve_points(
    y: np.ndarray, score: np.ndarray, max_points: int = 200
) -> dict[str, list[float]]:
    """PR + ROC curve points, thinned to ``max_points`` for JSON size."""
    precision, recall, _ = precision_recall_curve(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    pr_idx = np.unique(
        np.linspace(0, len(precision) - 1, min(max_points, len(precision))).astype(int)
    )
    roc_idx = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).astype(int))
    return {
        "pr": {
            "recall": [float(r) for r in recall[pr_idx]],
            "precision": [float(p) for p in precision[pr_idx]],
        },
        "roc": {
            "fpr": [float(v) for v in fpr[roc_idx]],
            "tpr": [float(v) for v in tpr[roc_idx]],
        },
    }


def _calibration_bins(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-frequency calibration bins: mean prediction, observed rate, count."""
    order = np.argsort(prob, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    for chunk in bins:
        if len(chunk) == 0:
            continue
        rows.append(
            {
                "mean_pred": float(prob[chunk].mean()),
                "observed": float(y[chunk].mean()),
                "count": int(len(chunk)),
            }
        )
    return rows


def _finite(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so ``curves.json`` is valid JSON."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _build_curves(profile: str, cfg, val_thr: pd.DataFrame, y_thr: np.ndarray) -> dict[str, Any]:
    models = json.loads((results_dir() / f"models_{profile}.json").read_text("utf-8"))
    out: dict[str, Any] = {
        "split": "val_thr (later half of the validation window; calibrator-fit slice excluded)",
        "ks": [float(k) for k in cfg.simulation.ks],
        "models": {},
    }
    for name in MODELS_FOR_CURVES:
        if name not in models:
            continue
        version = models[name]
        scores_path = artifacts_dir(version) / "val_scores.parquet"
        if not scores_path.exists():
            continue
        bundle = model_io.load_serving_bundle(version)
        scores = pd.read_parquet(scores_path).set_index("transaction_id")
        aligned = val_thr[["transaction_id"]].merge(
            scores["raw_score"].rename("raw_score"),
            left_on="transaction_id",
            right_index=True,
            how="left",
        )
        raw = aligned["raw_score"].to_numpy(dtype=float)
        calibrator = bundle["calibrator"]
        prob = calibrator.predict(raw) if calibrator is not None else raw
        calibrated = calibrator is not None
        out["models"][name] = {
            "model_version": version,
            "calibrated": bool(calibrated),
            **_curve_points(y_thr, prob),
            "calibration": _calibration_bins(y_thr, prob),
            "recall_at_k": {
                f"{k:g}": float(recall_at_top_k(y_thr, prob, k)) for k in cfg.simulation.ks
            },
        }

    final = json.loads((results_dir() / "final_results.json").read_text("utf-8"))
    out["test_calibration_bins"] = final.get("calibration", {})
    return out


def _build_psi_by_month(cfg, train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Per-feature PSI of each validation month against the train window.

    Delegates to :func:`fraudlens.monitoring.metrics.psi_by_period` so the app and
    the Part 15 monitoring layer share one PSI implementation.
    """
    features = [
        c for c in FEATURE_COLUMNS
        if c in train.columns and pd.api.types.is_numeric_dtype(train[c])
    ]
    current = val.copy()
    current["month"] = current["timestamp"].dt.strftime("%Y-%m")
    return psi_by_period(train, current, features, period_col="month").rename(
        columns={"period": "month"}
    )


def _opt(value: Any) -> str | None:
    """``None`` for missing scalars, else ``str`` (keeps JSON clean of ``NaN``)."""
    return None if value is None or pd.isna(value) else str(value)


def _presets(sample: pd.DataFrame, raw: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    """Four fixed-seed example transactions, one per demo story, with history."""
    rng = np.random.default_rng(seed)
    raw_by_id = raw.set_index("transaction_id")

    def one(mask: pd.Series, label: str, story: str, *, highest_risk: bool = False) -> dict | None:
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            return None
        if highest_risk:
            row = sample.iloc[idx[np.argmax(sample["fraud_probability"].to_numpy()[idx])]]
        else:
            row = sample.iloc[int(rng.choice(idx))]
        tx = raw_by_id.loc[row["transaction_id"]].to_dict()
        return {
            "label": label,
            "story": story,
            "transaction_id": row["transaction_id"],
            "expected_action": row["recommended_action"],
            "transaction": {
                "customer_id": str(tx["customer_id"]),
                "amount": float(tx["amount"]),
                "timestamp": str(pd.Timestamp(tx["timestamp"])),
                "merchant_category": _opt(tx["merchant_category"]),
                "merchant_id": _opt(tx["merchant_id"]),
                "device_id": _opt(tx["device_id"]),
                "location_id": _opt(tx["location_id"]),
                "transaction_type": str(tx["transaction_type"]),
                "channel": str(tx["channel"]),
                "recipient_id": _opt(tx["recipient_id"]),
                "include_explanation": True,
            },
        }

    candidates = [
        one(
            (sample["is_fraud"] == 0) & (sample["recommended_action"] == "APPROVE"),
            "Normal purchase",
            "A routine, low-risk transaction the system approves.",
        ),
        one(
            (sample["is_fraud"] == 1) & (sample["fraud_scenario"] == "burst"),
            "Suspicious burst",
            "A fraud burst: several transactions in a short window.",
            highest_risk=True,
        ),
        one(
            (sample["hard_negative_type"] == "travel_trip"),
            "Hard negative: travel",
            "A legitimate trip overseas that looks unusual for the card.",
            highest_risk=True,
        ),
        one(
            (sample["fraud_scenario"] == "structuring"),
            "Structuring-like",
            "Amounts just below the reporting threshold, repeatedly.",
            highest_risk=True,
        ),
    ]
    return [c for c in candidates if c is not None]


def _cost_params_and_capacity(
    cfg, policy: dict | None, dates: np.ndarray
) -> tuple[CostParams, int]:
    if policy is not None:
        return CostParams(**policy["costs"]), int(policy["capacity_per_day"])
    return resolve_cost_params(cfg.simulation.costs), daily_capacity(dates, cfg)[0]


def build_app_artifacts(
    profile: str,
    *,
    out_dir: Path | None = None,
    seed: int = SEED,
    n_scored: int = N_SCORED,
    n_shap: int = N_SHAP,
) -> Path:
    """Build every app artifact for ``profile`` and return the output directory.

    ``out_dir`` overrides the default ``artifacts/app/<profile>`` (used by the
    smoke test to build ``tiny`` into a temp dir; never committed).
    """
    cfg = load_config(profile)
    model_version = _resolve_model_version(profile)
    bundle = model_io.load_serving_bundle(model_version)
    calibrator, policy = bundle["calibrator"], bundle["policy"]
    if calibrator is None or policy is None:
        logger.warning("%s has no calibrator/policy; smoke mode uses raw scores as probabilities",
                       model_version)

    df, labels = load_feature_dataset(profile)
    split = make_temporal_split(df, cfg)
    val = df.iloc[split.val_idx].reset_index(drop=True)
    train = df.iloc[split.train_idx]
    _, val_thr_idx = chronological_fraction_split(
        split.val_idx, cfg.simulation.calibration.cal_fraction
    )
    val_thr = df.iloc[val_thr_idx].reset_index(drop=True)

    labels_min = labels[["transaction_id", "is_fraud", "fraud_scenario", "hard_negative_type"]]
    val_thr = val_thr.merge(labels_min, on="transaction_id", how="left")

    scores = pd.read_parquet(artifacts_dir(model_version) / "val_scores.parquet").set_index(
        "transaction_id"
    )
    aligned = val_thr[["transaction_id"]].merge(
        scores["raw_score"].rename("raw_score"),
        left_on="transaction_id",
        right_index=True,
        how="left",
    )
    if aligned["raw_score"].isna().any():
        missing = int(aligned["raw_score"].isna().sum())
        raise RuntimeError(
            f"{missing} val_thr rows missing from {model_version} val_scores.parquet"
        )

    raw = aligned["raw_score"].to_numpy(dtype=float)
    prob = calibrator.predict(raw) if calibrator is not None else raw
    y_thr = val_thr["is_fraud"].to_numpy()

    dates = np.array(val_thr["timestamp"].dt.date.tolist(), dtype=object)
    params, capacity = _cost_params_and_capacity(cfg, policy, dates)
    amounts = val_thr["amount"].to_numpy(dtype=float)
    actions = amount_aware_actions(prob, amounts, params)
    capped = enforce_capacity(dates, prob, amounts, actions, params, capacity)

    sample = val_thr.copy()
    sample["raw_score"] = raw
    sample["fraud_probability"] = prob
    sample["risk_score"] = risk_score(prob)
    sample["recommended_action"] = action_name(capped.actions)
    sample["risk_level"] = risk_level(capped.actions)
    customers = pd.read_parquet(raw_data_dir(profile) / "customers.parquet")[
        ["customer_id", "archetype"]
    ]
    sample = sample.merge(customers, on="customer_id", how="left")

    keep = _sample_positions(y_thr, n_scored, seed)
    sample = sample.iloc[keep].reset_index(drop=True)

    top_reasons = {tid: "" for tid in sample["transaction_id"]}
    explainer = ShapExplainer(bundle["model"])
    n_explain = min(n_shap, len(sample))
    explain_pos = np.argsort(-sample["fraud_probability"].to_numpy(), kind="stable")[:n_explain]
    x_encoded = encode_features(sample.iloc[explain_pos])
    for tid, reason in zip(
        sample["transaction_id"].to_numpy()[explain_pos],
        _top_raise_reasons(explainer, x_encoded),
        strict=True,
    ):
        top_reasons[tid] = reason
    sample["top_reason"] = sample["transaction_id"].map(top_reasons)

    out = Path(out_dir) if out_dir else app_artifacts_dir(profile)
    out.mkdir(parents=True, exist_ok=True)

    columns = [
        "transaction_id", "customer_id", "timestamp", "amount", "archetype",
        "raw_score", "fraud_probability", "risk_score", "recommended_action",
        "risk_level", "top_reason", "is_fraud", "fraud_scenario", "hard_negative_type",
    ]
    sample[columns].to_parquet(out / "scored_sample.parquet", index=False)

    curves = _build_curves(profile, cfg, val_thr, y_thr)
    hist, _ = np.histogram(sample["fraud_probability"], bins=np.linspace(0, 1, 21))
    curves["val_score_hist"] = {
        "bin_edges": [float(v) for v in np.linspace(0.0, 1.0, 21)],
        "counts": [int(v) for v in hist],
    }
    (out / "curves.json").write_text(json.dumps(_finite(curves), indent=2), encoding="utf-8")

    psi_table = _build_psi_by_month(cfg, train, val)
    psi_table.to_csv(out / "psi_by_month.csv", index=False)

    raw_tx = pd.read_parquet(raw_data_dir(profile) / "transactions.parquet")
    presets = _presets(sample, raw_tx, seed)
    (out / "presets.json").write_text(json.dumps(presets, indent=2), encoding="utf-8")

    manifest = {
        "profile": profile,
        "model_version": model_version,
        "policy_version": policy["policy_version"] if policy else None,
        "calibrated": calibrator is not None,
        "capacity_per_day": int(capacity),
        "costs": (policy["costs"] if policy else params.__dict__),
        "windows": {
            "t_train_end": split.boundaries["t_train_end"],
            "t_val_end": split.boundaries["t_val_end"],
        },
        "n_scored_sample": int(len(sample)),
        "n_shap_explained": int(n_explain),
        "n_val_thr": int(len(val_thr)),
        "seed": seed,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("app artifacts written to %s (%d sample rows)", out, len(sample))
    return out


def load_app_artifacts(profile: str) -> AppArtifacts:
    """Load the precomputed artifacts, with an error naming the build script."""
    directory = app_artifacts_dir(profile)
    required = [
        "scored_sample.parquet", "curves.json", "psi_by_month.csv", "presets.json", "manifest.json"
    ]
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"app artifacts missing for profile {profile!r} in {directory}: {missing}. "
            f"Run `python {MAKE_SCRIPT} --profile {profile}`."
        )
    return AppArtifacts(
        directory=directory,
        scored_sample=pd.read_parquet(directory / "scored_sample.parquet"),
        curves=json.loads((directory / "curves.json").read_text(encoding="utf-8")),
        psi_by_month=pd.read_csv(directory / "psi_by_month.csv"),
        presets=json.loads((directory / "presets.json").read_text(encoding="utf-8")),
        manifest=json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
    )
