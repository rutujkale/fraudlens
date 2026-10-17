"""FastAPI scoring service for the frozen FraudLens decision system (Part 13A).

Thin handlers only: history lookup → shared feature code → model → calibrator →
amount-aware policy → optional SHAP. The app refuses to start if any frozen
artifact hash in ``results/freeze.json`` has moved.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated
from uuid import uuid4

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import ValidationError

from fraudlens.api.deps import Scorer, _jsonable, get_scorer
from fraudlens.api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerSummaryResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from fraudlens.utils import get_logger

logger = get_logger(__name__)

ScorerDep = Annotated[Scorer, Depends(get_scorer)]
CsvFile = Annotated[UploadFile, File()]


def _check_batch_size(scorer: Scorer, n: int) -> None:
    max_batch = int(scorer.cfg.simulation.api.max_batch)
    if n > max_batch:
        raise HTTPException(
            status_code=422, detail=f"batch size {n} exceeds max_batch {max_batch}"
        )


def create_app(
    *,
    profile: str | None = None,
    model_version: str | None = None,
    history_backend: str | None = None,
    freeze_path: str | None = None,
    verify_freeze: bool = True,
    write_predictions: bool | None = None,
) -> FastAPI:
    """Build the app; the frozen bundle is loaded once in the lifespan."""
    resolved_profile = profile or os.environ.get("FRAUDLENS_PROFILE", "dev")
    load_kwargs = {
        "profile": resolved_profile,
        "model_version": model_version,
        "history_backend": history_backend,
        "freeze_path": freeze_path,
        "verify_freeze": verify_freeze,
        "write_predictions": write_predictions,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = Scorer(**load_kwargs)
        yield

    app = FastAPI(title="FraudLens scoring API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info("%s %s -> %d (rid=%s)", request.method, request.url.path,
                    response.status_code, request_id)
        return response

    @app.get("/health", response_model=HealthResponse)
    def health(scorer: ScorerDep) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=scorer.model_version,
            history_backend=scorer.provider.backend,
            history_ok=scorer.provider.ok(),
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(scorer: ScorerDep) -> ModelInfoResponse:
        results = scorer.final_results
        model_metrics = {
            row["model"]: {
                key: row.get(key)
                for key in ("pr_auc", "pr_auc_lo", "pr_auc_hi", "recall_at_top_0.01", "fpr_at_thr")
            }
            for row in results.get("model_comparison", [])
        }
        decision = {
            policy["policy"]: {
                "total_realized_cost": policy.get("total_realized_cost"),
                "recall": policy.get("recall"),
                "feasible": policy.get("feasible"),
            }
            for policy in results.get("decision", {}).get("policies", [])
        }
        freeze = scorer.freeze
        return ModelInfoResponse(
            profile=scorer.profile,
            profile_note=(
                "Frozen system is the dev profile, NOT main (main was never generated); "
                "no metric here implies production readiness."
            ),
            model_version=scorer.model_version,
            feature_version=scorer.metadata["feature_version"],
            calibration=scorer.policy["selected_calibrator"],
            policy_version=scorer.policy["policy_version"],
            train_window=scorer.metadata["train_window"],
            val_window=scorer.metadata["val_window"],
            base_rate=scorer.base_rate,
            base_rate_source="simulation_report.fraud_rate (dev population rate)",
            cost_assumptions=scorer.policy["costs"],
            capacity_per_day=scorer.capacity,
            val_metrics=scorer.metadata["metrics"],
            test_metrics=_jsonable({"model_comparison": model_metrics}),
            decision=_jsonable(decision),
            freeze_hashes={
                "freeze_git_commit": freeze.get("git_commit"),
                "created_at": freeze.get("created_at"),
                "config_hash": freeze.get("config_hash"),
                "dataset_version": freeze.get("dataset_version"),
                "models": {
                    name: info.get("model_version")
                    for name, info in freeze.get("models", {}).items()
                },
            },
            model_size_bytes=scorer.model_size_bytes,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, scorer: ScorerDep) -> PredictResponse:
        return PredictResponse(**scorer.score(payload))

    @app.get("/customer/{customer_id}/summary", response_model=CustomerSummaryResponse)
    def customer_summary(
        customer_id: str, scorer: ScorerDep, before: datetime | None = None
    ) -> CustomerSummaryResponse:
        return CustomerSummaryResponse(**scorer.customer_summary(customer_id, before))

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(payload: BatchRequest, scorer: ScorerDep) -> BatchResponse:
        _check_batch_size(scorer, len(payload.transactions))
        rows, summary = scorer.score_batch(payload.transactions)
        return BatchResponse(rows=rows, summary=summary)

    @app.post("/predict/batch/csv", response_model=BatchResponse)
    def predict_batch_csv(
        file: CsvFile, scorer: ScorerDep
    ) -> BatchResponse:
        frame = pd.read_csv(file.file)
        frame = frame.dropna(axis=1, how="all")
        records = frame.where(pd.notna(frame), None).to_dict("records")
        try:
            requests = [PredictRequest(**record) for record in records]
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        _check_batch_size(scorer, len(requests))
        rows, summary = scorer.score_batch(requests)
        return BatchResponse(rows=rows, summary=summary)

    return app


app = create_app()
