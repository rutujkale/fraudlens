"""Pydantic v2 request/response schemas for the scoring API (Part 13A).

Enums are built from the simulator's single source lists (``archetypes.py``), so a
category added to the data cannot silently drift from the API contract. Timestamps
are normalised to naive IST here; the feature code always sees naive datetimes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraudlens.data_gen.archetypes import CATEGORIES, CHANNELS, TRANSACTION_TYPES

TransactionType = Enum("TransactionType", {v: v for v in TRANSACTION_TYPES}, type=str)
Channel = Enum("Channel", {v: v for v in CHANNELS}, type=str)
MerchantCategory = Enum("MerchantCategory", {v: v for v in CATEGORIES}, type=str)

_IST = ZoneInfo("Asia/Kolkata")


class PredictRequest(BaseModel):
    """One raw transaction to score."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0, description="Transaction amount in INR (> 0).")
    timestamp: datetime = Field(
        ...,
        description="ISO-8601. A timezone-aware value is converted to IST, then stored naive.",
    )
    merchant_category: MerchantCategory | None = Field(
        None, description="NULL for P2P transfers (matches the raw data)."
    )
    merchant_id: str | None = Field(
        None,
        description="Optional. Required for exact parity on new_merchant_flag / merchant_freq_30d.",
    )
    device_id: str | None = None
    location_id: str | None = None
    transaction_type: TransactionType = TransactionType.PURCHASE
    channel: Channel = Channel.APP
    recipient_id: str | None = None
    transaction_id: str | None = None
    include_explanation: bool = False

    @field_validator("timestamp")
    @classmethod
    def _to_ist_naive(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:
            value = value.astimezone(_IST).replace(tzinfo=None)
        return value


class TimingMs(BaseModel):
    """Per-component wall-clock timings for one scored transaction."""

    history_fetch: float
    features: float
    model: float
    calibration: float
    policy: float
    explanation: float


class ShapContribution(BaseModel):
    """One feature's signed SHAP push on the raw margin (log-odds), not the risk score."""

    feature: str
    value: float | None = Field(None, description="Feature value (None when missing).")
    shap_value: float = Field(..., description="Signed SHAP value in margin/log-odds space.")
    direction: str = Field(..., description="'increases' or 'decreases'.")


class PredictResponse(BaseModel):
    """Scoring result for one transaction."""

    transaction_id: str
    fraud_probability: float = Field(..., description="Calibrated probability in [0, 1].")
    raw_score: float = Field(..., description="Uncalibrated XGBoost margin-to-probability score.")
    risk_score: float = Field(..., description="round(100 x calibrated probability, 1).")
    base_rate: float = Field(
        ...,
        description=(
            "Dev population fraud rate (simulation_report.fraud_rate, read at startup). "
            "Used as the denominator of lift_over_base_rate."
        ),
    )
    lift_over_base_rate: float = Field(
        ...,
        description="fraud_probability / base_rate, where base_rate is the dev population rate.",
    )
    risk_level: str = Field(..., description="LOW/ MEDIUM/HIGH/CRITICAL, derived from the action.")
    recommended_action: str = Field(..., description="APPROVE/STEP_UP/REVIEW/BLOCK.")
    reasons: list[str] = Field(default_factory=list, description="Top risk-raising sentences.")
    mitigating_factors: list[str] = Field(
        default_factory=list, description="Top risk-lowering sentences."
    )
    contributions: list[ShapContribution] = Field(
        default_factory=list,
        description="Per-feature SHAP pushes on log-odds (empty unless include_explanation).",
    )
    model_version: str
    feature_version: str
    policy_version: str
    latency_ms: float = Field(..., description="End-to-end server time for this request.")
    timing_ms: TimingMs


class BatchRequest(BaseModel):
    """A batch of transactions (capped at ``common.api.max_batch``)."""

    model_config = ConfigDict(extra="forbid")

    transactions: list[PredictRequest] = Field(..., min_length=1)


class BatchRow(BaseModel):
    """One scored row inside a batch response."""

    transaction_id: str
    amount: float
    fraud_probability: float
    risk_score: float
    risk_level: str
    recommended_action: str
    expected_saving: float = Field(
        ..., description="cost(APPROVE) - cost(chosen action), used for ordering and demotion."
    )


class BatchSummary(BaseModel):
    """Batch-level counts and the capacity outcome."""

    n: int
    counts_by_action: dict[str, int]
    review_workload: int = Field(
        ..., description="Number of REVIEW actions remaining after the per-day capacity cap."
    )
    n_demoted: int = Field(
        ..., description="REVIEW candidates demoted to their next-best action by the cap."
    )
    total_expected_loss_avoided: float = Field(
        ...,
        description="Sum over rows of cost(APPROVE) - cost(chosen action), in INR (expected).",
    )
    capacity_note: str = Field(
        ...,
        description="States that the cap binds REVIEW only; BLOCK and STEP_UP are never capped.",
    )


class BatchResponse(BaseModel):
    """Scored batch, ordered by descending expected saving."""

    rows: list[BatchRow]
    summary: BatchSummary


class HealthResponse(BaseModel):
    status: str
    model_version: str
    history_backend: str
    history_ok: bool


class RecentTransaction(BaseModel):
    """One of a customer's last few strictly-prior transactions."""

    transaction_id: str
    timestamp: datetime
    amount: float
    merchant_category: str | None = None
    channel: str
    transaction_type: str


class CustomerSummaryResponse(BaseModel):
    """Compact customer context for the analyst UI (strictly-prior history only)."""

    customer_id: str
    before: datetime | None = Field(
        None, description="Cutoff used; only rows strictly earlier are summarised."
    )
    n_prior_tx: int
    typical_amount: float | None = Field(
        None, description="Median of the customer's prior amounts."
    )
    usual_hours: list[int] = Field(
        default_factory=list, description="Most common transaction hours."
    )
    known_devices: list[str] = Field(default_factory=list)
    recent: list[RecentTransaction] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    """Frozen-system provenance for the analyst UI and reviewers."""

    profile: str
    profile_note: str
    model_version: str
    feature_version: str
    calibration: str
    policy_version: str
    train_window: dict[str, str]
    val_window: dict[str, str]
    base_rate: float
    base_rate_source: str
    cost_assumptions: dict[str, float]
    capacity_per_day: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, object]
    decision: dict[str, object]
    freeze_hashes: dict[str, object]
    model_size_bytes: int
