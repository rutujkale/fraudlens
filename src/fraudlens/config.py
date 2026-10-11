"""Configuration loading and validation.

All simulation parameters live in ``config/config.yaml`` under a ``common``
section plus named profiles (tiny/dev/main/drift_lab/unseen_lab). ``load_config``
merges ``common`` with one profile and validates invariant assumptions
(scenario mixes sum to 1, fractions in (0,1)).

The config path can be overridden with the ``FRAUDLENS_CONFIG_PATH`` env var
(used by tests).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

CONFIG_FILENAME = "config.yaml"


class StrictModel(BaseModel):
    """Base model that rejects unknown keys so YAML typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class DriftConfig(StrictModel):
    start_fraction: float | None = None
    strength: float


class SplitsConfig(StrictModel):
    train_end_fraction: float
    val_end_fraction: float


class FeaturesConfig(StrictModel):
    version: str
    windows_minutes: list[int]


class CostConfig(StrictModel):
    fraud_loss_rate: float
    fraud_fixed_fee: float
    review_cost: float
    friction_cost: float
    review_catch_rate: float
    stepup_cost: float
    stepup_friction_cost: float
    stepup_stop_rate: float
    block_friction_cost: float
    capacity_pct_of_daily_volume: float
    analyst_daily_capacity: float | None = None


class CostScenario(StrictModel):
    """Partial overrides applied on top of ``costs`` for a named scenario."""

    friction_cost: float | None = None
    review_cost: float | None = None
    fraud_loss_rate: float | None = None
    fraud_fixed_fee: float | None = None


class BootstrapConfig(StrictModel):
    n: int
    cluster_by: str


class ApiConfig(StrictModel):
    max_batch: int


class ValidationConfig(StrictModel):
    """Data-quality thresholds (Part 03)."""

    max_amount: float
    max_fraud_rate: float


class BoostedConfig(StrictModel):
    """Reference gradient-boosted settings for the separability gate (Part 03)."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float


class SeparabilityGateConfig(StrictModel):
    """Crude-signal gate thresholds, all tunable in one place (Part 03)."""

    max_single_signal_roc_auc: float
    max_single_signal_pr_auc: float
    tree3_vs_boosted_min_pr_auc_gap: float
    min_hard_negative_overlap_signals: int
    boosted: BoostedConfig


class CommonConfig(StrictModel):
    seed: int
    currency: str
    start_date: date
    reporting_threshold: float
    customers_archetype_mix: dict[str, float]
    fraud_scenarios: list[str]
    scenario_mix_period_a: dict[str, float]
    scenario_mix_period_b: dict[str, float]
    stealth_beta_period_a: list[float]
    stealth_beta_period_b: list[float]
    hard_negative_target_share: float
    fraud_device_reuse_prob: float
    family_shared_device_share: float
    splits: SplitsConfig
    features: FeaturesConfig
    costs: CostConfig
    cost_scenarios: dict[str, CostScenario]
    ks: list[float]
    precision_targets: list[float]
    bootstrap: BootstrapConfig
    api: ApiConfig
    validation: ValidationConfig
    separability_gate: SeparabilityGateConfig


class ProfileConfig(StrictModel):
    customers: int
    days: int
    target_fraud_rate: float
    label_delay_days: list[int]
    drift: DriftConfig
    scenario_mix_override: dict[str, float] | None = None


class RawConfig(StrictModel):
    common: CommonConfig
    profiles: dict[str, ProfileConfig]


class SimulationConfig(StrictModel):
    """Merged ``common`` + one profile: everything the pipeline needs."""

    profile: str
    seed: int
    currency: str
    start_date: date
    reporting_threshold: float
    customers: int
    days: int
    target_fraud_rate: float
    label_delay_days: list[int]
    drift: DriftConfig
    scenario_mix_override: dict[str, float] | None = None
    customers_archetype_mix: dict[str, float]
    fraud_scenarios: list[str]
    scenario_mix_period_a: dict[str, float]
    scenario_mix_period_b: dict[str, float]
    stealth_beta_period_a: list[float]
    stealth_beta_period_b: list[float]
    hard_negative_target_share: float
    fraud_device_reuse_prob: float
    family_shared_device_share: float
    splits: SplitsConfig
    features: FeaturesConfig
    costs: CostConfig
    cost_scenarios: dict[str, CostScenario]
    ks: list[float]
    precision_targets: list[float]
    bootstrap: BootstrapConfig
    api: ApiConfig
    validation: ValidationConfig
    separability_gate: SeparabilityGateConfig

    @property
    def embargo_days(self) -> int:
        return max(self.label_delay_days)

    @model_validator(mode="after")
    def _validate_mix_invariants(self) -> SimulationConfig:
        for field in ("scenario_mix_period_a", "scenario_mix_period_b"):
            validate_scenario_mix(getattr(self, field), field)
        if self.scenario_mix_override is not None:
            validate_scenario_mix(self.scenario_mix_override, "scenario_mix_override")
        return self


def validate_scenario_mix(mix: dict[str, float], name: str) -> None:
    """Raise ValueError unless fractions are in (0,1) and sum to 1 (±1e-6)."""
    if any(v <= 0.0 or v >= 1.0 for v in mix.values()):
        bad = {k: v for k, v in mix.items() if v <= 0.0 or v >= 1.0}
        raise ValueError(f"{name}: fractions must be in (0,1), got {bad}")
    total = sum(mix.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name}: does not sum to 1 (±1e-6), got {total}")


def _config_path() -> Path:
    override = os.environ.get("FRAUDLENS_CONFIG_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "config" / CONFIG_FILENAME


def load_config(profile: str = "dev") -> Config:
    """Merge ``common`` with ``profile`` and validate the result."""
    with open(_config_path(), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw_cfg = RawConfig.model_validate(raw)
    if profile not in raw_cfg.profiles:
        raise ValueError(
            f"Unknown profile {profile!r}. Available: {sorted(raw_cfg.profiles)}"
        )
    merged: dict[str, Any] = raw_cfg.common.model_dump(mode="python")
    merged.update(raw_cfg.profiles[profile].model_dump(mode="python", exclude_none=True))
    merged["profile"] = profile
    simulation = SimulationConfig.model_validate(merged)
    return Config(simulation=simulation)


class Config(StrictModel):
    """Loaded configuration: the singular source of truth for a run."""

    simulation: SimulationConfig

    @property
    def embargo_days(self) -> int:
        return self.simulation.embargo_days


def config_hash(config: Config) -> str:
    """Stable sha256 over the canonical JSON of a config, for reproducibility."""
    canonical = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()