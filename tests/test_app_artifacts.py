"""Part 14A: precomputed app artifacts build/load + recompute wrappers.

The fixture builds the ``tiny`` artifacts into a **temp** directory (never
``artifacts/app/``) so the smoke suite needs no committed artifacts and no API.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fraudlens.api.schemas import PredictRequest
from fraudlens.appdata import artifacts, recompute
from fraudlens.config import load_config
from fraudlens.evaluation.cost_model import resolve_cost_params
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.splits import make_temporal_split
from fraudlens.features.registry import FEATURE_COLUMNS
from fraudlens.monitoring.drift import psi_flag

PROFILE = "tiny"


@pytest.fixture(scope="module")
def built_dir(tmp_path_factory) -> tuple:
    base = tmp_path_factory.mktemp("appbase")
    out = artifacts.build_app_artifacts(PROFILE, out_dir=base / PROFILE, n_shap=50)
    return base, out


@pytest.fixture(scope="module")
def loaded(built_dir):
    base, _ = built_dir
    monkey = pytest.MonkeyPatch()
    monkey.setenv("FRAUDLENS_APP_ARTIFACTS_DIR", str(base))
    yield artifacts.load_app_artifacts(PROFILE)
    monkey.undo()


def test_build_writes_into_given_dir_only(built_dir, tmp_path):
    base, out = built_dir
    assert out == base / PROFILE
    for name in ("scored_sample.parquet", "curves.json", "psi_by_month.csv",
                 "presets.json", "manifest.json"):
        assert (out / name).exists(), name
    assert str(out).startswith(str(base))


def test_scored_sample_schema_and_val_only(loaded):
    sample = loaded.scored_sample
    required = {
        "transaction_id", "customer_id", "timestamp", "amount", "archetype",
        "raw_score", "fraud_probability", "risk_score", "recommended_action",
        "risk_level", "top_reason", "is_fraud",
    }
    assert required.issubset(sample.columns)
    assert set(sample["is_fraud"].unique()) <= {0, 1}
    assert not sample.empty

    df, _ = load_feature_dataset(PROFILE)
    split = make_temporal_split(df, load_config(PROFILE))
    t_train_end = pd.Timestamp(split.boundaries["t_train_end"])
    t_val_end = pd.Timestamp(split.boundaries["t_val_end"])
    ts = pd.to_datetime(sample["timestamp"])
    assert (ts >= t_train_end).all() and (ts < t_val_end).all(), "sample left the validation window"


def test_no_raw_feature_columns_leak_into_sample(loaded):
    # `amount` is a raw transaction field (rendered as-is), not a derived feature.
    assert set(FEATURE_COLUMNS).intersection(loaded.scored_sample.columns) <= {"amount"}


def test_manifest_fields(loaded):
    manifest = loaded.manifest
    assert manifest["profile"] == PROFILE
    assert manifest["n_scored_sample"] == len(loaded.scored_sample)
    assert manifest["calibrated"] is False  # tiny has no calibrator
    assert manifest["capacity_per_day"] >= 1


def test_presets_have_history_shape(loaded):
    presets = loaded.presets
    assert 1 <= len(presets) <= 4
    for preset in presets:
        assert set(preset) >= {"label", "story", "expected_action", "transaction"}
        PredictRequest(**preset["transaction"])  # schema-valid


def test_curves_shapes(loaded):
    curves = loaded.curves
    assert curves["models"], "expected at least one validation model"
    for model in curves["models"].values():
        assert len(model["pr"]["recall"]) == len(model["pr"]["precision"])
        assert len(model["roc"]["fpr"]) == len(model["roc"]["tpr"])
        assert model["calibration"], "calibration bins missing"
        assert {"mean_pred", "observed", "count"} <= set(model["calibration"][0])
    assert "test_calibration_bins" in curves


def test_psi_by_month_val_months_only(loaded):
    psi = loaded.psi_by_month
    assert {"month", "feature", "psi", "flag", "n"}.issubset(psi.columns)
    assert psi["feature"].isin(FEATURE_COLUMNS).all()
    assert psi["flag"].isin({psi_flag(v) for v in (0.0, 0.2, 0.5)}).all()
    assert psi["month"].str.match(r"\d{4}-\d{2}").all()

    df, _ = load_feature_dataset(PROFILE)
    split = make_temporal_split(df, load_config(PROFILE))
    val = df.iloc[split.val_idx]
    expected_months = {str(m) for m in val["timestamp"].dt.to_period("M").unique()}
    assert set(psi["month"].unique()) == expected_months


def test_missing_artifacts_names_the_script(tmp_path):
    monkey = pytest.MonkeyPatch()
    monkey.setenv("FRAUDLENS_APP_ARTIFACTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as excinfo:
        artifacts.load_app_artifacts(PROFILE)
    monkey.undo()
    assert "scripts/make_app_artifacts.py" in str(excinfo.value)


def test_recompute_best_thresholds_matches_scan(loaded):
    sample = loaded.scored_sample
    cfg = load_config(PROFILE)
    params = resolve_cost_params(cfg.simulation.costs)
    best = recompute.best_thresholds(sample, params, capacity=10)

    scanned = recompute.scan(sample, params)
    cost_row = scanned.loc[scanned["total_realized_cost"].idxmin()]
    assert best["cost_optimal_threshold"] == pytest.approx(float(cost_row["threshold"]))
    assert best["cost_optimal_realized_cost"] == pytest.approx(
        float(cost_row["total_realized_cost"])
    )


def test_recompute_capacity_caps_review_only(loaded):
    sample = loaded.scored_sample
    cfg = load_config(PROFILE)
    params = resolve_cost_params(cfg.simulation.costs)
    metrics = recompute.policy_metrics(sample, params, capacity=1)
    n_days = len(np.unique(pd.to_datetime(sample["timestamp"]).dt.date))
    assert metrics["review_workload"] <= 1 * n_days
    counts = metrics["counts_by_action"]
    total = counts["APPROVE"] + counts["STEP_UP"] + counts["REVIEW"] + counts["BLOCK"]
    assert total == metrics["n"]


def test_curves_json_is_valid_after_build(built_dir):
    _, out = built_dir
    json.loads((out / "curves.json").read_text(encoding="utf-8"))
