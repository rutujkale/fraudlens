"""Part 10 tests: SHAP wrapper, reason templates, gallery rules, fairness audit.

Model-backed tests load the dev artifact plus a small validation sample; the rest use
synthetic frames so they stay fast and independent of the database.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from fraudlens.evaluation.cost_model import APPROVE, BLOCK, REVIEW, STEP_UP
from fraudlens.evaluation.experiments_lib import load_supervised
from fraudlens.explain.fairness import (
    AUDIT_ATTRIBUTES,
    age_band,
    fpr_metric,
    group_table,
    load_audit_attributes,
    proxy_check,
    recall_metric,
)
from fraudlens.explain.gallery import select_cases
from fraudlens.explain.reasons import fallback_reason, format_reason, reasons_for
from fraudlens.explain.shap_explainer import Contribution, ShapExplainer, stratified_val_sample
from fraudlens.features.registry import FEATURE_COLUMNS
from fraudlens.models import io as model_io
from fraudlens.paths import results_dir
from fraudlens.utils import set_global_seed

PROFILE = "dev"


@lru_cache(maxsize=1)
def _dev_setup():
    """(explainer, x_sample, y_sample) from the dev artifact — loaded once."""
    set_global_seed(42)
    sup = load_supervised(PROFILE)
    bundle = model_io.load_serving_bundle("xgb_20261011_v1")
    explainer = ShapExplainer(bundle["model"])
    y_val = sup.y_val()
    sample = stratified_val_sample(y_val, max_frauds=200, max_legit=400, seed=42)
    return explainer, sup.x_val().iloc[sample].reset_index(drop=True), y_val[sample]


def _gallery_frame() -> pd.DataFrame:
    """Synthetic val_thr frame with one clean row per gallery rule."""
    n = 12
    frame = pd.DataFrame(
        {
            "is_fraud": [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            "p": [0.99, 0.5, 0.4, 0.9, 0.2, 0.1, 0.7, 0.3, 0.997, 0.6, 0.95, 0.05],
            "action": [
                BLOCK, APPROVE, APPROVE, STEP_UP, APPROVE, APPROVE,
                REVIEW, APPROVE, BLOCK, STEP_UP, REVIEW, APPROVE,
            ],
            "amount": [100.0, 200.0, 300.0, 400.0, 50_000.0, 500.0, 600.0, 700.0,
                       800.0, 900.0, 1000.0, 1100.0],
            "stealth": [0.1, 0.95, 0.99, np.nan, np.nan, np.nan, np.nan, np.nan,
                        0.99, 0.2, np.nan, np.nan],
            "travel_speed_kmh": [0.0] * 3 + [5000.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "history_days": [50.0, 60.0, 70.0, 80.0, 90.0, 1.5, 1.9, -5.0, 55.0, 65.0, 75.0, 85.0],
            "transaction_id": [f"T{i:04d}" for i in range(n)],
        }
    )
    return frame


def test_gallery_returns_every_case_once():
    cases = select_cases(_gallery_frame())
    assert [c.case for c in cases] == [
        "clear_tp",
        "subtle_tp",
        "false_negative",
        "hard_negative_fp",
        "high_amount_tn",
        "cold_start",
    ]
    assert all(0 <= c.position < 12 for c in cases)
    assert all(c.pool_size >= 1 for c in cases)


def test_gallery_rules_pick_the_intended_row():
    frame = _gallery_frame()
    cases = {c.case: c for c in select_cases(frame)}
    assert frame.iloc[cases["clear_tp"].position]["p"] == frame["p"].max()
    assert frame.iloc[cases["false_negative"].position]["stealth"] == 0.99
    assert frame.iloc[cases["hard_negative_fp"].position]["travel_speed_kmh"] == 5000.0
    assert frame.iloc[cases["high_amount_tn"].position]["amount"] == 50_000.0
    assert frame.iloc[cases["cold_start"].position]["history_days"] == 1.9

    # subtle_tp: the strongest (stealth, p) row inside the pre-registered pool
    flagged_fraud = (frame["is_fraud"] == 1) & (frame["action"] != APPROVE)
    stealth_cut = np.nanpercentile(frame.loc[frame["is_fraud"] == 1, "stealth"], 75)
    pool = frame[flagged_fraud & (frame["stealth"] >= stealth_cut)]
    best = pool.sort_values(["stealth", "p"], ascending=False).index[0]
    assert cases["subtle_tp"].position == best
    assert cases["subtle_tp"].pool_size == len(pool)


def test_gallery_cold_start_excludes_negative_history():
    frame = _gallery_frame()
    cold = next(c for c in select_cases(frame) if c.case == "cold_start")
    history = frame["history_days"].to_numpy()
    assert "0 <= history_days" in cold.rule
    assert 0 <= frame.iloc[cold.position]["history_days"] < 2
    assert cold.pool_size == int(((history >= 0) & (history < 2)).sum())


def test_shap_additivity_and_best_iteration():
    explainer, x_sample, _ = _dev_setup()
    assert explainer._model.best_iteration is not None
    assert explainer.check_additivity(x_sample) < 1e-3


def test_shap_values_deterministic_and_nan_safe():
    explainer, x_sample, _ = _dev_setup()
    first = explainer.shap_values(x_sample)
    second = explainer.shap_values(x_sample)
    assert first.shape == (len(x_sample), len(FEATURE_COLUMNS))
    assert np.allclose(first, second, equal_nan=True)
    assert np.all(np.isfinite(first[~np.isnan(x_sample.to_numpy(dtype=float))]))


def test_explain_global_and_local_shapes():
    explainer, x_sample, _ = _dev_setup()
    global_df = explainer.explain_global(x_sample)
    assert list(global_df.columns) == ["rank", "feature", "mean_abs_shap", "mean_shap"]
    assert global_df["rank"].tolist() == list(range(1, len(FEATURE_COLUMNS) + 1))
    assert global_df["mean_abs_shap"].is_monotonic_decreasing

    contributions = explainer.explain_local(x_sample.iloc[[0]])
    assert len(contributions) == len(FEATURE_COLUMNS)
    assert abs(contributions[0].shap_value) >= abs(contributions[-1].shap_value)


def test_top_drivers_split_by_sign():
    explainer, x_sample, _ = _dev_setup()
    raising, lowering = explainer.top_drivers(x_sample.iloc[[0]], k=3)
    assert all(c.shap_value > 0 for c in raising)
    assert all(c.shap_value < 0 for c in lowering)


def test_reason_templates_cover_every_feature():
    for feature in FEATURE_COLUMNS:
        sentence = format_reason(feature, 5.0)
        assert sentence
        assert sentence != fallback_reason(feature, 5.0), feature
    assert format_reason("totally_unknown_feature", 5.0) == "totally unknown feature = 5"
    assert "encoded category code" in format_reason("merchant_category", 3.0)


def test_reasons_are_nan_safe_and_deduped():
    assert "no prior history" in format_reason("tx_from_device_prior", float("nan"))
    contributions = [
        Contribution("amount", 100.0, 0.5, "increases"),
        Contribution("log_amount", 4.615, 0.4, "increases"),
        Contribution("minutes_since_prev_tx", 3.0, -0.3, "decreases"),
        Contribution("hour", 3.0, -0.2, "decreases"),
    ]
    row = pd.Series({"amount": 100.0, "log_amount": np.log1p(100.0)})
    reasons = reasons_for(row, contributions, k=5)
    assert len(reasons.raises_risk) == 1
    assert len(reasons.lowers_risk) == 2


def test_noise_controls_are_not_relied_on():
    path = results_dir() / "exp_8D_importance_dev.csv"
    if not path.exists():
        pytest.skip("8D evidence table not present")
    table = pd.read_csv(path).set_index("feature")
    for control in ("_random_noise", "_shuffled_minutes_since_prev_tx"):
        assert table.loc[control, "perm_pr_auc_drop"] <= 0
    assert table.loc["_random_noise", "mean_abs_shap"] > 0


def test_fairness_attributes_are_not_model_inputs():
    assert not set(AUDIT_ATTRIBUTES) & set(FEATURE_COLUMNS)


def test_no_label_fields_in_features_or_explanations():
    from fraudlens.features.registry import FORBIDDEN_COLUMNS

    assert not set(FORBIDDEN_COLUMNS) & set(FEATURE_COLUMNS)
    for field in ("is_fraud", "fraud_scenario", "episode_id", "stealth", "hard_negative_type"):
        assert format_reason(field, 1.0) == fallback_reason(field, 1.0)
        assert field not in FEATURE_COLUMNS


def test_age_bands_are_contiguous():
    assert age_band(17) == "unknown"
    assert age_band(18) == "<25"
    assert age_band(24.9) == "<25"
    assert age_band(25) == "25-34"
    assert age_band(80) == "65+"
    assert age_band(float("nan")) == "unknown"


def _fairness_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for group, n, p_fraud in (("regular", 1500, 0.05), ("ribbon", 400, 0.04)):
        for i in range(n):
            y = int(rng.random() < p_fraud)
            action = int(rng.choice([APPROVE, STEP_UP, REVIEW, BLOCK], p=[0.95, 0.02, 0.02, 0.01]))
            rows.append(
                {
                    "customer_id": f"C{group[:2]}-{i // 10:04d}",
                    "y": y,
                    "p": float(rng.random()),
                    "action": action,
                    "archetype": group,
                    "age_band": "<25",
                    "income_band": "low",
                    "home_region_group": "North",
                }
            )
    return pd.DataFrame(rows)


def test_fairness_suppression_and_dual_fpr():
    table = group_table(_fairness_frame(), calibration_metrics=True, n_boot=50, seed=1)
    metrics = table.metrics
    archetype = metrics[metrics["attribute"] == "archetype"]
    assert set(archetype["group"]) == {"regular", "ribbon"}
    small = archetype.loc[archetype["group"] == "ribbon"].iloc[0]
    assert "positives=" in small["suppressed_reason"]
    assert np.isnan(small["recall"])
    assert (metrics["fpr_any"] >= metrics["fpr_strict"]).all()
    counts = metrics[["n_approve", "n_step_up", "n_review", "n_block"]].sum(axis=1)
    assert (counts == metrics["n"]).all()
    supp = table.suppression.set_index("attribute")
    assert supp.loc["archetype", "n_groups"] == 2
    assert set(supp.index) == set(AUDIT_ATTRIBUTES)


def test_fpr_metrics_use_action_definition():
    y = np.array([0, 0, 1, 1])
    actions = np.array([APPROVE, STEP_UP, REVIEW, BLOCK])
    assert fpr_metric(y, actions) == pytest.approx(0.5)  # 1 of 2 legit rows flagged
    assert recall_metric(y, actions) == pytest.approx(1.0)


def test_proxy_check_flags_strong_association():
    rng = np.random.default_rng(3)
    band = np.repeat(["low", "mid", "high", "mid_high"], 250)
    rank = {"low": 0, "mid": 1, "mid_high": 2, "high": 3}
    banded = np.array([rank[b] for b in band], dtype=float) * 10
    features = pd.DataFrame(
        {
            "amount": banded + rng.normal(0, 0.01, 1000),
            "noise": rng.normal(0, 1, 1000),
        }
    )
    audit = pd.DataFrame(
        {
            "age_band": "<25",
            "income_band": band,
            "archetype": "regular",
            "home_region_group": "North",
        }
    )
    out = proxy_check(features, audit)
    amount_row = out[(out["feature"] == "amount") & (out["attribute"] == "income_band")].iloc[0]
    assert amount_row["statistic"] == "spearman"
    assert amount_row["strong"]
    noise_row = out[(out["feature"] == "noise") & (out["attribute"] == "income_band")].iloc[0]
    assert not noise_row["strong"]


def test_audit_attributes_load_for_dev():
    audit = load_audit_attributes(PROFILE)
    assert list(audit.columns) == ["customer_id", *AUDIT_ATTRIBUTES]
    assert audit["home_region_group"].notna().any()
    assert audit["age_band"].isin(["<25", "25-34", "35-49", "50-64", "65+", "unknown"]).all()


def test_shap_latency_budget():
    path = results_dir() / f"shap_latency_{PROFILE}.json"
    if not path.exists():
        pytest.skip("latency json not present")
    latency = json.loads(path.read_text(encoding="utf-8"))
    assert latency["p50_ms"] < 50
    assert latency["p95_ms"] < 100
