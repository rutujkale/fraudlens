"""XGBoost tests (Part 06F)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import load_config
from fraudlens.features.registry import FEATURE_COLUMNS
from fraudlens.models.xgb import (
    ablation_pr_auc,
    encode_features,
    gain_importances,
    random_search_xgb,
    scale_pos_weight_value,
)


def _features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS})
    df["transaction_type"] = rng.choice(["PURCHASE", "P2P_TRANSFER", "BILL_PAY"], n)
    df["channel"] = rng.choice(["APP", "WEB", "POS", "ATM"], n)
    df["merchant_category"] = rng.choice(["grocery", "fuel", "travel"], n)
    return df


def test_encode_features_is_consistent():
    df = _features(20)
    enc = encode_features(df)
    assert list(enc.columns) == FEATURE_COLUMNS
    assert enc["channel"].dtype == float
    # unknown categories -> NaN, known -> numeric code
    bad = df.copy()
    bad.loc[0, "channel"] = "NOT_A_CHANNEL"
    assert np.isnan(encode_features(bad).loc[0, "channel"])


def test_scale_pos_weight_modes():
    assert scale_pos_weight_value("one", 130.0) == 1.0
    assert scale_pos_weight_value("sqrt", 130.0) == pytest.approx(np.sqrt(130.0))
    assert scale_pos_weight_value("full", 130.0) == 130.0
    with pytest.raises(ValueError):
        scale_pos_weight_value("nope", 130.0)


def test_gain_importances_shares_sum_to_one():
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = (x["f0"] + rng.normal(0, 0.5, 300) > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, tree_method="hist").fit(x, y)
    imp = gain_importances(model)
    assert imp["gain_share"].sum() == pytest.approx(1.0)
    assert imp["gain"].is_monotonic_decreasing


def test_ablation_runs():
    from xgboost import XGBClassifier

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    y = (x["f0"] + rng.normal(0, 0.5, 300) > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, tree_method="hist").fit(x, y)
    ablated = ablation_pr_auc(model, x, y, "f0")
    assert isinstance(ablated, float)


@pytest.mark.slow
def test_random_search_uses_all_discrete_modes():
    cfg = load_config("tiny").simulation.models.xgb
    cfg = cfg.model_copy(update={"n_trials": 5, "n_estimators": 40, "early_stopping_rounds": 5})
    x = encode_features(_features(600))
    y = ((x["tx_count_10m"] + x["amount"]) > 1).astype(int)
    cut = 400
    model, results = random_search_xgb(
        x.iloc[:cut], y[:cut], x.iloc[cut:], y[cut:], cfg, seed=1
    )
    assert len(results) == cfg.n_trials
    assert results["val_pr_auc"].notna().all()
    # every discrete mode (1, sqrt, full) appears among the trials
    neg_pos = (len(y[:cut]) - y[:cut].sum()) / y[:cut].sum()
    got = results["scale_pos_weight"].to_numpy()
    for expected in (1.0, np.sqrt(neg_pos), neg_pos):
        assert np.any(np.isclose(got, expected))
    # and no weight escapes the configured candidate envelope
    assert results["scale_pos_weight"].between(
        cfg.scale_pos_weight_continuous[0], cfg.scale_pos_weight_continuous[1]
    ).all()
    assert model.predict_proba(x.iloc[cut:]).shape[0] == len(y) - cut
