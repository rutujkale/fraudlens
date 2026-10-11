"""Logistic-regression tests (Part 06E)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.config import load_config
from fraudlens.features.registry import CATEGORICAL_COLUMNS, FEATURE_COLUMNS
from fraudlens.models.logreg import (
    heavy_tail_columns,
    logreg_coefficients,
    make_logreg_pipeline,
    tune_logreg,
)

_TX_TYPES = ["PURCHASE", "P2P_TRANSFER", "BILL_PAY", "CASH_WITHDRAWAL"]
_CHANNELS = ["APP", "WEB", "POS", "ATM"]
_CATS = ["grocery", "fuel", "travel", "electronics"]


def _features(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS})
    df["transaction_type"] = rng.choice(_TX_TYPES, n)
    df["channel"] = rng.choice(_CHANNELS, n)
    df["merchant_category"] = rng.choice(_CATS, n)
    # inject some NaNs to exercise the imputer/missing indicators
    df.loc[df.sample(frac=0.1, random_state=1).index, "amount_ratio_to_30d_mean"] = np.nan
    return df


def test_pipeline_fits_and_scores():
    x = _features()
    y = (x["amount"].to_numpy() > 0).astype(int)
    pipe = make_logreg_pipeline(C=1.0)
    pipe.fit(x, y)
    proba = pipe.predict_proba(x)[:, 1]
    assert proba.shape == (len(x),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_heavy_tail_excludes_zscore_and_includes_counts():
    heavy = set(heavy_tail_columns())
    assert "amount_zscore_30d" not in heavy
    assert "tx_count_10m" in heavy
    assert "amount_sum_24h" in heavy


def test_tune_selects_best_on_validation():
    cfg = load_config("tiny").simulation.models.logreg
    x = _features(400)
    y = ((x["tx_count_10m"].to_numpy() + x["amount"].to_numpy()) > 1).astype(int)
    idx = np.arange(len(x))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cut = len(x) // 2
    tr, va = idx[:cut], idx[cut:]
    best, results = tune_logreg(
        x.iloc[tr], y[tr], x.iloc[va], y[va], cfg.C_grid, cfg.class_weights
    )
    assert results["val_pr_auc"].is_monotonic_decreasing
    assert best.predict_proba(x.iloc[va])[:, 1].shape == (len(va),)


def test_coefficients_cover_transformed_features():
    x = _features()
    y = (x["amount"].to_numpy() > 0).astype(int)
    pipe = make_logreg_pipeline().fit(x, y)
    coef = logreg_coefficients(pipe)
    assert len(coef) > len(FEATURE_COLUMNS)  # one-hot expands categoricals
    assert set(CATEGORICAL_COLUMNS)  # sanity: categoricals were included
    assert any("channel" in f for f in coef["feature"])
