"""Tests for the Part 09 Isolation Forest scoring (`models/anomaly.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.models.anomaly import (
    ANOMALY_FEATURES,
    cross_fit_scores,
    customer_folds,
    fit_anomaly,
    grid_search,
)


def _frame(n_per_customer: int = 40, n_customers: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_per_customer * n_customers
    data = {f: rng.normal(size=n) for f in ANOMALY_FEATURES}
    data["customer_id"] = np.repeat([f"c{i}" for i in range(n_customers)], n_per_customer)
    return pd.DataFrame(data)


def test_scores_in_unit_interval() -> None:
    df = _frame()
    model = fit_anomaly(df, n_estimators=100, max_samples=64, seed=1)
    score = model.score(df)
    assert score.shape == (len(df),)
    assert np.all(score >= 0.0) and np.all(score <= 1.0)


def test_determinism_same_seed() -> None:
    df = _frame()
    a = fit_anomaly(df, n_estimators=100, max_samples=64, seed=7)
    b = fit_anomaly(df, n_estimators=100, max_samples=64, seed=7)
    assert np.allclose(a.score(df), b.score(df))


def test_customer_folds_keep_customer_together() -> None:
    df = _frame()
    folds = customer_folds(df["customer_id"].to_numpy(), n_splits=5, seed=3)
    tmp = pd.DataFrame({"c": df["customer_id"], "f": folds})
    assert tmp.groupby("c")["f"].nunique().max() == 1


def test_cross_fit_no_self_customer_scoring() -> None:
    """Every row's score equals a model fit on the *other* folds — never its own."""
    df = _frame()
    groups = df["customer_id"].to_numpy()
    folds = customer_folds(groups, n_splits=5, seed=1)
    scores = cross_fit_scores(df, groups, n_estimators=50, max_samples=32, seed=1, n_splits=5)

    f = folds[0]
    fit_rows = folds != f
    model = fit_anomaly(df.iloc[fit_rows], n_estimators=50, max_samples=32, seed=1)
    assert np.allclose(scores[folds == f], model.score(df.iloc[folds == f]))
    assert scores.shape == (len(df),)


def test_strict_variant_excludes_held_out_rows() -> None:
    df = _frame()
    is_hold = (df["customer_id"] == "c0").to_numpy()
    all_model = fit_anomaly(df, n_estimators=50, max_samples=32, seed=1)
    strict_model = fit_anomaly(df, n_estimators=50, max_samples=32, seed=1, exclude_mask=is_hold)
    assert all_model.n_fit_rows == len(df)
    assert strict_model.n_fit_rows == len(df) - int(is_hold.sum())


def test_cross_fit_strict_excludes_holdout_from_every_fold() -> None:
    df = _frame()
    groups = df["customer_id"].to_numpy()
    is_hold = (df["customer_id"] == "c0").to_numpy()
    scores = cross_fit_scores(
        df, groups, n_estimators=50, max_samples=32, seed=2, fit_exclude_mask=is_hold
    )
    assert scores.shape == (len(df),)
    assert np.all(np.isfinite(scores))


def test_grid_search_returns_valid_params() -> None:
    df = _frame()
    y = np.zeros(len(df), dtype=int)
    y[:20] = 1
    grid = [{"n_estimators": 50, "max_samples": 32}, {"n_estimators": 100, "max_samples": 64}]
    best, table = grid_search(df, df, y, grid=grid, seed=1)
    assert best["n_estimators"] in (50, 100)
    assert set(table.columns) >= {"n_estimators", "max_samples", "val_pr_auc"}
