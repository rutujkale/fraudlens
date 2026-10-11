"""Tests for temporal splits and the test-set guard (Part 06A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import load_config
from fraudlens.evaluation.guard import TestSetGuard
from fraudlens.evaluation.splits import (
    customer_holdout,
    make_temporal_split,
    max_train_label_available_at,
    rolling_origin_folds,
    save_split,
    sort_transactions,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config("dev")


def _make_tx(days: int = 100, per_day: int = 10, start: str = "2026-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = days * per_day
    ts = pd.Timestamp(start) + pd.to_timedelta(np.sort(rng.uniform(0, days, n)), unit="D")
    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:07d}" for i in range(n)],
            "customer_id": [f"C{i % 50:04d}" for i in range(n)],
            "timestamp": ts,
        }
    )


def test_windows_disjoint_and_ordered(cfg):
    tx = sort_transactions(_make_tx())
    split = make_temporal_split(tx, cfg)
    assert set(split.train_idx).isdisjoint(split.val_idx)
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert set(split.val_idx).isdisjoint(split.test_idx)
    ts = tx["timestamp"].to_numpy()
    assert ts[split.train_idx].max() < ts[split.val_idx].min()
    assert ts[split.val_idx].max() < ts[split.test_idx].min()


def test_embargo_gap_at_least_e_days(cfg):
    tx = sort_transactions(_make_tx())
    split = make_temporal_split(tx, cfg)
    ts = tx["timestamp"].to_numpy()
    e = pd.Timedelta(int(cfg.embargo_days), unit="D")
    gap1 = pd.Timestamp(ts[split.val_idx].min()) - pd.Timestamp(ts[split.train_idx].max())
    gap2 = pd.Timestamp(ts[split.test_idx].min()) - pd.Timestamp(ts[split.val_idx].max())
    assert gap1 > e
    assert gap2 > e


def test_train_labels_mature(cfg):
    tx = sort_transactions(_make_tx())
    split = make_temporal_split(tx, cfg)
    rng = np.random.default_rng(1)
    delay = rng.integers(2, 8, size=len(tx))
    labels = pd.DataFrame(
        {
            "transaction_id": tx["transaction_id"],
            "label_available_at": tx["timestamp"] + pd.to_timedelta(delay, unit="D"),
        }
    )
    latest = max_train_label_available_at(labels, tx, split)
    t_train_end = pd.Timestamp(split.boundaries["t_train_end"])
    assert latest <= t_train_end + pd.Timedelta(int(cfg.embargo_days), unit="D")


def test_customer_holdout_is_disjoint_and_sized():
    customers = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(100)]})
    seen, heldout = customer_holdout(customers, fraction=0.2, seed=7)
    assert len(heldout) == 20
    assert set(seen).isdisjoint(heldout)
    assert len(seen) + len(heldout) == 100


def test_customer_holdout_dedupes_repeated_transaction_ids():
    # transaction-level ids repeat; the fraction must apply to unique customers.
    ids = np.array([f"C{i % 50:04d}" for i in range(1000)])
    seen, heldout = customer_holdout(ids, fraction=0.2, seed=7)
    assert len(heldout) == 10
    assert len(seen) == 40
    assert set(seen).isdisjoint(heldout)


def test_holdout_customers_excluded_from_train(cfg):
    tx = sort_transactions(_make_tx())
    split = make_temporal_split(tx, cfg)
    seen, heldout = customer_holdout(tx["customer_id"], fraction=0.2, seed=7)
    train = tx.iloc[split.train_idx]
    # held-out customers do appear in the raw train window (so filtering matters) ...
    assert set(train["customer_id"]) & set(heldout)
    # ... and after applying the holdout filter, none remain.
    train_seen = train[train["customer_id"].isin(seen)]
    assert set(train_seen["customer_id"]).isdisjoint(set(heldout))


def test_rolling_origin_folds_are_chronological(cfg):
    tx = sort_transactions(_make_tx(days=120))
    folds = rolling_origin_folds(tx, cfg, n_folds=3)
    assert len(folds) == 3
    ts = tx["timestamp"].to_numpy()
    for tr, va in folds:
        assert len(tr) and len(va)
        assert ts[tr].max() < ts[va].min()


def test_save_split_roundtrip(cfg, tmp_path):
    tx = sort_transactions(_make_tx(days=40))
    split = make_temporal_split(tx, cfg)
    out = save_split(split, profile="tiny")
    assert out.exists()
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_train"] == len(split.train_idx)
    assert set(data["boundaries"]) >= {"t0", "t_train_end", "t_val_end", "t_end"}


def test_unsorted_frame_raises(cfg):
    tx = _make_tx(days=5)
    tx = tx.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError):
        make_temporal_split(tx, cfg)


def test_guard_denies_until_final(tmp_path):
    guard = TestSetGuard(tmp_path / "log.csv")
    with pytest.raises(PermissionError):
        guard.access("evaluate_val", model_version="xgb_v1", final=False)
    assert guard.access("part12_eval", model_version="xgb_v1", final=True) is True
    rows = (tmp_path / "log.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3  # header + 2 accesses
    assert rows[0].startswith("timestamp,caller")
