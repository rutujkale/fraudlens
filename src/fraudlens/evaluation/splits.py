"""Temporal splits with embargo, customer holdout and rolling-origin folds (Part 06A).

Splits are chronological and cut on the **time axis** (not row counts), because
transactions are not uniformly spaced in time. After each cut an embargo gap of
``embargo_days`` (= the maximum label delay) is removed from every set: a label
for a transaction near the boundary may not have matured until later, so those
rows cannot be used for training or validation.

Contract: ``tx`` must already be sorted by ``(timestamp, transaction_id)``
(``build_features`` output is). Returned index arrays are positional indices into
that sorted frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.config import Config
from fraudlens.paths import results_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)

SORT_COLUMNS = ["timestamp", "transaction_id"]


def sort_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Return ``tx`` sorted by ``(timestamp, transaction_id)`` with a fresh index."""
    return tx.sort_values(SORT_COLUMNS, kind="stable").reset_index(drop=True)


@dataclass
class Split:
    """Positional indices for train/val/test plus boundary timestamps."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    boundaries: dict[str, str]
    embargo_days: int
    profile: str

    def to_json(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "embargo_days": self.embargo_days,
            "boundaries": self.boundaries,
            "n_train": int(len(self.train_idx)),
            "n_val": int(len(self.val_idx)),
            "n_test": int(len(self.test_idx)),
            "train_idx": [int(i) for i in self.train_idx],
            "val_idx": [int(i) for i in self.val_idx],
            "test_idx": [int(i) for i in self.test_idx],
        }


def make_temporal_split(tx: pd.DataFrame, cfg: Config) -> Split:
    """Chronological train/val/test split with an embargo gap after each cut."""
    sim = cfg.simulation
    return chronological_window_split(
        tx,
        train_end_fraction=sim.splits.train_end_fraction,
        val_end_fraction=sim.splits.val_end_fraction,
        embargo_days=cfg.embargo_days,
        profile=sim.profile,
    )


def chronological_window_split(
    tx: pd.DataFrame,
    *,
    train_end_fraction: float,
    val_end_fraction: float,
    embargo_days: int,
    profile: str,
) -> Split:
    """Time-axis 60/20/20-style split with an embargo, on explicit parameters.

    Same contract as :func:`make_temporal_split` but the fractions and embargo are
    passed directly so the external dataset (Part 11) can use its own axes without
    a synthetic profile config.
    """
    if not tx["timestamp"].is_monotonic_increasing:
        raise ValueError("chronological_window_split requires tx sorted by (timestamp, id)")
    ts = tx["timestamp"].to_numpy()
    t0 = pd.Timestamp(ts[0])
    t_end = pd.Timestamp(ts[-1])
    span = t_end - t0
    t_train_end = t0 + pd.Timedelta(span * train_end_fraction)
    t_val_end = t0 + pd.Timedelta(span * val_end_fraction)
    embargo = pd.Timedelta(int(embargo_days), unit="D")

    train = (ts >= np.datetime64(t0)) & (ts < np.datetime64(t_train_end))
    val = (ts >= np.datetime64(t_train_end + embargo)) & (ts < np.datetime64(t_val_end))
    test = (ts >= np.datetime64(t_val_end + embargo)) & (ts <= np.datetime64(t_end))

    boundaries = {
        "t0": str(t0),
        "t_train_end": str(t_train_end),
        "t_val_end": str(t_val_end),
        "t_end": str(t_end),
        "embargo_days": str(int(embargo_days)),
    }
    return Split(
        train_idx=np.flatnonzero(train),
        val_idx=np.flatnonzero(val),
        test_idx=np.flatnonzero(test),
        boundaries=boundaries,
        embargo_days=int(embargo_days),
        profile=profile,
    )


def chronological_fraction_split(
    idx: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split an already-time-sorted index array into (first ``fraction``, rest).

    ``fraction`` is applied by **position**, so the two parts are contiguous in
    time: the first covers the earlier window, the second the later window. This
    is the calibrator/threshold cut (Part 07); it must never be random, or future
    rows leak into fitting the calibrator.
    """
    idx = np.asarray(idx)
    if len(idx) < 2:
        return idx, idx[:0]
    cut = int(round(len(idx) * fraction))
    cut = max(1, min(len(idx) - 1, cut))
    return idx[:cut], idx[cut:]


def max_train_label_available_at(
    labels: pd.DataFrame, tx: pd.DataFrame, split: Split
) -> pd.Timestamp:
    """Latest ``label_available_at`` among training rows (for the maturity check)."""
    train_tx = tx.iloc[split.train_idx]
    merged = train_tx[["transaction_id"]].merge(labels, on="transaction_id", how="left")
    return pd.Timestamp(merged["label_available_at"].max())


def customer_holdout(
    customers: pd.DataFrame | np.ndarray, fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split customer ids into (seen, heldout); heldout are excluded from training."""
    ids = (
        customers["customer_id"].to_numpy()
        if isinstance(customers, pd.DataFrame)
        else np.asarray(customers)
    )
    ids = np.unique(ids)
    rng = np.random.default_rng(seed)
    n_hold = int(round(len(ids) * fraction))
    heldout = np.sort(rng.choice(ids, size=n_hold, replace=False))
    seen = ids[~np.isin(ids, heldout)]
    return seen, heldout


def rolling_origin_folds(
    tx: pd.DataFrame, cfg: Config, n_folds: int = 3
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological expanding-window folds inside the train window (for Part 08)."""
    sim = cfg.simulation
    ts = tx["timestamp"].to_numpy()
    t0 = pd.Timestamp(ts[0])
    t_end = pd.Timestamp(ts[-1])
    train_span = (t_end - t0) * sim.splits.train_end_fraction
    embargo = pd.Timedelta(int(cfg.embargo_days), unit="D")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_folds + 1):
        val_start = t0 + pd.Timedelta(train_span * (k / (n_folds + 1)))
        val_end = t0 + pd.Timedelta(train_span * ((k + 1) / (n_folds + 1)))
        tr = (ts >= np.datetime64(t0)) & (ts < np.datetime64(val_start - embargo))
        va = (ts >= np.datetime64(val_start)) & (ts < np.datetime64(val_end))
        folds.append((np.flatnonzero(tr), np.flatnonzero(va)))
    return folds


def save_split(split: Split, profile: str | None = None) -> Path:
    """Write ``results/split_<profile>.json`` and return the path."""
    name = profile or split.profile
    out = Path(results_dir()) / f"split_{name}.json"
    out.write_text(json.dumps(split.to_json(), indent=2), encoding="utf-8")
    return out
