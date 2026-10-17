"""History providers for the scoring API (Part 13A).

A target transaction's features depend on exactly two row sets, both **strictly
before** ``T``:

* its customer's full prior history — lifetime features (``amount_vs_lifetime_median``,
  ``n_prior_tx``, ``is_cold_start``) use every earlier row, not a 30-day window; and
* its device's full prior history **across all customers** — ``accounts_seen_on_device``
  and ``device_age_days`` are global device state.

The provider returns the union. Extra device rows from other customers never change
the target's customer-level features because ``build_features`` groups by ``customer_id``.

``HISTORY_BACKEND=postgres|memory`` selects the implementation.
"""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

import numpy as np
import pandas as pd

from fraudlens.db import queries
from fraudlens.paths import raw_data_dir


def history_backend_from_env() -> str:
    """``HISTORY_BACKEND`` (default ``memory``, which needs no database)."""
    backend = os.environ.get("HISTORY_BACKEND", "memory").strip().lower()
    if backend not in {"postgres", "memory"}:
        raise ValueError(f"HISTORY_BACKEND must be 'postgres' or 'memory', got {backend!r}")
    return backend


class HistoryProvider(Protocol):
    """Supplies strictly-prior history for a transaction."""

    backend: str
    load_seconds: float

    def history(
        self, customer_id: str, device_id: str | None, before_ts: Any
    ) -> pd.DataFrame: ...

    def ok(self) -> bool: ...


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union of history frames, de-duplicated by ``transaction_id`` and sorted."""
    present = [f for f in frames if len(f)]
    if not present:
        return pd.DataFrame()
    combined = pd.concat(present, ignore_index=True)
    combined = combined.drop_duplicates(subset="transaction_id", keep="first")
    return combined.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(
        drop=True
    )


class PostgresHistoryProvider:
    """Reads history from ``raw.transactions`` via :mod:`fraudlens.db.queries`."""

    backend = "postgres"

    def __init__(self) -> None:
        self.load_seconds = 0.0

    def history(
        self, customer_id: str, device_id: str | None, before_ts: Any
    ) -> pd.DataFrame:
        frames = [pd.DataFrame(queries.fetch_customer_history(customer_id, before_ts))]
        if device_id:
            frames.append(pd.DataFrame(queries.fetch_device_history(device_id, before_ts)))
        return _combine(frames)

    def ok(self) -> bool:
        try:
            queries.fetch_transactions(limit=1)
            return True
        except Exception:  # noqa: BLE001 - any failure means not usable
            return False


class InMemoryHistoryProvider:
    """Loads the whole raw transactions table once (dev: ~2 s) and slices in memory."""

    backend = "memory"

    def __init__(self, profile: str) -> None:
        start = time.perf_counter()
        tx = pd.read_parquet(raw_data_dir(profile) / "transactions.parquet")
        tx = tx.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)
        self._tx = tx
        self._by_customer = tx.groupby("customer_id", sort=False).indices
        self._by_device = tx.groupby("device_id", sort=False).indices
        self.load_seconds = time.perf_counter() - start

    def history(
        self, customer_id: str, device_id: str | None, before_ts: Any
    ) -> pd.DataFrame:
        positions: list[np.ndarray] = []
        customer_pos = self._by_customer.get(customer_id)
        if customer_pos is not None:
            positions.append(customer_pos)
        if device_id:
            device_pos = self._by_device.get(device_id)
            if device_pos is not None:
                positions.append(device_pos)
        if not positions:
            return pd.DataFrame()
        idx = np.unique(np.concatenate(positions))
        sub = self._tx.iloc[idx]
        sub = sub[sub["timestamp"] < pd.Timestamp(before_ts)]
        return sub.reset_index(drop=True)

    def ok(self) -> bool:
        return True


def get_provider(profile: str, backend: str) -> HistoryProvider:
    """Build the provider named by ``backend``."""
    if backend == "postgres":
        return PostgresHistoryProvider()
    if backend == "memory":
        return InMemoryHistoryProvider(profile)
    raise ValueError(f"unknown history backend {backend!r}")
