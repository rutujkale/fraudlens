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
    """Loads the whole raw transactions table once (dev: ~2 s) and slices in memory.

    Rows are read with pyarrow-backed dtypes and are **not** pre-sorted:
    both consumers (``build_features`` and ``summarize_history``) sort their
    inputs themselves, and skipping the defensive sort halves the peak RSS of
    the load (matters on the 512 MB free-tier plan). Three columns no consumer
    reads (``account_id``, ``currency``, ``created_at``) are skipped, and
    columns containing nulls become categoricals so downstream numpy loops see
    ``np.nan`` (never ``pd.NA``).
    """

    backend = "memory"

    _COLUMNS = [
        "transaction_id",
        "customer_id",
        "timestamp",
        "amount",
        "merchant_id",
        "merchant_category",
        "device_id",
        "location_id",
        "recipient_id",
        "transaction_type",
        "channel",
    ]

    def __init__(self, profile: str) -> None:
        start = time.perf_counter()
        tx = pd.read_parquet(
            raw_data_dir(profile) / "transactions.parquet",
            columns=InMemoryHistoryProvider._COLUMNS,
            dtype_backend="pyarrow",
        )
        for col in tx.columns:
            if isinstance(tx[col].dtype, pd.ArrowDtype) and tx[col].isna().any():
                tx[col] = (
                    tx[col].astype(object).where(tx[col].notna(), np.nan).astype("category")
                )
        self._tx = tx
        self._by_customer = tx.groupby("customer_id", sort=False, observed=True).indices
        self._by_device = tx.groupby("device_id", sort=False, observed=True).indices
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


def summarize_history(
    customer_id: str,
    history: pd.DataFrame,
    *,
    before: Any = None,
    recent_n: int = 5,
    usual_hours: int = 3,
) -> dict[str, Any]:
    """Compact context for one customer from a strictly-prior ``history`` frame.

    Only the customer's **own** rows are used (the provider may also return device
    rows from other customers). Pure function, so it can be reused and tested
    against either backend.
    """
    own = history[history["customer_id"] == customer_id] if len(history) else history
    recent: list[dict[str, Any]] = []
    typical_amount: float | None = None
    usual: list[int] = []
    devices: list[str] = []
    if len(own):
        own = own.copy()
        own["timestamp"] = pd.to_datetime(own["timestamp"])
        own = own.sort_values(["timestamp", "transaction_id"], kind="stable")
        typical_amount = float(own["amount"].median())
        usual = [int(h) for h in own["timestamp"].dt.hour.value_counts().head(usual_hours).index]
        devices = sorted({str(d) for d in own["device_id"].dropna().unique()})
        for _, row in own.tail(recent_n).iloc[::-1].iterrows():
            recent.append(
                {
                    "transaction_id": str(row["transaction_id"]),
                    "timestamp": row["timestamp"].to_pydatetime(),
                    "amount": float(row["amount"]),
                    "merchant_category": (
                        None
                        if pd.isna(row.get("merchant_category"))
                        else str(row["merchant_category"])
                    ),
                    "channel": str(row["channel"]),
                    "transaction_type": str(row["transaction_type"]),
                }
            )
    return {
        "customer_id": customer_id,
        "before": before,
        "n_prior_tx": int(len(own)),
        "typical_amount": typical_amount,
        "usual_hours": usual,
        "known_devices": devices,
        "recent": recent,
    }
