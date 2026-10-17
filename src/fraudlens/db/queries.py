"""Typed, parameterized query helpers over the raw warehouse (Part 04).

These feed the API in Part 13. All history queries are strictly point-in-time:
they never return a row whose ``timestamp`` is ``>=`` the supplied cutoff.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from fraudlens.db.loader import get_engine

_TX_COLUMNS = (
    "transaction_id, customer_id, account_id, timestamp, amount, currency, "
    "merchant_id, merchant_category, device_id, location_id, recipient_id, "
    "transaction_type, channel"
)


def _rows(result) -> list[dict[str, Any]]:
    return [dict(row._mapping) for row in result]


def fetch_transactions(
    *,
    customer_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 1000,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """Fetch transactions, optionally filtered by customer and a time range."""
    clauses, params = [], {"limit": limit}
    if customer_id is not None:
        clauses.append("customer_id = :customer_id")
        params["customer_id"] = customer_id
    if start is not None:
        clauses.append("timestamp >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("timestamp < :end")
        params["end"] = end
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = text(
        f"SELECT {_TX_COLUMNS} FROM raw.transactions {where} "
        "ORDER BY timestamp, transaction_id LIMIT :limit"
    )
    eng = engine or get_engine()
    with eng.connect() as conn:
        return _rows(conn.execute(sql, params))


def _history_rows(
    column: str,
    value: str,
    before_ts: datetime,
    lookback_days: int | None,
    engine: Engine | None,
) -> list[dict[str, Any]]:
    """Rows for one key strictly before ``before_ts`` (unbounded when no lookback).

    ``column`` is an internal literal (``customer_id`` / ``device_id``), never user
    input. ``lookback_days=None`` drops the lower bound: the API needs the full
    prior history to reproduce lifetime and cross-customer device features (Part 13).
    """
    window = "AND timestamp >= :window_start " if lookback_days is not None else ""
    sql = text(
        f"SELECT {_TX_COLUMNS} FROM raw.transactions "
        f"WHERE {column} = :value "
        "AND timestamp < :before_ts "
        f"{window}"
        "ORDER BY timestamp, transaction_id"
    )
    params: dict[str, Any] = {"value": value, "before_ts": before_ts}
    if lookback_days is not None:
        params["window_start"] = before_ts - timedelta(days=lookback_days)
    eng = engine or get_engine()
    with eng.connect() as conn:
        return _rows(conn.execute(sql, params))


def fetch_customer_history(
    customer_id: str,
    before_ts: datetime,
    lookback_days: int | None = None,
    *,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """A customer's transactions strictly before ``before_ts``.

    ``lookback_days`` bounds the window; ``None`` (default) fetches all prior rows.
    """
    return _history_rows("customer_id", customer_id, before_ts, lookback_days, engine)


def fetch_device_history(
    device_id: str,
    before_ts: datetime,
    lookback_days: int | None = None,
    *,
    engine: Engine | None = None,
) -> list[dict[str, Any]]:
    """A device's transactions strictly before ``before_ts`` (any customer).

    ``lookback_days=None`` (default) fetches all prior rows; the device's earlier
    uses by other customers set ``account_seen_on_device`` and ``device_age_days``.
    """
    return _history_rows("device_id", device_id, before_ts, lookback_days, engine)
