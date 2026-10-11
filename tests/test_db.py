"""PostgreSQL integration tests (Part 04).

Marked ``db``; skipped gracefully when PostgreSQL is unreachable or the ``tiny``
parquet tables are absent, so the fast suite still runs in CI without Docker.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from fraudlens.data_gen.generate import FORBIDDEN_COLUMNS
from fraudlens.data_quality.validation import load_raw_tables
from fraudlens.db import queries
from fraudlens.db.loader import RAW_TABLES, get_engine, load_all, table_counts

pytestmark = pytest.mark.db


def _engine_or_skip():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as exc:  # noqa: BLE001 - any connection failure means skip
        pytest.skip(f"PostgreSQL not reachable: {exc}")


@pytest.fixture(scope="module")
def tiny_tables() -> dict[str, pd.DataFrame]:
    tables = load_raw_tables("tiny")
    if not tables or "transactions" not in tables:
        pytest.skip("tiny parquet data missing; run scripts/generate_data.py --profile tiny")
    return tables


@pytest.fixture(scope="module")
def loaded(tiny_tables):
    _engine_or_skip()
    load_all("tiny", reset=True, write_explain=False)
    return tiny_tables


def test_row_counts_match_parquet(loaded):
    counts = table_counts()
    for schema, table in RAW_TABLES:
        assert counts[f"{schema}.{table}"] == len(loaded[table]), table


def test_transactions_have_no_label_columns(loaded):
    engine = get_engine()
    with engine.connect() as conn:
        cols = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'raw' AND table_name = 'transactions'"
                )
            )
        }
    assert FORBIDDEN_COLUMNS.isdisjoint(cols), sorted(FORBIDDEN_COLUMNS & cols)


def test_foreign_keys_enforced(loaded):
    engine = get_engine()
    with engine.begin() as conn:
        with pytest.raises(IntegrityError):
            conn.execute(
                text(
                    "INSERT INTO raw.customer_devices (customer_id, device_id) "
                    "VALUES ('NOPE', 'NOPE')"
                )
            )


def test_fetch_customer_history_is_strictly_prior(loaded):
    tx = loaded["transactions"]
    top = tx["customer_id"].value_counts().idxmax()
    before_ts = tx["timestamp"].max() + pd.to_timedelta(1, unit="D")
    rows = queries.fetch_customer_history(top, before_ts.to_pydatetime(), lookback_days=365)

    assert rows, "expected history for the busiest customer"
    assert all(r["timestamp"] < before_ts.to_pydatetime() for r in rows)

    # Boundary: cutoff at exactly the customer's own max timestamp must exclude it.
    cust_max = tx.loc[tx["customer_id"] == top, "timestamp"].max()
    boundary_rows = queries.fetch_customer_history(
        top, cust_max.to_pydatetime(), lookback_days=365
    )
    assert all(r["timestamp"] < cust_max.to_pydatetime() for r in boundary_rows)


def test_views_return_rows(loaded):
    engine = get_engine()
    views = [
        "curated.transactions_enriched",
        "curated.daily_summary",
        "curated.scenario_summary",
        "curated.customer_summary",
        "curated.hourly_profile",
        "curated.review_load_by_day",
        "curated.rolling_24h_sql_demo",
    ]
    with engine.connect() as conn:
        for view in views:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {view}")).scalar_one()
            assert count > 0, view


def test_loader_is_idempotent(loaded):
    before = table_counts()
    load_all("tiny", reset=False, write_explain=False)
    after = table_counts()
    assert before == after
