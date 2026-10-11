"""Bulk-load simulator parquet tables into PostgreSQL (Part 04).

Design notes
------------
* The connection URL comes from ``DATABASE_URL`` (loaded from ``.env`` via
  python-dotenv).
* Bulk load uses ``psycopg2`` ``copy_expert`` with an **in-memory ``io.StringIO``
  CSV buffer** — never ``COPY FROM '/host/path'``. With Docker Desktop the server
  runs inside a container, so a host path would not exist server-side; streaming
  the CSV over the client connection sidesteps that entirely.
* Tables load in dependency order; indexes are created *after* load (the loader
  runs ``indexes.sql`` last) so COPY is not slowed by index maintenance.
* Idempotent: ``--reset`` drops and recreates the schemas; otherwise the raw
  tables are truncated and reloaded (documented choice over upsert).
"""

from __future__ import annotations

import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from fraudlens.config import load_config
from fraudlens.data_gen.drift import drift_start_date
from fraudlens.data_quality.validation import load_raw_tables
from fraudlens.paths import reports_dir, sql_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)

SCHEMAS = ["raw", "curated", "features", "model", "monitoring"]

# (schema, table, columns-in-copy-order). Order = dependency order: parents first.
TABLE_SPECS: list[tuple[str, str, list[str]]] = [
    ("raw", "locations", [
        "location_id", "country", "state", "city", "latitude", "longitude", "region_group",
    ]),
    ("raw", "merchants", [
        "merchant_id", "merchant_category", "region", "merchant_risk_group",
    ]),
    ("raw", "customers", [
        "customer_id", "age", "income_band", "home_region", "home_location_id",
        "account_open_date", "archetype", "baseline_daily_tx", "baseline_amount",
        "amount_sigma", "preferred_categories", "active_hour_mean", "active_hour_sd",
        "travel_frequency",
    ]),
    ("raw", "accounts", [
        "account_id", "customer_id", "account_type", "opening_date", "status", "base_currency",
    ]),
    ("raw", "devices", [
        "device_id", "first_seen", "last_seen", "device_type", "is_shared",
    ]),
    ("raw", "customer_devices", ["customer_id", "device_id"]),
    ("raw", "transactions", [
        "transaction_id", "customer_id", "account_id", "timestamp", "amount", "currency",
        "merchant_id", "merchant_category", "device_id", "location_id", "recipient_id",
        "transaction_type", "channel", "created_at",
    ]),
    ("raw", "labels", [
        "transaction_id", "is_fraud", "fraud_scenario", "episode_id", "stealth",
        "hard_negative_type", "label_available_at",
    ]),
]

RAW_TABLES = [(s, t) for s, t, _ in TABLE_SPECS]

# The customer-history query the API will use (Part 13). Explained before and
# after indexing to show the access-path change.
CUSTOMER_HISTORY_SQL = (
    "SELECT transaction_id, timestamp, amount, merchant_category, channel "
    "FROM raw.transactions "
    "WHERE customer_id = %s AND timestamp < %s "
    "ORDER BY timestamp DESC LIMIT 50"
)


@dataclass
class TableLoad:
    schema: str
    table: str
    rows: int
    seconds: float


def load_database_url() -> str:
    """Return ``DATABASE_URL`` from the environment (after loading ``.env``)."""
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env (make db-up first)."
        )
    return url


def get_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or load_database_url(), future=True)


def _to_json_value(v: object) -> str | None:
    """JSON-encode a list-like cell (parquet list -> JSONB); None/NaN -> NULL."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return json.dumps([str(x) for x in v])
    return json.dumps(v)


def _csv_buffer(df: pd.DataFrame, columns: list[str]) -> io.StringIO:
    """Build a text-mode CSV buffer for ``COPY ... FROM STDIN``.

    Text mode (``io.StringIO``) is required: ``copy_expert`` expects a text
    file-like for CSV; a ``BytesIO`` raises a confusing encoding error.
    """
    out = df[columns].copy()
    if "preferred_categories" in out.columns:
        out["preferred_categories"] = out["preferred_categories"].apply(_to_json_value)
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map({True: "true", False: "false"})
    buf = io.StringIO()
    out.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    return buf


def _run_sql_file(cur, path: Path) -> None:
    cur.execute(path.read_text(encoding="utf-8"))


def _load_table(cur, schema: str, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    col_list = ", ".join(columns)
    copy_sql = (
        f"COPY {schema}.{table} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '')"
    )
    cur.copy_expert(copy_sql, _csv_buffer(df, columns))


def _truncate_raw(cur) -> None:
    targets = ", ".join(f"{s}.{t}" for s, t in RAW_TABLES)
    cur.execute(f"TRUNCATE {targets} RESTART IDENTITY CASCADE")


def _set_drift_boundary(cur, profile: str) -> None:
    sim = load_config(profile).simulation
    boundary = drift_start_date(sim)
    cur.execute("DELETE FROM curated.drift_boundary")
    cur.execute(
        "INSERT INTO curated.drift_boundary (boundary_ts) VALUES (%s)",
        (pd.Timestamp(boundary).to_pydatetime() if boundary else None,),
    )


def explain_analyze(cur, sql: str, params: tuple) -> str:
    """Run ``EXPLAIN (ANALYZE, BUFFERS)`` and return the plan as text."""
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS) {sql}", params)
    return "\n".join(row[0] for row in cur.fetchall())


def _reference_customer(cur) -> tuple[str, object]:
    """A customer with the most transactions + the dataset's max timestamp."""
    cur.execute(
        "SELECT customer_id FROM raw.transactions "
        "GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 1"
    )
    customer_id = cur.fetchone()[0]
    cur.execute("SELECT MAX(timestamp) FROM raw.transactions")
    return customer_id, cur.fetchone()[0]


def load_all(profile: str, reset: bool = False, write_explain: bool = True) -> dict:
    """Load the ``profile`` parquet tables; return row counts and timings."""
    engine = get_engine()
    tables = load_raw_tables(profile)
    missing = [t for _, t, _ in TABLE_SPECS if t not in tables]
    if missing:
        raise FileNotFoundError(
            f"missing parquet tables for {profile}: {missing}. Run the simulator first."
        )

    conn = engine.raw_connection()
    loads: list[TableLoad] = []
    explain_before = explain_after = ""
    try:
        cur = conn.cursor()
        if reset:
            logger.info("Reset: dropping schemas %s", SCHEMAS)
            cur.execute(
                "DROP SCHEMA IF EXISTS raw, curated, features, model, monitoring CASCADE"
            )

        _run_sql_file(cur, sql_dir() / "schema.sql")
        _truncate_raw(cur)
        _set_drift_boundary(cur, profile)

        for schema, table, columns in TABLE_SPECS:
            df = tables[table]
            t0 = time.perf_counter()
            _load_table(cur, schema, table, df, columns)
            secs = time.perf_counter() - t0
            loads.append(TableLoad(schema, table, len(df), secs))
            logger.info("  loaded %s.%s: %d rows in %.2fs", schema, table, len(df), secs)

        ref_customer, ref_before = _reference_customer(cur)
        explain_before = explain_analyze(
            cur, CUSTOMER_HISTORY_SQL, (ref_customer, ref_before)
        )

        logger.info("Creating indexes (indexes.sql)")
        _run_sql_file(cur, sql_dir() / "indexes.sql")

        explain_after = explain_analyze(
            cur, CUSTOMER_HISTORY_SQL, (ref_customer, ref_before)
        )

        logger.info("Creating views (views.sql)")
        _run_sql_file(cur, sql_dir() / "views.sql")

        # Verify row counts against parquet.
        for spec in loads:
            cur.execute(f"SELECT COUNT(*) FROM {spec.schema}.{spec.table}")
            got = int(cur.fetchone()[0])
            if got != spec.rows:
                raise AssertionError(
                    f"{spec.schema}.{spec.table}: parquet {spec.rows} != db {got}"
                )
        conn.commit()
        logger.info("All row counts verified against parquet.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    if write_explain and explain_before and explain_after:
        out = Path(reports_dir()) / f"db_explain_{profile}.txt"
        out.write_text(
            f"# EXPLAIN (ANALYZE, BUFFERS) — customer-history query — profile {profile}\n"
            f"# reference customer: {ref_customer}\n"
            f"# query: {CUSTOMER_HISTORY_SQL}\n\n"
            f"===== BEFORE indexes.sql =====\n{explain_before}\n\n"
            f"===== AFTER indexes.sql =====\n{explain_after}\n",
            encoding="utf-8",
        )
        logger.info("EXPLAIN before/after written to %s", out)

    return {
        "profile": profile,
        "loads": [spec.__dict__ for spec in loads],
        "total_seconds": round(sum(s.seconds for s in loads), 3),
    }


def table_counts(database_url: str | None = None) -> dict[str, int]:
    """Quick ``schema.table`` -> row count map (used by tests)."""
    engine = get_engine(database_url)
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for schema, table in RAW_TABLES:
            counts[f"{schema}.{table}"] = int(
                conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}")).scalar_one()
            )
    return counts
