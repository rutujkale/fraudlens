# Part 04 — PostgreSQL Warehouse: Schema, Loader, Analytical Views

## Goal
Store the simulated data in PostgreSQL with a reproducible schema, fast bulk loading, sensible indexes, and a handful of analytical views. This part exists to show real SQL and data-modeling skill, so keep it **time-boxed**: PostgreSQL is the system of record and analytics layer. Feature logic stays in Python (Part 05) to guarantee training/serving consistency.

## Prerequisites
Parts 01–03 complete; Docker Postgres running (`make db-up`); `data/raw/dev/` validated.

## Schemas
`raw`, `curated`, `features`, `model`, `monitoring`.

## Tasks

### 4A — DDL (`sql/schema.sql`, idempotent: `CREATE ... IF NOT EXISTS`)
- `raw.customers, raw.accounts, raw.devices, raw.customer_devices, raw.merchants, raw.locations, raw.transactions, raw.labels` mirroring the parquet tables, with proper types (`TIMESTAMP`, `NUMERIC(14,2)` for amount, `BOOLEAN`, text enums with `CHECK` constraints), primary keys and foreign keys. `raw.labels` is a separate table keyed by `transaction_id` and must never be joined into feature tables.
- `features.feature_snapshot`: `transaction_id` PK, `feature_as_of_timestamp`, `feature_version`, plus a JSONB column `features` (the wide table lives in parquet; JSONB keeps the schema stable while features evolve). Document this trade-off in a SQL comment.
- `model.predictions` (prediction_id, transaction_id nullable, model_version, raw_probability, calibrated_probability, risk_score, action, scored_at, latency_ms) and `model.model_registry` (model_version, feature_version, trained_at, training_window, calibration, policy_version, metrics JSONB).
- `monitoring.metrics` (metric_date, model_version, metric_name, metric_value, extra JSONB).
- A `curated.transactions_enriched` **view** joining transactions with customers, merchants and locations (no labels).

### 4B — Indexes (`sql/indexes.sql`)
- `raw.transactions (customer_id, timestamp)`, `(device_id, timestamp)`, `(timestamp)`, BRIN on `timestamp` as a comparison; `raw.labels (is_fraud)` partial index where `is_fraud`.
- Add a comment above each index explaining the query it serves. Create indexes **after** bulk load for speed (the loader runs `indexes.sql` last).
- In the progress file, include `EXPLAIN (ANALYZE)` output for one customer-history query before and after indexing (this is the same query the API will use in Part 13).

### 4C — Loader (`db/loader.py`, `scripts/load_db.py --profile dev [--reset]`)
- SQLAlchemy engine from `DATABASE_URL`. `--reset` drops and recreates schemas.
- Bulk load with PostgreSQL `COPY` (via `psycopg2` `copy_expert` from an in-memory CSV buffer or chunked streaming), in dependency order, inside a transaction. Row counts per table printed and verified against parquet.
- Idempotent: re-running without `--reset` must not duplicate data (truncate-and-load per profile, or upsert — choose and document).

### 4D — Analytical views (`sql/views.sql`)
Write these with window functions / CTEs where appropriate:
1. `curated.daily_summary`: date, tx count, fraud count, fraud rate, total amount, fraud amount (joins labels — analytics only).
2. `curated.scenario_summary`: by scenario and drift period: count, avg amount, share.
3. `curated.customer_summary`: per customer tx count, avg/median amount (`percentile_cont`), distinct devices, distinct locations, first/last tx, fraud count.
4. `curated.hourly_profile`: tx volume and fraud rate by hour × weekday.
5. `curated.review_load_by_day`: counts per day of transactions above a configurable amount percentile (placeholder for the review-load analysis in Part 07).
6. `curated.rolling_24h_sql_demo`: for each transaction, count of the same customer's *prior* transactions in the last 24 h using `COUNT(*) OVER (PARTITION BY customer_id ORDER BY timestamp RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW)`. (This is a SQL cross-check for one feature in Part 05; verify that `EXCLUDE CURRENT ROW`, available in PostgreSQL 11+, behaves as strict-prior. Note that rows with the *same* timestamp are peers and `EXCLUDE CURRENT ROW` does not exclude peers — if it differs from the Part 05 definition, use `EXCLUDE TIES` or a self-join with `<` and record the nuance in `docs/what_went_wrong.md`.)

### 4E — Query helpers (`db/queries.py`)
Typed functions: `fetch_transactions(profile/time range)`, `fetch_customer_history(customer_id, before_ts, lookback_days)` (parameterized, strictly `timestamp < before_ts`), `fetch_device_history(device_id, before_ts, lookback_days)`. These feed the API in Part 13.

### 4F — Tests (`tests/test_db.py`, marker `db`; skip gracefully if Postgres is unreachable)
- Load `tiny`; row counts equal parquet; FK constraints hold; no label columns in `raw.transactions`; `fetch_customer_history` never returns a row with `timestamp >= before_ts`; views return rows; loader idempotency.

## Acceptance criteria
```bash
make db-up
python scripts/load_db.py --profile dev --reset
pytest -m db -q
psql "$DATABASE_URL" -c "select * from curated.daily_summary limit 5;"
```
Progress file: row counts, load time, index before/after timings, view sample outputs.

## Do not
Compute the ML features in SQL (only the one cross-check view). Put labels in `raw.transactions`. Spend more than a small slice of the project on this part.
