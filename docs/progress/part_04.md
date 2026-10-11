# Part 04 — PostgreSQL Warehouse: Schema, Loader, Analytical Views

## What was built

### 4A — DDL (`sql/schema.sql`)
- Idempotent (`CREATE ... IF NOT EXISTS`) schemas: `raw`, `curated`, `features`, `model`, `monitoring`.
- 8 `raw.*` tables mirroring the parquet columns, with proper types: `TIMESTAMP`,
  `NUMERIC(14,2)`, `BOOLEAN`, and `CHECK` text enums (categories, transaction types,
  channels, income bands, archetypes, account type/status, device type,
  merchant risk group, fraud scenarios, hard-negative types, currency). PKs and FKs
  throughout.
- `raw.transactions` deliberately has **no** label columns; `raw.labels` is a separate
  table keyed by `transaction_id` (never joined into feature tables).
- `features.feature_snapshot` (`transaction_id` PK, `feature_as_of_timestamp`,
  `feature_version`, `features JSONB`) with the trade-off documented in a SQL comment.
- `model.model_registry`, `model.predictions`, `monitoring.metrics`.
- `curated.drift_boundary` — one-row config metadata the loader populates from config,
  so `scenario_summary` can label period A/B without hard-coded dates.

### 4B — Indexes (`sql/indexes.sql`)
Created **after** bulk load (last step). Each index has a comment naming the query it
serves:
- `idx_transactions_customer_ts (customer_id, timestamp)` — the API customer-history query.
- `idx_transactions_device_ts (device_id, timestamp)`; `idx_transactions_ts (timestamp)`.
- `brin_transactions_ts USING BRIN (timestamp)` — compact alternative, kept as a comparison.
- `idx_labels_fraud (transaction_id) WHERE is_fraud` — partial index.

### 4C — Loader (`db/loader.py`, `scripts/load_db.py`)
- `get_engine()` reads `DATABASE_URL` from `.env` (via `python-dotenv`).
- **Bulk load = psycopg2 `copy_expert` with an in-memory `io.StringIO` (text-mode) CSV
  buffer.** No `COPY FROM '/host/path'` anywhere, so Docker Desktop's host/container
  filesystem split is irrelevant. Booleans normalised to `true`/`false`; the parquet
  list column `preferred_categories` JSON-encoded into `JSONB`.
- Dependency-ordered load; row counts verified against parquet; per-table timing.
- Idempotent: `--reset` drops/recreates schemas; otherwise TRUNCATE-and-reload (documented).

### 4D — Views (`sql/views.sql`)
`CREATE OR REPLACE`: `curated.transactions_enriched` (no labels) + 6 analytical views
(`daily_summary`, `scenario_summary`, `customer_summary` via `percentile_cont`,
`hourly_profile`, `review_load_by_day` placeholder, `rolling_24h_sql_demo`).

### 4E — Query helpers (`db/queries.py`)
`fetch_transactions`, `fetch_customer_history`, `fetch_device_history` — parameterized
(`text()` + bind params), strictly `timestamp < before_ts`.

### 4F — Tests (`tests/test_db.py`, marker `db`)
6 tests, skip gracefully if Postgres unreachable or `tiny` parquet missing: row counts ==
parquet, no label columns in `raw.transactions`, FK enforcement (negative insert raises),
`fetch_customer_history` strictly prior (incl. exact-boundary case), all 7 views return
rows, loader idempotency.

## Acceptance commands — actual outputs

| Command | Result |
|---|---|
| `make db-up` | container `fraudlens_db` up and healthy (host 5432) |
| `python scripts/load_db.py --profile dev --reset` | **606 915 rows / 8 tables in 13.8 s**, all counts verified |
| `python scripts/load_db.py --profile tiny --reset` | 300→4716 rows, 0.48 s |
| `pytest -m db -q` | `6 passed` |
| `make test-fast` | `57 passed, 6 deselected` |
| `make lint` | `All checks passed!` |

### Row counts (dev)

| table | rows | | table | rows |
|---|---:|---|---|---:|
| raw.locations | 42 | | raw.devices | 8 702 |
| raw.merchants | 625 | | raw.customer_devices | 7 765 |
| raw.customers | 5 000 | | raw.transactions | 289 870 |
| raw.accounts | 5 041 | | raw.labels | 289 870 |

Load time (dev): transactions 10.0 s, labels 3.4 s, all others < 0.2 s (total 13.8 s).
Index creation ran after load (`indexes.sql`), then `views.sql`.

### Index before/after — `reports/db_explain_dev.txt` (customer-history query)

Reference customer `C0003346` (busiest), `EXPLAIN (ANALYZE, BUFFERS)`:

| | node | rows removed / cond | Buffers | Execution |
|---|---|---|---|---|
| **before** | `Seq Scan` → `Sort` (top-N) | Rows Removed by Filter: 289 517 | shared hit **5267** | **23.72 ms** |
| **after** | `Index Scan Backward using idx_transactions_customer_ts` | Index Cond on `(customer_id, timestamp)` | shared hit 46 + read 3 (**49**) | **0.12 ms** |

~190× faster; the plan changes from a full table scan + sort to a single index scan that
also satisfies the `ORDER BY ... DESC LIMIT 50` (no sort node).

### View sample outputs (dev)

- `curated.daily_summary` — 2026-01-01: 3859 tx, 9 fraud (0.00233), ₹22.69 M total.
- `curated.scenario_summary` (drift visible): period A structuring share 0.152 → period B
  0.365; burst A 0.449 → B 0.291.
- `curated.customer_summary` — `C0003346`: 353 tx, median ₹19 669.55, 5 devices,
  16 locations, 3 fraud.
- `curated.hourly_profile` — peak hour 13 on weekday 6 (5467 tx).
- `curated.review_load_by_day` — 2026-01-01: 3859 tx, 180 above the p95 amount cutoff.
- `curated.rolling_24h_sql_demo` — `C0001860` reached 29 prior tx in 24 h.
- `curated.transactions_enriched` — joins customer/merchant/location context, no labels.

## Deviations from the prompt (and why)

1. **`psql` acceptance adapted.** `psql` is not installed on the host; the acceptance
   `psql "$DATABASE_URL" -c ...` was run as
   `docker exec fraudlens_db psql -U fraudlens -d fraudlens -c ...`.
2. **`preferred_categories` stored as `JSONB`** (parquet list) rather than a text array —
   simplest faithful type; documented in `schema.sql`.
3. **`curated.drift_boundary` table** added so `scenario_summary` gets the config-driven
   drift date without hard-coding it in SQL. Populated by the loader from
   `drift_start_date(config)`; NULL for no-drift profiles (all period A).
4. **Enum CHECK values are mirrored** from the Python constants. They are the only
   duplication; a mismatch fails loudly at load time rather than silently.
5. **Idempotency choice = truncate-and-reload** (over upsert), documented in the loader.
6. **`rolling_24h_sql_demo` peer nuance:** verified 0 same-timestamp peer groups in `dev`,
   so `EXCLUDE CURRENT ROW` matches strict-prior here; caveat recorded in
   `docs/what_went_wrong.md` for Part 05.
7. **`review_load_by_day`** uses a placeholder global p95 cutoff; Part 07 replaces it with
   the cost/capacity policy.

## Open issues / notes

- Load is single-transaction; a 290 k-row load is fast (14 s). At `main` scale it would
  still be well within budget.
- The `db`-marked tests reload `tiny` with `--reset`, so after running them the DB holds
  `tiny`; reload `dev` for the psql acceptance (`python scripts/load_db.py --profile dev --reset`).
- No feature computation in SQL (only the one cross-check view); labels never enter feature tables.
