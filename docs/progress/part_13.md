# Part 13 progress — FastAPI scoring service

**Status:** 13A (API package) + 13B (tests) + 13C (latency benchmark) + 13D (docs) complete.
· **Committed:** yes (3 commits: api code+tests; docs+results; review correction) · **Pushed:** yes

Frozen system served as-is: `xgb_dev_20261015_v1` (XGBoost + sigmoid calibrator +
`policy.json`). No retraining, no re-calibration, no policy re-optimisation. The **test window was
never opened** by Part 13.

## Correction pass (post-review)

A review of the first Part 13 commit surfaced four issues, all fixed here and re-measured on `dev`
(no `db`-marked pytest run between the reload and the benchmark):

1. **Postgres latency provenance was unrecorded.** The cells were measured on `dev`, but nothing in
   the output proved it. `benchmark_latency.py` now queries and stores
   `db: {expected_profile, transactions_rows, customers}` in `results/latency.json`
   (this run: dev, 277,607 transactions, 5,000 customers).
2. **Parity was only proven on the memory backend.** The calibrated 1e-6 parity test is `dev`+`memory`;
   the only postgres test checked strictness, not equality. Added
   `test_postgres_history_matches_memory_features` (db) proving postgres and in-memory providers yield
   the **same feature row** on tiny (feature-level, since tiny has no calibrator).
3. **The sample was mislabelled "1:1".** It is fraud-enriched: 360 fraud / 1,640 legit (18%,
   ~20x validation prevalence). Wording fixed here, in `docs/latency.md`, and in the
   `benchmark_latency.py` docstring.
4. **The headline cell was chosen post-hoc.** The rule is now explicit and stored
   (`headline_rule`), and this file states the cell was picked **after** the matrix was measured, not
   pre-registered (the earlier plan named `memory|conc1|explain-off`).

## What was built

- `src/fraudlens/api/schemas.py` — Pydantic v2 request/response models. `extra="forbid"`, `amount > 0`,
  tz-aware timestamps converted to IST then made naive, optional `merchant_id` (needed so
  `new_merchant_flag` / `merchant_freq_30d` reproduce exactly; `null` when absent) alongside
  `merchant_category`, `device_id`, `location_id`, `recipient_id`, `transaction_id`,
  `include_explanation`.
- `src/fraudlens/api/history.py` — `HistoryProvider` protocol with a PostgreSQL implementation
  (`db/queries.py`) and an in-memory implementation (loads all dev transactions once at start-up).
  `HISTORY_BACKEND=postgres|memory` (default `memory`).
- `src/fraudlens/api/deps.py` — `Scorer`: verifies freeze hashes, loads the serving bundle, builds the
  provider and entities, lazily builds the SHAP explainer. `score()` (single, no capacity) and
  `score_batch()` (cost policy + `enforce_capacity`, rows sorted by expected saving).
- `src/fraudlens/api/main.py` — `create_app(...)` (lifespan builds the `Scorer`), request-id
  middleware, `/health`, `/model-info`, `/predict`, `/predict/batch`, `/predict/batch/csv`; batch size
  guard returns 422 above `api.max_batch` (1000).
- `scripts/run_api.py` (uvicorn CLI) and `scripts/benchmark_latency.py` (latency matrix).
- `tests/test_api.py` — 20 tests (schema/validation, tz-normalisation, cold-start, batch size limit,
  REVIEW-only capacity cap, capacity-never-touches-BLOCK/STEP_UP, ordering, determinism, memory +
  postgres history strictness, **parity**, **postgres-vs-memory feature parity** [db], **tamper
  refusal**).
- `Makefile` — `api` and `benchmark-latency` targets.
- Docs: `docs/api.md`, `docs/latency.md`, this file; `what_went_wrong.md` gained a concurrency finding;
  `reports/headline_numbers.md` (and `docs/final_results.md`) regenerated so the Part 12 latency
  placeholder is now the measured p95.

## Deviations from the prompt (and why)

1. **History is unbounded, not 30 days.** The prompt asserts 30 days of history suffices. It does not:
   `n_prior_tx`, `is_cold_start` and the lifetime customer amount medians need the customer's **entire**
   prior history, and `accounts_seen_on_device` / `device_age_days` need the device's **entire** prior
   history across all customers. `db/queries.py` was extended with an optional `lookback_days`
   (`None` ⇒ no lower bound) plus a shared `_history_rows` helper; the provider requests the full
   prior window. Existing callers that pass an explicit integer are unchanged.
2. **Fraud-enriched sample in the latency benchmark.** Cells use 2,000 validation rows: **all 360
   validation-window frauds** (fewer than the 1,000 target) then a legit fill, so **360 fraud / 1,640
   legit = 18%** — about **20x** the ~0.89% validation prevalence. It is **not** a 1:1 sample. Seed 42
   keeps the row set stable across reruns; the enrichment changes the action mix, not per-request
   latency. All sampling is inside `[t_train_end, t_val_end)` (hard-asserted).
3. **In-process replay, not real HTTP.** The benchmark drives the app through
   `httpx.ASGITransport`; numbers are a lower bound (no TLS, no network, no uvicorn serialisation).
   Stated in `docs/latency.md` and `docs/api.md`.
4. **Headline latency cell = `postgres`, concurrency 1, explanation off.** The rule is recorded in
   `results/latency.json` (`headline_rule`) so the quoted figure is self-describing: postgres
   (deployment backend) × concurrency 1 (single-request serving) × explanation off (API default).
   **This cell was chosen after the latency matrix was measured, not pre-registered** — the earlier
   plan named `memory|conc1|explain-off`. Concurrency-8 cells are kept but not headlined because
   throughput *falls* when concurrency rises (see `what_went_wrong.md`).
5. **Benchmark-time DB provenance is recorded.** `results/latency.json` now carries
   `db: {expected_profile, transactions_rows, customers}` captured at benchmark time, so the postgres
   cells are provably measured against the claimed profile.
6. `reports/db_explain_dev.txt` was regenerated (timing-only noise) by a `db`-marked test; reverted so
   the commit diff stays meaningful.

## Commands run (actual output)

- `pytest -m db -q` (after the correction commit's new postgres parity test):
  **9 passed, 244 deselected in 10.20s** — includes the SQL leakage cross-check, the backward-compat
  query test, and the new postgres-vs-memory feature-parity test.
- `python scripts/load_db.py --profile dev --reset`:
  `Loaded 582,236 rows across 8 tables in 12.87s (profile=dev)`; `raw.transactions` **277,607 rows**,
  `raw.customers` **5,000** (the provenance recorded in `results/latency.json`).
- `pytest tests/test_api.py -q`:
  **20 tests** (19 non-db + 1 db). Parity printed
  `parity: n=220 max_abs_diff=0.000e+00` (API calibrated probability equals the offline pipeline
  exactly on the 220-row validation sample; threshold 1e-6 never approached) — this is the
  **dev/memory** path. The new `db`-marked `test_postgres_history_matches_memory_features` asserts
  **feature-level** equality (identical strictly-prior history ⇒ identical feature row) on tiny, where
  no calibrator/policy exists. The tamper test copies an artifact to a temp dir, corrupts it, asserts
  the app refuses to start (`RuntimeError`), and verifies the real artifact hash is unchanged.
- `pytest -q`: **252 passed, 1 skipped**.
- `python scripts/run_api.py --profile dev --history-backend memory` (smoke): start-up **0.68 s**
  (provider load 0.42 s); `examples/sample_request.json` (real val fraud `T0198776`, amount ~81,616)
  → `p=0.0783`, `raw=0.0644`, `risk_score 7.8`, `BLOCK`/`CRITICAL`, latency ~210 ms; `/model-info`
  returns `profile_note` + `base_rate 0.007648`.
- `python scripts/benchmark_latency.py --profile dev --n 2000 --warmup 100 --backends postgres,memory`
  → `results/latency.json` + `docs/latency.md`.
  - **Headline `postgres|conc1|explain-off`: p50 92.0 ms, p95 127.0 ms, p99 150.3 ms, 10.2 req/s**
    (history 46.4 ms p50 = DB round-trip; features 41.4 ms p50; SHAP 0.0).
  - Memory `conc1|explain-off`: p50 49.7 ms, p95 90.7 ms, 16.9 req/s (history ~0.8 ms).
  - History provider start-up: memory **0.409 s**, postgres ~0 s (queries per request).
  - 0 errors in every cell. Concurrency 8 degrades (see below).

## Offline validation-window action mix (context)

Whole validation window (40,585 rows): APPROVE 39,526 · STEP_UP 607 · REVIEW 91 · BLOCK 361. This
confirms the REVIEW cap (23/day) is not binding on validation volume; the capacity tests therefore
use a synthetic cap of 1 to exercise demotion.

## Base rate

`base_rate` is read at start-up from `data/raw/dev/simulation_report.json["fraud_rate"]` **0.007648**,
labelled **dev population rate** in the schema description, `docs/api.md`, and `/model-info`
(`base_rate_source`). No hard-coded copy.

## Open issues / limitations

- **Concurrency does not scale** in one Python process — logged as a finding in `what_went_wrong.md`
  with the per-component evidence; the GIL explanation is a **hypothesis**, not a proven cause.
- **Batch capacity is per IST day within the batch only** and caps **only REVIEW**; it is unaware of
  requests outside the batch unless `WRITE_PREDICTIONS=1` persists scores. Stated in every batch
  response (`capacity_note`) and in `docs/api.md`.
- In-process latency is a lower bound; a real uvicorn/HTTP measurement would be higher.
- Single profile (`dev`), no auth, no rate limiting — out of scope for a portfolio service.
