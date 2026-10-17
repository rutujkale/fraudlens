# Part 13 progress — FastAPI scoring service

**Status:** 13A (API package) + 13B (tests) + 13C (latency benchmark) + 13D (docs) complete.
· **Committed:** yes (2 commits: api code+tests; docs+results) · **Pushed:** yes

Frozen system served as-is: `xgb_dev_20261015_v1` (XGBoost + sigmoid calibrator +
`policy.json`). No retraining, no re-calibration, no policy re-optimisation. The **test window was
never opened** by Part 13.

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
- `tests/test_api.py` — 19 tests (schema/validation, tz-normalisation, cold-start, batch size limit,
  REVIEW-only capacity cap, capacity-never-touches-BLOCK/STEP_UP, ordering, determinism, memory +
  postgres history strictness, **parity**, **tamper refusal**).
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
2. **Balanced sample in the latency benchmark.** Cells use 2,000 validation rows at a 1:1
   fraud/legit ratio (seed 42) so the mix is stable across reruns; this changes the action mix, not
   per-request latency. All sampling is inside `[t_train_end, t_val_end)` (hard-asserted).
3. **In-process replay, not real HTTP.** The benchmark drives the app through
   `httpx.ASGITransport`; numbers are a lower bound (no TLS, no network, no uvicorn serialisation).
   Stated in `docs/latency.md` and `docs/api.md`.
4. **Headline latency cell = `postgres`, concurrency 1, explanation off.** Chosen as the conservative,
   deployment-like single-request number; concurrency-8 cells are kept but not headlined because
   throughput *falls* when concurrency rises (see `what_went_wrong.md`).
5. `reports/db_explain_dev.txt` was regenerated (timing-only noise) by a `db`-marked test; reverted so
   the commit diff stays meaningful.

## Commands run (actual output)

- `pytest -m db` (after the `queries.py` change, before the dev reload):
  **7 passed, 226 deselected in 18.39s** — includes the SQL leakage cross-check and the
  backward-compat query test.
- `python scripts/load_db.py --profile dev --reset`:
  `Loaded 582,236 rows across 8 tables in 12.30s (profile=dev)` (wall **15.29 s**).
- `pytest tests/test_api.py -q`:
  **19 passed in 25.53s**. Parity printed
  `parity: n=220 max_abs_diff=0.000e+00` (API calibrated probability equals the offline pipeline
  exactly on the 220-row validation sample; threshold 1e-6 never approached). The tamper test copies
  an artifact to a temp dir, corrupts it, asserts the app refuses to start (`RuntimeError`), and
  verifies the real artifact hash is unchanged.
- `pytest -q`: **251 passed, 1 skipped in 55.70s**.
- `python scripts/run_api.py --profile dev --history-backend memory` (smoke): start-up **0.68 s**
  (provider load 0.42 s); `examples/sample_request.json` (real val fraud `T0198776`, amount ~81,616)
  → `p=0.0783`, `raw=0.0644`, `risk_score 7.8`, `BLOCK`/`CRITICAL`, latency ~210 ms; `/model-info`
  returns `profile_note` + `base_rate 0.007648`.
- `python scripts/benchmark_latency.py --profile dev --n 2000 --warmup 100 --backends postgres,memory`
  → `results/latency.json` + `docs/latency.md`.
  - **Headline `postgres|conc1|explain-off`: p50 90.4 ms, p95 124.0 ms, p99 142.6 ms, 10.3 req/s**
    (history 45.7 ms p50 = DB round-trip; features 40.6 ms p50; SHAP 0.0).
  - Memory `conc1|explain-off`: p50 63.6 ms, p95 113.2 ms (history ~1 ms).
  - History provider start-up: memory **0.504 s**, postgres ~0 s (queries per request).
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
  with the per-component evidence.
- **Batch capacity is per IST day within the batch only** and caps **only REVIEW**; it is unaware of
  requests outside the batch unless `WRITE_PREDICTIONS=1` persists scores. Stated in every batch
  response (`capacity_note`) and in `docs/api.md`.
- In-process latency is a lower bound; a real uvicorn/HTTP measurement would be higher.
- Single profile (`dev`), no auth, no rate limiting — out of scope for a portfolio service.
