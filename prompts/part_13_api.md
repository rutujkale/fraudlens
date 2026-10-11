# Part 13 — FastAPI Scoring Service & Latency Benchmark

## Goal
Serve the frozen decision system: raw transaction in → features computed with the **same code as training** → raw score → calibrated probability → risk score → action → explanation. Measure latency by component and prove training/serving parity.

## Prerequisites
Parts 04–07, 10, 12 complete: frozen artifacts, Postgres loaded for `main` (or `dev`), `compute_features_for_transaction` available.

## Design

### Request flow
1. Validate input (Pydantic v2).
2. **History lookup:** fetch the customer's recent history (default 30 days + enough for 7d-vs-prev-7d trend = 14 days, so 30 is sufficient) strictly before `timestamp`, and the device's prior customer count, via a `HistoryProvider`:
   - `PostgresHistoryProvider` (uses `db/queries.py`, connection pool, the indexes from Part 04);
   - `InMemoryHistoryProvider` (loads parquet into a dict keyed by customer for demos/CI without a database).
   Selected by env `HISTORY_BACKEND=postgres|memory`.
3. `compute_features_for_transaction(...)` from `features/serving.py` (no feature logic in `api/`).
4. XGBoost raw score → calibrator → `risk_score`, `lift` → policy decision → `risk_level` derived from the action.
5. Optional SHAP explanation (`include_explanation=true`): top drivers + reasons from Part 10.
6. Optional write to `model.predictions` (env flag, off by default for benchmarking).

### Endpoints
- `GET /health` — status, model_version, history backend ok.
- `GET /model-info` — model_version, feature_version, calibration, policy_version, training window, validation/test metrics summary from `final_results.json`, cost assumptions, capacity.
- `POST /predict`
  Request: `customer_id`, `amount` (>0), `timestamp` (ISO-8601; tz-aware converted to IST, naive stored), `merchant_category` (enum), `device_id` (optional), `location_id` (optional), `transaction_type` (enum, default PURCHASE), `channel` (enum, default APP), `recipient_id` (optional), `transaction_id` (optional), `include_explanation` (bool, default false).
  Response: `transaction_id`, `fraud_probability` (calibrated), `raw_score`, `risk_score`, `lift_over_base_rate`, `risk_level`, `recommended_action`, `reasons` (list, only if requested), `mitigating_factors`, `model_version`, `feature_version`, `policy_version`, `latency_ms`, `timing_ms` (`history_fetch`, `features`, `model`, `calibration`, `policy`, `explanation`).
- `POST /predict/batch` — list of up to `max_batch` requests (config) **or** CSV upload (`/predict/batch/csv`); returns scored rows sorted by expected saving and applies the **daily capacity** allocation across the batch (using the policy from Part 07), with a summary: counts per action, estimated review workload, total expected loss avoided.
- Unknown `customer_id` ⇒ valid cold-start scoring (not an error); unknown `device_id` ⇒ new-device semantics; invalid enums/timestamps ⇒ 422 with clear messages.

### Engineering
- App factory with lifespan loading artifacts once (model, calibrator, policy, explainer, registry). Dependency injection in `api/deps.py`; schemas in `api/schemas.py`; strict response models; CORS off by default; structured logging with request id; no secrets in code (env).
- `scripts/run_api.py` starting uvicorn.

## Tasks

### 13A — Implementation (`api/main.py`, `schemas.py`, `deps.py`, `history.py`)
As specified. Keep handlers thin.

### 13B — Tests (`tests/test_api.py`, FastAPI `TestClient`, `memory` backend on the `tiny` profile)
- Valid request ⇒ 200 with schema-correct response; probability in [0,1]; action ∈ enum; `risk_level` consistent with the action.
- Invalid: negative amount, bad timestamp, missing `customer_id`, unknown category ⇒ 422.
- Cold-start customer ⇒ 200 and `is_cold_start` reflected in explanation.
- Batch: size limit enforced; capacity never exceeded per day; ordering by expected saving.
- **Training/serving parity (critical):** for ≥ 200 historical validation transactions, the API probability (history = rows strictly before the transaction) equals the offline pipeline's calibrated probability within 1e-6. Failure means feature skew; fix it before anything else and log it in `docs/what_went_wrong.md` if it occurs.
- **No future leakage via history provider:** the provider never returns rows with `timestamp ≥` request timestamp (also test with `postgres` backend under marker `db`).
- Determinism; latency fields present and non-negative; `/model-info` reflects the frozen artifact hashes.

### 13C — Latency benchmark (`scripts/benchmark_latency.py`)
- Sample 2,000+ real historical transactions (stratified, fixed seed) from the **validation** window (or any non-test sample); 100 warm-up requests excluded.
- Run sequentially (concurrency 1) and with 8 threads; both backends if available; with and without explanation.
- Report mean, p50, p95, p99 for each component and end-to-end, requests/sec, hardware description (CPU/RAM/OS), backend, model size. Write `results/latency.json` and `docs/latency.md`. Use the p95 end-to-end figure for the resume only if it comes from this file, and quote its conditions (backend, explanation on/off, concurrency).
- Update `reports/headline_numbers.md` (Part 12 placeholder) from this JSON via the existing script.

### 13D — Docs
`docs/api.md`: endpoints, sample curl commands, example JSON (use real output from a run), environment variables, how to start (`make api`).

## Acceptance criteria
```bash
pytest tests/test_api.py -q
HISTORY_BACKEND=memory python scripts/run_api.py &    # then:
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/sample_request.json
python scripts/benchmark_latency.py --profile main --backend memory
```
Progress file: parity test result, latency table by component, observations (what dominates latency), known limitations (e.g. history lookup cost, no feature store).

## Do not
Re-implement features in `api/`. Compute SHAP for batch requests by default. Report a latency number measured under unstated conditions. Use the test window for benchmarking samples.
