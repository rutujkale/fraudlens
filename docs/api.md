# Scoring API (Part 13)

FastAPI service that scores a transaction through the **frozen** bundle
(`artifacts/xgb_dev_20261015_v1/`: XGBoost model + sigmoid calibrator + `policy.json`)
using the same point-in-time feature code as training (`features/serving.py`).

> **Scope.** This is a portfolio/research service, not a production banking API. It runs the
> **`dev`** profile (the frozen system is dev, not `main` — `main` was never generated), on
> synthetic data. No metric here implies production readiness, and the frozen system must not be
> retrained or re-tuned.

## Run

```powershell
make db-up                                  # PostgreSQL 16 (only needed for HISTORY_BACKEND=postgres)
make api HISTORY_BACKEND=memory             # uvicorn on 127.0.0.1:8000
# or directly:
.venv/Scripts/python scripts/run_api.py --profile dev --history-backend memory
```

Interactive docs at `/docs`. `examples/sample_request.json` is a real validation-window fraud row.

Environment variables: `FRAUDLENS_PROFILE` (default `dev`), `HISTORY_BACKEND` (`memory`|`postgres`,
default `memory`), `WRITE_PREDICTIONS` (default off).

## Frozen-bundle verification (refuse to start on mismatch)

At start-up the scorer recomputes the sha256 of every frozen artifact and compares it with
`results/freeze.json` (`evaluation/locked.py::verify_artifact_hashes`). On any mismatch the process
**refuses to start** (`RuntimeError`). This is covered by the tamper-refusal test, which copies an
artifact to a temp dir, corrupts it, and asserts the app fails to boot while the real file is
untouched (`tests/test_api.py::test_startup_refuses_on_tampered_artifact`).

`GET /model-info` reports the freeze hashes (freeze git commit, config hash, dataset version, model
versions), the val/test metrics read from the published `results/final_results.json`, the cost
assumptions, capacity, base rate and `model_size_bytes`. The test-window numbers it shows are the
frozen Part 12 results — the API never re-reads the test window.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + model version + history backend health |
| GET | `/model-info` | frozen-bundle metadata, freeze hashes, published metrics, cost/capacity |
| POST | `/predict` | score one transaction (no capacity — that is a batch concept) |
| POST | `/predict/batch` | score up to `api.max_batch` (1000) transactions, apply the REVIEW cap |
| POST | `/predict/batch/csv` | same, from a CSV upload |

`/predict` returns the calibrated `fraud_probability`, `raw_score`, `risk_score` (0–100),
`risk_level`, `recommended_action`, optional `reasons`/`mitigating_factors`, and per-component
`timing_ms`. `extra="forbid"` and `amount > 0` are enforced; a timezone-aware timestamp is converted
to IST and made naive, matching the training convention. Invalid input returns 422.

## History backend

`HISTORY_BACKEND` selects how prior transactions are fetched for the point-in-time features.

- **`postgres`** — queries `raw.transactions` per request (`db/queries.py`, unbounded lookback:
  the target customer's *entire* prior history and the target device's *entire* prior history across
  all customers, both strictly before the transaction timestamp).
- **`memory`** — loads all dev transactions once at start-up into pandas and serves history from
  in-memory indices. Needs no database; **start-up cost** on this machine was **0.50 s** (dev,
  277,607 transactions).

`/health` reports `history_ok` so a missing/unreachable database is visible without a stack trace.
Both backends return the identical feature vector — the parity test asserts the API's calibrated
probability matches the offline pipeline to within 1e-6 (measured: 220 validation rows,
`max_abs_diff = 0.000e+00`).

## Batch and analyst-capacity semantics

- `/predict` is **per-transaction and applies no capacity**. Capacity is a batch/day concept.
- `/predict/batch` scores every row, applies the amount-aware cost policy, then `enforce_capacity`.
  The cap is **per IST day and only caps REVIEW**: `BLOCK` and `STEP_UP` are chosen by expected cost
  and are **never** bounded by the cap. Rows are returned sorted by expected saving (largest first),
  and the summary reports `counts_by_action`, `review_workload`, `n_demoted` and
  `total_expected_loss_avoided`.
- The cap is **not aware of requests outside the batch** unless predictions are persisted
  (`WRITE_PREDICTIONS=1` writes to `model.predictions`; off by default). This is a real limitation for
  a multi-instance deployment and is stated in every batch response (`capacity_note`).

## Base rate and lift

`base_rate` is read at start-up from `data/raw/dev/simulation_report.json["fraud_rate"]` = **0.007648**
— the **dev population rate**, not a production base rate. `lift_over_base_rate` = `p / base_rate`
carries that label in the response schema and here; it is a dev-population statistic.

## Explanations

SHAP explanations are **off by default** and opt-in per request (`include_explanation: true`);
batch endpoints never explain. When on, the explainer is built lazily (honouring the model's
`best_iteration`) and adds **~6 ms (memory) / ~10 ms (postgres)** to p50 latency.

## Latency

Measured on this machine (16-core AMD, 15.3 GB RAM, Python 3.12, in-process ASGI replay):

- **Headline (postgres, concurrency 1, explanation off): p50 92 ms, p95 127 ms** over 2,000
  validation transactions, on `dev` (277,607 rows) — profile `dev`, PostgreSQL history backend,
  concurrency 1, explanation off. History (the DB round-trip) is 46.4 ms p50, feature build 41.4 ms
  p50. This cell was **chosen after the matrix was measured** (see `headline_rule` in
  `results/latency.json`), not pre-registered.
- Sample is **fraud-enriched, not balanced**: all 360 validation frauds + 1,640 legit (18%, ~20x
  validation prevalence).
- Memory backend: p50 50 ms, p95 91 ms (history ~1 ms).
- Raising concurrency to 8 does **not** help — the feature build is the bottleneck and the most likely
  cause is the GIL serialising it in this single-process replay (**hypothesis**), so p95 rises ~15x.
  Horizontal scaling needs multiple uvicorn workers.

Full matrix, hardware and conditions: [`docs/latency.md`](latency.md) (from `results/latency.json`).

## Guarantees and limitations

- **Frozen system, no retraining.** The service loads the frozen bundle; it never trains, calibrates
  or re-optimises the policy.
- **Test window stays closed.** The API reads only published Part 12 results; parity and latency were
  measured on the validation window only.
- **In-process latency is a lower bound.** `docs/latency.md` numbers exclude the network stack, TLS and
  uvicorn serialisation.
- **No auth, no rate limiting, single profile.** Out of scope for a portfolio service.
