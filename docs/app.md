# Analyst console (Part 14)

A thin, three-page Streamlit app over the Part 13 API and the Part 12 result files.
It is a decision demo, not a model playground: **no feature, scoring or cost logic
lives in `app/`**. Live scores come from the API; offline views come from
precomputed files.

> **postgres and memory providers return identical feature rows on a tiny sample;
> calibrated-probability parity (1e-6, n=220) is proven on dev/memory.**

## Run it

```bash
make db-up                        # PostgreSQL (only needed for the postgres history backend)
make app-artifacts PROFILE=dev    # scripts/make_app_artifacts.py -> artifacts/app/dev/ (~45 s)
make api                          # scripts/run_api.py; HISTORY_BACKEND default memory
make app                          # streamlit run app/streamlit_app.py
```

`FRAUDLENS_PROFILE` selects the profile (default `dev`); `FRAUDLENS_APP_ARTIFACTS_DIR`
overrides the artifacts root; `FRAUDLENS_API_URL` overrides the API base URL.

## Pages and what each reads

| Page | Live (API) | Offline files |
|------|-----------|----------------|
| **1. Transaction Analyzer** | `POST /predict` (with `include_explanation`), `GET /customer/{id}/summary` | `artifacts/app/<profile>/presets.json` |
| **2. Review Queue & Cost Optimization** | `POST /predict/batch/csv` (upload) | `scored_sample.parquet`, `manifest.json`, `results/threshold_scan_dev.csv` semantics via package code |
| **3. Model Performance & Drift** | — | `results/final_results.json`, `results/final_model_comparison.csv`, `results/val_comparison_<profile>.csv`, `results/holdout_<profile>.csv`, `results/fairness_groups_primary_<profile>.csv`, `results/exp_09_unseen_decision_*.json`, `results/latency.json`, `artifacts/app/<profile>/curves.json`, `artifacts/app/<profile>/psi_by_month.csv`, `reports/figures/final_*.png` |

Sidebar on every page: profile, model version, policy version, cost assumptions,
capacity note (*"Capacity caps REVIEW only (per IST day); BLOCK and STEP_UP are
chosen by expected cost and are never bounded."*) and the banner
*"Synthetic data demo — not a production fraud system."*

## Code layout

```
app/
├── streamlit_app.py            # entry point (3-line sys.path shim)
├── pages/
│   ├── 1_Transaction_Analyzer.py
│   ├── 2_Review_Queue.py
│   └── 3_Model_Performance.py
└── components/
    ├── api_client.py           # httpx client + set_client/get_client (test seam)
    ├── artifacts.py            # cached load of artifacts/app/<profile>/
    ├── context.py              # sidebar + config + capacity/banner constants
    ├── charts.py               # Plotly figures (pure plotting)
    └── limitations.py          # docs/limitations.md if present, else placeholder
```

Every page starts with a three-line `sys.path` shim inserting `app/` (pages use
`parents[1]`, the entry uses `parent`) so `from components import ...` resolves both
under `streamlit run` and under `streamlit.testing.v1.AppTest`.

## Precomputation (`scripts/make_app_artifacts.py`)

Writes to `artifacts/app/<profile>/` (gitignored, **never committed**):

- `scored_sample.parquet` — the `val_thr` slice (later half of validation) with raw
  score, calibrated probability, risk score, action, risk level, archetype,
  precomputed top reason (top-k rows by score) and the true label (rendered only
  behind the "show labels" toggle). `--n-scored`/`--n-shap`/`--seed` control size.
- `curves.json` — validation PR/ROC/calibration points and recall@K per model, the
  frozen test calibration bins (`results/final_results.json`), and the validation
  score histogram.
- `psi_by_month.csv` — per-feature PSI of each **validation** month against train.
- `presets.json` — four fixed-seed example transactions (normal purchase, suspicious
  burst, travel hard negative, structuring-like) with enough fields for a valid
  `/predict` call, including `customer_id` so history exists.
- `manifest.json` — profile, model/policy version, `calibrated`, costs, capacity,
  window boundaries, sample sizes, seed, `generated_at`.

Business logic lives in `fraudlens.appdata.artifacts` / `fraudlens.appdata.recompute`;
the script is a thin CLI, and the pages only *render* what those return.

## Recomputation on the sliders (Page 2)

The sliders change cost/capacity inputs only. Every recompute goes through
`fraudlens.appdata.recompute`, which calls the Part 07 package code
(`evaluation.cost_model`, `evaluation.threshold_search`, `risk.policy`):

- `policy_metrics` — amount-aware actions + `enforce_capacity`, expected and realized
  cost breakdown, counts by action, review workload/day, `n_demoted`.
- `best_thresholds` — best global thresholds (F1 / cost / cost-feasible) with the
  realized cost from `scan_global_thresholds` + `op_at_threshold`.
- `scan` — the global-threshold scan (expected-loss curve).
- `queue_table` — per-row capacity-capped action and expected saving.
- `scenario_table` — the four preset cost scenarios on the same sample.

`tests/test_app_smoke.py::test_page2_recompute_uses_package_code` asserts
`recompute.scan` equals `scan_global_thresholds` on the same `val_thr` slice and that
the cost-optimal threshold is its argmin — the page cannot drift from Part 07.

## Testing

```bash
pytest tests/test_app_smoke.py -q
```

`AppTest` runs each page against `tiny` artifacts built into a temp dir, with a fake
API client whose responses are built from `fraudlens.api.schemas` and `.model_dump()`ed
(schema drift fails the test). The tests assert: pages compile and run without
exceptions; the API-down message names `scripts/run_api.py`; the missing-artifact
message names `scripts/make_app_artifacts.py`; Page 2 hides `is_fraud` and the
realized-cost KPI by default; and the recompute path equals the package code.

## Graceful errors

- API down → `st.error` with the reason and `st.info` naming the start command.
- Missing artifacts → the page stops and names `scripts/make_app_artifacts.py`.
- Profile models without a calibrator/policy (e.g. `tiny`) → sidebar warning; raw
  scores are treated as probabilities (`manifest["calibrated"] = false`).
