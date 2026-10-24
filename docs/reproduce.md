<!-- generated from docs/reproduce_template.md by scripts/render_readme.py; do not edit docs/reproduce.md by hand -->
# Reproducing FraudLens

Every published result is reproducible from a seed and the committed
`config/config.yaml`. Primary results use the **`dev`** profile; `tiny` is the CI /
smoke profile and `main` was never generated (see `docs/limitations.md`).

## Prerequisites

- Python 3.11 / 3.12
- Docker Desktop (for PostgreSQL 16 via `docker-compose.yml`)
- `make setup` creates `.venv` and installs `requirements.txt`

`make` uses `PY ?= .venv/Scripts/python` on Windows. Set `PROFILE=dev` (the
default) or `PROFILE=tiny` to pick a profile.

## One-shot local CI (`make ci-local`)

Reproduces `.github/workflows/ci.yml` against the local Docker database:

```bash
make db-up          # start PostgreSQL 16
make ci-local       # ruff check + tiny pipeline + pytest -m "not slow and not needs_dev"
```

`ci-local` builds the **tiny** profile end to end (generate → validate → features →
train → load DB) and runs the tests that need no locally built `dev` bundle. Tests
marked `needs_dev` (`tests/test_api.py`, `tests/test_explain.py`) load the frozen
`dev` artifact and dev data, which are gitignored and built locally by the `dev`
pipeline below; they are excluded in CI.

## Full `dev` pipeline (`make all-dev`)

```bash
make db-up
make all-dev
```

`all-dev` expands to, in order:

```
simulate (generate_data) -> validate-data -> db-load -> build-features ->
train -> evaluate-val -> calibrate -> optimize-policy ->
experiment-8a 8b 8c 8d -> shap -> fairness -> app-artifacts
```

Then, to reproduce the published headline outputs:

```bash
make monitor        # per-period drift monitoring over train + validation
make render         # README.md, docs/model_card.md, docs/interview_notes.md from results
python scripts/final_audit.py
```

## Exact command sequence (what each step writes)

| Step | Command | Writes |
|---|---|---|
| raw data | `make simulate` | `data/raw/<profile>/` |
| quality | `make validate-data` | validation report |
| DB load | `make db-load` | `raw.*` tables in PostgreSQL |
| features | `make build-features` | `data/features/<profile>/`, `docs/data_dictionary_features.md` |
| train | `make train` | `artifacts/<model_version>/`, `results/models_<profile>.json` |
| validation eval | `make evaluate-val` | `results/<model>_<profile>_val*` |
| calibrate | `make calibrate` | calibrator into the artifact |
| policy | `make optimize-policy` | `policy.json`, `results/policy_<profile>*.csv` |
| experiments | `make experiments` | `results/exp_8A..8D*`, `docs/experiments/*.md` |
| SHAP | `make shap` | `results/shap_global_<profile>.csv`, figures |
| fairness | `make fairness` | `docs/fairness_audit.md` |
| app data | `make app-artifacts` | `artifacts/appdata_<profile>/` |
| monitoring | `make monitor` | `results/monitoring_<profile>.csv`, `results/monitoring_psi_<profile>.csv`, figures |
| sim-to-real | `make sim-to-real` | `results/sim_to_real.json` (primary R1−R2 gap only) |
| external report | `make external-report` | the generated tables in `docs/external_validation.md` (primary gap only) |
| render | `make render` | `README.md`, `docs/model_card.md`, `docs/interview_notes.md`, `docs/trade_offs.md`, `docs/reproduce.md` |

The locked test evaluation is **not** part of `all-dev`: it is a one-shot run
(`make final`) guarded by `evaluation/guard.py` and the access log. Reproducing the
published test numbers means re-running that guarded evaluation (it will refuse a
second unprompted run). See `docs/final_results.md`.

## Expected runtimes (measured on this machine: Windows, Python 3.12, Docker Desktop)

Rough figures — hardware differs; treat them as order-of-magnitude.

| Step | `tiny` | `dev` |
|---|---|---|
| generate | ~1 s | ~55 s |
| build features | seconds | ~1–2 min |
| train (all models) | ~10 s | ~1–2 min |
| calibrate + policy | seconds | < 1 min |
| experiment 8A (5 seeds) | — | ~35 min |
| cluster bootstrap in `make fairness` (fairness audit) | — | ~70 s |
| monitoring | — | ~1 min |
| CI (`make ci-local`, tiny) | target < 10 min | — |

## Expected rough outputs

- `results/final_results.json` — the four pre-registered outcomes and the decision
  tables. XGBoost test PR-AUC 0.9619 (CI 0.9340-0.9798); cost reduction
  773,067 INR (83.23% point estimate), CI 361,480 to 1,281,632. The percentage is a **point estimate**: the CI
  (361,480-1,281,632) applies to the INR figure, not the ratio.
- `results/monitoring_dev.csv` — one row per month with a `sample` column
  (`in_sample` / `validation` / `mixed`), prevalence, calibration, PR-AUC and recall at
  the frozen policy (train + validation only; only `validation` rows are out-of-sample).
- `results/sim_to_real.json` — the primary R1−R2 gap only (no cross-dataset differenced
  lift; the raw record stays in `results/external_gap.csv`).
- `README.md` — no hand-typed numbers: every figure is a template slot filled from
  `results/final_results.json` and `results/latency.json` by `scripts/render_readme.py`
  (as are `docs/model_card.md`, `docs/interview_notes.md`, `docs/trade_offs.md` and
  this file).

## Seeds

`seed: 42` (`config/config.yaml`) drives the simulator, splits, model training,
bootstrap and the anomaly experiment. Same seed + same config ⇒ identical output.
Every artifact records `seed`, `config_hash`, `git_commit` and its
`training_window` in `metadata.json`.

## Regenerating the `main` profile

`main` was intentionally **not** generated for this build, because generating it
would create a new dataset and test window after the freeze, invalidating the
frozen hashes and the "one test window, evaluated once" discipline
(`docs/limitations.md`). If you want a larger run, on your own copy:

```bash
PROFILE=main make all-dev
```

Do this **before** freezing; it produces ~20,000 customers over 270 days
(`config/config.yaml:profiles.main`) and a fresh, unpublished test window.
