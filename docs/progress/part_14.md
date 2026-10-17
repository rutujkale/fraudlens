# Part 14 progress — Streamlit analyst app

**Status:** 14A (precompute) + 14B (`/customer/{id}/summary`) + 14C (app + smoke tests) + 14D (docs) complete.
· **Committed:** yes · **Pushed:** yes

The frozen system `xgb_dev_20261015_v1` (XGBoost + sigmoid calibrator + `cost_policy_v1`) is served
as-is. The **test window was never opened** by Part 14: every app artifact and every smoke run uses
the validation window only.

> Wording rule for this part: the app says **no** claim of "parity verified on postgres". The accurate
> sentence is *"postgres and memory providers return identical feature rows on a tiny sample;
> calibrated-probability parity (1e-6, n=220) is proven on dev/memory."*

## What was built

- `src/fraudlens/appdata/artifacts.py` — `build_app_artifacts(profile, out_dir, n_scored, n_shap, seed)`
  (writes `scored_sample.parquet`, `curves.json`, `psi_by_month.csv`, `presets.json`, `manifest.json`)
  and `load_app_artifacts(profile)` (missing build raises `FileNotFoundError` naming the script).
  Uses the package's `evaluation`, `risk`, `explain` and `monitoring` code; the `val_thr` slice only.
- `src/fraudlens/appdata/recompute.py` — thin recompute wrappers (`scan`, `best_thresholds`,
  `policy_metrics`, `queue_table`, `scenario_table`, `sample_arrays`, `threshold_grid`) delegating to
  the Part 07 package code.
- `scripts/make_app_artifacts.py` — thin CLI (`--profile`, `--out-dir`, `--n-scored`, `--n-shap`, `--seed`).
- `app/components/` — `api_client` (httpx + `set_client`/`get_client` test seam), `artifacts` (cached
  loader), `context` (sidebar, cost labels, `BANNER`, `CAPACITY_NOTE`, `get_config`), `charts` (Plotly),
  `limitations` (`docs/limitations.md` if present, else placeholder).
- `app/streamlit_app.py` + `app/pages/{1_Transaction_Analyzer,2_Review_Queue,3_Model_Performance}.py`.
- API: `GET /customer/{id}/summary` (`api/schemas.py::CustomerSummaryResponse`/`RecentTransaction`,
  `api/history.py::summarize_history`, `deps.Scorer.customer_summary`); `predict` now also returns
  `contributions` (per-feature SHAP on log-odds) so the UI never reloads the model.
- `tests/test_app_artifacts.py` (11 tests), `tests/test_app_smoke.py` (7 tests), 3 new API tests.
- `Makefile` — `app-artifacts` and `app` targets.
- Docs: `docs/app.md`, `docs/demo_script.md`, `docs/screenshots/README.md`, this file;
  `what_went_wrong.md` gained the tiny-validation-degeneracy finding.

## Command outputs (actual)

```
ruff check .                                    -> All checks passed!
python scripts/make_app_artifacts.py --profile dev
  -> app artifacts written to artifacts/app/dev (20293 sample rows)
python -c "build_app_artifacts('tiny', out_dir=<temp>/tiny, n_scored=2000, n_shap=200)"
  -> ... (243 sample rows)   # 2 of 4 presets; 0 frauds in tiny val
pytest tests/test_app_artifacts.py -q           -> 11 passed
pytest tests/test_app_smoke.py -q               -> 7 passed
pytest -m db -q                                 -> 10 passed, 257 deselected
pytest -q -m "not db"                           -> 264 passed, 1 skipped, 10 deselected
```

Manifest sanity (dev): `model_version xgb_dev_20261015_v1`, `policy_version cost_policy_v1`,
`calibrated true`, `capacity_per_day 23`, `n_scored_sample 20293`, `n_shap_explained 5000`, `seed 42`,
4 presets.

## Files each page reads

See `docs/app.md` for the full table. Summary: Page 1 = API (`/predict`, `/customer/{id}/summary`) +
`presets.json`; Page 2 = `scored_sample.parquet` + `manifest.json` through package recompute + the
batch CSV API; Page 3 = `results/final_results.json`, `results/final_model_comparison.csv`,
`results/val_comparison_<profile>.csv`, `results/holdout_<profile>.csv`,
`results/fairness_groups_primary_<profile>.csv`, `results/exp_09_unseen_decision_*.json`,
`results/latency.json`, `artifacts/app/<profile>/curves.json`, `artifacts/app/<profile>/psi_by_month.csv`
and the frozen `reports/figures/final_*.png`.

## Recompute path

The Page 2 sliders change cost/capacity inputs only; all math runs through
`fraudlens.appdata.recompute`, which calls `evaluation.cost_model`, `evaluation.threshold_search` and
`risk.policy`. `test_page2_recompute_uses_package_code` asserts `recompute.scan` equals
`scan_global_thresholds` on the same `val_thr` slice and that the cost-optimal threshold is its argmin.

## Deviations from the prompt (and why)

1. **`--profile dev`, not `main`.** The prompt's acceptance uses `--profile main`, but the API is frozen
   on `dev` and there is no `main` artifact set in this repo. The app defaults to `dev`
   (`FRAUDLENS_PROFILE` overrides).
2. **Frozen test curve points are absent.** `results/final_results.json` has no PR/ROC/calibration
   curve *points* for the test window, only scalar metrics and the calibration `bins`. Page 3 draws the
   validation curves from `curves.json` and the frozen test calibration curve from the stored test
   `bins`; the frozen test PR/ROC figures are shown as the committed `final_*.png`.
3. **`psi_by_month` is validation-only.** Test-month PSI needs the test window and is deferred to
   Part 15; Page 3 states this on screen.
4. **App artifacts are not committed.** They are built into `artifacts/app/<profile>/` (gitignored);
   the smoke test builds `tiny` into a temp dir, never `artifacts/app/`.
5. **Confusion-matrix selector is validation-only**, with the frozen test point shown beside it
   (derived from `decision.policies.amount_aware_capacity` + `composition`, calibrated scale).
6. **Page 2 recomputes on the `val_thr` slice only** (20,293 dev rows), not the full validation window.
7. **`predict` response gained `contributions`.** The prompt wants a SHAP bar on Page 1 but forbids
   model logic in `app/`; the API now returns the per-feature SHAP values it already computes for the
   reason sentences.

## Known gaps / open issues

- **`tiny` validation has 0 frauds**, so tiny curve artifacts are degenerate and only 2 of 4 presets
  exist. Tiny is CI-smoke only; the demo uses `dev`. (Recorded in `what_went_wrong.md`.)
- `docs/limitations.md` does not exist yet (Part 15); Page 3's Limitations expander shows a
  file-sourced placeholder until then.
- The app is single-page-per-process (no auth, no persistence beyond the API's decision log), by design
  for a portfolio demo.
