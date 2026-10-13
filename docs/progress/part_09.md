# Part 09 progress

**Status:** done (holdout: structuring) · **Committed:** (pending) · **Pushed:** no

## What was built

- `src/fraudlens/models/anomaly.py` — curated numeric behavioural set (28 features); `AnomalyModel`
  with `score = rank-normalised raw score` in `[0,1]`; `fit_anomaly()` (supports exclude-mask, records
  `n_fit_rows`); `customer_folds()` (grouped CV); `cross_fit_scores()` (prevents scoring own customer,
  test enforces); `grid_search()` (selects best IF by val PR-AUC on non-held-out fraud).
- `scripts/run_unseen_experiment.py` — E15/E16 for `--holdout`; builds x_va with anomaly column for
  S1/S1s and for the non-held-out val subsets (S1/S1s early-stopped on non-held-out val only); S0–S4,
  per-seed + seed-bagged paired CIs (cluster-bootstrap), recall@0.5/1/2/5%, per-scenario, figures,
  idempotent logging.
- `tests/test_anomaly.py` — 7 tests (cross-fit leakage guard, score range, fit_exclude_mask, rank
  monotonicity, grid runs, S4>0, S2-like sanity).

## Key results

- IF best params: n_estimators=400, max_samples=1024 (val PR-AUC 0.128).
- S2 (IF alone): structuring ROC-AUC 0.969; recall@1% **0.204**, @5% **0.823**.
- Paired (structuring): S1−S0 bagged +0.0291 [+0.0075,+0.0521] (per-seed +0.0099 ± 0.0107); S1s−S0
  bagged +0.0029 [−0.0065,+0.0123] (per-seed +0.0087 ± 0.0041). S3 weight tuned to 0.
- Adoption: **not adopted** (S1s shows no benefit, S1 gain < 0.02 per-seed, gain for S1 driven by
  seed bagging; gain requires detector to have seen the pattern unlabeled). No IF artifact saved.

## Commands run

```powershell
.venv\Scripts\python.exe scripts\run_unseen_experiment.py --profile unseen_lab --holdout structuring
.venv\Scripts\python.exe -m pytest -v tests\test_anomaly.py
.venv\Scripts\python.exe -m ruff check .
```

Actual summaries:
- Tests passed: 7/7 (test_anomaly).
- Script runtime ~35 min (grid/cross-fit + 5 seeds × S0–S4).
- Outputs: `results\exp_09_unseen*.csv/json`, `exp_09_if_*.csv`, `exp_09_paired_*`, figures in
  `reports\figures\exp_09_*`; E15 appended (duplicated from a prior run) and deduplicated to 6.

## Deviations (from prompt)

- **Pre-registration / adoption rule change.** The outcome categories (A/B/C) and an initial adoption
  rule were written down in the plan before running. After the **first run** (which saved an IF artifact
  under the initial rule and would have adopted S1), the adoption rule was **tightened** to require
  **both** S1 and S1s to show supported benefit and added a per-seed materiality threshold (gain ≥ 0.02);
  under that tightened rule the final run concluded **not adopted** (S1s bagged CI included 0,
  per-seed S1 gain < 0.02). The experiment doc (`09_unseen_fraud.md`) was finalized after the runs.
- Optional robustness sweep (hold out `geo_anomaly` / `amount_anomaly`) was **skipped**. Rationale:
  on `unseen_lab` those held-out scenarios have only 34 / 24 validation positives — far below ~300.
  Reporting them would be underpowered; `--holdout` is exercised by design. To run them properly
  requires a much larger regeneration.
- S2 (IF-alone) has no training variance across seeds (no XGB), so std/CI entries are NaN in the
  per-seed summary tables; this is consistent with reporting IF as the standalone detector.
- The adoption decision text reflects both: the seed-bagged S1 supports it, but per-seed < 0.02 and
  S1s bagged CI includes 0, so the conservative call is "not adopted" — the code's reason matches.

## Bugs / what_went_wrong

Added entry in `docs/what_went_wrong.md`: an early check (unused var from a column-append test)
  caught that S1/S1s would reference `anomaly_score` on `x_va` where it hadn't been added. Fixed by
  building `x_va_anom_all/strict` and the `x_va_nohold_*` variants used for early-stopping on
  non-held-out val.

## Open items

- Consider logging `model_version` for the "not adopted" case (even when no artifact). Not required.
- If decision were to adopt, the IF artifact would need to be saved. Currently not needed.
