# Part 10 progress

**Status:** done (SHAP explainability + subgroup fairness audit) · **Committed:** `2751cc5` (code/tests/scripts) and `a74c922` (docs/results/figures) · **Pushed:** yes (`915cd57..a74c922`)

## What was built

- `src/fraudlens/explain/shap_explainer.py` — `ShapExplainer` around
  `shap.TreeExplainer(model, model_output="raw")` (`expected_value`, `raw_margin`,
  `shap_values`, `interaction_values`, `check_additivity`, `explain_global`, `explain_local`,
  `top_drivers`) and `stratified_val_sample(y, max_frauds, max_legit, seed)`. The explainer is
  rebuilt on load (not pickled) and honours `best_iteration` in both SHAP and `raw_margin`.
- `src/fraudlens/explain/reasons.py` — 53-feature template map (`format_reason`,
  `fallback_reason`, `reasons_for`); NaN-safe ("no prior history"), duplicate-sentence dedupe
  (e.g. `amount` vs `log_amount`), grammatical plurals, encoded-code note for the 3 categoricals.
- `src/fraudlens/explain/plots.py` — global bar, beeswarm, dependence (binary features drawn as
  jittered strips), waterfall (from a `shap.Explanation`); every figure carries the
  usage-vs-utility caveat.
- `src/fraudlens/explain/gallery.py` — deterministic, pre-registered case selection with pool
  sizes; ties break on `transaction_id`.
- `src/fraudlens/explain/fairness.py` — `load_audit_attributes`, `group_table` (per-group metrics
  with min-positives/min-rows suppression + cluster-bootstrap CIs), `metric_ratios`,
  `paired_group_diff` (shared customer resample), `proxy_check` (Spearman/eta²/Cramér's V).
- `src/fraudlens/explain/report.py` — markdown renderers for the two generated docs (numbers come
  from the same frames that are written to `results/`, so docs cannot drift from CSVs).
- `scripts/run_shap.py` (10A/C/D + gallery + latency + E17), `scripts/fairness_audit.py` (10E),
  `tests/test_explain.py` (18 tests), `Makefile` targets `shap`, `fairness`, `explain`.
- Shared plumbing: `models/io.py::load_serving_bundle` (model + calibrator + policy) and
  `evaluation/experiments.py::drop_experiment_rows` (idempotent experiment logging).
- Docs: generated `docs/explanation_gallery.md`, `docs/fairness_audit.md`; hand-written
  methodology sections (SHAP-is-margin, simulated-gaps caveat).

## Key results (profile `dev`, model `xgb_20261011_v1`, validation only)

- **Global SHAP** on 15,342 sampled validation rows (342 frauds + 15k legit). Top-10 mean|SHAP|:
  `tx_from_device_prior` 1.0851, `merchant_category` 1.0264, `minutes_since_prev_tx` 0.8258,
  `new_merchant_flag` 0.7619, `device_count_30d` 0.7461, `hour_deviation` 0.6636, `amount` 0.5240,
  `is_new_recipient` 0.5132, `hour` 0.4929, `region_freq_30d` 0.3897.
- **Usage ≠ utility:** `tx_from_device_prior` is rank 1 by mean|SHAP| but only 0.0085 permutation
  PR-AUC drop (8D); the largest utility drop belongs to `is_new_recipient` (0.0513, SHAP rank 8).
  Both columns are in `results/shap_global_dev.csv`.
- **Usage ≠ utility (8D triangulation).** The two rankings agree at the top of the *utility* list —
  `is_new_recipient` 0.0513, `minutes_since_prev_tx` 0.0457, `new_merchant_flag` 0.0408 are both the
  strongest permutation drops and top-8 by mean|SHAP| — and disagree exactly where it matters:
  `tx_from_device_prior` and `merchant_category` are rank 1–2 by mean|SHAP| but only 0.0085/0.0279 by
  permutation drop. The Part 08 control story repeats: `_random_noise` ranks 26/55 by mean|SHAP| while
  contributing nothing (perm drop −0.000144). Where they disagree, the permutation column wins.
- **Additivity:** `|ΣSHAP + base − raw_margin|` = 1.65e-5 over 200 rows; using all 600 trees
  (ignoring `best_iteration=592`) is off by 0.108.
- **Noise-control regression** (from the recorded 8D evidence): `_random_noise` perm drop
  −0.000144 (mean|SHAP| 0.1408), `_shuffled_minutes_since_prev_tx` −0.000260 (0.2342). Test asserts
  both ≤ 0; ranks are logged but not thresholded (the 8D 55-column model was never persisted).
- **Latency** (warm, single-row `explain_local`, 200 calls): p50 7.44 ms, p95 8.71 ms, max 9.60 ms
  → `results/shap_latency_dev.json`.
- **Gallery** (pool sizes in `results/shap_gallery_dev.csv`): clear_tp 101, subtle_tp 29,
  false_negative 5, hard_negative_fp 17, high_amount_tn 180, cold_start 44. The missed fraud is the
  immediately preceding transaction of the same episode as `subtle_tp` (same customer, 02:05 vs 02:06)
  — the rules were fixed before selection, so this coincidence is reported, not curated away.
- **Fairness (primary `val_thr`, 122 frauds / 20,657 rows):** 19 groups, only **5 fully reported**
  (age_band 1/4, income_band 2/4, archetype 1/5, region 1/6). FPR ratio is computable only for
  income_band: `fpr_any` 1.964 (high 0.0123 vs mid 0.0063), `flag_rate_any` 1.687, `recall` 1.062.
  11 of 39 group-vs-reference metric differences have a cluster-bootstrap CI excluding 0 (all FPR/flag
  rate; recall differences are never distinguishable with 122 frauds). These are **uncorrected for
  multiple comparisons** — 39 tests at 95% would yield ≈2 chance positives and the tests share
  bootstrap draws, so they are exploratory signals (stated in `docs/fairness_audit.md`), not confirmed
  disparities.
- **Proxy check:** 29 of 212 feature×attribute pairs cross the pre-registered thresholds — all 19
  `archetype` pairs (behavioural features encode archetype; `n_prior_tx` eta² 0.817), 6 `age_band`
  (amount features, |ρ| up to 0.467), 4 `income_band`, **0** `home_region_group`.

## Commands run

```powershell
.venv\Scripts\python.exe scripts\run_shap.py --profile dev
.venv\Scripts\python.exe scripts\fairness_audit.py --profile dev
.venv\Scripts\python.exe -m pytest -q -m "not db and not slow"
.venv\Scripts\python.exe -m ruff check .
```

Actual summaries:
- `run_shap`: ~9 s end-to-end (global SHAP 8 s, gallery/latency <2 s); wrote 5 `results/shap_*` files
  (`shap_global_dev.csv`, `shap_scenario_dev.csv`, `shap_dependence_dev.csv`, `shap_gallery_dev.csv`,
  `shap_latency_dev.json`), 11 figures (`shap_global_bar`, `shap_beeswarm`, 3× `shap_dependence_*`,
  6× `shap_waterfall_*`), `docs/explanation_gallery.md`, 1 legible `E17` row.
- `fairness_audit`: ~75 s (cluster bootstrap dominates); wrote 6 `results/fairness_*` files,
  `docs/fairness_audit.md`, 1 `E18` row. (Bootstrap is also the slowest part to re-run.)
- Tests: `pytest -q -m "not db and not slow"` → **183 passed, 8 deselected**; full `pytest -q` →
  **184 passed, 7 skipped** (the extra pass is a `db`-marked test). `ruff check .` clean.
- `results/experiments.csv`: 56 → 58 data rows (E17, E18; `git diff` shows exactly +2 lines). No other
  tracked result/figure changed (S0's frozen dev artifacts were read, never written).

## Test formula / model for every result

Every number above is written by the scripts to `results/*.csv|json` and `reports/figures/*.png`;
nothing is hand-edited. Test-window guard stayed closed (`TestSetGuard`); the calibrator/thresholds
were **not** refit; the model was not retrained or altered.

## Deviations (from prompt)

- **`plots.py`, `gallery.py`, `report.py` are extra files.** The prompt lists the gallery and figures
  as deliverables but not their modules; splitting them keeps `run_shap.py` a thin CLI and keeps doc
  rendering next to the tables it renders.
- **Gallery cold-start rule is `0 <= history_days < 2`,** not `history_days < 2` (see the finding
  below): negative history is a simulator defect, not a cold start.
- **Gallery rule constants — when they were set (exact chronology).** The shipped values
  (`CLEAR_TP_P = 0.95` with `0.90` fallback, `STEALTH_PCT = 75`, `TRAVEL_PCT = 90`,
  `AMOUNT_PCT = 99`, `COLD_START_HISTORY_DAYS = 2`, `0 <= history_days` lower bound) were written into
  `gallery.py` **before any case was selected** — the first execution of `run_shap.py` produced the
  gallery with these constants already in place, so the *selection* was not tuned on the selected rows.
  They do differ from an earlier plan **draft** that mentioned p ≥ 0.99 / stealth ≥ p90 / history < 3d;
  the draft numbers were never in code, and the shipped ones are deliberately **less** selective so the
  pool is not near-empty. One rule was tightened **after** seeing output: the cold-start rule, initially
  `history_days < 2` as drafted, was changed to `0 <= history_days < 2` because the first selection
  returned a row with `history_days = -11.87` (the simulator defect below). That post-hoc change is a
  researcher degree of freedom taken after observing one selected row; it narrows the pool to genuinely
  opened accounts, it is disclosed here, and it is the only rule change made after seeing a case. The
  final values are printed in `docs/explanation_gallery.md` and in `results/shap_gallery_dev.csv`.
- **Dependence colour partner** comes from a 500-row `shap_interaction_values` subsample (10 s), not a
  pre-chosen feature; if that call ever fails the code falls back to the next global feature. No
  failure was observed.
- **Exploratory fairness view excludes calibration gap and Brier** (calibrator saw half the window),
  as planned; the primary view's calibration gap/Brier are out-of-sample and kept.
- **`E17`/`E18` rows leave `train_window`/`val_window`/`test_window` empty,** matching the existing
  E12/E15 rows in `experiments.csv`.
- **Random-noise SHAP is not near zero, and the test checks utility instead.** Prompt 10A suggests a
  "noise feature SHAP ≈ 0" sanity check; measured `mean|SHAP|` for `_random_noise` is 0.1408 (rank
  26/55 by usage). Treating that as failure would contradict the honesty rule (mean|SHAP| measures
  usage). The test therefore asserts the recorded 8D evidence: both controls have
  `perm_pr_auc_drop ≤ 0`, ranks are logged but not thresholded, and the doc cites 8D.
- **`reasons_for` returns a `Reasons` dataclass, not `list[str]`.** Positive and mitigating factors must
  be separated (the prompt asks for both), so the return type is `Reasons(raises_risk, lowers_risk)`
  rather than a flat list.
- **Six `results/fairness_*_<profile>.csv` files** instead of one `results/fairness_<profile>.csv`
  (groups × primary/exploratory, suppression, ratios, pairwise, proxy), so each table has one stable
  schema; the doc reads the same frames.
- **Gallery selection is rule-based, not seeded.** Pre-registered thresholds plus
  `transaction_id` tie-breaks make selection reproducible without an RNG, which is stronger than a
  fixed seed (documented in `gallery.py`).

## Bugs / what_went_wrong

- New entry: **cold-start customers have transactions before their `account_open_date`**
  (`history_days < 0`; 12,329 of 289,870 dev rows, 468 customers). Detected while selecting the
  gallery cold-start case (the row printed `history_days = -11.87`). Not fixed here: regenerating
  `data/raw/dev` would invalidate the frozen artifact this part exists to explain. Fix path and
  deferred data-quality assertion are recorded in `docs/what_went_wrong.md`; the gallery rule and the
  generated doc both state the caveat.
- Re-used the Part 09 lesson on logging: `drop_experiment_rows` is now in
  `evaluation/experiments.py` and called by both new scripts (re-runs no longer duplicate E17/E18).
  Part 08/09 scripts keep their own dedupe; they were left untouched.

## Part 08 / 09 blast radius (S0 repair)

**No.** The S0 artifact-clobber incident (tiny run overwriting the dev calibrator/policy on
2026-10-12 19:59) was checked before this part: no Part 08/09 output produced after that timestamp
read the clobbered files (8A–8E fit calibrators in memory; Part 09 loaded no artifact), and the dev
calibrator/policy were regenerated with `--profile dev` and asserted back to the committed numbers.
All Part 10 analysis reads the repaired `xgb_20261011_v1` bundle.

## Open items

- **Simulator fix requires regeneration** (Part 10 knowingly left it): gate transactions on
  `account_open_date`, regenerate all profiles, re-run Parts 04–09, add the `history_days >= 0`
  data-quality assertion. Until then the dataset under-tests cold start.
- **`mean|SHAP|` vs utility divergence** is documented but not modelled; if Part 12 wants a single
  importance ranking it should use the 8D permutation drops, not `mean|SHAP|`.
- **Fairness power:** with 122 primary frauds, 14 of 19 groups are suppressed and recall differences
  are never distinguishable. Reporting more requires a larger regeneration, not a looser min-positives
  threshold.
- **Bootstrap runtime** (~70 s for the audit) will grow with `bootstrap.n`; consider `n_boot` per-profile
  (e.g. 200 on `dev`) if Part 15 reruns the audit often.
