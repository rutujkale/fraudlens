# Part 02b — Cold-start fix and regeneration

**Status:** in progress (fix + tests landed; regeneration pending)
**Tag:** `pre-coldstart-fix` → `3ff9577` (pushed) marks the last commit whose data may contain
transactions before `account_open_date`
**Data version label:** `pre_coldstart_fix` (old rows) → `post_coldstart_fix` (config
`common.dataset_version`, new rows)

## Why this exists

`data_gen/entities.py` gives ~10% of customers a mid-window `account_open_date`, but **no
generator read it**, so customers transacted before their account existed. On dev that was 12,329 of
289,870 rows (4.25%) across 486 customers, giving negative `history_days` and a cold-start cohort that
was not actually cold. Documented in `docs/what_went_wrong.md` (Part 10). This part fixes the
simulator and regenerates everything downstream.

## Decisions (user-approved)

1. Episode placement: **redraw the day** inside the customer's valid window, and **fall back to a
   same-archetype customer** when the drawn customer's account opens after the last valid episode
   start. Both counts are logged per profile.
2. `results/experiments.csv` gains a **`dataset_version`** column; old rows keep
   `pre_coldstart_fix`, re-runs replace only their own version's rows (8E and Part 09 scripts now
   filter on it). Old rows are history, never deleted.
3. Profiles regenerated: `tiny`, `dev`, `drift_lab`, `unseen_lab`. `main` **has never been
   generated** (20k customers / 270 days) and stays out of scope.
4. No config/seed change to the simulation. The only config addition is the non-simulation
   `dataset_version` label.

## Pre-registered before/after metric list

Written **before** regenerating, so the comparison cannot be curated after the fact. Every value is
read from the generated artefacts (`results/*.csv|json`, `simulation_report.json`), never typed by
hand. The **pre** column is taken from the tracked results at tag `pre-coldstart-fix`
(`git show pre-coldstart-fix:results/...`) — no snapshot is needed because `results/` is committed;
the **post** column after all parts are re-run.

| # | Quantity | Source | Pre | Post |
|---|---|---|---|---|
| 1 | `n_transactions` | `simulation_report.json` ×4 profiles | | |
| 2 | `n_fraud_transactions` / `n_fraud_episodes` | same | | |
| 3 | realised fraud rate vs target (flag if > ±5%) | same, new `fraud_rate_vs_target_pct` | | |
| 4 | pre-open rows (must be 0) and affected customers | `transactions` × `customers` | | |
| 5 | cold-start gate counters (`normal_rows_masked`, `episode_day_redraws`, `episode_customer_fallbacks`) | `simulation_report.json` (new `cold_start_gate`) | n/a | |
| 6 | separability gate verdict + `min_pr_auc_gap`, max single-signal ROC/PR-AUC | gate output / `docs/eda_summary.md` | | |
| 7 | val PR-AUC: rules / logreg / xgb | `results/models_dev.json` | | |
| 8 | calibration: chosen calibrator, ECE, Brier, and the Part 07 "calibration did nothing" null result (does sigmoid still beat raw, does isotonic still not help) | `results/calibration_dev.*` | | |
| 9 | policy: capacity/day, realised ₹ cost, saving vs `f1_optimal`, recall/precision/FPR, action counts, `n_demoted`, `cost_optimal_threshold` | `results/policy_dev.*` | | |
| 10 | Part 08 headline rows (E10–E14): imbalance winner, ablation deltas, drift PSI/adversarial AUC | `results/exp_08*` | | |
| 11 | Part 09: S1−S0 paired CI, adoption decision | `results/exp_09*` | | |
| 12 | Part 10: top mean\|SHAP\| feature, additivity error, latency p50/p95, gallery row ids, fairness "distinguishable" count | `results/shap_*`, `results/fairness_*` | | |
| 13 | min `history_days` (must be ≥ 0) | features parquet | | |

Already-known deltas from this part (found before regeneration, from the code change itself):

- tiny: 4,716 → 4,626 transactions; the `structuring` scenario **reappears** (see bug below);
  gate counters `{normal_rows_masked: 92, episode_day_redraws: 1, episode_customer_fallbacks: 0}`;
  realised fraud rate 0.019023 vs target 0.02.

## Changes

- `data_gen/normal.py`: `open_day_index()` (single source of the gate) and an **emit-time** filter in
  `generate_normal` — pre-opening days are dropped while the vectorized draws and per-customer state
  are untouched, so open customers' rows are byte-identical to an ungated run.
- `data_gen/fraud.py` / `hard_negatives.py`: episode start is drawn after the customer's opening day
  (redraw counted), with a same-archetype fallback (counted) when the drawn customer has no room;
  every branch that picked a day is gated.
- `data_gen/generate.py`: gate counters passed through; `cold_start_gate` in the report;
  realised-vs-target fraud rate in the report; **assertion** that no transaction precedes its
  customer's `account_open_date` before anything is written.
- `data_quality/validation.py`: new `transactions_after_account_open` check.
- `features/build_features.py`: guard raising on any `history_days < 0`.
- `evaluation/experiments.py`: `dataset_version` column, version-scoped `drop_experiment_rows`,
  `backfill_dataset_version`, header self-heal so an added column cannot misalign old rows.
- `scripts/run_experiment_8E.py`, `scripts/run_unseen_experiment.py`: version-scoped row replacement.
- `config/config.yaml` + `config.py`: `dataset_version` label (not a simulation parameter).

## Bugs found while fixing (full entries in `docs/what_went_wrong.md`)

1. **Pre-open transactions** — the original defect (12,329 dev rows).
2. **`if bu < 4: break` in the structuring branch** (`fraud.py`) — regenerated tiny hit the
   structuring draw with a small remaining budget, and the old `break` ended fraud generation
   entirely, leaving the profile with **no structuring episodes at all**. Changed to `continue`
   (the outer loop already stops on `bud <= 0`). Exposed by the re-run, not by the gate itself.
3. **Vectorised masking shifted the normal stream** — masking the pre-opening days before the
   vectorized draws changed `nt` and therefore re-assigned random numbers to every subsequent row,
   silently changing all customers' data. Fixed by filtering at emit time instead, which is what the
   identity test asserts.

## Tests

- `tests/test_simulator.py`: `test_no_transactions_before_account_open`,
  `test_normal_rows_for_open_customers_unchanged` (byte-identity for already-open customers,
  normal stage only), `test_episode_falls_back_when_account_opens_after_window`.
- `tests/test_validation.py`: `test_transaction_before_account_open_fails`,
  `test_account_open_before_window_is_fine`.
- `tests/test_experiments.py`: version-scoped drop, backfill idempotence, legacy-header alignment.

`pytest -q -m "not db and not slow"` → **191 passed, 8 deselected**; `ruff check .` clean.

## Re-run order (recorded before execution)

`generate_data` ×4 → `validate_data` ×4 → `eda` (dev, tiny) → `separability_gate --profile dev` →
`load_db --profile tiny --reset` → `build_features` ×4 → `train --profile dev --model all` and
`--profile tiny` → `evaluate_val --profile dev` → `calibrate --profile dev` → `optimize_policy
--profile dev` → `run_experiment_8A..8D --profile dev` → `run_experiment_8E --profile drift_lab` →
`run_unseen_experiment --profile unseen_lab --holdout structuring` → `run_shap --profile dev` →
`fairness_audit --profile dev` → full `pytest -q` + `ruff check .`. Stop at the first failure and
report real output. If the separability gate fails, follow Part 03D with thresholds unchanged.

## Results

_Pending: filled in after the regeneration run (numbers copied from the artefacts above)._
