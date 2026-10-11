# Part 06 — Temporal Splits, Metrics & Three Core Models

## What was built

Chronological evaluation machinery plus three models (rules → logistic regression →
XGBoost), each fit on the training window and selected on the validation window. The test
window is **never** touched; all access is funnelled through `TestSetGuard`.

| Path | Purpose |
|---|---|
| `evaluation/splits.py` | `make_temporal_split` (time-axis cuts + embargo gap), `customer_holdout`, `rolling_origin_folds`, `max_train_label_available_at`, `save_split` |
| `evaluation/guard.py` | `TestSetGuard` — logs every access, raises unless `final=True` (Part 12 only) |
| `evaluation/metrics.py` | PR-AUC, ROC-AUC, recall@top-K, recall@precision, FPR, Brier, ECE, best-F1, lift |
| `evaluation/bootstrap.py` | `cluster_bootstrap` — resamples **customers**, not transactions |
| `evaluation/experiments.py` | `log_experiment` → `results/experiments.csv` (16-col schema) |
| `evaluation/dataset.py` | `load_feature_dataset` — rejoins `customer_id`/`timestamp` from raw |
| `evaluation/reporting.py` | `build_comparison`, `per_scenario_recall`, `group_metrics`, `holdout_comparison`, `to_markdown` |
| `evaluation/figures.py` | validation PR curves + recall@K figures |
| `models/rules.py` | `RuleEngine` + `write_rules_doc` |
| `models/logreg.py` | `tune_logreg`, `make_logreg_pipeline`, `logreg_coefficients` |
| `models/xgb.py` | `encode_features`, `random_search_xgb`, `gain_importances`, `ablation_pr_auc` |
| `models/io.py` | `build_metadata`, `save_model`, `load_model`, `load_metadata` |
| `scripts/train.py` | CLI: fit + select; writes split, doc, coefficients, search log, artifacts, experiment rows |
| `scripts/evaluate_val.py` | CLI: comparison table, per-scenario recall, holdout, figures |
| `tests/test_splits.py` | 10 tests (splits, embargo, holdout dedupe, guard) |
| `tests/test_metrics.py`, `test_bootstrap.py` | 13 tests |
| `tests/test_experiments.py` | 1 test |
| `tests/test_rules.py`, `test_logreg.py`, `test_xgb.py` | 12 tests |

Config gained `models:` (xgb search space + scale_pos_weight plan, logreg grids) and `rules:`
(6 named rules with percentile/domain thresholds). Makefile gained `train` / `evaluate-val`.

### Split & leakage discipline

- `make_temporal_split` cuts on the time axis at `splits.*_fraction`, inserting an embargo gap
  of `embargo_days` after each cut; requires the frame sorted by `(timestamp, transaction_id)`.
- Labels are joined **only** at evaluation: `is_fraud` to score, `fraud_scenario` for the
  per-scenario breakdown — never as features. `FORBIDDEN_COLUMNS` still guards feature columns.
- `save_split` writes `results/split_<profile>.json`; the test window is recorded but untouched.

## Commands run and results

```bash
python scripts/build_features.py --profile tiny           # 4,716 rows (tiny features existed for dev only)
python scripts/train.py --profile tiny --model all         # rules 0.125 < logreg 0.509 < xgb 0.767
python scripts/evaluate_val.py --profile tiny              # pipeline smoke-test
python scripts/train.py --profile dev --model all          # 288 s; xgb val PR-AUC 0.9667 (STOP guard trip)
python scripts/evaluate_val.py --profile dev               # 28 s; comparison + scenario + holdout + figures
make test-fast                                             # 116 passed, 8 deselected
make lint                                                  # All checks passed!
```

### Validation comparison (`dev`, train 174,834 / val 41,313 / test 39,380)

CI = cluster bootstrap over customers (n=500). Test window untouched.

| model | PR-AUC | 95% CI | ROC-AUC | best-F1 | recall@P=0.9 | recall@top-1% |
|---|---|---|---|---|---|---|
| rules | 0.3787 | 0.2418–0.4943 | 0.8091 | 0.5612 | 0.0322 | 0.4678 |
| logreg | 0.9021 | 0.8531–0.9379 | 0.9939 | 0.8469 | 0.7924 | 0.8889 |
| **xgboost** | **0.9667** | 0.9466–0.9810 | 0.9987 | 0.9212 | 0.9152 | 0.9561 |

Ordering is as expected: `rules < logreg < xgb` on every metric. Accuracy is never reported.

**XGBoost best hyperparameters** (random search, 30 trials, early stop on val aucpr):
`scale_pos_weight 50.98` (continuous), `max_depth 7`, `learning_rate 0.042`, `min_child_weight 3`,
`subsample 0.971`, `colsample_bytree 0.699`, `reg_alpha 0.602`, `reg_lambda 2.443`,
`best_iteration 592`. Logreg selected `C=10, class_weight=None`.

### Per-scenario recall at top-1% budget (`dev`, best model = xgboost)

| scenario | val frauds | captured | recall |
|---|---|---|---|
| account_takeover | 55 | 54 | 0.982 |
| amount_anomaly | 32 | 27 | 0.844 |
| burst | 189 | 180 | 0.952 |
| geo_anomaly | 23 | 23 | 1.000 |
| structuring | 43 | 43 | 1.000 |

### Unseen-customer holdout (XGB refit without 20% of customers)

| setting | n | frauds | PR-AUC | ROC-AUC | recall@top-1% |
|---|---|---|---|---|---|
| A_seen_customers | 33,082 | 297 | 0.9639 | 0.9986 | 0.929 |
| B_unseen_customers | 8,231 | 45 | 0.9349 | 0.9981 | 0.956 |

A modest generalisation gap (0.964 → 0.935). Figures: `reports/figures/val_pr_curves_dev.png`,
`reports/figures/val_recall_at_k_dev.png`.

## The >0.95 stop-guard (per the Part 06 directive)

XGBoost val PR-AUC **0.9667 > 0.95** tripped the guard; `train.py` exited 3 after writing
`results/xgb_importance_dev.json`. Diagnostics:

- **No dominant feature** (`dominant_feature: null`); top-5 gain shares are diffuse:
  `minutes_since_prev_tx` 0.126, `days_since_last_tx` 0.093, `tx_count_10m` 0.091,
  `tx_count_1h` 0.064, `is_night` 0.060.
- **`is_new_device` ablation:** 0.9667 → 0.9667 (drop −5.5e−06).

Interpretation: not a single leaker. The signal is spread across velocity/recency features and
the most plausible shortcut feature contributes nothing — consistent with the Part 03 gate
passing (best single-signal PR-AUC 0.183) and Part 05's invariance tests passing. This is
synthetic separability, not temporal leakage. Logged in `docs/what_went_wrong.md`; per the
directive the simulator was **not** changed, and the user accepted 0.9667 before Part 06 closed.

## Incidents (see `docs/what_went_wrong.md` for full text)

1. **High XGB val PR-AUC with no dominant feature** — logged, accepted as synthetic
   separability (above).
2. **`customer_holdout` did not deduplicate transaction-level ids** — the first
   `evaluate_val --profile dev` printed `A_seen n=164` vs `B_unseen n=41149` (seen PR-AUC 1.0
   on 4 frauds). Fixed with `np.unique` in `splits.py`; after the fix the holdout reads
   0.9639 seen vs 0.9349 unseen. Regression test
   `test_customer_holdout_dedupes_repeated_transaction_ids` added.

## Deviations from the prompt

- **Validation use:** the validation window is used for **both** XGB early stopping **and**
  hyperparameter/threshold selection. This mildly optimises the validation estimate (selection
  optimism); it is the prompt's intended single-val design, stated here for honesty. The test
  window is the unbiased check in Part 12.
- **`tiny` features** had to be built before the `tiny` smoke-test (only `dev` existed).
- **Rule thresholds** are training-legit percentiles (or domain minima when a percentile is
  undefined on thin data), config-driven in `rules:` — not hard-coded.
- **`logreg` `penalty="l2"`** omitted: deprecated in scikit-learn ≥1.8 (default is already L2).
- **Figures** saved as PNG only (matplotlib Agg); no interactive output.
- **`main` profile not run** — `dev` only, per the approved plan.

## Open issues

- Part 03 gate margins remain thin; the 0.9667 progress vs the gate's single-signal cap
  (0.183) is worth re-checking in Part 08's experiment suite (a gain-share < 0.60 invariant
  test is a candidate there).
- All models are **uncalibrated** (calibration is Part 07); `ece` is already reported for
  reference but no Platt/isotonic step exists yet.
- `bootstrap.n=500` on `dev` is fast; `main` may need more resamples for stable CIs.
