# Part 06 — Temporal Splits, Metrics, and the Three Models (Rules, Logistic Regression, XGBoost)

## Goal
Build the evaluation foundation (temporal splits with embargo, metrics, cluster bootstrap, experiment log, test-set guard) and train the three core models on the **training window**, evaluating on the **validation window only**. The test window is not touched in this part.

## Prerequisites
Parts 01–05 complete; `data/features/dev/features_v1.parquet` (and ideally `main`) exists and leakage tests pass.

## Tasks

### 6A — Temporal splits (`evaluation/splits.py`)
- `make_temporal_split(tx, cfg) -> Split` with boundaries from `splits.train_end_fraction`, `val_end_fraction` and `embargo_days` (= max label delay):
  `train = [t0, t_train_end)`, embargo, `val = [t_train_end + E, t_val_end)`, embargo, `test = [t_val_end + E, t_end]`. Rows inside the embargo gaps are excluded from every set (their labels may not have matured). Return index arrays plus a dict of boundary timestamps; save to `results/split_<profile>.json`.
- `customer_holdout(customers, fraction=0.2, seed)`: choose customers excluded from training. Evaluate on the validation window for both held-out customers (Setting B, unseen customers) and seen customers (Setting A). Report separately.
- `rolling_origin_folds(...)`: optional utility returning 3 chronological folds inside the training window (with embargo) for use in Part 08.
- `evaluation/guard.py`: `TestSetGuard` — any function that touches the test window must go through it. It logs each access to `results/test_access_log.csv` (timestamp, caller, model_version, git commit) and raises unless the caller passes `final=True` (only Part 12 does). Add a test that attempting test access without `final=True` raises.
- Tests: windows disjoint; ordering correct; embargo ≥ E days; all train labels mature (`label_available_at ≤ t_train_end + E`) for train rows; no customer-holdout customer in train.

### 6B — Metrics (`evaluation/metrics.py`, `evaluation/bootstrap.py`)
Functions: `pr_auc` (average precision), `roc_auc`, `precision_recall_f1_at_threshold`, `confusion_at_threshold`, `fpr_at_threshold`, `recall_at_top_k(y, score, k_frac)`, `precision_at_top_k`, `recall_at_precision(y, score, target)` (max recall with precision ≥ target; NaN if unattainable), `brier`, `ece` (equal-frequency bins; document the binning), `best_f1_threshold`, `lift_over_base_rate`. A `evaluate_scores(y, score, ks, precision_targets) -> dict` convenience.
- `cluster_bootstrap(y, score, groups, metric_fn, n, seed) -> (point, lo, hi)` resampling `customer_id`s with replacement (vectorized: precompute per-customer index lists).
- Tests with tiny hand-computed arrays: perfect ranking, random ranking (PR-AUC ≈ prevalence), top-K counts, recall@precision edge cases, ties in scores, bootstrap CI contains the point estimate and shrinks with more data.

### 6C — Experiment log (`evaluation/experiments.py`)
`log_experiment(row: dict)` appends to `results/experiments.csv` (schema from CLAUDE.md), adds `git_commit`, `timestamp`, `config_hash`. Experiment IDs: E01 Rules, E02 LogReg, E03 XGBoost, … (later parts continue the numbering).

### 6D — Rules baseline (`models/rules.py`)
A transparent human-designed risk policy, **defined before looking at validation labels**. Each rule has a name, condition and weight, e.g.: `tx_count_10m ≥ 5`; `amount_ratio_to_30d_mean ≥ 5`; `is_new_device and amount_ratio_to_30d_mean ≥ 2`; `travel_speed_kmh ≥ 800`; `is_night and new_category_flag`; `count_near_threshold_24h ≥ 3`. Score = weighted sum of fired rules normalized to [0,1]; rank by score (ties broken by amount). Provide `explain(row)` listing the rules that fired. Thresholds come from general domain intuition and percentiles of **training legit** feature distributions, not from fraud labels. Record the rule definitions in `docs/rules_baseline.md`. The rules engine must not copy the simulator's internal logic.

### 6E — Logistic Regression (`models/logreg.py`)
sklearn `Pipeline`: median imputation with missing-indicator columns → `log1p` on heavy-tailed non-negative numerics (amounts, counts, sums, ratios) → `StandardScaler` → one-hot for low-cardinality categoricals. L2 penalty; tune `C` over a log grid and `class_weight ∈ {None, "balanced"}` on **validation PR-AUC**. Save the fitted pipeline. Also export standardized coefficients (top ± features) to `results/logreg_coefficients.csv`.

### 6F — XGBoost (`models/xgb.py`)
- `tree_method="hist"`, `objective="binary:logistic"`, `eval_metric="aucpr"`, native NaN handling, categoricals via one-hot or label codes consistent across train/val.
- Early stopping on validation `aucpr`; random search of ~30 configs over `max_depth [3–8]`, `learning_rate [0.02–0.2]`, `min_child_weight [1–20]`, `subsample [0.6–1]`, `colsample_bytree [0.5–1]`, `reg_alpha`, `reg_lambda`, `scale_pos_weight ∈ {1, sqrt(neg/pos), neg/pos}`; seeded; record every trial in `results/xgb_search.csv`. Select by validation PR-AUC. (State in the progress file that validation is used for both early stopping and selection and what that implies.)
- Save with `models/io.py`: model file + `metadata.json` (model_version `xgb_<date>_v1`, feature_version, training window, hyperparameters, seed, git commit, config hash, validation metrics). Raw (uncalibrated) validation scores saved to `artifacts/<model_version>/val_scores.parquet` (transaction_id, customer_id, raw score).

### 6G — Training & evaluation scripts
`scripts/train.py --profile dev --model {rules,logreg,xgb,all}` and `scripts/evaluate_val.py`. Output a **validation comparison table** `results/val_comparison_<profile>.csv` and markdown with PR-AUC (with cluster-bootstrap CI), ROC-AUC, recall@top-K for all K in config, recall at 90/80/70% precision, F1 at the best-F1 threshold. Add per-scenario recall at the top-1% alert budget (using `labels.fraud_scenario`; labels are used **only for evaluation**). Also report the customer-holdout comparison (seen vs unseen customers) for XGBoost. Figures: PR curves (all models), recall@K curves → `reports/figures/`.

## Acceptance criteria
```bash
pytest tests/ -q -m "not db and not slow"
python scripts/train.py --profile dev --model all
python scripts/evaluate_val.py --profile dev
cat results/val_comparison_dev.csv
```
Expectations (sanity, not targets): XGBoost > LogReg > Rules on validation PR-AUC; if XGBoost is *suspiciously* near-perfect (PR-AUC > ~0.95), stop: re-check the leakage tests and feature importances and treat it as a possible simulator shortcut (log it in `docs/what_went_wrong.md`) rather than celebrating. If Rules ≥ XGBoost, investigate before continuing.

Progress file: the comparison table, bootstrap CIs, per-scenario recall, holdout results, observations (facts derived from the numbers only).

## Do not
Touch the test window. Use accuracy as a headline. Use a random split. Apply SMOTE/oversampling. Tune the rules on validation labels.
