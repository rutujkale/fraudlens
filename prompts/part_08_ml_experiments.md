# Part 08 — ML-Understanding Experiments

## Goal
Run five focused experiments that show you understand *why* the models behave as they do. Each answers a specific question, uses train/validation data only, produces a CSV + figure + short findings note, and appends to the experiment log. The aim is insight, not a better score.

## Prerequisites
Parts 06–07 complete. Datasets: `main` (or `dev`) for experiments 8A–8D; `drift_lab` for 8E (generate it and build features first: `generate_data.py --profile drift_lab`, `build_features.py --profile drift_lab`).

## Common rules
- Same temporal split, same random seeds (run ≥3 seeds where a model is retrained; report mean ± std), same evaluation protocol as Part 06.
- Selection and reporting on validation only. Cluster-bootstrap CIs on key numbers.
- Each experiment has a script `scripts/run_experiment_8X.py`, writes `results/exp_8X_*.csv`, `reports/figures/exp_8X_*.png`, and `docs/experiments/8X_<name>.md` containing: Question, Setup, Results table (generated), Observations (only statements supported by the numbers), Caveats. Do not speculate presented as fact; label hypotheses as hypotheses.

## 8A — Imbalance handling (E10)
**Question:** What does each imbalance strategy do to ranking quality and to probability calibration?
Compare, with otherwise identical XGBoost settings: (1) no handling, (2) `scale_pos_weight = neg/pos`, (3) `scale_pos_weight = sqrt(neg/pos)`, (4) random undersampling of negatives (train only; ratios 1:10, 1:50), and for LogReg (5) `class_weight="balanced"`. Metrics: PR-AUC, recall@1%, Brier, ECE **before calibration**, and Brier/ECE **after** sigmoid calibration. Expected insight: ranking metrics change little, raw probabilities become badly miscalibrated under reweighting/undersampling, calibration repairs them. Add a note explaining why SMOTE-style oversampling is excluded (interpolated rows are behaviorally impossible, distort probabilities, and leak across time-ordered data).

## 8B — Feature-group ablation × fraud scenario (E11)
**Question:** Which signals catch which fraud scenario?
Groups: transaction, recency+frequency, amount-sums, amount-deviation, merchant, device, recipient, geographic, time-of-day, trend. Two modes: **drop-one-group** (retrain without the group) and **only-one-group**. Report overall PR-AUC and **recall by fraud scenario at the top-1% alert budget** (scenario labels used for evaluation only). Output a heatmap (scenario × group, change in recall vs full model). Observations should connect results to how each scenario was simulated (e.g. structuring depends on amount-sum/near-threshold features) and note where the model relies on a single group (a fragility signal).

## 8C — Bias–variance and model capacity (E12)
**Question:** How does complexity trade off, and is the gap between LogReg and XGBoost about nonlinearity/interactions?
- XGBoost: train vs validation PR-AUC across `max_depth ∈ {2,3,4,6,8,10}` and across boosting rounds (learning curves with early-stopping point marked); report the overfitting gap.
- Learning curve over training-set size (10%, 25%, 50%, 100% of the earliest data).
- LogReg variants: base; + manual interaction terms (e.g. `is_new_device × amount_ratio_to_30d_mean`, `tx_count_1h × amount_ratio`, `distance_from_home × is_night`); + binned numerics (splines/KBins). Does adding interactions close the gap to XGBoost? Report PR-AUC with CIs.
- Optional: a depth-limited single decision tree as a "what a human rule would reach" reference.

## 8D — Importance triangulation (E13)
**Question:** Do different importance methods agree, and why or why not?
Compute for the final XGBoost on validation: gain importance, permutation importance (PR-AUC drop, 5 repeats, **grouped by correlated feature clusters** as well as per feature), and mean |SHAP| (use a 10–20k stratified-by-class subsample). Report rank correlations (Spearman) among methods and a figure of the top-15 across methods. Observations: where they disagree and which correlated features share credit. (Full SHAP work comes in Part 10; here SHAP is only a comparison point.) Also run a **sanity check**: add a random-noise feature and a shuffled copy of the strongest feature; verify they rank near the bottom.

## 8E — Concept drift and retraining strategies (E14) — on `drift_lab`
**Question:** How much does performance degrade when the fraud strategy changes, and which retraining policy recovers it?
Dataset `drift_lab` has an abrupt shift at 50% of the timeline. Backtest in monthly steps after the shift (use the embargo/label-delay rules; at each step only labels matured by that date are available for training):
1. **Frozen:** train once on the pre-drift period, never retrain.
2. **Expanding retrain:** retrain every month on all data to date.
3. **Sliding window:** retrain every month on the last 60 days only.
4. **Recency-weighted:** expanding window with exponentially decayed sample weights (half-life 30 days).
Report monthly PR-AUC and recall@1% per strategy, the degradation after the shift vs before, how many months to recover, and the drift's feature-level footprint: PSI per feature (pre vs post) and an **adversarial validation** AUC (classifier distinguishing pre vs post rows; high AUC ⇒ strong covariate shift) — show the top features driving it. Implement PSI in `monitoring/drift.py` (reused in Part 15): quantile bins from the reference sample, epsilon smoothing, thresholds 0.1 / 0.25 as conventional rules of thumb. Observation: separate **covariate drift** (features move) from **concept drift** (the feature→fraud relationship changes); report evidence for each.

## Wrap-up
- `docs/experiments/README.md`: one paragraph per experiment (question → finding → caveat) generated from the result files, plus a "what would I do differently" list.
- Append a row per experiment variant to `results/experiments.csv` (E10…E14).
- If any experiment surprised you or exposed a bug, add it to `docs/what_went_wrong.md`.

## Acceptance criteria
```bash
python scripts/run_experiment_8A.py --profile main
python scripts/run_experiment_8B.py --profile main
python scripts/run_experiment_8C.py --profile main
python scripts/run_experiment_8D.py --profile main
python scripts/run_experiment_8E.py --profile drift_lab
pytest tests/test_drift.py -q          # PSI correctness: identical dist ≈ 0, shifted dist > threshold
```
Progress file: one-line finding per experiment with the key numbers, runtime per script, and any deviations.

## Do not
Touch the test window. Cherry-pick seeds. Present a hypothesis as a finding. Re-tune hyperparameters inside the ablations beyond what the experiment specifies (use the Part 06 best config unless stated).
