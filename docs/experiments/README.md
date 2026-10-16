# Part 08 — ML understanding experiments

Five controlled experiments that probe *why* the models behave as they do. Each
one is a script (`scripts/run_experiment_8X.py`) writing a CSV to `results/`, a
figure to `reports/figures/`, and rows to `results/experiments.csv`.

| ID | Experiment | Profile | Script | Result files |
|----|------------|---------|--------|--------------|
| 8A | Imbalance handling | `dev` | `run_experiment_8A.py` | `exp_8A_imbalance*.csv` |
| 8B | Feature-group ablation | `dev` | `run_experiment_8B.py` | `exp_8B_ablation_dev.csv`, `exp_8B_scenario_delta_dev.csv` |
| 8C | Model capacity / data-size | `dev` | `run_experiment_8C.py` | `exp_8C_{capacity,trainsize,rounds,logreg}_dev.csv` |
| 8D | Feature-importance agreement | `dev` | `run_experiment_8D.py` | `exp_8D_{importance,method_agreement}_dev.csv` |
| 8E | Drift and retraining policy | `drift_lab` | `run_experiment_8E.py` | `exp_8E_{backtest,psi,adversarial*}_drift_lab.*` |

Common rules (from the Part 08 prompt): the same temporal split, at least three
seeds whenever a model is retrained (reported as mean ± std), cluster-bootstrap
CIs for headline metrics, and **validation only** — no test-window access.

Per-experiment notes:

- [8A — Imbalance handling](8A_imbalance.md)
- [8B — Feature-group ablation](8B_feature_ablation.md)
- [8C — Model capacity and data size](8C_model_capacity.md)
- [8D — Feature-importance agreement](8D_feature_importance.md)
- [8E — Drift and retraining policy](8E_drift.md)

## Headline findings

1. **`scale_pos_weight = neg/pos` is the best imbalance lever** (PR-AUC
   0.9271 vs 0.9258 unweighted) and it barely distorts probabilities on this
   near-separable task; random undersampling *hurts* ranking and distorts the
   base rate, and sigmoid calibration only repairs the distorted models. (8A)
2. **No single feature group is load-bearing overall** — dropping any one group
   costs ≤ 0.04 PR-AUC — but per-scenario effects are sharp and match how each
   scenario was *simulated* (device↔account-takeover, geographic↔geo-anomaly,
   transaction↔amount-anomaly). Structuring is the most redundant scenario. (8B)
3. **Saturation looks data-limited, not capacity-limited** — a *trend*, not a
   measured law: val PR-AUC saturates at depth ≈ 6–8, train PR-AUC hits 1.0 by
   depth 6, and doubling the training set (+0.028) buys more than any depth
   change, but the 10%-row point is ±0.07 and the claim rests on three seeds of
   one synthetic split. A linear model plateaus ~0.06 below XGBoost even with
   interactions/binning. (8C)
4. **The three importance methods agree on the top families (ρ 0.52–0.58) but
   disagree on the tail**, and gain/permutation correctly flag the planted
   control columns as unimportant while path-dependent mean|SHAP| does not. (8D)
5. **Behavioural drift is mild (adversarial AUC 0.6995)**; the naive
   "adversarial AUC 0.9821" on all features is almost entirely the monotonic
   tenure clock, not fraud drift. Retraining recovers only ~0.04 PR-AUC over a
   frozen model — a property of this simulator's scenario-mix drift, to re-test
   on real data (Part 11). (8E)

> Numbers above are post-coldstart-fix (Part 02b, dataset regenerated 2026-10-16); the
> per-experiment notes below still show the pre-fix figures under a superseded banner.
