# Part 08 — ML Understanding Experiments

> **Superseded by the Part 02b cold-start fix (2026-10-16).** The numbers below were produced
> on the pre-fix dataset with model `xgb_20261011_v1`; the data was regenerated and the dev model
> retrained as `xgb_dev_20261015_v1`. See [part_02b.md](part_02b.md) for the before/after table.

## What was built

Five controlled experiments (8A–8E) that probe *why* the Part 06/07 models behave
as they do, plus the first piece of monitoring code (PSI). All experiments use
the same temporal split and the validation window only; the test window is never
touched. `drift_lab` was generated and featurised up front so 8E could run on it.

| Path | Purpose |
|---|---|
| `src/fraudlens/evaluation/experiments_lib.py` | Shared helpers: `load_supervised`, `best_xgb_params`, `best_logreg_params`, `fit_xgb`, `predict_scores`, `mean_std`, `calibration_before_after` |
| `src/fraudlens/monitoring/drift.py` | `psi`, `psi_flag`, `psi_table` (quantile bins, own missing bucket, eps floor) |
| `tests/test_drift.py` | 6 tests (identical→0, shift→>0.25, monotonicity, missing-rate shift, eps-finite, table/flags) |
| `src/fraudlens/models/logreg.py` | `make_logreg_pipeline` gained `extra_numeric` / `extra_categorical` (backward-compatible) |
| `scripts/run_experiment_8A.py` | Imbalance handling sweep |
| `scripts/run_experiment_8B.py` | Feature-group ablation (drop-one / only-one / per-scenario) |
| `scripts/run_experiment_8C.py` | Capacity, data size, rounds, linear reference |
| `scripts/run_experiment_8D.py` | Importance agreement (gain / permutation / grouped / SHAP) + planted controls |
| `scripts/run_experiment_8E.py` | Drift backtest (4 retraining policies) + PSI + adversarial validation |
| `docs/experiments/*.md` | Per-experiment notes + README |

`docs/experiments/README.md` and one note per experiment were written, each with
Question / Setup / Results / Observations / Caveats.

## 8A — Imbalance handling (`dev`, 3 seeds)

| config | PR-AUC | recall@1% | Brier before→after | ECE before→after |
|---|---|---|---|---|
| `spw_full` | **0.9663 ± 0.0008** | **0.9542** | 0.00099→0.00118 | 0.00029→0.00091 |
| `spw_sqrt` | 0.9653 ± 0.0005 | 0.9493 | 0.00098→0.00119 | 0.00017→0.00091 |
| `none` | 0.9614 ± 0.0014 | 0.9405 | 0.00108→0.00129 | 0.00061→0.00087 |
| `undersample_1_50` | 0.9567 ± 0.0011 | 0.9405 | 0.00119→0.00138 | 0.00036→0.00091 |
| `undersample_1_10` | 0.9413 ± 0.0004 | 0.9230 | 0.00226→**0.00162** | 0.00459→**0.00107** |
| `logreg_balanced` | 0.8680 | 0.8392 | 0.01610→**0.00226** | 0.03346→**0.00131** |

`scale_pos_weight = neg/pos` is the best lever; undersampling hurts ranking and
distorts the base rate; sigmoid calibration only helps the **distorted** models
(`undersample_1_10`, `logreg_balanced`) — matching the Part 07 null result.

## 8B — Feature-group ablation (`dev`, 3 seeds)

Full PR-AUC 0.9657; baseline scenario recall@top-1%: ATO 0.970, amount_anomaly
0.823, burst 0.956, geo_anomaly 0.986, structuring 1.000.

Drop-one PR-AUC: recency_frequency 0.9234 (−0.042), device 0.9325 (−0.033),
transaction 0.9438 (−0.022), then ≤ −0.010 for the rest. Per-scenario the drops
line up with the simulator design: device→ATO (−0.176), geographic→geo_anomaly
(−0.217), recency_frequency→burst (−0.072), transaction→amount_anomaly (−0.167).

**Falsified hypothesis:** dropping `amount_sums` leaves structuring recall at
**1.000** (cost −0.001 PR-AUC), and `only_amount_sums` catches just 0.333 of it.
Structuring is caught redundantly by amount-vs-baseline (0.884 alone), transaction
type (0.519), trend (0.527) and others. The signal is not concentrated in the
threshold-count feature. The simulation-design link is written out in
`docs/experiments/8B_feature_ablation.md`.

## 8C — Capacity / data size (`dev`, 3 seeds)

| depth | val PR-AUC | train PR-AUC | gap |
|---:|---:|---:|---:|
| 2 | 0.9204 ± 0.0012 | 0.9300 | 0.0096 |
| 4 | 0.9539 ± 0.0009 | 0.9980 | 0.0441 |
| 6 | 0.9634 ± 0.0005 | 1.0000 | 0.0366 |
| **8** | **0.9657 ± 0.0006** | 1.0000 | 0.0343 |
| 10 | 0.9655 ± 0.0010 | 1.0000 | 0.0345 |

Data size at depth 8: 10% rows 0.6783 ± 0.1436, 25% 0.8893, 50% 0.9308, 100%
0.9657. Rounds: val plateaus ~0.9653 by round ~490. LogReg base 0.9021 [+interactions
0.9044, +binned 0.9022] vs XGB 0.9657 → a stable ~0.06 gap linear models cannot
close. **Saturation looks data-limited, not capacity-limited — stated as a trend,
not a measured law:** the 10%-row point is ±0.14 and saturation rests on three
seeds of one synthetic split (50→100% gives +0.0349, more than any depth change);
re-test on real data in Part 11.

## 8D — Importance agreement (`dev`)

Sanity model val PR-AUC 0.9651. Method agreement (Spearman, 55 features): gain↔perm
0.59, gain↔SHAP 0.57, perm↔SHAP 0.65, perm-group↔perm 0.41. Top families agree
(device, recency/velocity, merchant novelty). **Planted controls:** rank last under
gain (49, 51) and permutation (51, 52), but **17th/26th of 55 under mean|SHAP|**.
Split counts confirm the booster really fits them (736 / 1065 internal nodes) — a
real overfit-vs-usefulness discrepancy, not a bug. Grouped permutation reveals
cluster effects (`days_since_last_tx` 0.004 alone → 0.176 with its recency cluster).

## 8E — Drift and retraining policy (`drift_lab`)

Monthly post-shift PR-AUC (mean): frozen **0.9440**, expanding **0.9824**,
sliding_60d **0.9830**, recency_weighted **0.9787**. Covariate drift (pre vs post):
adversarial AUC **0.9998 all features** but **0.7072 behavioural** — the gap is the
four monotonic tenure features (`history_days` alone gives 0.974). Behavioural PSI
flags only `hour_deviation` (0.533). **Conclusion: covariate "drift" was mostly the
clock; real concept drift is mild (frozen stays at 0.9440) and a 60-day sliding
window matches full-history retraining.** Caveat: the frozen model staying
competitive and the ~0.04 retraining gain are dataset-specific — the 2026-08-29
window is easiest for *every* policy (0.9610/0.9876/0.9893/0.9884), so read the
gap, not the level; re-test on real data in Part 11.

## Commands run (actual summaries)

```
python scripts/run_experiment_8A.py --profile dev    # 6 E10 rows; summary in results/exp_8A_imbalance_summary_dev.csv
python scripts/run_experiment_8B.py --profile dev    # 21 E11 rows; ablation + scenario deltas
python scripts/run_experiment_8C.py --profile dev    # best depth by mean = 8; 9 E12 rows
python scripts/run_experiment_8D.py --profile dev    # 1 E13 row; sanity model PR-AUC 0.9651
python scripts/run_experiment_8E.py --profile drift_lab  # 5 E14 rows; ~2 min; 1,156,589 rows
pytest -q                                            # 156 passed, 7 skipped
ruff check src/ scripts/ tests/                      # All checks passed
```

## Deviations from the prompt

- **8C depth selection** — the sweep originally chose "best depth" from a single
  (depth, seed) pair; fixed to choose by mean val PR-AUC (deviations logged in
  `what_went_wrong.md`). Because depth 8/10 are tied, the effect was cosmetic.
- **8E uses the whole `drift_lab` timeline** (point-in-time training, labels
  matured per step). `drift_lab` is a lab profile, so this does not touch the
  Part-12 test-window protocol.
- **8D uses default path-dependent TreeSHAP.** Interventional SHAP (background
  dataset) is deferred to Part 10.
- **drift_lab generated early** (before 8E) per the working protocol, so the long
  simulation job was not on the critical path.

## Open issues

- 8D's mean|SHAP| behaviour is a **constraint for Part 10**: raw global mean|SHAP|
  must not be used as an importance ranking; use signed per-transaction
  attribution on a calibrated model. Logged in `what_went_wrong.md`.
- Categorical drift (PSI on `transaction_type`/`channel` mix) is only captured
  indirectly via adversarial AUC; ordered-bin PSI for categoricals is a possible
  Part 13 improvement.
- All findings except 8E are on `dev`; magnitudes are synthetic-specific (Part 11
  IEEE-CIS will re-test capacity, calibration and drift on real data).
