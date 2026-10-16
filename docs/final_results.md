# FraudLens — final locked evaluation results

Profile `dev` · frozen evaluation · freeze commit `0953884369` · eval commit `c31c8c8db0` · generated 2026-10-16T17:01:21.

Test window (held out, accessed once): 39,823 rows · 393 frauds · prevalence 0.987%.  All 14 frozen hashes matched `results/freeze.json` (True); config reconstruction proof OK.

## 1. Model comparison (test, frozen thresholds)

| Model | PR-AUC (CI) | ROC-AUC | F1@thr | Prec | Rec | FPR | Lift@1% | recall@1% | recall@90%P | Accuracy* |
|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.2036 (0.0873-0.3492) | 0.7390 | 0.3646 | 0.6600 | 0.2519 | 0.00129 | 29.21 | 0.2926 | n/a | 0.9913 |
| logreg | 0.8873 (0.8383-0.9233) | 0.9938 | 0.8352 | 0.9453 | 0.7481 | 0.00043 | 83.55 | 0.8372 | 0.7786 | 0.9971 |
| xgb | 0.9619 (0.9340-0.9798) | 0.9991 | 0.9026 | 0.9346 | 0.8728 | 0.00061 | 90.41 | 0.9059 | 0.9033 | 0.9981 |

*Accuracy is a footnote only (imbalanced data; PR-AUC is primary). CI = cluster bootstrap over customers (n=500, seed 42).

## 2. Calibration (frozen calibrator)

| Model | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
|---|---|---|---|---|
| xgb | 0.0014 | 0.0013 | 0.000198 | 0.000290 |
| logreg | 0.0024 | 0.0025 | 0.000130 | 0.000173 |

Test base rate 0.00987.

## 3. Decision policy (test, frozen validation thresholds)

| Policy | Flagged/day | Recall | Precision | FPR | Fraud loss (INR) | Review (INR) | Friction (INR) | Total cost (INR) | % vs F1 | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| approve_all | 0.0 | 0.0000 | n/a | 0.00000 | 4,839,298 | 0 | 0 | 4,839,298 | -420.98 | True |
| review_all | 2212.4 | 1.0000 | 0.0099 | 1.00000 | 483,930 | 2,986,725 | 5,914,500 | 9,385,155 | -910.38 | False |
| f1_optimal | 20.4 | 0.8728 | 0.9346 | 0.00061 | 897,751 | 27,525 | 3,600 | 928,876 | 0.00 | True |
| cost_optimal_global | 27.6 | 0.9491 | 0.7505 | 0.00314 | 704,488 | 37,275 | 18,600 | 760,363 | 18.14 | False |
| cost_optimal_global_capacity | 23.5 | 0.9160 | 0.8511 | 0.00160 | 765,170 | 31,725 | 9,450 | 806,345 | 13.19 | False |
| amount_aware | 5.9 | 0.9669 | 0.3287 | 0.01968 | 84,848 | 8,025 | 62,935 | 155,808 | 83.23 | True |
| amount_aware_capacity | 5.9 | 0.9669 | 0.3287 | 0.01968 | 84,848 | 8,025 | 62,935 | 155,808 | 83.23 | True |

F1-optimal threshold 0.400721; cost-optimal global 0.075663; under capacity 0.188227. Capacity 23/day (avg daily volume 2254.8). n_demoted 0. Amount-aware capacity reduction vs F1: 83.23% (CI 361,480 to 1,281,632 INR, excludes 0: True).

## 4. Cost sensitivity (val threshold re-derived per scenario, evaluated on test)

| Scenario | val thr | Test cost (INR) | Recall | Flagged/day | % saving vs F1 |
|---|---|---|---|---|---|
| low_fp_cost | 0.015351 | 630,795 | 0.9796 | 46.4 | 32.09 |
| balanced | 0.075663 | 760,363 | 0.9491 | 27.6 | 18.14 |
| high_fp_cost | 0.075663 | 813,728 | 0.9491 | 27.6 | 12.40 |
| high_fraud_loss | 0.075663 | 901,875 | 0.9491 | 27.6 | 2.91 |

## 5. Drift: validation (pre-switch) vs test (post-switch)

| Model | val PR-AUC | test PR-AUC |
|---|---|---|
| rules | 0.2923 | 0.2036 |
| logreg | 0.8638 | 0.8873 |
| xgb | 0.9291 | 0.9619 |

Score PSI val→test 0.0015. Drift switch `2026-04-16` is inside the test window: 4,574 rows / 83 frauds pre, 35,249 rows / 310 frauds post — **79% of test frauds are post-switch**.

| Feature (top PSI) | PSI | Flag |
|---|---|---|
| history_days | 4.214 | shift |
| device_age_days | 2.574 | shift |
| n_prior_tx | 1.455 | shift |
| tx_from_device_prior | 1.040 | shift |
| hour_deviation | 0.450 | shift |
| active_days_30d | 0.233 | monitor |
| activity_change_7d_vs_prev_7d | 0.129 | monitor |
| amount_change_7d_vs_prev_7d | 0.128 | monitor |

## 6. Customer holdout (Setting B): unseen vs seen customers on test

| Group | Rows | Frauds | PR-AUC | ROC-AUC | recall@1% |
|---|---|---|---|---|---|
| seen | 38,268 | 356 | 0.9654 | 0.9992 | 0.9242 |
| unseen | 1,555 | 37 | 0.9390 | 0.9969 | 0.4324 |

_unseen customers = test rows whose `customer_id` never appears in train+validation_ (1,555 rows).

## 7. Per-scenario recall at the frozen policy

| Scenario | n fraud | recall@frozen | recall@top1% |
|---|---|---|---|
| structuring | 156 | 1.0000 | 0.9487 |
| burst | 127 | 0.9921 | 0.9685 |
| account_takeover | 63 | 0.9048 | 0.7619 |
| amount_anomaly | 25 | 0.7600 | 0.7200 |
| geo_anomaly | 22 | 1.0000 | 0.8636 |
| none | 0 | n/a | n/a |

## 8. Fairness slice (test, frozen policy); cells with n_pos<30 suppressed

| Attribute | groups | suppressed groups | reported groups |
|---|---|---|---|
| age_band | 5 | 2 | 3 |
| income_band | 4 | 0 | 4 |
| archetype | 5 | 2 | 3 |
| home_region_group | 6 | 1 | 5 |

## 9. Sim-to-real finalization (no cross-scale differencing)

Synthetic test XGB PR-AUC 0.9619 at prevalence 0.00987; features 53 vs 38 shared with IEEE-CIS. Synthetic PR-AUC and incidence are reported beside each other; the IEEE-CIS external numbers are not subtracted from these (different scale, prevalence and feature count: 53 vs 38).

## Figures

- `reports/figures/final_pr_curves.png`
- `reports/figures/final_roc.png`
- `reports/figures/final_calibration.png`
- `reports/figures/final_recall_at_k.png`
- `reports/figures/final_cost_bars.png`
- `reports/figures/final_sensitivity.png`

## Pre-registered outcomes (verbatim)

- **H1_model** — holds=True: XGB PR-AUC CI lower bound exceeds the Rules CI upper bound. LogReg ordering is reported either way.
- **H2_decision** — holds=True: The amount-aware capacity policy's realized-cost reduction versus F1-optimal has a CI excluding 0.
- **H3_unseen** — holds=True: The post-fix Part 09 "not adopted" decision stands, with no new Part 09 claim.
- **H4_engineering** — holds=True: Counts come from code, and the latency figure comes from Part 13.

## Interpretation

**Models.** XGBoost leads on the test window (PR-AUC 0.9619, CI 0.9340-0.9798) ahead of LogReg (0.8873) and far ahead of the Rules baseline (0.2036); H1 holds (XGB lower bound clears the Rules upper bound). Absolute brier/ECE are small because prevalence is 0.99%, so calibration is best read on the reliability diagram rather than on raw magnitude; the frozen calibrator is applied and reported both ways.

**Decision.** Analysts are capped at 23/day but the amount-aware policy flags only 5.9/day, so capacity never binds (n_demoted 0) and the policy is identical with or without the cap. It cuts realized cost 83.23% versus the F1-optimal threshold (CI 361,480-1,281,632 INR, excluding 0), driven by catching high-value fraud while accepting a higher flag volume than F1 — H2 holds. Sensitivity shows the saving is positive under all four cost scenarios, largest when false-positive friction is cheap.

**Drift & unseen fraud.** The 2026-04-16 drift switch sits inside the test window; 79% of test frauds (310/393) are post-switch and the mixture shifts toward structuring. Despite that, XGB PR-AUC does not degrade from validation to test (0.9291 → 0.9619); the ranking survives the shift even though the top PSI features are the tenure/behavioural counters. On unseen customers (no training history) recall@1% drops to 0.4324 from 0.9242, the honest cost of thin history. The Part 09 isolation forest remains not adopted, so there is no anomaly before/after gain to claim.
