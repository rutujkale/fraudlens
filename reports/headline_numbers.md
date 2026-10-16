# FraudLens — headline numbers

Profile `dev` · test window · frozen evaluation (freeze commit `0953884369`, eval commit `c31c8c8db0`).

## 1. Model performance

- XGB 0.9619 (CI 0.9340-0.9798)
- Rules 0.2036 (CI 0.0873-0.3492)
- LogReg 0.8873 (CI 0.8383-0.9233)
- recall@1% budget: 0.9059; recall at capacity policy: 0.9669
- _assumptions:_ PR-AUC is primary (imbalanced data; accuracy is a footnote). CI = cluster bootstrap over customers (n=500, seed 42). Test prevalence 0.987%.

## 2. Business decision

- realized-cost reduction vs F1-optimal: 773,067 INR (83.23%), CI 361,480 to 1,281,632
- CI excludes 0: True
- demand vs capacity: 5.9/day vs 23/day
- n_demoted: 0
- _assumptions:_ Costs (INR): alpha=0.85 of amount lost, review_cost=75, friction_cost=150, step-up 5/40, block friction 450; analyst capacity 23/day (avg daily volume 2254.8). Realized cost on test with the frozen validation threshold; the reduction CI resamples customers.

## 3. Unseen fraud & drift

- anomaly before/after: None — Part 09 isolation forest was re-run post cold-start fix and NOT adopted (S1-S0 and S1s-S0 bagged CIs include 0); no anomaly feature is in the policy, so there is no before/after gain to report.
- XGB PR-AUC val->test: {'val': 0.9290862366465465, 'test': 0.9619214120306975}
- drift switch 2026-04-16; post-switch share of test frauds: 0.7888040712468194
- _assumptions:_ The drift switch 2026-04-16 sits inside the test window; 79% of test frauds are post-switch, so val->test degradation mixes drift with the window change.

## 4. Engineering

- features: 53
- automated tests: 233; leakage tests: 12
- API p95 latency: None (Filled in Part 13 (FastAPI deployment); placeholder until then.)

## Pre-registered outcomes (verbatim)

- **H1_model** — holds=True: XGB PR-AUC CI lower bound exceeds the Rules CI upper bound. LogReg ordering is reported either way.
- **H2_decision** — holds=True: The amount-aware capacity policy's realized-cost reduction versus F1-optimal has a CI excluding 0.
- **H3_unseen** — holds=True: The post-fix Part 09 "not adopted" decision stands, with no new Part 09 claim.
- **H4_engineering** — holds=True: Counts come from code, and the latency figure comes from Part 13.

## Resume bullet candidates (real numbers)

- Built an end-to-end fraud-risk system; frozen XGBoost test PR-AUC 0.962 (CI 0.934-0.980) vs Rules 0.204 at 0.99% prevalence [results/final_results.json:model_comparison.xgb]
- Cost-based capacity policy cut realized fraud cost 83.2% vs F1-optimal (CI 361,480-1,281,632 INR) under a 23/day analyst cap [final_results.json:decision.meta.reduction_ci]
- Reported concept drift as a finding: 79% of test frauds fall after the drift switch [final_results.json:drift.post_switch]
- 53-feature registry, 12 leakage tests, 233 automated tests [final_results.json:meta]
