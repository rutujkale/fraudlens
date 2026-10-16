# FraudLens — headline numbers

Profile `dev` · test window · frozen evaluation (freeze commit `0953884369`, eval commit `c31c8c8db0`).

## 1. Model performance

- XGB 0.9619 (CI 0.9340-0.9798)
- Rules 0.2036 (CI 0.0873-0.3492)
- LogReg 0.8873 (CI 0.8383-0.9233)
- recall@1% budget: 0.9059; recall at capacity policy: 0.9669
- CI width: with only 393 test frauds the XGB PR-AUC CI (0.934-0.980) is wide; read the point estimate with that in mind.
- _assumptions:_ PR-AUC is primary (imbalanced data; accuracy is a footnote). CI = cluster bootstrap over customers (n=500, seed 42). Test prevalence 0.987%.

## 2. Business decision

- realized-cost reduction vs F1-optimal: 773,067 INR (83.23%), CI 361,480 to 1,281,632
- CI excludes 0: True
- demand vs capacity: 5.9/day vs 23/day
- n_demoted: 0
- capacity: Capacity did not bind on test: the policy flags 5.9/day against a 23/day cap with 0 demotions, so `enforce_capacity` was never exercised and the capacity-aware result is NOT evidence the cap works — it is the unconstrained policy.
- _assumptions:_ Costs (INR): alpha=0.85 of amount lost, review_cost=75, friction_cost=150, step-up 5/40, block friction 450; analyst capacity 23/day (avg daily volume 2254.8). Realized cost on test with the frozen validation threshold; the reduction CI resamples customers. The 83.23% is a point estimate: the CI is reported in absolute INR, and no separate CI for the ratio is persisted because its denominator (the resampled F1 cost) is itself random.

## 3. Unseen fraud & drift

- anomaly before/after: None — the Part 09 isolation forest was re-run post cold-start fix and NOT adopted — the S1-S0 and S1s-S0 bagged CIs are 0.0000 (they include 0) — so there is no anomaly before/after gain to report; this is a null result, not a failure.
- drift: **no degradation observed** (XGB PR-AUC 0.9291 val -> 0.9619 test, higher on test)
- drift switch 2026-04-16; post-switch share of test frauds: 79%
- Hypotheses for why test beats val (not established): the structuring share of frauds rose from 18% pre-switch to 45% post-switch (structuring is an easier, high-signal scenario), and the validation window may be selection-optimistic.
- _assumptions:_ The drift switch 2026-04-16 sits inside the test window; 79% of test frauds are post-switch, so drift and the window change are conflated; the observed val->test movement is upward (no degradation observed).

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
- Amount-aware cost policy cut realized fraud cost 83.2% vs F1-optimal (CI 361,480-1,281,632 INR); capacity 23/day was non-binding on this window [final_results.json:decision.meta.reduction_ci]
- Reported concept drift as a finding: 79% of test frauds fall after the drift switch, yet XGB val->test PR-AUC rose (0.929->0.962, no degradation observed) [final_results.json:drift]
- 53-feature registry, 12 leakage tests, 233 automated tests [final_results.json:meta]
