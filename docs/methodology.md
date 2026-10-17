# Methodology notes

Short, evidence-backed notes on design choices whose *reasoning* is not obvious from
the code alone. Every number here is produced by a committed script or test; nothing is
estimated.

## Point-in-time features use one window primitive (Part 05)

All behavioural features are computed with the strict window `[T − w, T)`: a feature for a
transaction at time `T` may only use rows with `timestamp < T`. Two consequences:

- **Current row and same-timestamp peers are excluded.** `features/windows.window_bounds`
  returns `(searchsorted(ts, T−w, "left"), searchsorted(ts, T, "left"))`; the right-hand
  `side="left"` drops every row sharing `T`.
- **Tie-group processing.** Per customer, rows at the same timestamp are handled as one
  group: features are computed from state *as of before the group*, then the group updates
  the running state (seen merchants/categories/recipients/devices/regions, hour histogram).
  Cross-customer device state (`accounts_seen_on_device`, `device_age_days`) uses the same
  snapshot-then-apply pattern in global `(timestamp, transaction_id)` order.

### Why `searchsorted` lives in exactly one function

`window_bounds` is the **only** call site of `np.searchsorted` in the feature package
(guarded by `tests/test_leakage.py::test_searchsorted_lives_in_windows_only`). Every
aggregate routes through it, so the leakage canary (`test_8`) can replace a *single*
function and exercise the whole feature set. If one feature smuggled in its own bounds,
the canary could pass while that feature leaked — the single injection point is what makes
the test suite trustworthy.

### Invariance tests beat eyeballing

The suite proves point-in-time correctness with invariants rather than spot checks:
truncation invariance (features for rows `≤ T_c` are identical whether or not later rows
exist), future-mutation invariance (shuffling/dropping later rows changes nothing), and
training/serving parity (the live path is the same code). `test_8` deliberately breaks the
window three ways and asserts the right tests fail — a test suite that cannot fail is not
evidence.

## PostgreSQL index evidence for the customer-history query (Part 04)

The API (Part 13) and the point-in-time SQL cross-check both fetch a customer's recent
history with `ORDER BY timestamp DESC LIMIT n`. `reports/db_explain_dev.txt` records
`EXPLAIN (ANALYZE, BUFFERS)` before and after `sql/indexes.sql` on `dev`
(`raw.transactions`, 277,607 rows):

| plan | execution time | buffers | scan |
|---|---|---|---|
| without `idx_transactions_customer_ts` | 22.739 ms | 5075 | `Seq Scan` + `Sort`, 277,257 rows removed by filter |
| with `idx_transactions_customer_ts` | 0.131 ms | 52 | `Index Scan Backward`, index cond pushdown, no sort |

That is **≈174× faster** (22.739 → 0.131 ms) and **≈98× fewer buffers** (5075 → 52),
because the composite `(customer_id, timestamp)` index turns "scan the whole table then
sort the top 50" into "walk the index backwards from the cutoff and stop at 50". The index
also serves the strictly-prior `timestamp < before_ts` predicate directly
(`Index Cond`), which is exactly the access pattern the service layer needs.

## SHAP values are margin pushes, never points on the risk score (Part 10)

`TreeExplainer(model, model_output="raw")` attributes the **raw margin** (log-odds) that
`predict_proba` sees, so `base_value + Σ SHAP = margin` and the attested identity holds to
`1.2e-5` on dev (`tests/test_explain.py::test_shap_additivity_and_best_iteration`). Two
consequences:

- **Never plot a SHAP value as a position on the 0–100 risk score.** `risk_score` is
  `100 × calibrated probability`, and the sigmoid calibrator is a non-linear map of the
  margin, so a "+0.6 SHAP" contribution is not "0.6 risk points". SHAP answers *which
  features moved the model rank / margin*, and the calibrated probability (plus the
  amount-aware action) is the only thing quoted as a risk number.
- **The explainer must honour `best_iteration`.** The dev artifact early-stopped at 450 of
  481 built rounds; explaining past `best_iteration` would attribute trees the model does not
  score with (margin off by 0.410 worst-case, 0.198 mean). `ShapExplainer.raw_margin` passes
  `iteration_range=(0, best_iteration + 1)`, matching the booster default.

`mean|SHAP|` is a **usage** measure, not a utility measure: a feature can dominate
`mean|SHAP|` while contributing nothing out-of-sample. Experiment 8D is the utility
evidence (permutation PR-AUC drop), and the two are reported next to each other in
`results/shap_global_dev.csv` rather than conflated.

## Simulated subgroup gaps are not real-world fairness claims (Part 10)

The subgroup audit (`docs/fairness_audit.md`) reports per-group operating metrics for the
frozen policy, but the data is synthetic: group differences are the simulator's assumptions
about age/income/archetype, not demographic reality. Suppression (groups with < 30 frauds
or < 1,000 rows) is reported *up front*, per-group calibration/Brier is excluded from the
exploratory full-validation view (the calibrator saw half of it), and every interval is a
**cluster** bootstrap over customers, because one customer's transactions are correlated.
The audit is not a compliance claim.

## Simulator assumptions (Part 02)

`data_gen/` produces entities (customers, accounts, devices, merchants) and transactions
from config (`config/config.yaml`), with archetype mixes, hard negatives and drift all
seeded (`seed: 42`) so a run is exactly reproducible. The behavioural story is deliberately
imperfect: a share of legitimate transactions share devices and a configured share of fraud
reuses known devices, so "new device" is not a giveaway (`hard_negatives.py`). Fraud is
generated as **episodes** (account takeover, structuring, amount/geo anomaly, burst) with a
configured `label_delay_days`; hard negatives (travel, payday, shared device, near-threshold)
are simulated as explicit episodes and measured, not ignored. Drift is a config-driven
change in the scenario mix (`drift.start_fraction` / `strength`). The separability gate
(`data_quality/validation.py`) fails a profile whose crude signals solve fraud too easily,
so an accidental shortcut cannot ship as a result. Details and the cold-start fix are in
`docs/progress/part_02.md`, `part_02b.md`.

## Validation strategy (Parts 06, 12)

Splits are **chronological with an embargo equal to the maximum label delay**; a random
split is never the primary evaluation. `evaluation/splits.py` fixes `t0`, `t_train_end`,
`t_val_end` (and the test window beyond it); the embargo gap drops rows whose labels could
not have been known at the boundary. Hyperparameters, calibrators and thresholds are fit on
train/validation only; the test window is evaluated **once**, in Part 12, through
`evaluation/guard.py`, and every access is logged
(`results/test_access_log.csv`). Confidence intervals are **cluster bootstraps over
customers** (`evaluation/bootstrap.py`), because one customer's transactions are correlated.

## Imbalance (Parts 06, 8A)

The positive class is ~1% (`composition.prevalence`). The headline metric is **PR-AUC**,
with recall@top-K, recall at a fixed precision, FPR, Brier, ECE and expected/realized cost
reported alongside; accuracy is never a headline metric. **No SMOTE** or any synthetic
oversampling is used. Experiment 8A (`docs/experiments/8A_imbalance.md`) shows that
reweighting/calibration choices move the ranking little here, and that raw XGBoost
probabilities were already close to calibrated — the experiment is the evidence, not an
assumption.

## Calibration (Part 07)

Probabilities are calibrated (sigmoid on a held-out validation slice) because the decision
rule needs a probability that means "frequency", not a margin. `models/calibration.py` fits
the calibrator on `val_cal` (half of validation); Brier and ECE are reported before/after in
`results/final_results.json`. A calibrated probability is the only thing converted to a risk
score (`risk_score = 100 × p`); the threshold is derived from it.

## Cost model and the closed-form threshold (Parts 07, 12)

Per transaction with calibrated `p` and amount `A`, the four actions are priced
(`results/freeze.json:cost_params`):

```
cost(APPROVE) = p·(α·A + fixed_fee)
cost(STEP_UP) = S + (1-p)·F_step + p·(1-s)·(α·A + fixed_fee)
cost(REVIEW)  = R + (1-p)·F + p·(1-c)·(α·A + fixed_fee)
cost(BLOCK)   = (1-p)·F_block
```

With only APPROVE vs REVIEW and a perfect catch rate (`c = 1`), intervening is worth it
exactly when

```
p > (R + F) / (α·A + fixed_fee + F)
```

— an **amount-dependent** threshold, valid only because `p` is calibrated. On `dev`,
`α = 0.85`, `R = 75`, `F = 150`, `fixed_fee = 0` (`results/freeze.json:cost_params`), so the
cutoff rises with `A`: small transactions need a higher probability before a review pays for
itself.

## Capacity allocation (Parts 07, 12)

Analyst reviews are capped at `capacity_per_day = 23` (`results/freeze.json`). The policy
ranks candidates by **largest expected saving** and **demotes** everything over capacity to
the next-best action (STEP_UP or APPROVE). Risk **levels derive from actions**
(APPROVE→LOW, STEP_UP→MEDIUM, REVIEW→HIGH, BLOCK→CRITICAL), not from fixed probability
cutoffs. Honest caveat: on the test window the policy flagged `5.9/day` and `n_demoted = 0`,
so the cap never bound — the capacity result is the unconstrained policy
(`docs/limitations.md`).

## Anomaly experiment design (Part 09)

The Isolation Forest is evaluated as an **unseen-fraud** detector: train on the normal
population, score fraud archetypes held out of supervised training, and compare bagged
confidence intervals for the S1−S0 / S1s−S0 detection improvements over the supervised
baseline. The result is **null / not adopted** — the bagged CIs include 0 after the
cold-start fix — and is reported as such
(`docs/experiments/09_unseen_fraud.md`, `results/final_results.json:drift`).

## Drift and retraining (Parts 8E, 15A)

Covariate drift (inputs move) and concept drift (input→label mapping moves) are separated.
On the temporal split, naive adversarial validation is dominated by **monotonic tenure
features by construction**; strip them and behavioural drift is moderate
(adversarial AUC `0.6995` on the post-fix `drift_lab`). Feature/score PSI, prevalence /
calibration drift and delayed-label performance are monitored per month over train +
validation, with pre-stated retraining rules (`docs/retraining_plan.md`,
`docs/experiments/8E_drift.md`). The published test-window drift is "no degradation observed
(0.9291 val → 0.9619 test)".

## External validation and the sim-to-real gap (Part 11)

The IEEE-CIS benchmark is used as a **secondary** validation via `external/ieee_cis.py`:
transferable FraudLens features are mapped onto the public schema (UID heuristic), a model
is trained/validated temporally, and the gap between synthetic and real PR-AUC is recorded
in `results/sim_to_real.json`. The benchmark shows the direction of transfer, not parity;
the real dataset is not redistributed (`docs/external_validation.md`, `LICENSE`).
