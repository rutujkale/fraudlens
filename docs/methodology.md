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
