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
(`raw.transactions`, 289,870 rows):

| plan | execution time | buffers | scan |
|---|---|---|---|
| without `idx_transactions_customer_ts` | 23.519 ms | 5267 | `Seq Scan` + `Sort`, 289,517 rows removed by filter |
| with `idx_transactions_customer_ts` | 0.136 ms | 49 | `Index Scan Backward`, index cond pushdown, no sort |

That is **≈173× faster** (23.519 → 0.136 ms) and **≈107× fewer buffers** (5267 → 49),
because the composite `(customer_id, timestamp)` index turns "scan the whole table then
sort the top 50" into "walk the index backwards from the cutoff and stop at 50". The index
also serves the strictly-prior `timestamp < before_ts` predicate directly
(`Index Cond`), which is exactly the access pattern the service layer needs.
