# Part 05 — Point-in-Time Feature Engineering & Leakage Test Suite

## What was built

A single reusable feature module that computes behavioural features for every
transaction using only strictly-earlier data, plus an automated leakage test suite
with a canary that proves the tests can fail.

| Path | Purpose |
|---|---|
| `src/fraudlens/features/registry.py` | `FeatureSpec`, `FEATURE_COLUMNS`, `CATEGORICAL_COLUMNS`, `FORBIDDEN_COLUMNS`, `assert_no_target_columns`, auto-generated data dictionary |
| `src/fraudlens/features/windows.py` | `window_bounds` — the **only** `np.searchsorted` call site — plus `count_in_window`/`sum_in_window`/`lifetime_bounds` |
| `src/fraudlens/features/build_features.py` | `build_features(tx, entities, cfg)`; one block per feature group |
| `src/fraudlens/features/serving.py` | `compute_features_for_transaction` — appends to history and calls the **same** `build_features` |
| `scripts/build_features.py` | CLI → `data/features/<profile>/features_v1.parquet` + NaN-rate CSV + stats + data dictionary |
| `tests/test_features.py` | 13 hand-computed feature tests |
| `tests/test_leakage.py` | 11 leakage tests (incl. canary + single-`searchsorted` guard) |
| `docs/methodology.md` | point-in-time method + the Part 04 index evidence |
| `docs/data_dictionary_features.md` | generated from the registry |

`FeaturesConfig` gained `near_threshold_band`, `travel_speed_cap_kmh`,
`hour_deviation_halfwidth_hours`, `hour_deviation_min_prior`, `amount_std_floor`,
`trend_window_days`; Makefile target `build-features` added.

**53 features** registered (`FEATURE_VERSION = "features_v1"`), covering transaction,
recency, frequency, amount-sum, amount-deviation, merchant/category, device, recipient,
geographic, time-of-day and trend groups.

### Strictness

Window `[T − w, T)`, current row and same-timestamp peers excluded. Per customer, rows are
processed in equal-timestamp groups (`prior_end` drives recency/lifetime/hour features and
the previous-row pointer; running state is applied after the group). Cross-customer device
state uses the same snapshot-then-apply pattern globally. `is_new_device` and
`device_age_days` are derived from strictly-prior transactions, never from the static
`customer_devices` / `devices.first_seen` tables.

## Commands run and results

```bash
pytest tests/test_features.py tests/test_leakage.py -q            # 23 passed, 1 deselected (db)
pytest tests/test_leakage.py::test_10_sql_cross_check -q          # 1 passed (PostgreSQL up)
python scripts/build_features.py --profile dev                    # 289,870 rows, 53 features, 55.2 s
python -c "import pandas as pd; print(pd.read_parquet('data/features/dev/features_v1.parquet').shape)"
# (289870, 55)  -> 53 features + transaction_id + feature_as_of_timestamp
make test-fast                                                     # 80 passed, 7 deselected
make lint                                                          # All checks passed!
```

Runtime: **`tiny` 4,716 rows → 0.8 s; `dev` 289,870 rows → 55.2 s** (well inside the
"few minutes" target for `main`). The pipeline is a per-customer loop over `window_bounds`;
no pandas `rolling`, no whole-history `groupby().transform`.

### NaN-rate highlights (`dev`, full table in `data/features/dev/features_v1_nan_rates.csv`)

NaN is deliberate for missing history (models handle it; the module never imputes).

| feature | NaN rate | reason |
|---|---|---|
| `is_new_recipient` | 0.852 | recipient is NULL except P2P transfer |
| `velocity_ratio_1h_vs_24h`, `max_amount_24h` | 0.438 | no prior tx in 24 h |
| `device_age_days` | 0.347 | device NULL (non-APP) or first-ever use |
| `is_new_device`, `tx_from_device_prior`, `accounts_seen_on_device`, `device_count_30d` | 0.317 | device NULL |
| `new_merchant_flag`, `merchant_freq_30d` | ~0.20 | merchant NULL for P2P/bill-pay |
| `hour_deviation` | 0.086 | fewer than 5 prior transactions |
| everything else | ≤ 0.02 | only first-transaction cold start |

## Leakage suite outcomes (11 tests)

All pass. Covered: truncation invariance (test 1), future-mutation invariance (test 2),
current-row exclusion + O(n²) brute-force cross-check (test 3), same-timestamp ties (test 4),
no target columns / poisoned frame raises (test 5), train/validation boundary (test 6),
label independence (test 7), training/serving parity over 220 rows (test 9), SQL cross-check
vs `curated.rolling_24h_sql_demo` (test 10), and the single-`searchsorted` guard.

**Canary (test 8).**

| leaky `window_bounds` | window | tests that must fail | observed |
|---|---|---|---|
| `_leaky_future` | `[T−w, T+w]` | 1 & 2 | ✔ fails 1 & 2 |
| `_leaky_lifetime` | `[0, len(ts)]` | 1 & 2 | ✔ fails 1 & 2 |
| `_leaky_inclusive` | `[T−w, T]` (`side="right"`) | 3 & 4 | ✔ fails 3 & 4 |

Note (correction to the prompt): an *inclusive* window does **not** fail test 1 — the
current row / peer exists on both sides of a truncation cut, so truncation invariance still
holds. It fails tests 3 & 4. Test 1 only fails for a future/lifetime leak. This mapping is
the one asserted.

## Incidents (see `docs/what_went_wrong.md` for full text)

1. **Same-timestamp peers leaked into recency/running-state features** (found by test 9:
   offline `minutes_since_prev_tx=0.0` vs serving `39.0`). Fixed by tie-group processing.
2. **Part 05E leakage demo** (`results/leakage_incident_dev.json`): future-window leak
   inflates val PR-AUC 0.9560 → **0.9927** and is caught by tests 1 & 2; inclusive-window
   leak barely moves the metric (+0.0007) and is caught by tests 3 & 4. The clean 0.956
   baseline is synthetic separability, **not** leakage (invariance tests pass) and is logged
   for Part 06.

## Deviations from the prompt

- **53 features** rather than "~40"; every row of the prompt's table is implemented.
- **`history_days`** is defined as days since account opening (static, always defined), so
  the cold-start row keeps a non-NaN value as the prompt requires. `account_age_days` is not
  a separate column.
- **Inventory-not-implemented:** `--from-db` (per decision, parquet-only) and the optional
  `--to-db` write to `features.feature_snapshot` were not built (the table stays empty;
  YAGNI until Part 13 needs it). `build_features` reads only `entities.customers` and
  `entities.locations`.
- **Feature groups in one `build_features.py`** (the prompt allowed a single file; it stayed
  under a screen of scrolling per group).
- **`amount_std_floor` from config** guards the z-score (prompt said "guard small std").
- Base rate note: clean val PR-AUC 0.956 is high; flagged for Part 06 (not a Part 05 defect).

## Open issues

- Gate margins from Part 03 remain thin (`tree3_vs_boosted_gap 0.0648` vs `0.05`, overlap
  `2/2`); feature richness (0.956 PR-AUC) makes the Part 03 gate's boosted-vs-tree signal
  worth revisiting in Part 06.
- `main`-profile feature timing not yet measured (optional; `dev` is representative).
