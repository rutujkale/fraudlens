# Part 05 — Point-in-Time Feature Engineering & Leakage Test Suite

## Goal
One reusable feature module that computes behavioral features for each transaction using only data strictly before it, plus an automated test suite that **proves** no future information leaks in, including a test that detects deliberately leaky implementations. The same code path serves training, the API and Streamlit.

## Prerequisites
Parts 01–03 complete (Part 04 optional for this part, but needed for the SQL cross-check test).

## Design requirements

- Public API in `features/build_features.py`:
  - `build_features(tx: DataFrame, entities: Entities, cfg) -> DataFrame` — computes features for **every row** of `tx` (sorted by `(timestamp, transaction_id)`), using only rows strictly earlier in time (timestamp `<` T). Returns `transaction_id`, `feature_as_of_timestamp` (= tx timestamp), and feature columns.
  - `features/serving.py: compute_features_for_transaction(tx_row: dict, history: DataFrame, entities, cfg) -> pd.Series` — appends the new transaction to the history and calls the **same** `build_features`, returning only the last row. No separate re-implementation.
- `features/registry.py`: a registry listing every feature with `name, group, dtype, description, window, requires_columns`, `FEATURE_VERSION = "features_v1"`, `FEATURE_COLUMNS`, `CATEGORICAL_COLUMNS`, `FORBIDDEN_COLUMNS` (is_fraud, fraud_scenario, episode_id, stealth, hard_negative_type, label_available_at, anything target-derived). `assert_no_target_columns(df)` raises if a forbidden column appears. The registry auto-generates `docs/data_dictionary_features.md`.
- **Strictness:** window `[T − w, T)`. The current row and rows with the same timestamp are excluded.
- **Performance approach (important):** per customer (and per device), keep arrays of sorted timestamps and cumulative sums (`cumsum` of amount and amount²). For each row, locate the window bounds with `np.searchsorted(ts, T - w, side="left")` and `np.searchsorted(ts, T, side="left")`. This is exact, strict-prior and fast. Use it for counts, sums, means, standard deviations (via sum of squares). Distinct counts (devices, recipients, merchants in a window) may use per-customer loops. Target: `main` profile feature build in a few minutes; profile it and report timing.
- **Cold start:** customers with no prior history get NaN for history-based statistics, plus `n_prior_tx`, `history_days`, `is_cold_start`. Do not impute inside the feature module; models handle missing values.
- Static customer attributes allowed: home location, `account_age_days` (computed as T − account_open_date). **Not** used as features: age, income_band (reserved for the fairness audit), archetype, merchant_risk_group.

## Feature set (~40; implement all, adjust only with justification)

| Group | Features |
|---|---|
| Transaction | amount, log_amount, hour, day_of_week, is_weekend, is_night, transaction_type, channel, merchant_category (categoricals) |
| Recency | minutes_since_prev_tx, days_since_last_tx, n_prior_tx, history_days, is_cold_start |
| Frequency | tx_count_10m, 1h, 6h, 24h, 7d; active_days_7d, active_days_30d; velocity_ratio_1h_vs_24h |
| Amount sums | amount_sum_1h, amount_sum_24h, max_amount_24h, count_near_threshold_24h (tx with amount in [0.55, 1.0)×`reporting_threshold`) |
| Amount deviation | avg_amount_7d, avg_amount_30d, std_amount_30d, amount_ratio_to_7d_mean, amount_ratio_to_30d_mean, amount_zscore_30d (guard small std), amount_vs_lifetime_median |
| Merchant / category | new_merchant_flag, new_category_flag, category_freq_30d, merchant_freq_30d |
| Device | is_new_device (never seen for this customer before T), device_age_days, tx_from_device_prior, accounts_seen_on_device (distinct customers who used the device before T), device_count_30d |
| Recipient | unique_recipients_1h, unique_recipients_24h, is_new_recipient |
| Geographic | distance_from_home_km (haversine), distance_from_prev_km, travel_speed_kmh (capped), new_region_flag, region_freq_30d |
| Time-of-day behavior | hour_deviation (share of the customer's prior transactions within ±2 h of this hour; NaN if < 5 prior) |
| Trend | activity_change_7d_vs_prev_7d, amount_change_7d_vs_prev_7d |

Feature names must be consistent with the registry and used verbatim in later parts.

## Tasks

### 5A — Implement windows and features
`features/windows.py` (searchsorted + cumsum helpers, generic and unit-testable), one function per group in `build_features.py` (or split modules `behavioral.py, velocity.py, device.py, geographic.py, temporal.py` under `features/`). Register every feature.

### 5B — Offline pipeline script
`scripts/build_features.py --profile dev` reads raw parquet (or PostgreSQL via a flag), writes `data/features/<profile>/features_v1.parquet` and, optionally, `features.feature_snapshot`. Prints feature count, NaN rates, runtime.

### 5C — Unit tests (`tests/test_features.py`)
Hand-computed mini example (≈10 transactions for 2 customers) with expected values for every feature group: counts at window edges (a tx exactly 1 h earlier is **inside** `[T−1h, T)`; a tx exactly at T is excluded), first transaction ⇒ zeros/NaNs, haversine known distance (Mumbai–Delhi ≈ 1,150 km), travel speed capping, zero-std guard, new-device logic, near-threshold count.

### 5D — Leakage test suite (`tests/test_leakage.py`)
1. **Truncation invariance (core test):** for random cut times `T_c`, features computed on the full dataset for rows with `timestamp ≤ T_c` must equal (`assert_frame_equal` with tolerance) features computed on `tx[tx.timestamp ≤ T_c]` only.
2. **Future mutation test:** copy the dataset; for all rows after `T_c`, shuffle amounts, change devices and locations, drop rows. Features for rows `≤ T_c` must be unchanged.
3. **Current-row exclusion:** for each customer's first transaction, all count features are 0 and all history stats NaN; `tx_count_1h` for a row equals the brute-force count over a naive O(n²) reference implementation on a small random dataset (property test comparing fast vs naive).
4. **Same-timestamp ties:** two transactions with equal timestamps do not count each other.
5. **No target columns:** `assert_no_target_columns(feature_df)`; `FEATURE_COLUMNS ∩ FORBIDDEN_COLUMNS = ∅`; features computed from a transactions frame lacking label columns give identical results.
6. **Train/validation boundary:** features for training rows are identical whether or not validation/test rows exist in the input (special case of 1; implement with the actual split boundaries from the config).
7. **Label-independence:** randomly permute `is_fraud` in the labels table and recompute features → identical.
8. **Canary test (tests the tests):** create deliberately leaky variants in the test module (e.g. a window that includes the current row; a window `[T−w, T+w]`; a mean over the whole customer history). Assert that tests 1–2 **fail** on each leaky variant. This proves the suite can detect leakage.
9. **Training/serving parity:** for ≥ 200 random historical transactions, `compute_features_for_transaction(tx, history_before_tx)` equals the offline row (tolerance 1e-9, NaN-aware).
10. **SQL cross-check (marker `db`):** `tx_count_24h` computed in Python equals the SQL view from Part 04 on a sample (explain any tie-handling difference).

### 5E — Leakage "incident" demo (for the write-up)
Temporarily introduce an inclusive-window bug in a scratch branch/copy, show that test 1 fails and that a quick XGBoost on validation looks suspiciously better, then revert. Record in `docs/what_went_wrong.md` ("Leakage incident": how detected, performance change, test that prevents it). Use actual numbers.

## Acceptance criteria
```bash
pytest tests/test_features.py tests/test_leakage.py -q
python scripts/build_features.py --profile dev
python -c "import pandas as pd; df=pd.read_parquet('data/features/dev/features_v1.parquet'); print(df.shape)"
```
Progress file: feature count, NaN-rate table, runtime on dev (and main if generated), leakage-suite results including the canary outcomes, the incident write-up.

## Do not
Use pandas `rolling` with closed on both sides, `groupby().transform` over the whole customer history, or any aggregate computed over the full dataset. Import label tables inside `features/`. Duplicate feature logic elsewhere.
