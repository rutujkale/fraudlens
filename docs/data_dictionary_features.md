# Feature data dictionary (features_v1)

53 features. Generated from `features/registry.py`; do not edit by hand.

| feature | group | dtype | window | requires | description |
|---|---|---|---|---|---|
| `amount` | transaction | float | row | amount | Transaction amount (INR). |
| `log_amount` | transaction | float | row | amount | log1p(amount); tames the long right tail. |
| `hour` | transaction | int | row | timestamp | Hour of day (0-23) in IST. |
| `day_of_week` | transaction | int | row | timestamp | ISO day of week (0=Mon .. 6=Sun). |
| `is_weekend` | transaction | int8 | row | timestamp | 1 if Saturday/Sunday. |
| `is_night` | transaction | int8 | row | timestamp | 1 if hour >= 22 or hour < 6. |
| `transaction_type` | transaction | category | row | transaction_type | PURCHASE / P2P_TRANSFER / BILL_PAY / CASH_WITHDRAWAL. |
| `channel` | transaction | category | row | channel | APP / WEB / POS / ATM. |
| `merchant_category` | transaction | category | row | merchant_category | Merchant category (NULL for P2P). |
| `minutes_since_prev_tx` | recency | float | lifetime | timestamp | Minutes since this customer's previous transaction (NaN if none). |
| `days_since_last_tx` | recency | float | lifetime | timestamp | Days since previous transaction (NaN if none). |
| `n_prior_tx` | recency | int | lifetime | timestamp | Count of strictly-prior transactions (0 for the first). |
| `history_days` | recency | float | lifetime | timestamp | Days since account opening (static, always defined). |
| `is_cold_start` | recency | int8 | lifetime | timestamp | 1 if the customer has no strictly-prior transaction. |
| `tx_count_10m` | frequency | int | 10m | timestamp | Prior tx in the last 10 minutes. |
| `tx_count_1h` | frequency | int | 1h | timestamp | Prior tx in the last hour. |
| `tx_count_6h` | frequency | int | 6h | timestamp | Prior tx in the last 6 hours. |
| `tx_count_24h` | frequency | int | 24h | timestamp | Prior tx in the last 24 hours. |
| `tx_count_7d` | frequency | int | 7d | timestamp | Prior tx in the last 7 days. |
| `active_days_7d` | frequency | int | 7d | timestamp | Distinct calendar days with a prior tx in the last 7d. |
| `active_days_30d` | frequency | int | 30d | timestamp | Distinct calendar days with a prior tx in the last 30d. |
| `velocity_ratio_1h_vs_24h` | frequency | float | 24h | timestamp | tx_count_1h / tx_count_24h (NaN if none in 24h). |
| `amount_sum_1h` | amount_sums | float | 1h | amount | Sum of prior amounts in the last hour. |
| `amount_sum_24h` | amount_sums | float | 24h | amount | Sum of prior amounts in the last 24 hours. |
| `max_amount_24h` | amount_sums | float | 24h | amount | Max prior amount in the last 24 hours (NaN if none). |
| `count_near_threshold_24h` | amount_sums | int | 24h | amount | Prior 24h tx with amount in the structuring band. |
| `avg_amount_7d` | amount_deviation | float | 7d | amount | Mean prior amount over 7d (NaN if none). |
| `avg_amount_30d` | amount_deviation | float | 30d | amount | Mean prior amount over 30d (NaN if none). |
| `std_amount_30d` | amount_deviation | float | 30d | amount | Std of prior amount over 30d (NaN if none). |
| `amount_ratio_to_7d_mean` | amount_deviation | float | 7d | amount | amount / avg_amount_7d (NaN if no mean). |
| `amount_ratio_to_30d_mean` | amount_deviation | float | 30d | amount | amount / avg_amount_30d (NaN if no mean). |
| `amount_zscore_30d` | amount_deviation | float | 30d | amount | Z-score of amount vs prior 30d (std floored). |
| `amount_vs_lifetime_median` | amount_deviation | float | lifetime | amount | amount / median prior amount (NaN if none/0). |
| `new_merchant_flag` | merchant_category | float | lifetime | merchant_id | 1 if merchant never seen for this customer (NaN if NULL). |
| `new_category_flag` | merchant_category | float | lifetime | merchant_category | 1 if category never seen for this customer (NaN if NULL). |
| `category_freq_30d` | merchant_category | float | 30d | merchant_category | Share of prior 30d tx in this category (NaN if none). |
| `merchant_freq_30d` | merchant_category | float | 30d | merchant_id | Share of prior 30d tx at this merchant (NaN if none). |
| `is_new_device` | device | float | lifetime | device_id | 1 if device never used by this customer before T (NaN if NULL). |
| `device_age_days` | device | float | lifetime | device_id, timestamp | Days since the device's first prior use, any customer (NaN if new/NULL). |
| `tx_from_device_prior` | device | float | lifetime | device_id | Prior tx by this customer from this device (NaN if NULL). |
| `accounts_seen_on_device` | device | float | lifetime | device_id | Distinct customers who used the device before T (NaN if NULL). |
| `device_count_30d` | device | float | 30d | device_id | Prior 30d tx by this customer from this device (NaN if NULL). |
| `unique_recipients_1h` | recipient | int | 1h | recipient_id | Distinct recipients in prior hour. |
| `unique_recipients_24h` | recipient | int | 24h | recipient_id | Distinct recipients in prior 24h. |
| `is_new_recipient` | recipient | float | lifetime | recipient_id | 1 if recipient never used by this customer before T (NaN if NULL). |
| `distance_from_home_km` | geographic | float | row | location_id | Haversine distance from the customer's home city (km). |
| `distance_from_prev_km` | geographic | float | lifetime | location_id | Distance from the previous transaction's city (NaN if none). |
| `travel_speed_kmh` | geographic | float | lifetime | location_id | Implied speed since previous tx, capped (NaN if none/0 dt). |
| `new_region_flag` | geographic | int | lifetime | location_id | 1 if the transaction region was never seen before by this customer. |
| `region_freq_30d` | geographic | float | 30d | location_id | Share of prior 30d tx in this region (NaN if none). |
| `hour_deviation` | temporal | float | lifetime | timestamp | Share of prior tx within +/-2h of this hour (NaN if <5 prior). |
| `activity_change_7d_vs_prev_7d` | trend | float | 14d | timestamp | tx_count_7d / count in the previous 7d window (NaN if prev=0). |
| `amount_change_7d_vs_prev_7d` | trend | float | 14d | amount | amount_sum_7d / sum in the previous 7d window (NaN if prev=0). |
