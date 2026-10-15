# Local explanation gallery (Part 10D)

Six validation cases selected by **pre-registered rules** (not by looks): each row below records the pool it was drawn from, so a reader can see how representative it is. SHAP here explains the raw model margin; the reported probability is the calibrated one (the honesty rule in `docs/methodology.md`).

Caveat: the simulator gives ~10% of customers a mid-window `account_open_date` but still generates their transactions from day 0, so ~4% of rows have `history_days < 0` (see `docs/what_went_wrong.md`). The cold-start case therefore requires `0 <= history_days < 2`; a negative-history row would be a generation defect, not a cold start.

**Policy:** amount-aware policy `cost_policy_v1` (scenario `balanced`), calibrator `sigmoid`, capacity 23/day

| case | transaction_id | fraud | scenario | stealth | amount (Rs) | p_calibrated | action | pool |
|---|---|---|---|---|---|---|---|---|
| clear_tp | T0217039 | 1 | burst | 0.4021 | 833.08 | 1.000000 | BLOCK | 101 |
| subtle_tp | T0230915 | 1 | amount_anomaly | 0.6867 | 1,411.48 | 0.999932 | BLOCK | 29 |
| false_negative | T0230912 | 1 | amount_anomaly | 0.6867 | 1,174.53 | 0.006097 | APPROVE | 5 |
| hard_negative_fp | T0229331 | 0 | None | n/a | 710.16 | 0.431502 | STEP_UP | 17 |
| high_amount_tn | T0215456 | 0 | None | n/a | 312,045.64 | 0.000139 | APPROVE | 180 |
| cold_start | T0231027 | 0 | None | n/a | 5,600.91 | 0.012758 | APPROVE | 44 |

## Selection rules

- **clear_tp** — fraud, action != APPROVE, p >= 0.95; highest p
- **subtle_tp** — fraud, action != APPROVE, stealth >= p75 (0.501); highest stealth
- **false_negative** — fraud, action == APPROVE; highest stealth then highest p
- **hard_negative_fp** — legit, action != APPROVE, travel_speed >= p90 (29 km/h); highest p
- **high_amount_tn** — legit, action == APPROVE, amount >= p99 (Rs 55,096); largest amount
- **cold_start** — 0 <= history_days < 2; highest p

## clear_tp

`T0217039` — Correctly flagged fraud (burst) with p=1.0000, action BLOCK. The drivers below are the ones that pushed the margin up; the model did not need anything exotic to catch this one.

- **Raises risk:** Only 1 minute since the previous transaction; First time this customer pays this recipient; 5 transactions in the previous hour; 0.0 days since the previous transaction
- **Mitigating:** merchant category (encoded category code) = 10; Only 70% of the customer's prior transactions fall within +/-2h of this hour; Transaction at 14:00 IST; 96% of prior 30-day spend is in this region
- **Figure:** `reports/figures/shap_waterfall_clear_tp_dev.png`

## subtle_tp

`T0230915` — High-stealth fraud (stealth=0.687, scenario amount_anomaly) still flagged, p=0.9999. Stealth raises the chance the transaction looks ordinary in-level; the catch comes from the combination of drivers rather than a single extreme value.

- **Raises risk:** Only 1 minute since the previous transaction; 0.0 days since the previous transaction; 0 prior transactions from this device; device age days: no prior history
- **Mitigating:** Amount is 1.8x the customer's lifetime median; 100% of prior 30-day spend is in this region; 1 transaction in the previous hour; The customer's 30-day amount volatility is ₹ 1,228
- **Figure:** `reports/figures/shap_waterfall_subtle_tp_dev.png`

## false_negative

`T0230912` — Missed fraud: p=0.0061 -> APPROVE. This is the highest-stealth missed fraud in val_thr (stealth=0.687, scenario amount_anomaly). It looks like the customer's own behaviour, and the features the model relies on do not fire; a review budget cannot catch every deliberate imitation.

- **Raises risk:** Only 0% of the customer's prior transactions fall within +/-2h of this hour; Transaction at 02:00 IST; merchant category (encoded category code) = 4; Transaction during the night window (22:00-06:00 IST)
- **Mitigating:** 37 prior transactions from this device; Only 2,202 minutes since the previous transaction; Amount is 1.5x the customer's lifetime median; Amount is 0.6x this customer's 7-day average
- **Figure:** `reports/figures/shap_waterfall_false_negative_dev.png`

## hard_negative_fp

`T0229331` — False positive on a legitimate transaction (p=0.4315 -> STEP_UP). A travel/velocity pattern that mimics account takeover is exactly the kind of hard negative the model is expected to trade against; the cost of this review is Rs 75 of analyst time plus friction on a good customer.

- **Raises risk:** merchant category (encoded category code) = 1; 0% of prior 30-day spend is in this region; First time at this merchant; 983 km from the previous transaction's city
- **Mitigating:** Only 94% of the customer's prior transactions fall within +/-2h of this hour; Transaction at 12:00 IST; Device first seen 90 days ago; 15 prior transactions from this device
- **Figure:** `reports/figures/shap_waterfall_hard_negative_fp_dev.png`

## high_amount_tn

`T0215456` — Large legitimate transaction (Rs 312,046) correctly approved (p=0.0001). High amounts are not fraudulent by themselves; the amount features are relative to the customer's own baseline, which is what keeps this one clear.

- **Raises risk:** Amount is 14.1x the customer's lifetime median; Amount is 11.0x this customer's 30-day average; Transaction amount of ₹ 312,046; 0 km from the customer's home city
- **Mitigating:** 185 prior transactions from this device; Only 48% of the customer's prior transactions fall within +/-2h of this hour; merchant category (encoded category code) = 10; 73 transactions from this device in the prior 30 days
- **Figure:** `reports/figures/shap_waterfall_high_amount_tn_dev.png`

## cold_start

`T0231027` — Cold start by the book: the account opened 1.3 days before this transaction, p=0.0128 -> APPROVE. Note the contradiction the rule exposes: this customer already has n_prior_tx=240 and the device shows prior activity, i.e. the simulator generated this customer's history before the declared `account_open_date` (see the `history_days < 0` finding in `docs/what_went_wrong.md`). A genuinely empty-history cold start cannot be exhibited with this dataset; what this case does show is that the model is comfortable approving a low-risk brand-new-account transaction.

- **Raises risk:** Only 1 minute since the previous transaction; 0.0 days since the previous transaction; merchant category (encoded category code) = 5; 2 prior transactions just below the reporting threshold in 24h
- **Mitigating:** 51 prior transactions from this device; 10 transactions from this device in the prior 30 days; Account opened 1.3 days ago; Amount is 0.4x the customer's lifetime median
- **Figure:** `reports/figures/shap_waterfall_cold_start_dev.png`
