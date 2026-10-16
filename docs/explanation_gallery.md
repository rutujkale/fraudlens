# Local explanation gallery (Part 10D)

Six validation cases selected by **pre-registered rules** (not by looks): each row below records the pool it was drawn from, so a reader can see how representative it is. SHAP here explains the raw model margin; the reported probability is the calibrated one (the honesty rule in `docs/methodology.md`).

Caveat: since the Part 02b cold-start fix, generated transactions are gated on the customer's `account_open_date`, so no row has `history_days < 0` (the features builder asserts this). The cold-start case is defined by `0 <= history_days < 2` — a genuinely brand-new account, not the pre-fix generation defect (see `docs/what_went_wrong.md`).

**Policy:** amount-aware policy `cost_policy_v1` (scenario `balanced`), calibrator `sigmoid`, capacity 23/day

| case | transaction_id | fraud | scenario | stealth | amount (Rs) | p_calibrated | action | pool |
|---|---|---|---|---|---|---|---|---|
| clear_tp | T0212883 | 1 | burst | 0.4421 | 23,471.08 | 0.999963 | BLOCK | 125 |
| subtle_tp | T0200869 | 1 | amount_anomaly | 0.7903 | 2,532.22 | 0.201278 | STEP_UP | 51 |
| false_negative | T0200867 | 1 | amount_anomaly | 0.7903 | 1,731.29 | 0.015410 | APPROVE | 14 |
| hard_negative_fp | T0210287 | 0 | None | n/a | 2,246.89 | 0.844267 | BLOCK | 58 |
| high_amount_tn | T0203070 | 0 | None | n/a | 189,756.14 | 0.000321 | APPROVE | 111 |
| cold_start | T0218309 | 0 | None | n/a | 5,600.91 | 0.531980 | BLOCK | 70 |

## Selection rules

- **clear_tp** — fraud, action != APPROVE, p >= 0.95; highest p
- **subtle_tp** — fraud, action != APPROVE, stealth >= p75 (0.442); highest stealth
- **false_negative** — fraud, action == APPROVE; highest stealth then highest p
- **hard_negative_fp** — legit, action != APPROVE, travel_speed >= p90 (30 km/h); highest p
- **high_amount_tn** — legit, action == APPROVE, amount >= p99 (Rs 55,083); largest amount
- **cold_start** — 0 <= history_days < 2; highest p

## clear_tp

`T0212883` — Correctly flagged fraud (burst) with p=1.0000, action BLOCK. The drivers below are the ones that pushed the margin up; the model did not need anything exotic to catch this one.

- **Raises risk:** Only 1 minute since the previous transaction; First time this customer pays this recipient; channel (encoded category code) = 2; First time at this merchant
- **Mitigating:** Only 60% of the customer's prior transactions fall within +/-2h of this hour; merchant category (encoded category code) = 10; Transaction amount of ₹ 23,471; 100% of prior 30-day spend is in this region
- **Figure:** `reports/figures/shap_waterfall_clear_tp_dev.png`

## subtle_tp

`T0200869` — High-stealth fraud (stealth=0.790, scenario amount_anomaly) still flagged, p=0.2013. Stealth raises the chance the transaction looks ordinary in-level; the catch comes from the combination of drivers rather than a single extreme value.

- **Raises risk:** Only 1 minute since the previous transaction; 0.0 days since the previous transaction; merchant category (encoded category code) = 4; Only 18% of the customer's prior transactions fall within +/-2h of this hour
- **Mitigating:** 22 prior transactions from this device; Transaction on weekday index 6 (Mon=0); Transaction amount of ₹ 2,532; 91% of prior 30-day spend is in this region
- **Figure:** `reports/figures/shap_waterfall_subtle_tp_dev.png`

## false_negative

`T0200867` — Missed fraud: p=0.0154 -> APPROVE. This is the highest-stealth missed fraud in val_thr (stealth=0.790, scenario amount_anomaly). It looks like the customer's own behaviour, and the features the model relies on do not fire; a review budget cannot catch every deliberate imitation.

- **Raises risk:** merchant category (encoded category code) = 2; 0 transactions from this device in the prior 30 days; 1 prior transaction from this device; 0% of prior 30-day spend is at this merchant
- **Mitigating:** Only 435 minutes since the previous transaction; Device first seen 46 days ago; Amount is 1.1x this customer's 7-day average; Transaction on weekday index 6 (Mon=0)
- **Figure:** `reports/figures/shap_waterfall_false_negative_dev.png`

## hard_negative_fp

`T0210287` — False positive on a legitimate transaction (p=0.8443 -> BLOCK). A travel/velocity pattern that mimics account takeover is exactly the kind of hard negative the model is expected to trade against; the cost of this review is Rs 75 of analyst time plus friction on a good customer.

- **Raises risk:** 2% of prior 30-day spend is in this region; 2 transactions in the previous hour; merchant category (encoded category code) = 1; Only 5 minutes since the previous transaction
- **Mitigating:** 19 prior transactions from this device; The customer's 30-day average amount is about ₹ 1,794; Amount is 1.5x this customer's 7-day average; Device first seen 90 days ago
- **Figure:** `reports/figures/shap_waterfall_hard_negative_fp_dev.png`

## high_amount_tn

`T0203070` — Large legitimate transaction (Rs 189,756) correctly approved (p=0.0003). High amounts are not fraudulent by themselves; the amount features are relative to the customer's own baseline, which is what keeps this one clear.

- **Raises risk:** channel (encoded category code) = 2; Account opened 88.8 days ago; Largest prior amount today is ₹ 31,747; Amount is 7.6x the customer's lifetime median
- **Mitigating:** 7% of prior 30-day spend is at this merchant; tx from device prior: no prior history; device count 30d: no prior history; 217 km from the customer's home city
- **Figure:** `reports/figures/shap_waterfall_high_amount_tn_dev.png`

## cold_start

`T0218309` — Cold start by the book: the account opened 1.34 days before this transaction, p=0.5320 -> BLOCK. The account is brand-new, so its own behavioural history is thin (n_prior_tx=3) and adds little; this case shows how the amount-aware policy treats a low-history account, not a generation defect (post Part 02b the simulator no longer emits pre-open rows).

- **Raises risk:** Only 1 minute since the previous transaction; 0.0 days since the previous transaction; 0 prior transactions from this device; merchant category (encoded category code) = 5
- **Mitigating:** Account opened 1.3 days ago; Amount is 0.2x the customer's lifetime median; 1-hour activity is 0.3x the 24-hour rate; hour deviation: no prior history
- **Figure:** `reports/figures/shap_waterfall_cold_start_dev.png`
