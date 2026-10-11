# Part 02 — Behavioral Transaction & Fraud Simulator

## Goal
Build a config-driven simulator that produces a **noisy, non-trivial** financial ecosystem: customers with behavioral archetypes, legitimate transactions, hard negatives (legit but fraud-like), 5 fraud scenarios whose signals overlap with legitimate behavior, concept drift, and delayed labels. The data must NOT be solvable by a simple rule such as "new device AND high amount".

This is the most important component for credibility. Build it in the sub-steps below and test after each.

## Prerequisites
Part 01 complete (config profiles, utils, paths).

## Output (per profile) → `data/raw/<profile>/*.parquet`
| Table | Columns |
|---|---|
| customers | customer_id, age, income_band, home_region, home_location_id, account_open_date, archetype, baseline_daily_tx, baseline_amount, amount_sigma, preferred_categories (list/str), active_hour_mean, active_hour_sd, travel_frequency |
| accounts | account_id, customer_id, account_type, opening_date, status, base_currency |
| devices | device_id, first_seen, last_seen, device_type, is_shared (bool; simulation metadata) |
| customer_devices | customer_id, device_id (known devices) |
| merchants | merchant_id, merchant_category, region, merchant_risk_group |
| locations | location_id, country, state, city, latitude, longitude, region_group |
| transactions | transaction_id, customer_id, account_id, timestamp, amount, currency, merchant_id, merchant_category, device_id, location_id, recipient_id (nullable), transaction_type, channel, created_at |
| labels | transaction_id, is_fraud, fraud_scenario, episode_id, stealth, hard_negative_type, label_available_at |
| simulation_report.json | actual counts and rates (see 2G) |

Rules: **`transactions` has no label columns.** `archetype` and any field in `customers` must not encode fraud. `merchant_risk_group` is random and independent of fraud generation. `transaction_id` is assigned **after** a global sort by timestamp (`T0000001`…), so IDs carry no hint about origin. Categories: grocery, fuel, dining, electronics, travel, utilities, healthcare, fashion, entertainment, jewelry, transfers, wallet_topup. Transaction types: PURCHASE, P2P_TRANSFER, BILL_PAY, CASH_WITHDRAWAL. Channels: APP, WEB, POS, ATM.

## Sub-steps

### 2A — Entities (`entities.py`, `archetypes.py`)
- ~30–40 Indian cities as `locations` with approximate real lat/lon, state, region_group (North, South, East, West, Central, North-East). Seeded sampling only; no network calls.
- ~400–2,000 merchants (scale with customers) across categories and regions.
- Customers by archetype mix (config). Archetype parameter ranges (sample per-customer with noise, not fixed constants):
  - conservative: 0.05–0.15 tx/day, median amount ₹300–1,200, 1 device, small region radius, narrow hours.
  - regular: 0.2–0.5 tx/day, ₹500–2,500, 1–2 devices.
  - high_spend: 0.15–0.4 tx/day, ₹3,000–15,000, wider category diversity, higher tolerance for large amounts.
  - frequent_digital: 0.6–1.5 tx/day, ₹200–1,500, 2–3 devices, many categories.
  - business_like: 1.0–2.5 tx/day, ₹3,000–25,000, many recipients, multiple regions.
- Per-customer: preferred category weights (Dirichlet), active-hour mixture (mean/sd), 1–3 known devices (4% of customers share one device with a family member → `is_shared`), contacts (P2P recipients, 3–25), home location. ~10% of accounts open mid-simulation (cold-start customers).
- Age and income_band are generated but correlate only weakly with archetype (they will be used for fairness auditing in Part 10, never as model features).

### 2B — Normal behavior (`normal.py`)
For each customer-day: Poisson count with rate = baseline × day-of-week factor × customer-day random effect (gamma-distributed, so some days are busy) × festival/payday modifiers. For each transaction:
- **Time:** from the customer's hour mixture with noise; weekday/weekend shift.
- **Amount:** log-normal with customer-specific location/scale × category multiplier; ~2% legitimate outliers (3–8× typical).
- **Category/merchant:** preferred categories most of the time, ~8% exploration to less common categories; merchant sampled within category with loyalty (repeat merchants more likely).
- **Location:** home location ~88%, nearby cities ~9%, random other ~3%. 
- **Device:** known devices; channel consistent with device/category (POS/ATM transactions have `device_id` null or a fixed POS pseudo-device — choose one design and document it).
- **Recipient:** for P2P_TRANSFER draw from the customer's contacts (with ~5% new contacts); else null.
Implementation: vectorize per customer with numpy; avoid per-transaction Python loops where possible. Target: `dev` profile generates in under ~2 minutes.

### 2C — Hard negatives (`hard_negatives.py`)
Injected legitimate episodes, tagged in `labels.hard_negative_type`, together ≈ `hard_negative_target_share` of legit transactions:
- `new_phone`: new device appears and is used for the following weeks (becomes a known device).
- `travel_trip`: location jumps to a distant city for 2–10 days; first transaction at the destination within hours of the last home transaction; slightly higher amounts; travel categories.
- `holiday_splurge`: 2–6 large purchases (2–5× typical) in fashion/electronics/jewelry/travel on festival dates, with new merchants.
- `medical_bill`: single very large payment (5–30× typical) in healthcare.
- `emergency_odd_hour`: 1–3 night-time transactions at 2–4× typical.
- `payday_burst`: 3–8 bill-pay/transfer transactions within an hour.
- `power_user_burst` (frequent_digital/business_like only): high-velocity legitimate runs.
Each type must overlap with at least one fraud signal (amount, new device, location jump, velocity, odd hour).

### 2D — Fraud episodes (`fraud.py`, `drift.py`)
Fraud is generated as **episodes** on victim customers, each with a latent `stealth ∈ [0,1]` drawn from a Beta distribution (period A and B parameters in config; interpolated by `drift.strength`). Every signal's deviation is scaled by `(1 − stealth)`, plus noise, so fraud ranges from blatant to nearly indistinguishable. Roughly: amount multiplier = `exp((1 − stealth)·N(μ,σ))`, device-new probability = `(1 − stealth)·0.9 + 0.1`, and so on. Also vary the *combination* of signals per episode.

| Scenario | Shape (all parameters randomized, overlapping with legit ranges) |
|---|---|
| burst | 4–25 tx within 5–45 min; amounts near customer typical ×(1 + (1−stealth)·U(0,3)); 1–3 categories; multiple recipients for transfers; device known w.p. 0.5+0.4·stealth |
| account_takeover | 2–10 tx over 1–6 h; new device w.p. (1−stealth)·0.9+0.1 (fraud devices reused across victims w.p. `fraud_device_reuse_prob`); location shift far from home w.p. (1−stealth); odd hours w.p. (1−stealth); unusual categories (electronics, jewelry, transfers) w.p. 0.7; amount ratio ≥ 1 |
| amount_anomaly | 1–2 tx with amount multiplier log-mean 1.2 + 1.8·(1−stealth); otherwise normal context (known device, home location, normal hours) |
| geo_anomaly | 1–4 tx placed right after a legit tx (< 2 h) at a far location (implausible travel speed); device may be known; normal-ish amount |
| structuring | cumulative ≈ U(1.5, 6)× `reporting_threshold`, split into 4–14 tx each = threshold × U(0.55, 0.97), spaced 5 min–6 h over ≤ 2 days; known device; mostly P2P_TRANSFER / CASH_WITHDRAWAL. Single-transaction features must be weak; the signal lives in window sums/counts |

- Victims sampled with mild weighting (e.g. business_like ×1.5); episode start time uniform over the allowed period; the scenario mix depends on whether the start is before/after the drift date (interpolated mix: `(1−s)·A + s·B`). If `drift.start_fraction` is null there is no drift.
- Episode counts: sample episodes sequentially from the mix until fraud *transaction* count reaches `target_fraud_rate × total_tx` (±5%). Report both episode share and transaction share per scenario.
- Optional `scenario_mix_override` (used by `unseen_lab`).
- Fraud transactions respect basic physics: no two transactions of one customer at identical timestamps; timestamps within the simulation window.

### 2E — Labels with delay (`labels.py`)
`label_available_at = timestamp + delay` where fraud delay ~ U(label_delay_days) and legit delay = 0 (assumed legit until reported). Record `episode_id` for fraud and hard negatives, and `stealth` for fraud.

### 2F — Orchestration (`generate.py`, `scripts/generate_data.py`)
`python scripts/generate_data.py --profile dev [--seed N]` runs: entities → normal → hard negatives → fraud → merge → sort → assign IDs → write parquet + report. Deterministic given seed. Print a one-screen summary.

### 2G — Simulation report (`simulation_report.json`)
n_customers, n_transactions, date range, fraud rate (actual), fraud count, fraud share by scenario (episodes and transactions) overall and per period (A/B), hard-negative share by type, tx per archetype, drift settings, label-delay settings, seed, config hash, git commit.

## Tests (`tests/test_simulator.py`, use the `tiny` profile)
- Determinism: same seed ⇒ identical frames; different seed ⇒ different.
- Schema/dtypes; unique IDs; amounts > 0; timestamps inside window and sorted; `transactions` has no label columns; IDs assigned after sorting.
- Referential integrity (customer/device/merchant/location/account FKs).
- Fraud rate within ±10% (relative) of target; every scenario appears; hard negatives of each type exist.
- Drift: with strength 1.0, the scenario mix and mean stealth after the drift date differ from before (statistical check).
- Label delay: `label_available_at ≥ timestamp`; fraud delays inside configured range.
- Structuring episodes: sum of amounts in the episode ≥ 1.5× threshold; individual amounts < threshold.
- `merchant_risk_group` is independent of fraud (chi-square p > 0.01 on `main`-like sample or a weak effect-size bound).

## Acceptance criteria
```bash
pytest tests/test_simulator.py -q
python scripts/generate_data.py --profile tiny
python scripts/generate_data.py --profile dev
cat data/raw/dev/simulation_report.json
```
Report in the progress file: actual rows, fraud rate, fraud count per scenario, hard-negative share, generation time. Part 03 will decide whether the data is hard enough; do not tune to make models look good.

## Do not
Use fraud rules like `IF new_device AND amount > X THEN fraud`. Encode fraud in customer attributes, merchant risk groups or ID patterns. Generate labels from features after the fact.
