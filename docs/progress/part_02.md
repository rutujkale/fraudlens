# Part 02 — Behavioral Transaction & Fraud Simulator

## What was built

Config-driven, seeded simulator under `src/fraudlens/data_gen/`:

- **`archetypes.py`** — 12 categories, 4 transaction types, 4 channels, income bands; per-archetype parameter ranges (conservative/regular/high_spend/frequent_digital/business_like) matching the spec (daily rate, median amount, devices, hours, category count, contacts, travel, P2P rate). `sample_archetype_params` draws every parameter per customer with noise (Dirichlet category weights, narrow/gamma hours, 1–3 devices).
- **`entities.py`** — 41 Indian cities with real lat/lon + `region_group`; 400–2000 merchants scaled with customers (risk group random); customers (age, income band weakly tied to archetype, home location, opening date — ~10% cold-start mid-window); accounts (1/customer, extra CURRENT for ~20% of business_like); devices + `customer_devices`; ~4% of customers share one family device; P2P contacts. `DeviceRegistry` mutates as episodes add devices. `cat_weights`/`contacts`/`near_locations`/`home_location` lookups exposed via an `Entities` dataclass.
- **`normal.py`** — per customer-day Poisson count = rate × day-of-week × gamma day-effect × payday/festival multipliers; vectorized time/amount spans; per-tx channel/device/location/merchant(loyalty)/recipient. `RowBuffer` accumulates transaction + parallel label columns (the single shared accumulator).
- **`hard_negatives.py`** — the spec's **7 types** (`new_phone`, `travel_trip`, `holiday_splurge`, `medical_bill`, `emergency_odd_hour`, `payday_burst`, `power_user_burst` restricted to frequent_digital/business_like), together ≈ `hard_negative_target_share` of legit tx, each overlapping ≥1 fraud signal.
- **`fraud.py`** — 5 scenarios (burst, account_takeover, amount_anomaly, geo_anomaly, structuring). Signals scaled by `(1 − stealth)`; device-new prob `(1−s)·0.9+0.1`; fraud-device reuse pool; victims weighted business_like ×1.5. Episodes drawn until fraud tx count ≈ `target_fraud_rate` (±5%). No conditional fraud rules.
- **`drift.py`** — step switch at `start_fraction`; scenario mix and stealth Beta interpolate from period A toward B by `drift.strength` (1.0 = full switch). No-drift profiles stay in A.
- **`labels.py`** — fraud delay ~ U(label_delay_days), legit delay 0; `episode_id` for fraud + hard negatives, `stealth` for fraud.
- **`generate.py`** — orchestration: entities → normal → hard negatives → fraud → merge → sort by `(timestamp)` → strictly-increasing per-customer fixup → `T%07d` IDs **after** sort → labels → `FORBIDDEN_COLUMNS` guard → parquet + `simulation_report.json`. `write_outputs` + `main` + `_build_report`.
- **`scripts/generate_data.py`** — thin CLI (`--profile`, `--seed`). `make simulate PROFILE=...` target added.

## Acceptance commands — actual outputs

| Command | Result |
|---|---|
| `pytest tests/test_simulator.py -q` | `18 passed` |
| `make test-fast` | `33 passed in 4.25s` (15 config + 18 simulator) |
| `make lint` | `All checks passed!` |
| `python scripts/generate_data.py --profile tiny` | 4715 tx, 91 fraud (0.0193), 14 episodes |
| `python scripts/generate_data.py --profile dev` | 289 879 tx, 2216 fraud (0.007645), 331 episodes; **~15 s** (< 2 min target) |
| `cat data/raw/dev/simulation_report.json` | written (see below) |

### dev report highlights
- fraud rate 0.007645 vs target 0.008 (−4.4%).
- scenario shares (tx): burst 0.474, account_takeover 0.234, structuring 0.153, amount_anomaly 0.071, geo_anomaly 0.067.
- drift (start 2026-04-16, strength 0.5): period-A structuring share 29/296 = 9.8 % vs period-B 10/35 = 28.6 % (period B = last 12% of window); burst concentrated in A.
- hard-negative tx share of legit: power_user_burst 1.12 %, new_phone 0.95 %, travel_trip 0.58 %, payday_burst 0.42 %, holiday_splurge 0.36 %, emergency_odd_hour 0.14 %, medical_bill 0.085 % (sum ≈ 3.7 %).
- `config_hash` 64-char sha256; `git_commit` recorded.

## Deviations from the prompt (and why)

1. **Drift is a step, scaled by `strength`.** Config has `start_fraction` + `strength` (no ramp). Before the drift date: period A. From it on: mix/Beta = `A + strength·(B − A)` (dev/main 0.5 → partial, drift_lab 1.0 → full). Matches the config comments.
2. **Hard-negative types** follow the spec's exact seven names/behaviours; an earlier draft used invented types and was replaced.
3. **Structuring ratio clamps.** The literal `low = max(1.5, 0.55n+0.05)`, `high = min(6, 0.97n)` is inconsistent for n≥11 (`low > high`) and can force a negative residual. Fixed with `low = min(..., high)` and `last = clip(residual, 0.5, 0.98)`; the tested guarantees (sum ≥ 1.5×threshold, every amount < threshold) still hold.
4. **Channel/device design:** APP transactions carry a known `device_id`; WEB/POS/ATM have `device_id = NULL` (documented choice — Part 05 must treat NULL as "not a device event", not as "new device"). **APP-scoped** `is_new_device`.
5. **Festival calendar** is an approximate fixed (month, day) list for 2026 (very rough) — sufficient as a spending-shift modifier only.
6. **`scenario_mix_override`** pins both periods, so `unseen_lab` gets a fixed mix and no drift.
7. `B905` added to the ruff ignore list (intentional equal-length `zip()`s); `pyproject.toml` line-length 100 retained.

## Open issues / notes

- `dev` transaction volume is ~290k (≈0.48 tx/customer/day average), dominated by frequent_digital/business_like — expected from the archetype rates.
- `new_phone` adds a device via the registry, which later fraud episodes may then treat as a "known" device; this is intended realistic overlap, not a bug.
- Part 03 will judge whether the data is hard enough; no tuning to flatter models was done.
