# Part 03 — Data Validation, EDA, and the Separability Gate

## What was built

### 3A — Validation (`data_quality/validation.py`, `scripts/validate_data.py`)
- `validate_dataset(tables, *, window_start, window_end, max_amount=1e7, max_fraud_rate=0.05) -> ValidationReport`
  with `CheckResult` / `ValidationReport` (`passed`, `failed`, `to_dict`, `summary`) and
  **23 named checks**: unique IDs per table; no nulls in required columns; `amount > 0` and
  `≤ 1e7`; timestamps parsable and inside the window; valid category/type/channel enums; orphan
  foreign keys; `labels` covers every transaction exactly once; the `FORBIDDEN_COLUMNS` are absent
  from `transactions`; `label_available_at ≥ timestamp`; fraud-rate sanity (`0 < rate < 5%`).
- `load_raw_tables(profile)` loads the parquet tables from `data/raw/<profile>/`.
- `scripts/validate_data.py` (thin CLI): writes `reports/data_validation_<profile>.json`, exit ≠ 0
  on failure.
- `tests/test_validation.py`: 22 tests. A `_clean()` builder makes a valid dataset, then each test
  injects exactly one defect and asserts the matching check flips to fail.

### 3B — EDA (`data_quality/eda.py`, `scripts/eda.py`)
- 5 figures in `reports/figures/eda_dev_*.png` (all numbers computed by code, nothing hard-coded):
  `class_balance` (weekly fraud rate + tx/day + volume by archetype), `amounts` (legit vs fraud log
  scale, by archetype), `hours` (legit vs fraud vs hard-negative hour-of-day), `drift` (scenario
  shares + stealth distribution, period A vs B), `overlap` (fraud vs hard-negative vs legit on the
  crude signals).
- Generated docs: `docs/eda_summary.md` (per-scenario signal-profile table incl. an approximate
  `tx_in_1h` for EDA only) and `docs/data_dictionary.md` (raw-table stub; the feature dictionary
  comes in Part 05).
- **New-device indicator in EDA** = static `customer_devices` lookup only (no first-seen variant),
  matching the gate and Part 02's definition.

### 3C — Separability gate (`data_quality/signals.py`, `data_quality/gate.py`, `scripts/separability_gate.py`)
- `compute_crude_signals(...)`: five row + static-attribute signals, all oriented higher = more
  suspicious: `amount_ratio`, `hour_dev` (circular), `dist_home_km` (haversine to static home),
  `new_device_static` (APP only; NULL device ⇒ **not** new), `cat_not_preferred`. NaN → median.
  These are used **only** here and must never become Part 05 features.
- `run_separability_gate(...)`: single-signal ROC/PR-AUC over the whole set; depth-1 stump, depth-3
  tree and an XGBoost reference on a temporal split; hard-negative overlap; top-3 boosted gain
  importances with a `dominant_feature` flag when a single gain share > 0.60 (context for Parts
  05/08 only, not acted on here).
- Writes `results/separability_gate_<profile>.json` **and** `docs/separability_gate.md`; CLI exits
  ≠ 0 on FAIL.
- `tests/test_separability_gate.py`: 2 tests on synthetic data — a perfectly separable single signal
  must make the gate FAIL (guards the core purpose + condition plumbing + markdown rendering).

## Acceptance commands — actual outputs

| Command | Result |
|---|---|
| `pytest tests/test_validation.py -q` | `22 passed` |
| `pytest tests/test_separability_gate.py -q` | `2 passed` |
| `make test-fast` | `57 passed in 5.61s` (15 config + 18 simulator + 22 validation + 2 gate) |
| `make lint` | `All checks passed!` |
| `python scripts/validate_data.py --profile dev` | `PASS: all 23 checks passed`, exit 0 |
| `python scripts/eda.py --profile dev` | 5 figures + 2 docs written; exit 0 |
| `python scripts/separability_gate.py --profile dev` | **PASS**, exit 0 |

### Gate metrics (`results/separability_gate_dev.json`, profile `dev`, 289 870 tx / 2 207 fraud / 0.7614 %)

| condition | observed | threshold | result |
|---|---|---|---|
| best single-signal ROC-AUC | 0.8105 | < 0.85 | PASS |
| best single-signal PR-AUC | 0.1831 | < 0.20 | PASS |
| depth-3 tree vs boosted PR-AUC gap | 0.0648 | ≥ 0.05 | PASS |
| hard-negative overlap (signals) | 2 | ≥ 2 | PASS |

- Best single signal = `new_device_static` (ROC 0.782, PR 0.183); next `hour_dev` (ROC 0.811 but
  PR only 0.086). No single crude signal dominates.
- PR-AUC: stump(d=1) 0.1752 → tree(d=3) 0.5769 → boosted 0.6417 (interactions genuinely matter).
- Hard negatives overlap on `amount_ratio` and `dist_home_km` (2/5 ≥ threshold).
- `dominant_feature: null` — top boosted gain shares: `new_device_static` 0.558,
  `cat_not_preferred` 0.376, `hour_dev` 0.024 (none > 0.60).

## Simulator changes made because the gate initially failed (3D)

The gate failed on first run. Per 3D, the simulator was adjusted (never the models):

1. **Amount multipliers now fall to baseline at high stealth.** `logmean = (1 − stealth)·U(...)` in
   `fraud.py` for `burst`, `account_takeover` and `amount_anomaly`, so a stealthy fraud row no longer
   carries an obvious amount spike. (`amount_ratio` was the strongest shortcut.)
2. **Stealth raised and made drift-aware.** `stealth_beta_period_a: [2.0, 3.0]` (mean ≈ 0.40),
   `stealth_beta_period_b: [3.0, 2.0]` (mean ≈ 0.60) — period B is subtler.
3. **`fraud_device_reuse_prob: 0.35`** — fraud often reuses an already-known device, weakening the
   `new_device_static` shortcut while keeping realistic device overlap.

After regeneration (`tiny` + `dev`) the gate passes. **Margins are thin** and are recorded as the
main open issue.

## Figures produced

`reports/figures/eda_dev_class_balance.png`, `eda_dev_amounts.png`, `eda_dev_hours.png`,
`eda_dev_drift.png`, `eda_dev_overlap.png`.

## Deviations from the prompt (and why)

1. **`label_available_at` semantics.** The check is `label_available_at ≥ timestamp` (as literally
   specified). Fraud labels have a delay; legit labels are available immediately, so the inequality
   holds for both.
2. **Crude-signal definition of "new device".** NULL `device_id` (WEB/POS/ATM) is treated as *not a
   device event*, hence not new — consistent with Part 02's documented channel/device design. The EDA
   overlap plot uses the same static `customer_devices` lookup (the prompt's parenthetical mentions
   "device first-seen"; a first-seen variant would require ordering and is redundant with the static
   view, so the simplest correct option was chosen — stated here per the ambiguity rule).
3. **Gate condition 3 (hard-negative overlap)** is implemented as the literal pooled reading:
   `mean(hard-negative rows)` vs `median(fraud rows)` per signal, requiring ≥ 2 signals. Per-type
   overlap is additionally written to the report for transparency.
4. **Gate importances** (top-3 gain + `dominant_feature`) are logged as context only; nothing is
   tuned on them.
5. **`tx_in_1h`** in the EDA per-scenario table is a quick groupby, explicitly EDA-only; the rigorous
   window features are Part 05's job.
6. **The gate is a data check, not a model experiment** — it does **not** append to
   `results/experiments.csv`.

## Open issues / notes

- **Thin gate margins** (ROC 0.8105 vs 0.85; PR 0.1831 vs 0.20; gap 0.0648 vs 0.05; overlap exactly 2/2).
  The data passes but with little slack. Thresholds are frozen (must not be lowered); if a future
  regeneration regresses, the simulator is the knob to turn.
- EDA `overlap` figure: the legit bar for `new_device_static` is ~0.01 (near-invisible) — expected,
  not a plotting error.
- Part 02 bugs surfaced by this validation and fixed: `income_band` garbage floats and the
  `customer_devices` export leak (details in `docs/what_went_wrong.md`).
