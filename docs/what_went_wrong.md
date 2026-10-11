# What went wrong

Every bug, leakage incident, simulator shortcut or failed experiment goes here.
That file is interview material: it shows how problems are found and fixed.

## Template

### Incident
_What happened, one line._

- **How detected:** _test name, guard, manual check, ..._
- **Effect:** _what wrong result it could have produced._
- **Fix:** _what was changed and where._
- **Test added:** _which test now guards against regression._

---

### Incident: prompt files missing from `prompts/`
- **How detected:** `git status` before the Part 01 commit showed only `00_START_HERE.md` in `prompts/`; the `part_*.md` build files were absent.
- **Effect:** the repo would have shipped without its build log/instructions.
- **Fix:** re-copied with `Get-ChildItem -Filter "part_*.md" | Copy-Item` (the original `Copy-Item -LiteralPath ... part_*.md` does not expand wildcards).
- **Test added:** eyeball check of `prompts/` listing before commit (build-process issue, no unit test).

---

### Incident: `date + timedelta(minutes=...)` silently dropped the time component
- **How detected:** `account_takeover` crashed with `AttributeError: 'hour' is an invalid keyword argument for replace()` while placing odd-hour transactions.
- **Effect:** all fraud/hard-negative/episode timestamps built from `simulation.start_date` (a `datetime.date`, not `Timestamp`) collapsed to midnight — every episode would have been at 00:00, destroying time-of-day and velocity signals.
- **Fix:** wrap the window start in `pd.Timestamp(...)` in `fraud.py` and `hard_negatives.py`.
- **Test added:** `test_strict_per_customer_timestamp_order`, `test_amounts_positive_and_sorted`, `test_label_delay_ranges_and_legit_immediate` (all exercise real timestamps).

---

### Incident: structuring episode parameters inconsistent for large `n`
- **How detected:** `tiny` generation crashed with `ValueError: high - low < 0` (and would have produced negative amounts).
- **Effect:** the literal `low = max(1.5, 0.55n+0.05)`, `high = min(6, 0.97n)` makes `low > high` for n ≥ 11; per-tx clipping could force `residual < 0`.
- **Fix:** `low = min(max(1.5, 0.55n+0.05), high)` and `last = clip(residual, 0.5, 0.98)` in `fraud.py`.
- **Test added:** `test_structuring_below_threshold_and_aggregate_above`.

---

### Incident: device registry used a `set` (non-deterministic order)
- **How detected:** code review — a `set`'s iteration order varies with `PYTHONHASHSEED`, so `rng.choice(known_devices)` could pick different devices across runs.
- **Effect:** same-seed runs could produce different frames — reproducibility would silently break.
- **Fix:** `DeviceRegistry.customer_devices` stores insertion-ordered lists with dedupe.
- **Test added:** `test_determinism`.

---

### Incident: `first_account` returned the second account for business_like customers
- **How detected:** code review of the account loop (dict comprehension last-wins).
- **Effect:** `account_id` on transactions would not be the customers' primary account.
- **Fix:** `setdefault` so the first (primary) account wins.
- **Test added:** `test_referential_integrity`.

---

### Incident: payday modifier was a no-op
- **How detected:** code review — `_payday_mask` returned `1.0` in both branches.
- **Effect:** the specified payday spending bump never occurred.
- **Fix:** return `1.15` on the 1st of the month.
- **Test added:** covered indirectly by distribution/report tests (no isolated test).

---

### Incident: `income_band` sampled weights instead of band names
- **How detected:** Part 03 validation — `income_band` enum check failed; the column held floats.
- **Effect:** `entities.create_entities` did `rng.choice(...)` on the *probability weights*
  (`INCOME_BAND_WEIGHTS_BY_ARCHETYPE[a]`) rather than on `INCOME_BANDS`, so every customer's
  `income_band` was a garbage probability value. Any downstream segmentation by income band, and the
  validation enum check, would be wrong.
- **Fix:** `rng.choice(INCOME_BANDS, p=INCOME_BAND_WEIGHTS_BY_ARCHETYPE[a])` in `entities.py`.
- **Test added:** `tests/test_validation.py::test_invalid_income_band_fails` (enum check) plus the
  simulator's enum/coverage tests; the validation suite now fails on bad income bands.

---

### Incident: exported `customer_devices` leaked fraud-added devices (target-derived leak)
- **How detected:** Part 03 code review while building the gate's `new_device_static` signal — the
  `customer_devices` table contained device edges registered *during* episode generation, i.e. after
  account opening, including devices added by fraud (`new_phone` HN, account-takeover, etc.).
- **Effect:** using the exported table as "devices known at account opening" would leak future/fraud
  information into the `is_new_device` feature — a point-in-time violation.
- **Fix:** freeze device-edge export at `n_known_edges = len(devreg.customer_device_rows)` immediately
  after `make_device_registry` in `generate.py`; only the account-opening snapshot is written.
- **Test added:** `tests/test_validation.py` referential/logic checks plus
  `tests/test_separability_gate.py` (static-device signal exercised); the frozen-edge behaviour is
  asserted by the simulator's registry tests.

---

### Incident: simulator shortcut — crude signals solved the data (separability gate FAIL)
- **How detected:** `python scripts/separability_gate.py --profile dev` FAILed on first run: a single
  crude signal had too high discrimination, i.e. fraud was trivially separable by one cheap rule.
- **Effect:** a model would have scored well for the wrong reason (one cheap threshold), which is
  exactly what the gate exists to catch.
- **Fix:** in `fraud.py`, amount multipliers now decay with stealth (`logmean = (1 − stealth)·U(...)`)
  so high-stealth fraud keeps a baseline-like amount; stealth raised to `period_a [2.0,3.0]` /
  `period_b [3.0,2.0]` (means ≈0.40/0.60); `fraud_device_reuse_prob: 0.35` so fraud often reuses a
  known device. Regenerated `tiny` + `dev`; gate now PASSes. Thresholds were **not** lowered.
- **Before → after (gate conditions, `dev`):**

  | condition | threshold | before (failed run) | after (passing run) |
  |---|---|---|---|
  | best single-signal ROC-AUC | < 0.85 | **FAIL** (`amount_ratio` too high) | 0.8105 PASS |
  | best single-signal PR-AUC | < 0.20 | **FAIL** | 0.1831 PASS |
  | depth-3 vs boosted PR-AUC gap | ≥ 0.05 | — | 0.0648 PASS |
  | hard-negative overlap signals | ≥ 2 | — | 2/2 PASS |

  **Honesty note:** the *exact numeric* metrics of the original failing run were **not persisted** —
  the CLI wrote the failing report to `results/separability_gate_dev.json` and the next (passing) run
  overwrote it. Only the after numbers are recorded verbatim in that file. The before column above
  records the conditions that failed and the culprit signal, not invented numbers. Process fix:
  the CLI now also writes a preserved `results/separability_gate_<profile>.FAILED.json` on failure
  (written once, never clobbered by a pass).
- **Test added:** `tests/test_separability_gate.py` (perfect single-signal separator must FAIL the
  gate), plus the CLI run recorded in `docs/separability_gate.md`.

---

### Incident: `RANGE ... EXCLUDE CURRENT ROW` does not exclude same-timestamp peers
- **How detected:** code review while writing `sql/views.sql` (`curated.rolling_24h_sql_demo`), per
  the Part 04 prompt.
- **Effect:** in a window frame `RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW`, rows
  sharing the current row's `timestamp` are *peers*; `EXCLUDE CURRENT ROW` removes only the current
  physical row, not its peers. If two of a customer's transactions share a timestamp, the demo view
  can count a peer as "prior", over-counting vs a strict `timestamp < before_ts` definition.
- **Fix / status:** the view is a **cross-check only** and is documented as such in a SQL comment.
  Part 05's feature logic stays the source of truth (strictly `timestamp < T`). **Verified on `dev`:**
  `SELECT ... GROUP BY customer_id, timestamp HAVING count(*) > 1` returns **0 peer groups** (Part 02
  guarantees strictly-increasing per-customer timestamps), so `EXCLUDE CURRENT ROW` already matches the
  strict-prior definition on this data. The caveat stays for safety; Part 05 should re-confirm and use
  `EXCLUDE TIES` / a `<` self-join if peers ever appear.
- **Test added:** `tests/test_db.py` covers the demo view + prior-only semantics; the full window
  leakage suite arrives in Part 05.

---

### Incident: same-timestamp peers leaked into recency + running-state features
- **How detected:** `tests/test_leakage.py::test_9_training_serving_parity` — for a row whose timestamp
  tied with a peer, the offline build returned `minutes_since_prev_tx = 0.0` while serving (history
  filtered to `timestamp < T`) returned `39.0`.
- **Effect:** the per-customer loop used the positional predecessor (`cts[i-1]`) for
  `minutes_since_prev_tx`, `days_since_last_tx`, `distance_from_prev_km`, `travel_speed_kmh`, and
  `n_prior_tx`/`is_cold_start`/`hour_deviation`, and updated the running seen-sets and hour histogram
  after *every* row. A same-timestamp peer therefore became visible to the row beside it — a strict
  point-in-time violation that serving (which filters `< T`) would not reproduce.
- **Fix:** `features/build_features.py` now processes each customer in equal-timestamp groups:
  `prior_end` (== `#rows with ts < t`) drives recency/lifetime/hour features and the previous-row
  pointer, and all running state (seen merchants/categories/recipients/devices/regions, device counts,
  hour histogram) is updated only *after* the whole group. Window counts/sums were already strict.
- **Test added:** `test_9_training_serving_parity` (the detector) plus
  `test_4_same_timestamp_ties`. Part 02 also guarantees strictly-increasing per-customer timestamps, so
  this is defence-in-depth; no real profile currently contains per-customer ties.

---

### Incident: leakage "incident" demo — metric-blindness of subtle leaks (Part 05E)
- **How detected:** deliberately leaky `window_bounds` variants (the same ones the canary test uses)
  were monkeypatched in a scratch script (`%TEMP%/leakage_incident.py`, no repo edit), features rebuilt
  on `dev`, and a 200-tree XGBoost trained with the `dev` temporal split (train 60% / val 60–80%).
- **Effect / measured (`results/leakage_incident_dev.json`, 289,870 tx, 470 val positives):**

  | variant | val PR-AUC | val ROC-AUC | caught by |
  |---|---|---|---|
  | clean (correct `[T-w, T)`) | 0.9560 | 0.9987 | passes all tests |
  | `[T-w, T]` inclusive | 0.9567 | 0.9989 | tests 3 & 4 |
  | `[T-w, T+w]` future | **0.9927** | **0.9999** | tests 1 & 2 |

  The future-window leak inflates val PR-AUC by ~0.037 and ROC-AUC to 0.9999 — the "suspiciously
  better" signature. The inclusive-window leak, by contrast, barely moves the metric (+0.0007): a
  metric-only review would have missed it. The invariance tests catch **both** deterministically.
- **Fix:** none needed in the shipped code — the leak existed only in the scratch monkeypatch; the
  canary test (`test_8`) keeps both detectors honest. Correcting the prompt's assumption: an
  *inclusive* window does **not** fail test 1 (its current-row/same-timestamp leak is invisible to a
  truncation test because the peer/current row exists on both sides of the cut); it fails tests 3 & 4.
  Test 1 fails only for a future/lifetime leak.
- **Open concern (not a leak):** the *clean* val PR-AUC of 0.956 is very high for a 0.8% positive rate.
  Because clean features pass truncation, future-mutation and serving-parity invariance, this is not
  temporal leakage — it is synthetic separability (the simulator's fraud is genuinely learnable). It is
  logged for Part 06, where model optimism vs the Part 03 gate should be re-examined.
- **Test added:** `tests/test_leakage.py::test_8_canary_tests_detect_leakage`.

---

### Incident: Part 06 high XGBoost val PR-AUC (0.9667) with no dominant feature
- **How detected:** the Part 06 stop-guard in `scripts/train.py` — XGBoost best validation PR-AUC
  `0.9667 > 0.95` triggers an automatic diagnostic (`results/xgb_importance_dev.json`,
  `results/xgb_importance_dev.csv`). This closes the "Open concern (not a leak)" left by the Part 05E
  entry above (clean val PR-AUC 0.956).
- **Measurements (`dev`, 289,870 tx; train 174,834 / val 41,313 / test 39,380 untouched):**
  - rules val PR-AUC **0.3787**, logreg **0.9021**, xgb **0.9667** (ordering as expected).
  - top-5 XGBoost gain shares are **diffuse** — no single feature dominates:

    | rank | feature | gain share |
    |---|---|---|
    | 1 | `minutes_since_prev_tx` | 0.126 |
    | 2 | `days_since_last_tx` | 0.093 |
    | 3 | `tx_count_10m` | 0.091 |
    | 4 | `tx_count_1h` | 0.064 |
    | 5 | `is_night` | 0.060 |

  - `dominant_feature = null` (no feature exceeds the 0.60 dominance share).
  - `is_new_device` ablation (NaN the feature, re-score, no retrain): **0.9667 → 0.9667** (drop −5.5e−06).
- **Interpretation:** this is **not** a single leaker. The high PR-AUC is spread across many velocity
  / recency features, and the one feature most likely to be a target-derived shortcut
  (`is_new_device`) contributes nothing. This is consistent with the Part 03 separability gate passing
  (best single-signal PR-AUC 0.1831) and with genuinely learnable synthetic fraud, not temporal leakage
  (Part 05's truncation / future-mutation / serving-parity invariance tests all pass on these features).
- **Effect:** a headline "0.97 PR-AUC" could be mistaken for leakage; the diagnostic shows it is
  synthetic separability. Model optimism vs the gate is now documented rather than hidden.
- **Fix:** none to the simulator — per the Part 06 directive, no simulator change without approval, and
  no dominant feature justifies one.
- **Status:** logged and reported; **awaiting go-ahead** on whether 0.9667 is accepted as genuine or
  warrants a simulator-side investigation.
- **Test added:** none yet — the stop-guard is a CLI behaviour. A regression test pinning the
  no-dominant-feature invariant (gain share < 0.60) is a candidate for Part 08's experiment suite.

---

### Incident: `customer_holdout` did not deduplicate transaction-level customer ids
- **How detected:** `scripts/evaluate_val.py --profile dev` printed a nonsensical holdout split —
  `A_seen_customers n=164` vs `B_unseen_customers n=41149` (seen PR-AUC 1.0000 on 4 frauds). A 20%
  customer holdout should leave ~80% of the validation rows *seen*, not 0.4%.
- **Root cause:** `customer_holdout` was handed `df["customer_id"]`, an array with **one element per
  transaction** (57,974 rows, 5,000 unique customers). It did `np.array(sorted(ids))` and
  `seen = set(ids) - set(heldout)` without deduplicating, and `rng.choice(ids, size=0.2·len(ids))`
  sampled from the duplicated bag: `heldout` contained ~11,600 duplicates covering ~99.6% of unique
  customers, collapsing `seen` to 124 unique customers.
- **Effect:** the "unseen customers" experiment (Setting A vs B) measured the wrong partition —
  virtually all validation rows landed in the held-out group, hiding the true seen/unseen
  generalisation gap. Worse, the same helper feeds the Part 09 unseen-fraud design.
- **Fix:** `np.unique(ids)` before sampling in `splits.py::customer_holdout`, and
  `seen = ids[~np.isin(ids, heldout)]`. Fraction now applies to unique customers.
- **Result after fix (`dev`):** holdout seen PR-AUC **0.9639** (n=33,082, 297 frauds) vs unseen
  PR-AUC **0.9349** (n=8,231, 45 frauds) — a modest, believable generalisation gap.
- **Test added:** `tests/test_splits.py::test_customer_holdout_dedupes_repeated_transaction_ids`
  (1000 transaction ids over 50 customers → exactly 10 heldout / 40 seen).
