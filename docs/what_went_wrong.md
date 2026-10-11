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
