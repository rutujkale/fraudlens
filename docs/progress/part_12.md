# Part 12 progress — Freeze, locked test evaluation, headline numbers

**Status:** 12A (freeze) complete and committed; 12B/12C **not run** — awaiting review before the
one-shot test evaluation. · **Committed:** yes (freeze) · **Pushed:** no

## Deviation from the prompt

- Everything runs on the **`dev`** profile, not `main`. `main` (20k customers / 270 days) has
  **never been generated** (Part 02b decision). This must also be recorded in `part_15.md`.
- No README/resume text may quote `main` numbers.

## Freeze (12A)

- Tooling: `src/fraudlens/evaluation/freeze.py` + thin `scripts/freeze.py` + `tests/test_freeze.py`
  (7 tests, `ruff` clean). **Commit A** `0953884` (`freeze: Part 12A tooling`);
  `results/freeze.json` **Commit B** `139dab8`.
- `freeze.json` records `git_commit 095388436943d70e8cabd2106062d4b76784283b` = the code commit whose
  `freeze.py` produced it (the commit *before* `freeze.json` was added); `tree_clean: true`.
- **No retrain:** `retrain: false`. `xgb_dev_20261015_v1` / `logreg_dev_20261015_v1` are used as-is
  (a retrain would fork a new timestamped artifact).
- **Systems frozen:** `["rules", "logreg", "xgb"]`. Anomaly is **not** evaluated: the Part 09
  isolation forest was re-run post cold-start fix and the decision is unchanged (**not adopted**;
  S1−S0 and S1s−S0 bagged CIs = 0.0000), and no IF artifact exists.
- Frozen ids match `results/models_dev.json`; both artifact `metadata.json` `profile == "dev"`
  (asserted — guards the earlier cross-profile clobbering).
- sha256 of **every** artifact file (model, calibrator, `calibration.json`, `metadata.json`,
  `policy.json` for xgb, `val_scores.parquet`) **and** the three data files
  (`features_v1.parquet`, `transactions.parquet`, `labels.parquet`).
- Split boundaries: `t0 2026-01-01 00:17:00`, `train_end 2026-03-13 23:59:00`,
  `val_end 2026-04-06 23:53:00`, `end 2026-04-30 23:47:00`, `embargo 7d`; sizes
  **train 163,323 / val 40,585 / test 39,823** (recomputed and cross-checked against
  `results/split_dev.json`).
- Cost config (balanced) frozen; **capacity 23/day**, `avg_daily_volume 2254.78` (from `policy.json`);
  frozen thresholds `f1 0.40072`, `cost_optimal 0.07566`, `cost_optimal_capacity 0.18823`,
  calibrator `sigmoid`.
- **Fitted rule thresholds recorded** (velocity_10m 5.0, amount_ratio_30d 8.1465, new_device_amount
  2.0, travel_speed 800.0, night_new_category null, near_threshold 3.0). The final evaluation aborts
  if they differ at run time.
- **config-hash delta, proven.** The runtime `config_hash 4279ce47…` differs from each artifact's
  `config_hash_at_training 7b6d94bf…`. `reconstructed_training_config_hash()` removes exactly the
  keys added *after* training — the Part 11 `external:` block (`dc9c8f6`) **and** the two Part 02b
  `models.xgb.suspicious_pr_auc_*` leakage-guard flags (`230ab14`) — and reproduces `7b6d94bf…`
  exactly. So the dev-relevant sections (simulation incl. `fraud_device_reuse_prob 0.50`, splits,
  costs, model hyperparameters, rules, features, scenario mixes, seed) are byte-identical to the
  training-time config; only external-validation settings and a run-time guard flag postdate the
  artifacts. Both hashes are recorded. Proof:
  `python -c "from fraudlens.config import load_config; from fraudlens.evaluation.locked import reconstructed_training_config_hash as r; print(r(load_config('dev')))"`
  → `7b6d94bf3274232b9370448f301b51d82d4a6d24511f14e5cbea71f9cec3db0a` (test
  `test_reconstructed_hash_matches_frozen_training_hash` asserts this against `freeze.json`).

### Pre-registered headline success (H1–H4) — written before the test window opens

- **H1 (model):** XGB PR-AUC CI lower bound exceeds the Rules CI upper bound. LogReg ordering is
  reported either way. **Power note (pre-registered):** the dev test window has only **393 frauds**,
  so cluster-bootstrap PR-AUC CIs are wide; if H1 fails, that most likely reflects low power rather
  than poor model quality.
- **H2 (decision):** the amount-aware capacity policy's realized-cost reduction versus F1-optimal has
  a CI excluding 0.
- **H3 (unseen):** the post-fix Part 09 "not adopted" decision stands, with no new Part 09 claim.
- **H4 (engineering):** counts come from code, and the latency figure comes from Part 13.

Whichever way these come out, they are reported as findings.

### Test window and drift (pre-registered)

- The dev drift switch is at **2026-04-16 00:00:00** (`data_gen/drift.py::drift_start_date` =
  `start_date + int(0.88 × 120) d`, i.e. 2026-01-01 + 105 d; **not 14:14** — a span-based 14:14
  estimate is wrong because the code uses integer-day offsets). It falls **inside** the test window.
- Test window = **2026-04-13 23:54 → 2026-04-30 23:47 (17.0 days)**, so the switch is ~2.6 days in.
  Most of the window is **post-drift**: pre-switch 4,574 rows / 83 frauds; post-switch 35,249 rows /
  **310 of 393 frauds (79%)**. Structuring's share of frauds rises from 15/83 (18%) pre-switch to
  141/310 (45%) post-switch — the pre-registered concept-drift signature. Validation is entirely
  pre-drift.
- Consequence: any val→test degradation mixes **drift with the window change** and must be read that
  way, not as model failure.

## Pre-guard disclosure — test-window label composition

The dev **test-window label counts only** were read to pre-register which breakdowns fall below
`n_pos < 30` and must be suppressed. This is **composition only**: no model scores, no features, no
metrics, no evaluation. `freeze.py` itself never inspects test labels (it maps `is_fraud` only to
select *training*-legit rows for the rules thresholds).

Actual order of events:
1. Counts first read during planning (before the freeze; read-only).
2. `scripts/freeze.py --profile dev` ran on the clean tree and wrote `results/freeze.json` at
   **2026-10-16T16:23:44** (Commit A `0953884`, `tree_clean: true`).
3. Counts re-verified and the audit row written to `results/test_access_log.csv` at
   **2026-10-16T16:25:08** (`caller=part12_label_composition`, `final=False`, git `d11d34c`) —
   **after** the freeze, not before.

Read (verified): test 39,823 rows, **393 frauds, prevalence 0.987%**; scenario structuring 156,
burst 127, account_takeover 63, amount_anomaly 25, geo_anomaly 22; archetype (fraud) regular 230,
conservative 90, frequent_digital 33, high_spend 27, business_like 13; hard-negative (legit)
power_user_burst 478, travel_trip 404, new_phone 227, emergency_odd_hour 63, medical_bill 44,
payday_burst 16.

**Pre-registered suppressions (`n_pos < 30`), reported as suppressed with a reason, not dropped:**
scenarios `amount_anomaly` (25), `geo_anomaly` (22); archetypes `high_spend` (27),
`business_like` (13); hard-negative type `payday_burst` (16).

## Open items

- 12B locked evaluation not run (awaiting review). The verifier is implemented in
  `src/fraudlens/evaluation/locked.py` (unit-tested in `tests/test_final_guard.py`, 8 tests): hash
  checks accepting both path separators, clean-tree + HEAD-contains-`freeze.json` precondition, and
  a `final=False` disclosure row that does **not** count as a prior evaluation. The run will open the
  test window only via `TestSetGuard(final=True)`.
- Item 9 sim-to-real: use the main-frozen XGB **test** PR-AUC with its prevalence beside it; note the
  53 (synthetic) vs 38 (IEEE shared-subset) feature-set difference; **no cross-dataset differencing**.
- Part 15: carry the dev-not-`main` deviation forward.
