# Part 07 — Probability Calibration, Cost Model, and Decision Policy

## Goal
Turn model scores into **trustworthy probabilities**, then into **actions** by minimizing business cost under analyst capacity. Compare an F1-optimal threshold against cost-optimal policies, and show how sensitive the result is to cost assumptions. All selection uses validation data only.

## Prerequisites
Part 06 complete; XGBoost (and LogReg) artifacts with raw validation scores exist.

## Tasks

### 7A — Calibration (`models/calibration.py`)
- Do not use `CalibratedClassifierCV(cv="prefit")` (deprecated). Operate on scores directly:
  - `SigmoidCalibrator`: logistic regression on `logit(clip(raw))` (Platt).
  - `IsotonicCalibrator`: `IsotonicRegression(out_of_bounds="clip")`.
  - Common interface `fit(raw, y)`, `predict(raw)`, serializable via joblib.
- **Data split inside validation (chronological):** first ~60% of the validation window = `val_cal` (fit calibrators), last ~40% = `val_thr` (select thresholds/policy and evaluate calibration). Never fit a calibrator and select thresholds on the same rows. Report positives available in each slice; if `val_cal` has fewer than ~150 positives, flag that isotonic is unstable and prefer sigmoid (report both with CIs regardless).
- Evaluate on `val_thr`: reliability diagram (equal-frequency bins), Brier score, ECE, log-loss, before vs after, for each calibrator, for XGBoost and LogReg. Also report mean predicted probability vs observed fraud rate. Save `results/calibration_<profile>.csv`, `reports/figures/calibration_*.png`. Choose the calibrator by Brier/log-loss on `val_thr` with a bootstrap CI; ties → the simpler (sigmoid).
- Because XGBoost may use `scale_pos_weight`, show explicitly how class weighting distorts raw probabilities and that calibration repairs it.

### 7B — Cost model (`evaluation/cost_model.py`)
Implement the framework in `CLAUDE.md` ("Decision framework") with a `CostConfig` dataclass loaded from config (+ scenario overrides).
- `expected_cost(action, p, amount, cfg)` for APPROVE / STEP_UP / REVIEW / BLOCK (vectorized).
- `realized_cost(action, is_fraud, amount, cfg)`: the cost that actually occurs given the true label (fraud missed ⇒ α·A + fee; legit reviewed ⇒ R + F; step-up/block analogs; successful catches cost the intervention only; review catch rate and step-up stop rate treated as deterministic expectations for realized cost, i.e. a fraud under review incurs `(1−c)·(α·A+fee)`).
- Baselines: approve-all, review-all (ignoring capacity), review-top-K-by-probability.
- Unit tests with hand-computed numbers; closed-form check: with APPROVE vs REVIEW only, `c=1`, fixed `A`, the cost-minimizing threshold equals `(R+F)/(αA+fee+F)` (verify numerically on a grid). Monotonicity tests (higher p never lowers the recommended action severity for fixed A).

### 7C — Threshold search (`evaluation/threshold_search.py`)
1. **Global-threshold policies** (single probability cutoff → REVIEW vs APPROVE): scan a fine grid on `val_thr`; for each threshold compute flagged count, flagged/day, TP/FP/FN, precision, recall, expected & realized fraud loss, review cost, friction cost, total cost. Produce:
   - `F1-optimal` threshold,
   - `cost-optimal (unconstrained)`,
   - `cost-optimal under capacity` (largest feasible flagged/day ≤ capacity, then min cost).
2. **Per-transaction (amount-aware) policy** (`risk/policy.py`): choose the action with min expected cost for each transaction; then enforce the daily review capacity greedily: within each day rank REVIEW candidates by expected saving `cost(APPROVE) − cost(REVIEW)`, keep the top N = capacity, demote the rest to their next-best action. Return actions + diagnostics (how many demoted).
3. **Ranking policies under capacity:** compare ranking by `p` vs ranking by `p·A` (expected loss at stake) vs ranking by expected saving.
- **Capacity:** `capacity_per_day = analyst_daily_capacity` if set, else `ceil(capacity_pct_of_daily_volume × average daily volume of val_thr)`. Report the value used.
- **Realized vs expected:** headline cost reductions use **realized** cost on `val_thr` labels; expected cost is shown as a diagnostic (and as a calibration sanity check: expected ≈ realized).
- Output: `results/policy_comparison_<profile>.csv` with rows for approve-all, review-all, F1-optimal, cost-optimal (global), cost-optimal (global, capacity), amount-aware (no capacity), amount-aware (capacity), top-K by p, top-K by p·A. Columns: policy, flagged_per_day, recall, precision, fpr, fraud_loss, review_cost, friction_cost, total_realized_cost, total_expected_cost, % cost reduction vs F1-optimal, feasible (bool).

### 7D — Figures
Expected-loss curve vs threshold with F1-optimal and cost-optimal points marked and the feasible region shaded; cost breakdown stacked bars per policy; flagged-per-day vs capacity line; amount-dependent threshold curve `p*(A)` from the closed form. → `reports/figures/`.

### 7E — Sensitivity analysis
For each scenario in `cost_scenarios` (low_fp_cost, balanced, high_fp_cost, high_fraud_loss) recompute the cost-optimal global threshold and the amount-aware policy: optimal threshold, realized cost, recall, flagged/day, % saving vs F1-optimal. Output `results/sensitivity_<profile>.csv` and a figure. State the conclusion that there is no universally correct threshold without cost assumptions. A second sensitivity: vary `review_catch_rate` (0.7–1.0) and `stepup_stop_rate`.

### 7F — Risk score and policy artifact (`risk/scoring.py`, `risk/policy.py`)
- `risk_score = round(100 × calibrated_probability, 1)`; also expose `lift = p / base_rate`.
- `risk_level` is **derived from the action** (APPROVE→LOW, STEP_UP→MEDIUM, REVIEW→HIGH, BLOCK→CRITICAL), not from fixed probability cutoffs. Report the empirical distribution of calibrated probabilities at the observed prevalence and explain in the progress file why fixed 0–29/30–69/70–89/90–100 bands would be misleading at ~0.5% prevalence.
- Save `artifacts/<model_version>/calibrator.joblib` and `policy.json` (policy_version `cost_policy_v1`, cost config used, capacity, thresholds or the rule to recompute them, selected calibrator, validation metrics). Append experiments E04 (XGB + calibration + F1 threshold) and E05 (XGB + calibration + cost policy) to the log.

### 7G — Scripts and tests
`scripts/calibrate.py`, `scripts/optimize_policy.py --profile dev [--scenario balanced]`. Tests: calibrator monotonicity and fit/predict shapes; capacity never exceeded on any day; demotion logic; cost scenario overrides; policy serialization round trip; run on `tiny` quickly.

## Acceptance criteria
```bash
pytest tests/test_calibration.py tests/test_cost_model.py tests/test_policy.py -q
python scripts/calibrate.py --profile dev
python scripts/optimize_policy.py --profile dev
cat results/policy_comparison_dev.csv
```
Progress file: calibration before/after metrics, selected calibrator and why, policy comparison table, sensitivity table, capacity value, any case where cost-optimal is *not* better than F1-optimal (report honestly), and the interpretation of the amount-aware policy vs the global threshold.

## Do not
Fit calibrators or choose thresholds on the test window or on the same rows used to fit the calibrator. Report expected cost as if it were realized. Claim a % loss reduction without stating the cost assumptions next to it.
