# Part 12 — Freeze, Locked Test Evaluation, and Headline Numbers

## Goal
Run the **one-shot** evaluation on the held-out test window with everything frozen, and generate the results that appear in the README and on your resume. After this part, no model, feature, threshold or cost decision may change in response to test results.

## Prerequisites
Parts 05–11 complete and committed. All experiments that need tuning are done. Working tree clean.

## Tasks

### 12A — Freeze (`scripts/freeze.py`)
Before touching the test window, write `results/freeze.json`:
- git commit (must be a clean tree; refuse otherwise), config hash, feature version, model_version(s), sha256 of each artifact (models, calibrator, policy.json, iforest if adopted), split boundaries, cost config + capacity value, list of systems to be evaluated (Rules, LogReg, XGBoost, and XGBoost + anomaly score **only if adopted in Part 09**), seeds.
- Final models: retraining on train+validation is allowed **only if decided now and recorded in the freeze file** (state the rationale: more recent data; embargo respected). If retrained, thresholds/calibrator must be re-derived by the same documented procedure on the final validation slice, **not** on test. Default: do not retrain; use the Part 06–07 artifacts as-is.

### 12B — Locked evaluation (`scripts/final_evaluation.py`)
- Verifies artifact hashes against `freeze.json`; aborts on mismatch.
- Accesses the test window only via `TestSetGuard(final=True)`. A second run requires `--allow-rerun --reason "<text>"`, and logs the reason in `results/test_access_log.csv`. (Rerunning because results were disappointing is not allowed; the legitimate reason would be a crash.)
- Computes, on the test window, with cluster-bootstrap CIs (n from config):
  1. **Model comparison:** Rules, LogReg, XGBoost (+ anomaly score if adopted): PR-AUC, ROC-AUC, precision/recall/F1 at the **frozen** thresholds, recall@top-K for all K, recall at 90/80/70% precision, FPR, Brier, ECE (before/after calibration), lift over base rate. Prevalence of the test window reported. Accuracy shown only in a footnote column.
  2. **Calibration:** reliability diagram, Brier, ECE with the frozen calibrator.
  3. **Decision policy:** realized cost for approve-all, review-all, F1-optimal threshold (frozen from validation), cost-optimal global, cost-optimal global under capacity, amount-aware under capacity; flagged/day vs capacity; % cost reduction vs F1-optimal with CI; breakdown of fraud loss / review / friction costs.
  4. **Sensitivity:** the four cost scenarios applied with thresholds/policies **re-derived on validation under each scenario** and evaluated on test.
  5. **Per-scenario recall** at the frozen policy and at top-1% budget; per-archetype recall/FPR.
  6. **Drift evidence:** compare validation (pre-drift) vs test (post-drift) metrics for each model; PSI of key features and of the score between validation and test.
  7. **Customer holdout (Setting B):** evaluate unseen customers vs seen customers on test.
  8. **Fairness slice** (reuse Part 10 code) on test at the frozen policy.
  9. **Sim-to-real table finalization:** replace the validation-based synthetic figure in Part 11's gap table with the test figure.
  10. **Top false positives and false negatives:** counts by hard-negative type and by scenario/stealth bucket (answers "top sources of false positives").
- Outputs: `results/final_results.json` (single source of truth), `results/final_model_comparison.csv`, `results/final_policy_comparison.csv`, figures in `reports/figures/final_*.png` (PR curves, ROC, calibration, recall@K, expected-loss curve with frozen thresholds, cost bars, sensitivity).

### 12C — Headline numbers (`scripts/make_headline_numbers.py`)
Generate `reports/headline_numbers.md` and keys in `final_results.json` for exactly these four results, with the associated assumptions printed beside each:
1. **Model performance:** XGBoost PR-AUC (CI) vs Rules PR-AUC (CI) vs LogReg, test prevalence, recall@1% and recall@K for the capacity budget.
2. **Business decision:** realized-cost reduction % of the best capacity-feasible policy vs the F1-optimal threshold (CI), flagged/day vs capacity, with the cost assumptions (α, R, F, capacity) stated.
3. **Unseen fraud:** structuring recall before → after the anomaly score at the stated alert budget (from Part 09; the result may be null — report as is) and the drift degradation (val → test).
4. **Engineering:** placeholders for API p95 latency (filled in Part 13) and feature count, number of automated tests (counted by code via `pytest --collect-only -q`), leakage-test count.
Also generate resume-ready bullet **candidates** with real numbers inserted (never invented), each mapped to the file/key it came from.

### 12D — Reporting rules
- Report **actual** prevalence in train/val/test.
- Where a result is unfavorable (e.g. policy no better than F1-optimal, anomaly feature no help, strong degradation under drift), headline it as a finding, not a failure.
- Add `docs/final_results.md` (generated from the JSON): tables + figure links + three paragraphs of interpretation limited to what the numbers support.

### 12E — Tests (`tests/test_final_guard.py`)
Guard raises without `final=True`; hash mismatch aborts; headline script fails if a required key is missing; test access log appends.

## Acceptance criteria
```bash
python scripts/freeze.py
python scripts/final_evaluation.py --profile main
python scripts/make_headline_numbers.py
cat reports/headline_numbers.md
cat results/test_access_log.csv
```
Progress file: freeze details, the four headline results, test-access log, anything surprising. Commit with message `final evaluation (locked)`; tag `v1.0-eval`.

## Do not
Evaluate on test before freezing. Re-tune anything after seeing test numbers. Re-run the evaluation to get better numbers. Round or "clean up" results by hand.
