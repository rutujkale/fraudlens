# Part 09 — Isolation Forest & the Unseen-Fraud Experiment

## Goal
Answer one precise question with a fair experiment:

> **Can an unsupervised anomaly score help a supervised fraud model detect a fraud pattern that was absent from supervised training?**

A null or negative result is acceptable and must be reported honestly. Validation data only; the test window is not touched.

## Prerequisites
Parts 05–07 complete. Generate and featurize the lab dataset:
```bash
python scripts/generate_data.py --profile unseen_lab
python scripts/build_features.py --profile unseen_lab
```
`unseen_lab` oversamples **structuring** (≈35% of fraud episodes) so the held-out scenario has enough positives to evaluate (report the count; if fewer than ~300 structuring fraud rows land in the evaluation window, increase customers via config and regenerate).

## Design

### Held-out scenario
`structuring` is removed from **supervised** training: drop all transactions belonging to structuring episodes from the supervised training set (do not relabel them as legit). Validation/evaluation data keeps them. Other scenarios remain.

### Isolation Forest (`models/anomaly.py`)
- Features: a curated numeric subset of behavioral features (amount deviation, velocity counts, amount sums, near-threshold counts, distance/travel, hour deviation, device/recipient novelty). No IDs, no labels. Median-impute with missing indicators.
- Fit **without labels**. Two variants:
  - `IF-all`: fit on all training-window rows (realistic: an anomaly detector sees all unlabeled traffic, including unseen attack types).
  - `IF-strict`: fit on training-window rows excluding structuring episodes (strictest "unseen" test; shows how much the detector itself depends on having seen the pattern).
- Hyperparameters: `n_estimators` 200–400, `max_samples` 256–1024, `contamination="auto"`; choose by a small grid using validation PR-AUC of the *IF-alone* score on **non-held-out** fraud only (do not tune on structuring).
- Anomaly score = negated `score_samples`, then rank-normalized to [0, 1] using the training distribution.

### Avoiding in-sample skew (important)
If the anomaly score is used as a supervised feature, training rows must not be scored by an IF that was fit on those same rows. Use **customer-grouped 5-fold cross-fitting** on the training window: each training row's score comes from an IF fit on the other four folds' customers; validation rows use an IF fit on all training rows. Add a test that no training row is scored by a model fit on its own customer. Record this in the progress file and in `docs/what_went_wrong.md` if you first tried the naive way and saw a difference.

### Systems compared (same XGBoost hyperparameters, 5 seeds each)
| ID | System |
|---|---|
| S0 | XGBoost, trained **without** structuring (the "unseen" baseline) |
| S1 | S0 + anomaly score as an extra feature (IF-all, cross-fitted) |
| S1s | S0 + anomaly score (IF-strict) |
| S2 | Isolation Forest alone (IF-all) |
| S3 | Rank-average blend: S0 score ranked + anomaly score ranked (weight tuned on **non-structuring** validation fraud only) |
| S4 (oracle) | XGBoost trained **with** structuring — the ceiling, shows the pattern is learnable from the features |

### Evaluation (validation window of `unseen_lab`)
- **Structuring-only recall** at fixed alert budgets (top 0.5%, 1%, 2%, 5% of all transactions) — the headline metric.
- **PR-AUC overall** and **PR-AUC structuring-vs-legit only** (treat other fraud as excluded for this slice).
- Recall on **seen** scenarios (to check the anomaly feature doesn't hurt known fraud).
- Cluster-bootstrap CIs; a **paired** comparison S1 vs S0 (same bootstrap resamples) with the CI of the difference. Report mean ± std across the 5 training seeds.
- Per-scenario IF-alone ROC-AUC/PR-AUC (does the anomaly score pick out structuring at all?) and score distributions for structuring vs legit vs other fraud (figure).
- Detectability check: confirm structuring is separable from window features (S4 ≫ S0) — if S4 is also weak, the experiment is uninformative; say so and fix feature coverage (amount-sum / near-threshold windows) before concluding.

### Interpreting outcomes (write in `docs/experiments/09_unseen_fraud.md`)
Pre-register these three outcomes in the document **before running**: (A) no meaningful benefit; (B) better unseen-fraud recall with little change in overall PR-AUC; (C) meaningful overall improvement (then check robustness: other seeds, other held-out scenarios). Then report which occurred, with numbers. If the first attempt shows no benefit, document what you changed afterward and why (e.g. feature subset, cross-fitting), as an entry in `docs/what_went_wrong.md` ("Isolation Forest failure") — and be clear which version is final and that the final configuration was not chosen using structuring labels.

### Optional robustness
Repeat holding out `geo_anomaly` (and one other scenario) with the same code path via a `--holdout` argument. Report the table.

## Deliverables
- `models/anomaly.py`, `scripts/run_unseen_experiment.py --profile unseen_lab --holdout structuring`, tests in `tests/test_anomaly.py` (cross-fit separation, determinism, scores in [0,1], strict variant excludes held-out rows).
- `results/exp_09_unseen.csv`, figures (structuring recall vs alert budget per system; anomaly-score distributions), log entry E15 (and E16 for holdout variants).
- Final trained anomaly model saved at `artifacts/<model_version>/iforest.joblib` **only if** it is adopted into the final system; otherwise record "not adopted" with reasons. Part 12 will use whichever decision you make here ("XGBoost + anomaly score" as a row only if adopted).

## Acceptance criteria
```bash
pytest tests/test_anomaly.py -q
python scripts/run_unseen_experiment.py --profile unseen_lab --holdout structuring
cat results/exp_09_unseen.csv
```
Progress file: structuring positives in the evaluation window, the system comparison table with CIs, which pre-registered outcome occurred, decision on adoption.

## Do not
Tune anything using structuring labels. Fit the anomaly model on a row and then use that row's in-sample score as a training feature. Report only the seed that looks best. Call the anomaly score useful unless the paired CI supports it.
