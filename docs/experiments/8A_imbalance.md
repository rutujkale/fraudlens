# 8A — Imbalance handling

**Script:** `scripts/run_experiment_8A.py` · **Profile:** `dev` · **Seeds:** 42–44

## Question

Fraud is ~1 in 126 transactions. Which imbalance treatment gives the best
*ranking* (PR-AUC, recall@1%) and the least *probability distortion* (Brier,
ECE) — and does sigmoid calibration repair whichever distortion remains?

## Setup

- Temporal split (no shuffle): train 174,834 rows (1,385 fraud) / validation
  41,313 rows (342 fraud); negative:positive ≈ 126.2.
- Fixed XGBoost hyperparameters (Part 06 best config); only the imbalance
  treatment changes. Three seeds per config (mean ± std).
- Ranking metrics (PR-AUC, recall@1%) on the full validation window.
- Calibration check: sigmoid fit on the **first chronological half** of
  validation (`val_cal`, 20,656 rows / 220 fraud) and Brier/ECE read on the
  **second half** (`val_thr`, 20,657 rows / 122 fraud) — fit and evaluation rows
  never overlap.
- Treatments: unweighted (`none`); `scale_pos_weight = neg/pos` (`spw_full`);
  `sqrt(neg/pos)` (`spw_sqrt`); random undersampling of negatives to 1:10 and
  1:50; and logistic regression with `class_weight="balanced"` (different model,
  included as the "linear + reweighting" reference).

## Results

| config | PR-AUC (mean ± std) | recall@1% | Brier before → after | ECE before → after |
|--------|--------------------:|----------:|---------------------:|-------------------:|
| `spw_full` | **0.9663 ± 0.0008** | **0.9542** | 0.00099 → 0.00118 | 0.00029 → 0.00091 |
| `spw_sqrt` | 0.9653 ± 0.0005 | 0.9493 | 0.00098 → 0.00119 | 0.00017 → 0.00091 |
| `none` | 0.9614 ± 0.0014 | 0.9405 | 0.00108 → 0.00129 | 0.00061 → 0.00087 |
| `undersample_1_50` | 0.9567 ± 0.0011 | 0.9405 | 0.00119 → 0.00138 | 0.00036 → 0.00091 |
| `undersample_1_10` | 0.9413 ± 0.0004 | 0.9230 | 0.00226 → **0.00162** | 0.00459 → **0.00107** |
| `logreg_balanced` | 0.8680 (single fit) | 0.8392 | 0.01610 → **0.00226** | 0.03346 → **0.00131** |

Brier/ECE measured on `val_thr`; calibrator fit on `val_cal`.

## Observations

- **`scale_pos_weight = neg/pos` is the best ranking lever** (+0.005 PR-AUC,
  +0.014 recall@1% over unweighted), because it keeps every majority example
  while shifting the loss toward the minority class. `sqrt` weighting is between
  the two, as expected.
- **Undersampling hurts ranking and its benefit is only cosmetic.** Discarding
  negatives to 1:10 costs ~0.025 PR-AUC vs `spw_full`; 1:50 recovers most of it
  but still trails. Undersampling **distorts the base rate** — raw Brier
  0.00226 (≈2× the unweighted model) — and that is the *only* setting where
  sigmoid calibration clearly helps (0.00226 → 0.00162).
- **Calibration does not help the well-ranked models.** For `none`, `spw_full`,
  `spw_sqrt`, `undersample_1_50` the calibrated Brier/ECE is equal or slightly
  *worse* on `val_thr`. This repeats the Part 07 finding: on near-separable
  synthetic fraud the raw score is already calibrated, so a sigmoid only adds
  variance. `logreg_balanced` is the exception — `class_weight` genuinely
  inflates its probabilities (raw Brier 0.0161) and calibration repairs them
  (0.00226).
- **Hypothesis (not proven here):** the reason `spw_full` does not need
  calibration is the same reason Part 07 saw: positives score ≈1 and negatives
  ≈0, so the mean predicted probability tracks the base rate regardless of the
  weight. This would change on the Part 11 IEEE-CIS data, where classes overlap.

## Caveats

- `dev` is synthetic and near-separable; ranking differences between `spw`
  variants are small (±0.001). The direction, not the magnitude, is the result.
- `logreg_balanced` retrains a different model class, so it is a reference
  point, not a like-for-like imbalance arm.
- Single public split; ± std over three seeds only captures XGBoost's seeding
  variance, not split uncertainty (cluster-bootstrap CIs are in the summary CSV).
