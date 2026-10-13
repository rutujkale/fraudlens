# 8C — Model capacity and data size

**Script:** `scripts/run_experiment_8C.py` · **Profile:** `dev` · **Seeds:** 42–44

## Question

Is the model limited by **capacity** (it could fit more if we let it) or by
**data** (more complexity does not help, more rows would)? And how much of the
gap to a linear model is *representable* by a linear model at all?

Per the Part 08 decision, **all hyperparameters except `max_depth` are fixed at
the Part 06 best values** — only depth, training size and boosting rounds vary,
so the comparisons are clean. Early stopping uses the validation split.

## Setup

- Same temporal split: train 174,834 / validation 41,313 (342 fraud).
- **Capacity:** `max_depth ∈ {2,3,4,6,8,10}`, up to 800 rounds, early stop 30.
- **Data size:** fractions {0.10, 0.25, 0.50, 1.00} of the (earliest) training
  rows, at the best depth.
- **Rounds:** train PR-AUC vs val PR-AUC per boosting round at the best depth.
- **Linear reference:** logistic regression base; + explicit pairwise
  interactions; + binned numerics.

## Results — capacity (mean over 3 seeds)

| max_depth | val PR-AUC | train PR-AUC | gap | best_iter |
|----------:|-----------:|-------------:|----:|----------:|
| 2 | 0.9204 ± 0.0012 | 0.9300 | 0.0096 | 678 |
| 3 | 0.9261 ± 0.0158 | 0.9474 | 0.0213 | 424 |
| 4 | 0.9539 ± 0.0009 | 0.9980 | 0.0441 | 799 |
| 6 | 0.9634 ± 0.0005 | 1.0000 | 0.0366 | 698 |
| **8** | **0.9657 ± 0.0006** | 1.0000 | 0.0343 | 486 |
| 10 | 0.9655 ± 0.0010 | 1.0000 | 0.0345 | 480 |

## Results — data size (at depth 8)

| training fraction | rows | val PR-AUC | train PR-AUC |
|------------------:|-----:|-----------:|-------------:|
| 0.10 | 17,483 | 0.6783 ± 0.1436 | 0.9595 |
| 0.25 | 43,708 | 0.8893 ± 0.0501 | 0.9828 |
| 0.50 | 87,417 | 0.9308 ± 0.0014 | 1.0000 |
| **1.00** | 174,834 | **0.9657 ± 0.0006** | 1.0000 |

## Results — rounds and linear reference

- **Rounds:** val PR-AUC plateaus at 0.9653 by round ~490 and is flat to the 577
  the booster kept; train PR-AUC is 1.0 throughout. Added trees add nothing.
- **LogReg (cluster-bootstrap 95% CI):** base 0.9021 [0.8531, 0.9379];
  +interactions 0.9044 [0.8581, 0.9407]; +binned 0.9022 [0.8543, 0.9386].
  XGB reference at the same split: **0.9657**.

## Observations

- **Saturation looks data-limited, not capacity-limited — read it as a trend,
  not a measured law.** Val PR-AUC rises through depth 6 and then *saturates*
  (0.9634 → 0.9657 → 0.9655); depth 8 and 10 are within one std. Meanwhile train
  PR-AUC reaches **1.0000 at depth 6** — the trees can already memorise the
  training window, and more depth only adds variance, not signal. The overfit gap
  is roughly flat (0.034–0.037) from depth 4 upward.
- **More data is the clearest remaining lever.** Going 50% → 100% of the training
  window adds **+0.0349** PR-AUC — larger than the entire 2→10 depth change
  (+0.045 total, mostly before depth 6). The 10%-row curve is not just lower but
  **high-variance (±0.14)**, the signature of a model that hasn't seen enough
  fraud examples. Because that ±0.14 and the saturation both rest on three seeds
  of a single synthetic split, treat "more data helps" as a *trend*, not a
  measured law; confirm capacity-vs-data on real data (Part 11).
- **A linear model cannot close the gap.** Interactions and binning move LogReg
  by ≤0.002 (CIs overlap), leaving a stable **~0.06 PR-AUC gap** to XGBoost. The
  fraud signal here is genuinely higher-order/interaction-shaped, not something a
  richer linear basis recovers.
- **Hypothesis (not proven here):** the saturation is a property of the
  synthetic generator — episodes are constructed from a finite set of scenario
  templates, so beyond depth 6 the trees have already partitioned the template
  space. On real data (Part 11/12) we would expect capacity to matter more and
  the train/val gap to keep widening.

## Caveats

- Depth 3 is noisy (±0.016) because early stopping sometimes fired very early
  (best_iter = 144 for one seed); read the trend, not depth 3.
- "Best depth" is chosen by mean val PR-AUC (depth 8); depth 8 and 10 are
  statistically tied, so the data-size and rounds curves at depth 8 are
  representative but not uniquely optimal.
- Single temporal split; ± std is seeding variance, not split uncertainty.
