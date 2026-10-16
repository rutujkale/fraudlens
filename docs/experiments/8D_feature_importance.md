# 8D — Feature-importance agreement

> **Superseded by the Part 02b cold-start fix (2026-10-16).** The numbers below were produced
> on the pre-fix dataset with model `xgb_20261011_v1`; the data was regenerated and the dev model
> retrained as `xgb_dev_20261015_v1`. See [../progress/part_02b.md](../progress/part_02b.md) for the before/after table.

**Script:** `scripts/run_experiment_8D.py` · **Profile:** `dev`

## Question

Three importance methods are in common use — XGBoost **gain**, **permutation**
importance on the held-out validation window, and TreeSHAP **mean|SHAP|**. Do
they agree on which features matter? Where they disagree, which one is telling
the truth? We answer with two **planted controls** that must rank last:

- `_random_noise` — i.i.d. Gaussian, no relationship to the label.
- `_shuffled_minutes_since_prev_tx` — a permutation of the strongest recency
  feature, which therefore carries no usable signal.

## Setup

- Same temporal split; one model (fixed Part 06 hyperparameters, seed 42) with
  the 53 real features **plus the two controls** = 55 columns.
- **gain** — normalised split gain from the booster.
- **permutation** — mean PR-AUC drop when a column is shuffled on validation
  (5 repeats, ± std).
- **permutation (group)** — same, but shuffling every feature in a correlation
  cluster together, so correlated features cannot mask one another.
- **mean|SHAP|** — mean absolute TreeSHAP value over the validation rows.
- Sanity model val PR-AUC **0.9651** (controls did not change the model).

## Results — top features by each method

| rank | by gain (`gain_share`) | by permutation (PR-AUC drop) | by mean\|SHAP\| |
|-----:|------------------------|------------------------------|-----------------|
| 1 | days_since_last_tx (0.115) | is_new_recipient (0.051) | tx_from_device_prior (1.099) |
| 2 | minutes_since_prev_tx (0.110) | minutes_since_prev_tx (0.046) | merchant_category (1.002) |
| 3 | tx_count_1h (0.076) | new_merchant_flag (0.041) | minutes_since_prev_tx (0.893) |
| 4 | is_night (0.070) | merchant_category (0.028) | device_count_30d (0.676) |
| 5 | accounts_seen_on_device (0.065) | transaction_type (0.017) | new_merchant_flag (0.658) |

## Results — planted controls and method agreement

| feature | mean\|SHAP\| (rank) | gain_share (rank) | perm drop (rank) |
|---------|--------------------:|------------------:|-----------------:|
| `_shuffled_minutes_since_prev_tx` | 0.2342 (**17** / 55) | 0.0018 (49) | −0.0003 (52) |
| `_random_noise` | 0.1408 (**26** / 55) | 0.0017 (51) | −0.0001 (51) |

Spearman rank correlation between methods (55 features):

| | gain | permutation | perm-group | mean\|SHAP\| |
|-|-----:|------------:|-----------:|-------------:|
| **gain** | 1.00 | 0.59 | 0.27 | 0.57 |
| **permutation** | 0.59 | 1.00 | 0.41 | 0.65 |
| **perm-group** | 0.27 | 0.41 | 1.00 | 0.42 |
| **mean\|SHAP\|** | 0.57 | 0.65 | 0.42 | 1.00 |

## Observations

- **The three methods agree on the *families* that matter (ρ 0.57–0.65)** —
  device identity, recency/velocity, and merchant novelty appear near the top of
  every list. For a headline "these are the drivers" story any of them is fine.
- **They disagree on the *tail*, and the controls expose how.** Both controls
  rank **last under gain (49, 51) and permutation (51, 52)** with a
  non-positive PR-AUC drop, exactly as they should. Under **mean|SHAP| they rank
  17th and 26th of 55** — i.e. the middle of the pack.
- **This is a real effect, not a bug.** Counting splits confirms the model does
  fit the controls: `_random_noise` is split on in **736 internal nodes across
  312 of 578 trees** and the shuffled column in **1065 nodes** (1.4% and 2.0% of
  the 54,152 internal nodes). Path-dependent mean|SHAP| measures *how much the
  model uses a feature*, including those spurious splits, so it credits noise;
  permutation measures *how much the feature helps generalisation on held-out
  data*, so it correctly reports ≈0. **When the model overfits, "used" and
  "useful" diverge, and mean|SHAP| follows "used."**
- **Gain is scale/usage-biased too.** It ranks `days_since_last_tx` first and
  `is_night`/`tx_count_1h` highly because they are used high in the trees or on
  many rows, but several of them drop sharply under permutation (redundant with
  sibling features). Permutation is the only method that answers "would removing
  this hurt?".
- **Correlated features need grouped permutation.** `days_since_last_tx` has
  permutation drop 0.0035 alone but **0.176 when its whole cluster is shuffled**
  (the recency cluster also contains `minutes_since_prev_tx`); `device_count_30d`
  goes 0.016 → 0.141 with its device cluster. That is why perm-group ranks
  differ most from single permutation (ρ 0.41) — single-feature permutation
  under-credits features whose signal is shared inside a cluster.
- **Practical takeaway for Part 10:** if SHAP is shown as "the reasons", it must
  be the *signed, per-transaction* attribution on a calibrated model, never raw
  mean|SHAP| presented as a global ranking — on this overfit task mean|SHAP|
  would put noise ahead of real features.

## Caveats

- One seed/model; correlations are rank-based and the top families are stable,
  but exact ranks are not.
- TreeSHAP here uses the default path-dependent perturbation; interventional
  SHAP with a background sample would likely push the controls lower but costs a
  background dataset and is the subject of Part 10.
- The controls inflate the feature count to 55, so a "rank of 17 of 55" is two
  positions worse than "17 of 53"; the conclusion is unchanged.
