# Part 09 — Isolation Forest and the unseen-fraud experiment (E15)

**Script:** `scripts/run_unseen_experiment.py --profile unseen_lab --holdout structuring`
**Profile:** `unseen_lab` (15,000 customers, 180 days, fraud rate 0.005, **structuring 35%** of
fraud episodes) · **Seeds:** 42–46 · **Model:** Part-06 XGBoost config

## Question

> Can an **unsupervised anomaly score** help a supervised fraud model detect a fraud pattern
> that was **absent from supervised training**?

Held-out-scenario recall is the headline. Validation window only; the test window is untouched.

## Pre-registered outcomes (fixed before running)

1. **(A) No meaningful benefit** — the anomaly score does not improve held-out recall.
2. **(B) Better unseen-fraud recall, little/neutral change in overall PR-AUC.**
3. **(C) Meaningful overall improvement** — then robustness on other seeds/scenarios is required.

Adoption rule (also fixed before running): adopt the anomaly feature only if **both** the
realistic (`IF-all`) and the strict (`IF-strict`) variants show a supported benefit (paired
cluster-bootstrap CI of the recall difference excludes 0) **and** the per-seed gain is ≥ 0.02.
A win that requires the detector to have *seen* the pattern (unlabeled) is not an unseen-fraud win.

## Setup

- **Held-out scenario:** `structuring` dropped from the **supervised training rows only**
  (76,010 → 794,360 rows keep; the 1,650 structuring training rows are removed, **not relabelled**).
  Validation keeps all scenarios. Eval window has **344 structuring positives** (≥ 300 target).
- **Isolation Forest** (`models/anomaly.py`): 28-feature curated behavioural subset (amount
  deviation, velocity/counts, amount sums, near-threshold count, distance/travel, hour deviation,
  device/recipient novelty). No IDs, no labels, no monotonic tenure features. Median impute +
  missing indicators; `contamination="auto"`. Grid chosen by **val PR-AUC of the IF-alone score on
  non-held-out fraud only** → `n_estimators=400, max_samples=1024` (val PR-AUC 0.128).
- **Cross-fitting:** each training row scored by an IF fit on the *other* folds' customers
  (**customer-grouped 5-fold**); validation rows scored by an IF fit on all training rows. A test
  asserts no training row is scored by a model fit on its own customer.
- **Systems** (5 seeds each): S0 XGB without structuring · S1 S0 + IF-all score · S1s S0 +
  IF-strict score · S2 IF-alone · S3 rank blend of S0 + IF (weight tuned on **seen** val fraud) ·
  S4 oracle XGB trained *with* structuring.
- **Metric:** held-out recall at top 0.5/1/2/5% of all val transactions; overall PR-AUC; seen-scenario
  recall@1%; cluster-bootstrap CIs; paired S1−S0 / S1s−S0 difference CIs.

## Results — system comparison (mean ± std over 5 seeds)

| system | struct rec@0.5% | struct rec@1% | struct rec@2% | struct rec@5% | PR-AUC (overall) | ATO rec@1% (seen) |
|--------|---:|---:|---:|---:|---:|---:|
| S0 (no structuring) | 0.000 | 0.0035 ± 0.0024 | 0.0314 ± 0.0091 | 0.2610 ± 0.0111 | 0.4523 ± 0.0012 | 0.884 |
| S1 (+ IF-all) | 0.0012 | 0.0134 ± 0.0108 | 0.0756 ± 0.0386 | 0.3122 ± 0.0381 | 0.4480 ± 0.0218 | 0.855 |
| S1s (+ IF-strict) | 0.0012 | 0.0122 ± 0.0056 | 0.1076 ± 0.0187 | 0.3076 ± 0.0799 | 0.4490 ± 0.0247 | 0.860 |
| **S2 (IF alone)** | **0.0378** | **0.2035** | **0.4331** | **0.8227** | 0.1206 | 0.265 |
| S3 (rank blend) | 0.000 | 0.0035 | 0.0314 | 0.2610 | 0.4523 | 0.884 |
| S4 (oracle, with structuring) | 1.000 | 1.000 | 1.000 | 1.000 | 0.9576 | 0.882 |

S3 picked blend weight **w = 0** in every seed (see below), so S3 ≡ S0.

## Results — isolation forest alone, per scenario (val)

| scenario | n_pos | IF ROC-AUC | IF PR-AUC |
|----------|---:|---:|---:|
| **structuring (held out)** | 344 | **0.969** | 0.040 |
| geo_anomaly | 34 | 0.960 | 0.006 |
| burst | 182 | 0.958 | 0.171 |
| account_takeover | 83 | 0.878 | 0.010 |
| amount_anomaly | 24 | 0.711 | 0.0004 |

## Results — paired comparison (held-out recall@1%)

| comparison | per-seed mean ± std | seed-bagged diff [95% CI] |
|---|---|---|
| S1 − S0 (IF-all) | +0.0099 ± 0.0107 | +0.0291 [+0.0075, +0.0521] |
| S1s − S0 (IF-strict) | +0.0087 ± 0.0041 | +0.0029 [−0.0065, +0.0123] |

Single-system held-out recall@1% CIs: S0 [0.000, 0.0096], S1 [0.010, 0.057], S2 [0.142, 0.261].

## Which pre-registered outcome occurred?

**Mostly (A), with a specific nuance — and a clear negative for the "unseen" framing.**

- **The unsupervised score *does* detect the held-out pattern.** IF-alone (S2) reaches structuring
  recall@1% **0.204** and @5% **0.823**, with structuring ROC-AUC **0.969** (the highest of any
  scenario). So structuring is anomaly-shaped and the detector finds it.
- **But the supervised model barely benefits.** Adding IF-all as a feature (S1) lifts held-out
  recall@1% from 0.0035 to 0.0134; the *seed-bagged* difference is supported (+0.0291, CI > 0) but
  the **per-seed** gain is only +0.0099 and is **seed-driven** (seed 42 alone gives +0.029;
  two seeds give exactly 0). Overall PR-AUC does **not** improve (0.448 vs 0.452).
- **The strict variant shows no supported benefit** (S1s−S0 bagged +0.0029, CI includes 0). So the
  small S1 gain requires the detector to have *seen* structuring rows (unlabeled) during training —
  it is **not** evidence for genuinely-unseen detection.
- **You cannot tune the blend without the unseen labels.** S3's weight, chosen on *seen* val fraud,
  collapses to **w = 0** in every seed — the anomaly score simply isn't useful on the scenarios the
  tuner can see, so it gets zero weight and the blend degrades to S0.
- **Detectability holds:** S4 (trained with structuring) reaches recall 1.000 everywhere, so the
  pattern is fully learnable from the window features — a weak S4 would have made the experiment
  uninformative.

## Adoption decision: **not adopted**

No `artifacts/.../iforest.joblib` is saved. The anomaly feature is **not** added to the supervised
system. Reason (`results/exp_09_unseen_decision_*.json`):

> the IF-strict variant shows no benefit (the gain requires the detector to have seen the pattern
> unlabeled) and the per-seed S1 gain (< 0.02) is driven by seed bagging.

The IF score is nonetheless a **useful standalone structuring/anomaly alert** (S2) and is carried
forward as a **monitoring signal**, not as a model feature. Part 12 will therefore *not* include an
"XGBoost + anomaly score" row.

## Caveats

- One profile and one held-out scenario; `unseen_lab` is a lab, no Part-12 test-window claim here.
- The optional robustness sweep (hold out `geo_anomaly` / `amount_anomaly`) was **skipped**: those
  scenarios have only 34 / 24 validation positives on `unseen_lab`, far below the ~300 needed, so a
  table would be noise. `--holdout` supports them; a much larger regeneration would be required.
- S2 has no training seed, so it is reported once (no ± std).
- The paired CI is reported two ways because they disagree: the seed-bagged score (mean of 5 seeds as
  one ranker) supports S1, the per-seed mean does not clear the materiality bar. Both are shown.
