# 8E — Drift and retraining policy

> **Superseded by the Part 02b cold-start fix (2026-10-16).** The numbers below were produced
> on the pre-fix dataset with model `xgb_20261011_v1`; the data was regenerated and the dev model
> retrained as `xgb_dev_20261015_v1`. See [../progress/part_02b.md](../progress/part_02b.md) for the before/after table.

**Script:** `scripts/run_experiment_8E.py` · **Profile:** `drift_lab` (1,156,589 rows, 6,595 fraud)

## Post-coldstart-fix re-check (2026-10-16)

The Part 02b cold-start fix changed `drift_lab` to 1,097,615 rows / 6,275 fraud. Re-running 8E on the
new data (behavioural adversarial AUC **0.6995**, all-features **0.9821**, tenure **0.9759**) leaves
**every conclusion below unchanged** — magnitudes moved, nothing flipped:

- The clock still dominates the naive metric: all-features 0.9821 vs behaviour-only 0.6995.
- Behavioural covariate drift stays mild. PSI is still led by the tenure features (`history_days`
  4.26, `device_age_days` 2.67, `n_prior_tx` 1.30, `tx_from_device_prior` 0.94); the top behavioural
  feature is again `hour_deviation` (0.26, down from 0.53).
- Retraining still recovers only ~0.04 PR-AUC (frozen 0.9475 → expanding 0.9878 in the last window).
- The dataset-specific caveat stands: quote **0.6995**, never 0.9821.

Source: `results/exp_8E_adversarial_summary_drift_lab.json`, `results/exp_8E_psi_drift_lab.csv`,
`results/exp_8E_backtest_drift_lab.csv`.

## Question

`drift_lab` switches fraud behaviour at day 120 of 240: the scenario mix becomes
structuring-heavy and stealth rises. Does a frozen model degrade? How much does
retraining recover, and does the *amount of history* used matter? We also
separate **covariate** drift (inputs move) from **concept** drift (the
input→label relationship moves).

## Setup

- Drift switch = day 120 (`drift.start_fraction 0.50`). Four monthly evaluation
  windows after it (2026-05-31 … 2026-08-29).
- **Point-in-time training** at each step: only rows with `timestamp < step`
  **and** `label_available_at ≤ step` (label delays 3–14 days are enforced, not
  assumed).
- Four policies (200 trees, fixed Part 06 space, seed 42):
  - **frozen** — trained once on pre-shift matured rows, never updated.
  - **expanding** — all matured rows up to the step.
  - **sliding_60d** — last 60 days of matured rows.
  - **recency_weighted** — all matured rows, weights `0.5^(age/30d)`.
- **Covariate drift:** PSI (reference = pre-shift, current = post-shift) and
  adversarial validation (XGBoost distinguishing pre from post, 50k each side).
- This experiment uses the whole `drift_lab` timeline; `drift_lab` is a dedicated
  lab, so there is no Part-12 test-window claim here.

## Results — monthly PR-AUC by policy

| step | frozen | expanding | sliding_60d | recency_weighted |
|------|-------:|----------:|------------:|-----------------:|
| 2026-05-31 | 0.9439 | 0.9827 | 0.9817 | 0.9777 |
| 2026-06-30 | 0.9373 | 0.9810 | 0.9817 | 0.9744 |
| 2026-07-30 | 0.9338 | 0.9781 | 0.9794 | 0.9741 |
| 2026-08-29 | 0.9610 | 0.9876 | 0.9893 | 0.9884 |
| **mean** | **0.9440** | **0.9824** | **0.9830** | **0.9787** |

(recall@1% tracks the same ordering: frozen 0.945, expanding 0.984, sliding
0.980, recency-weighted 0.978.)

## Results — covariate drift

PSI (pre-shift vs post-shift), all numeric features:

| feature | PSI | flag |
|---------|----:|------|
| device_age_days | 4.114 | shift |
| history_days | 4.043 | shift |
| n_prior_tx | 1.780 | shift |
| tx_from_device_prior | 1.141 | shift |
| hour_deviation | 0.533 | shift |
| amount_vs_lifetime_median | 0.170 | monitor |
| days_since_last_tx, minutes_since_prev_tx, travel_speed_kmh, distance_from_prev_km | ≈0.169 | monitor |

Adversarial validation (can a model tell pre from post?):

| feature set | adversarial AUC |
|-------------|----------------:|
| all 53 features | 0.9998 |
| the 4 tenure features only | 0.9989 |
| `history_days` alone | 0.9740 |
| **49 behavioural features (tenure excluded)** | **0.7072** |

## Observations

- **Naive drift metrics on a temporal split are dominated by the clock.** The
  four highest-PSI features (`device_age_days`, `history_days`, `n_prior_tx`,
  `tx_from_device_prior`) are **monotonic tenure/cumulative-count features** —
  they *must* differ between any earlier and later window. `history_days` alone
  separates pre from post with AUC 0.974, and the tenure four give 0.9989, so the
  headline "adversarial AUC 0.9998" is almost entirely time, not fraud drift.
  **Excluding them, behavioural adversarial AUC drops to 0.7072** — real but
  moderate covariate shift (consistent with the scenario-mix change: structuring
  alters channel/type/amount distributions).
- **Behavioural PSI is mild.** Only `hour_deviation` is a clear shift (0.533),
  which fits the simulator: the drifted mix is heavier on account-takeover-style
  odd-hour activity. The handful of `monitor`-level features (amount vs lifetime
  median, recency/geo distances) move together, again consistent with a shifted
  scenario mix rather than the customer base changing.
- **Concept drift is mild, and retraining recovers ~0.04 PR-AUC.** The frozen
  model still scores 0.9440 mean — the per-scenario feature signals the frozen
  trees learned are largely still valid after the switch. Retraining lifts this
  to 0.9824 (expanding) / 0.9830 (sliding-60d).
- **Recent data is enough.** `sliding_60d` matches or beats `expanding` in every
  window while training on ~1/4 the rows — for a drift-monitoring policy this
  argues for a rolling window, not unbounded accumulation.
  `recency_weighted` under-performs both despite seeing all rows, because soft
  weights still mix stale rows in; a hard (sliding) cutoff is cleaner here.
- **The frozen curve is not monotone** (0.944 → 0.937 → 0.934 → 0.961). The last
  window is easier for the frozen model, likely because the post-switch scenario
  mix stabilises toward types the pre-shift model already handles. We report it
  rather than smoothing it away.
- **Hypothesis (not proven here):** because the simulator changes the *mix* of
  scenarios rather than the feature→label mapping inside each scenario, this is
  closer to covariate than concept drift, which is why a frozen model stays
  competitive and why behavioural drift is small.

## Caveats

- The frozen model staying competitive and the ~0.04 retraining gain are
  properties of *this* simulator's scenario-mix drift, not established facts: the
  2026-08-29 window is easiest for **every** policy (frozen 0.9610, expanding
  0.9876, sliding 0.9893, recency 0.9884), so read the *gap* between policies, not
  the level. Re-test on real data in Part 11.
- PSI is computed on numeric features only (categorical PSI needs ordered bins);
  categorical drift (transaction_type/channel mix) is visible only indirectly,
  through the behavioural adversarial AUC.
- 200 trees, fixed seed, no early stopping in the backtest (a choice for speed);
  absolute PR-AUC would be a little higher with the full Part 06 round budget.
- The "behavioural" set is defined by four named monotonic features; a different
  definition of tenure would shift the adversarial AUC somewhat, though the
  tenure-vs-behavioural gap (0.9989 vs 0.7072) is far too large to be a
  definitional artefact.

## Interview framing

The drift number to quote is the **behavioural adversarial AUC 0.7072**, never the
0.9998: the all-features figure is almost entirely the monotonic tenure clock, so
"the model can tell pre from post with AUC 0.9998" would be a false drift claim.
The honest one-liner: *"Temporal splits make tenure features drift by construction;
strip them and real behavioural covariate drift is mild (0.7072), and retraining
buys only ~0.04 PR-AUC — a result I treat as dataset-specific to this simulator's
scenario-mix change until Part 11 re-tests it on real data."*
