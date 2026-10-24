# Retraining and monitoring plan

What is monitored, the pre-stated rules that would trigger a retrain, and the
evidence behind them. The evidence is Experiment 8E
(`docs/experiments/8E_drift.md`, `results/exp_8E_*`).

## What is monitored (`make monitor`, `monitoring/metrics.py`)

Per calendar month over the **train + validation** timeline only (the test window
is closed; the published test PSI is quoted from `results/final_results.json`, not
recomputed):

- **Feature PSI** for the key business features and the top-15 SHAP features,
  against the training window as reference. Bands: `< 0.1` stable, `0.1–0.25`
  monitor, `> 0.25` shift (conventional rules of thumb, not derived from this data).
- **Score PSI** (calibrated probability) per period.
- **Fraud prevalence** and **calibration drift** (mean predicted vs observed rate),
  using only labels **matured** by each period end (`label_delay_days = [2, 7]`).
- **Delayed-label performance**: per-period PR-AUC, precision/recall at the frozen
  policy threshold, and FPR.

Outputs: `results/monitoring_dev.csv`, `results/monitoring_psi_dev.csv`, and three
figures (`reports/figures/monitoring_dev_{psi_heatmap,score_dist,performance}.png`).
The per-period table carries a **`sample`** column: `in_sample` (the model was
trained on it), `validation` (held out), or `mixed` (straddles `train_end`). So the
`dev` months read: 2026-01 `in_sample`, 2026-02 `in_sample`, 2026-03 `mixed`,
2026-04 `validation`. **Only the validation rows are out-of-sample** — the January
and February PR-AUCs (~1.0000) are in-sample fits and must never be quoted as
performance, and March is part training, part validation.

## Pre-stated retraining rules (`monitoring.metrics.retraining_trigger`)

Documented **before** looking at any run:

1. **Score PSI > 0.25 for two consecutive periods** — the calibrated score
   distribution has moved enough that the frozen operating point is suspect.
2. **Recall at the frozen policy < 0.50 in any period** — the model is missing more
   than half the frauds at the deployed cutoff.

If either fires, retrain. On the `dev` run neither fires
(`retraining_trigger -> {'retrain': False, 'reasons': []}`), which matches the
published "no degradation observed" drift line.

**This "no retrain" verdict is close to guaranteed on this run and is not
evidence the trigger works under real drift.** The monitoring table ends at the
validation boundary — there is **no test or post-freeze month** — so no period can
show the degradation the rules are designed to catch. The trigger logic is
exercised only on **synthetic series in `tests/test_monitoring.py`** (PSI streak,
single-spike non-trip, recall-floor trip, stable series), never on observed drift.
Re-test the trigger against a post-freeze window before trusting the verdict.

## What 8E says about *how* to retrain

On the dedicated `drift_lab` (a scenario-mix shift at day 120 of 240), after the
cold-start fix:

- **Naive drift metrics are dominated by the tenure clock.** All-features
  adversarial AUC `0.9821` collapses to behavioural `0.6995` once the four
  monotonic tenure features are removed. Quote **0.6995**; the `0.9821` figure is
  almost entirely time, not fraud drift.
- **Concept drift is mild; retraining recovers ~0.04 PR-AUC** (frozen `0.9475` →
  expanding `0.9878` in the last window).
- **A rolling window is enough.** `sliding_60d` matches or beats `expanding` while
  training on ~1/4 the rows, so the plan is a **rolling ~60-day window**, not
  unbounded accumulation. `recency_weighted` under-performed the hard cutoff.

## Policy

- **Weekly / monthly** monitoring job (this repo: `make monitor`); alert on the two
  rules above, plus a dashboard review when any key-feature PSI enters the `> 0.25`
  band.
- **On trigger:** retrain XGBoost on the last ~60 days of matured rows, re-fit the
  calibrator on a held-out slice of the fresh window, re-derive the cost-based
  policy threshold, and re-run the calibration + cost checks before promoting.
- **Guardrail:** the published test window is immutable. A retrain creates a new
  model version and a new forward window; it never re-scores the frozen test set
  (`docs/final_results.md`).
- **Caveat:** the ~0.04 retraining gain and the mild-drift conclusion are properties
  of *this* simulator's scenario-mix change, not established facts about real fraud.
  Re-test on real data before trusting the magnitudes
  (`docs/experiments/8E_drift.md` caveats).
