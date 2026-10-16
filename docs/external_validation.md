# External validation on IEEE-CIS & the sim-to-real gap (Part 11)

> **Status: awaiting the manual Kaggle download.** The IEEE-CIS files
> (`train_transaction.csv`, `train_identity.csv`) are not in `data/external/ieee_cis/`, so no
> R1–R5 numbers exist yet. This file is the pre-registered analysis plan; the tables below are
> produced by `python scripts/run_external_validation.py` and written to `results/external_*.csv`.
> Nothing here is invented — every number is read from those files after the run. See
> [progress/part_11.md](progress/part_11.md) for the frozen decisions.

## Why this exists

The main credibility objection to a synthetic-data project is *"your model learned your own
simulator."* External validation answers it directly: it shows the **methodology** (point-in-time
features, temporal validation, calibration, cost-based decisions) carries over to real data, and it
**quantifies the sim-to-real gap**.

## Pre-registration (frozen before any external row was seen)

- `uid` = `card1|card2|card3|card5` (never a true customer); `addr1` = location proxy.
- pseudo-timestamp = `2017-11-30 + TransactionDT` (absolute date arbitrary; ordering/spacing only).
- external split 60/20/20 with a 3-day embargo; IEEE test evaluated **once**.
- shared subset `S` = 38 of 53 features; 8 structurally missing, 7 excluded from transfer
  (see `progress/part_11.md`).
- **primary gap = R1 − R2 on the IEEE test window**; secondary = dev-val `S` − IEEE-test R2,
  labelled **selection-optimistic**; **lift reported per dataset, never differenced**.
- separate guard log `results/test_access_log_external.csv`.

## Tables (generated on run)

`results/external_validation.csv` — one row per model/window with `prevalence`, `pr_auc`, `roc_auc`,
`brier`, `ece`, `best_f1`, `lift_at_top_0.01`, `recall_at_top_0.01`:

| dataset | prevalence | schema coverage | model | PR-AUC | lift@1% | recall@1% |
|---|---|---|---|---|---|---|
| IEEE-CIS (test) | *(run)* | 38/53 | R1 XGBoost | *(run)* | *(run)* | *(run)* |
| IEEE-CIS (test) | *(run)* | 38/53 | R2 transfer dev XGB | *(run)* | *(run)* | *(run)* |
| IEEE-CIS (test) | *(run)* | 38/53 | R3 transfer + sigmoid | *(run)* | *(run)* | *(run)* |
| IEEE-CIS (test) | *(run)* | native | R4 native (upper bound) | *(run)* | *(run)* | *(run)* |

`results/external_gap.csv` — the gap statement (`primary_gap_pr_auc_r1_minus_r2_on_ieee_test`,
`secondary_gap_dev_val_minus_ieee_test` + note). `results/external_calibration.csv` — R3 Brier/ECE
before/after sigmoid. `results/external_policy.csv` — realized cost for approve-all / F1-optimal /
cost-optimal / capacity-aware. `results/external_feature_availability.csv` — the mask.
`results/external_feature_nan_share.csv` — per-feature NaN share (device features are NaN wherever
`DeviceInfo` is missing; the device semantic mismatch is documented, not hidden).
`results/external_uid_stats_ieee_cis.json` — n_uids, transactions-per-uid distribution, single-tx
uid shares.

## Gap statement (filled on run)

The primary gap is the drop in PR-AUC from a real-data model (R1) to the simulator-trained model
transferred with the same features (R2), both measured on the **IEEE test window**. A positive gap
means simulator-trained signal does not fully transfer; the secondary dev-val-vs-IEEE figure is
larger and **optimistic** because dev-val is R2's own selection window. Lift is reported per dataset
only.

## Hypotheses if the gap is large (each labelled)

- **Schema mismatch** — IEEE lacks merchant/recipient/coordinates; `S` is a subset.
- **Pseudo-customer noise** — the `uid` heuristic merges/splits real customers; history features
  computed per `uid` are noisy.
- **Different fraud mechanisms** — IEEE fraud patterns differ from the simulator's five scenarios.
- **Simulator overlap/stealth assumptions** — the simulator's separability gate may make fraud
  "easier" than reality.
- **Label/prevalence differences** — ~3.5% prevalence and different labelling rules vs the simulator.

## What this does and does **not** prove

- **Does:** the point-in-time feature pipeline, temporal protocol, calibration step and cost-based
  decision framework run on real data and produce meaningful lift.
- **Does not:** validate simulator realism, claim state-of-the-art, or treat `uid` as a true
  customer. IEEE-CIS is one dataset with one fraud taxonomy; the result is evidence, not proof.

## Figures (11D)

`reports/figures/external_pr_curves.png` (R1 vs R2), `external_calibration.png` (R3 before/after),
`external_feature_availability.png` (mask heatmap). Lift-vs-K across datasets is reported per
dataset, never overlaid as a single comparison.
