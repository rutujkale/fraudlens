# Part 11 progress

**Status:** implemented; real run **not executed** — IEEE-CIS files absent (**pre-registered before data**).
**Committed:** (pending) · **Pushed:** no

> **Part 11 is NOT complete until a real IEEE-CIS run finishes.** Today only the loader/adapter,
> the pre-registration, the R1–R5 runner and its synthetic-frame tests exist; no external result
> file exists yet.

## Freeze notes (Part 12)

- **Config hash changed.** Adding `common.external` changed the config hash for **every** profile
  (dev = `4279ce47d91d0b9e77efeeeeacba0a1f20e081d58a7460e4e055a23ab2a833ae`). Experiment rows logged
  before this part keep their old hashes (history); every new run uses the new hash. This is expected
  and recorded, not a bug.
- **No further `config/` edits before the Part 12 test-window freeze.** The config hash is the freeze
  anchor: any later change to `config/config.yaml` must be recorded alongside the Part 12 result.

## Preflight: is the data present?

| file | present |
|---|---|
| `data/external/ieee_cis/train_transaction.csv` | **NO** |
| `data/external/ieee_cis/train_identity.csv` | **NO** |

`data/external/` did not exist; `git ls-files data/external` → 0 tracked files. So R1–R5 cannot
run yet. This part ships the loader/adapter, the pre-registered decisions, the runnable R1–R5
runner, synthetic-frame tests, and the clean-exit path. R1–R5 execute only after the two Kaggle
CSVs are placed in `data/external/ieee_cis/`.

## Pre-registration (fixed **before** any external row was seen)

Everything below is declared here and in `config/config.yaml` (`common.external`) and frozen by
code/tests in the Part 11 implementation commit, so no choice is tuned after looking at the data.

- **Pseudo-customer `uid`:** string join of `card1, card2, card3, card5`. `addr1` is **not** a uid
  field — it is the location proxy. The heuristic is noisy (shared cards, missing values); `uid` is
  never treated as a true customer.
- **Pseudo-timestamp:** `start_date (2017-11-30) + TransactionDT seconds`. The absolute date is
  arbitrary; only ordering and spacing matter, so all time-of-day features are meaningless here.
- **External split:** chronological 60 / 20 / 20 of the external dataset with a **3-day embargo**
  (  `internal axes`, separate from FraudLens). The IEEE test window is evaluated **once**, here.
- **Device semantic mismatch:** IEEE `DeviceInfo` is not a bank device id and is absent for many
  rows, so the device features are NaN exactly where it is missing; the per-feature NaN share is
  written to `results/external_feature_nan_share.csv` and the mismatch is documented rather than
  smoothed over.
- **Availability contract** (derived from `features/registry.py`):
  - structurally **missing** (8): `new_merchant_flag`, `merchant_freq_30d`, `unique_recipients_1h`,
    `unique_recipients_24h`, `is_new_recipient`, `distance_from_home_km`, `distance_from_prev_km`,
    `travel_speed_kmh` — no merchant id, no recipient id, no coordinates ( `addr1` is region-only).
  - **excluded from transfer** (7, available but semantically incomparable): `merchant_category`,
    `transaction_type`, `channel`, `hour`, `is_night`, `is_weekend`, `day_of_week` (ProductCD ≠
    merchant category; category base rates differ; time-of-day is aliased by the arbitrary origin).
  - **shared subset `S`** = 53 − 8 − 7 = **38 features**. R2 trains and scores on `S` only; the
    categoricals are not label-encoded for R2 (they fall outside `S`).
- **R1–R5 reporting** (each: dataset, prevalence, schema coverage `|S|` vs 53, model, PR-AUC,
  **lift = PR-AUC / prevalence**, recall@1%, Brier/ECE, n):
  - **R1** real baseline (shared pipeline; LogReg + XGBoost) — IEEE val, then test.
  - **R2** sim-to-real transfer — POST-FIX **dev** XGBoost retrained on `S`; applied to IEEE test.
  - **R3** recalibration — sigmoid fit on the **earlier** IEEE val slice, evaluated on the later one;
    ranking unchanged; the transfer gap is reported, not hidden.
  - **R4** native reference (upper bound only) — XGBoost on all native IEEE columns, outside the
    FraudLens schema.
  - **R5** cost/policy — external cost config; F1-optimal vs cost-optimal (thresholds from IEEE
    val), plus capacity-aware.
- **Gap statement:** **primary gap = R1 − R2 on the IEEE test window** (PR-AUC). **Secondary** =
  dev-val `S` PR-AUC − IEEE-test R2 PR-AUC, explicitly labelled **selection-optimistic** (dev-val is
  R2's own selection window). **Lift is reported per dataset and never differenced across
  datasets.** The gap shows the *pipeline/evaluation/decision methodology* carries over; it does
  **not** validate simulator realism.
- **Two guards kept apart:** the IEEE test is gated by
  `results/test_access_log_external.csv` (caller `external:ieee_cis:R1-R5`); the FraudLens
  `results/test_access_log.csv` and the synthetic test window are never touched by external code.
- **R2 artifact isolation:** the transfer model is written to `artifacts/external/…`; it never
  touches `results/models_dev.json` or the dev artifact directory.
- **External cost config** (`common.external.costs`), **differs** from the synthetic one because
  IEEE amounts are USD-scale and prevalence/mechanisms differ, so the amount-dependent APPROVE-vs-
  REVIEW threshold `(R+F)/(αA+fee+F)` shifts on both counts: `fraud_loss_rate 0.85`,
  `fraud_fixed_fee 0`, `review_cost 5.0`, `friction_cost 10.0`, `review_catch_rate 0.90`,
  `stepup_cost 0.5`, `stepup_friction_cost 3.0`, `stepup_stop_rate 0.60`, `block_friction_cost 30.0`,
  `capacity_pct_of_daily_volume 0.01`. Scenarios reuse the Part 07 relative overrides on this base.
- **Raw-data hygiene:** `data/external/ieee_cis/` is gitignored and a test asserts
  `git ls-files data/external` is empty; only derived aggregates are written to `results/`.
- **Experiment ids:** E20 (R1), E21 (R2), E22 (R3), E23 (R4), E24 (R5), all
  `dataset_profile=dataset_version="ieee_cis"`.

## What was built

- `src/fraudlens/external/ieee_cis.py` — memory-efficient loader (`usecols`/dtypes, left-join
  identity), pseudo-timestamp, `uid` + `uid_stats`, canonical adapter, synthesised
  `customers`/`locations` entities (no coordinates), `STRUCTURALLY_MISSING`/`TRANSFER_EXCLUDED`/
  `SHARED_FEATURES`, `availability_rows()`, chronological `external_split`, `prepare_dataset`,
  `IeeeCisMissingError`.
- `src/fraudlens/external/runner.py` — R1–R5, one guarded test block, gap table, external cost
  policy, figures, E20–E24 logging, `SyntheticRefs`/`load_synthetic_refs`, `native_matrix`,
  `best_cost_threshold`. Test seams: `out_dir`, `figures_out`, `log_experiments`, `save_artifacts`.
- `scripts/run_external_validation.py` — thin CLI; prints `skipped: data missing` (+ instructions)
  and exits 0 (2 with `--strict`) when the files are absent.
- `tests/test_ieee_adapter.py` — 12 tests on a **synthetic fake IEEE frame** (CI needs no real
  files): schema mapping, `TransactionDT`→timestamp, uid construction/stats, availability mask,
  masked-features-NaN, no label leakage, `encode_features` regression (synthetic codes unchanged +
  IEEE `ProductCD` added), default split boundaries unchanged, native matrix, clean-exit CLI,
  end-to-end runner on a fake frame, and no tracked files under `data/external`.
- Backward-compatible edits: `build_features` tolerates a `locations` frame without
  `latitude`/`longitude`; `splits.chronological_window_split` generalises `make_temporal_split`
  (identity for the default path, tested); `models/xgb.py` `_CAT_MAPS` gains IEEE `ProductCD`
  (append-only, synthetic codes unchanged); `config.py` + `config/config.yaml` add `common.external`;
  `paths.external_data_dir`; `.gitignore` explicit external entries; `Makefile` `external` target.

## Commands run

```powershell
.venv\Scripts\python.exe -c "from fraudlens.config import load_config; load_config('dev')"   # external block parses
.venv\Scripts\python.exe -m pytest tests\test_ieee_adapter.py -q     # 12 passed
.venv\Scripts\python.exe scripts\run_external_validation.py --profile dev   # skipped: data missing (exit 0)
.venv\Scripts\python.exe -m ruff check .                             # All checks passed
.venv\Scripts\python.exe -m pytest -q                                # 213 passed
```

Actual summaries: adapter/runner tests 12/12; full suite **213 passed** (was 201; +12);
`ruff` clean; clean-exit prints `skipped: data missing` and exits 0; no `results/external_*` files
written (no data). `data/external/ieee_cis/` created empty (gitignored).

## Deviations (from prompt)

- **uid fields** are `card1, card2, card3, card5` (prompt's example was `card1, card2, addr1`);
  `addr1` is kept as the location proxy, per the approved change.
- **R4 runs** as an upper-bound reference (prompt marked it optional).
- **E20–E24** (prompt said "E20–E23"); there are five experiments R1–R5. No id collisions existed
  (max prior id was E18).
- **`make_temporal_split` refactor:** split into `chronological_window_split` with explicit
  parameters, keeping the default path byte-identical (regression test added).
- **`build_features` change:** minimal guard so a regions-only `locations` frame yields NaN
  geographic features instead of raising (test added).

## Bugs / what_went_wrong

Two entries appended to `docs/what_went_wrong.md` (both caught by the new fake-frame tests before
any real data was loaded): structurally-missing `unique_recipients_*` computing as **0** instead of
NaN (misleading constant) — fixed by enforcing the availability contract in `prepare_dataset`; and a
`KeyError` when `build_features` mapped customer home locations against an empty coordinate map —
fixed with `.get(h)`.

## Open items

- R1–R5 results are **pending the manual Kaggle download**. After the files are placed, run
  `python scripts/run_external_validation.py` (or `make external`) and finalise the tables in
  `docs/external_validation.md`; the gap table is finalized in Part 12 against the locked synthetic
  test window.
- `cat results/external_validation.csv` will exist only after that run.
