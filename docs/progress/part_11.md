# Part 11 progress

**Status:** implemented; **real IEEE-CIS run complete** (R1–R5 + gap + truncation check).
**Committed:** yes (`dc9c8f6`, `942c3e3`, `1dd09d7`) · **Pushed:** yes

> **Part 11 is complete.** Pre-registration landed before any external row was seen; the run below
> is the second (post-R5-fix) real run. The first run's evidence is preserved verbatim under
> `results/pre_fix/` and explained in `docs/what_went_wrong.md`.

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
| `data/external/ieee_cis/train_transaction.csv` | yes (590,540 rows, 13 adapter cols, 3.499 % fraud) |
| `data/external/ieee_cis/train_identity.csv` | yes |

`data/external/` is gitignored (`git ls-files data/external` → 0 tracked files); only derived
aggregates are written to `results/`. External split 60/20/20 + 3-day embargo:
train=380,815 / val=95,763 / test=96,139 rows (`uids=14,845`, single-tx uid share 0.275).

## Pre-registration (fixed **before** any external row was seen)

Everything below is declared here and in `config/config.yaml` (`common.external`) and frozen by
code/tests in the Part 11 implementation commit, so no choice is tuned after looking at the data.

- **Pseudo-customer `uid`:** string join of `card1, card2, card3, card5`. `addr1` is **not** a uid
  field — it is the location proxy. The heuristic is noisy (shared cards, missing values); `uid` is
  never treated as a true customer.
- **Pseudo-timestamp:** `start_date (2017-11-30) + TransactionDT seconds`. The absolute date is
  arbitrary; only ordering and spacing matter, so all time-of-day features are meaningless here.
- **External split:** chronological 60 / 20 / 20 of the external dataset with a **3-day embargo**
  (  `internal axes`, separate from FraudLens). The IEEE test window is evaluated **twice, both
  logged** in `results/test_access_log_external.csv`: the initial run (which exposed the R5
  uncalibrated-score bug) and the post-R5-fix re-run. Both accesses are recorded with a `reason`;
  the first run's R1–R3 rows are preserved verbatim under `results/pre_fix/`.
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

## Real run results (post-fix)

Full generated tables live in [`docs/external_validation.md`](../external_validation.md)
(`scripts/external_validation_report.py`). Headline (IEEE test window, prevalence 0.0345):

| model | PR-AUC | lift@1% | recall@1% |
|---|---|---|---|
| R1_logreg | 0.1135 | 6.73 | 0.067 |
| R1_xgb | 0.1362 | 7.78 | 0.078 |
| R2 transfer dev XGB (S=38) | 0.0580 | 1.48 | 0.015 |
| R3 transfer + sigmoid | 0.0580 | 1.48 | 0.015 |
| R4 native (upper bound) | 0.4905 | 25.50 | 0.255 |

- **Primary gap = R1 − R2 = 0.0782** (simulator-trained signal partially transfers). Secondary
  dev-val `S` − IEEE test R2 = **0.8197**, labelled selection-optimistic; its raw-PR-AUC baselines
  differ by prevalence (dev ≈ 0.6 % vs IEEE 3.45 %), so it is **not** trustworthy as a magnitude
  and **must not feed any Part 12 headline number**.
- **R3** sigmoid: test Brier 0.0367 → 0.0335, ECE 0.0274 → 0.0075 (ranking unchanged).
- **R5** cost policy (IEEE test): approve_all 423,984.89; f1_optimal 396,497.56 (6.48 %);
  cost_optimal 403,048.47 (4.94 %); **amount_aware 269,018.95 (36.55 %)**; capacity_aware
  274,281.28 (35.31 %, 1,511 demoted).
- **Truncation check**: 3 cuts, max abs diff **0.0**, passed (adapter + real `build_features`).
- **Calibration matters on real data.** The first run's R1 raw scores (class-weighted, ECE 0.4247)
  had to be sigmoid-calibrated before the amount-aware policy could use them (ECE → 0.0018). This
  contrasts with the **Part 07 null result** on the near-separable synthetic data, where class
  weighting already left the raw probabilities calibrated (xgb raw ECE 0.00032) and calibration
  mildly *worsened* Brier. The direction flips between the synthetic and real regimes.
- **The first-run R1–R3 numbers existed before the R5 fix** and are preserved byte-for-byte in
  `results/pre_fix/`; only R5 (and R4's column set) changed in the re-run.

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
.venv\Scripts\python.exe -m pytest tests\test_ieee_adapter.py -q      # adapter/runner tests
.venv\Scripts\python.exe scripts\run_external_validation.py            # real run, exit 0
.venv\Scripts\python.exe scripts\external_validation_report.py         # fills docs tables
.venv\Scripts\python.exe -m ruff check .                               # All checks passed
.venv\Scripts\python.exe -m pytest -q                                  # full suite, 0 warnings
```

Actual summaries: real run **EXITCODE=0** in ~7 min; gap 0.0782; R5 amount_aware 36.55 % saving;
truncation `passed=True max_abs_diff=0`; R1–R3 rows in `external_validation.csv` reproduce the
`results/pre_fix/` values **exactly** (byte-for-byte on those rows) — only R4 changed, from the
13-column shortcut to the full native column set (test PR-AUC 0.1607 → 0.4905), as intended for an
upper-bound reference. `ruff` clean; full suite green with the imputer warning suppressed.

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
- **R5 calibration added (post-first-run fix):** amount-aware R5 now uses an R1 sigmoid calibrator
  fit on an earlier IEEE val slice (evaluated on the later slice, never test). Pre-registered cost
  values/thresholds/capacity unchanged. See `what_went_wrong.md`.
- **R4 uses the full native column set (V/C/D/M + identity)** via `load_ieee_cis_native`, not just
  the 13 adapter columns; it is an upper-bound reference only and nothing is selected on it.
- **`scripts/external_validation_report.py`** generates the `docs/external_validation.md` tables
  from `results/` (numbers-integrity rule); new file, not in the original prompt.

## Bugs / what_went_wrong

Five entries appended to `docs/what_went_wrong.md`: the two pre-data adapter bugs (structurally-
missing `unique_recipients_*` computed as 0 → masked; `build_features` `KeyError` on a regions-only
`locations` frame); and three from the real run — the `pd.NA` crash in the string columns, the CLI
"no data" test that ran the full pipeline and polluted tracked results, and the **R5 uncalibrated-
score bug** (raw class-weighted R1 scores fed to the amount-aware policy → −214 % saving), fixed
with an R1 sigmoid calibrator. The `enforce_capacity` REVIEW-only design gap is logged as an open
item (guarded by a test, not fixed).

## Part 12 note (do not start Part 12 here)

Part 12 (locked synthetic test window) will run on the **`dev`** profile, **not `main`** — `main`
was never generated. Recorded here so the deviation is not lost; `part_12.md`/`part_15.md` will
repeat it when those parts run.

## Open items

- Part 11 complete. Results are in `results/external_*.csv` and generated into
  `docs/external_validation.md`.
- `enforce_capacity` caps only REVIEW (BLOCK/STEP_UP pass through) — a design gap for amount-aware
  policies; logged, not changed here (the pre-registered Part 11 policy is REVIEW-capped).
- IEEE-CIS raw data is gitignored; only derived aggregates are tracked.
