# Part 03 — Data Validation, EDA, and the Separability Gate

## Goal
(1) Automated data-quality validation. (2) EDA that documents the simulator honestly. (3) A **separability gate** that proves the data is not trivially solvable, before any model work. If the gate fails, the simulator is redesigned. Do not move on until it passes.

## Prerequisites
Part 02 complete; `data/raw/dev/` exists.

## Tasks

### 3A — Validation module (`data_quality/validation.py`, `scripts/validate_data.py`)
`validate_dataset(tables: dict[str, DataFrame]) -> ValidationReport` with named checks, each returning pass/fail, count of offending rows, and examples:
- duplicate IDs (all tables); nulls in required columns; amount ≤ 0 or absurdly large (> 1e7); timestamps outside the window or unparsable; invalid category / type / channel enums; orphan foreign keys; label table covers every transaction exactly once; no label columns in `transactions`; `label_available_at ≥ timestamp`; fraud rate sanity (0 < rate < 5%).
- Output `reports/data_validation_<profile>.json` and exit code ≠ 0 on failure so CI can use it.
- Tests in `tests/test_validation.py`: build a clean tiny frame, then inject each defect type and assert the matching check fails.

### 3B — EDA script (`scripts/eda.py --profile dev`)
Generate figures into `reports/figures/eda_<profile>_*.png` and a generated `docs/eda_summary.md` (all numbers computed by code):
1. Class balance overall and over time (weekly fraud rate); transactions per day; per-archetype volume.
2. Amount distributions: legit vs fraud (log scale), and by archetype.
3. Hour-of-day distribution: legit vs fraud vs hard negatives.
4. Fraud scenario shares by period (A vs B) — visual proof of drift; stealth distribution by period.
5. Hard negatives vs fraud overlap on raw signals (amount ratio to the customer's baseline, new-device indicator via device first-seen, hour, location distance from home).
6. Per-scenario signal profile table: median amount ratio, share on new device, median distance from home, median transactions in 1 hour (use a quick, approximate groupby for EDA only; the rigorous features come in Part 05).
7. Data dictionary stub in `docs/data_dictionary.md` for raw tables (feature dictionary is generated in Part 05).

### 3C — Separability gate (`scripts/separability_gate.py`)
Purpose: show no single cheap signal solves the problem, and that depth is needed.
- Compute a **small set of crude, leakage-safe-by-construction tx-level signals** using only the row itself and static customer attributes: amount / customer baseline_amount, hour deviation from `active_hour_mean`, distance from home, `device_id ∉ known devices` (using the static `customer_devices` table), category not in preferred set. (These are NOT the Part 05 features, and are used only for this gate.)
- Report for each: ROC-AUC and PR-AUC vs `is_fraud` on the whole dataset.
- Fit depth-1 stumps and a depth-3 decision tree on these crude signals with a temporal split; report PR-AUC.
- Fit an unconstrained gradient-boosted model on the same crude signals for reference.
- **Gate conditions** (write them as assertions with thresholds in config so they can be tuned transparently):
  - best single-signal ROC-AUC < 0.85 and PR-AUC < 0.20;
  - depth-3 tree PR-AUC clearly below the boosted model (shows interactions matter);
  - hard-negative rows score on average at least as "suspicious" as the median fraud row on at least two of the crude signals (shows overlap exists).
- Write `results/separability_gate_<profile>.json` and a short `docs/separability_gate.md` with the observed values and PASS/FAIL.

### 3D — If the gate fails
Do **not** fiddle with models. Report which signal is too strong, adjust simulator parameters (e.g. widen stealth, add overlap, reduce deterministic coupling), regenerate, re-run. Log the failure and the fix in `docs/what_went_wrong.md` ("Simulator shortcut" entry). Re-run until the gate passes. Remember: the aim is *realistic difficulty*, not a particular model score.

## Acceptance criteria
```bash
pytest tests/test_validation.py -q
python scripts/validate_data.py --profile dev        # exit 0
python scripts/eda.py --profile dev
python scripts/separability_gate.py --profile dev    # PASS
```
Progress file lists: gate metrics, any simulator changes made, and the figures produced.

## Do not
Train the real models here. Use the gate's crude signals as model features later. Tune simulator parameters to hit a target model score.
