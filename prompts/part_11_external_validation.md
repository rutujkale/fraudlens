# Part 11 — External Validation on a Real Dataset (IEEE-CIS) & Sim-to-Real Gap

## Goal
Show that the *methodology* works on real data and quantify the **sim-to-real gap**. This answers the main credibility objection to synthetic data: "your model learned your own simulator."

## Prerequisites
Parts 05–07 complete. **Manual step for you:** download `train_transaction.csv` and `train_identity.csv` from the Kaggle competition *IEEE-CIS Fraud Detection* (requires a free Kaggle account and accepting the rules) and place them in `data/external/ieee_cis/`. The data is not redistributed and must not be committed. If the files are missing, scripts must exit with a clear message explaining how to obtain them, and the rest of the project must still work.

## Known facts about the dataset (verify in code, do not assume)
~590k transactions, ~3.5% fraud; `TransactionDT` is seconds from an unknown reference (covers ~6 months); `TransactionAmt`, `ProductCD`, `card1–card6`, `addr1/addr2`, `dist1/dist2`, email domains, count features `C1–C14`, timedelta features `D1–D15`, match flags `M1–M9`, Vesta features `V1–V339`; identity table with `DeviceType`, `DeviceInfo`, `id_01–id_38` (present for only a subset of transactions). There is no customer ID: any "customer" is a heuristic. Print the actual shape and fraud rate at load and record them.

## Tasks

### 11A — Loader and adapter (`external/ieee_cis.py`)
- Load with memory-efficient dtypes; merge identity (left join on `TransactionID`).
- Build a **pseudo-timestamp**: `start_date + TransactionDT seconds` (document that the absolute date is arbitrary; only ordering and spacing matter).
- Build a **pseudo-customer** `uid` from stable card/address fields (e.g. `card1`, `card2`, `addr1` combined) — state clearly in the docs that this heuristic is noisy and imperfect, and report the number of uids and the distribution of transactions per uid.
- Adapter maps IEEE columns to the **canonical transactions schema** used by the feature module: `customer_id=uid`, `timestamp`, `amount=TransactionAmt`, `merchant_category=ProductCD`, `device_id` from `DeviceInfo` if present (else null), `location_id` from `addr1` (region proxy; no coordinates ⇒ geographic distance features unavailable), `recipient_id` null, `transaction_type`/`channel` set to constants. A **feature availability mask** declares which Part 05 features are computable (frequency, amount-deviation, recency, merchant/category novelty, hour/day, device novelty where device is present) and which are structurally missing (distance, travel speed, recipients, accounts-on-device). Missing features are set to NaN for the transfer experiment and **dropped** for the native experiment. Reuse `build_features` unchanged (it must tolerate absent optional columns; if it doesn't, make the minimal backward-compatible change and add a test).
- Tests (`tests/test_ieee_adapter.py`, using a small synthetic fake IEEE-shaped frame so CI doesn't need the real files): schema mapping, timestamp monotonic with `TransactionDT`, uid construction, availability mask, no label leakage into features.

### 11B — Experiments (`scripts/run_external_validation.py`)
Use a chronological split of IEEE-CIS (e.g. first 60% train / next 20% validation / last 20% test **of the external dataset**; test is used here, it is a *separate* dataset from the FraudLens test window, and is evaluated once via its own guard-log entry). Apply an embargo of a few days.
1. **R1 — Real-data baseline with shared features:** train LogReg and XGBoost on the IEEE train window using only features computable via the shared pipeline; evaluate PR-AUC / recall@top-K on the IEEE validation and then test windows. Report **lift over base rate** (PR-AUC / prevalence) because prevalence differs from the simulator.
2. **R2 — Sim-to-real transfer:** apply the simulator-trained XGBoost (restricted to the shared feature subset, retrained on synthetic data using only those features so schemas match) to IEEE test data. Report PR-AUC and the gap versus its synthetic test performance (use the synthetic **validation** figure here, since the synthetic test is locked until Part 12; the gap table is finalized in Part 12).
3. **R3 — Recalibration/threshold adjustment:** keep the R2 ranking model but recalibrate on a small IEEE validation slice (sigmoid) and show calibration before/after — the ranking is unchanged but probabilities become meaningful. Do not hide the original transfer gap.
4. **R4 — Native reference (optional):** XGBoost on all native IEEE columns (V/C/D/M features) as an upper-bound reference, clearly labelled as using features outside the FraudLens schema.
5. Apply the cost/policy framework (Part 07) to the best real-data model with a documented cost configuration (state it differs from the synthetic one) and report the F1-optimal vs cost-optimal comparison — to show the decision methodology generalizes.
6. Leakage check on the real data: run the truncation-invariance test on a sample of IEEE rows through the adapter.

### 11C — Sim-to-real analysis (`docs/external_validation.md`, generated tables)
A table: dataset, prevalence, schema coverage, model, PR-AUC, lift over base rate, recall@1% — for synthetic vs IEEE. Discussion with hypotheses clearly labelled: schema mismatch, pseudo-customer noise, different fraud mechanisms, simulator overlap/stealth assumptions, label differences. State what this does and does **not** prove (it does not validate the simulator's realism; it shows the pipeline, evaluation protocol and decision framework carry over). Log findings as E20–E23.

### 11D — Figures
PR curves (real vs synthetic), calibration before/after on IEEE, feature availability heatmap, lift-vs-K for both datasets.

## Acceptance criteria
```bash
pytest tests/test_ieee_adapter.py -q
python scripts/run_external_validation.py           # exits cleanly with instructions if files are missing
cat results/external_validation.csv
```
Progress file: dataset shape/prevalence actually observed, uid statistics, feature availability, table of R1–R5 results, the gap statement with caveats.

## Do not
Commit the dataset or any derived file containing raw rows. Claim state-of-the-art performance. Describe PaySim as real data (it is itself synthetic). Treat uid as a true customer. Compare raw PR-AUC across datasets without lift/prevalence context.
