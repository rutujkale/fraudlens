# Part 10 — SHAP Explainability & Subgroup Fairness Audit

## Goal
(1) Give every prediction an honest, human-readable explanation. (2) Check whether error rates and calibration differ across customer subgroups. (3) Document what SHAP can and cannot say. Validation data only; the test window is not touched.

## Prerequisites
Parts 06–07 complete (final XGBoost, calibrator, policy artifacts). Part 09 decision on the anomaly feature made (if adopted, the model includes it; otherwise ignore it here).

## Tasks

### 10A — SHAP engine (`explain/shap_explainer.py`)
- `shap.TreeExplainer` on the XGBoost model, **raw margin (log-odds) output**. Cache the explainer in the model artifact folder (or rebuild on load), expose:
  - `explain_global(X_sample) -> DataFrame` (mean |SHAP| per feature) from a stratified subsample of validation (e.g. all frauds up to 5k + 15k legit);
  - `explain_local(x_row) -> list[Contribution]` with feature, value, shap_value, direction; top-k positive and negative drivers.
- **Honesty rule:** SHAP explains the raw model margin, which is monotonically (not linearly) related to the calibrated probability after calibration. Never present SHAP values as points on the 0–100 risk score. Display either (a) the ordered list of drivers with direction, or (b) "share of upward push": each positive driver's SHAP divided by the sum of positive SHAP values, labeled exactly as that. Document this in `docs/methodology.md` (stub now, finish in Part 15).
- Sanity tests: for a sample, `sum(shap) + expected_value` reproduces the raw margin within tolerance; SHAP of the random-noise feature (if you kept it from 8D) is near zero; local explanation is deterministic.

### 10B — Human-readable reasons (`explain/reasons.py`)
A template map from feature → sentence given the **raw feature value** and the customer's baseline, for the top ~25 features, e.g.:
- `amount_ratio_to_30d_mean` → "Amount is 11.4× this customer's 30-day average"
- `tx_count_1h` → "9 transactions in the previous hour"
- `is_new_device` → "First time this customer uses this device"
- `travel_speed_kmh` → "Implied travel speed of 1,240 km/h since the previous transaction"
- `hour_deviation`, `distance_from_home_km`, `new_category_flag`, `count_near_threshold_24h`, `unique_recipients_1h`, `accounts_seen_on_device`, etc.
Provide a safe fallback for unmapped features (feature name + value). `reasons_for(x_row, contributions, k=5) -> list[str]` returns the top positive drivers as sentences and the top mitigating factors separately ("Mitigating: known device, usual location"). Unit-test the formatters (units, rounding, NaN handling, cold-start customers).

### 10C — Global analysis figures
Mean |SHAP| bar chart; beeswarm summary; dependence plots for the top 3 features, with the interaction feature coloring. Add a short generated table linking top SHAP features to scenarios (mean SHAP of each top feature by `fraud_scenario`, labels used for analysis only). Cross-reference with Experiment 8D (importance triangulation) and note agreements/disagreements. Save to `reports/figures/shap_*.png` and `results/shap_global.csv`.

### 10D — Local explanation gallery
Pick 6 validation cases (fixed seed, chosen by category, not by looks): a clear true positive, a subtle (high-stealth) true positive, a missed fraud (false negative), a hard-negative false positive (e.g. travel), a high-amount legitimate true negative, a cold-start customer. For each: calibrated probability, action under the policy, top reasons, a waterfall figure. Write `docs/explanation_gallery.md`. The false-negative and false-positive cases should be discussed candidly (what the model could not see).

### 10E — Fairness / subgroup audit (`explain/fairness.py`, `scripts/fairness_audit.py`)
- Model inputs exclude `age`, `income_band` (assert in a test that they are not in `FEATURE_COLUMNS`). They are **audit attributes** only, together with `archetype` and `home_region_group`.
- Groups: age band (<25, 25–34, 35–49, 50–64, 65+), income band, archetype, region group.
- On `val_thr` using the final policy, per group: n, fraud prevalence, **flag rate**, precision, recall, **false-positive rate**, average calibrated probability vs observed fraud rate (calibration gap), Brier, and PR-AUC if enough positives. Add cluster-bootstrap CIs; suppress/mark metrics for groups with fewer than ~30 positives or ~1,000 rows.
- Report max/min ratios (e.g. FPR ratio between best and worst group) and which differences are statistically distinguishable from sampling noise.
- **Proxy check:** report whether model features correlate strongly with the audit attributes (e.g. archetype-driven features like `baseline amount`-related ratios) and whether disparities are better explained by legitimate behavioral differences. State clearly that, because the data is simulated, subgroup differences reflect the simulator's assumptions (age/income/archetype correlations), not real-world demographics.
- Write `docs/fairness_audit.md` (generated tables + candid discussion + "what I would examine with real data": group definitions, legal frameworks, intersectional groups, post-processing options) and `results/fairness_<profile>.csv`. Do not claim regulatory compliance.

### 10F — Tests
`tests/test_explain.py`: additivity check, reason formatters, top-k ordering, no label fields used, performance (explaining one row < ~50 ms on dev model — record the actual figure; used again in Part 13).

## Acceptance criteria
```bash
pytest tests/test_explain.py -q
python scripts/run_shap.py --profile dev
python scripts/fairness_audit.py --profile dev
ls reports/figures | grep shap
```
Progress file: top-10 features by mean |SHAP|, single-row explanation latency, the six-case gallery summary, fairness findings with CIs, limitations.

## Do not
Present SHAP as probability points. Include age/income as model inputs. Claim bias absence or compliance. Cherry-pick gallery examples to look good.
