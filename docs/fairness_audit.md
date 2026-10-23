# Fairness / subgroup audit (Part 10E)

Audit attributes (`age_band`, `income_band`, `archetype`, `home_region_group`) are
**not model inputs** — `tests/test_explain.py` asserts they are absent from
`FEATURE_COLUMNS`. They are joined from `data/raw/dev/customers.parquet`
(and `locations.region_group`) for analysis only.

Alert definition: a transaction is **flagged** when the frozen policy's action is
not `APPROVE`. `flag_any` counts `STEP_UP`/`REVIEW`/`BLOCK`; `flag_strict` counts
only `REVIEW`/`BLOCK`. The two differ in burden: `STEP_UP` costs the customer a
step-up (Rs 40 friction, Rs 5 handling) while `REVIEW` costs the bank Rs 75 and
`BLOCK` imposes Rs 450 of friction on a legitimate customer.

This audit is run on **simulated** data: subgroup differences reflect the
simulator's own assumptions about age/income/archetype, not real demographics.
Nothing here is a statement about fairness in the real world, and it is not a
regulatory compliance claim.


**Policy:** amount-aware policy `cost_policy_v1` (scenario `balanced`), calibrator `sigmoid`, capacity 23/day. Alert = `action != APPROVE`; this fires on APPROVE vs STEP_UP/REVIEW/BLOCK.


## Suppressed groups per attribute (reported up front)

A group is counted under `suppressed_positives` when it has fewer than 30 frauds (recall/precision/PR-AUC are then not reported) and under `suppressed_rows` when it has fewer than 1,000 rows (calibration gap, Brier and bootstrap CIs are then not reported). `reported` counts groups with no suppression at all.

| attribute | groups | positives<30 | rows<1000 | fully reported |
|---|---|---|---|---|
| age_band | 4 | 2 | 1 | 2 |
| income_band | 4 | 2 | 0 | 2 |
| archetype | 5 | 3 | 0 | 2 |
| home_region_group | 6 | 3 | 1 | 3 |

## Primary view — `val_thr` (the policy's own selection slice)

Primary view — this is the slice the policy and its capacity were selected on (20,293 rows, 183 frauds), so per-group metrics are the honest operating numbers of the frozen policy.

All 183 frauds / 20,293 rows the policy was selected on; the calibrator was fitted earlier in validation, so the calibration gap and Brier here are out-of-sample.

| attribute | group | n | n_fraud | prevalence | flag_rate_any | fpr_any | fpr_strict | recall | precision | pr_auc | mean_p | calibration_gap | brier | suppressed_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| age_band | 25-34 | 7553 | 91 | 0.012 | 0.0229 | 0.0115 | 0.0027 | 0.956 | 0.503 | 0.967 | 0.0123 | 0.0003 | 0.00117 |  |
| age_band | 35-49 | 9179 | 61 | 0.0066 | 0.0326 | 0.027 | 0.006 | 0.869 | 0.177 | 0.819 | 0.0069 | 0.0003 | 0.002571 |  |
| age_band | 50-64 | 690 | 21 | 0.0304 | 0.0725 | 0.0448 | 0.006 | suppressed | suppressed | suppressed | 0.0299 | suppressed | suppressed | rows=690<1000; positives=21<30 |
| age_band | <25 | 2871 | 10 | 0.0035 | 0.0042 | 0.001 | 0.0007 | suppressed | suppressed | suppressed | 0.0032 | -0.0003 | 0.001298 | positives=10<30 |
| income_band | high | 3483 | 27 | 0.0078 | 0.0411 | 0.0336 | 0.0067 | suppressed | suppressed | suppressed | 0.0087 | 0.0009 | 0.001169 | positives=27<30 |
| income_band | low | 3560 | 65 | 0.0183 | 0.0264 | 0.01 | 0.002 | 0.908 | 0.628 | 0.957 | 0.017 | -0.0013 | 0.002869 |  |
| income_band | mid | 7340 | 65 | 0.0089 | 0.0199 | 0.0118 | 0.003 | 0.923 | 0.411 | 0.899 | 0.0091 | 0.0002 | 0.002169 |  |
| income_band | mid_high | 5910 | 26 | 0.0044 | 0.0255 | 0.0218 | 0.0049 | suppressed | suppressed | suppressed | 0.0049 | 0.0005 | 0.001513 | positives=26<30 |
| archetype | business_like | 3638 | 7 | 0.0019 | 0.0577 | 0.0559 | 0.011 | suppressed | suppressed | suppressed | 0.0033 | 0.0014 | 0.000869 | positives=7<30 |
| archetype | conservative | 1548 | 25 | 0.0161 | 0.0304 | 0.0151 | 0.0033 | suppressed | suppressed | suppressed | 0.0191 | 0.003 | 0.003903 | positives=25<30 |
| archetype | frequent_digital | 7374 | 37 | 0.005 | 0.0066 | 0.0019 | 0.0004 | 0.946 | 0.714 | 0.951 | 0.0052 | 0.0001 | 0.000803 |  |
| archetype | high_spend | 1148 | 23 | 0.02 | 0.0828 | 0.064 | 0.0178 | suppressed | suppressed | suppressed | 0.0217 | 0.0017 | 0.001409 | positives=23<30 |
| archetype | regular | 6585 | 91 | 0.0138 | 0.0202 | 0.0082 | 0.002 | 0.879 | 0.602 | 0.909 | 0.0124 | -0.0014 | 0.003402 |  |
| home_region_group | Central | 4847 | 49 | 0.0101 | 0.0262 | 0.0167 | 0.0031 | 0.959 | 0.37 | 0.948 | 0.0101 | -0 | 0.00178 |  |
| home_region_group | East | 2385 | 45 | 0.0189 | 0.0319 | 0.015 | 0.0021 | 0.911 | 0.539 | 0.974 | 0.0174 | -0.0015 | 0.002458 |  |
| home_region_group | North | 5851 | 62 | 0.0106 | 0.0303 | 0.0207 | 0.0052 | 0.919 | 0.322 | 0.921 | 0.0101 | -0.0005 | 0.002335 |  |
| home_region_group | North-East | 365 | 0 | 0 | 0.0247 | 0.0247 | 0.0082 | suppressed | suppressed | suppressed | 0.0046 | suppressed | suppressed | rows=365<1000; positives=0<30 |
| home_region_group | South | 2985 | 13 | 0.0044 | 0.0228 | 0.0192 | 0.0061 | suppressed | suppressed | suppressed | 0.0056 | 0.0012 | 0.002232 | positives=13<30 |
| home_region_group | West | 3860 | 14 | 0.0036 | 0.0199 | 0.0166 | 0.0026 | suppressed | suppressed | suppressed | 0.0047 | 0.0011 | 0.00085 | positives=14<30 |

## Per-action counts (burden, not just flag rates)

| attribute | group | n_approve | n_step_up | n_review | n_block |
|---|---|---|---|---|---|
| age_band | 25-34 | 7380 | 68 | 13 | 92 |
| age_band | 35-49 | 8880 | 199 | 36 | 64 |
| age_band | 50-64 | 640 | 28 | 3 | 19 |
| age_band | <25 | 2859 | 5 | 1 | 6 |
| income_band | high | 3340 | 95 | 14 | 34 |
| income_band | low | 3466 | 31 | 7 | 56 |
| income_band | mid | 7194 | 71 | 14 | 61 |
| income_band | mid_high | 5759 | 103 | 18 | 30 |
| archetype | business_like | 3428 | 164 | 25 | 21 |
| archetype | conservative | 1501 | 20 | 5 | 22 |
| archetype | frequent_digital | 7325 | 16 | 2 | 31 |
| archetype | high_spend | 1053 | 52 | 11 | 32 |
| archetype | regular | 6452 | 48 | 10 | 75 |
| home_region_group | Central | 4720 | 71 | 10 | 46 |
| home_region_group | East | 2309 | 31 | 6 | 39 |
| home_region_group | North | 5674 | 96 | 21 | 60 |
| home_region_group | North-East | 356 | 6 | 1 | 2 |
| home_region_group | South | 2917 | 42 | 10 | 16 |
| home_region_group | West | 3783 | 54 | 5 | 18 |

## Ratios and sampling noise

- `recall` (age_band) max/min ratio = 1.100 (high: age_band=25-34; low: age_band=35-49)
- `fpr_any` (age_band) max/min ratio = 2.341 (high: age_band=35-49; low: age_band=25-34)
- `flag_rate_any` (age_band) max/min ratio = 1.422 (high: age_band=35-49; low: age_band=25-34)
- `recall` (income_band) max/min ratio = 1.017 (high: income_band=mid; low: income_band=low)
- `fpr_any` (income_band) max/min ratio = 1.180 (high: income_band=mid; low: income_band=low)
- `flag_rate_any` (income_band) max/min ratio = 1.327 (high: income_band=low; low: income_band=mid)
- `recall` (archetype) max/min ratio = 1.076 (high: archetype=frequent_digital; low: archetype=regular)
- `fpr_any` (archetype) max/min ratio = 4.277 (high: archetype=regular; low: archetype=frequent_digital)
- `flag_rate_any` (archetype) max/min ratio = 3.040 (high: archetype=regular; low: archetype=frequent_digital)
- `recall` (home_region_group) max/min ratio = 1.053 (high: home_region_group=Central; low: home_region_group=East)
- `fpr_any` (home_region_group) max/min ratio = 1.386 (high: home_region_group=North; low: home_region_group=East)
- `flag_rate_any` (home_region_group) max/min ratio = 1.216 (high: home_region_group=East; low: home_region_group=Central)

Differences vs the reference group (largest by row count), paired cluster bootstrap over customers — 95% CI excluding 0 means the difference is not explained by sampling noise alone:

| attribute | group | metric | point | lo | hi | distinguishable |
|---|---|---|---|---|---|---|
| age_band | 25-34 | recall | 0.087191 | -0.043427 | 0.237885 | False |
| age_band | 25-34 | fpr_any | -0.015455 | -0.021417 | -0.01021 | True |
| age_band | 25-34 | flag_rate_any | -0.00967 | -0.020101 | 0.0005 | False |
| age_band | <25 | recall | 0.031148 | -0.125 | 0.254902 | False |
| age_band | <25 | fpr_any | -0.025931 | -0.030564 | -0.021918 | True |
| age_band | <25 | flag_rate_any | -0.028395 | -0.035766 | -0.021224 | True |
| income_band | mid_high | recall | -0.038462 | -0.15835 | 0.107874 | False |
| income_band | mid_high | fpr_any | 0.009933 | 0.004154 | 0.015883 | True |
| income_band | mid_high | flag_rate_any | 0.005659 | -0.002092 | 0.013757 | False |
| income_band | low | recall | -0.015385 | -0.211823 | 0.10317 | False |
| income_band | low | fpr_any | -0.001807 | -0.00686 | 0.003805 | False |
| income_band | low | flag_rate_any | 0.006513 | -0.008928 | 0.023391 | False |
| income_band | high | recall | 0.076923 | 0.014719 | 0.167296 | True |
| income_band | high | fpr_any | 0.021744 | 0.013523 | 0.030708 | True |
| income_band | high | flag_rate_any | 0.021166 | 0.007407 | 0.037622 | True |
| archetype | regular | recall | -0.066825 | -0.167814 | 0.077042 | False |
| archetype | regular | fpr_any | 0.006253 | 0.003556 | 0.008722 | True |
| archetype | regular | flag_rate_any | 0.013552 | 0.004435 | 0.023023 | True |
| archetype | business_like | recall | 0.054054 | 0 | 0.176471 | False |
| archetype | business_like | fpr_any | 0.053999 | 0.045279 | 0.064606 | True |
| archetype | business_like | flag_rate_any | 0.051079 | 0.040898 | 0.064241 | True |
| archetype | conservative | recall | 0.014054 | -0.109297 | 0.16129 | False |
| archetype | conservative | fpr_any | 0.013194 | 0.006074 | 0.020749 | True |
| archetype | conservative | flag_rate_any | 0.023717 | 0.008778 | 0.045143 | True |
| archetype | high_spend | recall | 0.054054 | 0 | 0.2 | False |
| archetype | high_spend | fpr_any | 0.062092 | 0.045151 | 0.078912 | True |
| archetype | high_spend | flag_rate_any | 0.076108 | 0.044335 | 0.123259 | True |
| home_region_group | Central | recall | 0.039829 | -0.089274 | 0.205417 | False |
| home_region_group | Central | fpr_any | -0.004055 | -0.01119 | 0.003071 | False |
| home_region_group | Central | flag_rate_any | -0.004049 | -0.018947 | 0.009875 | False |
| home_region_group | West | recall | 0.009217 | -0.167449 | 0.2 | False |
| home_region_group | West | fpr_any | -0.004088 | -0.011086 | 0.00329 | False |
| home_region_group | West | flag_rate_any | -0.010303 | -0.02335 | 0.000143 | False |
| home_region_group | South | recall | -0.073201 | -0.437007 | 0.194529 | False |
| home_region_group | South | fpr_any | -0.00155 | -0.010043 | 0.008061 | False |
| home_region_group | South | flag_rate_any | -0.007471 | -0.02086 | 0.00498 | False |
| home_region_group | East | recall | -0.008244 | -0.160394 | 0.183259 | False |
| home_region_group | East | fpr_any | -0.005772 | -0.013484 | 0.002092 | False |
| home_region_group | East | flag_rate_any | 0.001615 | -0.015811 | 0.021382 | False |

**15 of 39** differences are distinguishable from noise — and these are **uncorrected for multiple comparisons**. 39 tests at a 95% level would throw up roughly 5% false positives by chance alone, and the tests share bootstrap draws and reference groups, so the family-wise error rate is above 5%. Read them as **exploratory signals**: a confirmatory analysis would pre-register a smaller family (e.g. FPR per attribute) and apply Holm-Bonferroni or Benjamini-Hochberg. Note also that every group here was reported, not selected — no group was dropped for being unflattering.


## Exploratory view — full validation window

360 frauds / 40,585 rows. **Exploratory only**: the calibrator was fitted on the first half of this window (`val_cal`), so calibration gap and Brier would be in-sample for half the rows and are therefore excluded here; the policy capacity was fitted on `val_thr`.

| attribute | group | n | n_fraud | prevalence | flag_rate_any | fpr_any | fpr_strict | recall | precision | pr_auc | mean_p | calibration_gap | brier | suppressed_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| age_band | 25-34 | 15022 | 176 | 0.0117 | 0.0224 | 0.0115 | 0.0022 | 0.943 | 0.493 | 0.957 | suppressed | suppressed | suppressed |  |
| age_band | 35-49 | 18468 | 133 | 0.0072 | 0.0333 | 0.0268 | 0.0055 | 0.925 | 0.2 | 0.896 | suppressed | suppressed | suppressed |  |
| age_band | 50-64 | 1420 | 21 | 0.0148 | 0.0514 | 0.0379 | 0.005 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | positives=21<30 |
| age_band | <25 | 5675 | 30 | 0.0053 | 0.006 | 0.0016 | 0.0005 | 0.833 | 0.735 | 0.83 | suppressed | suppressed | suppressed |  |
| income_band | high | 7166 | 48 | 0.0067 | 0.0402 | 0.034 | 0.006 | 0.958 | 0.16 | 0.947 | suppressed | suppressed | suppressed |  |
| income_band | low | 7200 | 91 | 0.0126 | 0.0199 | 0.0083 | 0.002 | 0.923 | 0.587 | 0.946 | suppressed | suppressed | suppressed |  |
| income_band | mid | 14656 | 151 | 0.0103 | 0.0214 | 0.0121 | 0.0027 | 0.914 | 0.439 | 0.916 | suppressed | suppressed | suppressed |  |
| income_band | mid_high | 11563 | 70 | 0.0061 | 0.0272 | 0.0216 | 0.0041 | 0.943 | 0.21 | 0.933 | suppressed | suppressed | suppressed |  |
| archetype | business_like | 7377 | 12 | 0.0016 | 0.0602 | 0.0587 | 0.0102 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | positives=12<30 |
| archetype | conservative | 3180 | 46 | 0.0145 | 0.0286 | 0.015 | 0.0045 | 0.957 | 0.484 | 0.929 | suppressed | suppressed | suppressed |  |
| archetype | frequent_digital | 14508 | 97 | 0.0067 | 0.0074 | 0.0014 | 0.0004 | 0.897 | 0.813 | 0.92 | suppressed | suppressed | suppressed |  |
| archetype | high_spend | 2323 | 26 | 0.0112 | 0.0663 | 0.0557 | 0.0122 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | positives=26<30 |
| archetype | regular | 13197 | 179 | 0.0136 | 0.0199 | 0.0075 | 0.0015 | 0.922 | 0.627 | 0.94 | suppressed | suppressed | suppressed |  |
| home_region_group | Central | 9790 | 87 | 0.0089 | 0.0244 | 0.0165 | 0.003 | 0.908 | 0.331 | 0.896 | suppressed | suppressed | suppressed |  |
| home_region_group | East | 4628 | 52 | 0.0112 | 0.0292 | 0.019 | 0.0028 | 0.923 | 0.356 | 0.963 | suppressed | suppressed | suppressed |  |
| home_region_group | North | 11681 | 93 | 0.008 | 0.0277 | 0.0205 | 0.0044 | 0.925 | 0.265 | 0.916 | suppressed | suppressed | suppressed |  |
| home_region_group | North-East | 783 | 7 | 0.0089 | 0.0268 | 0.018 | 0.0039 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | rows=783<1000; positives=7<30 |
| home_region_group | South | 5861 | 44 | 0.0075 | 0.0227 | 0.0158 | 0.0041 | 0.932 | 0.308 | 0.902 | suppressed | suppressed | suppressed |  |
| home_region_group | West | 7842 | 77 | 0.0098 | 0.0264 | 0.0173 | 0.003 | 0.948 | 0.353 | 0.973 | suppressed | suppressed | suppressed |  |

## Proxy check — do model features encode the audit attributes?

Ordinal attributes (age/income bands) use Spearman rho; nominal attributes use eta-squared for numeric features and Cramer's V for the categorical ones. Pre-registered thresholds: |rho| >= 0.30, eta^2 >= 0.14, V >= 0.30. A strong correlation means the feature is a *proxy* for the attribute, which is how a behavioural difference can look like a demographic disparity.

| attribute | feature | statistic | value | threshold |
|---|---|---|---|---|
| age_band | max_amount_24h | spearman | 0.465959 | 0.3 |
| age_band | avg_amount_7d | spearman | 0.396552 | 0.3 |
| age_band | avg_amount_30d | spearman | 0.370122 | 0.3 |
| age_band | amount | spearman | 0.330932 | 0.3 |
| age_band | log_amount | spearman | 0.330932 | 0.3 |
| archetype | n_prior_tx | eta_squared | 0.770216 | 0.14 |
| archetype | active_days_30d | eta_squared | 0.768185 | 0.14 |
| archetype | avg_amount_30d | eta_squared | 0.72827 | 0.14 |
| archetype | avg_amount_7d | eta_squared | 0.649198 | 0.14 |
| archetype | log_amount | eta_squared | 0.634541 | 0.14 |
| archetype | active_days_7d | eta_squared | 0.556694 | 0.14 |
| archetype | tx_count_7d | eta_squared | 0.552218 | 0.14 |
| archetype | std_amount_30d | eta_squared | 0.545405 | 0.14 |
| archetype | tx_from_device_prior | eta_squared | 0.510368 | 0.14 |
| archetype | device_count_30d | eta_squared | 0.455764 | 0.14 |
| archetype | max_amount_24h | eta_squared | 0.35944 | 0.14 |
| archetype | amount | eta_squared | 0.329535 | 0.14 |
| archetype | amount_sum_24h | eta_squared | 0.320595 | 0.14 |
| archetype | minutes_since_prev_tx | eta_squared | 0.260368 | 0.14 |
| archetype | days_since_last_tx | eta_squared | 0.260368 | 0.14 |
| archetype | hour_deviation | eta_squared | 0.238721 | 0.14 |
| archetype | count_near_threshold_24h | eta_squared | 0.180768 | 0.14 |
| archetype | is_new_recipient | eta_squared | 0.168514 | 0.14 |
| archetype | tx_count_24h | eta_squared | 0.158767 | 0.14 |
| income_band | max_amount_24h | spearman | 0.350022 | 0.3 |
| income_band | avg_amount_30d | spearman | 0.309881 | 0.3 |
| income_band | avg_amount_7d | spearman | 0.308549 | 0.3 |
| income_band | std_amount_30d | spearman | 0.300951 | 0.3 |

(28 of 212 feature/attribute pairs are strong; the full table is in `results/fairness_proxy_dev.csv`.)


## What I would examine with real data

- **Group definitions:** age/income bands are a modelling choice; real audits must
  pre-register them (and intersectional groups, e.g. age x region) before looking at
  outcomes, or the bands become a researcher degree of freedom.
- **Legal frameworks:** in lending/credit this would be evaluated against ECOA/Reg B
  (US), FCA fair-treatment rules (UK) or RBI/DPDP guidance (India) for the relevant
  jurisdiction; disparate-impact analysis needs a legally defined protected class and
  a business necessity defence, neither of which is available here.
- **Better statistics:** with more positives, per-group calibration curves, equalised
  odds tests and Wald CIs on group differences; here the 183 primary frauds force
  wide intervals.
- **Post-processing options:** group-specific thresholds, reject-option
  classification, or cost-sensitive thresholds — each trades overall cost for
  parity and must be evaluated on realised cost, not accuracy.
- **Monitoring:** track group flag rates and calibration drift per group once the
  system runs, and alert when a group's FPR departs from its baseline.

*This audit is not a compliance claim.*
