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
| age_band | 4 | 3 | 1 | 1 |
| income_band | 4 | 2 | 0 | 2 |
| archetype | 5 | 4 | 0 | 1 |
| home_region_group | 6 | 5 | 1 | 1 |

## Primary view — `val_thr` (the policy's own selection slice)

Primary view — this is the slice the policy and its capacity were selected on (20,657 rows, 122 frauds), so per-group metrics are the honest operating numbers of the frozen policy.

All 122 frauds / 20,657 rows the policy was selected on; the calibrator was fitted earlier in validation, so the calibration gap and Brier here are out-of-sample.

| attribute | group | n | n_fraud | prevalence | flag_rate_any | fpr_any | fpr_strict | recall | precision | pr_auc | mean_p | calibration_gap | brier | suppressed_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| age_band | 25-34 | 7599 | 23 | 0.003 | 0.0092 | 0.0063 | 0.0022 | suppressed | suppressed | suppressed | 0.0037 | 0.0007 | 0.000325 | positives=23<30 |
| age_band | 35-49 | 9393 | 72 | 0.0077 | 0.0196 | 0.0122 | 0.0045 | 0.972 | 0.38 | 0.935 | 0.009 | 0.0013 | 0.001971 |  |
| age_band | 50-64 | 712 | 0 | 0 | 0.0056 | 0.0056 | 0 | suppressed | suppressed | suppressed | 0.0022 | suppressed | suppressed | rows=712<1000; positives=0<30 |
| age_band | <25 | 2953 | 27 | 0.0091 | 0.0115 | 0.0031 | 0.001 | suppressed | suppressed | suppressed | 0.0091 | -0.0001 | 0.000964 | positives=27<30 |
| income_band | high | 3615 | 39 | 0.0108 | 0.0221 | 0.0123 | 0.0053 | 0.923 | 0.45 | 0.897 | 0.011 | 0.0002 | 0.002852 |  |
| income_band | low | 3620 | 9 | 0.0025 | 0.0099 | 0.0078 | 0.0028 | suppressed | suppressed | suppressed | 0.0042 | 0.0017 | 0.001062 | positives=9<30 |
| income_band | mid | 7393 | 52 | 0.007 | 0.0131 | 0.0063 | 0.0018 | 0.981 | 0.526 | 0.98 | 0.0082 | 0.0012 | 0.00097 |  |
| income_band | mid_high | 6029 | 22 | 0.0036 | 0.0131 | 0.0095 | 0.0033 | suppressed | suppressed | suppressed | 0.0043 | 0.0006 | 0.000511 | positives=22<30 |
| archetype | business_like | 3780 | 17 | 0.0045 | 0.0225 | 0.0183 | 0.0058 | suppressed | suppressed | suppressed | 0.0048 | 0.0003 | 0.001576 | positives=17<30 |
| archetype | conservative | 1573 | 17 | 0.0108 | 0.0242 | 0.0135 | 0.0051 | suppressed | suppressed | suppressed | 0.0171 | 0.0063 | 0.003212 | positives=17<30 |
| archetype | frequent_digital | 7411 | 29 | 0.0039 | 0.0061 | 0.0024 | 0.0007 | suppressed | suppressed | suppressed | 0.0044 | 0.0005 | 0.000428 | positives=29<30 |
| archetype | high_spend | 1181 | 10 | 0.0085 | 0.039 | 0.0307 | 0.0171 | suppressed | suppressed | suppressed | 0.0098 | 0.0013 | 0.002433 | positives=10<30 |
| archetype | regular | 6712 | 49 | 0.0073 | 0.0116 | 0.0047 | 0.0011 | 0.959 | 0.603 | 0.956 | 0.0077 | 0.0004 | 0.001096 |  |
| home_region_group | Central | 4965 | 23 | 0.0046 | 0.0121 | 0.0079 | 0.0024 | suppressed | suppressed | suppressed | 0.0055 | 0.0009 | 0.00115 | positives=23<30 |
| home_region_group | East | 2354 | 13 | 0.0055 | 0.0132 | 0.0081 | 0.0017 | suppressed | suppressed | suppressed | 0.0064 | 0.0008 | 0.0005 | positives=13<30 |
| home_region_group | North | 5941 | 49 | 0.0082 | 0.0177 | 0.0098 | 0.0032 | 0.959 | 0.448 | 0.93 | 0.0089 | 0.0007 | 0.001721 |  |
| home_region_group | North-East | 362 | 0 | 0 | 0.0083 | 0.0083 | 0.0028 | suppressed | suppressed | suppressed | 0.0005 | suppressed | suppressed | rows=362<1000; positives=0<30 |
| home_region_group | South | 3050 | 20 | 0.0066 | 0.0151 | 0.0086 | 0.0033 | suppressed | suppressed | suppressed | 0.0078 | 0.0012 | 0.001847 | positives=20<30 |
| home_region_group | West | 3985 | 17 | 0.0043 | 0.0118 | 0.0076 | 0.004 | suppressed | suppressed | suppressed | 0.0054 | 0.0012 | 0.000416 | positives=17<30 |

## Per-action counts (burden, not just flag rates)

| attribute | group | n_approve | n_step_up | n_review | n_block |
|---|---|---|---|---|---|
| age_band | 25-34 | 7529 | 31 | 11 | 28 |
| age_band | 35-49 | 9209 | 75 | 21 | 88 |
| age_band | 50-64 | 708 | 4 | 0 | 0 |
| age_band | <25 | 2919 | 6 | 2 | 26 |
| income_band | high | 3535 | 26 | 10 | 44 |
| income_band | low | 3584 | 18 | 5 | 13 |
| income_band | mid | 7296 | 35 | 2 | 60 |
| income_band | mid_high | 5950 | 37 | 17 | 25 |
| archetype | business_like | 3695 | 48 | 12 | 25 |
| archetype | conservative | 1535 | 13 | 2 | 23 |
| archetype | frequent_digital | 7366 | 13 | 3 | 29 |
| archetype | high_spend | 1135 | 16 | 15 | 15 |
| archetype | regular | 6634 | 26 | 2 | 50 |
| home_region_group | Central | 4905 | 27 | 9 | 24 |
| home_region_group | East | 2323 | 15 | 2 | 14 |
| home_region_group | North | 5836 | 40 | 8 | 57 |
| home_region_group | North-East | 359 | 2 | 0 | 1 |
| home_region_group | South | 3004 | 18 | 4 | 24 |
| home_region_group | West | 3938 | 14 | 11 | 22 |

## Ratios and sampling noise

- `recall` (age_band): not computable — fewer than two fully reported groups
- `fpr_any` (age_band): not computable — fewer than two fully reported groups
- `flag_rate_any` (age_band): not computable — fewer than two fully reported groups
- `recall` (income_band) max/min ratio = 1.062 (high: income_band=mid; low: income_band=high)
- `fpr_any` (income_band) max/min ratio = 1.964 (high: income_band=high; low: income_band=mid)
- `flag_rate_any` (income_band) max/min ratio = 1.687 (high: income_band=high; low: income_band=mid)
- `recall` (archetype): not computable — fewer than two fully reported groups
- `fpr_any` (archetype): not computable — fewer than two fully reported groups
- `flag_rate_any` (archetype): not computable — fewer than two fully reported groups
- `recall` (home_region_group): not computable — fewer than two fully reported groups
- `fpr_any` (home_region_group): not computable — fewer than two fully reported groups
- `flag_rate_any` (home_region_group): not computable — fewer than two fully reported groups

Differences vs the reference group (largest by row count), paired cluster bootstrap over customers — 95% CI excluding 0 means the difference is not explained by sampling noise alone:

| attribute | group | metric | point | lo | hi | distinguishable |
|---|---|---|---|---|---|---|
| age_band | 25-34 | recall | -0.0157 | -0.130347 | 0.057143 | False |
| age_band | 25-34 | fpr_any | -0.005895 | -0.009229 | -0.002478 | True |
| age_band | 25-34 | flag_rate_any | -0.010377 | -0.017537 | -0.003997 | True |
| age_band | <25 | recall | -0.046296 | -0.483871 | 0.035795 | False |
| age_band | <25 | fpr_any | -0.009155 | -0.012455 | -0.005655 | True |
| age_band | <25 | flag_rate_any | -0.008075 | -0.019114 | 0.006218 | False |
| income_band | mid_high | recall | 0.019231 | 0 | 0.086708 | False |
| income_band | mid_high | fpr_any | 0.003223 | -0.000219 | 0.006782 | False |
| income_band | mid_high | flag_rate_any | -1.7e-05 | -0.006997 | 0.007799 | False |
| income_band | low | recall | -0.09188 | -0.25 | 0.051316 | False |
| income_band | low | fpr_any | 0.001488 | -0.002026 | 0.005433 | False |
| income_band | low | flag_rate_any | -0.003176 | -0.010035 | 0.003357 | False |
| income_band | high | recall | -0.057692 | -0.307507 | 0.037282 | False |
| income_band | high | fpr_any | 0.006038 | 0.001424 | 0.010777 | True |
| income_band | high | flag_rate_any | 0.009009 | -0.003551 | 0.023148 | False |
| archetype | regular | recall | 0.028149 | -0.091042 | 0.234198 | False |
| archetype | regular | fpr_any | 0.002214 | 0.000174 | 0.004228 | True |
| archetype | regular | flag_rate_any | 0.005549 | -0.001881 | 0.013341 | False |
| archetype | business_like | recall | 0.010142 | -0.06383 | 0.202381 | False |
| archetype | business_like | fpr_any | 0.015898 | 0.010956 | 0.020543 | True |
| archetype | business_like | flag_rate_any | 0.016415 | 0.006882 | 0.027698 | True |
| archetype | conservative | recall | 0.068966 | 0 | 0.25 | False |
| archetype | conservative | fpr_any | 0.011058 | 0.00468 | 0.018495 | True |
| archetype | conservative | flag_rate_any | 0.018086 | 0.004664 | 0.034222 | True |
| archetype | high_spend | recall | 0.068966 | 0 | 0.25 | False |
| archetype | high_spend | fpr_any | 0.028305 | 0.016941 | 0.039491 | True |
| archetype | high_spend | flag_rate_any | 0.032878 | 0.017994 | 0.049489 | True |
| home_region_group | Central | recall | -0.04614 | -0.267359 | 0.057514 | False |
| home_region_group | Central | fpr_any | -0.001952 | -0.005874 | 0.002358 | False |
| home_region_group | Central | flag_rate_any | -0.005589 | -0.015208 | 0.004475 | False |
| home_region_group | West | recall | 0.040816 | 0 | 0.111111 | False |
| home_region_group | West | fpr_any | -0.002283 | -0.006599 | 0.001657 | False |
| home_region_group | West | flag_rate_any | -0.00588 | -0.015871 | 0.004887 | False |
| home_region_group | South | recall | 0.040816 | 0 | 0.111111 | False |
| home_region_group | South | fpr_any | -0.001263 | -0.005922 | 0.003007 | False |
| home_region_group | South | flag_rate_any | -0.002592 | -0.012709 | 0.00653 | False |
| home_region_group | East | recall | -0.036107 | -0.25 | 0.079619 | False |
| home_region_group | East | fpr_any | -0.001728 | -0.006572 | 0.002859 | False |
| home_region_group | East | flag_rate_any | -0.004505 | -0.015202 | 0.006978 | False |

**11 of 39** differences are distinguishable from noise — and these are **uncorrected for multiple comparisons**. 39 tests at a 95% level would throw up roughly 5% false positives by chance alone, and the tests share bootstrap draws and reference groups, so the family-wise error rate is above 5%. Read them as **exploratory signals**: a confirmatory analysis would pre-register a smaller family (e.g. FPR per attribute) and apply Holm-Bonferroni or Benjamini-Hochberg. Note also that every group here was reported, not selected — no group was dropped for being unflattering.


## Exploratory view — full validation window

342 frauds / 41,313 rows. **Exploratory only**: the calibrator was fitted on the first half of this window (`val_cal`), so calibration gap and Brier would be in-sample for half the rows and are therefore excluded here; the policy capacity was fitted on `val_thr`.

| attribute | group | n | n_fraud | prevalence | flag_rate_any | fpr_any | fpr_strict | recall | precision | pr_auc | mean_p | calibration_gap | brier | suppressed_reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| age_band | 25-34 | 15137 | 113 | 0.0075 | 0.0124 | 0.0053 | 0.0017 | 0.956 | 0.574 | 0.971 | suppressed | suppressed | suppressed |  |
| age_band | 35-49 | 18912 | 165 | 0.0087 | 0.0209 | 0.0124 | 0.004 | 0.988 | 0.412 | 0.96 | suppressed | suppressed | suppressed |  |
| age_band | 50-64 | 1476 | 0 | 0 | 0.0095 | 0.0095 | 0.0007 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | positives=0<30 |
| age_band | <25 | 5788 | 64 | 0.0111 | 0.0147 | 0.004 | 0.0012 | 0.969 | 0.729 | 0.975 | suppressed | suppressed | suppressed |  |
| income_band | high | 7345 | 73 | 0.0099 | 0.0231 | 0.0138 | 0.0051 | 0.959 | 0.412 | 0.926 | suppressed | suppressed | suppressed |  |
| income_band | low | 7310 | 41 | 0.0056 | 0.0112 | 0.0059 | 0.0021 | 0.951 | 0.476 | 0.965 | suppressed | suppressed | suppressed |  |
| income_band | mid | 14798 | 134 | 0.0091 | 0.0153 | 0.0065 | 0.0015 | 0.978 | 0.58 | 0.979 | suppressed | suppressed | suppressed |  |
| income_band | mid_high | 11860 | 94 | 0.0079 | 0.0173 | 0.0095 | 0.0029 | 0.989 | 0.454 | 0.98 | suppressed | suppressed | suppressed |  |
| archetype | business_like | 7641 | 38 | 0.005 | 0.0229 | 0.0182 | 0.0053 | 0.974 | 0.211 | 0.915 | suppressed | suppressed | suppressed |  |
| archetype | conservative | 3255 | 76 | 0.0233 | 0.0359 | 0.0129 | 0.0041 | 1 | 0.65 | 0.989 | suppressed | suppressed | suppressed |  |
| archetype | frequent_digital | 14645 | 111 | 0.0076 | 0.0095 | 0.0022 | 0.0006 | 0.964 | 0.77 | 0.976 | suppressed | suppressed | suppressed |  |
| archetype | high_spend | 2371 | 10 | 0.0042 | 0.0371 | 0.033 | 0.0131 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | positives=10<30 |
| archetype | regular | 13401 | 107 | 0.008 | 0.0122 | 0.0046 | 0.0012 | 0.963 | 0.628 | 0.964 | suppressed | suppressed | suppressed |  |
| home_region_group | Central | 10003 | 84 | 0.0084 | 0.0159 | 0.0079 | 0.0026 | 0.964 | 0.509 | 0.958 | suppressed | suppressed | suppressed |  |
| home_region_group | East | 4665 | 22 | 0.0047 | 0.0139 | 0.0095 | 0.0022 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | positives=22<30 |
| home_region_group | North | 11952 | 171 | 0.0143 | 0.0236 | 0.0098 | 0.0031 | 0.977 | 0.592 | 0.971 | suppressed | suppressed | suppressed |  |
| home_region_group | North-East | 777 | 0 | 0 | 0.0064 | 0.0064 | 0.0013 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | rows=777<1000; positives=0<30 |
| home_region_group | South | 5986 | 46 | 0.0077 | 0.0147 | 0.0072 | 0.0022 | 0.978 | 0.511 | 0.962 | suppressed | suppressed | suppressed |  |
| home_region_group | West | 7930 | 19 | 0.0024 | 0.0106 | 0.0082 | 0.0027 | suppressed | suppressed | suppressed | suppressed | suppressed | suppressed | positives=19<30 |

## Proxy check — do model features encode the audit attributes?

Ordinal attributes (age/income bands) use Spearman rho; nominal attributes use eta-squared for numeric features and Cramer's V for the categorical ones. Pre-registered thresholds: |rho| >= 0.30, eta^2 >= 0.14, V >= 0.30. A strong correlation means the feature is a *proxy* for the attribute, which is how a behavioural difference can look like a demographic disparity.

| attribute | feature | statistic | value | threshold |
|---|---|---|---|---|
| age_band | max_amount_24h | spearman | 0.467252 | 0.3 |
| age_band | avg_amount_7d | spearman | 0.398985 | 0.3 |
| age_band | avg_amount_30d | spearman | 0.376674 | 0.3 |
| age_band | amount | spearman | 0.333177 | 0.3 |
| age_band | log_amount | spearman | 0.333177 | 0.3 |
| age_band | std_amount_30d | spearman | 0.309598 | 0.3 |
| archetype | n_prior_tx | eta_squared | 0.816586 | 0.14 |
| archetype | active_days_30d | eta_squared | 0.798181 | 0.14 |
| archetype | avg_amount_30d | eta_squared | 0.717638 | 0.14 |
| archetype | log_amount | eta_squared | 0.627903 | 0.14 |
| archetype | avg_amount_7d | eta_squared | 0.597503 | 0.14 |
| archetype | active_days_7d | eta_squared | 0.559612 | 0.14 |
| archetype | tx_count_7d | eta_squared | 0.547173 | 0.14 |
| archetype | std_amount_30d | eta_squared | 0.529569 | 0.14 |
| archetype | tx_from_device_prior | eta_squared | 0.523548 | 0.14 |
| archetype | device_count_30d | eta_squared | 0.457211 | 0.14 |
| archetype | max_amount_24h | eta_squared | 0.355043 | 0.14 |
| archetype | amount_sum_24h | eta_squared | 0.323449 | 0.14 |
| archetype | amount | eta_squared | 0.315245 | 0.14 |
| archetype | minutes_since_prev_tx | eta_squared | 0.255975 | 0.14 |
| archetype | days_since_last_tx | eta_squared | 0.255975 | 0.14 |
| archetype | hour_deviation | eta_squared | 0.246776 | 0.14 |
| archetype | count_near_threshold_24h | eta_squared | 0.175128 | 0.14 |
| archetype | is_new_recipient | eta_squared | 0.168758 | 0.14 |
| archetype | tx_count_24h | eta_squared | 0.156047 | 0.14 |
| income_band | max_amount_24h | spearman | 0.362977 | 0.3 |
| income_band | avg_amount_30d | spearman | 0.317454 | 0.3 |
| income_band | avg_amount_7d | spearman | 0.316386 | 0.3 |
| income_band | std_amount_30d | spearman | 0.308435 | 0.3 |

(29 of 212 feature/attribute pairs are strong; the full table is in `results/fairness_proxy_dev.csv`.)


## What I would examine with real data

- **Group definitions:** age/income bands are a modelling choice; real audits must
  pre-register them (and intersectional groups, e.g. age x region) before looking at
  outcomes, or the bands become a researcher degree of freedom.
- **Legal frameworks:** in lending/credit this would be evaluated against ECOA/Reg B
  (US), FCA fair-treatment rules (UK) or RBI/DPDP guidance (India) for the relevant
  jurisdiction; disparate-impact analysis needs a legally defined protected class and
  a business necessity defence, neither of which is available here.
- **Better statistics:** with more positives, per-group calibration curves, equalised
  odds tests and Wald CIs on group differences; here the 122 primary frauds force
  wide intervals.
- **Post-processing options:** group-specific thresholds, reject-option
  classification, or cost-sensitive thresholds — each trades overall cost for
  parity and must be evaluated on realised cost, not accuracy.
- **Monitoring:** track group flag rates and calibration drift per group once the
  system runs, and alert when a group's FPR departs from its baseline.

*This audit is not a compliance claim.*
