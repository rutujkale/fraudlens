# 8B — Feature-group ablation

**Script:** `scripts/run_experiment_8B.py` · **Profile:** `dev` · **Seeds:** 42–44

## Question

Which of the ten feature groups actually carry the fraud signal? Two complementary
ablations answer this:

- **drop-one** — retrain on the full feature set *minus* one group (does the model
  miss it?), and
- **only-one** — train on *just* that group (is the group sufficient on its own?).

We also track **per-scenario recall@top-1%**, because a group can matter for one
fraud type and nothing else. The groups (registry `features_v1`):

| group | members |
|-------|---------|
| transaction | amount, log_amount, hour, day_of_week, is_weekend, is_night, transaction_type, channel, merchant_category |
| recency_frequency | minutes_since_prev_tx, days_since_last_tx, n_prior_tx, history_days, is_cold_start, tx_count_{10m,1h,6h,24h,7d}, active_days_{7d,30d}, velocity_ratio_1h_vs_24h |
| amount_sums | amount_sum_1h, amount_sum_24h, max_amount_24h, count_near_threshold_24h |
| amount_deviation | avg_amount_7d/30d, std_amount_30d, amount_ratio_to_7d/30d_mean, amount_zscore_30d, amount_vs_lifetime_median |
| merchant_category | new_merchant_flag, new_category_flag, category_freq_30d, merchant_freq_30d |
| device | is_new_device, device_age_days, tx_from_device_prior, accounts_seen_on_device, device_count_30d |
| recipient | unique_recipients_1h, unique_recipients_24h, is_new_recipient |
| geographic | distance_from_home_km, distance_from_prev_km, travel_speed_kmh, new_region_flag, region_freq_30d |
| temporal | hour_deviation |
| trend | activity_change_7d_vs_prev_7d, amount_change_7d_vs_prev_7d |

Full model PR-AUC **0.9657**. Baseline scenario recall@top-1%:
account_takeover **0.970**, amount_anomaly **0.823**, burst **0.956**,
geo_anomaly **0.986**, structuring **1.000**.

## Results — drop-one (PR-AUC and per-scenario recall delta)

| dropped group | PR-AUC | Δ PR-AUC | ATO | amount_anom | burst | geo_anom | structuring |
|---------------|-------:|---------:|----:|------------:|------:|---------:|------------:|
| recency_frequency | 0.9234 | −0.042 | −0.006 | −0.052 | **−0.072** | 0.014 | 0.000 |
| device | 0.9325 | −0.033 | **−0.176** | −0.063 | −0.018 | 0.014 | 0.000 |
| transaction | 0.9438 | −0.022 | −0.042 | **−0.167** | −0.007 | −0.072 | −0.023 |
| merchant_category | 0.9560 | −0.010 | −0.006 | −0.094 | −0.009 | −0.043 | 0.000 |
| geographic | 0.9577 | −0.008 | 0.018 | −0.031 | −0.004 | **−0.217** | 0.000 |
| temporal | 0.9599 | −0.006 | 0.000 | −0.073 | −0.005 | −0.058 | 0.000 |
| amount_deviation | 0.9612 | −0.004 | 0.024 | −0.010 | −0.007 | 0.000 | −0.054 |
| amount_sums | 0.9650 | −0.001 | 0.024 | −0.052 | −0.004 | 0.014 | 0.000 |
| recipient | 0.9644 | −0.001 | 0.012 | −0.031 | −0.012 | −0.015 | 0.000 |
| trend | 0.9645 | −0.001 | −0.006 | −0.010 | 0.002 | 0.014 | 0.000 |

Δ = recall − baseline (negative = the group helped catch that scenario).

## Results — only-one (PR-AUC and per-scenario recall)

| group kept | PR-AUC | ATO | amount_anom | burst | geo_anom | structuring |
|------------|-------:|----:|------------:|------:|---------:|------------:|
| device | 0.4977 | **0.885** | 0.062 | 0.677 | 0.652 | 0.000 |
| recency_frequency | 0.4650 | 0.333 | **0.500** | **0.638** | 0.275 | 0.287 |
| amount_sums | 0.2695 | 0.139 | 0.042 | 0.586 | 0.000 | 0.333 |
| recipient | 0.2315 | 0.055 | 0.000 | 0.513 | 0.000 | 0.395 |
| transaction | 0.1355 | 0.436 | 0.135 | 0.081 | 0.058 | 0.519 |
| amount_deviation | 0.1263 | 0.000 | 0.000 | 0.055 | 0.000 | **0.884** |
| geographic | 0.0685 | 0.461 | 0.000 | 0.011 | **0.942** | 0.000 |
| trend | 0.0335 | 0.055 | 0.000 | 0.081 | 0.000 | 0.527 |
| temporal | 0.0319 | 0.091 | 0.188 | 0.018 | 0.000 | 0.279 |
| merchant_category | 0.0288 | 0.018 | 0.000 | 0.093 | 0.000 | 0.000 |

## Observations

**No group is load-bearing overall.** Dropping the worst group
(`recency_frequency`) costs only −0.042 PR-AUC; the model is highly redundant,
which is what we want for robustness. But the per-scenario deltas are sharp and —
this is the point — they line up one-to-one with how each scenario was *simulated*:

- **`device` → account_takeover (−0.176).** The simulator builds ATO as a *new
  device* acting from a *far region* at an *odd hour*. So the device group
  (`is_new_device`, `device_age_days≈0`, `tx_from_device_prior=0`) is the planted
  cue, and dropping it is the single largest scenario hit. `only_device` still
  recovers ATO recall 0.885, confirming device is nearly sufficient for ATO.
- **`geographic` → geo_anomaly (−0.217).** geo_anomaly is simulated as a
  transaction far from the previous location a few minutes after a normal one, so
  `distance_from_prev_km` / `travel_speed_kmh` spike. Dropping geographic loses
  exactly that. Note `only_geographic` has a *tiny* overall PR-AUC (0.069) yet
  near-perfect **geo_anomaly recall 0.942** — a group can be worthless on average
  and decisive for one scenario, which is why per-scenario recall matters.
- **`recency_frequency` → burst (−0.072).** burst is simulated as many
  transactions within minutes; the velocity windows (`tx_count_1h`,
  `velocity_ratio_1h_vs_24h`, `minutes_since_prev_tx`) are the burst signal.
  `only_recency_frequency` gives the best burst recall (0.638).
- **`transaction` → amount_anomaly (−0.167).** amount_anomaly is 1–2 transactions
  elevated against the customer's own baseline; the raw `amount`/`log_amount` (and
  the merchant/category context they interact with) live in the transaction group,
  so removing it blinds the model to the elevated amount itself.

**Structuring is the most redundant scenario — and that falsifies the tempting
hypothesis.** Intuition says "structuring = sub-threshold splitting, so the
`amount_sums` group (`count_near_threshold_24h`, cumulative sums) must be what
catches it." The data say otherwise: dropping `amount_sums` leaves structuring
recall at **1.000** and costs only −0.001 PR-AUC, and `only_amount_sums` catches
just 0.333 of structuring. Structuring has *at least five* strong redundant
catchers: `only_amount_deviation` 0.884 (`amount_vs_lifetime_median` — the split
amounts are unusual for that customer), `only_trend` 0.527, `only_transaction`
0.519 (`amount`/`transaction_type` — CASH_WITHDRAWAL/ATM), `only_recipient` 0.395,
`only_recency_frequency` 0.287. The signal is spread across amount-vs-baseline,
transaction type and velocity, not concentrated in the threshold-count feature.
The only drop that dents structuring at all is `amount_deviation` (−0.054), which
is the strongest single structuring catcher — not `amount_sums`.

**Hypothesis (not proven here):** the redundancy is a property of the simulator
(sub-threshold amounts are simultaneously unusual *for the customer*, made via a
narrow set of transaction types, and repeated over 1–2 days), so several
independent feature views see the same episode. On real data, where structuring
is subtler, the threshold-count features may matter more.

## Caveats

- `dev` only; all groups are evaluated with the same fixed hyperparameters.
- `only-*` models are deliberately under-powered (few features), so their low
  PR-AUC is expected and only the *relative* per-scenario recalls are meaningful.
- PR-AUC on the full validation window with 342 frauds is noisy at the 3rd
  decimal; we read direction, not exact deltas.

## Interview framing

This experiment is a **falsified hypothesis**, stated plainly: I expected the
threshold-count features (`count_near_threshold_24h`, the `amount_sums` group) to
be what catches structuring. The ablation says otherwise — dropping `amount_sums`
leaves structuring recall at 1.000 (only −0.001 PR-AUC), and `only_amount_sums`
catches just 0.333. The signal is actually spread across amount-vs-baseline
(`only_amount_deviation` 0.884), transaction type (0.519) and trend (0.527). The
honest one-liner: *"I designed a feature I was sure would catch structuring; the
ablation showed it was redundant, and I reported that instead of quietly dropping
the test."*
