# Trade-offs

Each trade-off is stated with the evidence from **this** project. Numbers are cited
from committed results; nothing is estimated.

## Precision vs recall

At a fixed review capacity, raising recall costs precision. The frozen policy is
compared to an F1-optimal threshold at the same operating data
(`results/final_results.json:decision`): the F1-optimal cut maximises the balanced
point, while the cost-based cut maximises expected saving. Because frauds are rare,
the F1-optimal cut reviews too many cases for the analyst budget — the decision
framework charges for each review (`review_cost`), which is the whole point.

## Fraud loss vs customer friction

Every action a decision can take has a cost on the legitimate side: `stepup_cost` /
`stepup_friction` for a challenge, `friction_cost` for a review, `block_friction`
for a block. The cost model (`results/freeze.json:cost_params`) prices *both*
outcomes, so a model with a marginally better detector can lose once friction is
counted. The evidence is the amount-aware policy beating the F1-optimal threshold
**on realized cost**, not on PR-AUC (`H2_decision`).

## Review capacity vs detection

Analyst time is capped at `23/day` (`results/freeze.json:capacity_per_day`). The
policy ranks by largest expected saving and demotes the rest to the next-best
action. Honest caveat: on this test window the policy flagged `5.9/day` and the
cap never bound (`n_demoted = 0`), so the **capacity mechanism is untested** on
this data — it is the unconstrained policy result
(`results/final_results.json:decision.meta`).

## Complexity vs explainability

XGBoost buys ranking power (`0.9619` test PR-AUC) over Logistic Regression
(`0.8873`) and Rules (`0.2036`), but a boosted tree is not directly readable. The
project pays for that complexity with SHAP (global + local) and a reasons layer,
and is explicit that SHAP explains the **margin / rank**, never the 0–100 risk
score (`docs/methodology.md`, `docs/explanation_gallery.md`). The Rules baseline is
kept as a legible floor, not a competitor.

## Synthetic realism vs reproducibility

Every published result is synthetic so it is exactly reproducible from a seed and
committed config (`docs/reproduce.md`). The cost is realism: the sim-to-real gap is
real and measured (`results/sim_to_real.json`), and the subgroup audit is only a
statement about the simulator (`docs/fairness_audit.md`). A public benchmark
(IEEE-CIS) is used once, secondarily, to show direction of transfer.

## Benchmark transfer vs feature compatibility

Only the IEEE-CIS features that have a FraudLens analogue are transferable; the
adapter maps them via a UID heuristic. Full feature parity is impossible (different
schema, different semantics), so the benchmark result is evidence of *direction*,
not a reproduction (`docs/external_validation.md`).

## Latency vs feature richness

The registry has 53 features and the served model uses the frozen calibrator +
policy. Richer features (longer windows, more joins) raise serving cost. The
latency budget is measured, not assumed: p95 end-to-end is `127.031 ms` in an
in-process replay with explanation off (`results/latency.json`), and the feature
build is a single code path shared by training and serving so richness cannot
silently diverge (`docs/methodology.md`).

## Temporal purity vs data volume

A chronological split with an embargo equal to the maximum label delay throws away
data (the embargo gap) to prevent leakage. The alternative — a random split — leaks
future information and inflates every metric. The leakage suite (truncation
invariance, future-mutation invariance, canary) is the evidence the choice was
kept honest (`tests/test_leakage.py`, `docs/what_went_wrong.md`).
