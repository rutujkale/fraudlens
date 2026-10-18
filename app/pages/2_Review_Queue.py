"""Page 2 — Review Queue & Cost Optimization (Part 14).

Every recompute on this page calls the Part 07 package code through
``fraudlens.appdata.recompute``; the sliders only change the numbers fed in.

Layout follows ``design/stitch/review_queue_cost_optimization``: context strip,
dense KPI row, the operational-invariant banner, the queue table with an inspector,
the cost controls and the loss-vs-threshold chart.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from components import api_client, charts, context, theme  # noqa: E402

from fraudlens.appdata import recompute  # noqa: E402
from fraudlens.evaluation.cost_model import CostParams  # noqa: E402

CONTROL_KEYS = [
    "alpha",
    "fixed_fee",
    "review_cost",
    "friction_cost",
    "catch_rate",
    "stepup_cost",
    "stepup_friction",
    "stop_rate",
    "block_friction",
]

st.set_page_config(page_title="Review Queue — FraudLens", layout="wide")
context.sidebar()
st.title("Review Queue & Cost Optimization")

art = context.artifacts()
sample = art.scored_sample
costs = art.manifest["costs"]
cfg = context.get_config()

for key in CONTROL_KEYS:
    st.session_state.setdefault(f"p2_{key}", float(costs[key]))
st.session_state.setdefault("p2_capacity", int(art.manifest["capacity_per_day"]))

with st.expander("Cost & capacity controls", expanded=True):
    row1 = st.columns(4)
    alpha = row1[0].slider("Fraud loss rate α", 0.0, 1.0, step=0.05, key="p2_alpha")
    review_cost = row1[1].slider("Review cost (INR)", 0.0, 500.0, step=5.0, key="p2_review_cost")
    friction_cost = row1[2].slider(
        "Friction cost (INR)", 0.0, 1000.0, step=10.0, key="p2_friction_cost"
    )
    capacity = row1[3].number_input("Review capacity / day", min_value=1, key="p2_capacity")
    row2 = st.columns(4)
    fixed_fee = row2[0].slider(
        "Fixed fee per fraud (INR)", 0.0, 500.0, step=5.0, key="p2_fixed_fee"
    )
    catch_rate = row2[1].slider("Review catch rate", 0.0, 1.0, step=0.05, key="p2_catch_rate")
    stepup_cost = row2[2].slider("Step-up cost (INR)", 0.0, 200.0, step=1.0, key="p2_stepup_cost")
    stepup_friction = row2[3].slider(
        "Step-up friction (INR)", 0.0, 500.0, step=5.0, key="p2_stepup_friction"
    )
    row3 = st.columns(4)
    stop_rate = row3[0].slider("Step-up stop rate", 0.0, 1.0, step=0.05, key="p2_stop_rate")
    block_friction = row3[1].slider(
        "Block friction (INR)", 0.0, 2000.0, step=25.0, key="p2_block_friction"
    )
    if row3[2].button("Reset to frozen assumptions"):
        for key in CONTROL_KEYS:
            st.session_state.pop(f"p2_{key}", None)
        st.session_state.pop("p2_capacity", None)
        st.rerun()

params = CostParams(
    alpha=alpha,
    fixed_fee=fixed_fee,
    review_cost=review_cost,
    friction_cost=friction_cost,
    catch_rate=catch_rate,
    stepup_cost=stepup_cost,
    stepup_friction=stepup_friction,
    stop_rate=stop_rate,
    block_friction=block_friction,
)

show_labels = st.toggle(
    "Show labelled outcomes (realized cost and is_fraud)", value=False,
    help="Off by default: the queue is what an analyst sees, without ground truth.",
)

st.info(context.CAPACITY_NOTE)

metrics = recompute.policy_metrics(sample, params, int(capacity))
best = recompute.best_thresholds(sample, params, int(capacity))

kpi = st.columns(5)
kpi[0].metric("Transactions in sample", f"{metrics['n']:,}")
kpi[1].metric("Flagged (non-approve)", f"{metrics['flagged']:,}")
kpi[2].metric("Review / day vs capacity", f"{metrics['review_per_day']:.1f} / {capacity}")
kpi[3].metric("Expected loss (INR)", f"{metrics['expected_loss']:,.0f}")
kpi[4].metric("Expected loss avoided (INR)", f"{metrics['expected_loss_avoided']:,.0f}")
if show_labels:
    st.metric("Realized loss avoided (INR, labelled)", f"{metrics['realized_loss_avoided']:,.0f}")
theme.chips(
    [
        (f"cost-optimal τ {best['cost_optimal_threshold']:.4f}", "steel"),
        (f"F1-optimal τ {best['f1_threshold']:.4f}", ""),
        (f"feasible τ {best['cost_optimal_feasible_threshold']:.4f}", "gold"),
        (f"demoted {metrics['n_demoted']}", "warn" if metrics["n_demoted"] else "ok"),
    ]
)

with st.container(border=True):
    st.plotly_chart(charts.loss_curve(recompute.scan(sample, params)), use_container_width=True)
    st.caption(
        "Expected-loss curve from `scan_global_thresholds`; the policy chooses actions "
        "per transaction, not one global threshold."
    )

st.subheader("Queue")
queue = recompute.queue_table(sample, params, int(capacity))
queue = queue.sort_values("expected_saving", ascending=False, kind="stable")
action_filter = st.multiselect(
    "Actions", ["STEP_UP", "REVIEW", "BLOCK"], default=["STEP_UP", "REVIEW", "BLOCK"]
)
view = queue[queue["recommended_action"].isin(action_filter)]
columns = [
    "transaction_id", "timestamp", "amount", "fraud_probability", "risk_score",
    "recommended_action", "risk_level", "archetype", "expected_saving", "top_reason",
]
if show_labels:
    columns += ["is_fraud", "fraud_scenario"]
st.dataframe(view[columns].head(500), use_container_width=True)
st.download_button(
    "Download queue CSV", view[columns].to_csv(index=False).encode("utf-8"),
    file_name="review_queue.csv", mime="text/csv",
)

with st.expander("Batch scoring (CSV upload)"):
    upload = st.file_uploader("Transactions CSV", type="csv")
    if upload is not None:
        try:
            result = api_client.get_client().batch_csv(upload.getvalue(), upload.name)
        except api_client.ApiError as exc:
            st.error(f"Batch scoring failed: {exc}")
            st.info(f"Start the API first: `{api_client.API_COMMAND}`")
        else:
            rows = result.get("rows", [])
            summary = result.get("summary", {})
            st.success(
                f"Scored {summary.get('n', len(rows))} rows; review workload "
                f"{summary.get('review_workload')} (demoted {summary.get('n_demoted')})."
            )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

with st.expander("Preset cost scenarios"):
    st.dataframe(recompute.scenario_table(sample, cfg, int(capacity)), use_container_width=True)
