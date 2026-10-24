"""Page 1 — Transaction Analyzer (Part 14).

Scores one transaction through the API and shows its explanation and customer
context. No feature or model logic lives here.

Layout follows the design mockup: a two-pane workspace with
the payload form on the left and the decision / why / SHAP deep-dive on the right.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from components import api_client, charts, context, theme  # noqa: E402

from fraudlens.data_gen.archetypes import CATEGORIES, CHANNELS, TRANSACTION_TYPES  # noqa: E402

SHAP_CAPTION = "Contribution to the model's log-odds (not points on the risk score)"

_RISK_TONE = {"LOW": "ok", "MEDIUM": "gold", "HIGH": "warn", "CRITICAL": "warn"}

st.set_page_config(page_title="Transaction Analyzer — FraudLens", layout="wide")
context.sidebar()
st.title("Transaction Analyzer")
st.caption("Score a single transaction end-to-end through the Part 13 API.")

art = context.artifacts()
client = api_client.get_client()
presets = art.presets
names = ["Manual entry"] + [f"{p['label']} (expected {p['expected_action']})" for p in presets]
choice = st.segmented_control(
    "Preset", names, default=names[1] if presets else names[0]
)
choice = choice or "Manual entry"
preset = None if choice == "Manual entry" else presets[names.index(choice) - 1]
tx = preset["transaction"] if preset else {}

left, right = st.columns([5, 7], gap="large")

with left:
    with st.form("score"):
        st.markdown("**Transaction payload parameters**")
        theme.rule()
        customer_id = st.text_input("Customer ID", tx.get("customer_id", ""))
        c1, c2 = st.columns(2)
        amount = c1.number_input(
            "Amount (INR)", min_value=1.0, value=float(tx.get("amount", 2500.0))
        )
        timestamp = c2.text_input(
            "Timestamp (IST, naive)",
            tx.get("timestamp", art.manifest["windows"]["t_val_end"]),
        )
        device_id = st.text_input("Device ID", tx.get("device_id") or "")
        d1, d2 = st.columns(2)
        merchant_category = d1.selectbox(
            "Merchant category",
            CATEGORIES,
            index=(
                CATEGORIES.index(tx["merchant_category"])
                if tx.get("merchant_category") in CATEGORIES
                else 0
            ),
        )
        location_id = d2.text_input("Location ID", tx.get("location_id") or "")
        e1, e2 = st.columns(2)
        channel = e1.selectbox(
            "Channel",
            CHANNELS,
            index=CHANNELS.index(tx["channel"]) if tx.get("channel") in CHANNELS else 0,
        )
        transaction_type = e2.selectbox(
            "Transaction type",
            TRANSACTION_TYPES,
            index=(
                TRANSACTION_TYPES.index(tx["transaction_type"])
                if tx.get("transaction_type") in TRANSACTION_TYPES
                else 0
            ),
        )
        recipient_id = st.text_input("Recipient ID", tx.get("recipient_id") or "")
        include_explanation = st.checkbox("Include explanation (SHAP)", value=True)
        submitted = st.form_submit_button("Score transaction", type="primary")

with right:
    if submitted:
        payload = {
            "customer_id": customer_id,
            "amount": float(amount),
            "timestamp": timestamp,
            "merchant_category": merchant_category,
            "device_id": device_id or None,
            "location_id": location_id or None,
            "transaction_type": transaction_type,
            "channel": channel,
            "recipient_id": recipient_id or None,
            "include_explanation": include_explanation,
        }
        try:
            result = client.predict(payload)
        except api_client.ApiError as exc:
            st.error(f"Could not reach the scoring API: {exc}")
            st.info(f"Start it first: `{api_client.API_COMMAND}`")
            st.stop()

        st.header("Decision")
        kpi = st.columns(4)
        kpi[0].metric("Risk score (0-100)", f"{result['risk_score']:.1f}")
        kpi[1].metric("Calibrated probability", f"{result['fraud_probability']:.4f}")
        kpi[2].metric("Lift over base rate", f"{result['lift_over_base_rate']:.1f}x")
        kpi[3].metric("Recommended action", result["recommended_action"])
        theme.chips(
            [
                (f"risk level {result['risk_level']}", _RISK_TONE.get(result["risk_level"], "")),
                (f"model {result['model_version']}", "gold"),
                (f"policy {result['policy_version']}", "steel"),
            ]
        )
        st.caption(f"Base rate {result['base_rate']:.5f}")

        why, shap_col = st.columns(2)
        with why:
            with st.container(border=True):
                st.markdown("**Why (raising risk)**")
                for sentence in result["reasons"] or ["No strong risk-raising driver."]:
                    st.markdown(f"- {sentence}")
                st.markdown("**Mitigating factors**")
                for sentence in result["mitigating_factors"] or ["No mitigating driver."]:
                    st.markdown(f"- {sentence}")
        with shap_col:
            with st.container(border=True):
                st.plotly_chart(charts.shap_bar(result["contributions"]), use_container_width=True)
                st.caption(SHAP_CAPTION)

        st.caption(
            f"Latency {result['latency_ms']:.0f} ms — "
            + ", ".join(f"{k} {v:.0f} ms" for k, v in result["timing_ms"].items())
        )

        st.header("Customer context")
        try:
            summary = client.customer_summary(customer_id, before=timestamp)
        except api_client.ApiError as exc:
            st.warning(f"Customer summary unavailable: {exc}")
        else:
            ctx = st.columns(4)
            ctx[0].metric("Prior transactions", summary["n_prior_tx"])
            ctx[1].metric(
                "Typical amount (INR)",
                f"{summary['typical_amount']:,.0f}" if summary["typical_amount"] else "n/a",
            )
            ctx[2].metric(
                "Usual hours", ", ".join(str(h) for h in summary["usual_hours"]) or "n/a"
            )
            ctx[3].metric("Known devices", len(summary["known_devices"]))
            if summary["recent"]:
                st.dataframe(pd.DataFrame(summary["recent"]), use_container_width=True)
            st.caption(f"History strictly before `{summary['before']}`.")
    else:
        st.info("Pick a preset or fill the payload, then **Score transaction**.")
