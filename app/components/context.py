"""Shared page chrome: sidebar, model/policy/cost context, banner (Part 14)."""

from __future__ import annotations

import streamlit as st

from components.artifacts import PROFILE, artifacts
from fraudlens.config import Config, load_config

BANNER = "Synthetic data demo — not a production fraud system."

CAPACITY_NOTE = (
    "Capacity caps REVIEW only (per IST day); BLOCK and STEP_UP are chosen by "
    "expected cost and are never bounded."
)

COST_LABELS = {
    "alpha": "Fraud loss rate (alpha)",
    "fixed_fee": "Fixed fee per fraud (INR)",
    "review_cost": "Review cost (INR)",
    "friction_cost": "Customer friction (INR)",
    "catch_rate": "Review catch rate",
}


@st.cache_resource(show_spinner=False)
def get_config() -> Config:
    """The active profile's config (cached)."""
    return load_config(PROFILE)


def sidebar() -> None:
    """Render the sidebar: version, cost assumptions and the disclaimer."""
    art = artifacts()
    st.sidebar.title("FraudLens")
    st.sidebar.caption(BANNER)
    st.sidebar.markdown(f"**Profile** `{PROFILE}`")
    st.sidebar.markdown(f"**Model** `{art.manifest['model_version']}`")
    st.sidebar.markdown(f"**Policy** `{art.manifest['policy_version']}`")
    st.sidebar.divider()
    st.sidebar.markdown("**Cost assumptions**")
    for key, value in art.manifest["costs"].items():
        st.sidebar.markdown(f"- {COST_LABELS.get(key, key)}: `{value}`")
    st.sidebar.caption(CAPACITY_NOTE)
    if not art.manifest["calibrated"]:
        st.sidebar.warning(
            "This model has no calibrator/policy; raw scores are shown as probabilities."
        )
