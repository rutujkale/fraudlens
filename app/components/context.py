"""Shared page chrome: sidebar, model/policy/cost context, banner (Part 14).

UI polish: the sidebar also applies the shared stylesheet, and this module is the
single place that exposes the *locked* wording (banner, capacity note, SHAP
caption) and the Part 15 engineering strings. The engineering strings are read
straight from :mod:`fraudlens.reporting.render` so the app and the rendered docs
cannot drift; ``tests/test_app_smoke.py`` asserts the equality.
"""

from __future__ import annotations

import streamlit as st

from components import theme
from components.artifacts import PROFILE, artifacts
from fraudlens.config import Config, load_config
from fraudlens.reporting.render import build_context as _build_render_context

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


@st.cache_resource(show_spinner=False)
def _render_context() -> dict:
    """The Part 15 render context (numbers from ``results/``), read only.

    ``test_counts`` is stubbed so :func:`collect_test_counts` (which scans the
    test suite) is skipped — only the engineering strings are consumed here.
    """
    return _build_render_context(test_counts={"n_tests": None, "n_leakage_tests": None})


def engineering_latency() -> str:
    """p95 latency figure, exactly as ``reporting.render`` renders it."""
    return str(_render_context()["headline.engineering.latency"])


def engineering_latency_note() -> str:
    """Latency conditions line, exactly as ``reporting.render`` renders it."""
    return str(_render_context()["headline.engineering.latency_note"])


def engineering_parity() -> str:
    """Training/serving parity sentence, exactly as ``reporting.render`` renders it."""
    return str(_render_context()["headline.engineering.parity"])


def sidebar() -> None:
    """Render the sidebar: version, cost assumptions and the disclaimer."""
    theme.inject()
    art = artifacts()
    st.sidebar.title("FraudLens")
    st.sidebar.caption(BANNER)
    st.sidebar.markdown(
        theme.chip(f"profile {PROFILE}", "steel")
        + theme.chip(str(art.manifest["model_version"]), "gold")
        + theme.chip(str(art.manifest["policy_version"] or "no policy")),
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Cost assumptions**")
    for key, value in art.manifest["costs"].items():
        st.sidebar.markdown(f"- {COST_LABELS.get(key, key)}: `{value}`")
    st.sidebar.caption(CAPACITY_NOTE)
    if not art.manifest["calibrated"]:
        st.sidebar.warning(
            "This model has no calibrator/policy; raw scores are shown as probabilities."
        )
