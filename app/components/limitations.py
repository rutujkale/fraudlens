"""Limitations text for the Model Performance page (Part 14).

Reads ``docs/limitations.md`` when it exists (Part 15); until then a file-sourced
placeholder is shown so the page never invents claims.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

LIMITATIONS_FILE = "docs/limitations.md"

FALLBACK = (
    "A consolidated `docs/limitations.md` is not written yet (scheduled for Part 15). "
    "Known limitations of this demo, all visible elsewhere in the project: synthetic "
    "data only; the reported metrics are single-split point estimates with customer "
    "cluster-bootstrap CIs; the decision policy is evaluated on one held-out test "
    "window; the API is single-process on CPU; and fairness is audited only on "
    "simulator archetypes, not on protected attributes."
)


def body() -> str:
    """Limitations markdown, from the docs file if present else the placeholder."""
    try:
        return Path(LIMITATIONS_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return FALLBACK


def render() -> None:
    st.expander("Limitations", expanded=False).markdown(body())
