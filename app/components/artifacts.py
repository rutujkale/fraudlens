"""Load the precomputed app artifacts shared by every page (Part 14).

The artefacts are built by ``scripts/make_app_artifacts.py`` into
``artifacts/app/<profile>/`` (gitignored). Loading is cached, and a missing build
stops the page with the exact command that recreates it.
"""

from __future__ import annotations

import os

import streamlit as st

from fraudlens.appdata.artifacts import AppArtifacts, load_app_artifacts

PROFILE = os.environ.get("FRAUDLENS_PROFILE", "dev")


@st.cache_resource(show_spinner="Loading app artifacts...")
def _load(profile: str) -> AppArtifacts:
    return load_app_artifacts(profile)


def artifacts() -> AppArtifacts:
    """Cached artifacts for the active profile; stops the page with a hint if missing."""
    try:
        return _load(PROFILE)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
