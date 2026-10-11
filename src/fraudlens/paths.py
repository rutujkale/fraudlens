"""Standard filesystem locations, created on demand.

Everything a stage needs is derived from the repo root so paths stay
consistent across training, the API and Streamlit.
"""

from __future__ import annotations

from pathlib import Path

from fraudlens.utils import ensure_dir

REPO_ROOT = Path(__file__).resolve().parents[2]


def raw_data_dir(profile: str) -> Path:
    return ensure_dir(REPO_ROOT / "data" / "raw" / profile)


def features_data_dir(profile: str) -> Path:
    return ensure_dir(REPO_ROOT / "data" / "features" / profile)


def artifacts_dir(model_version: str) -> Path:
    return ensure_dir(REPO_ROOT / "artifacts" / model_version)


def results_dir() -> Path:
    return ensure_dir(REPO_ROOT / "results")


def reports_dir() -> Path:
    return ensure_dir(REPO_ROOT / "reports")


def figures_dir() -> Path:
    return ensure_dir(REPO_ROOT / "reports" / "figures")


def docs_dir() -> Path:
    return ensure_dir(REPO_ROOT / "docs")