"""Risk score, lift, and action-derived risk levels (Part 07F).

``risk_score = round(100 × calibrated_probability, 1)``. ``risk_level`` is derived
from the chosen **action**, not from fixed probability bands: at the observed
prevalence (~0.8% on dev) fixed 0–29/30–69/70–89/90–100 bands would label almost
everything LOW and a genuine fraud as MEDIUM, so the level must reflect what we
actually do.
"""

from __future__ import annotations

import numpy as np

from fraudlens.evaluation.cost_model import ACTION_NAMES, APPROVE, BLOCK, REVIEW, STEP_UP

RISK_LEVELS: dict[int, str] = {
    APPROVE: "LOW",
    STEP_UP: "MEDIUM",
    REVIEW: "HIGH",
    BLOCK: "CRITICAL",
}


def risk_score(probability) -> np.ndarray:
    """0–100 score on one decimal: ``round(100 × p, 1)``."""
    return np.round(100.0 * np.asarray(probability, dtype=float), 1)


def lift(probability, base_rate: float) -> np.ndarray:
    """Calibrated probability divided by the population base rate."""
    return np.asarray(probability, dtype=float) / base_rate


def risk_level(action) -> np.ndarray:
    """Map action codes to level names (LOW/MEDIUM/HIGH/CRITICAL)."""
    actions = np.asarray(action)
    out = np.empty(actions.shape, dtype=object)
    for code, name in RISK_LEVELS.items():
        out[actions == code] = name
    return out


def action_name(action) -> np.ndarray:
    actions = np.asarray(action)
    out = np.empty(actions.shape, dtype=object)
    for code, name in enumerate(ACTION_NAMES):
        out[actions == code] = name
    return out
