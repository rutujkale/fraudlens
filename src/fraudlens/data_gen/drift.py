"""Drift scheduling: switches fraud scenario mix and stealth from Period A to B.

The switch happens at a single moment (`drift.start_fraction` of the window).
Before it, s = 0 (Period A); from it on, s = 1 (Period B). A profile with no
drift (start_fraction null or strength 0) stays in Period A.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def drift_start_date(simulation) -> object:
    d = simulation.drift
    if d.start_fraction is None:
        return None
    return simulation.start_date + timedelta(days=int(d.start_fraction * simulation.days))


def position(simulation, ts) -> float:
    """s in {0, 1}: which fraud period a transaction belongs to."""
    d = simulation.drift
    if d.start_fraction is None or d.strength <= 0:
        return 0.0
    start = pd.Timestamp(drift_start_date(simulation))
    return 1.0 if pd.Timestamp(ts) >= start else 0.0


def scenario_mix(simulation, s: float) -> dict[str, float]:
    """Scenario weights for a period; period B is shifted from A by ``strength``.

    ``scenario_mix_override`` pins both periods. Otherwise after the drift
    moment the mix is ``A + strength * (B - A)`` (strength 1.0 = full switch).
    """
    if simulation.scenario_mix_override is not None:
        return dict(simulation.scenario_mix_override)
    a = simulation.scenario_mix_period_a
    if s < 1.0:
        return dict(a)
    b = simulation.scenario_mix_period_b
    w = simulation.drift.strength
    mixed = {k: (1 - w) * a[k] + w * b[k] for k in a}
    total = sum(mixed.values())
    return {k: v / total for k, v in mixed.items()}


def stealth_beta(simulation, s: float) -> tuple[float, float]:
    a = simulation.stealth_beta_period_a
    if s < 1.0:
        return (a[0], a[1])
    b = simulation.stealth_beta_period_b
    w = simulation.drift.strength
    return (a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]))


def draw_stealth(rng: np.random.Generator, simulation, s: float) -> float:
    a, b = stealth_beta(simulation, s)
    return float(rng.beta(a, b))