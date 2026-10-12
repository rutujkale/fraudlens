"""Action selection and analyst-capacity enforcement (Part 07C).

Two policies:

* **amount-aware** — pick the action with minimum *expected* cost per transaction;
* **ranking** — for a fixed alert budget, rank by a score key and review the top N.

When REVIEW demand exceeds the daily capacity, keep the candidates with the
largest expected saving ``cost(APPROVE) − cost(REVIEW)`` and demote the rest to
their next-best action (the cheapest non-REVIEW action for that row).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraudlens.evaluation.cost_model import (
    ACTION_NAMES,
    APPROVE,
    BLOCK,
    REVIEW,
    STEP_UP,
    CostParams,
    expected_cost,
)

_ALL_ACTIONS = np.array([APPROVE, STEP_UP, REVIEW, BLOCK])


def amount_aware_actions(prob, amount, params: CostParams) -> np.ndarray:
    """Per-row action minimizing expected cost."""
    prob = np.asarray(prob, dtype=float)
    amount = np.asarray(amount, dtype=float)
    costs = np.stack([expected_cost(a, prob, amount, params) for a in _ALL_ACTIONS], axis=1)
    return _ALL_ACTIONS[np.argmin(costs, axis=1)]


def expected_saving(prob, amount, params: CostParams) -> np.ndarray:
    """Expected saving from reviewing vs approving: ``cost(APPROVE) − cost(REVIEW)``."""
    return expected_cost(APPROVE, prob, amount, params) - expected_cost(
        REVIEW, prob, amount, params
    )


def _next_best_action(prob, amount, params: CostParams) -> np.ndarray:
    """Cheapest action excluding REVIEW (where a demoted review candidate goes)."""
    prob = np.asarray(prob, dtype=float)
    amount = np.asarray(amount, dtype=float)
    others = [APPROVE, STEP_UP, BLOCK]
    costs = np.stack([expected_cost(a, prob, amount, params) for a in others], axis=1)
    return np.array(others)[np.argmin(costs, axis=1)]


@dataclass
class CapacityResult:
    actions: np.ndarray
    n_demoted: int
    n_review: int


def enforce_capacity(
    dates: np.ndarray,
    prob,
    amount,
    actions,
    params: CostParams,
    capacity_per_day: int,
) -> CapacityResult:
    """Cap REVIEW actions per calendar day, demoting the lowest-saving candidates."""
    dates = np.asarray(dates)
    prob = np.asarray(prob, dtype=float)
    amount = np.asarray(amount, dtype=float)
    actions = np.asarray(actions).copy()
    saving = expected_saving(prob, amount, params)
    demoted = 0

    for day in np.unique(dates):
        day_mask = dates == day
        candidates = np.flatnonzero(day_mask & (actions == REVIEW))
        if len(candidates) <= capacity_per_day:
            continue
        order = candidates[np.argsort(-saving[candidates], kind="stable")]
        drop = order[capacity_per_day:]
        actions[drop] = _next_best_action(prob[drop], amount[drop], params)
        demoted += len(drop)

    return CapacityResult(
        actions=actions, n_demoted=int(demoted), n_review=int((actions == REVIEW).sum())
    )


def rank_key(prob, amount, params: CostParams, key: str) -> np.ndarray:
    """Ranking score for top-K policies: ``p``, ``p·A`` (loss at stake), or expected saving."""
    prob = np.asarray(prob, dtype=float)
    amount = np.asarray(amount, dtype=float)
    if key == "p":
        return prob
    if key == "p_times_amount":
        return prob * amount
    if key == "expected_saving":
        return expected_saving(prob, amount, params)
    raise ValueError(f"unknown rank key {key!r}")


def top_k_actions(
    dates: np.ndarray, key_scores: np.ndarray, capacity_per_day: int
) -> np.ndarray:
    """REVIEW the top ``capacity_per_day`` rows (by ``key_scores``) each day."""
    dates = np.asarray(dates)
    key_scores = np.asarray(key_scores, dtype=float)
    actions = np.full(len(dates), APPROVE, dtype=int)
    for day in np.unique(dates):
        day_idx = np.flatnonzero(dates == day)
        order = day_idx[np.argsort(-key_scores[day_idx], kind="stable")]
        actions[order[:capacity_per_day]] = REVIEW
    return actions


def action_names(actions) -> list[str]:
    return [ACTION_NAMES[int(a)] for a in np.asarray(actions)]
