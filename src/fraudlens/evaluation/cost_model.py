"""Business cost model for the four risk actions (Part 07B).

Implements the framework in ``CLAUDE.md`` ("Decision framework"). For a
transaction with calibrated probability ``p`` and amount ``A`` (INR):

    cost(APPROVE) = p·(α·A + fee)
    cost(STEP_UP) = S + (1-p)·F_step + p·(1-s)·(α·A + fee)
    cost(REVIEW)  = R + (1-p)·F      + p·(1-c)·(α·A + fee)
    cost(BLOCK)   = (1-p)·F_block

``expected_cost`` uses ``p`` (no labels needed). ``realized_cost`` uses the true
label, with the review catch rate ``c`` and step-up stop rate ``s`` treated as
deterministic expectations (a fraud under review incurs ``(1-c)·(α·A+fee)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraudlens.config import CostConfig, CostScenario

APPROVE, STEP_UP, REVIEW, BLOCK = 0, 1, 2, 3
ACTION_NAMES = ("APPROVE", "STEP_UP", "REVIEW", "BLOCK")
# Severity rises with the intrusion on the customer (used by the monotonicity test).
ACTION_SEVERITY = {APPROVE: 0, STEP_UP: 1, REVIEW: 2, BLOCK: 3}


@dataclass(frozen=True)
class CostParams:
    """Resolved business cost parameters."""

    alpha: float          # fraud_loss_rate
    fixed_fee: float      # fraud_fixed_fee
    review_cost: float    # R
    friction_cost: float  # F (legit friction when reviewed)
    catch_rate: float     # c (review catch rate)
    stepup_cost: float    # S
    stepup_friction: float  # F_step
    stop_rate: float      # s (step-up stop rate)
    block_friction: float  # F_block


def resolve_cost_params(costs: CostConfig, scenario: CostScenario | None = None) -> CostParams:
    """Build ``CostParams`` from config, applying a scenario's non-null overrides."""
    values = {
        "alpha": costs.fraud_loss_rate,
        "fixed_fee": costs.fraud_fixed_fee,
        "review_cost": costs.review_cost,
        "friction_cost": costs.friction_cost,
        "catch_rate": costs.review_catch_rate,
        "stepup_cost": costs.stepup_cost,
        "stepup_friction": costs.stepup_friction_cost,
        "stop_rate": costs.stepup_stop_rate,
        "block_friction": costs.block_friction_cost,
    }
    if scenario is not None:
        overrides = {
            "alpha": scenario.fraud_loss_rate,
            "fixed_fee": scenario.fraud_fixed_fee,
            "review_cost": scenario.review_cost,
            "friction_cost": scenario.friction_cost,
        }
        for key, val in overrides.items():
            if val is not None:
                values[key] = val
    return CostParams(**values)


def _loss(amount: np.ndarray, p: CostParams) -> np.ndarray:
    return p.alpha * np.asarray(amount, dtype=float) + p.fixed_fee


def expected_cost(action, prob, amount, params: CostParams) -> np.ndarray:
    """Vectorized expected cost of ``action`` (int code) for each row."""
    p = np.asarray(prob, dtype=float)
    loss = _loss(amount, params)
    approve = p * loss
    stepup = (
        params.stepup_cost
        + (1 - p) * params.stepup_friction
        + p * (1 - params.stop_rate) * loss
    )
    review = (
        params.review_cost
        + (1 - p) * params.friction_cost
        + p * (1 - params.catch_rate) * loss
    )
    block = (1 - p) * params.block_friction
    action = np.asarray(action)
    return np.select(
        [(action == APPROVE), (action == STEP_UP), (action == REVIEW), (action == BLOCK)],
        [approve, stepup, review, block],
        default=approve,
    )


def realized_cost(action, is_fraud, amount, params: CostParams) -> np.ndarray:
    """Vectorized realized cost given the true label (0/1)."""
    y = np.asarray(is_fraud).astype(bool)
    loss = _loss(amount, params)
    approve = np.where(y, loss, 0.0)
    stepup = params.stepup_cost + np.where(y, (1 - params.stop_rate) * loss, params.stepup_friction)
    review = params.review_cost + np.where(y, (1 - params.catch_rate) * loss, params.friction_cost)
    block = np.where(y, 0.0, params.block_friction)
    action = np.asarray(action)
    return np.select(
        [(action == APPROVE), (action == STEP_UP), (action == REVIEW), (action == BLOCK)],
        [approve, stepup, review, block],
        default=approve,
    )


def cost_breakdown(
    actions, is_fraud, amount, params: CostParams, *, realized: bool = True
) -> dict[str, float]:
    """Decompose total cost into fraud loss, review, step-up and friction channels."""
    action = np.asarray(actions)
    y = np.asarray(is_fraud).astype(bool)
    loss = _loss(amount, params)
    n_review = int((action == REVIEW).sum())
    n_stepup = int((action == STEP_UP).sum())
    n_block = int((action == BLOCK).sum())

    if realized:
        fraud_loss = 0.0
        fraud_loss += float(loss[(action == APPROVE) & y].sum())
        fraud_loss += float(((1 - params.stop_rate) * loss)[(action == STEP_UP) & y].sum())
        fraud_loss += float(((1 - params.catch_rate) * loss)[(action == REVIEW) & y].sum())
        friction = (
            params.friction_cost * int(((action == REVIEW) & ~y).sum())
            + params.stepup_friction * int(((action == STEP_UP) & ~y).sum())
            + params.block_friction * int(((action == BLOCK) & ~y).sum())
        )
        total = float(realized_cost(action, y, amount, params).sum())
    else:
        p = np.asarray(is_fraud, dtype=float)  # in expected mode callers pass prob here
        fraud_loss = 0.0
        fraud_loss += float((p * loss)[action == APPROVE].sum())
        fraud_loss += float((p * (1 - params.stop_rate) * loss)[action == STEP_UP].sum())
        fraud_loss += float((p * (1 - params.catch_rate) * loss)[action == REVIEW].sum())
        friction = (
            float(((1 - p) * params.friction_cost)[action == REVIEW].sum())
            + float(((1 - p) * params.stepup_friction)[action == STEP_UP].sum())
            + float(((1 - p) * params.block_friction)[action == BLOCK].sum())
        )
        total = float(expected_cost(action, p, amount, params).sum())

    review_cost = params.review_cost * n_review
    stepup_cost = params.stepup_cost * n_stepup
    return {
        "total": total,
        "fraud_loss": fraud_loss,
        "review_cost": review_cost,
        "stepup_cost": stepup_cost,
        "friction_cost": friction,
        "n_review": n_review,
        "n_stepup": n_stepup,
        "n_block": n_block,
    }


def approve_all(n: int) -> np.ndarray:
    return np.full(n, APPROVE, dtype=int)


def review_all(n: int) -> np.ndarray:
    return np.full(n, REVIEW, dtype=int)


def review_top_k(score: np.ndarray, k: int) -> np.ndarray:
    """REVIEW the ``k`` highest scores, APPROVE the rest."""
    actions = approve_all(len(score))
    if k > 0:
        actions[np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]] = REVIEW
    return actions
