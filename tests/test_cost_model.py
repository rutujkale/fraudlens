"""Tests for the cost model (Part 07B)."""

from __future__ import annotations

import numpy as np
import pytest

from fraudlens.evaluation.cost_model import (
    APPROVE,
    BLOCK,
    REVIEW,
    STEP_UP,
    CostParams,
    approve_all,
    cost_breakdown,
    expected_cost,
    realized_cost,
    resolve_cost_params,
    review_all,
    review_top_k,
)

PARAMS = CostParams(
    alpha=1.0, fixed_fee=0.0, review_cost=50.0, friction_cost=100.0, catch_rate=1.0,
    stepup_cost=5.0, stepup_friction=20.0, stop_rate=0.5, block_friction=200.0,
)


@pytest.mark.parametrize(
    "action,expected",
    [
        (APPROVE, 100.0),
        (STEP_UP, 73.0),
        (REVIEW, 140.0),
        (BLOCK, 180.0),
    ],
)
def test_expected_cost_hand_computed(action, expected):
    got = expected_cost(action, np.array([0.1]), np.array([1000.0]), PARAMS)
    assert got[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    "action,is_fraud,expected",
    [
        (APPROVE, 1, 1000.0),
        (APPROVE, 0, 0.0),
        (STEP_UP, 1, 5.0 + 0.5 * 1000.0),
        (STEP_UP, 0, 5.0 + 20.0),
        (REVIEW, 1, 50.0 + 0.0 * 1000.0),
        (REVIEW, 0, 50.0 + 100.0),
        (BLOCK, 1, 0.0),
        (BLOCK, 0, 200.0),
    ],
)
def test_realized_cost_hand_computed(action, is_fraud, expected):
    got = realized_cost(action, np.array([is_fraud]), np.array([1000.0]), PARAMS)
    assert got[0] == pytest.approx(expected)


def test_cost_breakdown_partitions_total():
    # cost_breakdown's channels must sum to the direct realized total.
    amount = np.array([500.0, 2000.0, 1000.0])
    y = np.array([1, 0, 1])
    actions = np.array([APPROVE, REVIEW, STEP_UP])
    breakdown = cost_breakdown(actions, y, amount, PARAMS, realized=True)
    direct = realized_cost(actions, y, amount, PARAMS).sum()
    assert breakdown["total"] == pytest.approx(direct)
    parts = (
        breakdown["fraud_loss"]
        + breakdown["review_cost"]
        + breakdown["stepup_cost"]
        + breakdown["friction_cost"]
    )
    assert parts == pytest.approx(direct)


def test_closed_form_threshold_approve_vs_review():
    # c=1 and fixed A: intervene iff p > (R+F)/(alpha*A+fee+F).
    params = CostParams(
        alpha=0.85, fixed_fee=0.0, review_cost=75.0, friction_cost=150.0, catch_rate=1.0,
        stepup_cost=5.0, stepup_friction=40.0, stop_rate=0.6, block_friction=450.0,
    )
    amount = 2000.0
    threshold = (params.review_cost + params.friction_cost) / (
        params.alpha * amount + params.fixed_fee + params.friction_cost
    )
    grid = np.linspace(0.0, 1.0, 501)
    approve = expected_cost(APPROVE, grid, np.full_like(grid, amount), params)
    review = expected_cost(REVIEW, grid, np.full_like(grid, amount), params)
    # At the analytic threshold the two expected costs are equal.
    a = expected_cost(APPROVE, np.array([threshold]), np.array([amount]), params)[0]
    r = expected_cost(REVIEW, np.array([threshold]), np.array([amount]), params)[0]
    assert a == pytest.approx(r, abs=1e-6)
    # Below the threshold APPROVE is cheaper; above it REVIEW is cheaper.
    assert np.all(approve[grid < threshold] <= review[grid < threshold] + 1e-9)
    assert np.all(review[grid > threshold] <= approve[grid > threshold] + 1e-9)


def test_higher_probability_never_lowers_severity():
    severity = {APPROVE: 0, STEP_UP: 1, REVIEW: 2, BLOCK: 3}
    amount = 1500.0
    p_grid = np.linspace(0.0, 1.0, 2001)
    costs = np.stack(
        [expected_cost(a, p_grid, np.full_like(p_grid, amount), PARAMS) for a in range(4)],
        axis=1,
    )
    chosen = np.argmin(costs, axis=1)
    sev = np.array([severity[a] for a in chosen])
    assert np.all(np.diff(sev) >= 0)


def test_scenario_override_changes_params():
    from fraudlens.config import load_config

    cfg = load_config("dev")
    base = resolve_cost_params(cfg.simulation.costs, None)
    scen = cfg.simulation.cost_scenarios["high_fp_cost"]
    over = resolve_cost_params(cfg.simulation.costs, scen)
    assert over.friction_cost == scen.friction_cost
    assert over.review_cost == scen.review_cost
    assert over.alpha == base.alpha  # untouched field


def test_baselines_shapes():
    assert (approve_all(5) == APPROVE).all()
    assert (review_all(5) == REVIEW).all()
    actions = review_top_k(np.array([0.1, 0.9, 0.5, 0.7]), k=2)
    assert actions[np.argsort(-np.array([0.1, 0.9, 0.5, 0.7]))[:2]].tolist() == [REVIEW, REVIEW]
    assert (actions == APPROVE).sum() == 2
