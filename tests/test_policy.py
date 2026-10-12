"""Tests for action selection, capacity and risk scoring (Part 07C/07D/07F)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fraudlens.evaluation.cost_model import (
    APPROVE,
    BLOCK,
    REVIEW,
    STEP_UP,
    CostParams,
    resolve_cost_params,
)
from fraudlens.evaluation.threshold_search import build_policy_comparison, daily_capacity
from fraudlens.risk.policy import (
    amount_aware_actions,
    enforce_capacity,
    rank_key,
    top_k_actions,
)
from fraudlens.risk.scoring import RISK_LEVELS, action_name, lift, risk_level, risk_score

PARAMS = CostParams(
    alpha=0.85, fixed_fee=5.0, review_cost=75.0, friction_cost=150.0, catch_rate=1.0,
    stepup_cost=5.0, stepup_friction=40.0, stop_rate=0.6, block_friction=450.0,
)


def test_amount_aware_returns_valid_actions():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    amount = rng.uniform(50, 100000, size=500)
    actions = amount_aware_actions(p, amount, PARAMS)
    assert set(np.unique(actions)) <= {APPROVE, STEP_UP, REVIEW, BLOCK}
    assert len(actions) == 500


def test_enforce_capacity_never_exceeds_per_day():
    rng = np.random.default_rng(1)
    n = 5000
    prob = rng.random(n)
    amount = rng.uniform(50, 50000, size=n)
    dates = rng.integers(0, 20, size=n)  # 20 days
    actions = amount_aware_actions(prob, amount, PARAMS)
    cap = 30
    result = enforce_capacity(dates, prob, amount, actions, PARAMS, cap)
    for day in np.unique(dates):
        assert (result.actions[dates == day] == REVIEW).sum() <= cap
    assert result.n_demoted >= 0
    assert result.n_demoted == sum(
        max((actions[dates == d] == REVIEW).sum() - cap, 0) for d in np.unique(dates)
    )


def test_enforce_capacity_keeps_highest_saving():
    # One day, 3 review-worthy rows, capacity 1 -> keep the highest expected saving.
    prob = np.array([0.6, 0.9, 0.7])
    amount = np.array([5000.0, 5000.0, 5000.0])
    dates = np.array([0, 0, 0])
    actions = np.array([REVIEW, REVIEW, REVIEW])
    result = enforce_capacity(dates, prob, amount, actions, PARAMS, capacity_per_day=1)
    assert (result.actions == REVIEW).sum() == 1
    assert result.actions[1] == REVIEW  # p=0.9 largest saving stays
    assert result.n_demoted == 2


def test_top_k_actions_caps_per_day():
    key = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.8])
    dates = np.array([0, 0, 0, 1, 1, 1])
    actions = top_k_actions(dates, key, capacity_per_day=2)
    assert (actions[dates == 0] == REVIEW).sum() == 2
    assert (actions[dates == 1] == REVIEW).sum() == 2
    assert actions[1] == REVIEW and actions[3] == REVIEW  # top of each day


def test_rank_key_variants():
    prob = np.array([0.2, 0.5, 0.1])
    amount = np.array([1000.0, 200.0, 5000.0])
    assert np.allclose(rank_key(prob, amount, PARAMS, "p"), prob)
    assert np.allclose(rank_key(prob, amount, PARAMS, "p_times_amount"), prob * amount)
    assert len(rank_key(prob, amount, PARAMS, "expected_saving")) == 3
    with pytest.raises(ValueError):
        rank_key(prob, amount, PARAMS, "nope")


def test_risk_levels_and_score():
    assert RISK_LEVELS[APPROVE] == "LOW"
    assert RISK_LEVELS[STEP_UP] == "MEDIUM"
    assert RISK_LEVELS[REVIEW] == "HIGH"
    assert RISK_LEVELS[BLOCK] == "CRITICAL"
    assert risk_score(0.1234) == 12.3
    assert risk_level(APPROVE) == "LOW"
    assert action_name(BLOCK) == "BLOCK"
    assert lift(0.10, 0.01) == pytest.approx(10.0)


def test_policy_comparison_smoke_and_columns():
    from fraudlens.config import load_config

    cfg = load_config("tiny")
    rng = np.random.default_rng(2)
    n = 3000
    prob = rng.random(n)
    amount = rng.uniform(50, 60000, size=n)
    dates = rng.integers(0, 15, size=n)
    y = (rng.random(n) < prob * 0.1).astype(int)
    params = resolve_cost_params(cfg.simulation.costs, None)
    df, scan, meta = build_policy_comparison(y, prob, amount, dates, params, cfg)
    expected = {
        "approve_all", "review_all", "f1_optimal", "cost_optimal_global",
        "cost_optimal_global_capacity", "amount_aware", "amount_aware_capacity",
        "top_k_by_p", "top_k_by_p_times_amount", "top_k_by_expected_saving",
    }
    assert expected == set(df["policy"])
    assert {"total_realized_cost", "total_expected_cost", "pct_cost_reduction_vs_f1"} <= set(df)
    assert meta["capacity_per_day"] >= 1
    cap, avg = daily_capacity(dates, cfg)
    assert meta["capacity_per_day"] == cap and avg == pytest.approx(n / 15)
    # json round-trip of the policy artefact payload
    policy = {"params": params.__dict__, "capacity_per_day": meta["capacity_per_day"]}
    restored = json.loads(json.dumps(policy, default=str))
    assert restored["capacity_per_day"] == meta["capacity_per_day"]
