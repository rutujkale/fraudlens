"""Rules-baseline tests (Part 06D)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlens.config import load_config
from fraudlens.models.rules import RuleEngine, rules_markdown


def _train_legit(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "tx_count_10m": rng.integers(0, 4, n),
            "amount_ratio_to_30d_mean": rng.uniform(0.2, 2.0, n),
            "is_new_device": np.zeros(n, dtype=int),
            "travel_speed_kmh": rng.uniform(0, 100, n),
            "is_night": np.zeros(n, dtype=int),
            "new_category_flag": np.zeros(n, dtype=int),
            "count_near_threshold_24h": np.zeros(n, dtype=int),
        }
    )


def test_score_bounds_and_fires():
    engine = RuleEngine(load_config("tiny").simulation.rules).fit(_train_legit())
    clean = _train_legit().head(5)
    assert (engine.score(clean) == 0).all()

    suspicious = pd.DataFrame(
        {
            "tx_count_10m": [10],
            "amount_ratio_to_30d_mean": [50.0],
            "is_new_device": [1],
            "travel_speed_kmh": [2000.0],
            "is_night": [1],
            "new_category_flag": [1],
            "count_near_threshold_24h": [5],
        }
    )
    s = engine.score(suspicious)
    assert s.iloc[0] == 1.0  # every rule fires
    fired = engine.explain(suspicious.iloc[0])
    assert set(fired) == set(engine.cfg.rules)
    assert 0.0 <= engine.score(_train_legit()).max() <= 1.0


def test_percentile_threshold_from_legit_only():
    train = _train_legit(1000)
    engine = RuleEngine(load_config("tiny").simulation.rules).fit(train)
    # velocity_10m threshold is a percentile of the legit column, so < a generous cap
    assert engine.thresholds["velocity_10m"] >= 3
    assert engine.thresholds["travel_speed"] == 800.0  # domain threshold unchanged


def test_rules_markdown_lists_all_rules():
    engine = RuleEngine(load_config("tiny").simulation.rules).fit(_train_legit())
    md = rules_markdown(engine, "tiny")
    for name in engine.cfg.rules:
        assert name in md
