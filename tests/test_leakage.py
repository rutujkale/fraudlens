"""Leakage test suite (Part 05D).

Proves features use only strictly-prior information:
1 truncation invariance, 2 future-mutation invariance, 3 current-row exclusion
(plus a brute-force count cross-check), 4 same-timestamp ties, 5 no target
columns, 6 train/validation boundary, 7 label independence, 8 canary (the tests
detect deliberately leaky windows), 9 training/serving parity, 10 SQL
cross-check, and a guard that ``np.searchsorted`` lives in exactly one module.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import fraudlens.features
import fraudlens.features.windows as W
from fraudlens.config import load_config
from fraudlens.evaluation.leakage import truncation_invariance
from fraudlens.features.build_features import build_features
from fraudlens.features.registry import (
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    assert_no_target_columns,
)
from fraudlens.features.serving import compute_features_for_transaction

COLS = ["transaction_id"] + FEATURE_COLUMNS

_LOCATIONS = pd.DataFrame(
    [
        {"location_id": "L0", "latitude": 19.0760, "longitude": 72.8777, "region_group": "West"},
        {"location_id": "L1", "latitude": 28.7041, "longitude": 77.1025, "region_group": "North"},
        {"location_id": "L2", "latitude": 18.5204, "longitude": 73.8567, "region_group": "West"},
    ]
)


@pytest.fixture(scope="module")
def cfg():
    return load_config("tiny")


def _entities(customers: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(customers=customers, locations=_LOCATIONS)


def _default_customers(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "home_location_id": [_LOCATIONS["location_id"].iloc[i % 3] for i in range(n)],
            "account_open_date": ["2025-12-01"] * n,
        }
    )


def _random_dataset(seed: int = 0, n_customers: int = 3, n_tx: int = 120):
    """Deterministic random transactions + entities (no simulator dependency)."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-03-01 00:00")
    ts = base + pd.to_timedelta(rng.integers(0, 6 * 24 * 60, size=n_tx), unit="m")
    tx = pd.DataFrame(
        {
            "transaction_id": [f"X{i:05d}" for i in range(n_tx)],
            "customer_id": [f"C{i}" for i in rng.integers(0, n_customers, size=n_tx)],
            "timestamp": ts,
            "amount": np.round(rng.uniform(100, 80000, size=n_tx), 2),
            "device_id": rng.choice([None, "D0", "D1", "D2"], size=n_tx),
            "location_id": rng.choice(_LOCATIONS["location_id"], size=n_tx),
            "merchant_category": rng.choice(["grocery", "fuel", "electronics", None], size=n_tx),
            "merchant_id": rng.choice(["M0", "M1", "M2", None], size=n_tx),
            "transaction_type": rng.choice(["PURCHASE", "P2P_TRANSFER", "BILL_PAY"], size=n_tx),
            "channel": rng.choice(["APP", "WEB", "POS", "ATM"], size=n_tx),
            "recipient_id": rng.choice([None, "C0", "C1", "C2"], size=n_tx),
        }
    )
    return tx, _entities(_default_customers(n_customers))


# --- reusable core assertions (also used by the canary) -----------------------


def _assert_truncation_invariance(tx, entities, cfg, cut_ts):
    result = truncation_invariance(tx, entities, cfg, cut_ts)
    assert result["passed"], result


def _assert_future_invariance(tx, entities, cfg, cut_ts, rng):
    cut = pd.Timestamp(cut_ts)
    full = build_features(tx, entities, cfg)
    mut = tx.copy()
    after = mut["timestamp"] > cut
    n_after = int(after.sum())
    mut.loc[after, "amount"] = rng.uniform(1, 1_000_000, size=n_after)
    mut.loc[after, "device_id"] = rng.choice(["D7", "D8"], size=n_after)
    mut.loc[after, "location_id"] = rng.choice(_LOCATIONS["location_id"], size=n_after)
    mut = mut.drop(index=mut.index[after][rng.random(n_after) < 0.5])
    mutated = build_features(mut, entities, cfg)
    a = (
        full[full["feature_as_of_timestamp"] <= cut]
        .sort_values("transaction_id")
        .reset_index(drop=True)
    )
    b = (
        mutated[mutated["feature_as_of_timestamp"] <= cut]
        .sort_values("transaction_id")
        .reset_index(drop=True)
    )
    assert set(a["transaction_id"]) == set(b["transaction_id"])
    assert_frame_equal(a[COLS], b[COLS], check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)


def _assert_current_row_exclusion(tx, entities, cfg):
    f = build_features(tx, entities, cfg)
    cust = dict(zip(tx["transaction_id"], tx["customer_id"]))
    f = f.assign(_cust=f["transaction_id"].map(cust))
    first_idx = f.groupby("_cust")["feature_as_of_timestamp"].idxmin()
    firsts = f.loc[first_idx]
    assert (firsts["n_prior_tx"] == 0).all()
    assert (firsts["tx_count_1h"] == 0).all()
    assert (firsts["is_cold_start"] == 1).all()


def _tie_frame():
    base = pd.Timestamp("2026-02-01 10:00")
    return pd.DataFrame(
        [
            {
                "transaction_id": "Y1",
                "customer_id": "C0",
                "timestamp": base,
                "amount": 100.0,
                "device_id": "D0",
                "location_id": "L0",
                "merchant_category": "grocery",
                "merchant_id": "M0",
                "transaction_type": "PURCHASE",
                "channel": "APP",
                "recipient_id": None,
            },
            {
                "transaction_id": "Y2",
                "customer_id": "C0",
                "timestamp": base,
                "amount": 200.0,
                "device_id": "D0",
                "location_id": "L0",
                "merchant_category": "grocery",
                "merchant_id": "M0",
                "transaction_type": "PURCHASE",
                "channel": "APP",
                "recipient_id": None,
            },
            {
                "transaction_id": "Y3",
                "customer_id": "C0",
                "timestamp": base + pd.to_timedelta(30, unit="m"),
                "amount": 300.0,
                "device_id": "D0",
                "location_id": "L0",
                "merchant_category": "grocery",
                "merchant_id": "M0",
                "transaction_type": "PURCHASE",
                "channel": "APP",
                "recipient_id": None,
            },
        ]
    )


def _assert_ties_excluded(entities, cfg):
    f = build_features(_tie_frame(), entities, cfg).set_index("transaction_id")
    assert f.loc["Y1", "tx_count_1h"] == 0
    assert f.loc["Y2", "tx_count_1h"] == 0
    assert f.loc["Y3", "tx_count_1h"] == 2


# --- the actual tests ---------------------------------------------------------


def test_1_truncation_invariance(cfg):
    tx, ent = _random_dataset(seed=11)
    for cut in ("2026-03-02 12:00", "2026-03-04 00:00", "2026-03-05 18:30"):
        _assert_truncation_invariance(tx, ent, cfg, cut)


def test_2_future_mutation_invariance(cfg):
    tx, ent = _random_dataset(seed=12)
    _assert_future_invariance(tx, ent, cfg, "2026-03-03 00:00", np.random.default_rng(7))


def test_3_current_row_exclusion_and_brute_force(cfg):
    tx, ent = _random_dataset(seed=13, n_tx=150)
    _assert_current_row_exclusion(tx, ent, cfg)

    # Brute-force O(n^2) reference for tx_count_1h.
    ordered = tx.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    window = pd.to_timedelta(60, unit="m")
    expected = {}
    for _, grp in ordered.groupby("customer_id"):
        times = grp["timestamp"].tolist()
        ids = grp["transaction_id"].tolist()
        for i, (t, tid) in enumerate(zip(times, ids, strict=True)):
            expected[tid] = sum(1 for tt in times[:i] if (t - window) <= tt < t)
    got = build_features(tx, ent, cfg).set_index("transaction_id")["tx_count_1h"]
    for tid, count in expected.items():
        assert got.loc[tid] == count, tid


def test_4_same_timestamp_ties(cfg):
    _assert_ties_excluded(_entities(_default_customers(1)), cfg)


def test_5_no_target_columns(cfg):
    tx, ent = _random_dataset(seed=14)
    feat = build_features(tx, ent, cfg)
    assert_no_target_columns(feat)
    assert set(FEATURE_COLUMNS).isdisjoint(FORBIDDEN_COLUMNS)

    poisoned = tx.copy()
    poisoned["is_fraud"] = 0
    with pytest.raises(AssertionError):
        build_features(poisoned, ent, cfg)


def test_6_train_validation_boundary(cfg):
    tx, ent = _random_dataset(seed=15, n_tx=400)
    lo, hi = tx["timestamp"].min(), tx["timestamp"].max()
    cut = lo + (hi - lo) * cfg.simulation.splits.train_end_fraction
    _assert_truncation_invariance(tx, ent, cfg, cut)


def test_7_label_independence(cfg):
    from fraudlens.data_gen.generate import generate_profile

    result = generate_profile("tiny")
    ent = SimpleNamespace(customers=result.customers, locations=result.locations)
    f1 = build_features(result.transactions, ent, cfg)

    labels = result.labels.copy()
    rng = np.random.default_rng(0)
    labels["is_fraud"] = rng.permutation(labels["is_fraud"].to_numpy())
    labels["fraud_scenario"] = rng.permutation(labels["fraud_scenario"].to_numpy())
    f2 = build_features(result.transactions, ent, cfg)
    assert_frame_equal(f1, f2, check_dtype=False)


def test_8_canary_tests_detect_leakage(cfg, monkeypatch):
    tx, ent = _random_dataset(seed=16, n_tx=150)
    tie_ent = _entities(_default_customers(1))

    def _leaky_inclusive(ts, t, w):
        return int(np.searchsorted(ts, t - w, side="left")), int(
            np.searchsorted(ts, t, side="right")
        )

    def _leaky_future(ts, t, w):
        return int(np.searchsorted(ts, t - w, side="left")), int(
            np.searchsorted(ts, t + w, side="right")
        )

    def _leaky_lifetime(ts, t, w):
        return 0, len(ts)

    # future / lifetime leaks must be caught by tests 1 and 2.
    for leak in (_leaky_future, _leaky_lifetime):
        monkeypatch.setattr(W, "window_bounds", leak)
        with pytest.raises(AssertionError):
            _assert_truncation_invariance(tx, ent, cfg, "2026-03-04 00:00")
        with pytest.raises(AssertionError):
            _assert_future_invariance(tx, ent, cfg, "2026-03-04 00:00", np.random.default_rng(1))

    # an inclusive window leaks the current row / ties: caught by tests 3 and 4.
    monkeypatch.setattr(W, "window_bounds", _leaky_inclusive)
    with pytest.raises(AssertionError):
        _assert_current_row_exclusion(tx, ent, cfg)
    with pytest.raises(AssertionError):
        _assert_ties_excluded(tie_ent, cfg)


def test_9_training_serving_parity(cfg):
    tx, ent = _random_dataset(seed=17, n_tx=400)
    offline = build_features(tx, ent, cfg)
    ordered = tx.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    rng = np.random.default_rng(3)
    sample = rng.choice(len(ordered), size=220, replace=False)
    for i in sample:
        row = ordered.iloc[i].to_dict()
        history = ordered[ordered["timestamp"] < row["timestamp"]]
        served = compute_features_for_transaction(row, history, ent, cfg)
        expected = offline[offline["transaction_id"] == row["transaction_id"]].iloc[0]
        assert served["transaction_id"] == row["transaction_id"]
        for col in FEATURE_COLUMNS:
            a, b = expected[col], served[col]
            if isinstance(a, str) or (a is None):
                assert a == b, (col, a, b)
            elif pd.isna(a):
                assert pd.isna(b), (col, a, b)
            else:
                assert b == pytest.approx(a), (col, a, b)


def test_searchsorted_lives_in_windows_only():
    feat_dir = Path(fraudlens.features.__file__).parent
    offenders = [
        p.name
        for p in feat_dir.glob("*.py")
        if p.name != "windows.py" and "searchsorted(" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"searchsorted( used outside windows.py: {offenders}"
    assert "searchsorted(" in (feat_dir / "windows.py").read_text(encoding="utf-8")


@pytest.mark.db
def test_10_sql_cross_check(cfg):
    from sqlalchemy import text

    from fraudlens.data_quality.validation import load_raw_tables
    from fraudlens.db.loader import get_engine, load_all

    tables = load_raw_tables("tiny")
    if not tables or "transactions" not in tables:
        pytest.skip("tiny parquet missing; run scripts/generate_data.py --profile tiny")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not reachable: {exc}")

    load_all("tiny", reset=True, write_explain=False)
    ent = SimpleNamespace(customers=tables["customers"], locations=tables["locations"])
    feat = build_features(tables["transactions"], ent, cfg)
    sql = pd.read_sql(
        "SELECT transaction_id, prior_24h_count FROM curated.rolling_24h_sql_demo", engine
    )
    merged = feat.merge(sql, on="transaction_id")
    mismatches = int((merged["tx_count_24h"] != merged["prior_24h_count"]).sum())
    assert mismatches == 0, f"{mismatches} rows differ between Python and SQL 24h counts"
