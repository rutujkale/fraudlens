"""Hand-computed feature tests (Part 05C).

A tiny 2-customer / 6-transaction dataset with known geography (Mumbai & Delhi)
and known timestamps lets every feature group be checked against a value worked
out by hand: window edges, first-transaction cold start, haversine distance,
speed capping, the zero-std guard, new-device logic and the near-threshold count.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import load_config
from fraudlens.features.build_features import build_features
from fraudlens.features.registry import FEATURE_COLUMNS, FORBIDDEN_COLUMNS

MUM = {"location_id": "L_MUM", "latitude": 19.0760, "longitude": 72.8777, "region_group": "West"}
DEL = {"location_id": "L_DEL", "latitude": 28.7041, "longitude": 77.1025, "region_group": "North"}

# (transaction_id, customer, timestamp, amount, device, loc, category, merchant,
#  type, channel, recipient)
_ROWS = [
    (
        "T1",
        "C1",
        "2026-01-01 10:00",
        1000.0,
        "D1",
        "L_MUM",
        "grocery",
        "M1",
        "PURCHASE",
        "APP",
        None,
    ),
    (
        "T2",
        "C1",
        "2026-01-01 10:30",
        30000.0,
        "D1",
        "L_MUM",
        "grocery",
        "M1",
        "PURCHASE",
        "APP",
        None,
    ),
    (
        "T3",
        "C1",
        "2026-01-01 11:00",
        2000.0,
        "D_SHARED",
        "L_DEL",
        "electronics",
        "M2",
        "PURCHASE",
        "APP",
        None,
    ),
    (
        "T4",
        "C1",
        "2026-01-02 09:00",
        50000.0,
        "D_SHARED",
        "L_MUM",
        "grocery",
        "M1",
        "PURCHASE",
        "APP",
        None,
    ),
    (
        "T5",
        "C2",
        "2026-01-01 11:00",
        500.0,
        "D_SHARED",
        "L_MUM",
        "fuel",
        "M3",
        "PURCHASE",
        "POS",
        None,
    ),
    ("T6", "C2", "2026-01-01 12:00", 500.0, None, "L_MUM", "fuel", "M3", "BILL_PAY", "WEB", None),
]


@pytest.fixture(scope="module")
def cfg():
    return load_config("tiny")


@pytest.fixture(scope="module")
def entities():
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "home_location_id": ["L_MUM", "L_MUM"],
            "account_open_date": ["2025-12-01", "2025-12-01"],
        }
    )
    locations = pd.DataFrame([MUM, DEL])
    return SimpleNamespace(customers=customers, locations=locations)


@pytest.fixture(scope="module")
def feats(cfg, entities) -> pd.DataFrame:
    tx = pd.DataFrame(
        [
            {
                "transaction_id": r[0],
                "customer_id": r[1],
                "timestamp": pd.Timestamp(r[2]),
                "amount": r[3],
                "device_id": r[4],
                "location_id": r[5],
                "merchant_category": r[6],
                "merchant_id": r[7],
                "transaction_type": r[8],
                "channel": r[9],
                "recipient_id": r[10],
            }
            for r in _ROWS
        ]
    )
    out = build_features(tx, entities, cfg)
    return out.set_index("transaction_id")


def _get(feats, tid, col):
    return feats.loc[tid, col]


def test_shape_and_no_forbidden_columns(feats):
    assert list(feats.columns[:1]) == ["feature_as_of_timestamp"]
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    assert FORBIDDEN_COLUMNS.isdisjoint(feats.columns)


def test_transaction_features(feats):
    assert _get(feats, "T1", "amount") == 1000.0
    assert _get(feats, "T1", "log_amount") == pytest.approx(np.log1p(1000.0))
    assert _get(feats, "T1", "hour") == 10
    assert _get(feats, "T1", "day_of_week") == 3  # 2026-01-01 is a Thursday
    assert _get(feats, "T1", "is_weekend") == 0
    assert _get(feats, "T1", "is_night") == 0
    assert _get(feats, "T1", "transaction_type") == "PURCHASE"
    assert _get(feats, "T1", "channel") == "APP"
    assert _get(feats, "T1", "merchant_category") == "grocery"


def test_first_transaction_is_cold_start(feats):
    for col in (
        "tx_count_10m",
        "tx_count_1h",
        "tx_count_6h",
        "tx_count_24h",
        "tx_count_7d",
        "active_days_7d",
        "active_days_30d",
        "amount_sum_1h",
        "amount_sum_24h",
        "count_near_threshold_24h",
        "unique_recipients_1h",
        "unique_recipients_24h",
    ):
        assert _get(feats, "T1", col) == 0, col
    for col in (
        "minutes_since_prev_tx",
        "days_since_last_tx",
        "max_amount_24h",
        "avg_amount_7d",
        "avg_amount_30d",
        "std_amount_30d",
        "amount_zscore_30d",
        "distance_from_prev_km",
        "hour_deviation",
        "activity_change_7d_vs_prev_7d",
    ):
        assert np.isnan(_get(feats, "T1", col)), col
    assert _get(feats, "T1", "n_prior_tx") == 0
    assert _get(feats, "T1", "is_cold_start") == 1
    assert _get(feats, "T1", "history_days") == pytest.approx(
        31.0 + 10.0 / 24.0
    )  # 2025-12-01 -> 2026-01-01 10:00


def test_window_edges(feats):
    # T3 at 11:00; T1 at 10:00 is exactly T-1h and must be INCLUDED in [T-1h, T).
    assert _get(feats, "T3", "tx_count_1h") == 2
    assert _get(feats, "T3", "tx_count_10m") == 0
    assert _get(feats, "T4", "tx_count_1h") == 0  # next day, nothing in hour
    assert _get(feats, "T4", "tx_count_24h") == 3
    assert _get(feats, "T4", "active_days_7d") == 1  # only 2026-01-01 so far


def test_amount_sums_and_ratios(feats):
    assert _get(feats, "T3", "amount_sum_1h") == 31000.0
    assert _get(feats, "T3", "max_amount_24h") == 30000.0
    assert _get(feats, "T4", "amount_sum_24h") == 33000.0
    assert _get(feats, "T2", "amount_ratio_to_7d_mean") == pytest.approx(30.0)
    assert _get(feats, "T4", "amount_vs_lifetime_median") == pytest.approx(25.0)  # 50000/2000


def test_zero_std_guard(feats):
    # Single prior tx -> std is 0 -> z-score must use the configured floor, not inf/NaN.
    expected = (30000.0 - 1000.0) / 1e-6
    assert _get(feats, "T2", "amount_zscore_30d") == pytest.approx(expected)


def test_near_threshold_count(feats):
    # 30000 lies in [0.55, 1.0) * 50000 = [27500, 50000).
    assert _get(feats, "T3", "count_near_threshold_24h") == 1
    assert _get(feats, "T4", "count_near_threshold_24h") == 1


def test_new_device_logic(feats):
    assert _get(feats, "T1", "is_new_device") == 1
    assert _get(feats, "T2", "is_new_device") == 0
    assert _get(feats, "T2", "tx_from_device_prior") == 1
    assert _get(feats, "T3", "is_new_device") == 1  # first use of D_SHARED by C1
    assert _get(feats, "T4", "is_new_device") == 0
    assert _get(feats, "T4", "tx_from_device_prior") == 1


def test_null_device_is_nan(feats):
    for col in (
        "is_new_device",
        "device_age_days",
        "tx_from_device_prior",
        "accounts_seen_on_device",
        "device_count_30d",
    ):
        assert np.isnan(_get(feats, "T6", col)), col


def test_device_cross_customer_ties_and_age(feats):
    # C1 and C2 both use D_SHARED at 2026-01-01 11:00: same-timestamp rows must not
    # see each other, so both report 0 prior distinct accounts.
    assert _get(feats, "T3", "accounts_seen_on_device") == 0
    assert _get(feats, "T5", "accounts_seen_on_device") == 0
    # By T4 both C1 and C2 have used D_SHARED, so 2 distinct accounts; device age = 22h.
    assert _get(feats, "T4", "accounts_seen_on_device") == 2
    assert _get(feats, "T4", "device_age_days") == pytest.approx(22.0 / 24.0)
    assert np.isnan(_get(feats, "T3", "device_age_days"))  # first ever use


def test_geography_and_speed_cap(feats):
    assert _get(feats, "T1", "distance_from_home_km") == 0.0
    dist = _get(feats, "T3", "distance_from_home_km")
    assert 1140.0 < dist < 1180.0, dist  # Mumbai <-> Delhi great-circle ~1150 km
    assert _get(feats, "T3", "distance_from_prev_km") == pytest.approx(dist)
    # ~1150 km in 30 min is ~2300 km/h; must be capped at 1000.
    assert _get(feats, "T3", "travel_speed_kmh") == 1000.0
    assert _get(feats, "T4", "travel_speed_kmh") < 1000.0


def test_region_and_merchant_novelty(feats):
    assert _get(feats, "T1", "new_region_flag") == 1
    assert _get(feats, "T2", "new_region_flag") == 0
    assert _get(feats, "T3", "new_region_flag") == 1  # Delhi/North first time
    assert _get(feats, "T4", "new_region_flag") == 0
    assert _get(feats, "T3", "new_merchant_flag") == 1
    assert _get(feats, "T3", "new_category_flag") == 1
    assert _get(feats, "T3", "merchant_freq_30d") == 0.0
    assert _get(feats, "T4", "merchant_freq_30d") == pytest.approx(2.0 / 3.0)


def test_recipient_features(feats):
    assert _get(feats, "T5", "unique_recipients_1h") == 0
    assert np.isnan(_get(feats, "T5", "is_new_recipient"))  # NULL recipient
