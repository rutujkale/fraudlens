"""Tests for ``data_quality.validation`` (Part 03A).

A clean minimal dataset is built once; every test injects exactly one defect
and asserts the matching check fails.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from fraudlens.data_quality.validation import validate_dataset

START = pd.Timestamp("2026-01-01 00:00:00")
N_TX = 100


def _clean() -> dict[str, pd.DataFrame]:
    ts = pd.date_range(START, periods=N_TX, freq="h")
    tx_ids = [f"T{i:07d}" for i in range(1, N_TX + 1)]
    transactions = pd.DataFrame(
        {
            "transaction_id": tx_ids,
            "customer_id": ["C0000001"] * 50 + ["C0000002"] * 50,
            "account_id": ["A0000001"] * 50 + ["A0000002"] * 50,
            "timestamp": ts,
            "amount": [100.0] * N_TX,
            "currency": ["INR"] * N_TX,
            "merchant_id": ["M0000001"] * N_TX,
            "merchant_category": ["grocery"] * N_TX,
            "device_id": [None] * N_TX,
            "location_id": ["L0000001"] * N_TX,
            "recipient_id": [None] * N_TX,
            "transaction_type": ["PURCHASE"] * N_TX,
            "channel": ["POS"] * N_TX,
            "created_at": ts,
        }
    )
    labels = pd.DataFrame(
        {
            "transaction_id": tx_ids,
            "is_fraud": [False] * N_TX,
            "fraud_scenario": [None] * N_TX,
            "episode_id": [None] * N_TX,
            "stealth": [None] * N_TX,
            "hard_negative_type": [None] * N_TX,
            "label_available_at": list(ts + pd.to_timedelta(1, unit="D")),
        }
    )
    labels.loc[0, "is_fraud"] = True
    customers = pd.DataFrame(
        {
            "customer_id": ["C0000001", "C0000002"],
            "age": [30, 40],
            "income_band": ["mid", "high"],
            "home_region": ["West", "South"],
            "home_location_id": ["L0000001", "L0000002"],
            "account_open_date": ["2025-01-01", "2025-01-01"],
            "archetype": ["regular", "high_spend"],
            "baseline_daily_tx": [2.0, 3.0],
            "baseline_amount": [500.0, 800.0],
            "amount_sigma": [0.5, 0.6],
            "preferred_categories": [["grocery"], ["dining"]],
            "active_hour_mean": [14.0, 18.0],
            "active_hour_sd": [2.0, 2.5],
            "travel_frequency": [1.0, 2.0],
        }
    )
    accounts = pd.DataFrame(
        {
            "account_id": ["A0000001", "A0000002"],
            "customer_id": ["C0000001", "C0000002"],
            "account_type": ["SAVINGS", "SAVINGS"],
            "opening_date": ["2025-01-01", "2025-01-01"],
            "status": ["ACTIVE", "ACTIVE"],
            "base_currency": ["INR", "INR"],
        }
    )
    devices = pd.DataFrame(
        {
            "device_id": ["D0000001"],
            "first_seen": [START],
            "last_seen": [START],
            "device_type": ["MOBILE"],
            "is_shared": [False],
        }
    )
    customer_devices = pd.DataFrame(
        {"customer_id": ["C0000001"], "device_id": ["D0000001"]}
    )
    merchants = pd.DataFrame(
        {
            "merchant_id": ["M0000001"],
            "merchant_category": ["grocery"],
            "region": ["West"],
            "merchant_risk_group": ["low"],
        }
    )
    locations = pd.DataFrame(
        {
            "location_id": ["L0000001", "L0000002"],
            "country": ["India", "India"],
            "state": ["MH", "KA"],
            "city": ["Mumbai", "Bengaluru"],
            "latitude": [19.076, 12.9716],
            "longitude": [72.8777, 77.5946],
            "region_group": ["West", "South"],
        }
    )
    return {
        "transactions": transactions,
        "labels": labels,
        "customers": customers,
        "accounts": accounts,
        "devices": devices,
        "customer_devices": customer_devices,
        "merchants": merchants,
        "locations": locations,
    }


def _failed(tables: dict[str, pd.DataFrame], **kwargs) -> set[str]:
    return {c.name for c in validate_dataset(tables, **kwargs).failed}


@pytest.fixture()
def clean() -> dict[str, pd.DataFrame]:
    return _clean()


def test_clean_dataset_passes(clean):
    assert validate_dataset(clean).passed


def test_duplicate_transaction_id_fails(clean):
    clean["transactions"].loc[1, "transaction_id"] = "T0000001"
    assert "unique_transactions.transaction_id" in _failed(clean)


def test_duplicate_customer_id_fails(clean):
    clean["customers"].loc[1, "customer_id"] = "C0000001"
    assert "unique_customers.customer_id" in _failed(clean)


def test_null_required_column_fails(clean):
    clean["transactions"].loc[0, "account_id"] = None
    assert "no_null_required_columns" in _failed(clean)


def test_non_positive_amount_fails(clean):
    clean["transactions"].loc[0, "amount"] = 0.0
    assert "amount_positive" in _failed(clean)


def test_absurd_amount_fails(clean):
    clean["transactions"].loc[0, "amount"] = 2.0e7
    assert "amount_not_absurd" in _failed(clean)


def test_timestamp_out_of_window_fails(clean):
    clean["transactions"].loc[0, "timestamp"] = START - pd.to_timedelta(5, unit="D")
    win_end = START + pd.to_timedelta(10, unit="D")
    failed = _failed(clean, window_start=START, window_end=win_end)
    assert "transactions.timestamp_in_window" in failed


def test_unparsable_timestamp_fails(clean):
    clean["transactions"]["timestamp"] = clean["transactions"]["timestamp"].astype(object)
    clean["transactions"].loc[0, "timestamp"] = "not-a-date"
    assert "transactions.timestamp_parsable" in _failed(clean)


def test_invalid_channel_fails(clean):
    clean["transactions"].loc[0, "channel"] = "CARRIER_PIGEON"
    assert "valid_channel" in _failed(clean)


def test_invalid_category_fails(clean):
    clean["transactions"].loc[0, "merchant_category"] = "crypto"
    assert "valid_category" in _failed(clean)


def test_invalid_transaction_type_fails(clean):
    clean["transactions"].loc[0, "transaction_type"] = "WIRE"
    assert "valid_transaction_type" in _failed(clean)


def test_invalid_archetype_fails(clean):
    clean["customers"].loc[0, "archetype"] = "whale"
    assert "valid_archetype" in _failed(clean)


def test_invalid_income_band_fails(clean):
    clean["customers"].loc[0, "income_band"] = 0.3
    assert "valid_income_band" in _failed(clean)


def test_orphan_foreign_key_fails(clean):
    clean["transactions"].loc[0, "customer_id"] = "C9999999"
    assert "no_orphan_foreign_keys" in _failed(clean)


def test_label_missing_transaction_fails(clean):
    clean["labels"] = clean["labels"].iloc[:-1]
    assert "labels_cover_transactions_exactly_once" in _failed(clean)


def test_extra_label_fails(clean):
    extra = clean["labels"].iloc[[0]].copy()
    extra["transaction_id"] = "T9999999"
    clean["labels"] = pd.concat([clean["labels"], extra], ignore_index=True)
    assert "labels_cover_transactions_exactly_once" in _failed(clean)


def test_label_column_in_transactions_fails(clean):
    clean["transactions"]["is_fraud"] = False
    assert "no_label_columns_in_transactions" in _failed(clean)


def test_label_available_before_timestamp_fails(clean):
    clean["labels"].loc[0, "label_available_at"] = START - pd.to_timedelta(1, unit="D")
    assert "label_available_at_ge_timestamp" in _failed(clean)


def test_fraud_rate_zero_fails(clean):
    clean["labels"]["is_fraud"] = False
    assert "fraud_rate_sanity" in _failed(clean)


def test_fraud_rate_too_high_fails(clean):
    clean["labels"]["is_fraud"] = True
    assert "fraud_rate_sanity" in _failed(clean)


def test_report_to_dict_is_json_serialisable(clean):
    import json

    clean["transactions"].loc[0, "amount"] = -1.0
    report = validate_dataset(clean)
    json.dumps(report.to_dict())
    assert not report.passed


def test_independent_mutations_use_fresh_copy(clean):
    snapshot = copy.deepcopy(clean)
    clean["transactions"].loc[0, "amount"] = -5.0
    assert validate_dataset(snapshot).passed
