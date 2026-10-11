"""Tests for the behavioral transaction/fraud simulator (Part 02).

Runs on the `tiny` profile so the suite stays fast. A drift-enabled variant of
`tiny` is built on the fly for the drift assertions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import chi2_contingency

from fraudlens import config as cfg_module
from fraudlens.data_gen.generate import (
    FORBIDDEN_COLUMNS,
    TX_COLUMNS,
    GeneratedResult,
    generate_profile,
)

SCENARIOS = {"burst", "account_takeover", "amount_anomaly", "geo_anomaly", "structuring"}
HN_TYPES = {
    "new_phone", "travel_trip", "holiday_splurge", "medical_bill",
    "emergency_odd_hour", "payday_burst", "power_user_burst",
}
THRESHOLD = 50000.0


@pytest.fixture(scope="module")
def result() -> GeneratedResult:
    return generate_profile("tiny")


@pytest.fixture()
def drift_result(tmp_path, monkeypatch) -> GeneratedResult:
    with open(cfg_module._config_path(), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["profiles"]["tiny"] = {
        "customers": 300,
        "days": 30,
        "target_fraud_rate": 0.05,
        "label_delay_days": [1, 3],
        "drift": {"start_fraction": 0.5, "strength": 1.0},
    }
    cfg_file = tmp_path / "config_drift.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("FRAUDLENS_CONFIG_PATH", str(cfg_file))
    return generate_profile("tiny")


def test_determinism():
    a = generate_profile("tiny")
    b = generate_profile("tiny")
    assert a.transactions.equals(b.transactions)
    assert a.labels.equals(b.labels)
    assert a.customers.equals(b.customers)


def test_different_seed_differs():
    a = generate_profile("tiny", seed=1)
    b = generate_profile("tiny", seed=2)
    assert not a.transactions["amount"].equals(b.transactions["amount"])


def test_amounts_positive_and_sorted():
    tx = generate_profile("tiny").transactions
    assert (tx["amount"] > 0).all()
    assert tx["timestamp"].is_monotonic_increasing


def test_transactions_schema(result):
    assert list(result.transactions.columns) == TX_COLUMNS
    assert result.transactions["transaction_id"].is_unique
    assert result.transactions["transaction_id"].str.match(r"^T\d{7}$").all()


def test_labels_schema_and_one_to_one(result):
    tx_ids = set(result.transactions["transaction_id"])
    assert result.labels["transaction_id"].is_unique
    assert set(result.labels["transaction_id"]) == tx_ids
    assert result.labels["is_fraud"].dtype == bool


def test_no_label_columns_in_transactions(result):
    assert not FORBIDDEN_COLUMNS.intersection(result.transactions.columns)


def test_referential_integrity(result):
    tx = result.transactions
    assert tx["customer_id"].isin(result.customers["customer_id"]).all()
    assert tx["account_id"].isin(result.accounts["account_id"]).all()
    has_m = tx["merchant_id"].notna()
    assert tx.loc[has_m, "merchant_id"].isin(result.merchants["merchant_id"]).all()
    has_d = tx["device_id"].notna()
    assert tx.loc[has_d, "device_id"].isin(result.devices["device_id"]).all()
    assert tx["location_id"].isin(result.locations["location_id"]).all()
    has_r = tx["recipient_id"].notna()
    assert tx.loc[has_r, "recipient_id"].isin(result.customers["customer_id"]).all()


def test_strict_per_customer_timestamp_order(result):
    ts = result.transactions[["customer_id", "timestamp"]].copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"])
    ts = ts.sort_values("timestamp", kind="stable")
    diffs = ts.groupby("customer_id")["timestamp"].diff().dropna()
    assert (diffs > pd.Timedelta(0)).all()


def test_fraud_rate_within_tolerance(result):
    target = result.report["target_fraud_rate"]
    assert abs(result.report["fraud_rate"] - target) <= 0.10 * target


def test_all_scenarios_present(result):
    assert set(result.labels["fraud_scenario"].dropna().unique()) == SCENARIOS


def test_all_hard_negative_types_present(result):
    present = set(result.labels["hard_negative_type"].dropna().unique())
    assert present == HN_TYPES


def test_label_delay_ranges_and_legit_immediate(result):
    merged = result.transactions[["transaction_id", "timestamp"]].merge(
        result.labels, on="transaction_id"
    )
    merged["delay_days"] = (
        pd.to_datetime(merged["label_available_at"]) - pd.to_datetime(merged["timestamp"])
    ).dt.days
    fraud = merged[merged["is_fraud"]]
    assert fraud["delay_days"].between(1, 3).all()
    legit = merged[~merged["is_fraud"]]
    assert (legit["delay_days"] == 0).all()


def test_structuring_below_threshold_and_aggregate_above(result):
    merged = result.transactions[["transaction_id", "amount"]].merge(
        result.labels, on="transaction_id"
    )
    struct = merged[merged["fraud_scenario"] == "structuring"]
    assert not struct.empty
    ep = struct.groupby("episode_id")["amount"]
    assert (ep.sum() >= 1.5 * THRESHOLD - 1e-6).all()
    assert (struct["amount"] < THRESHOLD).all()


def test_merchant_risk_group_values(result):
    assert set(result.merchants["merchant_risk_group"].unique()) == {"low", "medium", "high"}


def test_merchant_risk_group_independent_of_fraud(result):
    merged = result.transactions[["transaction_id", "merchant_id"]].merge(
        result.labels[["transaction_id", "is_fraud"]], on="transaction_id"
    ).merge(result.merchants[["merchant_id", "merchant_risk_group"]], on="merchant_id", how="inner")
    table = pd.crosstab(merged["merchant_risk_group"], merged["is_fraud"])
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n else 0.0
    assert cramers_v < 0.15


def test_report_has_required_sections(result):
    report = result.report
    for key in (
        "profile", "seed", "git_commit", "config_hash", "date_range",
        "n_transactions", "n_fraud_transactions", "n_fraud_episodes",
        "fraud_rate", "scenarios", "drift", "label_delay_days", "hard_negative",
        "tx_per_archetype", "fraud_per_archetype",
    ):
        assert key in report
    assert len(report["config_hash"]) == 64


def test_drift_shifts_scenario_mix(drift_result):
    scenarios = drift_result.report["scenarios"]
    struct = scenarios["structuring"]
    burst = scenarios["burst"]
    assert struct["episodes"] > 0
    assert struct["episodes_period_b"] > struct["episodes_period_a"]
    assert burst["episodes_period_a"] > burst["episodes_period_b"]


def test_drift_increases_mean_stealth(drift_result):
    merged = drift_result.transactions[["transaction_id", "timestamp"]].merge(
        drift_result.labels, on="transaction_id"
    )
    fraud = merged[merged["is_fraud"]].copy()
    start = pd.Timestamp(drift_result.report["drift"]["start_date"])
    fraud["period"] = (pd.to_datetime(fraud["timestamp"]) >= start).map({True: "B", False: "A"})
    mean_a = fraud.loc[fraud["period"] == "A", "stealth"].mean()
    mean_b = fraud.loc[fraud["period"] == "B", "stealth"].mean()
    assert mean_b > mean_a