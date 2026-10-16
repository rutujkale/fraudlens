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
        "target_fraud_rate": 0.12,
        "label_delay_days": [1, 3],
        "drift": {"start_fraction": 0.5, "strength": 1.0},
    }
    cfg_file = tmp_path / "config_drift.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("FRAUDLENS_CONFIG_PATH", str(cfg_file))
    return generate_profile("tiny")


@pytest.fixture()
def coverage_result(tmp_path, monkeypatch) -> GeneratedResult:
    """tiny at a high fraud rate so every scenario / hard-negative type appears.

    At the normal 2% rate a 30-day window yields ~15 fraud episodes, which is too
    few to guarantee all five scenarios (Part 02b: the device-reuse change moved
    the RNG stream and dropped ``geo_anomaly`` from tiny); the raised
    hard-negative share likewise lifts the hard-negative episode count. This
    fixture tests the generator's coverage, not the tiny profile's own mix.
    """
    with open(cfg_module._config_path(), encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["common"]["hard_negative_target_share"] = 0.12
    raw["profiles"]["tiny"] = {
        "customers": 300,
        "days": 30,
        "target_fraud_rate": 0.15,
        "label_delay_days": [1, 3],
        "drift": {"start_fraction": None, "strength": 0.0},
    }
    cfg_file = tmp_path / "config_coverage.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("FRAUDLENS_CONFIG_PATH", str(cfg_file))
    return generate_profile("tiny")


def test_determinism():
    a = generate_profile("tiny")
    b = generate_profile("tiny")
    assert a.transactions.equals(b.transactions)
    assert a.labels.equals(b.labels)
    assert a.customers.equals(b.customers)


def test_no_transactions_before_account_open(result):
    """Part 02b cold-start gate: nothing may precede the account's opening."""
    opened = pd.to_datetime(result.customers.set_index("customer_id")["account_open_date"])
    ts = pd.to_datetime(result.transactions["timestamp"])
    open_ts = result.transactions["customer_id"].map(opened)
    assert (ts >= open_ts).all()


def test_normal_rows_for_open_customers_unchanged():
    """The gate only drops pre-opening days for mid-window accounts.

    For customers already open when the window starts, the normal-stage rows are
    byte-identical to the ungated generator. No identity is claimed for the
    hard-negative/fraud stages, which consume the RNG stream differently
    (Part 02b).
    """
    import dataclasses

    from fraudlens.config import load_config
    from fraudlens.data_gen.entities import create_entities, make_device_registry
    from fraudlens.data_gen.normal import generate_normal, open_day_index

    sim = load_config("tiny").simulation
    ent = create_entities(np.random.default_rng(sim.seed), sim)
    devreg = make_device_registry(ent.devices, ent.customer_devices)

    gated = generate_normal(np.random.default_rng(sim.seed), sim, ent, devreg, counters={})

    flat = dataclasses.replace(
        ent, customers=ent.customers.assign(account_open_date=sim.start_date.isoformat())
    )
    ungated = generate_normal(np.random.default_rng(sim.seed), sim, flat, devreg)

    open_day = open_day_index(ent.customers, sim.start_date)
    open_customers = set(ent.customers.loc[open_day == 0, "customer_id"])
    assert len(open_customers) < len(ent.customers)  # the gate has something to do

    def keep(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["customer_id"].isin(open_customers)].reset_index(drop=True)

    a, b = keep(gated.tx_frame()), keep(ungated.tx_frame())
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)


def test_episode_falls_back_when_account_opens_after_window():
    """Customers whose accounts open after the window cannot host an episode.

    Half the customer base opens too late; every episode must fall back to a
    customer that is open, exercising the gate's fallback path (Part 02b).
    """
    import dataclasses

    from fraudlens.config import load_config
    from fraudlens.data_gen.entities import create_entities, make_device_registry
    from fraudlens.data_gen.fraud import generate_fraud
    from fraudlens.data_gen.hard_negatives import generate_hard_negatives
    from fraudlens.data_gen.normal import generate_normal

    sim = load_config("tiny").simulation
    ent = create_entities(np.random.default_rng(sim.seed), sim)
    late = (pd.Timestamp(sim.start_date) + pd.Timedelta(days=sim.days + 30)).isoformat()
    customers = ent.customers.copy()
    too_late = customers.index % 2 == 1
    customers.loc[too_late, "account_open_date"] = late
    customers.loc[~too_late, "account_open_date"] = pd.Timestamp(sim.start_date).isoformat()
    ent = dataclasses.replace(ent, customers=customers)
    devreg = make_device_registry(ent.devices, ent.customer_devices)

    gate = {"normal_rows_masked": 0, "episode_day_redraws": 0, "episode_customer_fallbacks": 0}
    normal = generate_normal(np.random.default_rng(sim.seed), sim, ent, devreg, counters=gate)
    hard = generate_hard_negatives(
        np.random.default_rng(sim.seed + 1), sim, ent, devreg, len(normal),
        normal.tx_times(), counters=gate,
    )
    fraud = generate_fraud(
        np.random.default_rng(sim.seed + 2), sim, ent, devreg,
        len(normal) + len(hard), normal.tx_times(), counters=gate,
    )

    open_customers = set(customers.loc[~too_late, "customer_id"])
    assert gate["episode_customer_fallbacks"] > 0
    assert len(fraud.tx_frame()) > 0
    # low-activity customers can legitimately have zero normal rows, hence subset
    assert set(normal.tx_frame()["customer_id"]) <= open_customers
    assert set(hard.tx_frame()["customer_id"]) <= open_customers
    assert set(fraud.tx_frame()["customer_id"]) <= open_customers


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


def test_all_scenarios_present(coverage_result):
    assert set(coverage_result.labels["fraud_scenario"].dropna().unique()) == SCENARIOS


def test_all_hard_negative_types_present(coverage_result):
    present = set(coverage_result.labels["hard_negative_type"].dropna().unique())
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