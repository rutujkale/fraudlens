"""Orchestrate data generation for a profile and persist parquet outputs.

Order: entities -> normal -> hard negatives -> fraud -> merge/sort -> ids ->
labels -> guard -> write parquet + simulation_report.json.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraudlens.config import Config, load_config
from fraudlens.data_gen import drift
from fraudlens.data_gen.entities import (
    DeviceRegistry,
    Entities,
    create_entities,
    make_device_registry,
)
from fraudlens.data_gen.fraud import generate_fraud
from fraudlens.data_gen.hard_negatives import generate_hard_negatives
from fraudlens.data_gen.labels import build_labels
from fraudlens.data_gen.normal import RowBuffer, generate_normal
from fraudlens.paths import raw_data_dir
from fraudlens.utils import get_logger, git_commit

logger = get_logger(__name__)

TX_COLUMNS = [
    "transaction_id", "customer_id", "account_id", "timestamp", "amount",
    "currency", "merchant_id", "merchant_category", "device_id",
    "location_id", "recipient_id", "transaction_type", "channel", "created_at",
]
LABEL_COLUMNS = [
    "transaction_id", "is_fraud", "fraud_scenario", "episode_id", "stealth",
    "hard_negative_type", "label_available_at",
]

# Target-derived columns that must never leak into the transactions table.
FORBIDDEN_COLUMNS = {
    "is_fraud", "fraud_scenario", "episode_id", "stealth", "hard_negative_type",
    "label_available_at",
}


@dataclass
class GeneratedResult:
    transactions: pd.DataFrame
    labels: pd.DataFrame
    customers: pd.DataFrame
    accounts: pd.DataFrame
    devices: pd.DataFrame
    customer_devices: pd.DataFrame
    merchants: pd.DataFrame
    locations: pd.DataFrame
    report: dict[str, Any]


def _stamp_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["transaction_id"] = [f"T{i:07d}" for i in range(1, len(df) + 1)]
    return df


def _ensure_strictly_increasing(df: pd.DataFrame) -> pd.DataFrame:
    """Break per-customer timestamp ties by nudging later rows +1 minute."""
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = df["timestamp"].to_numpy()
    groups = df.groupby("customer_id", sort=False).indices
    for idx in groups.values():
        prev = None
        for i in idx:
            if prev is not None and ts[i] <= prev:
                ts[i] = prev + np.timedelta64(1, "m")
            prev = ts[i]
    df["timestamp"] = ts
    return df


def _merge_buffers(buffers: list[RowBuffer]) -> pd.DataFrame:
    combined = RowBuffer()
    for b in buffers:
        combined.extend(b)
    tx = combined.tx_frame()
    meta = combined.label_frame()
    for col in meta.columns:
        if col != "timestamp":
            tx[col] = meta[col].to_numpy()
    return tx


def _build_report(
    cfg: Config, transactions: pd.DataFrame, labels: pd.DataFrame, ent: Entities
) -> dict[str, Any]:
    sim = cfg.simulation
    total = len(transactions)
    n_fraud = int(labels["is_fraud"].sum())
    drift_date = drift.drift_start_date(sim)
    has_drift = drift_date is not None and sim.drift.strength > 0

    merged = transactions[["transaction_id", "customer_id", "timestamp", "amount"]].merge(
        labels, on="transaction_id", how="left"
    )
    fraud = merged[merged["is_fraud"]].copy()
    fraud["period"] = "A"
    if has_drift:
        fraud["period"] = np.where(fraud["timestamp"] >= pd.Timestamp(drift_date), "B", "A")
    episodes = (
        fraud.groupby("episode_id")
        .agg(scenario=("fraud_scenario", "first"), period=("period", "first"))
        .reset_index()
    )

    def _counts(frame: pd.DataFrame, key: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in frame[key].value_counts().items()}

    n_episodes = int(len(episodes))
    period_episodes = episodes["period"].value_counts().to_dict()
    period_tx = fraud["period"].value_counts().to_dict()
    scenario_summary: dict[str, Any] = {}
    for scen in episodes["scenario"].unique():
        eps = episodes[episodes["scenario"] == scen]
        tx = fraud[fraud["fraud_scenario"] == scen]
        ep_period = eps["period"].value_counts().to_dict()
        tx_period = tx["period"].value_counts().to_dict()
        scenario_summary[str(scen)] = {
            "episodes": int(len(eps)),
            "transactions": int(len(tx)),
            "episode_share": round(len(eps) / max(1, n_episodes), 6),
            "transaction_share": round(len(tx) / max(1, n_fraud), 6),
            "episodes_period_a": int(ep_period.get("A", 0)),
            "episodes_period_b": int(ep_period.get("B", 0)),
            "transactions_period_a": int(tx_period.get("A", 0)),
            "transactions_period_b": int(tx_period.get("B", 0)),
            "amount_total": round(float(tx["amount"].sum()), 2) if len(tx) else 0.0,
        }

    arch_by_cust = dict(zip(ent.customers["customer_id"], ent.customers["archetype"]))
    merged["archetype"] = merged["customer_id"].map(arch_by_cust)
    tx_per_arch = _counts(merged, "archetype")
    fraud_per_arch = _counts(merged[merged["is_fraud"]], "archetype")

    legit = merged[~merged["is_fraud"]]
    hn_counts = _counts(legit[legit["hard_negative_type"].notna()], "hard_negative_type")
    n_legit = int(len(legit))
    hn_share = {k: round(v / max(1, n_legit), 6) for k, v in hn_counts.items()}

    return {
        "profile": sim.profile,
        "dataset_version": sim.dataset_version,
        "seed": sim.seed,
        "git_commit": git_commit(),
        "date_range": [
            str(transactions["timestamp"].min()),
            str(transactions["timestamp"].max()),
        ],
        "n_customers": int(sim.customers),
        "n_transactions": int(total),
        "n_fraud_transactions": n_fraud,
        "n_fraud_episodes": int(len(episodes)),
        "fraud_rate": round(n_fraud / max(1, total), 6),
        "target_fraud_rate": sim.target_fraud_rate,
        "fraud_rate_vs_target_pct": round(
            100.0 * ((n_fraud / max(1, total)) / sim.target_fraud_rate - 1.0), 3
        ),
        "fraud_rate_within_5pct_of_target": bool(
            abs(n_fraud / max(1, total) / sim.target_fraud_rate - 1.0) <= 0.05
        ),
        "period_totals": {
            "episodes": {str(k): int(v) for k, v in period_episodes.items()},
            "transactions": {str(k): int(v) for k, v in period_tx.items()},
        },
        "scenarios": scenario_summary,
        "drift": {
            "start_fraction": sim.drift.start_fraction,
            "strength": sim.drift.strength,
            "start_date": str(drift_date) if has_drift else None,
        },
        "label_delay_days": list(sim.label_delay_days),
        "hard_negative": {
            "target_share": sim.hard_negative_target_share,
            "transaction_counts": hn_counts,
            "transaction_share_of_legit": hn_share,
        },
        "tx_per_archetype": tx_per_arch,
        "fraud_per_archetype": fraud_per_arch,
    }


def generate_profile(profile: str = "dev", seed: int | None = None) -> GeneratedResult:
    """Generate the full dataset for ``profile`` and return frames + report."""
    cfg = load_config(profile)
    sim = cfg.simulation
    run_seed = sim.seed if seed is None else int(seed)
    rng = np.random.default_rng(run_seed)

    logger.info("Generating entities for profile %s (seed=%d)", profile, run_seed)
    ent = create_entities(rng, sim)
    devreg: DeviceRegistry = make_device_registry(ent.devices, ent.customer_devices)
    # Static "known devices" table = devices a customer had at account opening,
    # frozen before any episode adds a device (hard-negative new_phone or fraud).
    # Freezing later would make new-device a near-perfect fraud signal (Part 03
    # gate caught exactly that shortcut).
    n_known_edges = len(devreg.customer_device_rows)

    logger.info("Generating normal transactions")
    gate = {"normal_rows_masked": 0, "episode_day_redraws": 0, "episode_customer_fallbacks": 0}
    normal_buf = generate_normal(rng, sim, ent, devreg, counters=gate)
    legit_count = len(normal_buf)

    logger.info("Generating hard-negative episodes")
    normal_times = normal_buf.tx_times()
    hard_buf = generate_hard_negatives(
        rng, sim, ent, devreg, legit_count, normal_times, counters=gate
    )

    logger.info("Generating fraud episodes")
    fraud_buf = generate_fraud(
        rng, sim, ent, devreg, legit_count + len(hard_buf), normal_times, counters=gate
    )

    df = _merge_buffers([normal_buf, hard_buf, fraud_buf])
    df["created_at"] = df["timestamp"]
    df = _ensure_strictly_increasing(df)
    df = _stamp_ids(df)

    transactions = df[TX_COLUMNS].copy()
    meta_cols = ["transaction_id", "timestamp"] + [
        c for c in LABEL_COLUMNS if c not in ("transaction_id", "label_available_at")
    ]
    meta = df[meta_cols].copy()
    label_rng = np.random.default_rng(run_seed + 1000)
    labels = build_labels(meta, label_rng, sim)[LABEL_COLUMNS]

    leaked = FORBIDDEN_COLUMNS.intersection(transactions.columns)
    if leaked:  # pragma: no cover - guard
        raise AssertionError(f"label columns leaked into transactions: {sorted(leaked)}")

    # Cold-start gate (Part 02b): no transaction may precede its account opening.
    open_map = dict(
        zip(ent.customers["customer_id"], pd.to_datetime(ent.customers["account_open_date"]))
    )
    before_open = transactions["timestamp"] < transactions["customer_id"].map(open_map)
    if before_open.any():  # pragma: no cover - guard
        raise AssertionError(
            f"{int(before_open.sum())} transactions precede account_open_date "
            f"(first: {transactions.loc[before_open, 'transaction_id'].iloc[0]})"
        )

    devices, _ = devreg.to_frames()
    customer_devices = pd.DataFrame(devreg.customer_device_rows[:n_known_edges])
    report = _build_report(cfg, transactions, labels, ent)
    report["cold_start_gate"] = dict(gate)
    report["config_hash"] = _config_hash(cfg)

    result = GeneratedResult(
        transactions=transactions,
        labels=labels,
        customers=ent.customers,
        accounts=ent.accounts,
        devices=devices,
        customer_devices=customer_devices,
        merchants=ent.merchants,
        locations=ent.locations,
        report=report,
    )
    return result


def _config_hash(cfg: Config) -> str:
    from fraudlens.config import config_hash

    return config_hash(cfg)


def write_outputs(result: GeneratedResult, profile: str) -> dict[str, str]:
    """Persist all frames as parquet plus simulation_report.json."""
    import json

    out = raw_data_dir(profile)
    paths = {
        "transactions": out / "transactions.parquet",
        "labels": out / "labels.parquet",
        "customers": out / "customers.parquet",
        "accounts": out / "accounts.parquet",
        "devices": out / "devices.parquet",
        "customer_devices": out / "customer_devices.parquet",
        "merchants": out / "merchants.parquet",
        "locations": out / "locations.parquet",
    }
    result.transactions.to_parquet(paths["transactions"], index=False)
    result.labels.to_parquet(paths["labels"], index=False)
    result.customers.to_parquet(paths["customers"], index=False)
    result.accounts.to_parquet(paths["accounts"], index=False)
    result.devices.to_parquet(paths["devices"], index=False)
    result.customer_devices.to_parquet(paths["customer_devices"], index=False)
    result.merchants.to_parquet(paths["merchants"], index=False)
    result.locations.to_parquet(paths["locations"], index=False)
    report_path = out / "simulation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(result.report, fh, indent=2)
    return {k: str(v) for k, v in paths.items()} | {"report": str(report_path)}


def main(profile: str = "dev", seed: int | None = None) -> GeneratedResult:
    result = generate_profile(profile, seed=seed)
    write_outputs(result, profile)
    logger.info(
        "Done: %d tx, %d fraud (%.4f), %d episodes",
        result.report["n_transactions"],
        result.report["n_fraud_transactions"],
        result.report["fraud_rate"],
        result.report["n_fraud_episodes"],
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic FraudLens data.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args.profile, seed=args.seed)