"""EDA for a profile: figures + generated markdown (Part 03B).

All numbers are computed here and written to ``docs/eda_summary.md``; figures
go to ``reports/figures/eda_<profile>_*.png``. Nothing is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlens.data_quality.signals import (  # noqa: E402
    SIGNAL_COLUMNS,
    compute_crude_signals,
)
from fraudlens.data_quality.validation import load_raw_tables  # noqa: E402
from fraudlens.paths import REPO_ROOT, figures_dir, raw_data_dir  # noqa: E402
from fraudlens.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_RAW_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "transactions": {
        "transaction_id": "Primary key, assigned after sorting by (timestamp).",
        "customer_id": "FK -> customers; the transacting customer.",
        "account_id": "FK -> accounts; the account debited/credited.",
        "timestamp": "Naive IST event time.",
        "amount": "Positive amount in INR.",
        "currency": "ISO currency (INR).",
        "merchant_id": "FK -> merchants; null for P2P / cash withdrawal.",
        "merchant_category": "Category enum; null for cash withdrawal.",
        "device_id": "FK -> devices; only set for APP channel (null for WEB/POS/ATM).",
        "location_id": "FK -> locations; point of transaction.",
        "recipient_id": "FK -> customers; only for P2P transfers.",
        "transaction_type": "PURCHASE / P2P_TRANSFER / BILL_PAY / CASH_WITHDRAWAL.",
        "channel": "APP / WEB / POS / ATM.",
        "created_at": "Ingestion timestamp (equals event time).",
    },
    "labels": {
        "transaction_id": "FK -> transactions; one row per transaction.",
        "is_fraud": "Ground-truth fraud flag (never a feature).",
        "fraud_scenario": "Fraud scenario name, null when legitimate.",
        "episode_id": "Fraud / hard-negative episode id, null otherwise.",
        "stealth": "Fraud stealth in [0,1] (1 = looks almost normal), null when legit.",
        "hard_negative_type": "Hard-negative episode type, null otherwise.",
        "label_available_at": "When the label becomes known (>= timestamp).",
    },
    "customers": {
        "customer_id": "Primary key.",
        "age": "Age at simulation start.",
        "income_band": "low / mid / mid_high / high.",
        "home_region": "Region group of the home location.",
        "home_location_id": "FK -> locations; home city.",
        "account_open_date": "Account opening date (may be mid-window for cold start).",
        "archetype": "Behavioural archetype (no fraud signal).",
        "baseline_daily_tx": "Expected transactions per day.",
        "baseline_amount": "Expected transaction amount (INR).",
        "amount_sigma": "Log-normal amount spread.",
        "preferred_categories": "List of categories the customer usually uses.",
        "active_hour_mean": "Mean active hour of day (circular).",
        "active_hour_sd": "Active-hour spread.",
        "travel_frequency": "Relative propensity for out-of-home transactions.",
    },
    "accounts": {
        "account_id": "Primary key.",
        "customer_id": "FK -> customers.",
        "account_type": "SAVINGS / CURRENT.",
        "opening_date": "Account opening date.",
        "status": "ACTIVE.",
        "base_currency": "INR.",
    },
    "devices": {
        "device_id": "Primary key.",
        "first_seen": "First time the device appears.",
        "last_seen": "Last time the device appears.",
        "device_type": "MOBILE.",
        "is_shared": "Whether the device is shared across a family.",
    },
    "customer_devices": {
        "customer_id": "FK -> customers.",
        "device_id": "FK -> devices; devices known to the customer at account opening.",
    },
    "merchants": {
        "merchant_id": "Primary key.",
        "merchant_category": "Category enum.",
        "region": "Region group.",
        "merchant_risk_group": "low / medium / high (random; independent of fraud).",
    },
    "locations": {
        "location_id": "Primary key.",
        "country": "Country.",
        "state": "State.",
        "city": "City.",
        "latitude": "Latitude.",
        "longitude": "Longitude.",
        "region_group": "North / South / East / West / Central.",
    },
}


def _df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join([df.index.name or "index", *map(str, df.columns)]) + " |"
    sep = "|" + "|".join(["---"] * (len(df.columns) + 1)) + "|"
    rows = [
        "| " + " | ".join([str(idx), *(str(v) for v in row)]) + " |"
        for idx, row in zip(df.index, df.to_numpy())
    ]
    return "\n".join([header, sep, *rows])


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote %s", path)


def _load_report(profile: str) -> dict:
    path = raw_data_dir(profile) / "simulation_report.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _merged(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return transactions.merge(labels, on="transaction_id", how="left")


def _class_balance_fig(m: pd.DataFrame, profile: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    weekly = m.set_index("timestamp")["is_fraud"].resample("W").mean()
    axes[0].plot(weekly.index, weekly.values, marker="o")
    axes[0].set_title("Weekly fraud rate")
    axes[0].set_ylabel("fraud rate")
    axes[0].tick_params(axis="x", rotation=45)

    per_day = m.set_index("timestamp").resample("D").size()
    axes[1].plot(per_day.index, per_day.values)
    axes[1].set_title("Transactions per day")
    axes[1].set_ylabel("count")
    axes[1].tick_params(axis="x", rotation=45)

    arch = m["archetype"].value_counts()
    axes[2].bar(arch.index, arch.values)
    axes[2].set_title("Transactions per archetype")
    axes[2].tick_params(axis="x", rotation=30)
    _save(fig, out / f"eda_{profile}_class_balance.png")


def _amount_fig(m: pd.DataFrame, profile: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    legit = m.loc[~m["is_fraud"], "amount"]
    fraud = m.loc[m["is_fraud"], "amount"]
    bins = np.linspace(0, np.log10(m["amount"].max() + 1) + 0.1, 60)
    axes[0].hist(np.log10(legit + 1), bins=bins, alpha=0.6, label="legit", density=True)
    axes[0].hist(np.log10(fraud + 1), bins=bins, alpha=0.6, label="fraud", density=True)
    axes[0].set_title("Amount (log10, INR)")
    axes[0].set_xlabel("log10(amount + 1)")
    axes[0].legend()

    arch_order = m["archetype"].value_counts().index.tolist()
    data = [
        np.log10(m.loc[(m["archetype"] == a) & (~m["is_fraud"]), "amount"] + 1)
        for a in arch_order
    ]
    axes[1].boxplot(data, tick_labels=arch_order, showfliers=False)
    axes[1].set_title("Legit amount by archetype (log10)")
    axes[1].tick_params(axis="x", rotation=30)
    _save(fig, out / f"eda_{profile}_amounts.png")


def _hours_fig(m: pd.DataFrame, profile: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    hours = m["timestamp"].dt.hour
    for mask, label in (
        (~m["is_fraud"] & m["hard_negative_type"].isna(), "legit"),
        (m["is_fraud"], "fraud"),
        (m["hard_negative_type"].notna(), "hard negatives"),
    ):
        ax.hist(hours[mask], bins=range(25), alpha=0.5, density=True, label=label)
    ax.set_title("Hour-of-day distribution")
    ax.set_xlabel("hour")
    ax.legend()
    _save(fig, out / f"eda_{profile}_hours.png")


def _period_series(m: pd.DataFrame, drift_start) -> pd.Series:
    if drift_start is None:
        return pd.Series(["A"] * len(m), index=m.index)
    return pd.Series(
        np.where(m["timestamp"] >= pd.Timestamp(drift_start), "B", "A"), index=m.index
    )


def _drift_fig(m: pd.DataFrame, report: dict, profile: str, out: Path) -> None:
    drift_start = report.get("drift", {}).get("start_date")
    m = m.copy()
    m["period"] = _period_series(m, drift_start)
    fraud = m[m["is_fraud"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    if drift_start is not None:
        tab = pd.crosstab(fraud["fraud_scenario"], fraud["period"], normalize="columns")
        tab = tab.reindex(columns=[c for c in ("A", "B") if c in tab.columns])
        tab.plot(kind="bar", ax=axes[0])
        axes[0].set_title(f"Fraud scenario share by period (drift {drift_start})")
        axes[0].set_ylabel("share within period")
        axes[0].tick_params(axis="x", rotation=30)
    else:
        fraud["fraud_scenario"].value_counts().plot(kind="bar", ax=axes[0])
        axes[0].set_title("Fraud scenario share (no drift)")

    for period in ("A", "B"):
        vals = fraud.loc[fraud["period"] == period, "stealth"].dropna()
        if len(vals):
            axes[1].hist(vals, bins=20, alpha=0.5, density=True, label=f"period {period}")
    axes[1].set_title("Fraud stealth by period")
    axes[1].set_xlabel("stealth")
    axes[1].legend()
    _save(fig, out / f"eda_{profile}_drift.png")


def _overlap_fig(m: pd.DataFrame, signals: pd.DataFrame, profile: str, out: Path) -> None:
    sm = m[["transaction_id", "is_fraud", "hard_negative_type"]].merge(signals, on="transaction_id")
    groups = {
        "legit": sm["hard_negative_type"].isna() & ~sm["is_fraud"],
        "hard negatives": sm["hard_negative_type"].notna(),
        "fraud": sm["is_fraud"],
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    cols = ["amount_ratio", "hour_dev", "dist_home_km", "new_device_static"]
    logxs = [True, False, True, False]
    for ax, col, logx in zip(axes, cols, logxs):
        for label, mask in groups.items():
            vals = sm.loc[mask, col]
            if col == "new_device_static":
                ax.bar(label, vals.mean())
            else:
                data = np.log10(vals + 1) if logx else vals
                ax.hist(data, bins=30, alpha=0.5, density=True, label=label)
        ax.set_title(col + (" (log10)" if logx and col != "new_device_static" else ""))
        if col != "new_device_static":
            ax.legend(fontsize=8)
    _save(fig, out / f"eda_{profile}_overlap.png")


def _scenario_profile(m: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    sm = m[["transaction_id", "customer_id", "timestamp", "is_fraud", "fraud_scenario"]].merge(
        signals, on="transaction_id"
    )
    sm["hour_bucket"] = sm["timestamp"].dt.floor("h")
    velocity = (
        sm.groupby(["customer_id", "hour_bucket"])["transaction_id"]
        .transform("size")
    )
    sm["tx_in_1h"] = velocity
    fraud = sm[sm["is_fraud"]]
    prof = fraud.groupby("fraud_scenario").agg(
        transactions=("transaction_id", "size"),
        median_amount_ratio=("amount_ratio", "median"),
        share_new_device=("new_device_static", "mean"),
        median_dist_home_km=("dist_home_km", "median"),
        median_tx_in_1h=("tx_in_1h", "median"),
    )
    return prof.round(3)


def _write_data_dictionary(docs: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# Data Dictionary (raw tables)",
        "",
        "Generated by `scripts/eda.py` from the raw parquet schema. "
        "The feature dictionary is generated in Part 05.",
        "",
    ]
    for table in sorted(tables):
        lines.append(f"## `{table}`")
        lines.append("")
        lines.append("| column | dtype | description |")
        lines.append("|---|---|---|")
        desc = _RAW_DESCRIPTIONS.get(table, {})
        for col, dtype in tables[table].dtypes.items():
            lines.append(f"| `{col}` | {dtype} | {desc.get(col, '')} |")
        lines.append("")
    (docs / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", docs / "data_dictionary.md")


def _write_summary(
    docs: Path,
    profile: str,
    m: pd.DataFrame,
    signals: pd.DataFrame,
    report: dict,
    scenario_profile: pd.DataFrame,
) -> None:
    n = len(m)
    fraud = m[m["is_fraud"]]
    hn = m[m["hard_negative_type"].notna()]
    weekly = m.set_index("timestamp")["is_fraud"].resample("W").mean()

    lines = [
        f"# EDA Summary — profile `{profile}`",
        "",
        "Generated by `scripts/eda.py`. Every number below is computed from the raw tables.",
        "",
        "## Overview",
        "",
        f"- transactions: **{n:,}**",
        f"- customers: **{m['customer_id'].nunique():,}**",
        f"- fraud transactions: **{len(fraud):,}** (rate **{len(fraud) / n:.4%}**)",
        f"- hard-negative transactions: **{len(hn):,}**",
        f"- date range: `{m['timestamp'].min()}` → `{m['timestamp'].max()}`",
        "",
        "## Fraud rate over time (weekly)",
        "",
        "| week ending | fraud rate |",
        "|---|---|",
    ]
    for ts, val in weekly.items():
        lines.append(f"| {pd.Timestamp(ts).date()} | {val:.4%} |")
    lines += [
        "",
        "## Fraud scenario shares (overall and by period)",
        "",
        "| scenario | transactions | share |",
        "|---|---|---|",
    ]
    for scen, cnt in fraud["fraud_scenario"].value_counts().items():
        lines.append(f"| {scen} | {cnt} | {cnt / len(fraud):.2%} |")
    ptot = report.get("period_totals", {}).get("transactions", {})
    if ptot:
        lines.append("")
        lines.append(f"Fraud transactions by period: {ptot}")

    lines += [
        "",
        "## Per-scenario signal profile (crude signals)",
        "",
        _df_to_markdown(scenario_profile),
        "",
        "## Hard-negative vs fraud overlap (crude signals)",
        "",
    ]
    fraud_med = (
        fraud[["transaction_id"]].merge(signals, on="transaction_id")[SIGNAL_COLUMNS].median()
    )
    hn_sig = hn[["transaction_id"]].merge(signals, on="transaction_id")
    lines.append("| signal | hard-negative mean | fraud median |")
    lines.append("|---|---|---|")
    for col in SIGNAL_COLUMNS:
        lines.append(f"| {col} | {hn_sig[col].mean():.3f} | {fraud_med[col]:.3f} |")

    lines += [
        "",
        "## Figures",
        "",
        f"- `reports/figures/eda_{profile}_class_balance.png`",
        f"- `reports/figures/eda_{profile}_amounts.png`",
        f"- `reports/figures/eda_{profile}_hours.png`",
        f"- `reports/figures/eda_{profile}_drift.png`",
        f"- `reports/figures/eda_{profile}_overlap.png`",
        "",
        "## Honest notes",
        "",
        "- The `tx_in_1h` column in the scenario profile is a quick EDA approximation "
        "(customer-hour bucket count), not the rigorous Part 05 window feature.",
        "- Crude signals here are for EDA/gate use only and are never model inputs.",
        "",
    ]
    (docs / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", docs / "eda_summary.md")


def run_eda(profile: str = "dev") -> dict:
    """Generate figures + docs for ``profile`` and return a small summary dict."""
    tables = load_raw_tables(profile)
    report = _load_report(profile)
    m = _merged(tables["transactions"], tables["labels"])
    m["archetype"] = m["customer_id"].map(
        dict(zip(tables["customers"]["customer_id"], tables["customers"]["archetype"]))
    )
    signals = compute_crude_signals(
        tables["transactions"], tables["customers"], tables["locations"], tables["customer_devices"]
    )

    out = figures_dir()
    docs = REPO_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    _class_balance_fig(m, profile, out)
    _amount_fig(m, profile, out)
    _hours_fig(m, profile, out)
    _drift_fig(m, report, profile, out)
    _overlap_fig(m, signals, profile, out)

    scenario_profile = _scenario_profile(m, signals)
    _write_data_dictionary(docs, tables)
    _write_summary(docs, profile, m, signals, report, scenario_profile)

    summary = {
        "profile": profile,
        "n_transactions": int(len(m)),
        "n_fraud": int(m["is_fraud"].sum()),
        "fraud_rate": float(m["is_fraud"].mean()),
        "n_hard_negatives": int(m["hard_negative_type"].notna().sum()),
        "config_hash": report.get("config_hash"),
    }
    logger.info("EDA summary: %s", summary)
    return summary
