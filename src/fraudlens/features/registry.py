"""Feature registry: the authoritative list of every Part 05 feature (Part 05A).

One place lists each feature's name, group, dtype, human description, window and
the raw columns it needs. ``FEATURE_COLUMNS`` (order preserved) is what models,
the API and Streamlit consume. ``FORBIDDEN_COLUMNS`` is imported from the
simulator so the single definition of "target-derived" is shared, and
``assert_no_target_columns`` is the guard every feature frame passes through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraudlens.data_gen.generate import FORBIDDEN_COLUMNS as _FORBIDDEN
from fraudlens.paths import docs_dir

FEATURE_VERSION = "features_v1"

FORBIDDEN_COLUMNS: frozenset[str] = frozenset(_FORBIDDEN)

ID_COLUMNS = ("transaction_id", "feature_as_of_timestamp")


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for a single feature column."""

    name: str
    group: str
    dtype: str
    description: str
    window: str = "lifetime"
    requires_columns: tuple[str, ...] = ()


def _s(name, group, dtype, description, window="lifetime", requires=()):
    return FeatureSpec(name, group, dtype, description, window, tuple(requires))


SPECS: list[FeatureSpec] = [
    # --- transaction ---
    _s("amount", "transaction", "float", "Transaction amount (INR).", "row", ("amount",)),
    _s(
        "log_amount",
        "transaction",
        "float",
        "log1p(amount); tames the long right tail.",
        "row",
        ("amount",),
    ),
    _s("hour", "transaction", "int", "Hour of day (0-23) in IST.", "row", ("timestamp",)),
    _s(
        "day_of_week",
        "transaction",
        "int",
        "ISO day of week (0=Mon .. 6=Sun).",
        "row",
        ("timestamp",),
    ),
    _s("is_weekend", "transaction", "int8", "1 if Saturday/Sunday.", "row", ("timestamp",)),
    _s("is_night", "transaction", "int8", "1 if hour >= 22 or hour < 6.", "row", ("timestamp",)),
    _s(
        "transaction_type",
        "transaction",
        "category",
        "PURCHASE / P2P_TRANSFER / BILL_PAY / CASH_WITHDRAWAL.",
        "row",
        ("transaction_type",),
    ),
    _s("channel", "transaction", "category", "APP / WEB / POS / ATM.", "row", ("channel",)),
    _s(
        "merchant_category",
        "transaction",
        "category",
        "Merchant category (NULL for P2P).",
        "row",
        ("merchant_category",),
    ),
    # --- recency ---
    _s(
        "minutes_since_prev_tx",
        "recency",
        "float",
        "Minutes since this customer's previous transaction (NaN if none).",
        "lifetime",
        ("timestamp",),
    ),
    _s(
        "days_since_last_tx",
        "recency",
        "float",
        "Days since previous transaction (NaN if none).",
        "lifetime",
        ("timestamp",),
    ),
    _s(
        "n_prior_tx",
        "recency",
        "int",
        "Count of strictly-prior transactions (0 for the first).",
        "lifetime",
        ("timestamp",),
    ),
    _s(
        "history_days",
        "recency",
        "float",
        "Days since account opening (static, always defined).",
        "lifetime",
        ("timestamp",),
    ),
    _s(
        "is_cold_start",
        "recency",
        "int8",
        "1 if the customer has no strictly-prior transaction.",
        "lifetime",
        ("timestamp",),
    ),
    # --- frequency ---
    _s(
        "tx_count_10m",
        "frequency",
        "int",
        "Prior tx in the last 10 minutes.",
        "10m",
        ("timestamp",),
    ),
    _s("tx_count_1h", "frequency", "int", "Prior tx in the last hour.", "1h", ("timestamp",)),
    _s("tx_count_6h", "frequency", "int", "Prior tx in the last 6 hours.", "6h", ("timestamp",)),
    _s("tx_count_24h", "frequency", "int", "Prior tx in the last 24 hours.", "24h", ("timestamp",)),
    _s("tx_count_7d", "frequency", "int", "Prior tx in the last 7 days.", "7d", ("timestamp",)),
    _s(
        "active_days_7d",
        "frequency",
        "int",
        "Distinct calendar days with a prior tx in the last 7d.",
        "7d",
        ("timestamp",),
    ),
    _s(
        "active_days_30d",
        "frequency",
        "int",
        "Distinct calendar days with a prior tx in the last 30d.",
        "30d",
        ("timestamp",),
    ),
    _s(
        "velocity_ratio_1h_vs_24h",
        "frequency",
        "float",
        "tx_count_1h / tx_count_24h (NaN if none in 24h).",
        "24h",
        ("timestamp",),
    ),
    # --- amount sums ---
    _s(
        "amount_sum_1h",
        "amount_sums",
        "float",
        "Sum of prior amounts in the last hour.",
        "1h",
        ("amount",),
    ),
    _s(
        "amount_sum_24h",
        "amount_sums",
        "float",
        "Sum of prior amounts in the last 24 hours.",
        "24h",
        ("amount",),
    ),
    _s(
        "max_amount_24h",
        "amount_sums",
        "float",
        "Max prior amount in the last 24 hours (NaN if none).",
        "24h",
        ("amount",),
    ),
    _s(
        "count_near_threshold_24h",
        "amount_sums",
        "int",
        "Prior 24h tx with amount in the structuring band.",
        "24h",
        ("amount",),
    ),
    # --- amount deviation ---
    _s(
        "avg_amount_7d",
        "amount_deviation",
        "float",
        "Mean prior amount over 7d (NaN if none).",
        "7d",
        ("amount",),
    ),
    _s(
        "avg_amount_30d",
        "amount_deviation",
        "float",
        "Mean prior amount over 30d (NaN if none).",
        "30d",
        ("amount",),
    ),
    _s(
        "std_amount_30d",
        "amount_deviation",
        "float",
        "Std of prior amount over 30d (NaN if none).",
        "30d",
        ("amount",),
    ),
    _s(
        "amount_ratio_to_7d_mean",
        "amount_deviation",
        "float",
        "amount / avg_amount_7d (NaN if no mean).",
        "7d",
        ("amount",),
    ),
    _s(
        "amount_ratio_to_30d_mean",
        "amount_deviation",
        "float",
        "amount / avg_amount_30d (NaN if no mean).",
        "30d",
        ("amount",),
    ),
    _s(
        "amount_zscore_30d",
        "amount_deviation",
        "float",
        "Z-score of amount vs prior 30d (std floored).",
        "30d",
        ("amount",),
    ),
    _s(
        "amount_vs_lifetime_median",
        "amount_deviation",
        "float",
        "amount / median prior amount (NaN if none/0).",
        "lifetime",
        ("amount",),
    ),
    # --- merchant / category ---
    _s(
        "new_merchant_flag",
        "merchant_category",
        "float",
        "1 if merchant never seen for this customer (NaN if NULL).",
        "lifetime",
        ("merchant_id",),
    ),
    _s(
        "new_category_flag",
        "merchant_category",
        "float",
        "1 if category never seen for this customer (NaN if NULL).",
        "lifetime",
        ("merchant_category",),
    ),
    _s(
        "category_freq_30d",
        "merchant_category",
        "float",
        "Share of prior 30d tx in this category (NaN if none).",
        "30d",
        ("merchant_category",),
    ),
    _s(
        "merchant_freq_30d",
        "merchant_category",
        "float",
        "Share of prior 30d tx at this merchant (NaN if none).",
        "30d",
        ("merchant_id",),
    ),
    # --- device ---
    _s(
        "is_new_device",
        "device",
        "float",
        "1 if device never used by this customer before T (NaN if NULL).",
        "lifetime",
        ("device_id",),
    ),
    _s(
        "device_age_days",
        "device",
        "float",
        "Days since the device's first prior use, any customer (NaN if new/NULL).",
        "lifetime",
        ("device_id", "timestamp"),
    ),
    _s(
        "tx_from_device_prior",
        "device",
        "float",
        "Prior tx by this customer from this device (NaN if NULL).",
        "lifetime",
        ("device_id",),
    ),
    _s(
        "accounts_seen_on_device",
        "device",
        "float",
        "Distinct customers who used the device before T (NaN if NULL).",
        "lifetime",
        ("device_id",),
    ),
    _s(
        "device_count_30d",
        "device",
        "float",
        "Prior 30d tx by this customer from this device (NaN if NULL).",
        "30d",
        ("device_id",),
    ),
    # --- recipient ---
    _s(
        "unique_recipients_1h",
        "recipient",
        "int",
        "Distinct recipients in prior hour.",
        "1h",
        ("recipient_id",),
    ),
    _s(
        "unique_recipients_24h",
        "recipient",
        "int",
        "Distinct recipients in prior 24h.",
        "24h",
        ("recipient_id",),
    ),
    _s(
        "is_new_recipient",
        "recipient",
        "float",
        "1 if recipient never used by this customer before T (NaN if NULL).",
        "lifetime",
        ("recipient_id",),
    ),
    # --- geographic ---
    _s(
        "distance_from_home_km",
        "geographic",
        "float",
        "Haversine distance from the customer's home city (km).",
        "row",
        ("location_id",),
    ),
    _s(
        "distance_from_prev_km",
        "geographic",
        "float",
        "Distance from the previous transaction's city (NaN if none).",
        "lifetime",
        ("location_id",),
    ),
    _s(
        "travel_speed_kmh",
        "geographic",
        "float",
        "Implied speed since previous tx, capped (NaN if none/0 dt).",
        "lifetime",
        ("location_id",),
    ),
    _s(
        "new_region_flag",
        "geographic",
        "int",
        "1 if the transaction region was never seen before by this customer.",
        "lifetime",
        ("location_id",),
    ),
    _s(
        "region_freq_30d",
        "geographic",
        "float",
        "Share of prior 30d tx in this region (NaN if none).",
        "30d",
        ("location_id",),
    ),
    # --- temporal ---
    _s(
        "hour_deviation",
        "temporal",
        "float",
        "Share of prior tx within +/-2h of this hour (NaN if <5 prior).",
        "lifetime",
        ("timestamp",),
    ),
    # --- trend ---
    _s(
        "activity_change_7d_vs_prev_7d",
        "trend",
        "float",
        "tx_count_7d / count in the previous 7d window (NaN if prev=0).",
        "14d",
        ("timestamp",),
    ),
    _s(
        "amount_change_7d_vs_prev_7d",
        "trend",
        "float",
        "amount_sum_7d / sum in the previous 7d window (NaN if prev=0).",
        "14d",
        ("amount",),
    ),
]

FEATURE_COLUMNS: list[str] = [s.name for s in SPECS]
CATEGORICAL_COLUMNS: list[str] = ["transaction_type", "channel", "merchant_category"]
NUMERIC_COLUMNS: list[str] = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

_SPEC_BY_NAME = {s.name: s for s in SPECS}


def get_spec(name: str) -> FeatureSpec:
    """Return the registered spec for ``name`` (KeyError if unknown)."""
    return _SPEC_BY_NAME[name]


def assert_no_target_columns(df: pd.DataFrame) -> None:
    """Raise ``AssertionError`` if any target-derived column is present."""
    leaked = FORBIDDEN_COLUMNS.intersection(df.columns)
    if leaked:
        raise AssertionError(f"target-derived columns leaked into features: {sorted(leaked)}")


def data_dictionary_markdown() -> str:
    """Render the registry as a markdown table for ``docs/``."""
    lines = [
        f"# Feature data dictionary ({FEATURE_VERSION})",
        "",
        f"{len(SPECS)} features. Generated from `features/registry.py`; do not edit by hand.",
        "",
        "| feature | group | dtype | window | requires | description |",
        "|---|---|---|---|---|---|",
    ]
    for s in SPECS:
        req = ", ".join(s.requires_columns) if s.requires_columns else "-"
        lines.append(
            f"| `{s.name}` | {s.group} | {s.dtype} | {s.window} | {req} | {s.description} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_data_dictionary(path: str | Path | None = None) -> Path:
    """Write the generated data dictionary and return its path."""
    out = Path(path) if path is not None else docs_dir() / "data_dictionary_features.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(data_dictionary_markdown(), encoding="utf-8")
    return out
