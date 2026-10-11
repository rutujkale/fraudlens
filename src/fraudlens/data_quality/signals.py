"""Crude, leakage-safe-by-construction transaction signals (Part 03).

These exist **only** for the separability gate and the EDA overlap plot. They
are deliberately different from the real ``features_v1`` windows (Part 05):
each one uses only the transaction row itself plus **static** customer
attributes known at account opening / entity creation. No rolling window, no
count/velocity/aggregate over prior transactions.

All signals are oriented so that **higher = more suspicious**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = [
    "amount_ratio",
    "hour_dev",
    "dist_home_km",
    "new_device_static",
    "cat_not_preferred",
]

_EARTH_RADIUS_KM = 6371.0

_TX_COLS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "merchant_category",
    "device_id",
    "location_id",
]
_CUST_COLS = [
    "customer_id",
    "baseline_amount",
    "active_hour_mean",
    "home_location_id",
    "preferred_categories",
]


def _haversine(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    lat1, lon1, lat2, lon2 = (
        np.radians(v.to_numpy(dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return pd.Series(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def compute_crude_signals(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    locations: pd.DataFrame,
    customer_devices: pd.DataFrame,
) -> pd.DataFrame:
    """Return ``transaction_id`` plus the five crude signals (higher = suspicious)."""
    tx = transactions[_TX_COLS].copy()
    tx = tx.merge(customers[_CUST_COLS], on="customer_id", how="left")

    # 1. amount relative to the customer's static baseline
    tx["amount_ratio"] = tx["amount"] / tx["baseline_amount"].replace(0.0, np.nan)

    # 2. circular hour-of-day deviation from the customer's active-hour mean
    hour = tx["timestamp"].dt.hour + tx["timestamp"].dt.minute / 60.0
    raw = (hour - tx["active_hour_mean"]).abs() % 24.0
    tx["hour_dev"] = np.minimum(raw, 24.0 - raw)

    # 3. distance from home (row location vs static home location)
    loc = locations[["location_id", "latitude", "longitude"]]
    tx = tx.merge(
        loc.rename(columns={"latitude": "tx_lat", "longitude": "tx_lon"}),
        on="location_id",
        how="left",
    )
    tx = tx.merge(
        loc.rename(
            columns={
                "location_id": "home_location_id",
                "latitude": "home_lat",
                "longitude": "home_lon",
            }
        ),
        on="home_location_id",
        how="left",
    )
    tx["dist_home_km"] = _haversine(tx["tx_lat"], tx["tx_lon"], tx["home_lat"], tx["home_lon"])

    # 4. device not in the static known-device table. NULL device (WEB/POS/ATM)
    #    means "not a device event", i.e. NOT new.
    pairs = customer_devices[["customer_id", "device_id"]]
    known = set(map(tuple, pairs.itertuples(index=False, name=None)))
    has_device = tx["device_id"].notna()
    is_known = pd.Series(
        [(cid, did) in known for cid, did in zip(tx["customer_id"], tx["device_id"])],
        index=tx.index,
    )
    tx["new_device_static"] = (has_device & ~is_known).astype(float)

    # 5. merchant category outside the customer's preferred set (null -> not flagged)
    pref_by_id = {
        cid: set(p) if isinstance(p, (list, tuple, np.ndarray)) else set()
        for cid, p in zip(customers["customer_id"], customers["preferred_categories"])
    }
    tx["cat_not_preferred"] = [
        float(pd.notna(c) and c not in pref_by_id.get(cid, set()))
        for c, cid in zip(tx["merchant_category"], tx["customer_id"])
    ]

    out = tx[["transaction_id", *SIGNAL_COLUMNS]].copy()
    for col in ("amount_ratio", "hour_dev", "dist_home_km"):
        out[col] = out[col].fillna(out[col].median())
    return out
