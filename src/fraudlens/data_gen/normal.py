"""Legitimate (normal) transaction generation.

Per customer-day the count is Poisson from a noisy rate (day-of-week +
gamma day-effect + payday/festival multipliers). Per-transaction attributes
(amount, time, category, merchant, location, device, channel, recipient) are
sampled from that customer's own behavioural distribution.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

from fraudlens.data_gen.archetypes import CATEGORIES, TRANSACTION_TYPES
from fraudlens.data_gen.entities import CURRENCY, DeviceRegistry, Entities

# Field order as in spec table `transactions` (transaction_id stamped later).
TX_FIELDS = [
    "customer_id",
    "account_id",
    "timestamp",
    "amount",
    "currency",
    "merchant_id",
    "merchant_category",
    "device_id",
    "location_id",
    "recipient_id",
    "transaction_type",
    "channel",
]

# Field order as in spec table `labels` (transaction_id stamped later).
META_FIELDS = [
    "timestamp",
    "is_fraud",
    "fraud_scenario",
    "episode_id",
    "stealth",
    "hard_negative_type",
]

DOW_FACTOR = np.array([0.9, 1.0, 1.0, 1.0, 1.05, 1.25, 1.15])

# Amount multiplier by category (P2P / bill-pay / cash multiplier = 1.0).
CATEGORY_MULTIPLIER = {
    "grocery": 0.6,
    "fuel": 1.0,
    "dining": 0.8,
    "electronics": 1.5,
    "travel": 1.6,
    "utilities": 1.1,
    "healthcare": 1.2,
    "fashion": 1.2,
    "entertainment": 0.9,
    "jewelry": 2.5,
    "transfers": 1.0,
    "wallet_topup": 0.5,
}

# Approximate (very) festival calendar as (month, day) — see progress doc.
FESTIVALS = [
    (1, 1), (1, 26), (3, 4), (4, 8), (6, 20), (8, 15), (10, 2), (11, 8), (12, 25),
]


class RowBuffer:
    """Accumulates transaction + label columns and emits the two frames."""

    def __init__(self) -> None:
        self._tx: dict[str, list] = {k: [] for k in TX_FIELDS}
        self._meta: dict[str, list] = {k: [] for k in META_FIELDS}

    def add_row(self, row: dict) -> None:
        for k in TX_FIELDS:
            self._tx[k].append(row.get(k))
        for k in META_FIELDS:
            self._meta[k].append(row.get(k))

    def __len__(self) -> int:
        return len(self._tx["timestamp"])

    def tx_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._tx)

    def label_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._meta)

    def tx_times(self) -> tuple[np.ndarray, np.ndarray]:
        """Customer-ids and timestamps of buffered rows (for geo-anomaly lookups)."""
        return np.array(self._tx["customer_id"]), np.asarray(self._tx["timestamp"])

    def extend(self, other: RowBuffer) -> None:
        for k in TX_FIELDS:
            self._tx[k].extend(other._tx[k])
        for k in META_FIELDS:
            self._meta[k].extend(other._meta[k])


def _dow_factor(start: date, days: int) -> np.ndarray:
    return np.array([DOW_FACTOR[(start + timedelta(days=d)).weekday()] for d in range(days)])


def _payday_mask(start: date, days: int) -> np.ndarray:
    vals = []
    for d in range(days):
        dd = start + timedelta(days=d)
        vals.append(1.15 if dd.day == 1 else 1.0)
    return np.array(vals)


def _festival_dates(year: int) -> list[date]:
    try:
        return [date(year, m, d) for m, d in FESTIVALS]
    except ValueError:  # pragma: no cover - defensive for odd years
        return [date(year, m, d) for m, d in FESTIVALS if d <= 28]


def _festival_multiplier(start: date, days: int) -> np.ndarray:
    mult = np.ones(days)
    for fd in _festival_dates(start.year):
        offset = (fd - start).days
        if 0 <= offset < days:
            mult[offset] *= 1.30
    return mult


def generate_normal(
    rng: np.random.Generator, simulation, ent: Entities, devreg: DeviceRegistry
) -> RowBuffer:
    """Generate all normal (non-episode) legitimate transactions."""
    buf = RowBuffer()
    start = simulation.start_date
    ndays = simulation.days
    cust_df = ent.customers
    n = len(cust_df)

    merchants_by_cat: dict[str, list[str]] = defaultdict(list)
    for row in ent.merchants.to_dict("records"):
        merchants_by_cat[row["merchant_category"]].append(row["merchant_id"])

    all_customers = cust_df["customer_id"].tolist()
    all_locations = ent.locations["location_id"].tolist()

    rate = cust_df["baseline_daily_tx"].to_numpy(dtype=float)
    amt_mean = np.log(cust_df["baseline_amount"].to_numpy(dtype=float).clip(min=1))
    amt_sigma = cust_df["amount_sigma"].to_numpy(dtype=float)
    hour_mean = cust_df["active_hour_mean"].to_numpy(dtype=float)
    hour_sd = cust_df["active_hour_sd"].to_numpy(dtype=float)

    from fraudlens.data_gen.archetypes import ARCHETYPES

    cust_archetype = cust_df["archetype"].to_numpy()
    p2p = np.array([rng.uniform(*ARCHETYPES[a]["p2p_rate"]) for a in cust_archetype])

    dow = _dow_factor(start, ndays)
    payday = _payday_mask(start, ndays)
    fest = _festival_multiplier(start, ndays)

    # 1. per customer-day Poisson counts
    cust_idx: list[np.ndarray] = []
    day_idx: list[np.ndarray] = []
    for ci in range(n):
        day_effect = rng.gamma(3.0, 1.0 / 3.0, size=ndays)
        mult = payday * fest
        rr = rate[ci] * dow * day_effect * mult
        counts = rng.poisson(rr).astype(np.int64)
        hit = np.nonzero(counts)[0]
        hits = counts[hit]
        if hits.size:
            cust_idx.append(np.full(int(hits.sum()), ci))
            day_idx.append(np.repeat(hit, hits))
    if not cust_idx:
        return buf
    cust = np.concatenate(cust_idx).astype(np.int32)
    day = np.concatenate(day_idx).astype(np.int32)
    nt = len(cust)

    # 2. vectorized time & amount
    hour = (hour_mean[cust] + hour_sd[cust] * rng.standard_normal(nt)) % 24.0
    weekend = ((start.weekday() + day) % 7) >= 5
    hour = np.clip(hour + np.where(weekend, rng.uniform(-1, 1, nt), 0.0), 0, 23)
    minute = rng.integers(0, 60, size=nt)
    total_min = day * 1440 + np.round(hour * 60).astype(np.int64) + minute
    timestamps = pd.Timestamp(start) + pd.to_timedelta(total_min, unit="m")

    base_amt = np.exp(amt_mean[cust] + amt_sigma[cust] * rng.standard_normal(nt))
    outlier = rng.random(nt) < 0.02
    outlier_mult = np.where(outlier, rng.uniform(3.0, 8.0, nt), 1.0)

    # 3. transaction type (per-customer probabilities)
    pur = 1.0 - p2p - 0.10 - 0.05
    type_cum = np.column_stack([pur, pur + p2p, pur + p2p + 0.10, np.ones(n)])
    u_t = rng.random(nt)
    type_idx = (type_cum[cust] < u_t[:, None]).sum(axis=1)

    # 4. purchase category from per-customer Dirichlet weights (sparse cumsum)
    cum_w = np.zeros((n, len(CATEGORIES)))
    for ci, cust_id in enumerate(all_customers):
        acc = 0.0
        for cat, w in ent.cat_weights[cust_id].items():
            acc += w
            cum_w[ci, CATEGORIES.index(cat)] = acc
    u_c = rng.random(nt)
    cat_idx = (cum_w[cust] < u_c[:, None]).sum(axis=1).astype(np.int64)
    explore = rng.random(nt) < 0.08
    cat_idx[explore] = rng.integers(0, len(CATEGORIES), size=nt)[explore]

    categories = np.array(CATEGORIES)
    TX_TYPE = TRANSACTION_TYPES

    # 5. per-customer loop: channel/device/location/recipient/merchant
    device_by_cust: dict[str, list[str]] = defaultdict(list)
    for row in ent.customer_devices.to_dict("records"):
        device_by_cust[row["customer_id"]].append(row["device_id"])

    pos_starts = np.searchsorted(cust, np.arange(n), side="left")
    pos_ends = np.searchsorted(cust, np.arange(n), side="right")
    for ci in range(n):
        if pos_starts[ci] == pos_ends[ci]:
            continue
        cust_id = all_customers[ci]
        acct = ent.first_account[cust_id]
        known = devreg.known_devices(cust_id)
        home = ent.home_location[cust_id]
        near = ent.near_locations[home]
        contacts = ent.contacts[cust_id]
        last_merchant: dict[str, str] = {}
        for pos in range(pos_starts[ci], pos_ends[ci]):
            t = type_idx[pos]
            c = cat_idx[pos]
            txn_type = TX_TYPE[t]
            category = categories[c]

            if txn_type == "P2P_TRANSFER":
                category = "transfers"
            elif txn_type == "BILL_PAY":
                category = "utilities"
            elif txn_type == "CASH_WITHDRAWAL":
                category = None
            else:
                pass

            # channel & device
            if txn_type == "PURCHASE":
                pos_cats = {"grocery", "fuel", "dining", "fashion", "entertainment", "healthcare"}
                if category in pos_cats:
                    channel = "POS" if rng.random() < 0.65 else "APP"
                else:
                    channel = "APP" if rng.random() < 0.80 else "WEB"
            elif txn_type == "P2P_TRANSFER":
                channel = "APP"
            elif txn_type == "BILL_PAY":
                channel = "WEB" if rng.random() < 0.30 else "APP"
            else:
                channel = "ATM"
            device = str(rng.choice(known)) if (channel == "APP" and known) else None

            # location
            u_l = rng.random()
            if u_l < 0.88:
                location = home
            elif u_l < 0.97 and near:
                location = str(rng.choice(near))
            else:
                location = str(rng.choice(all_locations))

            # merchant (with per-customer loyalty) and recipient
            merchant = None
            recipient = None
            if txn_type in ("PURCHASE", "BILL_PAY") and category:
                repeat = rng.random() < 0.55 and category in last_merchant
                if repeat:
                    merchant = last_merchant[category]
                else:
                    pool = merchants_by_cat.get(category) or []
                    if pool:
                        merchant = str(rng.choice(pool))
                if merchant:
                    last_merchant[category] = merchant
            elif txn_type == "P2P_TRANSFER":
                if contacts and rng.random() < 0.95:
                    recipient = str(rng.choice(contacts))
                else:  # ~5% new contact
                    recipient = str(rng.choice(all_customers))

            mult = CATEGORY_MULTIPLIER.get(category or "transfers", 1.0)
            amount = round(float(base_amt[pos] * mult * outlier_mult[pos]), 2)

            buf.add_row(
                {
                    "customer_id": cust_id,
                    "account_id": acct,
                    "timestamp": timestamps[pos],
                    "amount": amount,
                    "currency": CURRENCY,
                    "merchant_id": merchant,
                    "merchant_category": category,
                    "device_id": device,
                    "location_id": location,
                    "recipient_id": recipient,
                    "transaction_type": txn_type,
                    "channel": channel,
                    "is_fraud": False,
                    "fraud_scenario": None,
                    "episode_id": None,
                    "stealth": None,
                    "hard_negative_type": None,
                }
            )
    return buf