"""Hard-negative episodes: legitimate behaviour that mimics fraud signals.

Seven types from the spec (2C). Each overlaps with at least one fraud signal
(new device, location jump, velocity, odd hour, large amount) but is genuinely
legitimate and stays ``is_fraud=False`` with a ``hard_negative_type`` tag.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from fraudlens.data_gen.entities import CURRENCY, DeviceRegistry, Entities
from fraudlens.data_gen.normal import RowBuffer, _festival_dates

HARD_NEGATIVE_TYPES = [
    "new_phone",
    "travel_trip",
    "holiday_splurge",
    "medical_bill",
    "emergency_odd_hour",
    "payday_burst",
    "power_user_burst",
]

_POWER_ARCHETYPES = {"frequent_digital", "business_like"}

_ROW_KEYS = [
    "customer_id", "account_id", "timestamp", "amount", "currency",
    "merchant_id", "merchant_category", "device_id", "location_id",
    "recipient_id", "transaction_type", "channel",
    "is_fraud", "fraud_scenario", "episode_id", "stealth", "hard_negative_type",
]


def _amount(rng, base: float, sigma: float, mult: float) -> float:
    return max(1.0, round(float(base * mult * np.exp(rng.normal(0, sigma))), 2))


def _txn(gaps_min: list[float], ts0: pd.Timestamp) -> list[pd.Timestamp]:
    out = [ts0]
    for g in gaps_min:
        out.append(out[-1] + timedelta(minutes=int(g)))
    return out


def _pick_merchant(rng, merchants_by_cat, category) -> str | None:
    pool = merchants_by_cat.get(category) or []
    return str(rng.choice(pool)) if pool else None


def _emit(buf, row_keys, values, **kw) -> None:
    buf.add_row(dict(zip(row_keys, values)))


def generate_hard_negatives(
    rng: np.random.Generator,
    simulation,
    ent: Entities,
    devreg: DeviceRegistry,
    legit_count: int,
    normal_times: tuple[np.ndarray, np.ndarray],
) -> RowBuffer:
    buf = RowBuffer()
    if legit_count <= 0:
        return buf
    target = int(round(simulation.hard_negative_target_share * legit_count))
    start = pd.Timestamp(simulation.start_date)
    days = simulation.days
    end_min = days * 1440

    customers = ent.customers.to_dict("records")
    merchants_by_cat: dict[str, list[str]] = defaultdict(list)
    for row in ent.merchants.to_dict("records"):
        merchants_by_cat[row["merchant_category"]].append(row["merchant_id"])
    power_customers = [c for c in customers if c["archetype"] in _POWER_ARCHETYPES]
    paydays = [d for d in range(days) if (simulation.start_date + timedelta(days=d)).day == 1]
    festivals = [fd for fd in _festival_dates(simulation.start_date.year)
                 if 0 <= (fd - simulation.start_date).days < days]
    normal_cust, normal_time = normal_times
    loc_region = dict(zip(ent.locations["location_id"], ent.locations["region_group"]))
    all_locs = ent.locations["location_id"].tolist()
    far_locs: dict[str, list[str]] = {}
    for reg in set(loc_region.values()):
        far_locs[reg] = [loc for loc in all_locs if loc_region[loc] != reg]

    total = 0
    episode_no = 0
    while total < 0.95 * target:
        ttype = str(rng.choice(HARD_NEGATIVE_TYPES))
        pool = power_customers if ttype == "power_user_burst" and power_customers else customers
        victim = pool[int(rng.integers(len(pool)))]
        cid = victim["customer_id"]
        ep_id = f"EP_H_{episode_no + 1:05d}"
        acct = ent.first_account[cid]
        home = ent.home_location[cid]
        near = ent.near_locations[home]
        known = devreg.known_devices(cid)
        base, sigma = victim["baseline_amount"], victim["amount_sigma"]

        def row(ts, amount, category=None, merchant=None, device=None,
                location=None, recipient=None, txn_type="PURCHASE", channel="APP",
                cid=cid, acct=acct, home=home, ep_id=ep_id, ttype=ttype):
            _emit(buf, _ROW_KEYS, [
                cid, acct, ts, amount, CURRENCY, merchant, category, device,
                location if location is not None else home, recipient, txn_type,
                channel, False, None, ep_id, None, ttype,
            ])

        n = 0
        if ttype == "new_phone":
            device = devreg.add_device(cid, start)  # becomes a known device going forward
            n = int(rng.integers(6, 16, endpoint=True))
            span_days = float(rng.uniform(14, 21))
            span_min = min(span_days * 1440, end_min - 60)
            ts0 = start + timedelta(minutes=int(rng.integers(0, max(1, end_min - span_min))))
            offsets = np.sort(rng.uniform(30, span_min, size=n))
            seq = _txn(np.diff(np.round(offsets)).tolist(), ts0)
            cats = list(victim["preferred_categories"])
            for ts in seq:
                cat = str(rng.choice(cats))
                row(ts, _amount(rng, base, sigma, float(rng.uniform(0.8, 1.6))),
                    category=cat, merchant=_pick_merchant(rng, merchants_by_cat, cat),
                    device=device, channel="APP")

        elif ttype == "travel_trip":
            mine = normal_cust == cid
            if not np.any(mine):
                continue
            ref = pd.Timestamp(normal_time[mine][int(rng.integers(int(mine.sum())))])
            dest = str(rng.choice(far_locs.get(loc_region[home]) or all_locs))
            n = int(rng.integers(3, 12, endpoint=True))
            trip_days = float(rng.uniform(2, 10))
            trip_min = trip_days * 1440
            ts0 = ref + timedelta(minutes=int(rng.uniform(10, 180)))
            if ts0 >= start + timedelta(minutes=end_min):
                continue
            remaining = (start + timedelta(minutes=end_min) - ts0).total_seconds() / 60
            trip_min = min(trip_min, remaining)
            offsets = np.sort(rng.uniform(30, max(60, trip_min), size=n))
            seq = _txn(np.diff(np.round(offsets)).tolist(), ts0)
            cats = ["travel", "dining", "entertainment", "fuel"]
            device = str(rng.choice(known)) if known else None
            for ts in seq:
                cat = str(rng.choice(cats))
                row(ts, _amount(rng, base, sigma, float(rng.uniform(1.1, 1.8))),
                    category=cat, merchant=_pick_merchant(rng, merchants_by_cat, cat),
                    device=device, location=dest)

        elif ttype == "holiday_splurge":
            if not festivals:
                continue
            fday = festivals[int(rng.integers(len(festivals)))]
            ts0 = pd.Timestamp(fday) + timedelta(hours=int(rng.uniform(10, 21)),
                                                 minutes=int(rng.integers(0, 60)))
            n = int(rng.integers(2, 6, endpoint=True))
            seq = _txn([rng.uniform(5, 120) for _ in range(n - 1)], ts0)
            cats = ["fashion", "electronics", "jewelry", "travel"]
            device = str(rng.choice(known)) if known else None
            for ts in seq:
                cat = str(rng.choice(cats))
                row(ts, _amount(rng, base, sigma, float(rng.uniform(2.0, 5.0))),
                    category=cat, merchant=_pick_merchant(rng, merchants_by_cat, cat),
                    device=device)

        elif ttype == "medical_bill":
            n = 1
            ts0 = start + timedelta(days=int(rng.integers(0, days)),
                                    hours=int(rng.uniform(9, 20)))
            row(ts0, _amount(rng, base, sigma, float(rng.uniform(5.0, 30.0))),
                category="healthcare",
                merchant=_pick_merchant(rng, merchants_by_cat, "healthcare"),
                device=str(rng.choice(known)) if known else None, txn_type="BILL_PAY",
                channel="WEB")

        elif ttype == "emergency_odd_hour":
            n = int(rng.integers(1, 3, endpoint=True))
            ts0 = start + timedelta(days=int(rng.integers(0, days)),
                                    hours=int(rng.integers(0, 4)),
                                    minutes=int(rng.integers(0, 60)))
            seq = _txn([rng.uniform(1, 40) for _ in range(n - 1)], ts0)
            device = str(rng.choice(known)) if known else None
            for ts in seq:
                cat = str(rng.choice(victim["preferred_categories"]))
                loc = str(rng.choice(near)) if near and rng.random() < 0.3 else home
                row(ts, _amount(rng, base, sigma, float(rng.uniform(2.0, 4.0))),
                    category=cat, merchant=_pick_merchant(rng, merchants_by_cat, cat),
                    device=device, location=loc)

        elif ttype == "payday_burst":
            day = int(rng.choice(paydays)) if paydays else int(rng.integers(0, days))
            ts0 = start + timedelta(days=day, hours=int(rng.uniform(9, 20)),
                                    minutes=int(rng.integers(0, 60)))
            n = int(rng.integers(3, 8, endpoint=True))
            seq = _txn([rng.uniform(1, 12) for _ in range(n - 1)], ts0)
            device = str(rng.choice(known)) if known else None
            for ts in seq:
                is_p2p = rng.random() < 0.5
                recipient = None
                if is_p2p and ent.contacts[cid]:
                    recipient = str(rng.choice(ent.contacts[cid]))
                row(ts, _amount(rng, base, sigma, float(rng.uniform(0.8, 1.6))),
                    category="transfers" if is_p2p else "utilities",
                    merchant=None if is_p2p else _pick_merchant(rng, merchants_by_cat, "utilities"),
                    device=device, recipient=recipient,
                    txn_type="P2P_TRANSFER" if is_p2p else "BILL_PAY",
                    channel="APP" if is_p2p else "WEB")

        else:  # power_user_burst
            n = int(rng.integers(8, 20, endpoint=True))
            ts0 = start + timedelta(days=int(rng.integers(0, days)),
                                    hours=int(rng.uniform(8, 21)),
                                    minutes=int(rng.integers(0, 60)))
            seq = _txn([rng.uniform(1, 8) for _ in range(n - 1)], ts0)
            device = str(rng.choice(known)) if known else None
            for ts in seq:
                cat = str(rng.choice(victim["preferred_categories"]))
                loc = str(rng.choice(near)) if near and rng.random() < 0.4 else home
                row(ts, _amount(rng, base, sigma, float(rng.uniform(0.9, 1.5))),
                    category=cat, merchant=_pick_merchant(rng, merchants_by_cat, cat),
                    device=device, location=loc)

        total += n
        episode_no += 1
    return buf