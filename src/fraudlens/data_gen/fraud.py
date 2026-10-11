"""Fraud episode generation across five scenarios.

Every scenario is a *signal driven by the base behaviours*, scaled by the
episode's stealth s in [0,1] (drawn from the per-period Beta via drift.py):
at s=1 the episode keeps amounts, times, devices and locations close to the
victim's normal behaviour. No conditional rules hardcode amounts/hours;
structuring alone follows the regulatory cash-splitting motif at the
reporting threshold.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from fraudlens.data_gen.archetypes import CATEGORIES
from fraudlens.data_gen.entities import CURRENCY, DeviceRegistry, Entities
from fraudlens.data_gen.normal import RowBuffer

SCENARIOS = ["burst", "account_takeover", "amount_anomaly", "geo_anomaly", "structuring"]

_ROW_KEYS = [
    "customer_id", "account_id", "timestamp", "amount", "currency",
    "merchant_id", "merchant_category", "device_id", "location_id",
    "recipient_id", "transaction_type", "channel",
    "is_fraud", "fraud_scenario", "episode_id", "stealth", "hard_negative_type",
]


def _lognorm_mult(rng: np.random.Generator, logmean: float, sigma: float) -> float:
    return float(np.exp(rng.normal(logmean, sigma)))


def _base_amount(rng, base_amt: float, amt_sigma: float, mult: float) -> float:
    return max(1.0, round(float(base_amt * mult * np.exp(rng.normal(0, amt_sigma))), 2))


def _device_for(
    rng, devreg, pool, cid, ts, known, p_new, reuse_prob
) -> tuple[str | None, list[str]]:
    if pool and rng.random() < reuse_prob:
        return str(rng.choice(pool)), pool
    if known and rng.random() >= p_new:
        return str(rng.choice(known)), pool
    d = devreg.add_device(cid, ts)
    return d, pool + [d]


def _gaps(rng, n: int, span_min: float) -> list[timedelta]:
    raw = np.sort(rng.uniform(1, span_min, size=n)) if n > 1 else np.array([0.0])
    gaps = np.concatenate([[0], np.diff(np.round(raw).astype(np.int64))])
    return [timedelta(minutes=int(g)) for g in np.clip(gaps, 1, None)]


def _far_location(rng, far_locs, home_region, all_locations) -> str:
    pool = far_locs.get(home_region) or all_locations
    return str(rng.choice(pool))


def generate_fraud(
    rng: np.random.Generator, simulation, ent: Entities, devreg: DeviceRegistry,
    legit_count: int, normal_times: tuple[np.ndarray, np.ndarray],
) -> RowBuffer:
    """Generate fraud episodes to hit `target_fraud_rate` of all transactions."""
    from fraudlens.data_gen import drift

    buf = RowBuffer()
    if legit_count <= 0:
        return buf
    target = legit_count * simulation.target_fraud_rate / (1 - simulation.target_fraud_rate)

    customers = ent.customers.to_dict("records")
    n_cust = len(customers)
    weights = np.array([1.5 if c["archetype"] == "business_like" else 1.0 for c in customers])
    weights = weights / weights.sum()
    all_customers = [c["customer_id"] for c in customers]
    merchants_by_cat: dict[str, list[str]] = defaultdict(list)
    for row in ent.merchants.to_dict("records"):
        merchants_by_cat[row["merchant_category"]].append(row["merchant_id"])

    all_locations = ent.locations["location_id"].tolist()
    loc_region = dict(zip(ent.locations["location_id"], ent.locations["region_group"]))
    far_locs: dict[str, list[str]] = {}
    for reg in set(loc_region.values()):
        far_locs[reg] = [loc for loc in all_locations if loc_region[loc] != reg]

    normal_cust, normal_time = normal_times

    cur = 0
    episode_no = 0
    max_start_minutes = simulation.days * 1440 - 2880
    window_start = pd.Timestamp(simulation.start_date)

    while cur < 0.95 * target:
        bu = int(np.floor(1.05 * target - cur))
        if bu <= 0:
            break
        ci = int(rng.choice(n_cust, p=weights))
        victim = customers[ci]
        cid = victim["customer_id"]

        s_ts = window_start + timedelta(minutes=int(rng.integers(0, max(1, max_start_minutes))))
        sval = drift.position(simulation, s_ts)
        mix = drift.scenario_mix(simulation, sval)
        scenario = str(rng.choice(list(mix.keys()), p=list(mix.values())))
        stealth = drift.draw_stealth(rng, simulation, sval)

        ctx = {
            "cid": cid,
            "account": ent.first_account[cid],
            "home": ent.home_location[cid],
            "near": ent.near_locations[ent.home_location[cid]],
            "known": devreg.known_devices(cid),
            "base_amt": victim["baseline_amount"],
            "amt_sigma": victim["amount_sigma"],
            "home_region": loc_region[ent.home_location[cid]],
            "pool": [],
            "others": [c for o, c in enumerate(all_customers) if o != ci],
        }
        # Far location must be computed lazily (needs home_region).

        rows: list[dict] = []
        ep_id = f"EP_F_{episode_no + 1:05d}"
        ts_anchor = s_ts

        if scenario == "burst":
            n = int(min(rng.integers(4, 25, endpoint=True), bu))
            span = float(rng.uniform(5, 45))
            ts_seq = [ts_anchor + g for g in _gaps(rng, n, span)]
            nmix = int(rng.integers(1, 3, endpoint=True))
            cats = [str(c) for c in rng.choice(CATEGORIES, size=nmix, replace=False)]
            p_new = 0.1 + 0.9 * (1 - stealth)
            pool = []
            pos_cats = {"grocery", "fuel", "dining", "fashion", "healthcare"}
            for k in range(n):
                ts = ts_seq[k]
                category = str(rng.choice(cats))
                device, pool = _device_for(
                    rng, devreg, pool, cid, ts, ctx["known"], p_new,
                    simulation.fraud_device_reuse_prob,
                )
                channel = "POS" if (category in pos_cats and rng.random() < 0.6) else "APP"
                merchant = None
                mp = merchants_by_cat.get(category) or []
                if mp:
                    merchant = str(rng.choice(mp))
                recipient = str(rng.choice(ctx["others"])) if rng.random() < 0.25 else None
                txn_type = "P2P_TRANSFER" if recipient else "PURCHASE"
                mult = _lognorm_mult(rng, 1.0 + (1 - stealth) * rng.uniform(0, 3), 0.25)
                amount = _base_amount(rng, ctx["base_amt"], ctx["amt_sigma"], mult)
                location = ctx["home"]
                rows.append(
                    dict(zip(
                        _ROW_KEYS,
                        [cid, ctx["account"], ts, amount, CURRENCY, merchant,
                         ("transfers" if recipient else category), device, location,
                         recipient, txn_type, channel,
                         True, scenario, ep_id, stealth, None],
                    ))
                )

        elif scenario == "account_takeover":
            n = int(min(rng.integers(2, 10, endpoint=True), bu))
            span = float(rng.uniform(30, 360))
            ts_seq = [ts_anchor + g for g in _gaps(rng, n, span)]
            p_new = 0.1 + 0.9 * (1 - stealth)
            odd_hour = rng.random() < (1 - stealth) * 0.7
            pool = []
            for k in range(n):
                ts = ts_seq[k]
                device, pool = _device_for(
                    rng, devreg, pool, cid, ts, ctx["known"], p_new,
                    simulation.fraud_device_reuse_prob,
                )
                far = rng.random() < (1 - stealth) * 0.8
                location = (
                    _far_location(rng, far_locs, ctx["home_region"], all_locations)
                    if far else ctx["home"]
                )
                if odd_hour:
                    ts = ts.replace(hour=int(rng.integers(0, 6)), minute=int(rng.integers(0, 60)))
                currencylike = rng.random() < 0.7
                if currencylike:
                    category = str(rng.choice(["electronics", "jewelry", "transfers"]))
                else:
                    category = str(rng.choice(CATEGORIES[:6]))
                recipient = str(rng.choice(ctx["others"])) if (category == "transfers") else None
                txn_type = "P2P_TRANSFER" if recipient else "PURCHASE"
                merchant = None
                mp = merchants_by_cat.get(category) or []
                if mp:
                    merchant = str(rng.choice(mp))
                channel = "APP" if device else "WEB"
                mult = _lognorm_mult(rng, 1.0 + (1 - stealth) * 0.8, 0.4)
                amount = _base_amount(rng, ctx["base_amt"], ctx["amt_sigma"], mult)
                rows.append(
                    dict(zip(
                        _ROW_KEYS,
                        [cid, ctx["account"], ts, amount, CURRENCY, merchant,
                         ("transfers" if recipient else category), device, location,
                         recipient, txn_type, channel,
                         True, scenario, ep_id, stealth, None],
                    ))
                )

        elif scenario == "amount_anomaly":
            n = int(min(rng.integers(1, 3, endpoint=True), bu))
            ts = ts_anchor
            for _ in range(n):
                mult = _lognorm_mult(rng, 1.2 + 1.8 * (1 - stealth), 0.4)
                amount = _base_amount(rng, ctx["base_amt"], ctx["amt_sigma"], mult)
                category = str(rng.choice(CATEGORIES[:6]))
                merchant = None
                mp = merchants_by_cat.get(category) or []
                if mp:
                    merchant = str(rng.choice(mp))
                if rng.random() < 0.1 or not ctx["known"]:
                    device = devreg.add_device(cid, ts)
                else:
                    device = str(rng.choice(ctx["known"]))
                channel = "APP" if device else "WEB"
                rows.append(
                    dict(zip(
                        _ROW_KEYS,
                        [cid, ctx["account"], ts, amount, CURRENCY, merchant, category,
                         device, ctx["home"], None, "PURCHASE", channel,
                         True, scenario, ep_id, stealth, None],
                    ))
                )
                ts = ts + timedelta(minutes=1)

        elif scenario == "geo_anomaly":
            mine = normal_cust == cid
            if not np.any(mine):
                continue
            refs = normal_time[mine]
            ref = pd.Timestamp(refs[int(rng.integers(len(refs)))])
            ts = ref + timedelta(minutes=int(rng.uniform(5, 110)))
            n = int(min(rng.integers(1, 4, endpoint=True), bu))
            p_new = 0.1 + 0.9 * (1 - stealth)
            for _k in range(n):
                location = _far_location(rng, far_locs, ctx["home_region"], all_locations)
                if rng.random() < p_new:
                    device = devreg.add_device(cid, ts)
                else:
                    device = str(rng.choice(ctx["known"])) if ctx["known"] else None
                category = str(rng.choice(CATEGORIES[:6]))
                merchant = None
                mp = merchants_by_cat.get(category) or []
                if mp:
                    merchant = str(rng.choice(mp))
                mult = _lognorm_mult(rng, 0.2, 0.3)
                amount = _base_amount(rng, ctx["base_amt"], ctx["amt_sigma"], mult)
                rows.append(
                    dict(zip(
                        _ROW_KEYS,
                        [cid, ctx["account"], ts, amount, CURRENCY, merchant, category,
                         device, location, None, "PURCHASE",
                         ("APP" if device else "POS"),
                         True, scenario, ep_id, stealth, None],
                    ))
                )
                ts = ts + timedelta(minutes=int(rng.uniform(1, 50)))

        elif scenario == "structuring":
            if bu < 4:
                break
            n = int(min(rng.integers(4, 14, endpoint=True), bu))
            high = min(6.0, 0.97 * n)
            low = min(max(1.5, 0.55 * n + 0.05), high)
            target_ratio = float(rng.uniform(low, high))
            ratios = np.clip(target_ratio / n + rng.uniform(-0.03, 0.03, size=n), 0.55, 0.97)
            residual = target_ratio - ratios[:-1].sum()
            ratios[-1] = float(np.clip(residual, 0.5, 0.98))
            amounts = [round(simulation.reporting_threshold * float(r), 2) for r in ratios]
            ts_seq = [ts_anchor]
            gaps = rng.uniform(5, 360, size=n - 1).cumsum()
            if gaps[-1] > 2880:
                gaps = gaps * (2880 / gaps[-1])
            for g in gaps:
                ts_seq.append(ts_anchor + timedelta(minutes=int(round(g))))
            for k in range(n):
                is_p2p = rng.random() < 0.6
                txn_type = "P2P_TRANSFER" if is_p2p else "CASH_WITHDRAWAL"
                recipient = str(rng.choice(ctx["others"])) if is_p2p else None
                merchant = None
                category = "transfers" if is_p2p else None
                device = str(rng.choice(ctx["known"])) if (is_p2p and ctx["known"]) else None
                channel = "APP" if is_p2p else "ATM"
                rows.append(
                    dict(zip(
                        _ROW_KEYS,
                        [cid, ctx["account"], ts_seq[k], amounts[k], CURRENCY, merchant,
                         category, device, ctx["home"], recipient, txn_type, channel,
                         True, scenario, ep_id, stealth, None],
                    ))
                )

        else:  # pragma: no cover - scenarios are fixed by config
            raise ValueError(f"unknown scenario {scenario}")

        for r in rows:
            buf.add_row(r)
        cur += len(rows)
        episode_no += 1
    return buf