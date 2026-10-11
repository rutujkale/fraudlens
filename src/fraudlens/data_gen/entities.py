"""Persistent entities: locations, merchants, customers, accounts, devices.

Everything is produced from a seeded RNG; no network calls. Customer attributes
carry no fraud signal (archetype, age, income etc. describe behaviour only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from fraudlens.data_gen.archetypes import INCOME_BANDS, sample_archetype_params

CURRENCY = "INR"

# Approximate real lat/lon and region grouping for a spread of Indian cities.
# (name, state, region_group, lat, lon)
CITIES = [
    ("Mumbai", "Maharashtra", "West", 19.076, 72.8777),
    ("Delhi", "Delhi", "North", 28.7041, 77.1025),
    ("Bengaluru", "Karnataka", "South", 12.9716, 77.5946),
    ("Hyderabad", "Telangana", "South", 17.385, 78.4867),
    ("Chennai", "Tamil Nadu", "South", 13.0827, 80.2707),
    ("Kolkata", "West Bengal", "East", 22.5726, 88.3639),
    ("Pune", "Maharashtra", "West", 18.5204, 73.8567),
    ("Ahmedabad", "Gujarat", "West", 23.0225, 72.5714),
    ("Jaipur", "Rajasthan", "North", 26.9124, 75.7873),
    ("Lucknow", "Uttar Pradesh", "Central", 26.8467, 80.9462),
    ("Kanpur", "Uttar Pradesh", "Central", 26.4499, 80.3319),
    ("Nagpur", "Maharashtra", "Central", 21.1458, 79.0882),
    ("Indore", "Madhya Pradesh", "Central", 22.7196, 75.8577),
    ("Bhopal", "Madhya Pradesh", "Central", 23.2599, 77.4126),
    ("Patna", "Bihar", "East", 25.5941, 85.1376),
    ("Vadodara", "Gujarat", "West", 22.3072, 73.1812),
    ("Ghaziabad", "Uttar Pradesh", "North", 28.6692, 77.4538),
    ("Ludhiana", "Punjab", "North", 30.901, 75.8573),
    ("Agra", "Uttar Pradesh", "Central", 27.1767, 78.0081),
    ("Nashik", "Maharashtra", "West", 19.9975, 73.7898),
    ("Faridabad", "Haryana", "North", 28.4089, 77.3178),
    ("Meerut", "Uttar Pradesh", "North", 28.9845, 77.7064),
    ("Rajkot", "Gujarat", "West", 22.3039, 70.8022),
    ("Varanasi", "Uttar Pradesh", "Central", 25.3176, 82.9739),
    ("Srinagar", "Jammu and Kashmir", "North", 34.0837, 74.7973),
    ("Aurangabad", "Maharashtra", "West", 19.8762, 75.3433),
    ("Dhanbad", "Jharkhand", "East", 23.7957, 86.4304),
    ("Amritsar", "Punjab", "North", 31.634, 74.8723),
    ("Navi Mumbai", "Maharashtra", "West", 19.033, 73.0297),
    ("Prayagraj", "Uttar Pradesh", "Central", 25.4358, 81.8463),
    ("Ranchi", "Jharkhand", "East", 23.3441, 85.3096),
    ("Howrah", "West Bengal", "East", 22.5958, 88.2636),
    ("Coimbatore", "Tamil Nadu", "South", 11.0168, 76.9558),
    ("Jabalpur", "Madhya Pradesh", "Central", 23.1815, 79.9864),
    ("Gwalior", "Madhya Pradesh", "Central", 26.2183, 78.1828),
    ("Vijayawada", "Andhra Pradesh", "South", 16.5062, 80.648),
    ("Jodhpur", "Rajasthan", "North", 26.2389, 73.0243),
    ("Madurai", "Tamil Nadu", "South", 9.9252, 78.1198),
    ("Raipur", "Chhattisgarh", "Central", 21.2514, 81.6296),
    ("Kota", "Rajasthan", "North", 25.2138, 75.8648),
    ("Guwahati", "Assam", "North-East", 26.1445, 91.7362),
    ("Chandigarh", "Chandigarh", "North", 30.7333, 76.7794),
]

INCOME_BAND_WEIGHTS_BY_ARCHETYPE = {
    # low, mid, mid_high, high  -- weak, noisy correlation with spending habits
    "conservative": [0.30, 0.40, 0.20, 0.10],
    "regular": [0.20, 0.45, 0.25, 0.10],
    "high_spend": [0.05, 0.25, 0.40, 0.30],
    "frequent_digital": [0.25, 0.40, 0.25, 0.10],
    "business_like": [0.05, 0.20, 0.35, 0.40],
}


@dataclass
class DeviceRegistry:
    """Mutable device catalogue so episodes can add (new) devices over time."""

    customer_devices: dict[str, list[str]] = field(default_factory=dict)
    devices: list[dict] = field(default_factory=list)
    customer_device_rows: list[dict] = field(default_factory=list)
    fraud_device_pool: list[str] = field(default_factory=list)
    _counter: int = 0

    def known_devices(self, customer_id: str) -> list[str]:
        return list(self.customer_devices.get(customer_id, []))

    def add_device(
        self, customer_id: str, ts, shared: bool = False, device_type: str = "MOBILE"
    ) -> str:
        """Register a device as known to a customer from ``ts`` onward."""
        self._counter += 1
        device_id = f"D{self._counter:07d}"
        bucket = self.customer_devices.setdefault(customer_id, [])
        if device_id not in bucket:
            bucket.append(device_id)
        self.devices.append(
            {
                "device_id": device_id,
                "first_seen": ts,
                "last_seen": ts,
                "device_type": device_type,
                "is_shared": bool(shared),
            }
        )
        self.customer_device_rows.append({"customer_id": customer_id, "device_id": device_id})
        return device_id

    def reuse_fraud_device(self, customer_id: str, ts: str | None = None) -> str:
        """Return a device previously used in a fraud episode (reuse scenario)."""
        device_id = None
        if self.fraud_device_pool:
            device_id = self.fraud_device_pool[0]
            bucket = self.customer_devices.setdefault(customer_id, [])
            if device_id not in bucket:
                bucket.append(device_id)
            if ts is not None:
                for row in self.devices:
                    if row["device_id"] == device_id:
                        row["last_seen"] = ts
        if device_id is None:
            device_id = self.add_device(customer_id, ts)
        return device_id

    def to_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            pd.DataFrame(self.devices),
            pd.DataFrame(self.customer_device_rows),
        )


def make_device_registry(devices: pd.DataFrame, customer_devices: pd.DataFrame) -> DeviceRegistry:
    reg = DeviceRegistry()
    for _, row in devices.iterrows():
        reg.devices.append(row.to_dict())
    for _, row in customer_devices.iterrows():
        bucket = reg.customer_devices.setdefault(row["customer_id"], [])
        if row["device_id"] not in bucket:
            bucket.append(row["device_id"])
        reg.customer_device_rows.append(row.to_dict())
    reg._counter = len(reg.devices)
    return reg


def build_locations() -> pd.DataFrame:
    rows = [
        {
            "location_id": f"LOC_{i:04d}",
            "country": "India",
            "state": state,
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "region_group": region,
        }
        for i, (city, state, region, lat, lon) in enumerate(CITIES, start=1)
    ]
    return pd.DataFrame(rows)


def build_merchants(rng: np.random.Generator, n_customers: int) -> pd.DataFrame:
    from fraudlens.data_gen.archetypes import CATEGORIES

    n_merchants = int(np.clip(n_customers // 8, 400, 2000))
    category_weights = np.array(
        [9, 6, 7, 4, 3, 5, 3, 4, 3, 1, 1, 2], dtype=float
    )  # grocery/fuel/dining dominate, jewelry/transfers rare
    category_weights /= category_weights.sum()
    categories = rng.choice(CATEGORIES, size=n_merchants, p=category_weights)
    cities = rng.integers(0, len(CITIES), size=n_merchants)
    risk_groups = rng.choice(["low", "medium", "high"], size=n_merchants)
    return pd.DataFrame(
        {
            "merchant_id": [f"M{i:07d}" for i in range(1, n_merchants + 1)],
            "merchant_category": categories,
            "region": [CITIES[i][0] for i in cities],
            "merchant_risk_group": risk_groups,
        }
    )


def create_entities(rng: np.random.Generator, simulation) -> Entities:
    """Create all persistent entities except any devices added during episodes."""
    sim = simulation
    start = sim.start_date
    n = sim.customers
    mix = sim.customers_archetype_mix

    locations = build_locations()
    merchants = build_merchants(rng, n)
    location_rows = locations.to_dict("records")
    loc_index = {row["location_id"]: i for i, row in enumerate(location_rows)}
    region_of_loc = {row["location_id"]: row["region_group"] for row in location_rows}

    # Per-customer behavioural sampling
    archetypes = rng.choice(list(mix.keys()), size=n, p=list(mix.values()))
    params = [sample_archetype_params(a, rng) for a in archetypes]

    home_indices = rng.integers(0, len(location_rows), size=n)
    home_locations = [location_rows[i]["location_id"] for i in home_indices]
    home_regions = [region_of_loc[h] for h in home_locations]
    income_bands = [
        rng.choice(INCOME_BANDS, p=INCOME_BAND_WEIGHTS_BY_ARCHETYPE[a])
        for a in archetypes
    ]

    customer_ids = [f"C{i:07d}" for i in range(1, n + 1)]

    # ~10% of accounts open mid-simulation (cold-start customers)
    cold_start_frac = 0.10
    open_offset_days = rng.integers(1, max(2, sim.days - 15), size=n)
    is_cold = rng.random(n) < cold_start_frac
    open_dates = [
        start + timedelta(days=int(off) if cold else 0)
        for cold, off in zip(is_cold, open_offset_days)
    ]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": [p["age"] for p in params],
            "income_band": income_bands,
            "home_region": home_regions,
            "home_location_id": home_locations,
            "account_open_date": [d.isoformat() for d in open_dates],
            "archetype": archetypes,
            "baseline_daily_tx": [p["baseline_daily_tx"] for p in params],
            "baseline_amount": [p["baseline_amount"] for p in params],
            "amount_sigma": [p["amount_sigma"] for p in params],
            "preferred_categories": [p["preferred_categories"] for p in params],
            "active_hour_mean": [p["active_hour_mean"] for p in params],
            "active_hour_sd": [p["active_hour_sd"] for p in params],
            "travel_frequency": [p["travel_frequency"] for p in params],
        }
    )

    # Accounts: 1 per customer; ~20% of business_like get a second (current) account
    account_rows = []
    account_counter = 0
    customer_accounts: dict[str, list[str]] = {}
    open_date_by_cust = dict(zip(customer_ids, [d.isoformat() for d in open_dates]))
    for cust, arch in zip(customer_ids, archetypes):
        account_counter += 1
        acct = f"A{account_counter:07d}"
        account_type = "CURRENT" if arch == "business_like" else "SAVINGS"
        customer_accounts[cust] = [acct]
        account_rows.append(
            {
                "account_id": acct,
                "customer_id": cust,
                "account_type": account_type,
                "opening_date": open_date_by_cust[cust],
                "status": "ACTIVE",
                "base_currency": CURRENCY,
            }
        )
        if arch == "business_like" and rng.random() < 0.2:
            account_counter += 1
            acct2 = f"A{account_counter:07d}"
            customer_accounts[cust].append(acct2)
            account_rows.append(
                {
                    "account_id": acct2,
                    "customer_id": cust,
                    "account_type": "CURRENT",
                    "opening_date": open_date_by_cust[cust],
                    "status": "ACTIVE",
                    "base_currency": CURRENCY,
                }
            )
    accounts = pd.DataFrame(account_rows)
    first_account: dict[str, str] = {}
    for a in account_rows:
        first_account.setdefault(a["customer_id"], a["account_id"])

    # Devices: per-customer known devices; ~4% of customers share one family device
    devices_rows: list[dict] = []
    customer_devices_rows: list[dict] = []
    customer_device_ids: dict[str, list[str]] = {}
    device_counter = 0
    for cust, _arch, p in zip(customer_ids, archetypes, params):
        device_ids = []
        for _ in range(p["device_count"]):
            device_counter += 1
            dev_id = f"D{device_counter:07d}"
            device_ids.append(dev_id)
            devices_rows.append(
                {
                    "device_id": dev_id,
                    "first_seen": start,
                    "last_seen": None,
                    "device_type": "MOBILE",
                    "is_shared": False,
                }
            )
            customer_devices_rows.append({"customer_id": cust, "device_id": dev_id})
        customer_device_ids[cust] = device_ids

    family_share_count = int(sim.family_shared_device_share * n) // 2
    shared_customers = rng.choice(customer_ids, size=family_share_count * 2, replace=False)
    for pair_start in range(0, len(shared_customers), 2):
        a, b = shared_customers[pair_start], shared_customers[pair_start + 1]
        device_counter += 1
        shared_dev = f"D{device_counter:07d}"
        devices_rows.append(
            {
                "device_id": shared_dev,
                "first_seen": start,
                "last_seen": None,
                "device_type": "MOBILE",
                "is_shared": True,
            }
        )
        for cust in (a, b):
            customer_devices_rows.append({"customer_id": cust, "device_id": shared_dev})
            customer_device_ids.setdefault(cust, []).append(shared_dev)

    devices = pd.DataFrame(devices_rows)
    customer_devices = pd.DataFrame(customer_devices_rows)

    # Contacts (P2P recipients): random other customers, per archetype range
    contacts: dict[str, list[str]] = {}
    for cust, _arch, p in zip(customer_ids, archetypes, params):
        lo, hi = p["contacts_range"]
        lo = int(lo)
        hi = int(hi)
        count = int(rng.integers(lo, hi + 1))
        others = [c for c in customer_ids if c != cust]
        choices = rng.choice(len(others), size=min(count, len(others)), replace=False)
        contacts[cust] = [others[i] for i in choices]

    # category weights lookup for normal generation
    cat_weights: dict[str, dict[str, float]] = {}
    for cust, p in zip(customer_ids, params):
        cat_weights[cust] = dict(zip(p["preferred_categories"], p["category_weights"]))

    return Entities(
        customers=customers,
        accounts=accounts,
        devices=devices,
        customer_devices=customer_devices,
        merchants=merchants,
        locations=locations,
        first_account=first_account,
        contacts=contacts,
        home_location={c: h for c, h in zip(customer_ids, home_locations)},
        near_locations=build_near_locations(loc_index, region_of_loc),
        cities_by_region=group_locations_by_region(location_rows),
        cat_weights=cat_weights,
    )


def build_near_locations(
    loc_index: dict[str, int], region_of_loc: dict[str, str]
) -> dict[str, list[str]]:
    """Same-region cities except the city itself (used for near-home tx)."""
    out: dict[str, list[str]] = {}
    locs = list(loc_index.keys())
    for loc in locs:
        out[loc] = [o for o in locs if o != loc and region_of_loc[o] == region_of_loc[loc]]
    return out


def group_locations_by_region(location_rows: list[dict]) -> dict[str, list[str]]:
    regions: dict[str, list[str]] = {}
    for row in location_rows:
        regions.setdefault(row["region_group"], []).append(row["location_id"])
    return regions


@dataclass
class Entities:
    customers: pd.DataFrame
    accounts: pd.DataFrame
    devices: pd.DataFrame
    customer_devices: pd.DataFrame
    merchants: pd.DataFrame
    locations: pd.DataFrame
    first_account: dict[str, str]
    contacts: dict[str, list[str]]
    home_location: dict[str, str]
    near_locations: dict[str, list[str]]
    cities_by_region: dict[str, list[str]]
    cat_weights: dict[str, dict[str, float]]