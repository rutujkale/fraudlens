"""Customer archetype definitions and per-customer parameter sampling.

Archetypes describe *behaviour* (frequency, amounts, device/hours/category
preferences). They never encode fraud risk — a customer's archetype must not
be a proxy for the fraud target (enforced by the Part 03 separability gate).
"""

from __future__ import annotations

import numpy as np

CATEGORIES = [
    "grocery",
    "fuel",
    "dining",
    "electronics",
    "travel",
    "utilities",
    "healthcare",
    "fashion",
    "entertainment",
    "jewelry",
    "transfers",
    "wallet_topup",
]

# Purchase categories a customer can *prefer* (P2P transfers and bill-pay are
# generated via transaction_type, not via category preference).
PREFERRED_POOL = [c for c in CATEGORIES if c not in ("transfers", "utilities")]

TRANSACTION_TYPES = ["PURCHASE", "P2P_TRANSFER", "BILL_PAY", "CASH_WITHDRAWAL"]
CHANNELS = ["APP", "WEB", "POS", "ATM"]

INCOME_BANDS = ["low", "mid", "mid_high", "high"]

# Parameterisation per archetype. Every range is sampled per-customer with
# noise (never fixed constants).
ARCHETYPES = {
    "conservative": {
        "daily": (0.05, 0.15),
        "amount": (300, 1200),
        "amount_sigma": (0.35, 0.55),
        "devices": (1, 1),
        "hour_mean": (10, 12),
        "hour_sd": (1.0, 1.6),
        "n_categories": (2, 4),
        "travel_freq": (0.0, 0.02),
        "p2p_rate": (0.04, 0.10),
        "contacts": (2, 10),
        "age_mean": 45,
    },
    "regular": {
        "daily": (0.2, 0.5),
        "amount": (500, 2500),
        "amount_sigma": (0.40, 0.60),
        "devices": (1, 2),
        "hour_mean": (11, 15),
        "hour_sd": (1.5, 2.5),
        "n_categories": (3, 6),
        "travel_freq": (0.0, 0.04),
        "p2p_rate": (0.08, 0.15),
        "contacts": (3, 25),
        "age_mean": 36,
    },
    "high_spend": {
        "daily": (0.15, 0.4),
        "amount": (3000, 15000),
        "amount_sigma": (0.45, 0.70),
        "devices": (1, 2),
        "hour_mean": (10, 16),
        "hour_sd": (1.8, 2.8),
        "n_categories": (4, 8),
        "travel_freq": (0.01, 0.06),
        "p2p_rate": (0.10, 0.18),
        "contacts": (3, 25),
        "age_mean": 38,
    },
    "frequent_digital": {
        "daily": (0.6, 1.5),
        "amount": (200, 1500),
        "amount_sigma": (0.35, 0.55),
        "devices": (2, 3),
        "hour_mean": (9, 20),
        "hour_sd": (3.0, 4.0),
        "n_categories": (5, 9),
        "travel_freq": (0.0, 0.04),
        "p2p_rate": (0.10, 0.20),
        "contacts": (5, 30),
        "age_mean": 27,
    },
    "business_like": {
        "daily": (1.0, 2.5),
        "amount": (3000, 25000),
        "amount_sigma": (0.40, 0.65),
        "devices": (1, 3),
        "hour_mean": (9, 18),
        "hour_sd": (2.5, 3.5),
        "n_categories": (5, 10),
        "travel_freq": (0.02, 0.10),
        "p2p_rate": (0.18, 0.30),
        "contacts": (8, 40),
        "age_mean": 42,
    },
}


def sample_archetype_params(archetype: str, rng: np.random.Generator) -> dict:
    """Draw one customer's behavioural parameters from the archetype ranges."""
    a = ARCHETYPES[archetype]
    n_categories = int(rng.integers(*a["n_categories"], endpoint=True))
    preferred = rng.choice(PREFERRED_POOL, size=n_categories, replace=False).tolist()
    return {
        "baseline_daily_tx": float(rng.uniform(*a["daily"])),
        "baseline_amount": float(rng.uniform(*a["amount"])),
        "amount_sigma": float(rng.uniform(*a["amount_sigma"])),
        "device_count": int(rng.integers(*a["devices"], endpoint=True)),
        "active_hour_mean": float(rng.uniform(*a["hour_mean"])),
        "active_hour_sd": float(rng.uniform(*a["hour_sd"])),
        "preferred_categories": preferred,
        "category_weights": rng.dirichlet(np.full(n_categories, 3.0)).tolist(),
        "travel_frequency": float(rng.uniform(*a["travel_freq"])),
        "p2p_rate": float(rng.uniform(*a["p2p_rate"])),
        "contacts_range": a["contacts"],
        "age": _age_for(archetype, rng),
    }


def _age_for(archetype: str, rng: np.random.Generator) -> int:
    mean = ARCHETYPES[archetype]["age_mean"]
    return int(np.clip(rng.normal(mean, 6.0), 18, 85))