"""Label table construction.

`is_fraud`, `fraud_scenario`, `episode_id`, `stealth` and `hard_negative_type`
come from the generator's parallel meta frame. Fraud labels become available
after a random delay `U[label_delay_days]` days; legitimate (including hard
negative) labels are available immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

META_TO_LABEL = [
    "is_fraud",
    "fraud_scenario",
    "episode_id",
    "stealth",
    "hard_negative_type",
]


def build_labels(meta: pd.DataFrame, rng: np.random.Generator, simulation) -> pd.DataFrame:
    """Attach `label_available_at` and return the labels frame (with transaction_id)."""
    lo, hi = simulation.label_delay_days
    n = len(meta)
    delay_days = np.where(
        meta["is_fraud"].to_numpy(),
        rng.integers(int(lo), int(hi) + 1, size=n),
        0,
    )
    timestamps = pd.to_datetime(meta["timestamp"])
    labels = pd.DataFrame({"transaction_id": meta["transaction_id"].to_numpy()})
    for col in META_TO_LABEL:
        labels[col] = meta[col].to_numpy()
    labels["label_available_at"] = timestamps + pd.to_timedelta(delay_days, unit="D")
    return labels