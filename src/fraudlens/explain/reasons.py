"""Human-readable reasons from SHAP contributions (Part 10B).

A template map turns a **raw feature value** into a sentence. Templates read only
feature columns (never labels), so an explanation can be produced at scoring time.
Unmapped features and missing values fall back to ``feature = value``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraudlens.explain.shap_explainer import Contribution
from fraudlens.features.registry import CATEGORICAL_COLUMNS

_MISSING = "no prior history"


def _is_na(value: float) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _plural(value: float, singular: str, *, fmt: str = ",.0f") -> str:
    """``"1 minute"`` / ``"5 minutes"`` (singular only at exactly one)."""
    return f"{value:{fmt}} {singular}{'' if abs(value - 1) < 1e-9 else 's'}"


_TEMPLATES: dict[str, Callable[[float], str]] = {
    # amount deviation
    "amount_ratio_to_30d_mean": lambda v: f"Amount is {v:.1f}x this customer's 30-day average",
    "amount_ratio_to_7d_mean": lambda v: f"Amount is {v:.1f}x this customer's 7-day average",
    "amount_vs_lifetime_median": lambda v: f"Amount is {v:.1f}x the customer's lifetime median",
    "amount_zscore_30d": lambda v: f"Amount is {v:.1f} SD away from the customer's 30-day norm",
    "avg_amount_30d": lambda v: f"The customer's 30-day average amount is about ₹ {v:,.0f}",
    # amount
    "amount": lambda v: f"Transaction amount of ₹ {v:,.0f}",
    "log_amount": lambda v: f"Transaction amount of ₹ {np.expm1(v):,.0f}",
    "amount_sum_1h": lambda v: f"₹ {v:,.0f} transacted in the previous hour",
    "amount_sum_24h": lambda v: f"₹ {v:,.0f} transacted in the previous 24 hours",
    "max_amount_24h": lambda v: f"Largest prior amount today is ₹ {v:,.0f}",
    "count_near_threshold_24h": lambda v: (
        f"{_plural(v, 'prior transaction')} just below the reporting threshold in 24h"
    ),
    # recency / frequency
    "minutes_since_prev_tx": lambda v: (
        f"Only {_plural(v, 'minute')} since the previous transaction"
    ),
    "days_since_last_tx": lambda v: f"{v:.1f} days since the previous transaction",
    "n_prior_tx": lambda v: f"{_plural(v, 'prior transaction')} on the account",
    "history_days": lambda v: f"Account opened {_plural(v, 'day', fmt='.1f')} ago",
    "is_cold_start": lambda v: (
        "No prior transaction on this account (cold start)" if v >= 1 else "Established account"
    ),
    "tx_count_10m": lambda v: f"{_plural(v, 'transaction')} in the previous 10 minutes",
    "tx_count_1h": lambda v: f"{_plural(v, 'transaction')} in the previous hour",
    "tx_count_6h": lambda v: f"{_plural(v, 'transaction')} in the previous 6 hours",
    "tx_count_24h": lambda v: f"{_plural(v, 'transaction')} in the previous 24 hours",
    "tx_count_7d": lambda v: f"{_plural(v, 'transaction')} in the previous 7 days",
    "velocity_ratio_1h_vs_24h": lambda v: f"1-hour activity is {v:.1f}x the 24-hour rate",
    # recipients
    "unique_recipients_1h": lambda v: f"{_plural(v, 'different recipient')} in the previous hour",
    "unique_recipients_24h": lambda v: (
        f"{_plural(v, 'different recipient')} in the previous 24 hours"
    ),
    "is_new_recipient": lambda v: (
        "First time this customer pays this recipient" if v >= 1 else "Known recipient"
    ),
    # merchant / category
    "new_merchant_flag": lambda v: (
        "First time at this merchant" if v >= 1 else "Merchant seen before for this customer"
    ),
    "new_category_flag": lambda v: (
        "First purchase in this merchant category" if v >= 1 else "Familiar merchant category"
    ),
    "category_freq_30d": lambda v: f"{v * 100:.0f}% of prior 30-day spend is in this category",
    "merchant_freq_30d": lambda v: f"{v * 100:.0f}% of prior 30-day spend is at this merchant",
    # device
    "is_new_device": lambda v: (
        "First time this customer uses this device" if v >= 1 else "Device known for this customer"
    ),
    "device_age_days": lambda v: f"Device first seen {_plural(v, 'day')} ago",
    "tx_from_device_prior": lambda v: f"{_plural(v, 'prior transaction')} from this device",
    "accounts_seen_on_device": lambda v: (
        f"This device has been used by {_plural(v, 'account', fmt='.0f')}"
    ),
    "device_count_30d": lambda v: (
        f"{_plural(v, 'transaction')} from this device in the prior 30 days"
    ),
    # geographic
    "distance_from_home_km": lambda v: f"{v:,.0f} km from the customer's home city",
    "distance_from_prev_km": lambda v: f"{v:,.0f} km from the previous transaction's city",
    "travel_speed_kmh": lambda v: (
        f"Implied travel speed of {v:,.0f} km/h since the previous transaction"
    ),
    "new_region_flag": lambda v: (
        "First transaction in this region" if v >= 1 else "Region seen before for this customer"
    ),
    "region_freq_30d": lambda v: f"{v * 100:.0f}% of prior 30-day spend is in this region",
    # temporal
    "hour": lambda v: f"Transaction at {int(v):02d}:00 IST",
    "is_night": lambda v: (
        "Transaction during the night window (22:00-06:00 IST)" if v >= 1 else "Daytime transaction"
    ),
    "hour_deviation": lambda v: (
        f"Only {v * 100:.0f}% of the customer's prior transactions fall within +/-2h of this hour"
    ),
    # trend
    "activity_change_7d_vs_prev_7d": lambda v: (
        f"7-day activity is {v:.1f}x the previous 7 days"
    ),
    "amount_change_7d_vs_prev_7d": lambda v: f"7-day spend is {v:.1f}x the previous 7 days",
    # raw calendar / baseline fields without a richer story
    "day_of_week": lambda v: f"Transaction on weekday index {int(v)} (Mon=0)",
    "is_weekend": lambda v: "Weekend transaction" if v >= 1 else "Weekday transaction",
    "active_days_7d": lambda v: f"{_plural(v, 'active day')} in the previous 7",
    "active_days_30d": lambda v: f"{_plural(v, 'active day')} in the previous 30",
    "avg_amount_7d": lambda v: f"The customer's 7-day average amount is about ₹ {v:,.0f}",
    "std_amount_30d": lambda v: f"The customer's 30-day amount volatility is ₹ {v:,.0f}",
}


@dataclass
class Reasons:
    """Top raising and lowering explanations for one transaction."""

    raises_risk: list[str] = field(default_factory=list)
    lowers_risk: list[str] = field(default_factory=list)

    def mitigations_line(self) -> str:
        return "Mitigating: " + "; ".join(self.lowers_risk) if self.lowers_risk else ""


def format_reason(feature: str, value: float) -> str:
    """One sentence for ``feature`` at ``value`` (safe fallback if unmapped/missing)."""
    template = _TEMPLATES.get(feature)
    if template is None:
        if feature in CATEGORICAL_COLUMNS and not _is_na(value):
            return f"{feature.replace('_', ' ')} (encoded category code) = {value:.0f}"
        return fallback_reason(feature, value)
    if _is_na(value):
        return f"{feature.replace('_', ' ')}: {_MISSING}"
    try:
        return template(float(value))
    except (ValueError, OverflowError):
        return fallback_reason(feature, value)


def fallback_reason(feature: str, value: float) -> str:
    """`feature = value` fallback (``not available`` when missing)."""
    if _is_na(value):
        return f"{feature.replace('_', ' ')} = not available"
    return f"{feature.replace('_', ' ')} = {value:g}"


def reasons_for(
    row: pd.Series, contributions: list[Contribution], k: int = 5
) -> Reasons:
    """Top ``k`` raising and lowering reasons for one encoded feature row.

    ``contributions`` comes from :meth:`ShapExplainer.explain_local` (already sorted
    by ``|SHAP|``); the row must be the same encoded features in the same order.
    """
    reasons = Reasons()
    seen: set[str] = set()
    for contribution in contributions:
        feature = contribution.feature
        value = row[feature] if feature in row else contribution.value
        sentence = format_reason(feature, float(value))
        if sentence in seen:
            continue
        if contribution.shap_value > 0 and len(reasons.raises_risk) < k:
            reasons.raises_risk.append(sentence)
        elif contribution.shap_value < 0 and len(reasons.lowers_risk) < k:
            reasons.lowers_risk.append(sentence)
        else:
            continue
        seen.add(sentence)
        if len(reasons.raises_risk) >= k and len(reasons.lowers_risk) >= k:
            break
    return reasons
