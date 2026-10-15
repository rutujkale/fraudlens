"""Deterministic case selection for the local explanation gallery (Part 10D).

Cases are chosen by **pre-registered rules**, not by looks; every rule records the
size of the pool it chose from so a reader can judge how cherry-picked the example
is. Ties break on ``transaction_id`` ascending, so selection needs no seed and is
reproducible from ``results/shap_gallery_<profile>.csv`` alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudlens.evaluation.cost_model import APPROVE

CASE_ORDER = (
    "clear_tp",
    "subtle_tp",
    "false_negative",
    "hard_negative_fp",
    "high_amount_tn",
    "cold_start",
)

STEALTH_PCT = 75
TRAVEL_PCT = 90
AMOUNT_PCT = 99
CLEAR_TP_P = 0.95
CLEAR_TP_P_FALLBACK = 0.90
COLD_START_HISTORY_DAYS = 2.0


@dataclass(frozen=True)
class GalleryCase:
    """One selected transaction and the rule that selected it."""

    case: str
    position: int
    pool_size: int
    rule: str

    def to_dict(self) -> dict:
        return {
            "case": self.case,
            "position": self.position,
            "pool_size": self.pool_size,
            "rule": self.rule,
        }


def _argbest(
    frame: pd.DataFrame, mask: np.ndarray, by: list[str], ascending: list[bool]
) -> int | None:
    """Positional index of the best row under ``mask`` (ties: transaction_id asc)."""
    sub = frame.loc[mask]
    if sub.empty:
        return None
    ordered = sub.sort_values(
        [*by, "transaction_id"], ascending=[*ascending, True], kind="stable"
    )
    return int(ordered.index[0])


def select_cases(frame: pd.DataFrame) -> list[GalleryCase]:
    """Pick the six gallery cases from the ``val_thr`` analysis frame.

    ``frame`` must carry a positional RangeIndex and the columns
    ``is_fraud, p, action, amount, stealth, travel_speed_kmh, history_days``.
    """
    fraud = frame["is_fraud"].to_numpy(int) == 1
    legit = ~fraud
    flagged = frame["action"].to_numpy() != APPROVE
    approved = ~flagged
    stealth = frame["stealth"].to_numpy(dtype=float)
    travel = frame["travel_speed_kmh"].to_numpy(dtype=float)
    amount = frame["amount"].to_numpy(dtype=float)
    p = frame["p"].to_numpy(dtype=float)

    selected: list[GalleryCase] = []

    # 1. clear true positive: flagged fraud with a very high calibrated probability.
    mask = fraud & flagged & (p >= CLEAR_TP_P)
    rule = f"fraud, action != APPROVE, p >= {CLEAR_TP_P}; highest p"
    if not mask.any():
        mask = fraud & flagged & (p >= CLEAR_TP_P_FALLBACK)
        rule = f"fraud, action != APPROVE, p >= {CLEAR_TP_P_FALLBACK} (relaxed); highest p"
    selected.append(
        GalleryCase("clear_tp", _argbest(frame, mask, ["p"], [False]) or 0, int(mask.sum()), rule)
    )

    # 2. subtle true positive: flagged fraud with the highest stealth.
    stealth_cut = float(np.nanpercentile(stealth[fraud], STEALTH_PCT))
    mask = fraud & flagged & (stealth >= stealth_cut)
    selected.append(
        GalleryCase(
            "subtle_tp",
            _argbest(frame, mask, ["stealth", "p"], [False, False]) or 0,
            int(mask.sum()),
            f"fraud, action != APPROVE, stealth >= p{STEALTH_PCT} ({stealth_cut:.3f}); "
            "highest stealth",
        )
    )

    # 3. false negative: missed fraud with the highest stealth (most deceptive miss).
    mask = fraud & approved
    selected.append(
        GalleryCase(
            "false_negative",
            _argbest(frame, mask, ["stealth", "p"], [False, False]) or 0,
            int(mask.sum()),
            "fraud, action == APPROVE; highest stealth then highest p",
        )
    )

    # 4. hard-negative false positive: flagged legit row with an impossible travel speed.
    travel_cut = float(np.nanpercentile(travel[legit], TRAVEL_PCT))
    mask = legit & flagged & (travel >= travel_cut)
    rule = (
        f"legit, action != APPROVE, travel_speed >= p{TRAVEL_PCT} "
        f"({travel_cut:,.0f} km/h); highest p"
    )
    if not mask.any():
        mask = legit & flagged
        rule = "legit, action != APPROVE (travel filter empty); highest p"
    selected.append(
        GalleryCase(
            "hard_negative_fp", _argbest(frame, mask, ["p"], [False]) or 0, int(mask.sum()), rule
        )
    )

    # 5. high-amount legitimate true negative: approved legit row, largest amount.
    amount_cut = float(np.nanpercentile(amount, AMOUNT_PCT))
    mask = legit & approved & (amount >= amount_cut)
    selected.append(
        GalleryCase(
            "high_amount_tn",
            _argbest(frame, mask, ["amount"], [False]) or 0,
            int(mask.sum()),
            f"legit, action == APPROVE, amount >= p{AMOUNT_PCT} "
            f"(Rs {amount_cut:,.0f}); largest amount",
        )
    )

    # 6. cold start: account opened within the window and used almost immediately.
    # ``history_days < 0`` rows (simulator emits them: see docs/what_went_wrong.md)
    # are excluded on purpose - they are a generation defect, not cold-start evidence.
    history = frame["history_days"].to_numpy(dtype=float)
    mask = (history >= 0) & (history < COLD_START_HISTORY_DAYS)
    selected.append(
        GalleryCase(
            "cold_start",
            _argbest(frame, mask, ["p"], [False]) or 0,
            int(mask.sum()),
            f"0 <= history_days < {COLD_START_HISTORY_DAYS:g}; highest p",
        )
    )
    return selected
