"""Separability gate (Part 03C).

Proves the simulated data is not trivially solvable by a few cheap signals:
no single crude signal dominates, a shallow tree is clearly weaker than a
gradient-boosted reference (interactions matter), and the hard negatives
overlap with fraud on the crude signals (they are genuinely hard).

The crude signals come from ``data_quality.signals`` — they are NOT the
Part 05 features and must never be reused as model inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fraudlens.config import SeparabilityGateConfig
from fraudlens.data_quality.signals import SIGNAL_COLUMNS

DOMINANT_GAIN_SHARE = 0.60


@dataclass
class GateReport:
    profile: str
    n_rows: int
    n_fraud: int
    fraud_rate: float
    signals: dict[str, dict[str, float]]
    best_single_signal_roc_auc: float
    best_single_signal_pr_auc: float
    tree_stump_pr_auc: float
    tree_depth3_pr_auc: float
    boosted_pr_auc: float
    tree3_vs_boosted_gap: float
    overlap: dict[str, dict[str, Any]]
    overlap_signals_passed: int
    overlap_by_type: dict[str, list[str]]
    top_gain_importances: list[dict[str, Any]]
    dominant_feature: dict[str, Any] | None
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _temporal_split(n: int, train_end_fraction: float) -> np.ndarray:
    cut = int(n * train_end_fraction)
    mask = np.zeros(n, dtype=bool)
    mask[:cut] = True
    return mask


def _pr_auc(model: Any, x_eval: pd.DataFrame, y_eval: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_eval)[:, 1]
    else:
        scores = model.predict(x_eval)
    return float(average_precision_score(y_eval, scores))


def run_separability_gate(
    signals: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    gate_cfg: SeparabilityGateConfig,
    train_end_fraction: float,
    seed: int,
    profile: str = "dev",
) -> GateReport:
    """Compute the gate metrics and PASS/FAIL conditions."""
    merged = signals.merge(
        labels[["transaction_id", "is_fraud", "hard_negative_type"]],
        on="transaction_id",
        how="inner",
    ).sort_values("transaction_id", kind="stable").reset_index(drop=True)

    x = merged[SIGNAL_COLUMNS]
    y = merged["is_fraud"].to_numpy().astype(int)

    # 1. single-signal discrimination over the whole dataset
    signal_metrics: dict[str, dict[str, float]] = {}
    for col in SIGNAL_COLUMNS:
        vals = merged[col].to_numpy(dtype=float)
        signal_metrics[col] = {
            "roc_auc": float(roc_auc_score(y, vals)),
            "pr_auc": float(average_precision_score(y, vals)),
        }
    best_roc = max(m["roc_auc"] for m in signal_metrics.values())
    best_pr = max(m["pr_auc"] for m in signal_metrics.values())

    # 2. temporal split -> stumps / depth-3 / boosted reference
    train_mask = _temporal_split(len(merged), train_end_fraction)
    x_train, y_train = x[train_mask], y[train_mask]
    x_eval, y_eval = x[~train_mask], y[~train_mask]

    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    stump.fit(x_train, y_train)
    stump_pr = _pr_auc(stump, x_eval, y_eval)

    tree3 = DecisionTreeClassifier(max_depth=3, random_state=seed)
    tree3.fit(x_train, y_train)
    tree3_pr = _pr_auc(tree3, x_eval, y_eval)

    boosted = XGBClassifier(
        n_estimators=gate_cfg.boosted.n_estimators,
        max_depth=gate_cfg.boosted.max_depth,
        learning_rate=gate_cfg.boosted.learning_rate,
        subsample=gate_cfg.boosted.subsample,
        colsample_bytree=gate_cfg.boosted.colsample_bytree,
        random_state=seed,
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
    )
    boosted.fit(x_train, y_train)
    boosted_pr = _pr_auc(boosted, x_eval, y_eval)
    gap = boosted_pr - tree3_pr

    # top gain importances (context for Part 05 / Part 08, not acted on here)
    gains = boosted.get_booster().get_score(importance_type="gain")
    total_gain = sum(gains.values()) or 1.0
    top = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_gain = [
        {"feature": feat, "gain": float(val), "gain_share": float(val / total_gain)}
        for feat, val in top
    ]
    dominant = next((t for t in top_gain if t["gain_share"] > DOMINANT_GAIN_SHARE), None)

    # 3. hard-negative overlap on the crude signals
    fraud = merged[merged["is_fraud"]]
    hn = merged[merged["hard_negative_type"].notna()]
    fraud_median = fraud[SIGNAL_COLUMNS].median()
    overlap: dict[str, dict[str, Any]] = {}
    passed_signals = 0
    for col in SIGNAL_COLUMNS:
        hn_mean = float(hn[col].mean()) if len(hn) else float("nan")
        med = float(fraud_median[col])
        ok = hn_mean >= med
        passed_signals += int(ok)
        overlap[col] = {"hard_negative_mean": hn_mean, "fraud_median": med, "passes": ok}

    # per-type transparency (how many signals each hard-negative type overlaps)
    overlap_by_type: dict[str, list[str]] = {}
    for hn_type, group in hn.groupby("hard_negative_type"):
        hits = [c for c in SIGNAL_COLUMNS if group[c].mean() >= fraud_median[c]]
        overlap_by_type[str(hn_type)] = hits

    conditions = {
        "single_signal_roc_auc": {
            "threshold": gate_cfg.max_single_signal_roc_auc,
            "observed": best_roc,
            "passed": best_roc < gate_cfg.max_single_signal_roc_auc,
        },
        "single_signal_pr_auc": {
            "threshold": gate_cfg.max_single_signal_pr_auc,
            "observed": best_pr,
            "passed": best_pr < gate_cfg.max_single_signal_pr_auc,
        },
        "tree3_below_boosted": {
            "threshold": gate_cfg.tree3_vs_boosted_min_pr_auc_gap,
            "observed": gap,
            "passed": gap >= gate_cfg.tree3_vs_boosted_min_pr_auc_gap,
        },
        "hard_negative_overlap": {
            "threshold": gate_cfg.min_hard_negative_overlap_signals,
            "observed": passed_signals,
            "passed": passed_signals >= gate_cfg.min_hard_negative_overlap_signals,
        },
    }
    return GateReport(
        profile=profile,
        n_rows=int(len(merged)),
        n_fraud=int(y.sum()),
        fraud_rate=float(y.mean()),
        signals=signal_metrics,
        best_single_signal_roc_auc=best_roc,
        best_single_signal_pr_auc=best_pr,
        tree_stump_pr_auc=stump_pr,
        tree_depth3_pr_auc=tree3_pr,
        boosted_pr_auc=boosted_pr,
        tree3_vs_boosted_gap=gap,
        overlap=overlap,
        overlap_signals_passed=passed_signals,
        overlap_by_type=overlap_by_type,
        top_gain_importances=top_gain,
        dominant_feature=dominant,
        conditions=conditions,
        passed=all(c["passed"] for c in conditions.values()),
    )


def write_gate_report(report: GateReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)


def gate_markdown(report: GateReport) -> str:
    """Human-readable PASS/FAIL summary written to ``docs/separability_gate.md``."""
    lines = [
        f"# Separability Gate — profile `{report.profile}`",
        "",
        "Generated by `scripts/separability_gate.py`. Crude signals only; these are "
        "**not** Part 05 features.",
        "",
        f"**Overall: {'PASS' if report.passed else 'FAIL'}**",
        "",
        f"- rows: {report.n_rows:,} | fraud: {report.n_fraud:,} "
        f"(rate {report.fraud_rate:.4%})",
        "",
        "## Conditions",
        "",
        "| condition | observed | threshold | result |",
        "|---|---|---|---|",
    ]
    for name, cond in report.conditions.items():
        lines.append(
            f"| {name} | {float(cond['observed']):.4f} | {float(cond['threshold']):.4f} "
            f"| {'PASS' if cond['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Single-signal discrimination",
        "",
        "| signal | ROC-AUC | PR-AUC |",
        "|---|---|---|",
    ]
    for sig, metrics in report.signals.items():
        lines.append(f"| {sig} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} |")
    lines += [
        "",
        "## Model comparison (temporal split)",
        "",
        f"- depth-1 stump PR-AUC: {report.tree_stump_pr_auc:.4f}",
        f"- depth-3 tree PR-AUC: {report.tree_depth3_pr_auc:.4f}",
        f"- boosted reference PR-AUC: {report.boosted_pr_auc:.4f}",
        f"- gap (boosted − depth-3): {report.tree3_vs_boosted_gap:.4f}",
        "",
        "## Hard-negative overlap",
        "",
        "| signal | hard-negative mean | fraud median | passes |",
        "|---|---|---|---|",
    ]
    for sig, ov in report.overlap.items():
        lines.append(
            f"| {sig} | {ov['hard_negative_mean']:.3f} | {ov['fraud_median']:.3f} "
            f"| {'yes' if ov['passes'] else 'no'} |"
        )
    lines += [
        "",
        f"Signals overlapped: **{report.overlap_signals_passed}**.",
        "",
        "### Per hard-negative type (signals overlapped)",
        "",
    ]
    for hn_type, hits in sorted(report.overlap_by_type.items()):
        lines.append(f"- {hn_type}: {', '.join(hits) if hits else '(none)'}")
    lines += ["", "## Top-3 boosted gain importances (context for Parts 05 / 08)", ""]
    for item in report.top_gain_importances:
        lines.append(f"- {item['feature']}: gain share {item['gain_share']:.3f}")
    if report.dominant_feature:
        lines.append(
            f"\n> NOTE: `{report.dominant_feature['feature']}` dominates "
            f"({report.dominant_feature['gain_share']:.2f} of gain)."
        )
    lines.append("")
    return "\n".join(lines)
