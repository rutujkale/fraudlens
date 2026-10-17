"""Plotly figures for the analyst console (Part 14). Pure plotting only."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

ACTION_COLORS = {
    "APPROVE": "#2ca02c",
    "STEP_UP": "#ff7f0e",
    "REVIEW": "#d62728",
    "BLOCK": "#7f0000",
    "increase": "#d62728",
    "decrease": "#1f77b4",
}
MODEL_COLORS = {"xgb": "#1f77b4", "logreg": "#ff7f0e", "rules": "#2ca02c"}


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def pr_curve(curves: dict[str, Any], model: str) -> go.Figure:
    entry = curves.get("models", {}).get(model)
    if not entry:
        return _empty(f"no validation PR curve for {model}")
    fig = go.Figure(
        go.Scatter(
            x=entry["pr"]["recall"],
            y=entry["pr"]["precision"],
            mode="lines",
            name=model,
            line={"color": MODEL_COLORS.get(model)},
        )
    )
    base = 0.01
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02])
    fig.add_hline(y=base, line_dash="dot", annotation_text="prevalence (illustrative ~0.01)")
    return fig


def roc_curve(curves: dict[str, Any], model: str) -> go.Figure:
    entry = curves.get("models", {}).get(model)
    if not entry:
        return _empty(f"no validation ROC curve for {model}")
    fig = go.Figure(
        go.Scatter(
            x=entry["roc"]["fpr"],
            y=entry["roc"]["tpr"],
            mode="lines",
            name=model,
            line={"color": MODEL_COLORS.get(model)},
        )
    )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "grey"})
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return fig


def calibration_curve(bins: list[dict[str, float]], name: str) -> go.Figure:
    if not bins:
        return _empty("no calibration bins")
    x = [b["mean_pred"] for b in bins]
    y = [b["observed"] for b in bins]
    counts = [b["count"] for b in bins]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name=name,
            marker={"size": 8, "color": "#1f77b4"},
            text=[f"n={c}" for c in counts],
        )
    )
    fig.add_scatter(
        x=[0, max(x) or 1],
        y=[0, max(x) or 1],
        mode="lines",
        line={"dash": "dash", "color": "grey"},
    )
    fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed fraud rate")
    return fig


def score_histogram(bin_edges: list[float], counts: list[int]) -> go.Figure:
    if not counts:
        return _empty("no score histogram")
    centers = [(a + b) / 2 for a, b in zip(bin_edges[:-1], bin_edges[1:], strict=True)]
    return go.Figure(go.Bar(x=centers, y=counts)).update_layout(
        xaxis_title="Calibrated fraud probability", yaxis_title="Sample rows"
    )


def loss_curve(scan: pd.DataFrame) -> go.Figure:
    if scan.empty:
        return _empty("no threshold scan")
    fig = go.Figure()
    fig.add_scatter(
        x=scan["threshold"], y=scan["total_realized_cost"], mode="lines", name="realized"
    )
    fig.add_scatter(
        x=scan["threshold"], y=scan["total_expected_cost"], mode="lines", name="expected"
    )
    fig.update_layout(xaxis_title="Review threshold", yaxis_title="Cost (INR)")
    return fig


def action_bar(counts: dict[str, int]) -> go.Figure:
    names = list(counts)
    fig = go.Figure(
        go.Bar(
            x=names,
            y=[counts[n] for n in names],
            marker_color=[ACTION_COLORS[n] for n in names],
        )
    )
    return fig.update_layout(yaxis_title="Sample rows")


def shap_bar(contributions: list[dict[str, Any]], top: int = 12) -> go.Figure:
    rows = [c for c in contributions if c.get("shap_value") is not None]
    rows.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    rows = rows[:top][::-1]
    if not rows:
        return _empty("no SHAP contributions")
    fig = go.Figure(
        go.Bar(
            x=[c["shap_value"] for c in rows],
            y=[c["feature"] for c in rows],
            orientation="h",
            marker_color=[
                ACTION_COLORS["increase"] if c["shap_value"] >= 0 else ACTION_COLORS["decrease"]
                for c in rows
            ],
        )
    )
    fig.update_layout(xaxis_title="SHAP push on log-odds", yaxis_title="")
    return fig


def psi_heatmap(psi_by_month: pd.DataFrame, top: int = 15) -> go.Figure:
    if psi_by_month.empty:
        return _empty("no PSI table")
    ranked = (
        psi_by_month.groupby("feature")["psi"].max().sort_values(ascending=False).head(top).index
    )
    pivot = psi_by_month[psi_by_month["feature"].isin(ranked)].pivot(
        index="feature", columns="month", values="psi"
    )
    pivot = pivot.reindex(ranked)
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Reds",
            colorbar={"title": "PSI"},
        )
    )
    fig.update_layout(xaxis_title="Validation month", yaxis_title="Feature")
    return fig
