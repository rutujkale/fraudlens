"""Thin CLI: Part 15A per-period monitoring over train + validation (no test access).

Writes ``results/monitoring_<profile>.csv`` (per-period metrics),
``results/monitoring_psi_<profile>.csv`` (per (period, feature) PSI) and three
figures. Prints the retraining-trigger verdict.
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlens.monitoring.metrics import (  # noqa: E402
    build_dashboard_metrics,
    retraining_trigger,
)
from fraudlens.paths import figures_dir, results_dir  # noqa: E402

DRIFT_DATES = {"dev": "2026-04-16"}


def _psi_heatmap(psi_long: pd.DataFrame, profile: str, out: object) -> None:
    pivot = psi_long.pivot(index="feature", columns="period", values="psi")
    pivot = pivot.reindex(pivot.max(axis=1).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * pivot.shape[1], 0.3 * len(pivot) + 2))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0.0)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_title(f"Feature PSI vs train window ({profile}, train+validation only)")
    fig.colorbar(im, ax=ax, label="PSI")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _labels(periods: pd.DataFrame) -> list[str]:
    return [f"{p}\n{s}" for p, s in zip(periods["period"], periods["sample"], strict=True)]


def _boundary_index(periods: pd.DataFrame) -> int | None:
    """First month that is not fully in-sample (where train_end falls)."""
    for i, sample in enumerate(periods["sample"]):
        if sample != "in_sample":
            return i
    return None


def _score_dist(periods: pd.DataFrame, out: object) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(periods)), periods["mean_pred"], label="mean predicted")
    ax.plot(range(len(periods)), periods["observed_rate"], "o-", color="black", label="observed")
    ax.set_xticks(range(len(periods)), _labels(periods), fontsize=8)
    boundary = _boundary_index(periods)
    if boundary is not None:
        ax.axvline(boundary - 0.5, color="grey", ls=":", lw=1)
        ax.text(
            boundary - 0.5,
            0.98,
            " model trained on data left of this line (in-sample)",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color="grey",
        )
    ax.set_ylabel("rate")
    ax.set_title("Calibration drift: mean predicted vs observed (matured labels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _performance(periods: pd.DataFrame, profile: str, out: object) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(periods)), periods["pr_auc"], "o-", label="PR-AUC (frozen model)")
    ax.plot(
        range(len(periods)),
        periods["recall_at_frozen"],
        "s--",
        label="recall @ frozen policy",
    )
    ax.set_xticks(range(len(periods)), _labels(periods), fontsize=8)
    boundary = _boundary_index(periods)
    if boundary is not None:
        ax.axvline(boundary - 0.5, color="grey", ls=":", lw=1)
        ax.text(
            0.01,
            0.98,
            "left of the line = in-sample (model trained on it); right = held-out validation",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="grey",
        )
    switch = DRIFT_DATES.get(profile)
    if switch:
        ax.text(
            0.99,
            0.02,
            f"drift switch {switch} (inside test window, beyond this plot)",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="grey",
        )
    ax.set_title(f"Delayed-label performance over time ({profile}, train+validation only)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FraudLens drift monitoring.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--model", default="xgb")
    args = parser.parse_args()

    metrics = build_dashboard_metrics(args.profile, model_name=args.model)
    periods: pd.DataFrame = metrics["periods"]  # type: ignore[assignment]
    psi_long: pd.DataFrame = metrics["psi_by_period"]  # type: ignore[assignment]

    out_dir = results_dir()
    periods_path = out_dir / f"monitoring_{args.profile}.csv"
    psi_path = out_dir / f"monitoring_psi_{args.profile}.csv"
    periods.to_csv(periods_path, index=False)
    psi_long.to_csv(psi_path, index=False)

    fig_dir = figures_dir()
    _psi_heatmap(psi_long, args.profile, fig_dir / f"monitoring_{args.profile}_psi_heatmap.png")
    _score_dist(periods, fig_dir / f"monitoring_{args.profile}_score_dist.png")
    _performance(periods, args.profile, fig_dir / f"monitoring_{args.profile}_performance.png")

    trigger = retraining_trigger(
        periods["score_psi"].tolist(), periods["recall_at_frozen"].tolist()
    )

    ref = metrics["reference"]
    sys.stdout.write(f"profile={args.profile} model={metrics['model_version']}\n")
    sys.stdout.write(f"threshold (frozen capacity)={metrics['threshold']:.6f}\n")
    sys.stdout.write(f"window: {ref['window']} (n={ref['n_rows']})\n")
    sys.stdout.write(periods.to_string(index=False) + "\n")
    sys.stdout.write(f"wrote {periods_path}\nwrote {psi_path}\n")
    sys.stdout.write(f"retraining_trigger -> {trigger}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
