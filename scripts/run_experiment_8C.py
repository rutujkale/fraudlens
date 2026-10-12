"""Experiment 8C (E12): capacity, nonlinearity and where the signal saturates.

Every XGBoost configuration pins the other hyperparameters at their Part 06 best
values and varies **only** ``max_depth`` (plus boosting rounds) — so the sweep
measures capacity, not tuning luck. Three views:

* **Capacity**: val and train PR-AUC vs ``max_depth`` in {2,3,4,6,8,10} (3 seeds),
  recording the early-stopping iteration.
* **Learning curves**: val/train PR-AUC vs training-set size (earliest 10/25/50/100%)
  and vs boosting round at the best depth.
* **Nonlinearity**: LogReg (linear) base vs +interaction terms vs +discretised
  numerics, against the XGBoost reference — how much of the tree model's edge is
  nonlinearity a linear model cannot see?

3 seeds (LogReg's lbfgs solver is deterministic, so it is fit once). Test window
never touched.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlens.config import config_hash  # noqa: E402
from fraudlens.evaluation.bootstrap import cluster_bootstrap  # noqa: E402
from fraudlens.evaluation.experiments import log_experiment  # noqa: E402
from fraudlens.evaluation.experiments_lib import (  # noqa: E402
    best_logreg_params,
    best_xgb_params,
    fit_xgb,
    load_supervised,
    mean_std,
    predict_scores,
)
from fraudlens.evaluation.metrics import pr_auc  # noqa: E402
from fraudlens.features.registry import FEATURE_COLUMNS  # noqa: E402
from fraudlens.models.logreg import make_logreg_pipeline  # noqa: E402
from fraudlens.paths import figures_dir, results_dir  # noqa: E402
from fraudlens.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)

SEEDS = [42, 43, 44]
DEPTHS = [2, 3, 4, 6, 8, 10]
SIZE_FRACTIONS = [0.10, 0.25, 0.50, 1.00]
INTERACTIONS = (
    ("is_new_device", "amount_ratio_to_30d_mean"),
    ("tx_count_1h", "amount_ratio_to_30d_mean"),
    ("distance_from_home_km", "is_night"),
)
BIN_COLUMNS = (
    "amount_ratio_to_30d_mean",
    "amount_zscore_30d",
    "distance_from_home_km",
    "tx_count_1h",
    "device_age_days",
)


def _augment_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    x = df.copy()
    names = []
    for a, b in INTERACTIONS:
        name = f"ix_{a}__{b}"
        x[name] = x[a].fillna(0.0) * x[b].fillna(0.0)
        names.append(name)
    return x, tuple(names)


def _augment_bins(train: pd.DataFrame, val: pd.DataFrame, q: int = 5):
    """Quantile-bin ``BIN_COLUMNS`` using edges fit on TRAIN, applied to both."""
    tr, va = train.copy(), val.copy()
    names = []
    for col in BIN_COLUMNS:
        edges = np.unique(
            np.nanquantile(tr[col].to_numpy(dtype=float), np.linspace(0, 1, q + 1)[1:-1])
        )
        name = f"{col}_bin"
        tr[name] = np.digitize(tr[col].to_numpy(dtype=float), edges).astype(float)
        va[name] = np.digitize(va[col].to_numpy(dtype=float), edges).astype(float)
        tr.loc[tr[col].isna(), name] = -1.0
        va.loc[va[col].isna(), name] = -1.0
        names.append(name)
    return tr, va, tuple(names)


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 8C: capacity and nonlinearity.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    seeds = SEEDS[: args.seeds]

    sup = load_supervised(args.profile)
    cfg = sup.cfg
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    base = best_xgb_params(args.profile)
    lr_base = best_logreg_params(args.profile)

    x_tr, y_tr = sup.x_train(), sup.y_train()
    x_va, y_va = sup.x_val(), sup.y_val()
    groups = sup.val_groups()
    logger.info("profile=%s train=%d val=%d", args.profile, len(y_tr), len(y_va))

    # --- Capacity sweep -----------------------------------------------------
    cap_rows = []
    best_depth = None
    for depth in DEPTHS:
        params = {**base, "max_depth": int(depth)}
        for seed in seeds:
            model = fit_xgb(
                x_tr, y_tr, x_va, y_va, params, seed, n_estimators=800, early_stopping_rounds=30
            )
            train_pr = pr_auc(y_tr, predict_scores(model, x_tr))
            val_pr = pr_auc(y_va, predict_scores(model, x_va))
            cap_rows.append(
                {
                    "depth": depth,
                    "seed": seed,
                    "best_iteration": int(getattr(model, "best_iteration", -1)),
                    "train_pr_auc": train_pr,
                    "val_pr_auc": val_pr,
                    "gap": train_pr - val_pr,
                }
            )
        logger.info(
            "8C depth=%-2d val PR-AUC=%.4f best_iter=%d",
            depth,
            val_pr,
            cap_rows[-1]["best_iteration"],
        )
    cap = pd.DataFrame(cap_rows)
    cap_summary = mean_std(cap, "depth", ["train_pr_auc", "val_pr_auc", "gap", "best_iteration"])
    cap.to_csv(results_dir() / f"exp_8C_capacity_{args.profile}.csv", index=False)
    # Pick the depth with the best MEAN val PR-AUC (not the single best seed).
    best_row = cap_summary.loc[cap_summary["val_pr_auc_mean"].idxmax()]
    best_depth, best_val = int(best_row["depth"]), float(best_row["val_pr_auc_mean"])
    logger.info("8C best depth by mean val PR-AUC = %s", best_depth)

    # --- Learning curve vs training size (earliest rows) --------------------
    n_tr = len(x_tr)
    size_rows = []
    for frac in SIZE_FRACTIONS:
        k = max(2, int(frac * n_tr))
        params = {**base, "max_depth": int(best_depth)}
        for seed in seeds:
            model = fit_xgb(x_tr.iloc[:k], y_tr[:k], x_va, y_va, params, seed)
            size_rows.append(
                {
                    "fraction": frac,
                    "n_train": k,
                    "seed": seed,
                    "train_pr_auc": pr_auc(y_tr[:k], predict_scores(model, x_tr.iloc[:k])),
                    "val_pr_auc": pr_auc(y_va, predict_scores(model, x_va)),
                }
            )
    size = pd.DataFrame(size_rows)
    size_summary = mean_std(size, "fraction", ["train_pr_auc", "val_pr_auc"])
    size_summary = size_summary.merge(
        size[["fraction", "n_train"]].drop_duplicates("fraction"), on="fraction"
    )
    size.to_csv(results_dir() / f"exp_8C_trainsize_{args.profile}.csv", index=False)

    # --- Learning curve vs boosting round at the best depth -----------------
    params = {**base, "max_depth": int(best_depth)}
    model = fit_xgb(
        x_tr, y_tr, x_va, y_va, params, seeds[0], n_estimators=800, early_stopping_rounds=30
    )
    best_iter = int(getattr(model, "best_iteration", 800)) or 800
    kmax = int(model.get_booster().num_boosted_rounds())
    grid = np.unique(np.clip(np.round(np.geomspace(1, kmax, 40)).astype(int), 1, kmax))
    round_rows = []
    for k in grid:
        tr = model.predict_proba(x_tr, iteration_range=(0, int(k)))[:, 1]
        va = model.predict_proba(x_va, iteration_range=(0, int(k)))[:, 1]
        round_rows.append(
            {"round": int(k), "train_pr_auc": pr_auc(y_tr, tr), "val_pr_auc": pr_auc(y_va, va)}
        )
    rounds = pd.DataFrame(round_rows)
    rounds.to_csv(results_dir() / f"exp_8C_rounds_{args.profile}.csv", index=False)

    # --- LogReg: linear vs interactions vs binned numerics ------------------
    train_raw = sup.df.iloc[sup.split.train_idx][FEATURE_COLUMNS]
    val_raw = sup.df.iloc[sup.split.val_idx][FEATURE_COLUMNS]
    C = float(lr_base.get("C", 1.0))
    variants = {}
    aug_tr_ix, ix_names = _augment_interactions(train_raw)
    aug_va_ix, _ = _augment_interactions(val_raw)
    variants["logreg_base"] = (make_logreg_pipeline(C=C), train_raw, val_raw)
    variants["logreg_interactions"] = (
        make_logreg_pipeline(C=C, extra_numeric=ix_names),
        aug_tr_ix,
        aug_va_ix,
    )
    bin_tr, bin_va, bin_names = _augment_bins(train_raw, val_raw)
    variants["logreg_binned"] = (
        make_logreg_pipeline(C=C, extra_categorical=bin_names),
        bin_tr,
        bin_va,
    )

    lr_rows = []
    for name, (pipe, tr, va) in variants.items():
        pipe.fit(tr, y_tr)
        score = pipe.predict_proba(va)[:, 1]
        point, lo, hi = cluster_bootstrap(
            y_va, score, groups, pr_auc, n=cfg.simulation.bootstrap.n, seed=cfg.simulation.seed
        )
        lr_rows.append({"variant": name, "pr_auc": point, "pr_auc_lo": lo, "pr_auc_hi": hi})
        logger.info("8C %-20s PR-AUC=%.4f [%.4f, %.4f]", name, point, lo, hi)
    lr = pd.DataFrame(lr_rows)
    lr.to_csv(results_dir() / f"exp_8C_logreg_{args.profile}.csv", index=False)

    # --- Figures ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    d = cap_summary["depth"].to_numpy()
    ax1.errorbar(
        d,
        cap_summary["val_pr_auc_mean"],
        yerr=cap_summary["val_pr_auc_std"],
        marker="o",
        label="val",
    )
    ax1.errorbar(
        d,
        cap_summary["train_pr_auc_mean"],
        yerr=cap_summary["train_pr_auc_std"],
        marker="s",
        label="train",
    )
    ax1.set_xlabel("max_depth")
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("Capacity: PR-AUC vs max_depth")
    ax1.legend(fontsize=8)
    ax2.bar([str(x) for x in d], cap_summary["gap_mean"], color="tab:orange")
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("train - val PR-AUC")
    ax2.set_title("Generalisation gap")
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8C_capacity_{args.profile}.png", dpi=120)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.errorbar(
        size_summary["n_train"],
        size_summary["val_pr_auc_mean"],
        yerr=size_summary["val_pr_auc_std"],
        marker="o",
        label="val",
    )
    ax1.errorbar(
        size_summary["n_train"],
        size_summary["train_pr_auc_mean"],
        yerr=size_summary["train_pr_auc_std"],
        marker="s",
        label="train",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("training rows (earliest)")
    ax1.set_ylabel("PR-AUC")
    ax1.set_title(f"Learning curve (depth={best_depth})")
    ax1.legend(fontsize=8)
    ax2.plot(rounds["round"], rounds["val_pr_auc"], label="val")
    ax2.plot(rounds["round"], rounds["train_pr_auc"], label="train")
    ax2.axvline(best_iter, color="grey", ls="--", lw=0.8, label=f"early stop {best_iter}")
    ax2.set_xlabel("boosting round")
    ax2.set_ylabel("PR-AUC")
    ax2.set_title(f"PR-AUC vs rounds (depth={best_depth})")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8C_learning_curves_{args.profile}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        lr["variant"],
        lr["pr_auc"],
        yerr=[lr["pr_auc"] - lr["pr_auc_lo"], lr["pr_auc_hi"] - lr["pr_auc"]],
        capsize=3,
    )
    ax.axhline(best_val, color="grey", ls="--", lw=0.8, label=f"XGB depth={best_depth}")
    ax.set_ylabel("PR-AUC")
    ax.set_title("LogReg variants vs XGB reference")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8C_logreg_{args.profile}.png", dpi=120)
    plt.close(fig)

    for _, r in cap_summary.iterrows():
        log_experiment(
            {
                "experiment_id": "E12",
                "part": "08",
                "dataset_profile": args.profile,
                "model": "xgboost",
                "features_version": "features_v1",
                "hyperparameters": {
                    "max_depth": int(r["depth"]),
                    "others": "part06_best_fixed",
                    "seeds": seeds,
                },
                "calibration": "none",
                "threshold_policy": "rank_by_score",
                "metrics": {
                    "val_pr_auc_mean": round(float(r["val_pr_auc_mean"]), 6),
                    "val_pr_auc_std": round(float(r["val_pr_auc_std"]), 6),
                    "train_pr_auc_mean": round(float(r["train_pr_auc_mean"]), 6),
                },
                "notes": "capacity sweep; only max_depth varies",
            },
            config_hash=chash,
        )
    for _, r in lr.iterrows():
        log_experiment(
            {
                "experiment_id": "E12",
                "part": "08",
                "dataset_profile": args.profile,
                "model": "logreg",
                "features_version": "features_v1",
                "hyperparameters": {"variant": r["variant"], "C": C},
                "calibration": "none",
                "threshold_policy": "rank_by_score",
                "metrics": {
                    "pr_auc": round(float(r["pr_auc"]), 6),
                    "pr_auc_lo": round(float(r["pr_auc_lo"]), 6),
                    "pr_auc_hi": round(float(r["pr_auc_hi"]), 6),
                },
                "notes": "nonlinearity probe (interactions / discretisation)",
            },
            config_hash=chash,
        )
    logger.info("best depth=%s (val PR-AUC %.4f); wrote 8C outputs", best_depth, best_val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
