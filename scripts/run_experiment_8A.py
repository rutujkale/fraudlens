"""Experiment 8A (E10): imbalance handling — does it help, and at what metric cost?

Variants, all on the same temporal split and the same fixed Part 06 XGBoost
config (only the imbalance treatment changes): none (scale_pos_weight=1),
scale_pos_weight = neg/pos, = sqrt(neg/pos), random undersampling of the
training negatives to 1:10 and 1:50 (train only), and LogReg
``class_weight="balanced"``. SMOTE is deliberately excluded (see Caveats).

Reports PR-AUC, recall@1%, and Brier/ECE **before vs after** sigmoid calibration
(calibrator fit on the first chronological half of validation, Brier/ECE read on
the second half). Test window never touched.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlens.config import config_hash  # noqa: E402
from fraudlens.evaluation.experiments import log_experiment  # noqa: E402
from fraudlens.evaluation.experiments_lib import (  # noqa: E402
    best_logreg_params,
    best_xgb_params,
    calibration_before_after,
    fit_xgb,
    load_supervised,
    mean_std,
    predict_scores,
    score_row,
)
from fraudlens.evaluation.figures import plot_recall_at_k  # noqa: E402
from fraudlens.evaluation.splits import chronological_fraction_split  # noqa: E402
from fraudlens.features.registry import FEATURE_COLUMNS  # noqa: E402
from fraudlens.models.logreg import make_logreg_pipeline  # noqa: E402
from fraudlens.paths import figures_dir, results_dir  # noqa: E402
from fraudlens.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)

SEEDS = [42, 43, 44]
SCORE_COLS = [
    "pr_auc",
    "pr_auc_lo",
    "pr_auc_hi",
    "recall_at_1pct",
    "brier",
    "ece",
    "brier_before",
    "brier_after",
    "ece_before",
    "ece_after",
]


def _undersample(
    x: pd.DataFrame, y: np.ndarray, ratio: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep all positives and ``ratio`` negatives per positive (train only)."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n_neg = min(len(neg), int(ratio * len(pos)))
    keep = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    keep.sort()
    return x.iloc[keep], y[keep]


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 8A: imbalance handling.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    seeds = SEEDS[: args.seeds]

    sup = load_supervised(args.profile)
    cfg = sup.cfg
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)

    xgb_base = best_xgb_params(args.profile)
    lr_base = best_logreg_params(args.profile)
    x_tr, y_tr = sup.x_train(), sup.y_train()
    x_va, y_va = sup.x_val(), sup.y_val()

    neg_pos = float((len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))
    logger.info(
        "profile=%s train=%d val=%d neg/pos=%.1f", args.profile, len(y_tr), len(y_va), neg_pos
    )

    val_idx = sup.split.val_idx
    cal_pos, thr_pos = chronological_fraction_split(
        np.arange(len(val_idx)), cfg.simulation.calibration.cal_fraction
    )
    y_cal, y_thr = y_va[cal_pos], y_va[thr_pos]
    logger.info(
        "val_cal=%d pos=%d | val_thr=%d pos=%d", len(y_cal), y_cal.sum(), len(y_thr), y_thr.sum()
    )

    variants = [
        ("none", {"scale_pos_weight": 1.0}),
        ("spw_full", {"scale_pos_weight": neg_pos}),
        ("spw_sqrt", {"scale_pos_weight": float(np.sqrt(neg_pos))}),
        ("undersample_1_10", {"scale_pos_weight": 1.0}),
        ("undersample_1_50", {"scale_pos_weight": 1.0}),
    ]

    rows: list[dict] = []
    scores_by_variant: dict[str, np.ndarray] = {}
    for name, override in variants:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            xt, yt = x_tr, y_tr
            if name == "undersample_1_10":
                xt, yt = _undersample(x_tr, y_tr, 10.0, rng)
            elif name == "undersample_1_50":
                xt, yt = _undersample(x_tr, y_tr, 50.0, rng)
            params = {**xgb_base, **override}
            params["max_depth"] = int(params["max_depth"])
            model = fit_xgb(xt, yt, x_va, y_va, params, seed)
            raw = predict_scores(model, x_va)
            row = score_row(name, y_va, raw, cfg, groups=sup.val_groups(), with_ci=True, seed=seed)
            row.update(
                calibration_before_after(
                    y_cal, raw[cal_pos], y_thr, raw[thr_pos], cfg.simulation.calibration.bins
                )
            )
            row["seed"] = seed
            row["train_rows"] = int(len(yt))
            rows.append(row)
            if seed == seeds[0]:
                scores_by_variant[name] = raw
            logger.info(
                "8A %-16s seed=%d PR-AUC=%.4f recall@1%%=%.3f Brier %.5f->%.5f",
                name,
                seed,
                row["pr_auc"],
                row["recall_at_1pct"],
                row["brier_before"],
                row["brier_after"],
            )

    for seed in seeds:
        pipe = make_logreg_pipeline(C=float(lr_base["C"]), class_weight="balanced")
        pipe.fit(sup.df.iloc[sup.split.train_idx][FEATURE_COLUMNS], y_tr)
        raw = pipe.predict_proba(sup.df.iloc[val_idx][FEATURE_COLUMNS])[:, 1]
        name = "logreg_balanced"
        row = score_row(name, y_va, raw, cfg, groups=sup.val_groups(), with_ci=True, seed=seed)
        row.update(
            calibration_before_after(
                y_cal, raw[cal_pos], y_thr, raw[thr_pos], cfg.simulation.calibration.bins
            )
        )
        row["seed"] = seed
        row["train_rows"] = int(len(y_tr))
        rows.append(row)
        scores_by_variant.setdefault(name, raw)
        logger.info("8A %-16s seed=%d PR-AUC=%.4f", name, seed, row["pr_auc"])

    per_seed = pd.DataFrame(rows)
    summary = mean_std(per_seed, "config", SCORE_COLS)
    out = results_dir()
    per_seed.to_csv(out / f"exp_8A_imbalance_{args.profile}.csv", index=False)
    summary.to_csv(out / f"exp_8A_imbalance_summary_{args.profile}.csv", index=False)
    logger.info("wrote %s", out / f"exp_8A_imbalance_summary_{args.profile}.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    s = summary.set_index("config").loc[[v[0] for v in variants] + ["logreg_balanced"]]
    ax1.bar(s.index, s["pr_auc_mean"], yerr=s["pr_auc_std"], color="tab:blue", capsize=3)
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("PR-AUC by imbalance treatment")
    ax1.tick_params(axis="x", rotation=40, labelsize=8)
    width = 0.38
    xs = np.arange(len(s))
    ax2.bar(xs - width / 2, s["brier_before_mean"], width, label="raw", color="tab:orange")
    ax2.bar(xs + width / 2, s["brier_after_mean"], width, label="sigmoid", color="tab:green")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(s.index, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Brier (val_thr)")
    ax2.set_title("Brier before vs after sigmoid")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8A_imbalance_{args.profile}.png", dpi=120)
    plt.close(fig)

    plot_recall_at_k(
        y_va,
        scores_by_variant,
        cfg.simulation.ks,
        figures_dir() / f"exp_8A_recall_at_k_{args.profile}.png",
    )

    for _, r in summary.iterrows():
        log_experiment(
            {
                "experiment_id": "E10",
                "part": "08",
                "dataset_profile": args.profile,
                "model": "xgboost" if r["config"] != "logreg_balanced" else "logreg",
                "features_version": "features_v1",
                "train_window": {
                    "start": sup.split.boundaries["t0"],
                    "end": sup.split.boundaries["t_train_end"],
                },
                "val_window": {
                    "start": sup.split.boundaries["t_train_end"],
                    "end": sup.split.boundaries["t_val_end"],
                },
                "hyperparameters": {"imbalance": r["config"], "seeds": seeds},
                "calibration": "raw_and_sigmoid",
                "threshold_policy": "rank_by_score",
                "metrics": {
                    k: round(float(r[k]), 6)
                    for k in (
                        "pr_auc_mean",
                        "pr_auc_std",
                        "recall_at_1pct_mean",
                        "brier_mean",
                        "ece_mean",
                    )
                },
                "notes": "imbalance sweep; 3 seeds; calib on val_cal, metrics on full val",
            },
            config_hash=chash,
        )
    logger.info("logged %d E10 rows", len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
