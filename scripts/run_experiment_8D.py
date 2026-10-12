"""Experiment 8D (E13): importance triangulation — gain vs permutation vs SHAP.

Three attribution methods are compared on the final Part 06 XGBoost config:

* **gain** — split-based, biased toward high-cardinality/continuous features.
* **permutation** — PR-AUC drop when a feature's values are shuffled, measured
  per feature AND per correlated group (features with |Spearman| > 0.8 permuted
  together, so a cluster of proxies is not credited individually).
* **mean|SHAP|** — mean absolute TreeSHAP value on a class-stratified subsample.

A random-noise column and a shuffled copy of the strongest feature are added as
a sanity check: all three methods should rank them at the bottom. Spearman rank
correlations between methods show where they agree. Test window never touched.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fraudlens.config import config_hash  # noqa: E402
from fraudlens.evaluation.experiments import log_experiment  # noqa: E402
from fraudlens.evaluation.experiments_lib import (  # noqa: E402
    best_xgb_params,
    fit_xgb,
    load_supervised,
    predict_scores,
)
from fraudlens.evaluation.metrics import pr_auc  # noqa: E402
from fraudlens.models.xgb import gain_importances  # noqa: E402
from fraudlens.paths import figures_dir, results_dir  # noqa: E402
from fraudlens.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)

SEED = 42
SHAP_ROWS = 10000
PERM_REPEATS = 5
CORR_THRESHOLD = 0.80


def _permutation_drop(
    model, x: pd.DataFrame, y: np.ndarray, cols: list[str], seed: int
) -> tuple[float, float]:
    """Mean (and std) PR-AUC drop when ``cols`` are jointly shuffled, 5 repeats."""
    base = pr_auc(y, predict_scores(model, x))
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(PERM_REPEATS):
        xp = x.copy()
        order = rng.permutation(len(xp))
        block = xp[cols].to_numpy().copy()
        xp[cols] = block[order]
        drops.append(base - pr_auc(y, predict_scores(model, xp)))
    return float(np.mean(drops)), float(np.std(drops))


def _corr_clusters(x: pd.DataFrame, threshold: float) -> dict[str, str]:
    """Union-find clusters of features with |Spearman| > ``threshold``."""
    corr = x.corr(method="spearman").abs().to_numpy()
    cols = list(x.columns)
    parent = list(range(len(cols)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if np.isfinite(corr[i, j]) and corr[i, j] > threshold:
                parent[find(i)] = find(j)
    roots = {find(i): f"cluster_{k}" for k, i in enumerate(range(len(cols)))}
    return {cols[i]: roots[find(i)] for i in range(len(cols))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 8D: importance triangulation.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()

    sup = load_supervised(args.profile)
    cfg = sup.cfg
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    params = best_xgb_params(args.profile)
    params["max_depth"] = int(params["max_depth"])

    x_tr, y_tr = sup.x_train(), sup.y_train()
    x_va, y_va = sup.x_val(), sup.y_val()
    logger.info(
        "profile=%s train=%d val=%d features=%d", args.profile, len(y_tr), len(y_va), x_tr.shape[1]
    )

    # Strongest feature from the production config, to build the shuffled control.
    prod = fit_xgb(x_tr, y_tr, x_va, y_va, params, SEED)
    top_feature = str(gain_importances(prod).iloc[0]["feature"])
    logger.info("top gain feature: %s", top_feature)

    # Sanity-check model: real features + a random-noise column + a shuffled copy.
    rng = np.random.default_rng(SEED)
    noise_col = "_random_noise"
    shuffled_col = f"_shuffled_{top_feature}"
    x_tr_s, x_va_s = x_tr.copy(), x_va.copy()
    x_tr_s[noise_col] = rng.normal(size=len(x_tr_s))
    x_va_s[noise_col] = rng.normal(size=len(x_va_s))
    x_tr_s[shuffled_col] = rng.permutation(x_tr[top_feature].to_numpy())
    x_va_s[shuffled_col] = rng.permutation(x_va[top_feature].to_numpy())
    model = fit_xgb(x_tr_s, y_tr, x_va_s, y_va, params, SEED)
    logger.info("sanity model val PR-AUC=%.4f", pr_auc(y_va, predict_scores(model, x_va_s)))

    # gain
    gain = gain_importances(model).set_index("feature")["gain_share"]

    # permutation (per feature and per correlated group)
    features = list(x_va_s.columns)
    clusters = _corr_clusters(x_tr_s, CORR_THRESHOLD)
    perm, perm_std, perm_grp = {}, {}, {}
    for f in features:
        m, s = _permutation_drop(model, x_va_s, y_va, [f], SEED)
        perm[f], perm_std[f] = m, s
    for cluster in sorted(set(clusters.values())):
        members = [f for f, c in clusters.items() if c == cluster]
        m, _ = _permutation_drop(model, x_va_s, y_va, members, SEED)
        for f in members:
            perm_grp[f] = m

    # SHAP on a class-stratified subsample
    _, sample_idx = train_test_split(
        np.arange(len(y_va)), test_size=min(SHAP_ROWS, len(y_va)), stratify=y_va, random_state=SEED
    )
    x_sample = x_va_s.iloc[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=features)

    table = pd.DataFrame(
        {
            "feature": features,
            "cluster": [clusters[f] for f in features],
            "gain_share": [float(gain.get(f, 0.0)) for f in features],
            "perm_pr_auc_drop": [perm[f] for f in features],
            "perm_pr_auc_drop_std": [perm_std[f] for f in features],
            "perm_group_drop": [perm_grp[f] for f in features],
            "mean_abs_shap": [float(mean_abs_shap[f]) for f in features],
        }
    )
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table.to_csv(results_dir() / f"exp_8D_importance_{args.profile}.csv", index=False)

    methods = ["gain_share", "perm_pr_auc_drop", "perm_group_drop", "mean_abs_shap"]
    corr = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            rho, _ = spearmanr(table[a], table[b])
            corr.loc[a, b] = float(rho)
    corr.to_csv(results_dir() / f"exp_8D_method_agreement_{args.profile}.csv")
    logger.info("method rank correlations:\n%s", corr.round(3).to_string())

    bottom = table.tail(4)[["feature", "mean_abs_shap", "perm_pr_auc_drop"]]
    logger.info(
        "lowest-ranked features (controls should be here):\n%s", bottom.to_string(index=False)
    )

    # Figure: top-15 by mean|SHAP| with normalised gain + permutation.
    top = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ypos = np.arange(len(top))
    ax.barh(
        ypos - 0.28,
        top["mean_abs_shap"] / top["mean_abs_shap"].max(),
        height=0.25,
        label="mean|SHAP| (norm)",
    )
    ax.barh(
        ypos,
        top["perm_pr_auc_drop"] / max(top["perm_pr_auc_drop"].max(), 1e-9),
        height=0.25,
        label="perm drop (norm)",
    )
    ax.barh(
        ypos + 0.28,
        top["gain_share"] / max(top["gain_share"].max(), 1e-9),
        height=0.25,
        label="gain (norm)",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("normalised importance")
    ax.set_title("Top-15 features by mean|SHAP| — three methods")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8D_importance_{args.profile}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8
            )
    ax.set_title("Spearman rank correlation between methods")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8D_method_agreement_{args.profile}.png", dpi=120)
    plt.close(fig)

    log_experiment(
        {
            "experiment_id": "E13",
            "part": "08",
            "dataset_profile": args.profile,
            "model": "xgboost",
            "features_version": "features_v1",
            "hyperparameters": {**params, "shap_rows": SHAP_ROWS, "perm_repeats": PERM_REPEATS},
            "calibration": "none",
            "threshold_policy": "rank_by_score",
            "metrics": {
                "shap_perm_spearman": round(
                    float(corr.loc["mean_abs_shap", "perm_pr_auc_drop"]), 4
                ),
                "gain_perm_spearman": round(float(corr.loc["gain_share", "perm_pr_auc_drop"]), 4),
                "top_feature": top_feature,
            },
            "notes": "gain vs permutation vs SHAP; noise+shuffled controls",
        },
        config_hash=chash,
    )
    logger.info("wrote 8D outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
