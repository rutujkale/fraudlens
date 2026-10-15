"""Experiment 8E (E14): drift and retraining policy on the ``drift_lab`` profile.

``drift_lab`` switches fraud behaviour at day 120 of 240 (scenario mix shifts to
structuring-heavy, stealth rises to subtler fraud). We backtest four retraining
policies month by month **after** the shift, always using only labels matured by
the step date (point-in-time):

* ``frozen``    — train once before the shift, never update.
* ``expanding`` — retrain on all matured data up to the step.
* ``sliding_60d`` — retrain on the last 60 days of matured data.
* ``recency_weighted`` — all matured data, samples weighted 0.5^(age/30d).

We separate **covariate** drift (PSI on features, adversarial-validation AUC)
from **concept** drift (does the model's ranking degrade on post-shift fraud?).
Uses the whole drift_lab timeline (a dedicated lab; no Part-12 test-window claim).
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from fraudlens.config import config_hash  # noqa: E402
from fraudlens.evaluation.experiments import drop_experiment_rows, log_experiment  # noqa: E402
from fraudlens.evaluation.experiments_lib import (  # noqa: E402
    best_xgb_params,
    load_supervised,
)
from fraudlens.evaluation.metrics import pr_auc, recall_at_top_k, roc_auc  # noqa: E402
from fraudlens.features.registry import NUMERIC_COLUMNS  # noqa: E402
from fraudlens.models.xgb import gain_importances  # noqa: E402
from fraudlens.monitoring.drift import psi_table  # noqa: E402
from fraudlens.paths import figures_dir, results_dir  # noqa: E402
from fraudlens.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)

SEED = 42
N_ESTIMATORS = 200
HALF_LIFE_DAYS = 30.0
SLIDING_DAYS = 60
ADV_SUBSAMPLE = 50000
STRATEGIES = ["frozen", "expanding", "sliding_60d", "recency_weighted"]
# Cumulative-count / age features that grow with the calendar. On a temporal
# split these alone separate pre from post almost perfectly (history_days is a
# clock), so they inflate naive PSI and adversarial AUC. Covariate-drift
# diagnostics must exclude them to see behavioural drift.
TENURE_FEATURES = ["history_days", "device_age_days", "n_prior_tx", "tx_from_device_prior"]


def _fit(x: pd.DataFrame, y: np.ndarray, params: dict, seed: int, sample_weight=None):
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        **params,
    )
    model.fit(x, y, sample_weight=sample_weight, verbose=False)
    return model


def _adv_auc(x: pd.DataFrame, idx: np.ndarray, y: np.ndarray, cols: list[str]) -> float:
    """AUC of an adversarial classifier that tries to separate pre from post."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x.iloc[idx][cols], y, verbose=False)
    return roc_auc(y, model.predict_proba(x.iloc[idx][cols])[:, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment 8E: drift backtest.")
    parser.add_argument("--profile", default="drift_lab")
    args = parser.parse_args()

    sup = load_supervised(args.profile)
    cfg = sup.cfg
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    # Replace only this dataset version's rows: earlier versions describe data that
    # no longer exists and are kept as history (Part 02b).
    drop_experiment_rows("E14", args.profile, dataset_version=cfg.simulation.dataset_version)

    params = best_xgb_params(args.profile)
    params = {k: (int(v) if k == "max_depth" else v) for k, v in params.items()}

    df = sup.df
    x_all = sup.x
    y_all = sup.y
    ts = pd.to_datetime(df["timestamp"])
    label_at = pd.to_datetime(df["label_available_at"])
    t0 = ts.min()
    shift = t0 + timedelta(days=int(cfg.simulation.drift.start_fraction * cfg.simulation.days))
    logger.info("profile=%s rows=%d shift=%s", args.profile, len(df), shift.date())

    steps = [shift + timedelta(days=30 * k) for k in range(1, 5)]
    frozen_mask = (ts < shift) & (label_at <= shift)

    def _train_mask(step: pd.Timestamp, strategy: str) -> np.ndarray:
        matured = (ts < step) & (label_at <= step)
        if strategy == "expanding":
            return matured
        if strategy == "sliding_60d":
            return matured & (ts >= step - timedelta(days=SLIDING_DAYS))
        return matured

    def _weights(step: pd.Timestamp, mask: np.ndarray) -> np.ndarray | None:
        age_days = (step - ts[mask]).dt.total_seconds() / 86400.0
        return np.power(0.5, age_days / HALF_LIFE_DAYS).to_numpy()

    backtest_rows = []
    frozen_model = None
    for step in steps:
        eval_mask = (ts > step - timedelta(days=30)) & (ts <= step)
        y_eval = y_all[eval_mask.to_numpy()]
        x_eval = x_all[eval_mask.to_numpy()]
        for strategy in STRATEGIES:
            if strategy == "frozen":
                if frozen_model is None:
                    fm = frozen_mask.to_numpy()
                    frozen_params = {**params, "scale_pos_weight": _spw(y_all[fm])}
                    frozen_model = _fit(x_all[fm], y_all[fm], frozen_params, SEED)
                model = frozen_model
                n_train = int(frozen_mask.sum())
            else:
                mask = _train_mask(step, strategy).to_numpy()
                spw = _spw(y_all[mask])
                weight = _weights(step, mask) if strategy == "recency_weighted" else None
                train_params = {**params, "scale_pos_weight": spw}
                model = _fit(x_all[mask], y_all[mask], train_params, SEED, weight)
                n_train = int(mask.sum())
            score = model.predict_proba(x_eval)[:, 1]
            backtest_rows.append(
                {
                    "step": str(step.date()),
                    "strategy": strategy,
                    "n_train": n_train,
                    "n_eval": int(eval_mask.sum()),
                    "n_fraud": int(y_eval.sum()),
                    "pr_auc": pr_auc(y_eval, score),
                    "recall_at_1pct": recall_at_top_k(y_eval, score, 0.01),
                }
            )
        logger.info("8E step=%s done", step.date())

    backtest = pd.DataFrame(backtest_rows)
    backtest.to_csv(results_dir() / f"exp_8E_backtest_{args.profile}.csv", index=False)

    # --- Covariate drift: PSI + adversarial validation ----------------------
    pre = df[ts < shift]
    post = df[ts >= shift]
    psi = psi_table(pre, post, NUMERIC_COLUMNS)
    psi["tenure"] = psi["feature"].isin(TENURE_FEATURES)
    psi.to_csv(results_dir() / f"exp_8E_psi_{args.profile}.csv", index=False)
    psi_behavioral = psi[~psi["tenure"]].reset_index(drop=True)
    logger.info(
        "PSI all: %d shifted; behavioural: %d shifted (top=%s)",
        int((psi["flag"] == "shift").sum()),
        int((psi_behavioral["flag"] == "shift").sum()),
        psi_behavioral.iloc[0]["feature"],
    )

    rng = np.random.default_rng(SEED)
    pre_pos = np.flatnonzero((ts < shift).to_numpy())
    post_pos = np.flatnonzero((ts >= shift).to_numpy())
    pre_s = rng.choice(pre_pos, size=min(ADV_SUBSAMPLE, len(pre_pos)), replace=False)
    post_s = rng.choice(post_pos, size=min(ADV_SUBSAMPLE, len(post_pos)), replace=False)
    adv_idx = np.concatenate([pre_s, post_s])
    adv_y = np.concatenate([np.zeros(len(pre_s)), np.ones(len(post_s))])

    all_cols = list(x_all.columns)
    beh_cols = [c for c in all_cols if not c.startswith(tuple(TENURE_FEATURES))]
    adv_auc_all = _adv_auc(x_all, adv_idx, adv_y, all_cols)
    adv_auc_beh = _adv_auc(x_all, adv_idx, adv_y, beh_cols)
    adv_auc_ten = _adv_auc(x_all, adv_idx, adv_y, TENURE_FEATURES)

    adv = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    adv.fit(x_all.iloc[adv_idx], adv_y, verbose=False)
    gain_importances(adv).head(15).to_csv(
        results_dir() / f"exp_8E_adversarial_{args.profile}.csv", index=False
    )
    (results_dir() / f"exp_8E_adversarial_summary_{args.profile}.json").write_text(
        json.dumps(
            {
                "adversarial_auc_all": adv_auc_all,
                "adversarial_auc_behavioral": adv_auc_beh,
                "adversarial_auc_tenure": adv_auc_ten,
                "tenure_features": TENURE_FEATURES,
                "n_pre": len(pre_s),
                "n_post": len(post_s),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "adversarial AUC: all=%.4f tenure=%.4f behavioural=%.4f",
        adv_auc_all,
        adv_auc_ten,
        adv_auc_beh,
    )

    # --- Figures ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy in STRATEGIES:
        sub = backtest[backtest["strategy"] == strategy]
        ax.plot(pd.to_datetime(sub["step"]), sub["pr_auc"], marker="o", label=strategy)
    ax.axvline(shift, color="grey", ls="--", lw=0.9, label="drift switch")
    ax.set_ylabel("PR-AUC (monthly)")
    ax.set_title("Post-drift PR-AUC by retraining policy")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8E_backtest_{args.profile}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    top = psi_behavioral.head(15).iloc[::-1]
    flag_color = {"shift": "tab:red", "monitor": "tab:orange", "stable": "tab:green"}
    colors = [flag_color[f] for f in top["flag"]]
    ax.barh(top["feature"], top["psi"], color=colors)
    ax.axvline(0.25, color="grey", ls="--", lw=0.8, label="shift 0.25")
    ax.axvline(0.10, color="grey", ls=":", lw=0.8, label="monitor 0.10")
    ax.set_xlabel("PSI (pre-shift vs post-shift)")
    ax.set_title("Top-15 behavioural feature drift (tenure features excluded)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_8E_psi_{args.profile}.png", dpi=120)
    plt.close(fig)

    for strategy in STRATEGIES:
        sub = backtest[backtest["strategy"] == strategy]
        log_experiment(
            {
                "experiment_id": "E14",
                "part": "08",
                "dataset_profile": args.profile,
                "model": "xgboost",
                "features_version": "features_v1",
                "hyperparameters": {"strategy": strategy, "n_estimators": N_ESTIMATORS},
                "calibration": "none",
                "threshold_policy": "rank_by_score",
                "metrics": {
                    "mean_pr_auc": round(float(sub["pr_auc"].mean()), 6),
                    "first_pr_auc": round(float(sub.iloc[0]["pr_auc"]), 6),
                    "last_pr_auc": round(float(sub.iloc[-1]["pr_auc"]), 6),
                },
                "notes": "post-drift monthly backtest; point-in-time training",
            },
            config_hash=chash,
        )
    log_experiment(
        {
            "experiment_id": "E14",
            "part": "08",
            "dataset_profile": args.profile,
            "model": "adversarial",
            "features_version": "features_v1",
            "hyperparameters": {"subsample": ADV_SUBSAMPLE},
            "calibration": "none",
            "threshold_policy": "n/a",
            "metrics": {
                "adversarial_auc_all": round(float(adv_auc_all), 6),
                "adversarial_auc_behavioral": round(float(adv_auc_beh), 6),
                "adversarial_auc_tenure": round(float(adv_auc_ten), 6),
                "psi_behavioral_shift_features": int((psi_behavioral["flag"] == "shift").sum()),
                "psi_behavioral_max": round(float(psi_behavioral.iloc[0]["psi"]), 6),
            },
            "notes": "covariate drift: pre vs post; full vs behavioural (tenure excluded)",
        },
        config_hash=chash,
    )
    logger.info("wrote 8E outputs")
    return 0


def _spw(y: np.ndarray) -> float:
    pos = max(1, int(y.sum()))
    neg = int((y == 0).sum())
    return float(neg / pos)


if __name__ == "__main__":
    raise SystemExit(main())
