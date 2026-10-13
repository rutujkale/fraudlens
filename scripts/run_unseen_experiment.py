"""Part 09 (E15/E16): can an unsupervised anomaly score help a supervised model
detect a fraud pattern that was *absent* from supervised training?

Held-out scenario (`--holdout`, default ``structuring``) is removed from the
supervised training rows only (not relabelled); validation keeps it. We compare:

* S0  — XGBoost without the held-out scenario (the "unseen" baseline)
* S1  — S0 + cross-fitted IF-all anomaly score
* S1s — S0 + cross-fitted IF-strict anomaly score (IF never sees the held-out rows)
* S2  — Isolation Forest alone
* S3  — rank-average blend of S0 and IF-all (weight tuned on non-held-out val fraud)
* S4  — XGBoost trained *with* the held-out scenario (oracle ceiling)

Headline metric is held-out-scenario recall at fixed alert budgets; the fair test
is the **paired** S1−S0 (and S1s−S0) cluster-bootstrap CI. Validation window only.
"""

from __future__ import annotations

import argparse
import json
from functools import partial

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraudlens.config import config_hash  # noqa: E402
from fraudlens.evaluation.bootstrap import cluster_bootstrap  # noqa: E402
from fraudlens.evaluation.experiments import log_experiment  # noqa: E402
from fraudlens.evaluation.experiments_lib import (  # noqa: E402
    best_xgb_params,
    fit_xgb,
    load_supervised,
    predict_scores,
)
from fraudlens.evaluation.metrics import pr_auc, recall_at_top_k, roc_auc  # noqa: E402
from fraudlens.models.anomaly import (  # noqa: E402
    cross_fit_scores,
    fit_anomaly,
    grid_search,
)
from fraudlens.models.calibration import paired_cluster_bootstrap_diff  # noqa: E402
from fraudlens.paths import artifacts_dir, figures_dir, results_dir  # noqa: E402
from fraudlens.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)

SEEDS = [42, 43, 44, 45, 46]
BUDGETS = [0.005, 0.01, 0.02, 0.05]
HEADLINE_BUDGET = 0.01
BLEND_WEIGHTS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)


def _rank01(x: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0, 1] (ties by stable order)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks / max(1, len(x) - 1)


def _recall_at_budget(y_pos: np.ndarray, score: np.ndarray, budget: float) -> float:
    """Recall over ``y_pos==1`` rows captured in the top ``budget`` of *all* rows."""
    return recall_at_top_k(y_pos, score, budget)


def _drop_prior_log_rows(exp_id: str, holdout: str) -> None:
    """Remove earlier ``experiments.csv`` rows for this experiment/holdout (idempotent log)."""
    path = results_dir() / "experiments.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    dupe = (df["experiment_id"] == exp_id) & df["notes"].astype(str).str.contains(
        f"holdout={holdout}", regex=False
    )
    if dupe.any():
        df[~dupe].to_csv(path, index=False)


def _eval_row(
    system: str,
    seed: int,
    score: np.ndarray,
    *,
    y_val_struct: np.ndarray,
    y_val: np.ndarray,
    seen_y: dict[str, np.ndarray],
) -> dict:
    row = {"system": system, "seed": seed}
    for b in BUDGETS:
        row[f"holdout_recall_{b:g}"] = _recall_at_budget(y_val_struct, score, b)
    row["pr_auc_overall"] = pr_auc(y_val, score)
    row["recall_1pct_overall"] = recall_at_top_k(y_val, score, HEADLINE_BUDGET)
    for sc, y_sc in seen_y.items():
        row[f"recall_1pct_{sc}"] = recall_at_top_k(y_sc, score, HEADLINE_BUDGET)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 09 unseen-fraud experiment.")
    parser.add_argument("--profile", default="unseen_lab")
    parser.add_argument("--holdout", default="structuring")
    args = parser.parse_args()

    sup = load_supervised(args.profile)
    cfg = sup.cfg
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)

    df, x, y, split = sup.df, sup.x, sup.y, sup.split
    scenario = df["fraud_scenario"].to_numpy()
    holdout = args.holdout
    is_hold = scenario == holdout
    seen = [s for s in cfg.simulation.fraud_scenarios if s != holdout]

    tr, va = split.train_idx, split.val_idx
    tr_nohold = tr[~is_hold[tr]]
    va_nohold = va[~is_hold[va]]
    y_tr = y[tr]
    y_tn = y[tr_nohold]
    y_va = y[va]
    y_va_struct = is_hold[va].astype(int)
    groups_tr = sup.groups[tr]
    groups_va = sup.groups[va]
    seen_y = {sc: (scenario[va] == sc).astype(int) for sc in seen}

    logger.info(
        "profile=%s holdout=%s train=%d (nohold=%d) val=%d holdout_pos=%d",
        args.profile, holdout, len(tr), len(tr_nohold), len(va), int(y_va_struct.sum()),
    )
    if int(y_va_struct.sum()) < 300:
        logger.warning(
            "only %d held-out positives in eval window (<300): results will be noisy",
            int(y_va_struct.sum()),
        )

    # --- IF hyperparameters chosen on non-held-out val fraud only ------------
    best_if, grid_tbl = grid_search(
        df.iloc[tr], df.iloc[va_nohold], y[va_nohold], seed=cfg.simulation.seed
    )
    grid_tbl.to_csv(results_dir() / f"exp_09_if_grid_{args.profile}.csv", index=False)
    logger.info("IF best params=%s (val PR-AUC %.4f)", best_if, grid_tbl.iloc[0]["val_pr_auc"])

    # --- anomaly scores (cross-fitted for train rows) ------------------------
    anom_all_tr = cross_fit_scores(df.iloc[tr], groups_tr, seed=cfg.simulation.seed, **best_if)
    anom_strict_tr = cross_fit_scores(
        df.iloc[tr], groups_tr, seed=cfg.simulation.seed,
        fit_exclude_mask=is_hold[tr], **best_if,
    )
    full_all = fit_anomaly(df.iloc[tr], seed=cfg.simulation.seed, **best_if)
    full_strict = fit_anomaly(
        df.iloc[tr], seed=cfg.simulation.seed, exclude_mask=is_hold[tr], **best_if
    )
    anom_all_va = full_all.score(df.iloc[va])
    anom_strict_va = full_strict.score(df.iloc[va])

    keep = ~is_hold[tr]
    anom_all_tn = anom_all_tr[keep]
    anom_strict_tn = anom_strict_tr[keep]

    x_tn, x_tr_all, x_va = x.iloc[tr_nohold], x.iloc[tr], x.iloc[va]
    x_va_nohold = x.iloc[va_nohold]
    y_va_nohold = y[va_nohold]

    params = best_xgb_params(args.profile)
    params = {k: (int(v) if k == "max_depth" else v) for k, v in params.items()}

    def with_anom(base: pd.DataFrame, col: np.ndarray) -> pd.DataFrame:
        out = base.copy()
        out["anomaly_score"] = col
        return out

    nh = ~is_hold[va]
    x_va_anom_all = with_anom(x_va, anom_all_va)
    x_va_anom_strict = with_anom(x_va, anom_strict_va)
    x_va_nohold_anom_all = with_anom(x_va_nohold, anom_all_va[nh])
    x_va_nohold_anom_strict = with_anom(x_va_nohold, anom_strict_va[nh])
    x_tn_anom_all = with_anom(x_tn, anom_all_tn)
    x_tn_anom_strict = with_anom(x_tn, anom_strict_tn)

    rows: list[dict] = []
    s0_scores: list[np.ndarray] = []
    s1_scores: list[np.ndarray] = []
    s1s_scores: list[np.ndarray] = []
    rank_anom = _rank01(anom_all_va)

    for seed in SEEDS:
        s0 = predict_scores(fit_xgb(x_tn, y_tn, x_va_nohold, y_va_nohold, params, seed), x_va)
        s1 = predict_scores(
            fit_xgb(
                x_tn_anom_all, y_tn, x_va_nohold_anom_all, y_va_nohold, params, seed
            ),
            x_va_anom_all,
        )
        s1s = predict_scores(
            fit_xgb(
                x_tn_anom_strict, y_tn, x_va_nohold_anom_strict, y_va_nohold, params, seed
            ),
            x_va_anom_strict,
        )
        s4 = predict_scores(fit_xgb(x_tr_all, y_tr, x_va_nohold, y_va_nohold, params, seed), x_va)
        s0_scores.append(s0)
        s1_scores.append(s1)
        s1s_scores.append(s1s)

        # S3: rank-average blend, weight tuned on non-held-out val fraud only.
        rank_s0 = _rank01(s0)
        best_w, best_auc = 0.0, -np.inf
        for w in BLEND_WEIGHTS:
            blended = w * rank_anom + (1 - w) * rank_s0
            auc = pr_auc(y_va_nohold, blended[nh])
            if auc > best_auc:
                best_w, best_auc = float(w), auc
        s3 = best_w * rank_anom + (1 - best_w) * rank_s0

        for name, s in [("S0", s0), ("S1", s1), ("S1s", s1s), ("S3", s3), ("S4", s4)]:
            rows.append(
                _eval_row(
                    name, seed, s,
                    y_val_struct=y_va_struct, y_val=y_va, seen_y=seen_y,
                )
            )
        rows[-2]["blend_weight"] = best_w  # S3 row (order: S0,S1,S1s,S3,S4)

    # S2: IF-alone — no training seed; report once.
    rows.append(
        _eval_row(
            "S2", -1, anom_all_va,
            y_val_struct=y_va_struct, y_val=y_va, seen_y=seen_y,
        )
    )

    results = pd.DataFrame(rows)
    out_name = (
        "exp_09_unseen.csv"
        if holdout == "structuring"
        else f"exp_09_unseen_variants_{holdout}.csv"
    )
    results.to_csv(results_dir() / out_name, index=False)

    metric_cols = [c for c in results.columns if c not in ("system", "seed", "blend_weight")]
    summary = results.groupby("system")[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["system"] + [f"{c}_{a}" for c, a in summary.columns[1:]]
    summary.to_csv(
        results_dir() / f"exp_09_unseen_summary_{args.profile}_{holdout}.csv", index=False
    )

    # --- paired S1/S1s vs S0 CI on the headline metric -----------------------
    # Report both the per-seed paired diffs (mean ± std over 5 training seeds) and
    # the seed-bagged comparison (the mean score across seeds as one ranker). If
    # the two disagree, the seed-bagged win is driven by bagging, not robustness.
    metric_fn = partial(_recall_at_budget, budget=HEADLINE_BUDGET)
    s0_arr = np.mean(np.vstack(s0_scores), axis=0)
    paired, per_seed_paired = {}, {}
    for name, seed_scores in [("S1", s1_scores), ("S1s", s1s_scores)]:
        diffs = []
        for i, seed in enumerate(SEEDS):
            d, lo_s, hi_s = paired_cluster_bootstrap_diff(
                y_va_struct, seed_scores[i], s0_scores[i], groups_va, metric_fn,
                n=cfg.simulation.bootstrap.n, seed=cfg.simulation.seed,
            )
            per_seed_paired.setdefault(name, []).append(
                {"seed": seed, "diff": d, "lo": lo_s, "hi": hi_s}
            )
            diffs.append(d)
        diff, lo, hi = paired_cluster_bootstrap_diff(
            y_va_struct, np.mean(np.vstack(seed_scores), axis=0), s0_arr, groups_va, metric_fn,
            n=cfg.simulation.bootstrap.n, seed=cfg.simulation.seed,
        )
        paired[name] = {
            "diff_bagged": diff,
            "lo_bagged": lo,
            "hi_bagged": hi,
            "diff_per_seed_mean": float(np.mean(diffs)),
            "diff_per_seed_std": float(np.std(diffs)),
        }
        logger.info(
            "paired %s-S0 recall@1%%: bagged %+.4f [%+.4f, %+.4f]; per-seed %+.4f ± %+.4f",
            name, diff, lo, hi, float(np.mean(diffs)), float(np.std(diffs)),
        )
    pd.DataFrame(
        [
            {"variant": name, **row}
            for name, rows in per_seed_paired.items()
            for row in rows
        ]
        + [
            {"variant": name, "seed": "bagged", **{k: v for k, v in vals.items()}}
            for name, vals in paired.items()
        ]
    ).to_csv(results_dir() / f"exp_09_paired_{args.profile}_{holdout}.csv", index=False)

    # single-system CI on the headline metric
    ci_rows = {}
    for name, arr in [("S0", s0_arr), ("S1", np.mean(np.vstack(s1_scores), axis=0)),
                      ("S2", anom_all_va), ("S4", None)]:
        if arr is None:
            continue
        _, lo, hi = cluster_bootstrap(
            y_va_struct, arr, groups_va,
            partial(_recall_at_budget, budget=HEADLINE_BUDGET),
            n=cfg.simulation.bootstrap.n, seed=cfg.simulation.seed,
        )
        ci_rows[name] = {"lo": lo, "hi": hi}

    # --- per-scenario IF-alone ROC/PR-AUC ------------------------------------
    per_rows = []
    for sc in cfg.simulation.fraud_scenarios:
        mask = (scenario[va] == sc) | (y_va == 0)
        yy = (scenario[va][mask] == sc).astype(int)
        if yy.sum() == 0:
            continue
        per_rows.append({
            "scenario": sc,
            "n_pos": int(yy.sum()),
            "roc_auc": roc_auc(yy, anom_all_va[mask]),
            "pr_auc": pr_auc(yy, anom_all_va[mask]),
            "is_holdout": sc == holdout,
        })
    per_scenario = pd.DataFrame(per_rows)
    per_scenario.to_csv(
        results_dir() / f"exp_09_if_perscenario_{args.profile}_{holdout}.csv", index=False
    )

    # --- adoption decision ---------------------------------------------------
    # Adopt only if the effect is *robust and genuinely unseen*: both the realistic
    # (IF-all) and the strict (IF-strict) variants must show a supported benefit and
    # the per-seed effect must be meaningful. A win that needs IF-all to have seen
    # the pattern (unlabeled) is not an unseen-fraud win.
    metric = f"holdout_recall_{HEADLINE_BUDGET:g}"
    s0_hold = float(results[results["system"] == "S0"][metric].mean())
    s4_hold = float(results[results["system"] == "S4"][metric].mean())
    s1_ok = paired["S1"]["lo_bagged"] > 0.0
    s1s_ok = paired["S1s"]["lo_bagged"] > 0.0
    robust = paired["S1"]["diff_per_seed_mean"] >= 0.02
    adopted = bool(s1_ok and s1s_ok and robust)
    reason = (
        "both IF-all and IF-strict variants improve held-out recall (bagged CI > 0) "
        "with a meaningful per-seed gain"
        if adopted else
        "not adopted: "
        + ("IF-strict shows no benefit (gain requires the detector to have seen the "
           "pattern unlabeled); " if not s1s_ok else "")
        + ("per-seed S1 gain < 0.02 (driven by seed bagging); " if not robust else "")
        + ("IF-all bagged CI does not exclude 0" if not s1_ok else "")
    ).strip().rstrip(";")
    decision = {
        "profile": args.profile,
        "holdout": holdout,
        "s0_holdout_recall_1pct": round(s0_hold, 4),
        "s4_holdout_recall_1pct": round(s4_hold, 4),
        "paired": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in paired.items()},
        "per_seed_paired": {
            k: [{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                 for kk, vv in row.items()} for row in rows]
            for k, rows in per_seed_paired.items()
        },
        "ci": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in ci_rows.items()},
        "adopted": adopted,
        "reason": reason,
    }
    (results_dir() / f"exp_09_unseen_decision_{args.profile}_{holdout}.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    logger.info("adoption decision: %s (%s)", adopted, decision["reason"])

    # --- figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for system in results["system"].unique():
        sub = results[results["system"] == system]
        ys = [sub[f"holdout_recall_{b:g}"].mean() for b in BUDGETS]
        ax.plot(BUDGETS, ys, marker="o", label=system)
    ax.set_xscale("log")
    ax.set_xlabel("Alert budget (fraction of transactions, top-K)")
    ax.set_ylabel(f"{holdout} recall")
    ax.set_title(f"Held-out '{holdout}' recall vs alert budget ({args.profile})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_09_recall_{args.profile}_{holdout}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    legit = (~is_hold[va]) & (y_va == 0)
    other = (~is_hold[va]) & (y_va == 1)
    hold = is_hold[va]
    bins = np.linspace(0, 1, 30)
    ax.hist(anom_all_va[legit], bins=bins, density=True, alpha=0.5, label="legit")
    ax.hist(anom_all_va[other], bins=bins, density=True, alpha=0.6, label="other fraud (seen)")
    ax.hist(anom_all_va[hold], bins=bins, density=True, alpha=0.7, label=f"{holdout} (held out)")
    ax.set_xlabel("IF-all anomaly score (rank-normalised)")
    ax.set_ylabel("density")
    ax.set_title(f"Anomaly-score distribution by class ({args.profile})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir() / f"exp_09_anomaly_dist_{args.profile}_{holdout}.png", dpi=120)
    plt.close(fig)

    # --- experiment log ------------------------------------------------------
    # Idempotent: drop any prior rows for this experiment/holdout so re-running
    # the script does not append duplicate log entries.
    exp_id = "E15" if holdout == "structuring" else "E16"
    notes = f"unseen-fraud holdout={holdout}; cross-fitted IF; paired S1-S0 CI"
    _drop_prior_log_rows(exp_id, holdout)
    for system in results["system"].unique():
        sub = results[results["system"] == system]
        log_experiment(
            {
                "experiment_id": exp_id,
                "part": "09",
                "dataset_profile": args.profile,
                "model": f"unseen:{system}",
                "features_version": "features_v1",
                "hyperparameters": {
                    "if_params": best_if,
                    "xgb": params,
                    "system": system,
                },
                "calibration": "none",
                "threshold_policy": "rank_at_budget",
                "metrics": {
                    "holdout_recall_1pct_mean": round(
                        float(sub[f"holdout_recall_{HEADLINE_BUDGET:g}"].mean()), 6
                    ),
                    "pr_auc_overall_mean": round(float(sub["pr_auc_overall"].mean()), 6),
                },
                "notes": notes,
            },
            config_hash=chash,
        )

    if adopted:
        import joblib

        version = f"iforest_{args.profile}_{holdout}_v1"
        adir = artifacts_dir(version)
        joblib.dump(full_all, adir / "iforest.joblib")
        (adir / "metadata.json").write_text(
            json.dumps(
                {
                    "model_version": version,
                    "component": "iforest_anomaly",
                    "feature_version": "features_v1",
                    "profile": args.profile,
                    "holdout": holdout,
                    "params": best_if,
                    "seed": cfg.simulation.seed,
                    "features": list(full_all.features),
                    "adoption": decision,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("adopted: saved %s", version)

    logger.info("wrote Part 09 outputs for holdout=%s", holdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
