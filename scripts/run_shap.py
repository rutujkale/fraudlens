"""Part 10A/C/D: global SHAP analysis, figures and the explanation gallery.

Explains the **production** XGBoost artifact on validation data only. Calibrated
probabilities come from the saved sigmoid calibrator; actions come from the frozen
amount-aware policy in ``policy.json`` (capacity from that file). The test window is
never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from fraudlens.config import config_hash, load_config
from fraudlens.evaluation.cost_model import ACTION_NAMES, resolve_cost_params
from fraudlens.evaluation.experiments import drop_experiment_rows, log_experiment
from fraudlens.evaluation.experiments_lib import load_supervised
from fraudlens.evaluation.scores import load_model_versions
from fraudlens.evaluation.splits import chronological_fraction_split
from fraudlens.explain import plots
from fraudlens.explain.gallery import select_cases
from fraudlens.explain.reasons import reasons_for
from fraudlens.explain.report import render_gallery_markdown
from fraudlens.explain.shap_explainer import ShapExplainer, stratified_val_sample
from fraudlens.features.registry import FEATURE_VERSION
from fraudlens.models import io as model_io
from fraudlens.paths import docs_dir, figures_dir, raw_data_dir, results_dir
from fraudlens.risk.policy import amount_aware_actions, enforce_capacity
from fraudlens.utils import get_logger, set_global_seed

logger = get_logger(__name__)

SAMPLE_FRAUDS = 5000
SAMPLE_LEGIT = 15000
INTERACTION_ROWS = 500
LATENCY_ROWS = 200
LATENCY_WARMUP = 20
TOP_SCENARIO_FEATURES = 10
CONTROL_PREFIX = "_"


def _load_artifacts(profile: str):
    versions = load_model_versions(profile)
    bundle = model_io.load_serving_bundle(versions["xgb"])
    return bundle["model"], bundle["calibrator"], bundle["policy"], versions["xgb"]


def _policy_actions(p: np.ndarray, amount: np.ndarray, dates: np.ndarray, params, capacity: int):
    actions = amount_aware_actions(p, amount, params)
    return enforce_capacity(dates, p, amount, actions, params, capacity)


def _interaction_partners(
    explainer: ShapExplainer, x_sample: pd.DataFrame, features: list[str], seed: int
) -> dict[str, str]:
    """Strongest interaction partner per feature, from a small TreeSHAP interaction run."""
    rng = np.random.default_rng(seed)
    take = min(INTERACTION_ROWS, len(x_sample))
    idx = np.sort(rng.choice(len(x_sample), size=take, replace=False))
    try:
        interactions = explainer.interaction_values(x_sample.iloc[idx])
    except Exception as exc:  # pragma: no cover - shap/version dependent
        logger.warning("interaction values unavailable (%s); falling back to correlation", exc)
        return {}
    strength = np.abs(interactions).mean(axis=0)
    partners = {}
    for feature in features:
        j = explainer.feature_names.index(feature)
        row = strength[j].copy()
        row[j] = -np.inf
        partners[feature] = explainer.feature_names[int(np.argmax(row))]
    return partners


def main() -> int:
    parser = argparse.ArgumentParser(description="Part 10 SHAP analysis + gallery.")
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    sup = load_supervised(args.profile)
    model, calibrator, policy, version = _load_artifacts(args.profile)
    params = resolve_cost_params(
        cfg.simulation.costs, cfg.simulation.cost_scenarios[policy["scenario"]]
    )
    capacity = int(policy["capacity_per_day"])

    x_val = sup.x_val()
    y_val = sup.y_val()
    val = sup.df.iloc[sup.split.val_idx].reset_index(drop=True)
    labels = pd.read_parquet(raw_data_dir(args.profile) / "labels.parquet")
    val = val.merge(labels[["transaction_id", "stealth"]], on="transaction_id", how="left")
    raw = model.predict_proba(x_val)[:, 1]
    p_all = calibrator.predict(raw)
    n_val = len(val)
    _, thr_pos = chronological_fraction_split(
        np.arange(n_val), cfg.simulation.calibration.cal_fraction
    )
    logger.info(
        "profile=%s model=%s val=%d val_thr=%d frauds=%d capacity/day=%d",
        args.profile, version, n_val, len(thr_pos), int(y_val[thr_pos].sum()), capacity,
    )

    dates_thr = val.iloc[thr_pos]["timestamp"].dt.date.to_numpy()
    capped = _policy_actions(
        p_all[thr_pos], val.iloc[thr_pos]["amount"].to_numpy(float), dates_thr, params, capacity
    )
    thr = pd.DataFrame(
        {
            "transaction_id": val.iloc[thr_pos]["transaction_id"].to_numpy(),
            "customer_id": val.iloc[thr_pos]["customer_id"].to_numpy(),
            "timestamp": val.iloc[thr_pos]["timestamp"].to_numpy(),
            "is_fraud": y_val[thr_pos],
            "p": p_all[thr_pos],
            "action": capped.actions,
            "amount": val.iloc[thr_pos]["amount"].to_numpy(float),
            "stealth": val.iloc[thr_pos]["stealth"].to_numpy(float),
            "fraud_scenario": val.iloc[thr_pos]["fraud_scenario"].to_numpy(),
            "travel_speed_kmh": val.iloc[thr_pos]["travel_speed_kmh"].to_numpy(float),
            "history_days": val.iloc[thr_pos]["history_days"].to_numpy(float),
            "n_prior_tx": val.iloc[thr_pos]["n_prior_tx"].to_numpy(float),
        }
    ).reset_index(drop=True)
    logger.info("val_thr actions: %s", np.bincount(capped.actions, minlength=4).tolist())

    explainer = ShapExplainer(model)

    # --- 10A/10C global analysis -------------------------------------------
    sample = stratified_val_sample(y_val, SAMPLE_FRAUDS, SAMPLE_LEGIT, cfg.simulation.seed)
    x_sample = x_val.iloc[sample]
    shap_sample = explainer.shap_values(x_sample)
    global_df = explainer.explain_global(x_sample)
    logger.info("global SHAP on %d rows (%.0f frauds)", len(sample), y_val[sample].sum())

    eight_d_path = results_dir() / f"exp_8D_importance_{args.profile}.csv"
    if eight_d_path.exists():
        eight_d = pd.read_csv(eight_d_path)
        eight_d = eight_d[~eight_d["feature"].str.startswith(CONTROL_PREFIX)]
        global_df = global_df.merge(
            eight_d[["feature", "perm_pr_auc_drop", "gain_share", "mean_abs_shap"]].rename(
                columns={"mean_abs_shap": "mean_abs_shap_8d"}
            ),
            on="feature",
            how="left",
        )
    global_df.to_csv(results_dir() / f"shap_global_{args.profile}.csv", index=False)
    top10 = global_df.head(10)
    logger.info(
        "top-10 by mean|SHAP|:\n%s", top10[["feature", "mean_abs_shap"]].to_string(index=False)
    )

    # top features per fraud scenario (labels used for analysis only)
    scenarios = sup.df["fraud_scenario"].to_numpy()[sup.split.val_idx][sample]
    scen_rows = []
    for scenario in sorted(pd.Series(scenarios).dropna().unique()):
        mask = scenarios == scenario
        for feature in global_df.head(TOP_SCENARIO_FEATURES)["feature"]:
            j = explainer.feature_names.index(feature)
            scen_rows.append(
                {
                    "fraud_scenario": scenario,
                    "feature": feature,
                    "n_rows": int(mask.sum()),
                    "mean_shap": float(shap_sample[mask, j].mean()),
                }
            )
    pd.DataFrame(scen_rows).to_csv(
        results_dir() / f"shap_scenario_{args.profile}.csv", index=False
    )

    plots.plot_shap_bar(global_df, figures_dir() / f"shap_global_bar_{args.profile}.png")
    plots.plot_shap_beeswarm(
        shap_sample, x_sample, figures_dir() / f"shap_beeswarm_{args.profile}.png"
    )

    # --- 10C dependence plots: top-3 by permutation utility (8D) ------------
    if eight_d_path.exists():
        dep_features = (
            pd.read_csv(eight_d_path)
            .query("~feature.str.startswith(@CONTROL_PREFIX)", engine="python")
            .sort_values("perm_pr_auc_drop", ascending=False)["feature"]
            .head(3)
            .tolist()
        )
    else:
        dep_features = global_df.head(3)["feature"].tolist()
    partners = _interaction_partners(explainer, x_sample, dep_features, cfg.simulation.seed)
    dep_meta = []
    for feature in dep_features:
        partner = partners.get(feature, "")
        if partner:
            color_feature = partner
        else:
            others = [f for f in global_df["feature"].head(5) if f != feature]
            color_feature = others[0] if others else ""
        values = x_sample[feature].dropna().unique()
        binary = set(np.unique(values).tolist()).issubset({0.0, 1.0})
        plots.plot_shap_dependence(
            shap_sample,
            x_sample,
            feature,
            color_feature,
            figures_dir() / f"shap_dependence_{feature}_{args.profile}.png",
            binary=binary,
        )
        dep_meta.append(
            {
                "feature": feature,
                "color_feature": color_feature,
                "binary": bool(binary),
                "perm_pr_auc_drop": float(
                    global_df.loc[global_df["feature"] == feature, "perm_pr_auc_drop"].iloc[0]
                )
                if "perm_pr_auc_drop" in global_df.columns
                else float("nan"),
            }
        )
    pd.DataFrame(dep_meta).to_csv(
        results_dir() / f"shap_dependence_{args.profile}.csv", index=False
    )
    logger.info("dependence features: %s", [(d["feature"], d["color_feature"]) for d in dep_meta])

    # --- 10D gallery --------------------------------------------------------
    cases = select_cases(thr)
    x_thr = x_val.iloc[thr_pos].reset_index(drop=True)
    gallery_rows = []
    for case in cases:
        row = thr.iloc[[case.position]]
        x_row = x_thr.iloc[[case.position]]
        contributions = explainer.explain_local(x_row)
        reasons = reasons_for(x_row.iloc[0], contributions, k=4)
        shap_row = explainer.shap_values(x_row)[0]
        figure = f"shap_waterfall_{case.case}_{args.profile}.png"
        plots.plot_waterfall(
            shap_row,
            explainer.expected_value,
            x_row,
            figures_dir() / figure,
            title=(
                f"{case.case}: {row.iloc[0]['transaction_id']} "
                f"(p={row.iloc[0]['p']:.3f}, {ACTION_NAMES[int(row.iloc[0]['action'])]})"
            ),
        )
        gallery_rows.append(
            {
                "case": case.case,
                "transaction_id": str(row.iloc[0]["transaction_id"]),
                "customer_id": str(row.iloc[0]["customer_id"]),
                "timestamp": str(row.iloc[0]["timestamp"]),
                "is_fraud": int(row.iloc[0]["is_fraud"]),
                "fraud_scenario": row.iloc[0]["fraud_scenario"],
                "stealth": round(float(row.iloc[0]["stealth"]), 4)
                if row.iloc[0]["is_fraud"]
                else float("nan"),
                "amount": round(float(row.iloc[0]["amount"]), 2),
                "p": round(float(row.iloc[0]["p"]), 6),
                "action": ACTION_NAMES[int(row.iloc[0]["action"])],
                "travel_speed_kmh": round(float(row.iloc[0]["travel_speed_kmh"]), 1),
                "history_days": round(float(row.iloc[0]["history_days"]), 2),
                "n_prior_tx": float(row.iloc[0]["n_prior_tx"]),
                "pool_size": case.pool_size,
                "rule": case.rule,
                "top_raisers": " | ".join(reasons.raises_risk),
                "top_lowers": " | ".join(reasons.lowers_risk),
                "figure": f"reports/figures/{figure}",
            }
        )
    gallery = pd.DataFrame(gallery_rows)
    gallery.to_csv(results_dir() / f"shap_gallery_{args.profile}.csv", index=False)
    logger.info("gallery pools: %s", {r["case"]: r["pool_size"] for r in gallery_rows})

    # --- latency ------------------------------------------------------------
    rows_for_latency = x_thr.iloc[[i % len(x_thr) for i in range(8)]]
    for _ in range(LATENCY_WARMUP):
        explainer.explain_local(x_thr.iloc[[0]])
    times = []
    for i in range(LATENCY_ROWS):
        x_one = rows_for_latency.iloc[[i % len(rows_for_latency)]]
        start = time.perf_counter()
        explainer.explain_local(x_one)
        times.append((time.perf_counter() - start) * 1000.0)
    times_arr = np.asarray(times)
    latency = {
        "profile": args.profile,
        "model_version": version,
        "n_calls": LATENCY_ROWS,
        "warmup_calls": LATENCY_WARMUP,
        "p50_ms": float(np.percentile(times_arr, 50)),
        "p95_ms": float(np.percentile(times_arr, 95)),
        "max_ms": float(times_arr.max()),
        "mean_ms": float(times_arr.mean()),
        "note": "warm TreeExplainer, single-row explain_local on dev validation rows",
    }
    (results_dir() / f"shap_latency_{args.profile}.json").write_text(
        json.dumps(latency, indent=2), encoding="utf-8"
    )
    logger.info("latency ms: p50=%.2f p95=%.2f", latency["p50_ms"], latency["p95_ms"])

    # --- docs ---------------------------------------------------------------
    policy_line = (
        f"amount-aware policy `{policy['policy_version']}` (scenario `{policy['scenario']}`), "
        f"calibrator `{policy['selected_calibrator']}`, capacity {capacity}/day"
    )
    doc_cases = []
    for row in gallery_rows:
        doc_cases.append(
            {
                **row,
                "top_raisers": [s for s in row["top_raisers"].split(" | ") if s],
                "top_lowers": [s for s in row["top_lowers"].split(" | ") if s],
                "commentary": _case_commentary(row, thr),
            }
        )
    (docs_dir() / "explanation_gallery.md").write_text(
        render_gallery_markdown(args.profile, doc_cases, policy_line), encoding="utf-8"
    )

    drop_experiment_rows("E17", args.profile, dataset_version=cfg.simulation.dataset_version)
    log_experiment(
        {
            "experiment_id": "E17",
            "part": "10",
            "dataset_profile": args.profile,
            "model": f"xgboost+calibration+shap ({version})",
            "features_version": FEATURE_VERSION,
            "hyperparameters": {"shap_rows": len(sample), "interaction_rows": INTERACTION_ROWS},
            "calibration": policy["selected_calibrator"],
            "threshold_policy": f"amount_aware_capacity_{capacity}",
            "metrics": {
                "top_feature": str(global_df.iloc[0]["feature"]),
                "top_feature_mean_abs_shap": round(float(global_df.iloc[0]["mean_abs_shap"]), 4),
                "additivity_max_err": round(explainer.check_additivity(x_sample.iloc[:200]), 8),
                "explain_p50_ms": round(latency["p50_ms"], 3),
                "explain_p95_ms": round(latency["p95_ms"], 3),
            },
            "notes": "SHAP on raw margin; gallery selected by pre-registered rules",
        },
        config_hash=chash,
    )
    logger.info("wrote Part 10 SHAP outputs")
    return 0


def _case_commentary(case: dict, thr: pd.DataFrame) -> str:
    """Candid one-paragraph commentary for a gallery case (numbers are live)."""
    action = case["action"]
    if case["case"] == "clear_tp":
        return (
            f"Correctly flagged fraud ({case['fraud_scenario']}) with p={case['p']:.4f}, action "
            f"{action}. The drivers below are the ones that pushed the margin up; the model "
            "did not need anything exotic to catch this one."
        )
    if case["case"] == "subtle_tp":
        return (
            f"High-stealth fraud (stealth={case['stealth']:.3f}, scenario "
            f"{case['fraud_scenario']}) still flagged, p={case['p']:.4f}. Stealth raises the "
            "chance the transaction looks ordinary in-level; the catch comes from the "
            "combination of drivers rather than a single extreme value."
        )
    if case["case"] == "false_negative":
        return (
            f"Missed fraud: p={case['p']:.4f} -> {action}. This is the highest-stealth missed "
            f"fraud in val_thr (stealth={case['stealth']:.3f}, scenario {case['fraud_scenario']}). "
            "It looks like the customer's own behaviour, and the features the model relies on "
            "do not fire; a review budget cannot catch every deliberate imitation."
        )
    if case["case"] == "hard_negative_fp":
        return (
            f"False positive on a legitimate transaction (p={case['p']:.4f} -> {action}). A "
            "travel/velocity pattern that mimics account takeover is exactly the kind of hard "
            "negative the model is expected to trade against; the cost of this review is Rs 75 "
            "of analyst time plus friction on a good customer."
        )
    if case["case"] == "high_amount_tn":
        return (
            f"Large legitimate transaction (Rs {case['amount']:,.0f}) correctly approved "
            f"(p={case['p']:.4f}). High amounts are not fraudulent by themselves; the amount "
            "features are relative to the customer's own baseline, which is what keeps this "
            "one clear."
        )
    return (
        f"Cold start by the book: the account opened {case['history_days']:.2f} days before this "
        f"transaction, p={case['p']:.4f} -> {action}. The account is brand-new, so its own "
        f"behavioural history is thin (n_prior_tx={case['n_prior_tx']:.0f}) and adds little; "
        "this case shows how the amount-aware policy treats a low-history account, not a "
        "generation defect (post Part 02b the simulator no longer emits pre-open rows)."
    )


if __name__ == "__main__":
    sys.exit(main())
