"""IEEE-CIS external validation: R1–R5 and the sim-to-real gap (Part 11).

Logic lives here (not in ``scripts/``) so training/evaluation imports stay in the
package. The IEEE **test** metrics are all computed in one guarded block against
``results/test_access_log_external.csv`` — separate from the FraudLens
``TestSetGuard`` log, because the synthetic test window stays locked until
Part 12. The R2 transfer model is written to ``artifacts/external/`` only.
"""

from __future__ import annotations

import contextlib
import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraudlens.config import Config, config_hash
from fraudlens.evaluation.cost_model import (
    APPROVE,
    REVIEW,
    approve_all,
    realized_cost,
    resolve_cost_params,
)
from fraudlens.evaluation.experiments import drop_experiment_rows, log_experiment
from fraudlens.evaluation.experiments_lib import best_xgb_params, fit_xgb, load_supervised
from fraudlens.evaluation.figures import plot_pr_curves, plot_reliability
from fraudlens.evaluation.guard import TestSetGuard
from fraudlens.evaluation.leakage import truncation_invariance
from fraudlens.evaluation.metrics import (
    best_f1_threshold,
    brier,
    ece,
    evaluate_scores,
    lift_over_base_rate,
    pr_auc,
)
from fraudlens.evaluation.splits import chronological_fraction_split
from fraudlens.external import ieee_cis
from fraudlens.features.registry import FEATURE_COLUMNS, FEATURE_VERSION
from fraudlens.models.calibration import calibration_bins, fit_calibrators
from fraudlens.models.io import build_metadata, save_model
from fraudlens.models.logreg import tune_logreg
from fraudlens.models.xgb import encode_features
from fraudlens.paths import figures_dir, results_dir
from fraudlens.risk.policy import amount_aware_actions, enforce_capacity
from fraudlens.utils import get_logger

logger = get_logger(__name__)

EXTERNAL_PROFILE = "ieee_cis"
R2_MODEL_VERSION = "external/ieee_cis_transfer_xgb"
_NATIVE_DROP = {"TransactionID", "isFraud"}


@dataclass
class SyntheticRefs:
    """Encoded POST-FIX dev frames for R2 (restricted to the shared subset S)."""

    profile: str
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_val: pd.DataFrame
    y_val: np.ndarray
    params: dict
    train_window: dict = field(default_factory=dict)
    val_window: dict = field(default_factory=dict)
    val_pr_auc: float = float("nan")


def load_synthetic_refs(cfg: Config) -> SyntheticRefs:
    """Build the R2 synthetic references from the POST-FIX dev feature matrix."""
    sup = load_supervised(cfg.simulation.profile)
    cols = ieee_cis.SHARED_FEATURES
    b = sup.split.boundaries
    return SyntheticRefs(
        profile=cfg.simulation.profile,
        x_train=sup.x_train()[cols],
        y_train=sup.y_train(),
        x_val=sup.x_val()[cols],
        y_val=sup.y_val(),
        params=best_xgb_params(cfg.simulation.profile),
        train_window={"start": b["t0"], "end": b["t_train_end"]},
        val_window={"start": b["t_train_end"], "end": b["t_val_end"]},
    )


def native_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """All-native numeric design matrix (identifiers/labels dropped, cats coded).

    ``float32`` throughout: the full native frame is ~590k x ~400 and R4 only
    needs an upper-bound reference, not full precision.
    """
    numeric = raw.select_dtypes(include=[np.number])
    numeric = numeric.drop(columns=[c for c in _NATIVE_DROP if c in numeric.columns])
    out = numeric.astype("float32").copy()
    for col in raw.columns:
        if raw[col].dtype.name in ("string", "object", "category"):
            out[f"{col}__code"] = (
                raw[col].astype("category").cat.codes.replace(-1, np.nan).astype("float32")
            )
    return out


@contextlib.contextmanager
def _quiet_all_nan_imputer():
    """Silence the expected "all values missing" imputer warning (external path).

    The 8 ``STRUCTURALLY_MISSING`` features are all-NaN on IEEE-CIS by design, so
    sklearn warns on every LogReg fit *and* transform. Applied to the whole
    ``run`` (both warn); the warning is expected, not a data problem.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values",
            category=UserWarning,
        )
        yield


def truncation_check(
    cfg: Config, tx: pd.DataFrame, *, sample_n: int = 50_000, n_cuts: int = 3, seed: int = 0
) -> dict:
    """Run the point-in-time truncation check on a prefix sample of IEEE rows.

    Uses the adapter's canonical ``tx`` + synthesised entities and the real
    ``build_features`` (via ``evaluation.leakage``); the isolated RNG keeps the
    shared global seed untouched so R1-R3 results are unaffected.
    """
    sample = tx.sort_values(["timestamp", "transaction_id"], kind="stable").head(sample_n)
    entities = ieee_cis.adapt_entities(sample)
    ts = pd.to_datetime(sample["timestamp"]).sort_values()
    quantiles = np.linspace(0.5, 0.85, n_cuts)
    cuts = [ts.iloc[min(len(ts) - 1, int(q * len(ts)))] for q in quantiles]
    rows = []
    for i, cut in enumerate(cuts):
        res = truncation_invariance(sample, entities, cfg, cut)
        rows.append({"cut_index": i, "cut": str(cut), **res})
    passed = all(r["passed"] for r in rows)
    max_diff = max(r["max_abs_diff"] for r in rows)
    return {"n_sample": int(len(sample)), "rows": rows, "passed": bool(passed),
            "max_abs_diff": float(max_diff)}


def best_cost_threshold(y, score, amount, params, n_grid: int = 200) -> float:
    """Threshold minimizing realized cost on ``(y, score, amount)`` (APPROVE vs REVIEW)."""
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.9995, n_grid)))
    best_t, best_c = 0.0, np.inf
    for t in grid:
        actions = np.where(np.asarray(score) >= t, REVIEW, APPROVE)
        cost = float(realized_cost(actions, y, amount, params).sum())
        if cost < best_c:
            best_t, best_c = float(t), cost
    return best_t


def _metric_row(name: str, y, score, ks, precision_targets) -> dict:
    m = evaluate_scores(y, score, ks, precision_targets)
    return {
        "model": name,
        "n": int(len(y)),
        "n_fraud": int(np.asarray(y).sum()),
        "prevalence": float(np.asarray(y).mean()),
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "brier": m["brier"],
        "ece": m["ece"],
        "best_f1": m["best_f1"],
        "lift_at_top_0.01": lift_over_base_rate(y, score, 0.01),
        "recall_at_top_0.01": m["recall_at_top_0.01"],
    }


def _action_counts(actions) -> dict:
    """Action-code counts in APPROVE/STEP_UP/REVIEW/BLOCK order."""
    c = np.bincount(np.asarray(actions, dtype=int), minlength=4)
    return {
        "n_approve": int(c[0]),
        "n_stepup": int(c[1]),
        "n_review": int(c[2]),
        "n_block": int(c[3]),
    }


def _log_external(
    experiment_id: str,
    model: str,
    ds: ieee_cis.ExternalDataset,
    *,
    metrics: dict,
    calibration: str,
    threshold_policy: str,
    hyperparameters: dict | None = None,
    notes: str = "",
    chash: str | None = None,
) -> None:
    b = ds.split.boundaries
    drop_experiment_rows(experiment_id, EXTERNAL_PROFILE, dataset_version=EXTERNAL_PROFILE)
    log_experiment(
        {
            "experiment_id": experiment_id,
            "part": "11",
            "dataset_profile": EXTERNAL_PROFILE,
            "dataset_version": EXTERNAL_PROFILE,
            "model": model,
            "features_version": FEATURE_VERSION,
            "train_window": {"start": b["t0"], "end": b["t_train_end"]},
            "val_window": {"start": b["t_train_end"], "end": b["t_val_end"]},
            "test_window": {"start": b["t_val_end"], "end": b["t_end"]},
            "hyperparameters": hyperparameters or {},
            "calibration": calibration,
            "threshold_policy": threshold_policy,
            "metrics": metrics,
            "notes": notes,
        },
        config_hash=chash,
    )


@_quiet_all_nan_imputer()
def run(
    cfg: Config,
    ds: ieee_cis.ExternalDataset,
    synthetic: SyntheticRefs | None = None,
    native_raw: pd.DataFrame | None = None,
    *,
    out_dir=None,
    figures_out=None,
    log_experiments: bool = True,
    save_artifacts: bool = True,
) -> dict:
    """Execute R1–R5 and write results/figures. Returns a summary dict.

    ``out_dir``/``figures_out``/``log_experiments``/``save_artifacts`` are seams
    for tests (write to a temp dir, skip the shared experiment log and artifacts).
    """
    if synthetic is None:
        synthetic = load_synthetic_refs(cfg)
    ext = cfg.simulation.external
    chash = config_hash(cfg)
    seed = cfg.simulation.seed
    ks = cfg.simulation.ks
    pt = cfg.simulation.precision_targets
    out = out_dir if out_dir is not None else results_dir()

    g = ds.df
    y = g["is_fraud"].to_numpy(int)
    tr, va, te = ds.split.train_idx, ds.split.val_idx, ds.split.test_idx
    x_raw = g[FEATURE_COLUMNS]
    x_enc = encode_features(g)
    r1_cols = [f for f in FEATURE_COLUMNS if f not in ieee_cis.STRUCTURALLY_MISSING]
    s_cols = ieee_cis.SHARED_FEATURES

    # ---- R1: real-data baseline on computable features ----------------------
    logreg_r1, _ = tune_logreg(
        x_raw.iloc[tr], y[tr], x_raw.iloc[va], y[va],
        cfg.simulation.models.logreg.C_grid, cfg.simulation.models.logreg.class_weights,
    )
    xgb_r1 = fit_xgb(
        x_enc[r1_cols].iloc[tr], y[tr], x_enc[r1_cols].iloc[va], y[va],
        best_xgb_params(cfg.simulation.profile), seed,
    )
    scores_val = {
        "R1_logreg": logreg_r1.predict_proba(x_raw.iloc[va])[:, 1],
        "R1_xgb": xgb_r1.predict_proba(x_enc[r1_cols].iloc[va])[:, 1],
    }
    rows = [
        _metric_row(name, y[va], s, ks, pt) | {"window": "val"}
        for name, s in scores_val.items()
    ]

    # ---- R2: sim-to-real transfer (dev xgb on S) ----------------------------
    model_r2 = fit_xgb(
        synthetic.x_train, synthetic.y_train, synthetic.x_val, synthetic.y_val,
        synthetic.params, seed,
    )
    dev_val_pr = float(pr_auc(synthetic.y_val, model_r2.predict_proba(synthetic.x_val)[:, 1]))
    r2_val = model_r2.predict_proba(x_enc[s_cols].iloc[va])[:, 1]
    rows.append(_metric_row("R2_transfer_dev_xgb", y[va], r2_val, ks, pt) | {"window": "val"})
    metadata = build_metadata(
        model_version=R2_MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        profile=synthetic.profile,
        train_window=synthetic.train_window or {},
        val_window=synthetic.val_window or {},
        calibration="none",
        hyperparameters=synthetic.params,
        seed=seed,
        metrics={"dev_val_pr_auc_shared_subset": dev_val_pr},
        config_hash=chash,
    )
    if save_artifacts:
        save_model(model_r2, R2_MODEL_VERSION, metadata)

    # ---- R3: sigmoid recalibration on an IEEE val slice ---------------------
    cal_pos, thr_pos = chronological_fraction_split(
        np.arange(len(va)), ext.calibration.slice_fraction
    )
    calibrator = fit_calibrators(r2_val[cal_pos], y[va][cal_pos], names=("sigmoid",))["sigmoid"]
    p_r2_thr = calibrator.predict(r2_val[thr_pos])
    y_thr = y[va][thr_pos]
    r3_table = [
        {"stage": "val_thr_raw", "brier": float(brier(y_thr, r2_val[thr_pos])),
         "ece": float(ece(y_thr, r2_val[thr_pos]))},
        {"stage": "val_thr_sigmoid", "brier": float(brier(y_thr, p_r2_thr)),
         "ece": float(ece(y_thr, p_r2_thr))},
    ]
    rows.append(_metric_row("R3_transfer_sigmoid", y_thr, p_r2_thr, ks, pt) | {"window": "val_thr"})

    # ---- R4: native reference (upper bound) ---------------------------------
    native = native_matrix(native_raw if native_raw is not None else ds.raw)
    model_r4 = fit_xgb(
        native.iloc[tr], y[tr], native.iloc[va], y[va],
        best_xgb_params(cfg.simulation.profile), seed,
    )
    r4_val = model_r4.predict_proba(native.iloc[va])[:, 1]
    rows.append(_metric_row("R4_native_upper_bound", y[va], r4_val, ks, pt) | {"window": "val"})

    # ---- R5: thresholds chosen on IEEE val only -----------------------------
    params = resolve_cost_params(ext.costs, ext.cost_scenarios.get("balanced"))
    amount = g["amount"].to_numpy(float)
    f1_thr, _ = best_f1_threshold(y[va], scores_val["R1_xgb"])
    cost_thr = best_cost_threshold(y[va], scores_val["R1_xgb"], amount[va], params)
    # The amount-aware expected-cost policy is valid only for *calibrated* p
    # (Part 07 framework). R1_xgb is class-weighted, so fit a sigmoid on the
    # earlier IEEE val slice (never test) and use it for the per-row decisions.
    calibrator_r1 = fit_calibrators(
        scores_val["R1_xgb"][cal_pos], y[va][cal_pos], names=("sigmoid",)
    )["sigmoid"]
    p_r1_thr = calibrator_r1.predict(scores_val["R1_xgb"][thr_pos])
    r3_table += [
        {"stage": "R1_val_thr_raw", "brier": float(brier(y_thr, scores_val["R1_xgb"][thr_pos])),
         "ece": float(ece(y_thr, scores_val["R1_xgb"][thr_pos]))},
        {"stage": "R1_val_thr_sigmoid", "brier": float(brier(y_thr, p_r1_thr)),
         "ece": float(ece(y_thr, p_r1_thr))},
    ]
    val_days = max(1, int(pd.to_datetime(g.iloc[va]["timestamp"]).dt.normalize().nunique()))
    capacity = max(1, int(round(ext.costs.capacity_pct_of_daily_volume * len(va) / val_days)))

    # ---- one guarded IEEE-test evaluation (all test metrics) ----------------
    guard = TestSetGuard(out / "test_access_log_external.csv")
    guard.access(
        "external:ieee_cis:R1-R5",
        model_version=R2_MODEL_VERSION,
        final=True,
        reason="post-fix re-run: R1 sigmoid calibration for amount-aware R5 + full-native R4",
    )
    y_test = y[te]
    test_amount = amount[te]
    test_dates = pd.to_datetime(g.iloc[te]["timestamp"]).dt.normalize().to_numpy()
    test_scores = {
        "R1_logreg": logreg_r1.predict_proba(x_raw.iloc[te])[:, 1],
        "R1_xgb": xgb_r1.predict_proba(x_enc[r1_cols].iloc[te])[:, 1],
        "R2_transfer_dev_xgb": model_r2.predict_proba(x_enc[s_cols].iloc[te])[:, 1],
        "R3_transfer_sigmoid": calibrator.predict(
            model_r2.predict_proba(x_enc[s_cols].iloc[te])[:, 1]
        ),
        "R4_native_upper_bound": model_r4.predict_proba(native.iloc[te])[:, 1],
    }
    test_rows = [
        _metric_row(name, y_test, s, ks, pt) | {"window": "test"}
        for name, s in test_scores.items()
    ]
    p_r1_test = calibrator_r1.predict(test_scores["R1_xgb"])

    # ---- gap table (primary = R1 - R2 on IEEE test; no cross-dataset lift) --
    r1_test = next(r for r in test_rows if r["model"] == "R1_xgb")
    r2_test = next(r for r in test_rows if r["model"] == "R2_transfer_dev_xgb")
    gap = {
        "primary_gap_pr_auc_r1_minus_r2_on_ieee_test": r1_test["pr_auc"] - r2_test["pr_auc"],
        "secondary_gap_dev_val_minus_ieee_test": dev_val_pr - r2_test["pr_auc"],
        "secondary_note": "selection-optimistic: dev-val is the R2 model's own selection window",
        "r1_ieee_test_pr_auc": r1_test["pr_auc"],
        "r2_ieee_test_pr_auc": r2_test["pr_auc"],
        "r2_dev_val_shared_subset_pr_auc": dev_val_pr,
        "ieee_test_prevalence": r1_test["prevalence"],
        "note": "lift is reported per dataset only; it is never differenced across datasets",
    }

    # ---- R5 realized-cost comparison on IEEE test ---------------------------
    f1_actions = np.where(test_scores["R1_xgb"] >= f1_thr, REVIEW, APPROVE)
    cost_actions = np.where(test_scores["R1_xgb"] >= cost_thr, REVIEW, APPROVE)
    aware = amount_aware_actions(p_r1_test, test_amount, params)
    capped = enforce_capacity(test_dates, p_r1_test, test_amount, aware, params, capacity)
    policy_rows = []
    for name, acts, demoted in [
        ("approve_all", approve_all(len(te)), 0),
        ("f1_optimal", f1_actions, 0),
        ("cost_optimal", cost_actions, 0),
        ("amount_aware", aware, 0),
        ("capacity_aware", capped.actions, capped.n_demoted),
    ]:
        total = float(realized_cost(acts, y_test, test_amount, params).sum())
        policy_rows.append(
            {
                "policy": name,
                "realized_cost": total,
                **_action_counts(acts),
                "n_demoted": int(demoted),
            }
        )
    base_cost = policy_rows[0]["realized_cost"]
    for row in policy_rows:
        row["pct_saving_vs_approve_all"] = (
            (base_cost - row["realized_cost"]) / base_cost * 100 if base_cost else float("nan")
        )
    policy = pd.DataFrame(policy_rows)

    # ---- point-in-time leakage check on IEEE rows (Part 11B item 6) ---------
    trunc = truncation_check(cfg, ds.tx, seed=seed)

    # ---- write outputs ------------------------------------------------------
    table = pd.DataFrame(rows + test_rows)
    table.to_csv(out / "external_validation.csv", index=False)
    pd.DataFrame([gap]).to_csv(out / "external_gap.csv", index=False)
    pd.DataFrame(r3_table).to_csv(out / "external_calibration.csv", index=False)
    policy.to_csv(out / "external_policy.csv", index=False)
    pd.DataFrame(trunc["rows"]).to_csv(out / "external_truncation_check.csv", index=False)
    pd.DataFrame(ieee_cis.availability_rows()).to_csv(
        out / "external_feature_availability.csv", index=False
    )
    nan_share = g[FEATURE_COLUMNS].isna().mean()
    pd.DataFrame({"feature": nan_share.index, "nan_share": nan_share.to_numpy()}).to_csv(
        out / "external_feature_nan_share.csv", index=False
    )
    device_cols = [
        "is_new_device",
        "device_age_days",
        "tx_from_device_prior",
        "accounts_seen_on_device",
        "device_count_30d",
    ]
    logger.info(
        "device-feature NaN shares: %s",
        {c: round(float(nan_share[c]), 3) for c in device_cols},
    )
    (out / f"external_uid_stats_{EXTERNAL_PROFILE}.json").write_text(
        json.dumps(ds.uid_stats, indent=2), encoding="utf-8"
    )
    if figures_out is not None or out_dir is None:
        _write_figures(
            y_test, test_scores, y_thr, r2_val[thr_pos], p_r2_thr,
            figures_out if figures_out is not None else figures_dir(),
        )

    # ---- experiment log -----------------------------------------------------
    if not log_experiments:
        logger.info("gap (primary R1-R2 on IEEE test): %.4f",
                    gap["primary_gap_pr_auc_r1_minus_r2_on_ieee_test"])
        return {
            "gap": gap,
            "policy": policy_rows,
            "test": test_rows,
            "uid_stats": ds.uid_stats,
            "truncation": trunc,
        }
    _log_external("E20", "R1 logreg+xgb (shared pipeline)", ds,
                  metrics={"logreg_val_pr_auc": float(pr_auc(y[va], scores_val["R1_logreg"])),
                           "xgb_val_pr_auc": float(pr_auc(y[va], scores_val["R1_xgb"])),
                           "xgb_test_pr_auc": r1_test["pr_auc"]},
                  calibration="none", threshold_policy="rank_by_score",
                  notes="real-data baseline on computable features; IEEE val+test", chash=chash)
    _log_external("E21", "R2 sim-to-real transfer xgb", ds,
                  metrics={"ieee_test_pr_auc": r2_test["pr_auc"],
                           "dev_val_shared_pr_auc": dev_val_pr,
                           "primary_gap": gap["primary_gap_pr_auc_r1_minus_r2_on_ieee_test"]},
                  calibration="none", threshold_policy="rank_by_score",
                  hyperparameters=synthetic.params,
                  notes=f"retrained on POST-FIX {synthetic.profile} over subset S "
                        f"(|S|={len(s_cols)})",
                  chash=chash)
    _log_external("E22", "R3 transfer + sigmoid recalibration", ds,
                  metrics={"brier_before": r3_table[0]["brier"],
                           "brier_after": r3_table[1]["brier"],
                           "ece_before": r3_table[0]["ece"], "ece_after": r3_table[1]["ece"]},
                  calibration="sigmoid", threshold_policy="rank_by_score",
                  notes="sigmoid fit on earlier IEEE val slice; ranking unchanged", chash=chash)
    _log_external("E23", "R4 native xgb (upper bound)", ds,
                  metrics={"ieee_val_pr_auc": float(pr_auc(y[va], r4_val)),
                           "ieee_test_pr_auc": next(
                               r for r in test_rows if r["model"] == "R4_native_upper_bound"
                           )["pr_auc"]},
                  calibration="none", threshold_policy="rank_by_score",
                  notes="all native IEEE columns (V/C/D/M + identity) outside the "
                        "FraudLens schema; reference only", chash=chash)
    _log_external("E24", "R5 cost policy on R1 xgb", ds,
                  metrics={"f1_threshold": f1_thr, "cost_threshold": cost_thr,
                           "realized_cost_f1": float(policy.set_index("policy").loc[
                               "f1_optimal", "realized_cost"]),
                           "realized_cost_cost_optimal": float(policy.set_index("policy").loc[
                               "cost_optimal", "realized_cost"]),
                           "realized_cost_approve_all": float(policy.set_index("policy").loc[
                               "approve_all", "realized_cost"]),
                           "realized_cost_capacity_aware": float(policy.set_index("policy").loc[
                               "capacity_aware", "realized_cost"]),
                           "n_demoted": int(policy.set_index("policy").loc[
                               "capacity_aware", "n_demoted"])},
                  calibration="sigmoid (R1, earlier IEEE val slice)",
                  threshold_policy="cost_based_capacity_aware",
                  notes=f"amount-aware uses R1 sigmoid-calibrated p; capacity={capacity}/day "
                        f"from IEEE val; F1/cost thresholds on raw val scores",
                  chash=chash)

    logger.info("gap (primary R1-R2 on IEEE test): %.4f",
                gap["primary_gap_pr_auc_r1_minus_r2_on_ieee_test"])
    logger.info("external test PR-AUC: %s",
                {k: round(float(pr_auc(y_test, v)), 4) for k, v in test_scores.items()})
    logger.info("R5 realized cost:\n%s", policy.to_string(index=False))
    logger.info("truncation check: passed=%s max_abs_diff=%.3g over %d rows",
                trunc["passed"], trunc["max_abs_diff"], trunc["n_sample"])
    return {
        "gap": gap,
        "policy": policy_rows,
        "test": test_rows,
        "uid_stats": ds.uid_stats,
        "truncation": trunc,
    }


def _write_figures(y_test, test_scores, y_thr, raw_thr, sig_thr, figs) -> None:
    """PR curves (real vs transfer), IEEE recalibration, feature availability."""
    try:
        plot_pr_curves(
            y_test,
            {k: v for k, v in test_scores.items() if k in ("R1_xgb", "R2_transfer_dev_xgb")},
            figs / "external_pr_curves.png",
        )
        plot_reliability(
            {
                "transfer_raw": (float(np.mean(y_thr)), calibration_bins(y_thr, raw_thr, 10)),
                "transfer_sigmoid": (float(np.mean(y_thr)), calibration_bins(y_thr, sig_thr, 10)),
            },
            figs / "external_calibration.png",
        )
        _plot_availability(figs / "external_feature_availability.png")
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("figure generation failed: %s", exc)


def _plot_availability(path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ieee_cis.availability_rows()
    colors = {"available": "#2ca02c", "excluded_from_transfer": "#ff7f0e", "missing": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(rows)), [1] * len(rows), color=[colors[r["status"]] for r in rows])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["feature"] for r in rows], rotation=90, fontsize=6)
    ax.set_yticks([])
    ax.set_title("IEEE-CIS availability (green=available, orange=transfer-excluded, red=missing)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
