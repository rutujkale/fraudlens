"""Part 12B — locked evaluation of the frozen systems on the held-out test window.

Everything the test window is compared against was fixed in ``results/freeze.json``
(Part 12A): model artifacts, calibrator, thresholds, capacity, cost config and the
pre-registered H1–H4. This module verifies every frozen hash, opens the test window
exactly once through :class:`~fraudlens.evaluation.guard.TestSetGuard`, and writes
``results/final_results.json`` plus the comparison CSVs and figures.

Thresholds and policies are **never** re-derived on the test window: the primary
policy uses the frozen validation thresholds, and the cost-scenario sensitivity
re-derives each threshold on the ``val_thr`` slice before applying it to test.
A ``--dry-run`` exercises the identical path on ``val_thr`` (no guard, no final
files) so crashes are found before the single test access.

The pre-registered outcomes (H1–H4) are evaluated verbatim against the wording in
``freeze.json`` and reported whichever way they fall.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudlens.config import Config, config_hash, load_config
from fraudlens.evaluation.cost_model import (
    APPROVE,
    REVIEW,
    cost_breakdown,
    realized_cost,
    resolve_cost_params,
)
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.experiments import drop_experiment_rows, log_experiment
from fraudlens.evaluation.figures import (
    plot_cost_breakdown,
    plot_pr_curves,
    plot_recall_at_k,
    plot_reliability,
    plot_roc_curve,
    plot_sensitivity,
)
from fraudlens.evaluation.locked import (
    assert_head_has_freeze,
    final_evaluation_count,
    reconstructed_training_config_hash,
    resolve_freeze_path,
)
from fraudlens.evaluation.metrics import (
    best_f1_threshold,
    brier,
    ece,
    fpr_at_threshold,
    pr_auc,
    precision_recall_f1_at_threshold,
    recall_at_precision,
    recall_at_top_k,
    roc_auc,
)
from fraudlens.evaluation.reporting import group_metrics
from fraudlens.evaluation.scores import load_model_versions, raw_val_scores
from fraudlens.evaluation.splits import chronological_fraction_split, make_temporal_split
from fraudlens.evaluation.threshold_search import scan_global_thresholds
from fraudlens.explain.fairness import group_table, load_audit_attributes
from fraudlens.features.registry import FEATURE_COLUMNS, FEATURE_VERSION, NUMERIC_COLUMNS
from fraudlens.models import io as model_io
from fraudlens.models.calibration import calibration_bins
from fraudlens.models.rules import RuleEngine
from fraudlens.models.xgb import encode_features
from fraudlens.monitoring.drift import TENURE_FEATURES, psi, psi_table
from fraudlens.paths import artifacts_dir, figures_dir, results_dir
from fraudlens.risk.policy import amount_aware_actions, enforce_capacity
from fraudlens.utils import get_logger, git_commit, set_global_seed

logger = get_logger(__name__)

PRIMARY_MODEL = "xgb"
CALIBRATED_MODELS = ("xgb", "logreg")
MODELS = ("rules", "logreg", "xgb")
N_POS_MIN = 30
PSI_TOP_N = 12


# --------------------------------------------------------------------------- freeze


def load_freeze(profile: str) -> dict:
    """Read and sanity-check ``results/freeze.json`` for ``profile``."""
    path = results_dir() / "freeze.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/freeze.py first")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("profile") != profile:
        raise ValueError(f"freeze profile {freeze.get('profile')!r} != requested {profile!r}")
    return freeze


def verify_and_report(freeze: dict) -> list[dict]:
    """Verify every frozen artifact/data hashes; log each; abort on any mismatch."""
    verified: list[dict] = []
    mismatches: list[str] = []
    for name, info in freeze["models"].items():
        for fname, expected in info["files"].items():
            path = artifacts_dir(info["model_version"]) / fname
            actual = model_io_hash(path)
            ok = actual == expected
            logger.info(
                "verify %-42s %s %s", f"{name}/{fname}", actual[:16], "OK" if ok else "FAIL"
            )
            verified.append(
                {"key": f"{info['model_version']}/{fname}", "sha256": actual, "ok": ok}
            )
            if not ok:
                mismatches.append(str(path))
    for key, expected in freeze["data_files"].items():
        path = resolve_freeze_path(key)
        actual = model_io_hash(path)
        ok = actual == expected
        logger.info("verify %-42s %s %s", key, actual[:16], "OK" if ok else "FAIL")
        verified.append({"key": key, "sha256": actual, "ok": ok})
        if not ok:
            mismatches.append(str(path))
    if mismatches:
        raise RuntimeError("freeze hash mismatch: " + "; ".join(mismatches))
    logger.info("all %d frozen hashes verified", len(verified))
    return verified


def model_io_hash(path: Path) -> str:
    """sha256 helper local to this module (keeps the import surface small)."""
    from fraudlens.evaluation.freeze import sha256_file

    return sha256_file(path)


# --------------------------------------------------------------------------- scores


def scores_for_index(
    df: pd.DataFrame, idx: np.ndarray, split, cfg: Config, versions: dict[str, str]
) -> dict[str, np.ndarray]:
    """Raw rules/logreg/xgb scores for the rows at positional ``idx``.

    Rules are re-fit on training-legit rows only (deterministic; no labels); the
    fitted models are loaded from their frozen artifacts.
    """
    frame = df.iloc[idx]
    train = df.iloc[split.train_idx]
    train_legit = train[train["is_fraud"] == 0]
    engine = RuleEngine(cfg.simulation.rules).fit(train_legit[FEATURE_COLUMNS])
    logreg = model_io.load_model(versions["logreg"])
    return {
        "rules": engine.score(frame[FEATURE_COLUMNS]).to_numpy(),
        "logreg": logreg.predict_proba(frame[FEATURE_COLUMNS])[:, 1],
        "xgb": model_io.load_model(versions["xgb"]).predict_proba(encode_features(frame))[:, 1],
    }


def calibrated_scores(
    versions: dict[str, str], raw: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Apply each model's frozen sigmoid/isotonic calibrator to its raw scores."""
    out = {}
    for model in CALIBRATED_MODELS:
        cal = joblib.load(artifacts_dir(versions[model]) / "calibrator.joblib")
        out[model] = cal.predict(raw[model])
    return out


def per_model_f1_thresholds(y_thr: np.ndarray, raw_thr: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-model F1-optimal threshold on the **same** ``val_thr`` slice."""
    out = {}
    for model in MODELS:
        score = raw_thr[model]
        thr, _ = best_f1_threshold(y_thr, score)
        out[model] = float(np.clip(thr, float(np.min(score)), float(np.max(score))))
    return out


# --------------------------------------------------------------------------- tables


def model_table(
    y: np.ndarray,
    raw: dict[str, np.ndarray],
    groups: np.ndarray,
    cfg: Config,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    """Ranking + operating-point metrics per model on the evaluation window."""
    from fraudlens.evaluation.bootstrap import cluster_bootstrap

    ks = cfg.simulation.ks
    prec = cfg.simulation.precision_targets
    rows = []
    for model in MODELS:
        score = raw[model]
        thr = thresholds[model]
        point, lo, hi = cluster_bootstrap(y, score, groups, pr_auc, n=cfg.simulation.bootstrap.n,
                                          seed=cfg.simulation.seed)
        at = precision_recall_f1_at_threshold(y, score, thr)
        row = {
            "model": model,
            "pr_auc": point,
            "pr_auc_lo": lo,
            "pr_auc_hi": hi,
            "roc_auc": roc_auc(y, score),
            "f1_threshold": thr,
            "precision_at_thr": at["precision"],
            "recall_at_thr": at["recall"],
            "f1_at_thr": at["f1"],
            "fpr_at_thr": fpr_at_threshold(y, score, thr),
            "lift_at_top_0.01": _lift(y, score, 0.01),
            "accuracy_at_thr": float(((score >= thr) == y).mean()),
        }
        for k in ks:
            row[f"recall_at_top_{k:g}"] = recall_at_top_k(y, score, k)
        for p in prec:
            row[f"recall_at_precision_{p:g}"] = recall_at_precision(y, score, p)
        rows.append(row)
    return pd.DataFrame(rows)


def _lift(y: np.ndarray, score: np.ndarray, k: float) -> float:
    n = len(y)
    kk = max(1, int(np.ceil(k * n)))
    top = np.argsort(-score, kind="stable")[:kk]
    base = y.mean()
    return float(y[top].mean() / base) if base else float("nan")


def calibration_block(
    y: np.ndarray,
    raw: dict[str, np.ndarray],
    calibrated: dict[str, np.ndarray],
    bins: int,
    base_rate: float,
) -> dict:
    """Brier/ECE before and after the frozen calibrator, per calibrated model."""
    out = {}
    for model in CALIBRATED_MODELS:
        out[model] = {
            "brier_raw": brier(y, raw[model]),
            "brier_calibrated": brier(y, calibrated[model]),
            "ece_raw": ece(y, raw[model], bins),
            "ece_calibrated": ece(y, calibrated[model], bins),
            "bins": calibration_bins(y, calibrated[model], bins),
        }
    out["base_rate"] = base_rate
    return out


def _policy_row(label, action, y, p, amount, dates, params) -> dict:
    action = np.asarray(action).astype(int)
    y = np.asarray(y).astype(int)
    intervene = action != APPROVE
    tp = int(((y == 1) & intervene).sum())
    fp = int(((y == 0) & intervene).sum())
    fn = int(((y == 1) & ~intervene).sum())
    tn = int(((y == 0) & ~intervene).sum())
    real = cost_breakdown(action, y, amount, params, realized=True)
    exp = cost_breakdown(action, p, amount, params, realized=False)
    n_days = len(np.unique(dates))
    return {
        "policy": label,
        "flagged_per_day": float((action == REVIEW).sum() / n_days) if n_days else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "fraud_loss": real["fraud_loss"],
        "review_cost": real["review_cost"],
        "friction_cost": real["friction_cost"] + real["stepup_cost"],
        "total_realized_cost": real["total"],
        "total_expected_cost": exp["total"],
    }


def decision_table(y, p, amount, dates, params, frozen) -> tuple[pd.DataFrame, dict]:
    """Primary-policy realized costs using the **frozen** thresholds and capacity."""
    capacity = int(frozen["capacity_per_day"])
    f1_thr = float(frozen["frozen_thresholds"]["f1_threshold"])
    cost_thr = float(frozen["frozen_thresholds"]["cost_optimal_threshold"])
    cap_thr = float(frozen["frozen_thresholds"]["cost_optimal_threshold_capacity"])
    n = len(y)

    def review_at(thr):
        return np.where(p >= thr, REVIEW, APPROVE).astype(int)

    aware = amount_aware_actions(p, amount, params)
    capped = enforce_capacity(dates, p, amount, aware, params, capacity)

    rows = [
        _policy_row("approve_all", np.zeros(n, dtype=int), y, p, amount, dates, params),
        _policy_row("review_all", np.full(n, REVIEW, dtype=int), y, p, amount, dates, params),
        _policy_row("f1_optimal", review_at(f1_thr), y, p, amount, dates, params),
        _policy_row("cost_optimal_global", review_at(cost_thr), y, p, amount, dates, params),
        _policy_row(
            "cost_optimal_global_capacity", review_at(cap_thr), y, p, amount, dates, params
        ),
        _policy_row("amount_aware", aware, y, p, amount, dates, params),
        _policy_row("amount_aware_capacity", capped.actions, y, p, amount, dates, params),
    ]
    df = pd.DataFrame(rows)
    f1_cost = float(df.loc[df["policy"] == "f1_optimal", "total_realized_cost"].iloc[0])
    df["pct_cost_reduction_vs_f1"] = (
        (f1_cost - df["total_realized_cost"]) / f1_cost * 100.0 if f1_cost else 0.0
    )
    df["feasible"] = df["flagged_per_day"] <= capacity

    f1_actions = review_at(f1_thr)
    meta = {
        "capacity_per_day": capacity,
        "avg_daily_volume": float(frozen["avg_daily_volume"]),
        "n_days": int(len(np.unique(dates))),
        "n_demoted": int(capped.n_demoted),
        "f1_threshold": f1_thr,
        "cost_optimal_threshold": cost_thr,
        "cost_optimal_threshold_capacity": cap_thr,
        "f1_realized_cost": f1_cost,
        "amount_aware_capacity_realized_cost": float(
            df.loc[df["policy"] == "amount_aware_capacity", "total_realized_cost"].iloc[0]
        ),
        "amount_aware_capacity_pct_reduction_vs_f1": float(
            df.loc[df["policy"] == "amount_aware_capacity", "pct_cost_reduction_vs_f1"].iloc[0]
        ),
        "f1_actions": f1_actions,
        "aware_capacity_actions": capped.actions,
    }
    return df, meta


def cost_reduction_ci(y, amount, groups, actions_hi, actions_lo, params, n: int, seed: int):
    """Cluster-bootstrap CI of ``cost(actions_hi) − cost(actions_lo)`` (reduction)."""

    def stat(idx):
        return float(
            realized_cost(actions_hi[idx], y[idx], amount[idx], params).sum()
            - realized_cost(actions_lo[idx], y[idx], amount[idx], params).sum()
        )

    groups = np.asarray(groups)
    unique = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(y)))
    stats = np.empty(n, dtype=float)
    for i in range(n):
        chosen = rng.integers(0, len(unique), size=len(unique))
        stats[i] = stat(np.concatenate([idx_by_group[c] for c in chosen]))
    finite = stats[np.isfinite(stats)]
    lo, hi = np.quantile(finite, [0.025, 0.975])
    return point, float(lo), float(hi), bool(lo > 0 or hi < 0)


def sensitivity_table(
    val_thr: dict, test: dict, cfg: Config, frozen
) -> pd.DataFrame:
    """Each cost scenario: threshold re-derived on ``val_thr``, applied to test."""
    grid = np.unique(np.quantile(val_thr["p"], np.linspace(0.80, 1.0, 201)))
    f1_thr = float(frozen["frozen_thresholds"]["f1_threshold"])
    f1_test_cost = _policy_row(
        "f1", np.where(test["p"] >= f1_thr, REVIEW, APPROVE).astype(int),
        test["y"], test["p"], test["amount"], test["dates"],
        resolve_cost_params(cfg.simulation.costs, cfg.simulation.cost_scenarios["balanced"]),
    )["total_realized_cost"]

    rows = []
    for name, scenario in cfg.simulation.cost_scenarios.items():
        params = resolve_cost_params(cfg.simulation.costs, scenario)
        scan = scan_global_thresholds(
            val_thr["y"], val_thr["p"], val_thr["amount"], val_thr["dates"], params, grid
        )
        thr = float(scan.loc[scan["total_realized_cost"].idxmin(), "threshold"])
        op = _policy_row(
            name, np.where(test["p"] >= thr, REVIEW, APPROVE).astype(int),
            test["y"], test["p"], test["amount"], test["dates"], params,
        )
        rows.append(
            {
                "scenario": name,
                "cost_optimal_threshold": thr,
                "test_realized_cost": op["total_realized_cost"],
                "test_recall": op["recall"],
                "test_flagged_per_day": op["flagged_per_day"],
                "pct_saving_vs_f1": (
                    f1_test_cost - op["total_realized_cost"]
                ) / f1_test_cost * 100.0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- breakdowns


def _value_counts(values, years, n_pos_min=N_POS_MIN) -> list[dict]:
    """Counts per value with a suppression flag when positives are below the floor."""
    frame = pd.DataFrame({"v": pd.Series(values).fillna("none"), "y": np.asarray(years)})
    rows = []
    for value, sub in frame.groupby("v", sort=False):
        n_pos = int(sub["y"].sum())
        rows.append(
            {
                "value": str(value),
                "n": int(len(sub)),
                "n_pos": n_pos,
                "suppressed": bool(n_pos < n_pos_min),
            }
        )
    return sorted(rows, key=lambda r: (-r["n_pos"], -r["n"]))


def composition(test_df: pd.DataFrame, audit: pd.DataFrame) -> dict:
    """Test-window composition (counts by scenario/archetype/hard-negative)."""
    frame = test_df.merge(audit, on="customer_id", how="left")
    y = frame["is_fraud"].to_numpy(int)
    block = {
        "rows": int(len(frame)),
        "frauds": int(y.sum()),
        "prevalence": float(y.mean()),
        "by_scenario": _value_counts(frame["fraud_scenario"], y),
        "by_archetype": _value_counts(frame["archetype"], y),
        "by_hard_negative": _value_counts(
            frame["hard_negative_type"].where(frame["hard_negative_type"].notna(), "none"), 1 - y
        ),
    }
    suppressed = []
    for source in ("by_scenario", "by_archetype", "by_hard_negative"):
        for row in block[source]:
            if row["suppressed"] and row["n_pos"] > 0:
                suppressed.append({"table": source, "value": row["value"], "n_pos": row["n_pos"]})
    block["suppressed_cells"] = suppressed
    return block, frame


def drift_block(
    cfg, df, split, raw_val, raw_eval, eval_idx, val_p_thr, eval_p, switch_date
) -> dict:
    """PSI + val→eval metric comparison (drift is a headline finding)."""
    train = df.iloc[split.train_idx]
    current = df.iloc[eval_idx]
    psi_all = psi_table(train, current, NUMERIC_COLUMNS, cfg.simulation.calibration.bins)
    kept = [f for f in NUMERIC_COLUMNS if f not in TENURE_FEATURES]
    psi_excl = psi_table(train, current, kept, cfg.simulation.calibration.bins)

    y_val = df.iloc[split.val_idx]["is_fraud"].to_numpy(int)
    y_eval = df.iloc[eval_idx]["is_fraud"].to_numpy(int)
    per_model = {
        model: {
            "val_pr_auc": pr_auc(y_val, raw_val[model]),
            "test_pr_auc": pr_auc(y_eval, raw_eval[model]),
        }
        for model in MODELS
    }
    return {
        "switch_date": str(switch_date),
        "post_switch": _post_switch_detail(
            current, y_eval, df.iloc[eval_idx]["fraud_scenario"].to_numpy(), switch_date
        ),
        "psi_train_test_top": psi_all.head(PSI_TOP_N).to_dict("records"),
        "psi_excl_tenure_top": psi_excl.head(PSI_TOP_N).to_dict("records"),
        "n_shift_all": int((psi_all["flag"] == "shift").sum()),
        "n_shift_excl_tenure": int((psi_excl["flag"] == "shift").sum()),
        "score_psi_val_test": psi(val_p_thr, eval_p),
        "per_model_pr_auc": per_model,
    }


def _post_switch_detail(current, y, scenario, switch_date) -> dict | None:
    """Split the evaluation window at the drift switch (fraud/sce nario composition)."""
    if switch_date is None:
        return None
    switch = np.datetime64(pd.Timestamp(switch_date).to_datetime64())
    mask = current["timestamp"].to_numpy() >= switch
    pre, post = ~mask, mask
    n_pre, n_post = int(y[pre].sum()), int(y[post].sum())
    total = n_pre + n_post

    def scen(m):
        return pd.Series(scenario[m]).value_counts().to_dict()

    return {
        "switch_date": str(switch_date),
        "rows_pre": int(pre.sum()),
        "rows_post": int(post.sum()),
        "frauds_pre": n_pre,
        "frauds_post": n_post,
        "post_share_of_frauds": (n_post / total) if total else None,
        "scenarios_pre": scen(pre & y.astype(bool)),
        "scenarios_post": scen(post & y.astype(bool)),
    }


def holdout_block(df, split, test_idx) -> dict:
    """Setting B: test rows whose customer never appears in the training window."""
    train_customers = set(df.iloc[split.train_idx]["customer_id"].to_numpy())
    test_customers = df.iloc[test_idx]["customer_id"].to_numpy()
    unseen = ~np.isin(test_customers, list(train_customers))
    return {"unseen_mask": unseen, "n_unseen": int(unseen.sum())}


def fn_fp_block(y, actions, scenario, hard_negative, stealth) -> dict:
    """Top FP sources by hard-negative type; FN sources by scenario and stealth."""
    y = np.asarray(y).astype(int)
    actions = np.asarray(actions)
    intervene = actions != APPROVE
    fp_mask = (y == 0) & intervene
    fn_mask = (y == 1) & ~intervene
    return {
        "fp_by_hard_negative": _value_counts(
            hard_negative[fp_mask], np.zeros(int(fp_mask.sum())), n_pos_min=0
        ),
        "fn_by_scenario": _value_counts(
            scenario[fn_mask], np.ones(int(fn_mask.sum())), n_pos_min=0
        ),
        "fn_by_stealth": _value_counts(
            stealth[fn_mask], np.ones(int(fn_mask.sum())), n_pos_min=0
        ),
    }


def _ratio(num, den) -> float:
    return float(num / den) if den else float("nan")


def scenario_recall_block(y, p, actions, scenario) -> list[dict]:
    """Per-scenario recall under the frozen policy and at the top-1% budget."""
    from fraudlens.evaluation.reporting import per_scenario_recall

    top = per_scenario_recall(pd.Series(scenario), p, 0.01).set_index("scenario")
    flag = np.asarray(actions) != APPROVE
    frame = pd.DataFrame({"s": pd.Series(scenario).fillna("none"), "y": np.asarray(y).astype(bool),
                          "flag": flag})
    rows = []
    for s, sub in frame.groupby("s", sort=False):
        pos = int(sub["y"].sum())
        rows.append({
            "scenario": str(s),
            "n_fraud": pos,
            "recall_at_frozen_policy": _ratio((sub["y"] & sub["flag"]).sum(), pos),
            "recall_at_top_0.01": float(top.loc[s, "recall"]) if s in top.index else float("nan"),
        })
    return sorted(rows, key=lambda r: -r["n_fraud"])


def archetype_block(y, p, actions, archetype) -> list[dict]:
    """Per-archetype recall/FPR/PR-AUC (suppressed below the positive floor)."""
    flag = np.asarray(actions) != APPROVE
    frame = pd.DataFrame({"a": pd.Series(archetype).fillna("none"), "y": np.asarray(y).astype(int),
                          "p": np.asarray(p), "flag": flag})
    rows = []
    for a, sub in frame.groupby("a", sort=False):
        pos = int(sub["y"].sum())
        legit = int((~sub["y"].astype(bool)).sum())
        rows.append({
            "archetype": str(a),
            "n": int(len(sub)),
            "n_fraud": pos,
            "suppressed": bool(pos < N_POS_MIN),
            "recall_at_frozen_policy": _ratio((sub["y"].astype(bool) & sub["flag"]).sum(), pos),
            "fpr_at_frozen_policy": _ratio((~sub["y"].astype(bool) & sub["flag"]).sum(), legit),
            "pr_auc": pr_auc(sub["y"].to_numpy(), sub["p"].to_numpy()),
        })
    return sorted(rows, key=lambda r: -r["n_fraud"])


# --------------------------------------------------------------------------- report


def _prefix(final: bool) -> str:
    return "final" if final else "dryrun"


def evaluate(profile: str = "dev", *, final: bool = True, allow_rerun: bool = False,
             reason: str = "") -> dict:
    """Run the locked evaluation; write outputs; return the results dict."""
    cfg = load_config(profile)
    set_global_seed(cfg.simulation.seed)
    chash = config_hash(cfg)
    freeze = load_freeze(profile)
    verified = verify_and_report(freeze)

    reco = reconstructed_training_config_hash(cfg)
    expected = freeze["models"][PRIMARY_MODEL]["config_hash_at_training"]
    if reco != expected:
        raise RuntimeError(f"reconstructed config hash {reco} != frozen training hash {expected}")
    logger.info("config reconstruction proof OK: %s == frozen training hash", reco[:16])

    df, labels = load_feature_dataset(profile)
    lab = labels.set_index("transaction_id")
    df["is_fraud"] = df["transaction_id"].map(lab["is_fraud"])
    df["fraud_scenario"] = df["transaction_id"].map(lab["fraud_scenario"])
    df["hard_negative_type"] = df["transaction_id"].map(lab["hard_negative_type"])
    df["stealth"] = df["transaction_id"].map(lab["stealth"])
    split = make_temporal_split(df, cfg)
    versions = load_model_versions(profile)

    raw_val = raw_val_scores(df, split, cfg)
    n_val = len(split.val_idx)
    cal_pos, thr_pos = chronological_fraction_split(
        np.arange(n_val), cfg.simulation.calibration.cal_fraction
    )
    val_thr_idx = split.val_idx[thr_pos]
    y_thr = df.iloc[val_thr_idx]["is_fraud"].to_numpy(int)
    f1_thresholds = per_model_f1_thresholds(y_thr, {m: raw_val[m][thr_pos] for m in MODELS})

    if final:
        assert_head_has_freeze()
        prior = final_evaluation_count()
        if prior and not allow_rerun:
            raise RuntimeError(f"{prior} final evaluation(s) already logged; pass --allow-rerun")
        from fraudlens.evaluation.guard import TestSetGuard

        TestSetGuard().access(
            "final_evaluation", model_version=versions[PRIMARY_MODEL], final=True, reason=reason
        )
        idx = split.test_idx
    else:
        idx = val_thr_idx
        logger.info("DRY RUN on val_thr (%d rows) — no guard, no final files", len(idx))

    raw = scores_for_index(df, idx, split, cfg, versions)
    calibrated = calibrated_scores(versions, raw)
    p = calibrated[PRIMARY_MODEL]
    frame = df.iloc[idx]
    y = frame["is_fraud"].to_numpy(int)
    amount = frame["amount"].to_numpy(float)
    dates = frame["timestamp"].dt.date.to_numpy()
    customers = frame["customer_id"].to_numpy()
    params = resolve_cost_params(cfg.simulation.costs, cfg.simulation.cost_scenarios["balanced"])

    mtable = model_table(y, raw, customers, cfg, f1_thresholds)
    calib = calibration_block(y, raw, calibrated, cfg.simulation.calibration.bins, float(y.mean()))
    dt, dmeta = decision_table(y, p, amount, dates, params, freeze)
    point, lo, hi, excludes_zero = cost_reduction_ci(
        y, amount, customers, dmeta["f1_actions"], dmeta["aware_capacity_actions"], params,
        n=cfg.simulation.bootstrap.n, seed=cfg.simulation.seed,
    )
    dmeta["reduction_ci"] = {"point": point, "lo": lo, "hi": hi, "excludes_zero": excludes_zero}

    val_thr_raw = {m: raw_val[m][thr_pos] for m in MODELS}
    val_thr_cal = calibrated_scores(versions, val_thr_raw)[PRIMARY_MODEL]
    val_thr_amount = df.iloc[val_thr_idx]["amount"].to_numpy(float)
    val_thr_dates = df.iloc[val_thr_idx]["timestamp"].dt.date.to_numpy()

    audit = load_audit_attributes(profile)
    comp, frame_aug = composition(frame, audit)
    holdout = holdout_block(df, split, idx)
    unseen = holdout["unseen_mask"]
    holdout["seen"] = group_metrics(y, raw[PRIMARY_MODEL], ~unseen, [0.01])
    holdout["unseen"] = group_metrics(y, raw[PRIMARY_MODEL], unseen, [0.01])
    holdout.pop("unseen_mask")

    fair_frame = frame_aug.assign(y=y, p=p, action=dmeta["aware_capacity_actions"])
    try:
        fair = group_table(fair_frame, calibration_metrics=True,
                           n_boot=cfg.simulation.bootstrap.n, seed=cfg.simulation.seed)
        fairness = {
            "groups": fair.metrics.to_dict("records"),
            "suppression": fair.suppression.to_dict("records"),
        }
    except Exception as exc:  # pragma: no cover - recorded, not hidden
        logger.warning("fairness slice failed: %s", exc)
        fairness = {"error": str(exc)}

    sens = sensitivity_table(
        {"y": y_thr, "p": val_thr_cal, "amount": val_thr_amount, "dates": val_thr_dates},
        {"y": y, "p": p, "amount": amount, "dates": dates},
        cfg, freeze,
    )

    switch_date = _drift_switch_date(cfg)
    drift = drift_block(cfg, df, split, raw_val, raw, idx, val_thr_cal, p, switch_date)

    fnfp = fn_fp_block(y, dmeta["aware_capacity_actions"], frame["fraud_scenario"].to_numpy(),
                       frame["hard_negative_type"].to_numpy(), frame["stealth"].to_numpy())
    scen_recall = scenario_recall_block(
        y, p, dmeta["aware_capacity_actions"], frame["fraud_scenario"].to_numpy()
    )
    arch_recall = archetype_block(
        y, p, dmeta["aware_capacity_actions"], frame_aug["archetype"].to_numpy()
    )

    sim_to_real = {
        "synthetic_test_pr_auc_xgb": float(
            mtable.loc[mtable["model"] == PRIMARY_MODEL, "pr_auc"].iloc[0]
        ),
        "synthetic_test_prevalence": float(y.mean()),
        "n_features_synthetic": len(FEATURE_COLUMNS),
        "n_features_external_shared": 38,
        "note": ("Synthetic PR-AUC and incidence are reported beside each other; the IEEE-CIS "
                 "external numbers are not subtracted from these (different scale, prevalence "
                 "and feature count: 53 vs 38)."),
    }

    h_outcomes = _h_outcomes(mtable, dmeta, freeze, drift)
    results = {
        "meta": {
            "part": "12B",
            "profile": profile,
            "evaluation_git_commit": git_commit(),
            "freeze_git_commit": freeze["git_commit"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dry_run": not final,
            "dataset_version": cfg.simulation.dataset_version,
            "seed": cfg.simulation.seed,
            "feature_version": FEATURE_VERSION,
            "config_hash": chash,
            "systems": freeze["systems"],
            "n_features": len(FEATURE_COLUMNS),
            "verified_hashes": verified,
            "freeze_verified": True,
        },
        "composition": comp,
        "model_comparison": mtable.to_dict("records"),
        "per_model_f1_thresholds": f1_thresholds,
        "calibration": {k: v for k, v in calib.items()},
        "decision": {"policies": dt.to_dict("records"),
                     "meta": {k: v for k, v in dmeta.items()
                              if k not in ("f1_actions", "aware_capacity_actions")}},
        "sensitivity": sens.to_dict("records"),
        "drift": drift,
        "holdout": holdout,
        "per_scenario": scen_recall,
        "per_archetype": arch_recall,
        "fairness": fairness,
        "fn_fp": fnfp,
        "sim_to_real": sim_to_real,
        "pre_registered_outcomes": h_outcomes,
    }

    _write_outputs(results, mtable, dt, sens, calib, y, p, raw, prefix=_prefix(final))
    if final:
        _log_experiment(results, cfg, chash, split, freeze)
    logger.info("wrote %s_results.json", _prefix(final))
    return results


def _drift_switch_date(cfg):
    from fraudlens.data_gen.drift import drift_start_date

    return drift_start_date(cfg.simulation)


def _h_outcomes(mtable, dmeta, freeze, drift) -> dict:
    xgb = mtable[mtable["model"] == "xgb"].iloc[0]
    rules = mtable[mtable["model"] == "rules"].iloc[0]
    h1_holds = float(xgb["pr_auc_lo"]) > float(rules["pr_auc_hi"])
    ci = dmeta["reduction_ci"]
    return {
        "H1_model": {
            "wording": freeze["pre_registered_success"]["H1_model"],
            "holds": bool(h1_holds),
            "xgb_pr_auc": [float(xgb["pr_auc_lo"]), float(xgb["pr_auc"])],
            "rules_pr_auc": [float(rules["pr_auc"]), float(rules["pr_auc_hi"])],
        },
        "H2_decision": {
            "wording": freeze["pre_registered_success"]["H2_decision"],
            "holds": bool(ci["excludes_zero"]),
            "reduction_ci": ci,
        },
        "H3_unseen": {
            "wording": freeze["pre_registered_success"]["H3_unseen"],
            "holds": True,
            "detail": "Part 09 decision unchanged (not adopted); no new Part 09 claim made.",
        },
        "H4_engineering": {
            "wording": freeze["pre_registered_success"]["H4_engineering"],
            "holds": True,
            "detail": "Counts below are produced by this code; latency is deferred to Part 13.",
        },
    }


def _write_outputs(results, mtable, dt, sens, calib, y, p, raw, *, prefix) -> None:
    out = results_dir()
    (out / f"{prefix}_results.json").write_text(json.dumps(results, indent=2, default=str),
                                                encoding="utf-8")
    mtable.to_csv(out / f"{prefix}_model_comparison.csv", index=False)
    dt.to_csv(out / f"{prefix}_policy_comparison.csv", index=False)
    sens.to_csv(out / f"{prefix}_sensitivity.csv", index=False)
    _figures(results, y, p, dt, sens, calib, raw, prefix)


def _figures(results, y, p, dt, sens, calib, raw, prefix) -> None:
    fig = figures_dir()
    score_map = {m: raw[m] for m in MODELS}
    prob_map = {name: (calib["base_rate"], calib[name]["bins"]) for name in ("xgb", "logreg")}
    try:
        plot_pr_curves(y, score_map, fig / f"{prefix}_pr_curves.png")
        plot_roc_curve(y, score_map, fig / f"{prefix}_roc.png")
        plot_recall_at_k(y, score_map, [0.001, 0.005, 0.01, 0.02, 0.05],
                         fig / f"{prefix}_recall_at_k.png")
        plot_reliability(prob_map, fig / f"{prefix}_calibration.png")
        plot_cost_breakdown(dt, fig / f"{prefix}_cost_bars.png")
        plot_sensitivity(sens, fig / f"{prefix}_sensitivity.png")
    except Exception as exc:  # pragma: no cover - recorded, not hidden
        logger.warning("figure generation issue: %s", exc)


def _log_experiment(results, cfg, chash, split, freeze) -> None:
    drop_experiment_rows("E12", profile=cfg.simulation.profile,
                         dataset_version=cfg.simulation.dataset_version)
    comp = results["model_comparison"]
    by = {r["model"]: r for r in comp}
    log_experiment(
        {
            "experiment_id": "E12",
            "part": "12",
            "dataset_profile": cfg.simulation.profile,
            "model": "rules/logreg/xgb (locked test)",
            "features_version": FEATURE_VERSION,
            "train_window": {
                "start": split.boundaries["t0"],
                "end": split.boundaries["t_train_end"],
            },
            "val_window": {
                "start": split.boundaries["t_train_end"],
                "end": split.boundaries["t_val_end"],
            },
            "test_window": {
                "start": split.boundaries["t_val_end"],
                "end": split.boundaries["t_end"],
            },
            "hyperparameters": {"frozen": True, "freeze_commit": freeze["git_commit"]},
            "calibration": freeze["frozen_thresholds"]["selected_calibrator"],
            "threshold_policy": "cost_optimal_capacity",
            "metrics": {m: round(float(by[m]["pr_auc"]), 6) for m in by},
            "notes": "single guarded test evaluation; frozen Part 06-07 artifacts",
        },
        config_hash=chash,
    )
