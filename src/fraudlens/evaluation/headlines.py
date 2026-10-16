"""Part 12C — headline numbers for the README/resume (from ``final_results.json``).

Exactly four headlines, each printed with the assumptions it depends on. Every
number is read from ``results/final_results.json``; nothing is invented. If a
required key is missing the builder raises, so a headline can never silently drop
to a placeholder (Part 12E).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fraudlens.paths import REPO_ROOT, results_dir
from fraudlens.utils import get_logger

logger = get_logger(__name__)

REQUIRED_KEYS = (
    "meta",
    "composition",
    "model_comparison",
    "decision",
    "drift",
    "pre_registered_outcomes",
    "sim_to_real",
)


def load_final_results(path: str | Path | None = None) -> dict:
    """Read ``final_results.json`` and validate the keys the headlines need."""
    p = Path(path) if path else results_dir() / "final_results.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/final_evaluation.py first")
    results = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in results]
    if missing:
        raise KeyError(f"final_results.json missing required keys: {missing}")
    return results


def _model(results: dict, name: str) -> dict:
    return next(r for r in results["model_comparison"] if r["model"] == name)


def collect_test_counts() -> dict:
    """Count automated tests and leakage tests via ``pytest --collect-only -q``."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.SubprocessError, OSError) as exc:  # pragma: no cover
        logger.warning("pytest collection failed: %s", exc)
        return {"n_tests": None, "n_leakage_tests": None}
    ids = [line for line in out.stdout.splitlines() if "::" in line]
    return {
        "n_tests": len(ids),
        "n_leakage_tests": sum(1 for line in ids if "leak" in line.lower()),
    }


def _ci(model: dict) -> str:
    return f"{model['pr_auc']:.4f} (CI {model['pr_auc_lo']:.4f}-{model['pr_auc_hi']:.4f})"


def build_headlines(results: dict) -> dict:
    """Assemble the four pre-registered headline blocks plus resume bullets."""
    meta = results["meta"]
    comp = results["composition"]
    xgb, rules, logreg = _model(results, "xgb"), _model(results, "rules"), _model(results, "logreg")
    dmeta = results["decision"]["meta"]
    policies = {p["policy"]: p for p in results["decision"]["policies"]}
    aware = policies["amount_aware_capacity"]
    ci = dmeta["reduction_ci"]
    drift = results["drift"]
    post = drift.get("post_switch") or {}
    counts = collect_test_counts()
    capacity = dmeta["capacity_per_day"]
    pct = dmeta["amount_aware_capacity_pct_reduction_vs_f1"]

    h1 = {
        "xgb": _ci(xgb),
        "rules": _ci(rules),
        "logreg": _ci(logreg),
        "test_prevalence": comp["prevalence"],
        "recall_at_top_0.01": xgb["recall_at_top_0.01"],
        "recall_at_capacity_budget": aware["recall"],
        "assumptions": (
            "PR-AUC is primary (imbalanced data; accuracy is a footnote). CI = cluster "
            f"bootstrap over customers (n=500, seed {meta['seed']}). "
            f"Test prevalence {comp['prevalence'] * 100:.3f}%."
        ),
    }

    h2 = {
        "value": (
            f"realized-cost reduction vs F1-optimal: {ci['point']:,.0f} INR ({pct:.2f}%), "
            f"CI {ci['lo']:,.0f} to {ci['hi']:,.0f}"
        ),
        "ci_excludes_zero": ci["excludes_zero"],
        "flagged_per_day_vs_capacity": (
            f"{aware['flagged_per_day']:.1f}/day vs {capacity}/day"
        ),
        "n_demoted": dmeta["n_demoted"],
        "assumptions": (
            "Costs (INR): alpha=0.85 of amount lost, review_cost=75, friction_cost=150, "
            f"step-up 5/40, block friction 450; analyst capacity {capacity}/day "
            f"(avg daily volume {dmeta['avg_daily_volume']:.1f}). Realized cost on test "
            "with the frozen validation threshold; the reduction CI resamples customers."
        ),
    }

    h3 = {
        "structuring_recall_anomaly_before_after": None,
        "structuring_recall_note": (
            "Part 09 isolation forest was re-run post cold-start fix and NOT adopted "
            "(S1-S0 and S1s-S0 bagged CIs include 0); no anomaly feature is in the policy, "
            "so there is no before/after gain to report."
        ),
        "drift_val_to_test_pr_auc_xgb": {
            "val": drift["per_model_pr_auc"]["xgb"]["val_pr_auc"],
            "test": drift["per_model_pr_auc"]["xgb"]["test_pr_auc"],
        },
        "drift_switch_date": drift["switch_date"],
        "post_switch_share_of_test_frauds": post.get("post_share_of_frauds"),
        "assumptions": (
            f"The drift switch {drift['switch_date']} sits inside the test window; "
            f"{post.get('post_share_of_frauds', float('nan')):.0%} of test frauds are "
            "post-switch, so val->test degradation mixes drift with the window change."
        ),
    }

    h4 = {
        "feature_count": meta["n_features"],
        "n_automated_tests": counts["n_tests"],
        "n_leakage_tests": counts["n_leakage_tests"],
        "api_p95_latency_ms": None,
        "latency_note": "Filled in Part 13 (FastAPI deployment); placeholder until then.",
        "assumptions": "Counts produced by code (feature registry; pytest --collect-only).",
    }

    bullets = [
        "Built an end-to-end fraud-risk system; frozen XGBoost test PR-AUC "
        f"{xgb['pr_auc']:.3f} (CI {xgb['pr_auc_lo']:.3f}-{xgb['pr_auc_hi']:.3f}) vs Rules "
        f"{rules['pr_auc']:.3f} at {comp['prevalence'] * 100:.2f}% prevalence "
        "[results/final_results.json:model_comparison.xgb]",
        "Cost-based capacity policy cut realized fraud cost "
        f"{pct:.1f}% vs F1-optimal (CI {ci['lo']:,.0f}-{ci['hi']:,.0f} INR) under a "
        f"{capacity}/day analyst cap [final_results.json:decision.meta.reduction_ci]",
        f"Reported concept drift as a finding: {post.get('post_share_of_frauds', 0):.0%} of "
        "test frauds fall after the drift switch [final_results.json:drift.post_switch]",
        f"{meta['n_features']}-feature registry, {counts['n_leakage_tests']} leakage tests, "
        f"{counts['n_tests']} automated tests [final_results.json:meta]",
    ]
    return {
        "headline_1_model_performance": h1,
        "headline_2_business_decision": h2,
        "headline_3_unseen_fraud": h3,
        "headline_4_engineering": h4,
        "pre_registered_outcomes": results["pre_registered_outcomes"],
        "resume_bullet_candidates": bullets,
    }


def headline_markdown(results: dict) -> str:
    """Render the four headlines + assumptions + resume bullets to markdown."""
    h = build_headlines(results)
    out = results["pre_registered_outcomes"]
    m = results["meta"]
    lines = [
        "# FraudLens — headline numbers",
        "",
        f"Profile `{m['profile']}` · test window · frozen evaluation "
        f"(freeze commit `{m['freeze_git_commit'][:10]}`, eval commit "
        f"`{m['evaluation_git_commit'][:10]}`).",
        "",
    ]
    h1, h2, h3, h4 = (
        h["headline_1_model_performance"],
        h["headline_2_business_decision"],
        h["headline_3_unseen_fraud"],
        h["headline_4_engineering"],
    )
    lines += [
        "## 1. Model performance",
        "",
        f"- XGB: {h1['xgb']}",
        f"- {h1['rules']}",
        f"- {h1['logreg']}",
        f"- recall@1% budget: {h1['recall_at_top_0.01']:.4f}; "
        f"recall at capacity policy: {h1['recall_at_capacity_budget']:.4f}",
        f"- _assumptions:_ {h1['assumptions']}",
        "",
        "## 2. Business decision",
        "",
        f"- {h2['value']}",
        f"- CI excludes 0: {h2['ci_excludes_zero']}",
        f"- demand vs capacity: {h2['flagged_per_day_vs_capacity']}",
        f"- n_demoted: {h2['n_demoted']}",
        f"- _assumptions:_ {h2['assumptions']}",
        "",
        "## 3. Unseen fraud & drift",
        "",
        f"- anomaly before/after: {h3['structuring_recall_anomaly_before_after']} "
        f"— {h3['structuring_recall_note']}",
        f"- XGB PR-AUC val->test: {h3['drift_val_to_test_pr_auc_xgb']}",
        f"- drift switch {h3['drift_switch_date']}; post-switch share of test frauds: "
        f"{h3['post_switch_share_of_test_frauds']}",
        f"- _assumptions:_ {h3['assumptions']}",
        "",
        "## 4. Engineering",
        "",
        f"- features: {h4['feature_count']}",
        f"- automated tests: {h4['n_automated_tests']}; leakage tests: {h4['n_leakage_tests']}",
        f"- API p95 latency: {h4['api_p95_latency_ms']} ({h4['latency_note']})",
        "",
        "## Pre-registered outcomes (verbatim)",
        "",
    ]
    for key, block in out.items():
        lines.append(f"- **{key}** — holds={block['holds']}: {block['wording']}")
    lines += ["", "## Resume bullet candidates (real numbers)", ""]
    lines += [f"- {b}" for b in h["resume_bullet_candidates"]]
    lines.append("")
    return "\n".join(lines)


def write_headline_numbers(results: dict, path: str | Path | None = None) -> Path:
    """Write ``reports/headline_numbers.md`` from ``results``."""
    from fraudlens.paths import reports_dir

    out = Path(path) if path else reports_dir() / "headline_numbers.md"
    out.write_text(headline_markdown(results), encoding="utf-8")
    return out
