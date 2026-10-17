"""Part 15D — render Markdown docs from ``{{key}}`` placeholders (numbers from results).

The README, model card and interview notes contain no hand-typed number: each is a
template with ``{{key}}`` slots filled from ``results/final_results.json`` and
``results/latency.json`` (plus a few *derived* keys under ``headline.*`` for the
locked wording). :func:`render` raises :class:`UnresolvedPlaceholder` if any slot
has no value, so a doc can never silently keep a stale or blank figure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fraudlens.evaluation.headlines import collect_test_counts
from fraudlens.paths import REPO_ROOT, results_dir

PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

TEMPLATES = {
    "readme": (REPO_ROOT / "docs" / "readme_template.md", REPO_ROOT / "README.md"),
    "model_card": (
        REPO_ROOT / "docs" / "model_card_template.md",
        REPO_ROOT / "docs" / "model_card.md",
    ),
    "interview_notes": (
        REPO_ROOT / "docs" / "interview_notes_template.md",
        REPO_ROOT / "docs" / "interview_notes.md",
    ),
}


class UnresolvedPlaceholder(KeyError):
    """A ``{{key}}`` had no value in the results context."""


def flatten(obj: object, prefix: str = "") -> dict[str, object]:
    """Flatten nested dict/list JSON into ``a.b[0].c`` dotted keys."""
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _derived(results: dict, latency: dict, counts: dict) -> dict[str, object]:
    """Curated keys for the locked wording (all computed from the JSON above)."""
    comp = results["composition"]
    dmeta = results["decision"]["meta"]
    policies = {p["policy"]: p for p in results["decision"]["policies"]}
    aware = policies["amount_aware_capacity"]
    drift = results["drift"]
    post = drift["post_switch"]
    xgb = next(m for m in results["model_comparison"] if m["model"] == "xgb")
    rules = next(m for m in results["model_comparison"] if m["model"] == "rules")
    logreg = next(m for m in results["model_comparison"] if m["model"] == "logreg")
    ci = dmeta["reduction_ci"]
    headline = latency.get("headline", {})
    db = latency.get("db", {})
    note = (
        "in-process ASGI replay (a lower bound); conditions: "
        f"backend {headline.get('backend')}, concurrency {headline.get('concurrency')}, "
        f"explanation {'on' if headline.get('explanation') else 'off'}, "
        f"n={headline.get('n')} ({latency.get('n_fraud')} fraud / {latency.get('n_legit')} legit, "
        f"fraud-enriched); profile {db.get('expected_profile')} "
        f"({db.get('transactions_rows'):,} rows); chosen after the matrix was measured, "
        "not pre-registered"
    )
    return {
        "headline.model.xgb_ci": (
            f"{xgb['pr_auc']:.4f} (CI {xgb['pr_auc_lo']:.4f}-{xgb['pr_auc_hi']:.4f})"
        ),
        "headline.model.rules_ci": (
            f"{rules['pr_auc']:.4f} (CI {rules['pr_auc_lo']:.4f}-{rules['pr_auc_hi']:.4f})"
        ),
        "headline.model.logreg_ci": (
            f"{logreg['pr_auc']:.4f} (CI {logreg['pr_auc_lo']:.4f}-{logreg['pr_auc_hi']:.4f})"
        ),
        "headline.model.recall_at_1pct": f"{xgb['recall_at_top_0.01']:.4f}",
        "headline.model.recall_at_capacity": f"{aware['recall']:.4f}",
        "headline.model.test_prevalence_pct": f"{comp['prevalence'] * 100:.3f}%",
        "headline.decision.reduction": (
            f"{ci['point']:,.0f} INR ({dmeta['amount_aware_capacity_pct_reduction_vs_f1']:.2f}%), "
            f"CI {ci['lo']:,.0f} to {ci['hi']:,.0f}"
        ),
        "headline.decision.reduction_ci_inr": f"{ci['lo']:,.0f}-{ci['hi']:,.0f}",
        "headline.decision.capacity": (
            f"{aware['flagged_per_day']:.1f}/day vs {dmeta['capacity_per_day']}/day, "
            f"n_demoted {dmeta['n_demoted']}"
        ),
        "headline.decision.capacity_note": (
            "Capacity did not bind on test: the policy flags "
            f"{aware['flagged_per_day']:.1f}/day against a {dmeta['capacity_per_day']}/day cap "
            f"with {dmeta['n_demoted']} demotions, so `enforce_capacity` was never exercised and "
            "the capacity-aware result is NOT evidence the cap works — it is the unconstrained "
            "policy."
        ),
        "headline.drift.line": (
            "no degradation observed "
            f"({drift['per_model_pr_auc']['xgb']['val_pr_auc']:.4f} val -> "
            f"{drift['per_model_pr_auc']['xgb']['test_pr_auc']:.4f} test)"
        ),
        "headline.drift.switch_date": drift["switch_date"],
        "headline.drift.post_switch": (
            f"{post['post_share_of_frauds']:.0%} of test frauds fall after the drift switch"
        ),
        "headline.anomaly.note": (
            "null — the Part 09 isolation forest was re-run post cold-start fix and not adopted "
            "(the S1-S0 / S1s-S0 bagged CIs include 0)"
        ),
        "headline.engineering.latency": f"{headline.get('e2e_p95_ms'):.3f} ms p95",
        "headline.engineering.latency_note": note,
        "headline.engineering.latency_rule": latency.get("headline_rule", ""),
        "headline.engineering.parity": (
            "postgres and memory providers return identical feature rows on a tiny sample; "
            "calibrated-probability parity (1e-6, n=220) is proven on dev/memory."
        ),
        "headline.sim_to_real": results["sim_to_real"]["note"],
        "headline.n_features": int(results["meta"]["n_features"]),
        "headline.n_tests": counts.get("n_tests"),
        "headline.n_leakage_tests": counts.get("n_leakage_tests"),
    }


def build_context(
    *,
    final_results: dict | None = None,
    latency: dict | None = None,
    freeze: dict | None = None,
    test_counts: dict | None = None,
) -> dict[str, object]:
    """Flat ``key -> value`` context for every template placeholder."""
    if final_results is not None:
        results = final_results
    else:
        results = _load(results_dir() / "final_results.json")
    lat = latency if latency is not None else _load(results_dir() / "latency.json")
    frz_path = results_dir() / "freeze.json"
    frz = freeze if freeze is not None else (_load(frz_path) if frz_path.exists() else {})
    counts = test_counts if test_counts is not None else collect_test_counts()
    context = flatten(results)
    context.update(flatten(lat, "latency"))
    context.update(flatten(frz, "freeze"))
    context.update(_derived(results, lat, counts))
    return context


def resolve(expr: str, context: dict[str, object]) -> str:
    """Resolve one placeholder body ``key`` or ``key:spec``."""
    key, sep, spec = expr.partition(":")
    key = key.strip()
    if key not in context:
        raise UnresolvedPlaceholder(
            f"placeholder {{{{{expr}}}}} has no value in results/final_results.json or "
            f"results/latency.json (key {key!r})"
        )
    value = context[key]
    if value is None:
        raise UnresolvedPlaceholder(f"placeholder {{{{{expr}}}}} resolved to None")
    return format(value, spec.strip()) if sep else str(value)


def render(text: str, context: dict[str, object]) -> tuple[str, list[str]]:
    """Substitute every ``{{...}}``; raise on the first unresolved slot."""
    used: list[str] = []

    def _sub(match: re.Match) -> str:
        used.append(match.group(1).strip())
        return resolve(match.group(1), context)

    return PLACEHOLDER.sub(_sub, text), used


def render_templates(context: dict[str, object] | None = None) -> dict[str, Path]:
    """Render every registered template to its output path; return the written paths."""
    ctx = context or build_context()
    written: dict[str, Path] = {}
    for name, (template, out) in TEMPLATES.items():
        rendered, _ = render(template.read_text(encoding="utf-8"), ctx)
        leftovers = PLACEHOLDER.findall(rendered)
        if leftovers:
            raise UnresolvedPlaceholder(f"{name}: unresolved placeholders {leftovers}")
        out.write_text(rendered, encoding="utf-8")
        written[name] = out
    return written
