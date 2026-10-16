"""Part 02b before/after table.

Compares the pre-cold-start-fix artefacts (tag ``pre-coldstart-fix``) with the
current working tree. Only git-tracked artefacts can be compared numerically;
``data/raw/*`` is gitignored, so simulator counts are cited to the tracked docs
that recorded them. Read-only apart from the single output CSV.

Usage: python scripts/part_02b_report.py
"""

from __future__ import annotations

import io
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG = "pre-coldstart-fix"
OUT = ROOT / "results" / "part_02b_before_after.csv"
SIMMD = "docs/what_went_wrong.md (data/raw is gitignored)"
POST_VERSION = "post_coldstart_fix"
PRE_VERSION = "pre_coldstart_fix"

GATE = "results/separability_gate_dev.json"
CAL = "results/calibration_selection_dev.json"
LAT = "results/shap_latency_dev.json"
E8E = "results/exp_8E_adversarial_summary_drift_lab.json"
E9 = "results/exp_09_unseen_decision_unseen_lab_structuring.json"


def _bytes(ref: str, rel: str) -> bytes:
    if ref == "post":
        return (ROOT / rel).read_bytes()
    proc = subprocess.run(
        ["git", "show", f"{TAG}:{rel}"], cwd=ROOT, capture_output=True, check=True
    )
    return proc.stdout


def _json(ref: str, rel: str) -> Any:
    return json.loads(_bytes(ref, rel).decode("utf-8"))


def _csv(ref: str, rel: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(_bytes(ref, rel).decode("utf-8")))


def _jv(ref: str, rel: str, *keys: str) -> Any:
    d = _json(ref, rel)
    for k in keys:
        d = d[k]
    return d


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _value(fn: Callable[[str], Any], ref: str) -> str:
    try:
        return _fmt(fn(ref))
    except Exception as exc:
        return f"n/a [{type(exc).__name__}]"


def _pre_open_rows(ref: str, profile: str) -> int:
    tx = pd.read_parquet(io.BytesIO(_bytes(ref, f"data/raw/{profile}/transactions.parquet")))
    cust = pd.read_parquet(io.BytesIO(_bytes(ref, f"data/raw/{profile}/customers.parquet")))
    m = tx[["customer_id", "timestamp"]].merge(
        cust[["customer_id", "account_open_date"]], on="customer_id", how="left"
    )
    return int((m["timestamp"] < m["account_open_date"]).sum())


def _exp_metrics(ref: str, exp_id: str, profile: str) -> dict:
    df = _csv(ref, "results/experiments.csv")
    mask = (df.experiment_id == exp_id) & (df.dataset_profile == profile)
    if "dataset_version" in df.columns:
        ver = PRE_VERSION if ref == "pre" else POST_VERSION
        mask &= df.dataset_version == ver
    return json.loads(df[mask].iloc[0]["metrics"])


def _policy(ref: str, key: str) -> Any:
    df = _csv(ref, "results/policy_comparison_dev.csv")
    return df[df.policy == "amount_aware_capacity"].iloc[0][key]


def _imbalance_winner(ref: str) -> str:
    d = _csv(ref, "results/exp_8A_imbalance_summary_dev.csv")
    r = d.loc[d.pr_auc_mean.idxmax()]
    return f"{r['config']} pr_auc={r['pr_auc_mean']:.4f} rec@1pct={r['recall_at_1pct_mean']:.4f}"


def _max_scenario_delta(ref: str) -> str:
    d = _csv(ref, "results/exp_8B_scenario_delta_dev.csv")
    cols = [c for c in d.columns if c.startswith("delta_")]
    col = max(cols, key=lambda c: abs(d[c]).max())
    return f"{col.replace('delta_', '')}={d[col][abs(d[col]).idxmax()]:+.4f}"


def _top_shap(ref: str) -> str:
    r = _csv(ref, "results/shap_global_dev.csv").iloc[0]
    return f"{r['feature']} ({r['mean_abs_shap']:.4f})"


def _gallery_ids(ref: str) -> str:
    g = _csv(ref, "results/shap_gallery_dev.csv")
    return ";".join(f"{r['case']}={r['transaction_id']}" for _, r in g.iterrows())


def _fairness_flags(ref: str) -> str:
    f = _csv(ref, "results/fairness_pairwise_dev.csv")
    return f"{int(f['distinguishable'].sum())} of {len(f)}"


def main() -> None:
    rows: list[dict[str, str]] = []

    def add(
        group: str,
        scope: str,
        quantity: str,
        source: str,
        pre: str,
        pre_src: str,
        post: str,
    ) -> None:
        rows.append(
            {
                "group": group,
                "scope": scope,
                "quantity": quantity,
                "source": source,
                "pre": pre,
                "pre_source": pre_src,
                "post": post,
            }
        )

    def add2(group: str, scope: str, quantity: str, source: str, fn: Callable[[str], Any]) -> None:
        add(group, scope, quantity, source, _value(fn, "pre"), f"git:{TAG}", _value(fn, "post"))

    # simulator counts: post from report json (untracked), pre cited to tracked docs
    for p in ("tiny", "dev", "drift_lab", "unseen_lab"):
        rep = f"data/raw/{p}/simulation_report.json"
        for key in ("n_transactions", "n_fraud_transactions", "n_fraud_episodes", "fraud_rate"):
            add("simulator", p, key, "data/raw/<p>/simulation_report.json", "",
                SIMMD, _value(lambda r, k=key, rp=rep: _jv(r, rp, k), "post"))
        add2(
            "simulator", p, "fraud_rate_vs_target_pct",
            "simulation_report.json (new field)",
            lambda r, rp=rep: _jv(r, rp, "fraud_rate_vs_target_pct"),
        )
        add2(
            "simulator", p, "cold_start_gate", "simulation_report.json (new field)",
            lambda r, rp=rep: _jv(r, rp, "cold_start_gate"),
        )
        add2("simulator", p, "pre_open_rows", "transactions x customers parquet",
             lambda r, q=p: _pre_open_rows(r, q))

    # separability gate
    for key in (
        "passed", "best_single_signal_roc_auc", "best_single_signal_pr_auc",
        "tree_depth3_pr_auc", "boosted_pr_auc", "tree3_vs_boosted_gap",
        "overlap_signals_passed",
    ):
        add2("separability_gate", "dev", key, GATE, lambda r, k=key: _jv(r, GATE, k))

    # models / calibration / policy rows from experiments.csv (E-rows are unambiguous)
    for group, profile, exp_id, key in (
        ("models", "dev", "E01", "pr_auc"),
        ("models", "dev", "E02", "pr_auc"),
        ("models", "dev", "E03", "pr_auc"),
        ("models", "tiny", "E03", "pr_auc"),
        ("calibration", "dev", "E04", "pr_auc"),
        ("calibration", "dev", "E04", "ece"),
        ("policy", "dev", "E05", "realized_cost_amount_aware"),
        ("policy", "dev", "E05", "pct_saving_vs_f1"),
        ("policy", "dev", "E05", "recall"),
        ("policy", "dev", "E05", "flagged_per_day"),
    ):
        add2(
            group, profile, f"{exp_id}.{key}", "results/experiments.csv",
            lambda r, e=exp_id, p=profile, k=key: _exp_metrics(r, e, p)[k],
        )

    add2("calibration", "dev", "selected_calibrator", CAL,
         lambda r: _jv(r, CAL, "xgb", "selected"))
    for key in ("flagged_per_day", "recall", "precision", "fpr", "total_realized_cost",
                "pct_cost_reduction_vs_f1"):
        add2("policy", "dev", key, "results/policy_comparison_dev.csv[amount_aware_capacity]",
             lambda r, k=key: _policy(r, k))

    # experiments 8 / 9
    add2("exp8E", "drift_lab", "adversarial_auc_behavioral", E8E,
         lambda r: _jv(r, E8E, "adversarial_auc_behavioral"))
    add2("exp9", "unseen_lab", "S1_holdout_recall_diff", E9,
         lambda r: _jv(r, E9, "paired", "S1", "diff_bagged"))
    add2("exp8", "dev", "exp8A.imbalance_winner", "results/exp_8A_imbalance_summary_dev.csv",
         _imbalance_winner)
    add2("exp8", "dev", "exp8B.max_scenario_delta", "results/exp_8B_scenario_delta_dev.csv",
         _max_scenario_delta)

    # Part 10
    add2("shap", "dev", "top_mean_abs_shap", "results/shap_global_dev.csv", _top_shap)
    for key in ("p50_ms", "p95_ms"):
        add2("shap", "dev", key, LAT, lambda r, k=key: _jv(r, LAT, k))
    add2("part10", "dev", "gallery_rows", "results/shap_gallery_dev.csv", _gallery_ids)
    add2("part10", "dev", "fairness_distinguishable", "results/fairness_pairwise_dev.csv",
         _fairness_flags)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    table = _to_markdown(out)
    md = OUT.with_suffix(".md")
    md.write_text(table, encoding="utf-8")
    _inject_into_progress(table)
    print(f"wrote {OUT.relative_to(ROOT)}, {md.relative_to(ROOT)} and the table in part_02b.md")
    print(out.to_string(index=False, max_colwidth=40))


def _inject_into_progress(table: str) -> None:
    """Replace the marker block in docs/progress/part_02b.md with ``table``."""
    doc = ROOT / "docs" / "progress" / "part_02b.md"
    begin, end = "<!-- BEGIN part_02b_table -->", "<!-- END part_02b_table -->"
    text = doc.read_text(encoding="utf-8")
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    doc.write_text(f"{pre}{begin}\n{table}{end}{post}", encoding="utf-8")


def _to_markdown(df: pd.DataFrame) -> str:
    """Compact markdown table (source/pre_source collapsed into a footnote)."""
    lines = [
        "<!-- generated by scripts/part_02b_report.py; do not edit by hand -->",
        "",
        "| group | scope | quantity | pre | post |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        pre = r["pre"] if r["pre"] != "" else r["pre_source"]
        lines.append(
            f"| {r['group']} | {r['scope']} | {r['quantity']} | {pre} | {r['post']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
