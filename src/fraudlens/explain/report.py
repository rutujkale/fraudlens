"""Markdown rendering for Part 10 docs (10D gallery, 10E fairness audit).

Tables are rendered from the same in-memory frames that are written to
``results/``, so a number in a doc cannot drift from its CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ROUND = {
    "prevalence": 4,
    "mean_p": 4,
    "calibration_gap": 4,
    "brier": 6,
    "flag_rate_any": 4,
    "flag_rate_strict": 4,
    "fpr_any": 4,
    "fpr_strict": 4,
    "recall": 3,
    "precision": 3,
    "pr_auc": 3,
    "flag_rate_any_lo": 4,
    "flag_rate_any_hi": 4,
    "fpr_any_lo": 4,
    "fpr_any_hi": 4,
    "recall_lo": 3,
    "recall_hi": 3,
}


def _fmt(value, spec: str | None = None, na: str = "suppressed") -> str:
    if isinstance(value, float) and np.isnan(value):
        return na
    if isinstance(value, float):
        return f"{value:{spec}}" if spec else f"{value:g}"
    return str(value)


def _md_table(
    df: pd.DataFrame,
    columns: list[str],
    headers: list[str] | None = None,
    formats: dict[str, str] | None = None,
    na: str = "suppressed",
) -> str:
    headers = headers or columns
    formats = formats or {}
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in df.iterrows():
        cells = [_fmt(row[c], formats.get(c), na) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rounded(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(_ROUND.get(col, 6))
    return out


def _partition_totals(df: pd.DataFrame) -> tuple[int, int]:
    """Rows and frauds of the slice behind a group table.

    Every audit attribute partitions the same rows, so sum each partition separately
    and require them to agree; a frame-wide sum would count the rows once per attribute.
    """
    per_attr = df.groupby("attribute")[["n", "n_fraud"]].sum()
    n_rows = int(per_attr["n"].max())
    n_fraud = int(per_attr["n_fraud"].max())
    if not ((per_attr["n"] == n_rows) & (per_attr["n_fraud"] == n_fraud)).all():
        raise ValueError("audit attributes do not partition the same rows")
    return n_rows, n_fraud


FAIRNESS_HEADER = """# Fairness / subgroup audit (Part 10E)

Audit attributes (`age_band`, `income_band`, `archetype`, `home_region_group`) are
**not model inputs** — `tests/test_explain.py` asserts they are absent from
`FEATURE_COLUMNS`. They are joined from `data/raw/{profile}/customers.parquet`
(and `locations.region_group`) for analysis only.

Alert definition: a transaction is **flagged** when the frozen policy's action is
not `APPROVE`. `flag_any` counts `STEP_UP`/`REVIEW`/`BLOCK`; `flag_strict` counts
only `REVIEW`/`BLOCK`. The two differ in burden: `STEP_UP` costs the customer a
step-up (Rs 40 friction, Rs 5 handling) while `REVIEW` costs the bank Rs 75 and
`BLOCK` imposes Rs 450 of friction on a legitimate customer.

This audit is run on **simulated** data: subgroup differences reflect the
simulator's own assumptions about age/income/archetype, not real demographics.
Nothing here is a statement about fairness in the real world, and it is not a
regulatory compliance claim.
"""


def render_fairness_markdown(
    *,
    profile: str,
    primary_note: str,
    primary: pd.DataFrame,
    primary_suppression: pd.DataFrame,
    secondary: pd.DataFrame,
    ratios: list[dict],
    pairwise: pd.DataFrame,
    proxy: pd.DataFrame,
    policy_line: str,
) -> str:
    """Render ``docs/fairness_audit.md`` from computed tables."""
    primary_rows, primary_frauds = _partition_totals(primary)
    secondary_rows, secondary_frauds = _partition_totals(secondary)
    parts = [FAIRNESS_HEADER.format(profile=profile), f"\n**Policy:** {policy_line}\n"]

    parts.append("\n## Suppressed groups per attribute (reported up front)\n")
    parts.append(
        "A group is counted under `suppressed_positives` when it has fewer than 30 "
        "frauds (recall/precision/PR-AUC are then not reported) and under "
        "`suppressed_rows` when it has fewer than 1,000 rows (calibration gap, Brier "
        "and bootstrap CIs are then not reported). `reported` counts groups with no "
        "suppression at all.\n"
    )
    parts.append(
        _md_table(
            primary_suppression,
            ["attribute", "n_groups", "suppressed_positives", "suppressed_rows", "reported"],
            ["attribute", "groups", "positives<30", "rows<1000", "fully reported"],
        )
    )

    parts.append("\n## Primary view — `val_thr` (the policy's own selection slice)\n")
    parts.append(
        f"{primary_note}\n\nAll {primary_frauds:,} frauds / {primary_rows:,} rows the "
        "policy was selected on; "
        "the calibrator was fitted earlier in validation, so the calibration gap and "
        "Brier here are out-of-sample.\n"
    )
    cols = [
        "attribute",
        "group",
        "n",
        "n_fraud",
        "prevalence",
        "flag_rate_any",
        "fpr_any",
        "fpr_strict",
        "recall",
        "precision",
        "pr_auc",
        "mean_p",
        "calibration_gap",
        "brier",
        "suppressed_reason",
    ]
    parts.append(_md_table(_rounded(primary, cols), cols))

    parts.append("\n## Per-action counts (burden, not just flag rates)\n")
    action_cols = ["attribute", "group", "n_approve", "n_step_up", "n_review", "n_block"]
    parts.append(_md_table(primary[action_cols], action_cols))

    parts.append("\n## Ratios and sampling noise\n")
    ratio_lines = []
    for r in ratios:
        if not np.isfinite(r["ratio"]):
            ratio_lines.append(
                f"- `{r['metric']}` ({r['attribute']}): not computable — "
                "fewer than two fully reported groups"
            )
            continue
        ratio_lines.append(
            f"- `{r['metric']}` ({r['attribute']}) max/min ratio = {r['ratio']:.3f} "
            f"(high: {r['hi_group']}; low: {r['lo_group']})"
        )
    parts.append("\n".join(ratio_lines))
    parts.append(
        "\nDifferences vs the reference group (largest by row count), paired cluster "
        "bootstrap over customers — 95% CI excluding 0 means the difference is not "
        "explained by sampling noise alone:\n"
    )
    pair_cols = ["attribute", "group", "metric", "point", "lo", "hi", "distinguishable"]
    parts.append(_md_table(_rounded(pairwise, pair_cols), pair_cols))
    n_dist = int(pairwise["distinguishable"].sum()) if len(pairwise) else 0
    parts.append(
        f"\n**{n_dist} of {len(pairwise)}** differences are distinguishable from noise — and these "
        f"are **uncorrected for multiple comparisons**. {len(pairwise)} tests at a 95% level would "
        "throw up roughly 5% false positives by chance alone, and the tests share bootstrap draws "
        "and reference groups, so the family-wise error rate is above 5%. Read them as "
        "**exploratory signals**: a confirmatory analysis would pre-register a smaller family "
        "(e.g. FPR per attribute) and apply Holm-Bonferroni or Benjamini-Hochberg. Note also that "
        "every group here was reported, not selected — no group was dropped for being "
        "unflattering.\n"
    )

    parts.append("\n## Exploratory view — full validation window\n")
    parts.append(
        f"{secondary_frauds:,} frauds / {secondary_rows:,} rows. **Exploratory only**: "
        "the calibrator was fitted on "
        "the first half of this window (`val_cal`), so calibration gap and Brier would "
        "be in-sample for half the rows and are therefore excluded here; the policy "
        "capacity was fitted on `val_thr`.\n"
    )
    parts.append(_md_table(_rounded(secondary, cols), cols))

    parts.append("\n## Proxy check — do model features encode the audit attributes?\n")
    parts.append(
        "Ordinal attributes (age/income bands) use Spearman rho; nominal attributes use "
        "eta-squared for numeric features and Cramer's V for the categorical ones. "
        "Pre-registered thresholds: |rho| >= 0.30, eta^2 >= 0.14, V >= 0.30. A strong "
        "correlation means the feature is a *proxy* for the attribute, which is how a "
        "behavioural difference can look like a demographic disparity.\n"
    )
    strong = proxy[proxy["strong"]]
    if strong.empty:
        parts.append("No feature/attribute pair crossed the pre-registered threshold.\n")
    else:
        parts.append(
            _md_table(
                strong[["attribute", "feature", "statistic", "value", "threshold"]],
                ["attribute", "feature", "statistic", "value", "threshold"],
            )
        )
    parts.append(
        f"\n({len(strong)} of {len(proxy)} feature/attribute pairs are strong; the full "
        "table is in `results/fairness_proxy_{profile}.csv`.)\n".format(profile=profile)
    )

    parts.append(
        f"""
## What I would examine with real data

- **Group definitions:** age/income bands are a modelling choice; real audits must
  pre-register them (and intersectional groups, e.g. age x region) before looking at
  outcomes, or the bands become a researcher degree of freedom.
- **Legal frameworks:** in lending/credit this would be evaluated against ECOA/Reg B
  (US), FCA fair-treatment rules (UK) or RBI/DPDP guidance (India) for the relevant
  jurisdiction; disparate-impact analysis needs a legally defined protected class and
  a business necessity defence, neither of which is available here.
- **Better statistics:** with more positives, per-group calibration curves, equalised
  odds tests and Wald CIs on group differences; here the {primary_frauds} primary frauds force
  wide intervals.
- **Post-processing options:** group-specific thresholds, reject-option
  classification, or cost-sensitive thresholds — each trades overall cost for
  parity and must be evaluated on realised cost, not accuracy.
- **Monitoring:** track group flag rates and calibration drift per group once the
  system runs, and alert when a group's FPR departs from its baseline.

*This audit is not a compliance claim.*
"""
    )
    return "\n".join(parts)


def render_gallery_markdown(profile: str, cases: list[dict], policy_line: str) -> str:
    """Render ``docs/explanation_gallery.md`` from selected cases."""
    lines = [
        "# Local explanation gallery (Part 10D)",
        "",
        "Six validation cases selected by **pre-registered rules** (not by looks): each "
        "row below records the pool it was drawn from, so a reader can see how "
        "representative it is. SHAP here explains the raw model margin; the reported "
        "probability is the calibrated one (the honesty rule in `docs/methodology.md`).",
        "",
        "Caveat: since the Part 02b cold-start fix, generated transactions are gated on the "
        "customer's `account_open_date`, so no row has `history_days < 0` (the features builder "
        "asserts this). The cold-start case is defined by `0 <= history_days < 2` — a genuinely "
        "brand-new account, not the pre-fix generation defect (see `docs/what_went_wrong.md`).",
        "",
        f"**Policy:** {policy_line}",
        "",
        _md_table(
            pd.DataFrame(cases),
            [
                "case",
                "transaction_id",
                "is_fraud",
                "fraud_scenario",
                "stealth",
                "amount",
                "p",
                "action",
                "pool_size",
            ],
            [
                "case",
                "transaction_id",
                "fraud",
                "scenario",
                "stealth",
                "amount (Rs)",
                "p_calibrated",
                "action",
                "pool",
            ],
            formats={"p": ".6f", "amount": ",.2f", "stealth": ".4f"},
            na="n/a",
        ),
        "",
        "## Selection rules",
        "",
    ]
    lines += [f"- **{c['case']}** — {c['rule']}" for c in cases]
    lines.append("")
    for case in cases:
        lines.append(f"## {case['case']}")
        lines.append("")
        lines.append(f"`{case['transaction_id']}` — {case['commentary']}")
        lines.append("")
        raisers = "; ".join(case["top_raisers"]) if case["top_raisers"] else "none"
        lowers = "; ".join(case["top_lowers"]) if case["top_lowers"] else "none"
        lines.append(f"- **Raises risk:** {raisers}")
        lines.append(f"- **Mitigating:** {lowers}")
        lines.append(f"- **Figure:** `{case['figure']}`")
        lines.append("")
    return "\n".join(lines)
