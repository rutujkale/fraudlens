"""Page 3 — Model Performance & Drift (Part 14).

Everything here is read from ``results/`` and the precomputed artifacts; the test
window is never recomputed. Wording of the headline numbers is fixed by Part 13/14
review: 83.23% with an INR CI, and the exact drift sentence below.

The engineering strip (latency, parity) is rendered **directly from**
``fraudlens.reporting.render.build_context`` via ``components.context`` so the app
and the rendered headline doc cannot drift.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from components import charts, context, limitations, theme  # noqa: E402

from fraudlens.paths import reports_dir, results_dir  # noqa: E402

st.set_page_config(page_title="Model Performance — FraudLens", layout="wide")
context.sidebar()
st.title("Model Performance & Drift")

art = context.artifacts()
results = results_dir()
figures = reports_dir() / "figures"
final = json.loads((results / "final_results.json").read_text(encoding="utf-8"))
val_csv = results / f"val_comparison_{art.manifest['profile']}.csv"
val_cmp = pd.read_csv(val_csv) if val_csv.exists() else pd.DataFrame()

theme.chips(
    [
        (f"profile {final['meta']['profile']}", "steel"),
        (str(art.manifest["model_version"]), "gold"),
        (str(art.manifest["policy_version"] or "no policy"), ""),
        (f"seed {final['meta']['seed']}", ""),
        (f"features {final['meta']['feature_version']}", ""),
    ]
)

st.subheader("Pre-registered outcomes (Part 12, test window)")
outcomes = final["pre_registered_outcomes"]
h1 = outcomes["H1_model"]
h2 = outcomes["H2_decision"]
cards = st.columns(4)


def _hold_tone(holds: bool) -> str:
    return "ok" if holds else "warn"


with cards[0], st.container(border=True):
    st.markdown("**H1 · Model ordering**")
    theme.chips([("holds" if h1["holds"] else "fails", _hold_tone(h1["holds"]))])
    st.caption(h1["wording"])
    st.markdown(
        f"XGB PR-AUC CI `{h1['xgb_pr_auc'][0]:.4f}`–`{h1['xgb_pr_auc'][1]:.4f}` "
        f"vs Rules `{h1['rules_pr_auc'][0]:.4f}`–`{h1['rules_pr_auc'][1]:.4f}`."
    )
with cards[1], st.container(border=True):
    st.markdown("**H2 · Decision policy**")
    theme.chips([("holds" if h2["holds"] else "fails", _hold_tone(h2["holds"]))])
    st.caption(h2["wording"])
    ci = h2["reduction_ci"]
    st.markdown(
        f"Reduction CI `{ci['lo']:,.0f}`–`{ci['hi']:,.0f}` INR "
        f"(point `{ci['point']:,.0f}`) · excludes zero: `{ci['excludes_zero']}`."
    )
with cards[2], st.container(border=True):
    st.markdown("**H3 · Unseen fraud**")
    theme.chips([("holds" if outcomes["H3_unseen"]["holds"] else "fails",
                  _hold_tone(outcomes["H3_unseen"]["holds"]))])
    st.caption(outcomes["H3_unseen"]["detail"])
with cards[3], st.container(border=True):
    st.markdown("**H4 · Engineering**")
    theme.chips([("holds" if outcomes["H4_engineering"]["holds"] else "fails",
                  _hold_tone(outcomes["H4_engineering"]["holds"]))])
    st.caption(outcomes["H4_engineering"]["detail"])

st.subheader("Engineering — serving latency & parity")
with st.container(border=True):
    eng_lat, eng_text = st.columns([1, 3])
    eng_lat.metric("API latency", context.engineering_latency())
    eng_text.caption(context.engineering_latency_note())
    st.caption(context.engineering_parity())

per_model = final["drift"]["per_model_pr_auc"]
xgb = per_model["xgb"]
test_pr_auc = xgb["test_pr_auc"]
val_pr_auc = xgb["val_pr_auc"]
reduction = final["decision"]["meta"]["amount_aware_capacity_pct_reduction_vs_f1"]
ci = final["decision"]["meta"]["reduction_ci"]

st.subheader("Frozen test-window headline")
st.metric("XGB PR-AUC (test)", f"{test_pr_auc:.4f}", f"val {val_pr_auc:.4f}")
st.markdown(
    f"- **Cost reduction** (amount-aware capacity policy vs F1-optimal, test window): "
    f"**{reduction:.2f}%**, with a customer-cluster-bootstrap INR CI of "
    f"{ci['lo']:,.0f}–{ci['hi']:,.0f} (point {ci['point']:,.0f} INR)."
)
st.info(f"Drift: no degradation observed ({val_pr_auc:.4f} val → {test_pr_auc:.4f} test)")

st.subheader("Model comparison (test window)")
comparison = pd.DataFrame(final["model_comparison"])
comparison["val_pr_auc"] = comparison["model"].map(
    {row["model"]: row["pr_auc"] for _, row in val_cmp.iterrows()}
)
st.dataframe(
    comparison[
        [
            "model", "pr_auc", "pr_auc_lo", "pr_auc_hi", "val_pr_auc", "roc_auc",
            "precision_at_thr", "recall_at_thr", "fpr_at_thr",
        ]
    ].rename(
        columns={
            "pr_auc": "test_PR_AUC", "pr_auc_lo": "test_PR_AUC_lo", "pr_auc_hi": "test_PR_AUC_hi",
            "val_pr_auc": "val_PR_AUC", "roc_auc": "test_ROC_AUC",
            "precision_at_thr": "precision@F1thr", "recall_at_thr": "recall@F1thr",
            "fpr_at_thr": "FPR@F1thr",
        }
    ),
    use_container_width=True,
)
st.caption("Bootstrap CIs resample **customers** (cluster bootstrap), not transactions.")

tab_pr, tab_roc, tab_cal, tab_k = st.tabs(
    ["PR (validation)", "ROC (validation)", "Calibration", "Recall@K"]
)
with tab_pr:
    for model in art.curves["models"]:
        st.plotly_chart(charts.pr_curve(art.curves, model), use_container_width=True)
with tab_roc:
    for model in art.curves["models"]:
        st.plotly_chart(charts.roc_curve(art.curves, model), use_container_width=True)
with tab_cal:
    cal = final["calibration"]["xgb"]
    st.caption(
        f"XGB test Brier raw {cal['brier_raw']:.5f} → calibrated {cal['brier_calibrated']:.5f}; "
        f"ECE raw {cal['ece_raw']:.5f} → calibrated {cal['ece_calibrated']:.5f}."
    )
    st.plotly_chart(
        charts.calibration_curve(art.curves["models"]["xgb"]["calibration"], "validation bins"),
        use_container_width=True,
    )
    test_bins = art.curves.get("test_calibration_bins", {}).get("xgb", {}).get("bins")
    if test_bins:
        st.plotly_chart(
            charts.calibration_curve(test_bins, "test bins (frozen)"), use_container_width=True
        )
with tab_k:
    rows = []
    for model, entry in art.curves["models"].items():
        for k, value in entry["recall_at_k"].items():
            rows.append({"model": model, "top_k": k, "val_recall_at_k": value})
    st.dataframe(pd.DataFrame(rows).pivot(index="top_k", columns="model", values="val_recall_at_k"))

st.subheader("Confusion matrix")
frozen_thr = float(final["decision"]["meta"]["cost_optimal_threshold_capacity"])
threshold = st.slider(
    "Review threshold (calibrated probability, validation sample)",
    min_value=0.0, max_value=1.0, value=frozen_thr, step=0.005,
)
sample = art.scored_sample
y_true = sample["is_fraud"].to_numpy().astype(int)
prob = sample["fraud_probability"].to_numpy(dtype=float)
flagged = prob >= threshold
tp = int((flagged & (y_true == 1)).sum())
fp = int((flagged & (y_true == 0)).sum())
fn = int((~flagged & (y_true == 1)).sum())
tn = int((~flagged & (y_true == 0)).sum())
c1, c2 = st.columns(2)
c1.markdown("**Validation sample (selectable)**")
c1.dataframe(
    pd.DataFrame(
        {"fraud (actual)": [tp, fn], "legit (actual)": [fp, tn]},
        index=["flagged", "not flagged"],
    ),
    use_container_width=True,
)
composition = final["composition"]
policies = pd.DataFrame(final["decision"]["policies"]).set_index("policy")
adopted = policies.loc["amount_aware_capacity"]
positives = composition["frauds"]
negatives = composition["rows"] - positives
t_tp = round(adopted["recall"] * positives)
t_fp = round(adopted["fpr"] * negatives)
c2.markdown("**Test window (frozen, adopted policy)**")
c2.dataframe(
    pd.DataFrame(
        {"fraud (actual)": [t_tp, positives - t_tp], "legit (actual)": [t_fp, negatives - t_fp]},
        index=["flagged", "not flagged"],
    ),
    use_container_width=True,
)
c2.caption("Frozen point from `results/final_results.json`; the test window is never recomputed.")

st.subheader("Drift")
drift = final["drift"]
top = drift["psi_excl_tenure_top"][0]
st.markdown(
    f"- Validation PSI is computed per month against train (max feature: "
    f"**{top['feature']}** = {top['psi']:.2f}, {top['flag']})."
)
st.caption(
    "Tenure-style features dominate train↔test PSI "
    f"(top: `{drift['psi_train_test_top'][0]['feature']}` "
    f"= {drift['psi_train_test_top'][0]['psi']:.2f}); test-month PSI is scheduled for Part 15."
)
st.plotly_chart(charts.psi_heatmap(art.psi_by_month), use_container_width=True)
hist = art.curves.get("val_score_hist")
if hist:
    st.plotly_chart(
        charts.score_histogram(hist["bin_edges"], hist["counts"]), use_container_width=True
    )
st.caption(
    f"Validation vs test score PSI = {drift['score_psi_val_test']:.5f} "
    f"(a score-distribution shift would show here; validation histogram shown above)."
)
monthly = (
    sample.assign(month=pd.to_datetime(sample["timestamp"]).dt.to_period("M").astype(str))
    .groupby("month")["is_fraud"]
    .agg(["mean", "size"])
    .rename(columns={"mean": "fraud_rate", "size": "rows"})
)
st.dataframe(monthly, use_container_width=True)
post = drift["post_switch"]
st.caption(
    f"Fraud prevalence shifts after the simulated {drift['switch_date']} regime switch: "
    f"{post['frauds_post']} of {post['rows_post']} post-switch rows are fraud "
    f"({post['post_share_of_frauds']:.2%} of all frauds)."
)

st.subheader("Unseen-fraud experiment (Part 09) and customer holdout")
decision = json.loads(
    (results / "exp_09_unseen_decision_unseen_lab_structuring.json").read_text(encoding="utf-8")
)
st.markdown(
    f"- Isolation Forest **{'adopted' if decision['adopted'] else 'not adopted'}**: "
    f"{decision['reason']}"
)
holdout_csv = results / f"holdout_{art.manifest['profile']}.csv"
if holdout_csv.exists():
    st.dataframe(pd.read_csv(holdout_csv), use_container_width=True)
    st.caption("Setting A = seen customers, B = unseen customers (customer holdout).")

st.subheader("Fairness (Part 10)")
fairness_csv = results / f"fairness_groups_primary_{art.manifest['profile']}.csv"
if fairness_csv.exists():
    fairness = pd.read_csv(fairness_csv)
    st.dataframe(
        fairness[
            [
                "attribute", "group", "n", "prevalence", "flag_rate_any", "fpr_any",
                "recall", "precision", "suppressed_reason",
            ]
        ],
        use_container_width=True,
    )
st.caption(
    "Groups with fewer than 1,000 rows or 30 positives are suppressed (blank metrics). "
    "This is an audit of simulator archetypes, not a protected-attribute compliance claim."
)

with st.expander("Frozen test-window figures (from Part 12)"):
    for name in ["final_pr_curves", "final_roc", "final_calibration", "final_cost_bars"]:
        path = figures / f"{name}.png"
        if path.exists():
            st.image(str(path), caption=name, use_container_width=True)

limitations.render()
