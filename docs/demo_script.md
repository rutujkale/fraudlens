# FraudLens — two-minute demo script

Persona: a fraud analyst. Every number below appears on screen and comes from
`results/`, the precomputed `artifacts/app/<profile>/` files, or the scoring API.

## 0. Start (15 s)

```bash
make db-up                       # PostgreSQL (already loaded)
make app-artifacts PROFILE=dev   # ~45 s, writes artifacts/app/dev/
make api                         # HISTORY_BACKEND=memory by default
make app                         # streamlit run app/streamlit_app.py
```

Banner on every page: *"Synthetic data demo — not a production fraud system."*
Sidebar: profile, model version, policy version, cost assumptions, capacity note.

## 1. The problem (10 s)

Fraud is rare (~0.77% of transactions) and a wrong decision costs money. FraudLens
picks **actions** (approve / step-up / review / block) by expected cost, not a
probability cutoff, and caps review work at analyst capacity.

## 2. Transaction Analyzer (35 s)

- Pick the **suspicious burst** preset → Score.
- Read the decision: risk score, calibrated probability, lift over base rate,
  recommended action and risk level.
- The "why" panel: top risk-raising sentences and mitigating factors.
- SHAP bar, captioned *"Contribution to the model's log-odds (not points on the risk score)"*.
- Customer context: prior transactions, typical amount, usual hours, known devices,
  last five transactions (strictly before this one).

## 3. Review Queue & Cost Optimization (40 s)

- Default view hides labels: it shows what an analyst sees, no ground truth.
- KPIs: sample size, flagged, review/day vs capacity, expected loss, expected loss avoided.
- Drag the **review capacity** slider down → review/day is re-capped and candidates
  demote to their next-best action (expected cost, via Part 07 policy code).
- Drag the **fraud loss rate α** and **review cost** sliders → the amount-aware
  threshold and the cost-optimal global threshold move; the expected-loss curve redraws.
- "Reset to frozen assumptions" restores the Part 12 policy.
- Toggle **"Show labelled outcomes"** to reveal the ground-truth column and the
  realized-cost KPI for the demo only.

## 4. Model Performance & Drift (30 s)

- XGB test PR-AUC and the drift line: *no degradation observed (0.9291 val → 0.9619 test)*.
- Cost reduction 83.23% vs F1-optimal with an INR CI (customer cluster bootstrap).
- Validation PR/ROC and calibration tabs; calibration "before/after" Brier and ECE.
- Confusion matrix: move the threshold on the validation sample, next to the frozen
  test-window point.
- Drift: monthly validation PSI heatmap, highest-drift callout, score-distribution PSI.
- Part 09 unseen-fraud decision (not adopted) and the customer-holdout A vs B table.
- Fairness table with the suppression caveat, and the Limitations expander.

## 5. Three headline results

1. **XGB beats rules decisively** — PR-AUC CI lower bound above the rules CI upper bound.
2. **Cost-based, capacity-aware policy** cuts realized cost by **83.23%** vs F1-optimal
   (INR CI 361,480–1,281,632, excludes zero) on the frozen test window.
3. **The system degrades gracefully** — no PR-AUC drop from validation to the test
   window (0.9291 → 0.9619), even though fraud prevalence and mix shift after the
   simulated 2026-04-16 regime switch.

Screenshots/GIF: record yourself into `docs/screenshots/` (see its README).
