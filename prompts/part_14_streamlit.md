# Part 14 — Streamlit Analyst App

## Goal
A thin, three-page analyst tool on top of the API and precomputed result files that demonstrates the decision system in a two-minute demo: score a transaction with an explanation, work a review queue under capacity and cost assumptions, and inspect model performance, calibration and drift. It is **not** a model playground: no feature or model logic lives here.

## Prerequisites
Parts 10, 12, 13 complete. The API runs locally (memory backend is fine). Final result files exist in `results/` and `reports/`.

## Design rules
- The app calls the API for scoring (`/predict`, `/predict/batch`). Offline views read precomputed artifacts from `results/` and `artifacts/` through helper functions in `app/components/` — nothing is recomputed on page load beyond cheap pandas operations; use `st.cache_data`.
- Cost/policy recomputation (Page 2 sliders) imports `evaluation/cost_model.py` and `risk/policy.py` from the package — the same code as Part 07 — applied to a precomputed scored sample.
- Every number on screen comes from files or the API. Show the model version, policy version and the cost assumptions in a sidebar on every page. Include a visible banner: *"Synthetic data demo — not a production fraud system."*
- A fraud analyst is the persona: concise, sortable, explainable. Avoid decorative widgets.

## Precomputation (`scripts/make_app_artifacts.py`)
Produce files under `artifacts/app/` (gitignored except tiny samples):
- `scored_sample.parquet`: a stratified sample (e.g. 100k transactions, all frauds kept) from the **validation window** with raw score, calibrated probability, amount, customer archetype, top reason text (precomputed SHAP for the top-k only, e.g. the highest-risk 5k rows), true label (for the demo's "what actually happened" column only, hidden by default), timestamp.
- `curves.json`: PR, ROC, calibration, recall@K curve points per model from `final_results.json` (test) and validation.
- `psi_by_month.csv` and score-distribution histograms (from Part 15 once available; until then compute PSI with `monitoring/drift.py` for validation vs test).
- `presets.json`: 4 prefilled example transactions picked from real data with fixed seed (normal purchase; suspicious burst; hard-negative travel; structuring-like sequence) including `customer_id` so history exists.
Add a flag to use the `tiny` profile so the app can run in CI/smoke mode.

## Pages (`app/streamlit_app.py` + `app/pages/`)

### Page 1 — Transaction Analyzer
- Form: customer (selectbox or text), amount, timestamp, merchant category, device, location, transaction type, channel, recipient. Preset buttons fill the form from `presets.json`.
- On submit → `POST /predict` with `include_explanation=true`.
- Output: calibrated probability, risk score + lift over base rate, risk level with color, **recommended action**, and why: top positive drivers as sentences, mitigating factors, and a horizontal bar chart of SHAP contributions captioned "Contribution to the model's log-odds (not points on the risk score)". Latency shown in small text with the component breakdown.
- A compact "customer context" panel: the customer's typical amount, usual hours, known devices, last 5 transactions (from the history provider via an API endpoint `GET /customer/{id}/summary` — add it to the API if missing, with a test).

### Page 2 — Review Queue & Cost Optimization
- Top: KPIs (transactions in sample, flagged under current policy, review volume/day vs capacity, expected loss, realized loss avoided).
- **Queue table:** sortable/filterable, columns: transaction id, timestamp, amount, calibrated probability, risk score, action, archetype, top reason; default sort by expected saving; row click shows the explanation. Download CSV.
- **Batch upload:** CSV upload → `/predict/batch/csv` → ranked table + summary of review workload; show invalid-row errors clearly.
- **Cost controls (sidebar or expander):** sliders for fraud loss rate α, fixed fee, review cost, friction cost, step-up parameters, review catch rate, and review capacity per day. On change, recompute: amount-aware policy and the best global threshold (using Part 07 code), then display optimal threshold, realized cost vs F1-optimal, fraud captured, review volume/day, and the **expected-loss curve** (Plotly) with the F1-optimal and cost-optimal points. Provide a "reset to frozen assumptions" button and a table comparing the four preset cost scenarios.

### Page 3 — Model Performance & Drift
- Model comparison table (Rules / LogReg / XGBoost [+ anomaly if adopted]) from `final_results.json` with CIs; PR curve, ROC curve, calibration curve (before/after), recall@top-K, confusion matrix at a selectable threshold (default = frozen).
- Unseen-fraud experiment results (from Part 09) and customer-holdout results (Setting A vs B).
- Drift: monthly PSI heatmap for key features, highest-drift feature callout, score-distribution comparison (validation vs test), fraud prevalence over time.
- Fairness table (from Part 10) with the caveat text.
- A "Limitations" expander summarizing `docs/limitations.md`.

## Quality
- Layout works on a laptop-width screen; charts with Plotly; consistent color for actions (LOW/MEDIUM/HIGH/CRITICAL).
- Graceful errors: API down ⇒ clear message with the command to start it; missing artifacts ⇒ message naming the script to run.
- `tests/test_app_smoke.py`: import every page module; run page functions with Streamlit's `AppTest` (`streamlit.testing.v1`) in smoke mode against the `tiny` artifacts; assert no exceptions and key elements exist.
- `docs/screenshots/` placeholder and a `docs/demo_script.md` with the two-minute demo flow (problem → score a suspicious transaction → review queue + capacity slider → cost curve → model/drift page → three headline results). You will record screenshots/GIF yourself.

## Acceptance criteria
```bash
python scripts/make_app_artifacts.py --profile main
HISTORY_BACKEND=memory python scripts/run_api.py &
streamlit run app/streamlit_app.py
pytest tests/test_app_smoke.py -q
```
Progress file: pages implemented, files read by each page, how the recompute-on-slider path reuses the package code, smoke-test results, known gaps.

## Do not
Put feature, scoring, or cost logic in the app. Show test-window labels by default on pages used for demos of "live" scoring. Hard-code numbers or metrics. Add more pages than the three above.
