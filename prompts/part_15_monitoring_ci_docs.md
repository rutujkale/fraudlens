# Part 15 — Monitoring, CI, Documentation & Interview Notes

## Goal
Finish the project as a reproducible, maintainable, honestly documented portfolio artifact: a small monitoring layer, a CI pipeline, a README whose numbers are generated from results, a model card, methodology, limitations, the "what went wrong" write-up, and interview notes.

## Prerequisites
Parts 01–14 complete and committed.

## Tasks

### 15A — Monitoring (`monitoring/drift.py`, `monitoring/metrics.py`, `scripts/run_monitoring.py`)
- PSI (already implemented in Part 8E; extend/test): quantile bins from the reference (training window), epsilon smoothing, handles NaN/constant features, categorical PSI. Alert levels: <0.1 stable, 0.1–0.25 moderate, >0.25 major (state these are conventional rules of thumb).
- Monthly (or weekly) monitoring over the whole timeline: feature PSI for the key features (amount_ratio_to_30d_mean, tx_count_1h, is_new_device rate, distance_from_home_km, amount_sum_24h, plus all top-15 SHAP features), **score PSI** (calibrated probability), fraud prevalence (label drift, using only labels matured by each date), mean predicted probability vs observed rate (calibration drift), delayed-label performance (PR-AUC, precision/recall at the frozen policy, FPR) computed per period with the label-delay rule.
- `monitoring/metrics.py`: `build_dashboard_metrics(...) -> dict/DataFrame` (also used by Streamlit Page 3).
- Write results to `results/monitoring_<profile>.csv` and optionally to `monitoring.metrics` in Postgres. Generate figures: PSI heatmap, score distribution by period, performance over time with the drift date marked.
- A **retraining trigger** function with explicit, documented rules (e.g. score PSI > 0.25 for 2 consecutive periods, or recall@K below a floor) and a short `docs/retraining_plan.md` linking to Experiment 8E's findings.
- Tests: PSI = 0 for identical samples, grows with shift, stable under constant features; trigger logic with synthetic series; monitoring respects label maturity.

### 15B — Continuous integration (`.github/workflows/ci.yml`)
Triggers: push and pull request. Steps: set up Python 3.11, install deps (cache pip), `ruff check` + `ruff format --check`, start a PostgreSQL service container, run `pytest -m "not slow"` (including `db` tests against the service), then a **smoke pipeline on the `tiny` profile**: generate data → validate → build features → leakage tests → quick XGBoost train → API smoke test via TestClient → `separability_gate` on tiny is *not* required. Upload test reports as artifacts. Total CI time target: under ~10 minutes. Add a README badge. Add `make ci-local` that reproduces the same steps locally.

### 15C — Reproducibility
- `Makefile` targets completed: `data`, `features`, `db`, `train`, `calibrate`, `policy`, `experiments`, `external`, `freeze`, `final`, `api`, `app`, `all-dev` (everything on `dev` profile, no external/test). A fresh clone must be able to reproduce the `dev` pipeline with documented commands.
- `docs/reproduce.md`: exact command sequence, expected runtimes, expected rough outputs, and how to regenerate the `main` profile. Seeds documented.

### 15D — Documentation (generated numbers only)
1. `README.md` (top-down, scannable in two minutes): What is it → Why it matters (rare fraud, hard negatives, evolving patterns, limited analysts) → What I built (architecture diagram, e.g. Mermaid) → **What I found** (the four headline results from `reports/headline_numbers.md`, rendered via `scripts/render_readme.py` which fills `{{key}}` placeholders from `results/final_results.json` and `results/latency.json`; fail if a placeholder has no value) → Demo (GIF/screenshots placeholders) → Quickstart → Repo map → Limitations → License. No number appears in the README unless it is a placeholder filled from results.
2. `docs/methodology.md`: simulator assumptions, feature definitions & leakage rules, validation strategy (temporal split, embargo, label delay, cluster bootstrap), imbalance, calibration, cost model (with the closed-form threshold), capacity allocation, anomaly experiment design, SHAP honesty note, drift & retraining, external validation.
3. `docs/model_card.md`: intended use (portfolio/research), not intended for real approvals, data (synthetic + IEEE-CIS secondary validation), target, evaluation (temporal), metrics, calibration, policy and cost assumptions, subgroup analysis summary, known limitations (synthetic assumptions, simulation bias, sim-to-real gap, uid heuristic on real data, no network signals), ethical considerations, model/feature/policy versions and artifact hashes.
4. `docs/limitations.md`: candid list, including what a real institution would have (richer device/network signals, chargebacks arriving late, investigator feedback loops, model-risk governance) and what you would change before real deployment.
5. `docs/what_went_wrong.md`: consolidate every logged incident into a polished, honest narrative with numbers: leakage incident, simulator shortcut, threshold mistake (F1-optimal review volume vs capacity), isolation forest first attempt, any API parity bug, anything else. Each entry: what happened → how detected → effect (numbers) → fix → test added.
6. `docs/trade_offs.md`: precision vs recall, fraud loss vs friction, review capacity vs detection, complexity vs explainability, synthetic realism vs reproducibility, benchmark transfer vs feature compatibility, latency vs feature richness — each with the evidence from this project.
7. `docs/data_dictionary.md` (raw tables) and `docs/data_dictionary_features.md` (generated from the registry); `docs/experiments/` index.
8. `LICENSE` (MIT) and a note that the IEEE-CIS data is not redistributed.

### 15E — Interview notes (`docs/interview_notes.md`)
For each of these questions write a 5–8 sentence answer **grounded in this project's actual results** (reference the file/key for each number), plus one follow-up question an interviewer might ask and its answer:
1. Why is fraud detection different from ordinary classification?
2. Why PR-AUC instead of accuracy? 3. Why was a random split inappropriate, and what is the embargo for?
4. How did you prevent and test for leakage (including the canary test)?
5. How did you build hard negatives, and why do they matter?
6. Why XGBoost; why keep Logistic Regression and rules?
7. What did the imbalance experiment show about calibration? 8. What does calibration do, and why is a raw probability not a trustworthy risk score?
9. How did you pick the threshold? Derive the amount-dependent threshold.
10. Why does review capacity matter, and how was it modeled?
11. What happens under concept drift; covariate vs concept drift; retraining strategies compared?
12. Did the Isolation Forest help on unseen fraud? Why or why not?
13. What do SHAP values mean and not mean? How do importance methods differ?
14. What did the fairness audit find, and what are its limits on synthetic data?
15. What is the sim-to-real gap you measured?
16. How is training/serving skew prevented here?
17. What did not work? 18. What are the limitations of synthetic fraud data?
19. What would you change before a real deployment? 20. How would you retrain and monitor in production?
Also include a **60-second and a 2-minute project pitch** using only verified numbers, and 3 resume-bullet candidates with source keys.

### 15F — Final audit (`scripts/final_audit.py`)
Checks and prints PASS/FAIL: all tests pass; ruff clean; `results/freeze.json` hash verification; `test_access_log.csv` shows exactly one final evaluation (or reasons); README has no unresolved placeholders; every number in `reports/headline_numbers.md` exists in `results/`; no `data/` or `.env` files tracked by git; no label columns in feature columns; leakage and parity tests present; all Definition-of-Done items below marked.

## Definition of Done (tick in `docs/progress/part_15.md` with evidence links)
- [ ] Simulator config-driven; prevalence and scenarios documented; hard negatives and drift present
- [ ] PostgreSQL schema reproducible, loader automated, indexes documented
- [ ] Point-in-time features documented; leakage suite (incl. truncation invariance + canary) passes
- [ ] Rules, LogReg, XGBoost; temporal test set with embargo; PR-AUC and recall@top-K with CIs
- [ ] Calibration (before/after); cost model; capacity; expected-loss curve; F1 vs cost-optimal; sensitivity
- [ ] Isolation Forest unseen-fraud experiment with honest outcome
- [ ] SHAP global + local; fairness audit; model card; limitations
- [ ] ML-understanding experiments 8A–8E documented
- [ ] External validation and sim-to-real gap documented
- [ ] FastAPI with parity test and latency benchmark; Streamlit three pages; batch scoring works
- [ ] Unit, data, leakage, API tests; CI green
- [ ] README with generated headline numbers; architecture diagram; screenshots/GIF; what-went-wrong

## Acceptance criteria
```bash
make ci-local
python scripts/run_monitoring.py --profile main
python scripts/render_readme.py
python scripts/final_audit.py
```
Progress file: audit output, remaining TODOs for you (screenshots, demo video, resume wording).

## Do not
Write a README number by hand. Hide or soften unfavorable findings. Claim production readiness. Leave placeholders unresolved. Add new modeling changes in this part.
