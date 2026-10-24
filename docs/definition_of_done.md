# Definition of Done

Completion checklist for the end-to-end build. Every item links to the evidence in this repository.

- [x] Simulator config-driven; prevalence and scenarios documented; hard negatives and drift present — `config/config.yaml`, `docs/methodology.md`, `docs/what_went_wrong.md`
- [x] PostgreSQL schema reproducible, loader automated, indexes documented — `sql/`, `docs/methodology.md` (index evidence), `reports/db_explain_dev.txt`
- [x] Point-in-time features documented; leakage suite (incl. truncation invariance + canary) passes — `docs/methodology.md`, `tests/test_leakage.py`
- [x] Rules, LogReg, XGBoost; temporal test set with embargo; PR-AUC and recall@top-K with CIs — `results/final_results.json:model_comparison`
- [x] Calibration (before/after); cost model; capacity; expected-loss curve; F1 vs cost-optimal; sensitivity — `results/final_results.json:decision`, `results/freeze.json`
- [x] Isolation Forest unseen-fraud experiment with honest outcome (null) — `docs/experiments/09_unseen_fraud.md`
- [x] SHAP global + local; fairness audit; model card; limitations — `docs/explanation_gallery.md`, `docs/fairness_audit.md`, `docs/model_card.md`, `docs/limitations.md`
- [x] ML-understanding experiments 8A–8E documented — `docs/experiments/`
- [x] External validation and sim-to-real gap documented — `docs/external_validation.md`, `results/sim_to_real.json`
- [x] FastAPI with parity test and latency benchmark; Streamlit three pages; batch scoring works — `docs/api.md`, `docs/app.md`, `results/latency.json`
- [x] Unit, data, leakage, API tests present; CI green — local `scripts/final_audit.py` and `pytest -m db` pass (current counts in `README.md` / `docs/reproduce.md`); **CI green observed**: GitHub Actions run `39033533336` on `60d4570` → `test` job success in 2m19s (Node-20 deprecation annotation only, non-fatal)
- [x] README with generated headline numbers; architecture diagram; screenshots/GIF; what-went-wrong — README/diagram/what-went-wrong done; **screenshots captured** (5 PNGs in `docs/screenshots/live/`; static screenshots, no GIF)
