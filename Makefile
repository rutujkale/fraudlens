# FraudLens development targets.
# `PY` points at the venv interpreter so recipes work without activation.
# Override on the command line if the venv lives elsewhere:
#   make setup PY="py -3.12"
PY ?= .venv/Scripts/python

.PHONY: setup db-up db-down lint format test test-fast simulate validate-data eda separability-gate db-load build-features train evaluate-val calibrate optimize-policy drift-data experiment-8a experiment-8b experiment-8c experiment-8d experiment-8e shap fairness explain external external-report sim-to-real api benchmark-latency app-artifacts app data features db policy freeze final experiments monitor render all-dev ci-local

## setup: create venv (if missing) and install everything.
setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e ".[dev]"

## db-up / db-down: start/stop the PostgreSQL container (Part 01).
db-up:
	docker compose up -d
	docker compose ps

db-down:
	docker compose down

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

test:
	$(PY) -m pytest

## test-fast: everything except tests marked `db` or `slow`.
test-fast:
	$(PY) -m pytest -m "not db and not slow"

## simulate: generate raw data for a profile (PROFILE=dev by default).
PROFILE ?= dev
simulate:
	$(PY) scripts/generate_data.py --profile $(PROFILE)

## validate-data: schema/sanity checks on raw data (exit != 0 on failure).
validate-data:
	$(PY) scripts/validate_data.py --profile $(PROFILE)

## eda: figures + generated docs/eda_summary.md + docs/data_dictionary.md.
eda:
	$(PY) scripts/eda.py --profile $(PROFILE)

## separability-gate: prove the data is not trivially solvable (exit != 0 on FAIL).
separability-gate:
	$(PY) scripts/separability_gate.py --profile $(PROFILE)

## db-load: load raw data into PostgreSQL (add RESET=--reset to drop schemas first).
RESET ?=
db-load:
	$(PY) scripts/load_db.py --profile $(PROFILE) $(RESET)

## build-features: point-in-time features -> data/features/<profile>/features_v1.parquet.
build-features:
	$(PY) scripts/build_features.py --profile $(PROFILE)

## train: fit rules/logreg/xgb, select on the validation window (MODEL=rules|logreg|xgb|all).
MODEL ?= all
train:
	$(PY) scripts/train.py --profile $(PROFILE) --model $(MODEL)

## evaluate-val: validation comparison table, per-scenario recall, holdout, figures.
evaluate-val:
	$(PY) scripts/evaluate_val.py --profile $(PROFILE)

## calibrate: fit sigmoid/isotonic on val_cal, select on val_thr (SCENARIO=balanced).
calibrate:
	$(PY) scripts/calibrate.py --profile $(PROFILE)

## optimize-policy: cost-based decision policy + capacity + sensitivity (+ SCENARIO).
SCENARIO ?= balanced
optimize-policy:
	$(PY) scripts/optimize_policy.py --profile $(PROFILE) --scenario $(SCENARIO)

# TODO(part 08): experiments     - imbalance/ablation/drift experiments
## drift-data: generate + featurise the drift_lab profile (needed once for 8E).
drift-data:
	$(PY) scripts/generate_data.py --profile drift_lab
	$(PY) scripts/build_features.py --profile drift_lab

## experiment-8a..8d: Part 08 ML-understanding experiments on $(PROFILE) (default dev).
experiment-8a:
	$(PY) scripts/run_experiment_8A.py --profile $(PROFILE)

experiment-8b:
	$(PY) scripts/run_experiment_8B.py --profile $(PROFILE)

experiment-8c:
	$(PY) scripts/run_experiment_8C.py --profile $(PROFILE)

experiment-8d:
	$(PY) scripts/run_experiment_8D.py --profile $(PROFILE)

## experiment-8e: drift backtest + PSI on the drift_lab profile.
experiment-8e:
	$(PY) scripts/run_experiment_8E.py --profile drift_lab

## shap: global SHAP, figures, latency + explanation gallery (docs/explanation_gallery.md).
shap:
	$(PY) scripts/run_shap.py --profile $(PROFILE)

## fairness: subgroup audit -> results/fairness_*.csv + docs/fairness_audit.md.
fairness:
	$(PY) scripts/fairness_audit.py --profile $(PROFILE)

## explain: both Part 10 outputs.
explain: shap fairness

## external: IEEE-CIS external validation (Part 11); prints "skipped: data missing" if absent.
external:
	$(PY) scripts/run_external_validation.py --profile $(PROFILE)
## sim-to-real: project the primary R1-R2 gap into results/sim_to_real.json (primary only).
sim-to-real:
	$(PY) scripts/build_sim_to_real.py
## external-report: regenerate the external-validation tables in docs/ (primary gap only).
external-report: sim-to-real
	$(PY) scripts/external_validation_report.py
# TODO(part 09): unseen-fraud    - Isolation Forest experiment
# TODO(part 12): evaluate        - locked one-shot test evaluation
## api: run the FastAPI scoring service (Part 13). HISTORY_BACKEND=memory|postgres.
HISTORY_BACKEND ?= memory
api:
	$(PY) scripts/run_api.py --profile $(PROFILE) --history-backend $(HISTORY_BACKEND)

## benchmark-latency: per-component latency matrix -> results/latency.json + docs/latency.md.
benchmark-latency:
	$(PY) scripts/benchmark_latency.py --profile $(PROFILE)
## app-artifacts: precompute the files the Streamlit app reads (Part 14).
app-artifacts:
	$(PY) scripts/make_app_artifacts.py --profile $(PROFILE)

## app: run the Streamlit analyst console (Part 14). Start `make api` first for scoring.
app:
	$(PY) -m streamlit run app/streamlit_app.py
# Part 15 targets. Aliases named as the prompt lists them; each delegates to the
# underlying script so business logic stays in src/fraudlens.
## data: simulate raw data for $(PROFILE).
data: simulate
## features: point-in-time features for $(PROFILE).
features: build-features
## db: load raw data into PostgreSQL for $(PROFILE).
db: db-load
## policy: cost-based decision policy for $(PROFILE).
policy: optimize-policy
## freeze: freeze the $(PROFILE) artifacts + data hashes (Part 12A).
freeze:
	$(PY) scripts/freeze.py --profile $(PROFILE)
## final: one-shot locked test evaluation (Part 12B); refuses if already done.
final:
	$(PY) scripts/final_evaluation.py --profile $(PROFILE)
## experiments: ML-understanding experiments 8A-8D on $(PROFILE).
experiments: experiment-8a experiment-8b experiment-8c experiment-8d
## monitor: per-period drift monitoring over train + validation (Part 15A).
monitor:
	$(PY) scripts/run_monitoring.py --profile $(PROFILE)
## render: regenerate README, model card, interview notes, trade-offs and reproduce from results (Part 15D).
render:
	$(PY) scripts/render_readme.py
## all-dev: the whole dev pipeline (no external data, no test-window access).
all-dev: simulate validate-data db-load build-features train evaluate-val calibrate optimize-policy experiments shap fairness app-artifacts
## ci-local: reproduce .github/workflows/ci.yml locally (needs `make db-up` first).
ci-local:
	$(PY) scripts/ci_local.py