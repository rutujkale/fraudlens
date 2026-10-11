# FraudLens development targets.
# `PY` points at the venv interpreter so recipes work without activation.
# Override on the command line if the venv lives elsewhere:
#   make setup PY="py -3.12"
PY ?= .venv/Scripts/python

.PHONY: setup db-up db-down lint format test test-fast simulate validate-data eda separability-gate

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

# TODO(part 04): db-load         - load raw data into PostgreSQL
# TODO(part 05): build-features  - point-in-time feature engineering
# TODO(part 06): train           - rules / logreg / xgboost baselines
# TODO(part 07): calibrate       - calibration + cost-based decision policy
# TODO(part 08): experiments     - imbalance/ablation/drift experiments
# TODO(part 09): unseen-fraud    - Isolation Forest experiment
# TODO(part 10): explain         - SHAP explanations + subgroup audit
# TODO(part 11): external        - IEEE-CIS external validation
# TODO(part 12): evaluate        - locked one-shot test evaluation
# TODO(part 13): api             - FastAPI service + latency benchmark
# TODO(part 14): streamlit       - analyst application
# TODO(part 15): monitor         - drift monitoring + CI + docs