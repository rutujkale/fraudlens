# FraudLens — Project Context (applies to every part)

FraudLens is a portfolio-grade, end-to-end **fraud-risk decision system**: behavioral transaction simulator → PostgreSQL → point-in-time features (with leakage tests) → Rules / Logistic Regression / XGBoost under temporal validation → probability calibration → cost-based decision policy with analyst-capacity limits → Isolation Forest unseen-fraud experiment → SHAP → FastAPI + Streamlit → monitoring.

**Purpose of the project:** demonstrate deep understanding of ML (imbalance, temporal validation, leakage, calibration, model comparison, drift, decision theory, explainability). Evidence beats feature count. Every major component must produce a measurable result, answer a business question, or protect the credibility of a result.

It is a **portfolio/research system, not a production banking system.** Synthetic data only (plus one public benchmark in Part 11).

## Working protocol (follow for every part)

1. Read this file and the part prompt fully. Write a short plan (files to create, sub-steps) before coding.
2. Implement in the sub-steps the prompt lists. Run tests after each sub-step.
3. Run the acceptance commands. Fix failures before finishing.
4. Write `docs/progress/part_XX.md` with: what was built; commands run and their **actual** output summaries; deviations from the prompt (and why); open issues.
5. Append any bug, leakage incident, simulator shortcut, or failed experiment to `docs/what_went_wrong.md` (what happened, how detected, effect, fix, test added).
6. **Stop at the end of the part.** Do not begin the next part.

If a requirement is ambiguous, choose the simplest option that preserves analytical correctness, state the choice in the progress file, and continue. Do not silently drop requirements.

## Tech stack

Python 3.11, pandas, numpy, pyarrow, scikit-learn, xgboost, shap, scipy, matplotlib, plotly, pyyaml, pydantic v2, SQLAlchemy 2 + psycopg2-binary, FastAPI, uvicorn, httpx, streamlit, joblib, pytest, ruff. PostgreSQL 16 via Docker Compose. No deep-learning frameworks. No SMOTE.

## Repository layout (target)

```text
fraudlens/
├── CLAUDE.md  README.md  pyproject.toml  requirements.txt  Makefile
├── docker-compose.yml  .env.example  .gitignore
├── config/config.yaml
├── sql/                     # schema.sql, indexes.sql, views.sql
├── src/fraudlens/
│   ├── config.py  paths.py  utils.py          # load_config(profile), seeds, logging
│   ├── data_gen/   archetypes.py entities.py normal.py hard_negatives.py fraud.py drift.py labels.py generate.py
│   ├── data_quality/ validation.py
│   ├── db/         loader.py queries.py
│   ├── features/   registry.py windows.py build_features.py serving.py
│   ├── models/     rules.py logreg.py xgb.py anomaly.py calibration.py io.py
│   ├── evaluation/ splits.py metrics.py bootstrap.py cost_model.py threshold_search.py experiments.py guard.py
│   ├── risk/       policy.py scoring.py
│   ├── explain/    shap_explainer.py reasons.py fairness.py
│   ├── monitoring/ drift.py metrics.py
│   ├── external/   ieee_cis.py
│   └── api/        main.py schemas.py deps.py history.py
├── app/            streamlit_app.py  pages/
├── scripts/        thin CLI entrypoints only (no business logic)
├── tests/
├── docs/           progress/  experiments/  what_went_wrong.md ...
├── data/           (gitignored) raw/<profile>/  features/<profile>/  external/
├── artifacts/      (gitignored) model artifacts per model_version
├── results/        small CSV/JSON results (tracked)
└── reports/figures/  PNG figures (tracked)
```

Business logic lives in `src/fraudlens/`. `scripts/`, `app/` and `api/` import from it. **Feature logic exists in exactly one place** (`features/`) and is used by training, API and Streamlit.

## Global conventions

- **Config-driven and reproducible.** All parameters in `config/config.yaml` with named profiles: `tiny` (CI), `dev`, `main`, `drift_lab`, `unseen_lab`. No hard-coded magic numbers in code. Every random process takes a seed from config; same seed ⇒ identical output.
- **Timestamps** are naive datetimes in IST (Asia/Kolkata). API input with a timezone is converted to IST then made naive. **Currency is INR (₹).**
- **Sort order / ties:** transactions are ordered by `(timestamp, transaction_id)`.
- **Point-in-time rule:** a feature for the transaction at time `T` may use only data with timestamp **strictly < T** (the current row is excluded; rows with the same timestamp are excluded). Static customer attributes known at account opening (e.g. home region) are allowed.
- **Labels** (`is_fraud`, `fraud_scenario`, `episode_id`, `hard_negative_type`, `label_available_at`) live in a separate `labels` table, never in `transactions`, and never appear in feature columns. A guard function enforces this.
- **Temporal validation only.** Splits are chronological with an embargo gap equal to the maximum label delay. Hyperparameters, calibrators and thresholds are fit on train/validation. The test window is evaluated **once**, in Part 12, through `evaluation/guard.py`.
- **Imbalance:** primary metric PR-AUC; also recall@top-K, recall at fixed precision, FPR, Brier, ECE, expected/realized cost. Accuracy is never a headline metric.
- **Bootstrap CIs resample customers (cluster bootstrap)**, not transactions, because a customer's transactions are correlated.
- **Numbers integrity:** never hard-code or invent a result. Results are written by code to `results/*.csv|json` and `reports/figures/`. Any table in docs is generated from those files.
- **Experiment log:** every experiment appends a row to `results/experiments.csv` (experiment_id, part, dataset_profile, model, features_version, train/val/test windows, hyperparameters, calibration, threshold_policy, metrics, notes, git_commit, timestamp).
- **Code quality:** type hints, docstrings on public functions, `ruff` clean, small pure functions, logging via `logging` (no print in library code), tests under `tests/`. Heavy/DB tests use pytest markers (`db`, `slow`).
- **Artifacts** carry metadata: `model_version`, `feature_version` (`features_v1`), `training_window`, `calibration`, `policy_version`, metrics, seed, git commit, config hash.

## Decision framework (used in Parts 07, 12, 13, 14)

Per transaction with calibrated probability `p`, amount `A`:

```text
cost(APPROVE) = p·(α·A + fixed_fee)
cost(STEP_UP) = S + (1-p)·F_step + p·(1-s)·(α·A + fixed_fee)
cost(REVIEW)  = R + (1-p)·F + p·(1-c)·(α·A + fixed_fee)
cost(BLOCK)   = (1-p)·F_block
```
α = fraud loss rate, R = review cost, F = friction cost, S = step-up cost, s = step-up stop rate, c = review catch rate, F_block = friction when blocking a legit transaction. With only APPROVE vs REVIEW and c = 1, intervene iff `p > (R+F)/(αA + fixed_fee + F)` — an amount-dependent threshold that is valid only for calibrated `p`. Analyst capacity caps REVIEW cases per day; allocate by largest expected saving, and demote the rest to the next-best action. Risk **levels derive from actions** (APPROVE→LOW, STEP_UP→MEDIUM, REVIEW→HIGH, BLOCK→CRITICAL), not from fixed probability cutoffs.

## Things you must never do

- Use a random split as primary evaluation. Tune on the test window. Peek at test results before Part 12.
- Put a target-derived field into features. Use SMOTE or any synthetic oversampling.
- Present raw SHAP values as points on the 0–100 risk score.
- Claim production readiness, or write README/resume numbers that are not in `results/`.
- Duplicate feature logic in notebooks, the API, or Streamlit.
- Use real personal or financial data.
