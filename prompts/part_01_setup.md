# Part 01 — Repository, Config, Database Container, Tooling

## Goal
Create the project skeleton so every later part has a stable home: package layout, validated YAML configuration with named profiles, Docker PostgreSQL, Makefile, lint/test tooling.

## Prerequisites
Empty git repo containing `CLAUDE.md` and `prompts/`.

## Tasks

### 1A — Packaging
- `pyproject.toml` with src layout, package `fraudlens`, Python ≥3.11, `[project.optional-dependencies] dev = [pytest, pytest-cov, ruff, httpx]`. Editable install must work: `pip install -e ".[dev]"`.
- `requirements.txt` with compatible-range pins for: pandas, numpy, pyarrow, scikit-learn, xgboost, shap, scipy, matplotlib, plotly, pyyaml, pydantic>=2, sqlalchemy>=2, psycopg2-binary, fastapi, uvicorn, streamlit, joblib, python-dotenv.
- `ruff` config in `pyproject.toml` (line length 100, rules E,F,I,UP,B).
- Create the full directory tree from CLAUDE.md with `__init__.py` files and empty placeholder modules only where useful (do not write stub logic).
- `.gitignore`: `data/`, `artifacts/`, `.env`, `__pycache__`, `.venv`, `*.parquet`, `.pytest_cache`, notebooks checkpoints. Keep `results/` and `reports/figures/` tracked (use `.gitkeep`).

### 1B — Configuration (`config/config.yaml` + `src/fraudlens/config.py`)
Create the YAML below (you may refine names, but keep the structure and values) and a pydantic v2 model hierarchy that validates it. `load_config(profile: str = "dev") -> Config` merges `common` with the profile section. Include a `config_hash(cfg)` helper (sha256 of canonical JSON).

```yaml
common:
  seed: 42
  currency: INR
  start_date: "2026-01-01"
  reporting_threshold: 50000        # structuring reference amount (₹)
  customers_archetype_mix: {conservative: 0.30, regular: 0.40, high_spend: 0.10, frequent_digital: 0.15, business_like: 0.05}
  fraud_scenarios: [burst, account_takeover, amount_anomaly, geo_anomaly, structuring]
  scenario_mix_period_a: {burst: 0.22, account_takeover: 0.28, amount_anomaly: 0.22, geo_anomaly: 0.16, structuring: 0.12}
  scenario_mix_period_b: {burst: 0.15, account_takeover: 0.25, amount_anomaly: 0.12, geo_anomaly: 0.08, structuring: 0.40}
  stealth_beta_period_a: [2.0, 5.0]   # obvious fraud (mean ≈ 0.29)
  stealth_beta_period_b: [4.0, 3.0]   # subtler fraud (mean ≈ 0.57)
  hard_negative_target_share: 0.04    # ≈4% of legit tx belong to hard-negative episodes
  fraud_device_reuse_prob: 0.15
  family_shared_device_share: 0.04    # legit shared devices so device-sharing is not a trivial signal
  splits: {train_end_fraction: 0.60, val_end_fraction: 0.80}   # embargo = max label delay (days)
  features: {version: features_v1, windows_minutes: [10, 60, 360, 1440, 10080, 43200]}
  costs:
    fraud_loss_rate: 0.85
    fraud_fixed_fee: 0
    review_cost: 75
    friction_cost: 150
    review_catch_rate: 0.90
    stepup_cost: 5
    stepup_friction_cost: 40
    stepup_stop_rate: 0.60
    block_friction_cost: 450
    capacity_pct_of_daily_volume: 0.01   # review capacity as a share of average daily volume
    analyst_daily_capacity: null         # optional absolute override
  cost_scenarios:
    low_fp_cost:    {friction_cost: 50,  review_cost: 40}
    balanced:       {}
    high_fp_cost:   {friction_cost: 400, review_cost: 120}
    high_fraud_loss: {fraud_loss_rate: 1.0, fraud_fixed_fee: 300}
  ks: [0.001, 0.005, 0.01, 0.02, 0.05]          # recall@top-K fractions
  precision_targets: [0.9, 0.8, 0.7]
  bootstrap: {n: 500, cluster_by: customer_id}
  api: {max_batch: 1000}

profiles:
  tiny:       {customers: 300,   days: 30,  target_fraud_rate: 0.02,  label_delay_days: [1, 3],  drift: {start_fraction: null, strength: 0.0}}
  dev:        {customers: 5000,  days: 120, target_fraud_rate: 0.008, label_delay_days: [2, 7],  drift: {start_fraction: 0.88, strength: 0.5}}
  main:       {customers: 20000, days: 270, target_fraud_rate: 0.005, label_delay_days: [3, 21], drift: {start_fraction: 0.88, strength: 0.5}}
  drift_lab:  {customers: 10000, days: 240, target_fraud_rate: 0.006, label_delay_days: [3, 14], drift: {start_fraction: 0.50, strength: 1.0}}
  unseen_lab: {customers: 15000, days: 180, target_fraud_rate: 0.005, label_delay_days: [3, 14], drift: {start_fraction: null, strength: 0.0}, scenario_mix_override: {burst: 0.15, account_takeover: 0.20, amount_anomaly: 0.15, geo_anomaly: 0.15, structuring: 0.35}}
```

Notes to implement:
- `embargo_days` for a profile = `max(label_delay_days)`.
- `main` places drift start at the beginning of the test window by design (val = pre-drift, test = post-drift). Document this in a config comment.
- Config validation must fail with a clear message if scenario mixes do not sum to 1 (±1e-6), or a fraction is outside (0, 1).

### 1C — Utilities
- `utils.py`: `set_global_seed`, `get_logger`, `git_commit()` (returns "unknown" if not a repo), `ensure_dir`.
- `paths.py`: functions returning `data/raw/<profile>/`, `data/features/<profile>/`, `artifacts/<model_version>/`, `results/`, `reports/figures/` (create on demand).

### 1D — Database container
- `docker-compose.yml`: `postgres:16`, db `fraudlens`, volume, healthcheck, port from `.env`. `.env.example` with `POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT, DATABASE_URL`.

### 1E — Makefile
Targets: `setup` (venv + install), `db-up`, `db-down`, `lint`, `format`, `test`, `test-fast` (exclude `db` and `slow` markers). Add placeholders (comments) for targets that later parts will fill.

### 1F — Docs skeleton
`docs/what_went_wrong.md` (template with headings: *Incident, How detected, Effect, Fix, Test added*), `docs/progress/` folder, minimal `README.md` stub (title + one-paragraph description + "work in progress").

### 1G — Tests
`tests/test_config.py`: every profile loads; mix sums validated; invalid config raises; `embargo_days` correct; `config_hash` stable across loads.

## Acceptance criteria
```bash
pip install -e ".[dev]"
make lint
make test-fast
python -c "from fraudlens.config import load_config; c = load_config('main'); print(c.simulation.customers, c.embargo_days)"
docker compose up -d && docker compose ps     # postgres healthy
```
All pass. `docs/progress/part_01.md` written.

## Do not
Write simulator, feature, or model logic yet. Add dependencies not listed.
