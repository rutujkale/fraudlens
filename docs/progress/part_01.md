# Part 01 — Repository, Config, Database Container, Tooling

## What was built

- **Repo skeleton** at `E:\project\fraudlens_project` (git repo): `pyproject.toml` (setuptools src-layout, `python>=3.11`, dev extras pytest/pytest-cov/ruff/httpx, ruff config E,F,I,UP,B, line length 100, pytest markers `db`/`slow`), `requirements.txt` (compatible-range pins for the full stack), `.gitignore` (data/, artifacts/, .env, .venv, *.parquet, etc.; `results/` and `reports/figures/` tracked via `.gitkeep`).
- **Package tree** under `src/fraudlens/`: empty `__init__.py` for data_gen, data_quality, db, features, models, evaluation, risk, explain, monitoring, external, api; `app/`, `scripts/`, `tests/`, `sql/`, `config/`, `docs/` created.
- **Config layer** (`config/config.yaml` + `src/fraudlens/config.py`): pydantic v2 models with `extra="forbid"`; `load_config(profile)` merges `common` + profile; `SimulationConfig` with nested splits/features/costs/cost_scenarios/drift; `embargo_days` property = `max(label_delay_days)`; `config_hash()` = sha256 of post-processed deterministic JSON (`sort_keys`). Validation: scenario mixes (period_a, period_b, per-profile override) must each have fractions in (0,1) and sum to 1 ±1e-6, else `ValueError`.
- **Utilities** (`src/fraudlens/utils.py`): `set_global_seed`, `get_logger` (stream handler, INFO), `git_commit` (falls back to "unknown"), `ensure_dir`. **Paths** (`src/fraudlens/paths.py`): `raw_data_dir`, `features_data_dir`, `artifacts_dir`, `results_dir`, `figures_dir`, all created on demand from repo root.
- **DB container**: `docker-compose.yml` (postgres:16, db `fraudlens`, named volume, `pg_isready` healthcheck, port/credentials from `.env`) + `.env.example`. `.env` was created by **copying** `.env.example` (file copy, not regenerated) and is gitignored.
- **Makefile**: `setup`, `db-up`, `db-down`, `lint`, `format`, `test`, `test-fast` (`-m "not db and not slow"`), plus `# TODO(part XX)` placeholder comments for the remaining parts.
- **Docs skeleton**: `docs/what_went_wrong.md` (Incident/How detected/Effect/Fix/Test added template), `docs/progress/`, minimal README stub.
- **Tests** (`tests/test_config.py`, 15 tests): every profile loads; all scenario mixes sum to 1; `embargo_days` == max label delay per profile; unknown profile raises; invalid mix raises (via temp config + `FRAUDLENS_CONFIG_PATH`); `config_hash` stable across loads and differs across profiles.

## Acceptance commands — actual outputs

| Command | Result |
|---|---|
| `pip install -e ".[dev]"` (after `-r requirements.txt`) | OK. venv on Python 3.12.10; slowest notable pins: xgboost 3.4.1, shap 0.53.0, fastapi 0.143.0, streamlit 1.65.0 |
| `python -c "import fraudlens"` (added smoke test) | `import ok` |
| `make lint` | `All checks passed!` (one UP037 fixed during the run) |
| `make test-fast` | `15 passed in 0.40s` |
| `python -c "from fraudlens.config import load_config; c = load_config('main'); print(c.simulation.customers, c.embargo_days)"` | `20000 21` |
| `docker compose up -d` then `docker compose ps` | `fraudlens_db` ... `Up 9 seconds (healthy)` on `0.0.0.0:5432->5432/tcp` |
| Connectivity check (psycopg2 via `DATABASE_URL`) | `PostgreSQL 16.15 (Debian ...) on x86_64-pc-linux-gnu` |

## Deviations from the prompt

1. **Python 3.12** used (3.12.10) instead of 3.11 default. Spec requires `>=3.11`; 3.12 chosen for wheel availability of xgboost/shap/psycopg2, documented here as agreed.
2. **PostgreSQL via Docker Desktop installed by the user** (Docker was not present; chosen over native Postgres to match the spec's containerized setup). Engine needed `Start-Process "Docker Desktop.exe"`, first `postgres:16` pull took a full run; healthcheck reported healthy ~5s after start.
3. **GNU make 3.81** installed via winget (`GnuWin32.Make`); it runs recipes through Git for Windows' `sh.exe`, so `C:\Program Files (x86)\GnuWin32\bin` + `C:\Program Files\Git\usr\bin` were added to the user PATH. Makefile uses `PY ?= .venv/Scripts/python` (Windows venv layout) so targets run without activation; `PY` is overridable.
4. **CLAUDE.md → AGENTS.md** at repo root (opencode's expected context file), per the rename guidance in `00_START_HERE.md`.
5. **Config path override**: `load_config` reads `FRAUDLENS_CONFIG_PATH` (default `config/config.yaml` at repo root); used by the invalid-config test instead of mutating the real file.
6. `validate_scenario_mix` is a public helper (not underscore-private) because the spec's tests exercise the validation rule directly.

## Open issues

- None for Part 01. `docker compose down` not yet exercised (`make db-down` available).
- Note: raw psycopg2 needs `postgresql://`; `.env` `DATABASE_URL` intentionally uses the SQLAlchemy dialect URL `postgresql+psycopg2://` for Part 04+.