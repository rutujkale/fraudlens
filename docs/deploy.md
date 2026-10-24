# Deploy — Render API + Streamlit Cloud (free tier)

End-to-end steps actually used for this repo, with the live URLs filled in. Synthetic-data
portfolio project; nothing here handles real PII.

## Live deployment (verified)

**Verified 2026-10-19** against the live deployment — **17/17 automated checks passed**
(Streamlit console loads, both sidebar chips correct, a transaction scores end-to-end through
the Render API, SHAP chart renders, zero app console/page/network errors, zero browser→Render
requests). Screenshots from that run: `docs/progress/live_deploy_screenshots/`.

| Piece | Live URL |
|---|---|
| Analyst console | <https://fraud-lens.streamlit.app/> |
| Scoring API | <https://fraudlens-api-pld9.onrender.com> |

Free instances sleep after ~15 min idle: the first console interaction after a quiet period
can take 30–50 s (the app waits on the Render wake).

## What ships

| Piece | Source | Runs on |
|---|---|---|
| Scoring API | `Dockerfile.api` → `ghcr.io/rutujkale/fraudlens-api:v1` | Render (free web service) |
| Analyst console | repo, `app/streamlit_app.py` | Streamlit Community Cloud (free) |

- The image build **fails** unless every file in `results/freeze.json` matches its sha256
  (build log line: `BUILD-TIME FROZEN-BUNDLE ASSERTION PASSED: 14 files sha256-match ...`).
- Measured container footprint (profile `dev`, `HISTORY_BACKEND=memory`, peak `VmHWM` from
  `/proc/1/status`, 5 fresh-container runs, 2026-10-23; 512 MB cap = 524,288 kB): `:v1` boot
  **446,820 / 439,888 kB** and first explanation **513,260 / 505,036 kB** (worst / median);
  `fraudlens-api:pinned` (the v1-versioned lock) boot **455,520 / 448,600 kB** and first
  explanation **523,264 / 515,252 kB**. First-explain headroom is thin (worst runs ~1-11 MB
  under the cap); the demo's Page 1 requests an explanation every time, and `:v1` has run live
  without an OOM. The earlier 441,756 / 472,812 kB reading was a low outlier.
  `MALLOC_ARENA_MAX=2` cut the worst run by 14.4 MB — below the pre-set 15 MB bar and
  within run-to-run noise (`:v1` alone varies ~40 MB), so it is not set.
- Free instances sleep after ~15 min without traffic; cold start on 0.1 CPU takes ~1 min and
  Render keeps retrying health checks for up to 15 min.

## 1. Build and smoke-test the image (local)

```bash
docker build -f Dockerfile.api -t fraudlens-api:local .
# expect: BUILD-TIME FROZEN-BUNDLE ASSERTION PASSED: 14 files sha256-match results/freeze.json; serving xgb_dev_20261015_v1

docker run -d --name fraudlens-smoke -p 8080:10000 -e PORT=10000 fraudlens-api:local
sleep 6
curl -s http://127.0.0.1:8080/health
# {"status":"ok","model_version":"xgb_dev_20261015_v1","history_backend":"memory","history_ok":true}

curl -s -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json | head -c 400
# HTTP 200, keys include fraud_probability, risk_score, recommended_action, contributions

docker rm -f fraudlens-smoke
```

## 2. Push the image to GHCR (one-time)

The `gh` CLI token needs the packages scopes (a plain `gh auth login` does not include them):

```bash
gh auth refresh -s write:packages,read:packages
echo "$(gh auth token)" | docker login ghcr.io -u rutujkale --password-stdin

docker tag fraudlens-api:local ghcr.io/rutujkale/fraudlens-api:v1
docker push ghcr.io/rutujkale/fraudlens-api:v1
```

Then make the package **public** — Render's free plan cannot pull private images:

1. Open <https://github.com/users/rutujkale/packages> → package **fraudlens-api**.
2. **Package settings** → **Change visibility** → **Public** → confirm.

If you skip this step, Render fails with **Image Pull Failed**.

## 3. Render web service

The repo contains `render.yaml` (blueprint: `runtime: image`, `plan: free`, health path
`/health`). Two equivalent ways:

**Blueprint (recommended)** — Render dashboard → **New +** → **Blueprint** → connect the
GitHub repo → Render picks up `render.yaml` → **Apply**.

**Manual** — Render dashboard → **New +** → **Web Service** → **Deploy an existing image**
→ image URL `ghcr.io/rutujkale/fraudlens-api:v1` → Instance Type **Free** →
Health Check Path `/health` → env vars:

| Key | Value |
|---|---|
| `FRAUDLENS_PROFILE` | `dev` |
| `HISTORY_BACKEND` | `memory` |
| `WRITE_PREDICTIONS` | `0` |

(`PORT` is injected by Render; the entrypoint honours it.)

Deploy, wait for the first green **Live**, and note the URL, e.g.
`https://fraudlens-api-pld9.onrender.com` (Render appends a random suffix, so the real URL
differs from the service name).

## 4. Streamlit Community Cloud

1. <https://share.streamlit.io> → **New app** → repo `rutujkale/fraudlens`, branch `main`,
   main file path **`app/streamlit_app.py`**.
2. **Advanced settings → Secrets** and paste (this is the only secret; it is never committed —
   `.streamlit/secrets.toml` is gitignored):

```toml
FRAUDLENS_API_URL = "https://fraudlens-api-pld9.onrender.com"
```

3. **Deploy**. The app falls back to `http://127.0.0.1:8000` when the secret is absent
   (local runs), so nothing else changes for local development.

## 5. Post-deploy smoke test

```bash
curl -s https://fraudlens-api-pld9.onrender.com/health
# {"status":"ok","model_version":"xgb_dev_20261015_v1",...}

curl -s -X POST https://fraudlens-api-pld9.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json | head -c 400
# HTTP 200, model_version xgb_dev_20261015_v1
```

Then open <https://fraud-lens.streamlit.app/> → **Transaction Analyzer** → Run; it should score
without an "API is down" banner.

## 6. Updating the image later

```bash
docker build -f Dockerfile.api -t fraudlens-api:local .
docker tag fraudlens-api:local ghcr.io/rutujkale/fraudlens-api:v1
docker push ghcr.io/rutujkale/fraudlens-api:v1
```

Render pins the digest it pulled: dashboard → service → **Manual Deploy** → **Deploy**
(re-pulls `v1`). For a new release, bump the tag (`v2`, …) in the commands above **and** in
`render.yaml`.

## 7. Refreshing `requirements.lock`

`requirements.txt` stays the loose, human-edited file; `Dockerfile.api` installs
`requirements.lock`. After editing `requirements.txt`, resolve on the deploy target (Linux,
Python 3.12) and freeze — this also captures Linux-only deps such as `nvidia-nccl-cu13`:

```bash
docker run --rm -v "${PWD}:/w" -w /w python:3.12-slim sh -c \
  "pip install -q -r requirements.txt && pip freeze | grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' > requirements.lock"
```

Then rebuild; the build prints `BUILD-TIME FROZEN-BUNDLE ASSERTION PASSED: 14 files ...`.

## Limits of this setup (honest)

- `HISTORY_BACKEND=memory` only — no PostgreSQL on the free tier; the dev raw parquet is baked
  into the image (the build context excludes every other profile via `.dockerignore`).
- `WRITE_PREDICTIONS=0` — no persistence; in-memory history loses nothing because it is baked.
- No auth, no rate limiting, single profile `dev` — portfolio scope, not production.
- Free plan: 512 MB RAM / 0.1 CPU, sleeps after ~15 min idle. If Render ever reports memory
  errors, the measured boot/explain numbers above are the ones to beat — re-measure with
  `docker stats` after any change to data loading.
