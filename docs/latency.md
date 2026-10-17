# Latency — scoring API

Profile `dev` · model `xgb_dev_20261015_v1` · n=2000 per cell (360 fraud / 1640 legit — all validation-window frauds plus a legit fill, so **fraud-enriched, not balanced**; seed 42), 100 warm-up excluded · replayed in-process via `httpx.ASGITransport` (sync endpoints on the AnyIO threadpool).

Hardware: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, 16 cores, 15.3 GB RAM, Python 3.12.10, Windows-11-10.0.26200-SP0.

Headline cell: **postgres|conc1|explain-off** — end-to-end p95 127.031 ms, p50 92.019 ms.

Headline rule: postgres (deployment backend) x concurrency 1 (single-request serving) x explanation off (API default); chosen after the latency matrix was measured, not pre-registered.

Full conditions for the quoted figure: **profile `dev`**, PostgreSQL history backend, concurrency 1, explanation off, dev **validation-window** sample (the test window is never read), in-process ASGI replay (**a lower bound**: no network, TLS or uvicorn overhead).

Benchmark-time PostgreSQL provenance: `raw.transactions` = **277,607 rows**, `raw.customers` = 5,000 (expected profile `dev`).

| cell | e2e p50 | e2e p95 | e2e p99 | mean | req/s | history | features | model | calib | policy | explain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| postgres|conc1|explain-off | 92.019 | 127.031 | 150.278 | 95.176 | 10.2 | 46.385 | 41.432 | 5.771 | 0.322 | 0.303 | 0.0 |
| postgres|conc8|explain-off | 479.766 | 1273.334 | 1532.161 | 598.133 | 12.1 | 172.109 | 275.36 | 16.959 | 0.357 | 0.364 | 0.001 |
| postgres|conc1|explain-on | 96.618 | 131.598 | 146.745 | 100.028 | 9.7 | 46.02 | 40.978 | 5.708 | 0.32 | 0.302 | 5.854 |
| postgres|conc8|explain-on | 515.848 | 1347.267 | 1640.133 | 637.304 | 11.4 | 177.138 | 280.784 | 17.806 | 0.36 | 0.365 | 17.463 |
| memory|conc1|explain-off | 49.658 | 90.711 | 104.824 | 55.993 | 16.9 | 0.802 | 41.878 | 5.87 | 0.316 | 0.3 | 0.0 |
| memory|conc8|explain-off | 345.759 | 1299.622 | 1681.912 | 506.984 | 13.5 | 6.148 | 307.673 | 18.039 | 0.344 | 0.355 | 0.001 |
| memory|conc1|explain-on | 54.963 | 96.194 | 108.936 | 61.692 | 15.4 | 0.818 | 41.282 | 5.928 | 0.318 | 0.31 | 6.003 |
| memory|conc8|explain-on | 385.12 | 1357.332 | 1737.295 | 545.541 | 12.5 | 6.899 | 320.887 | 19.382 | 0.351 | 0.364 | 18.515 |

Component columns are p50 ms. `explain` is 0.0 when explanations are off.

## Honest reading

- Best throughput is `memory|conc1|explain-off` at 16.9 req/s. Raising concurrency to 8 does **not** help here: the component timings show the feature build is the bottleneck. The
  most likely explanation is the GIL serialising that CPU-bound work in this single-process replay — a **hypothesis**, not evidence about a multi-worker deployment. Horizontal scaling needs uvicorn workers/processes.
- Headline `history` is 46.385 ms p50 (the PostgreSQL round-trip); in-memory answers
  history in ~1 ms, so the postgres numbers are the conservative, deployment-like choice.
- Warm SHAP explanation adds ~6 ms (memory) to ~10 ms (postgres) at p50.
- History provider start-up: postgres 0.0 s, memory 0.409 s. In-memory loads all dev transactions at
  start-up; PostgreSQL queries per request instead.

In-process replay (no network stack, no uvicorn worker); numbers are a lower bound on real HTTP latency and exclude TLS/serialisation overhead. Components are the server-side timers from the API response.
