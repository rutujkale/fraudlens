# Latency — scoring API

Profile `dev` · model `xgb_dev_20261015_v1` · n=2000 per cell (360 fraud / 1640 legit, seed 42), 100 warm-up excluded · replayed in-process via `httpx.ASGITransport` (sync endpoints on the AnyIO threadpool).

Hardware: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, 16 cores, 15.3 GB RAM, Python 3.12.10, Windows-11-10.0.26200-SP0.

Headline cell: **postgres|conc1|explain-off** — end-to-end p95 123.969 ms, p50 90.398 ms.

| cell | e2e p50 | e2e p95 | e2e p99 | mean | req/s | history | features | model | calib | policy | explain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| postgres|conc1|explain-off | 90.398 | 123.969 | 142.597 | 93.532 | 10.3 | 45.716 | 40.637 | 5.67 | 0.312 | 0.298 | 0.0 |
| postgres|conc8|explain-off | 711.063 | 1934.699 | 2488.51 | 898.738 | 8.0 | 249.624 | 423.286 | 25.112 | 0.484 | 0.541 | 0.001 |
| postgres|conc1|explain-on | 144.041 | 204.036 | 233.261 | 150.49 | 6.4 | 67.011 | 61.776 | 8.082 | 0.491 | 0.545 | 10.108 |
| postgres|conc8|explain-on | 744.279 | 1950.875 | 2390.791 | 918.296 | 7.9 | 240.603 | 411.938 | 26.056 | 0.476 | 0.551 | 25.545 |
| memory|conc1|explain-off | 63.644 | 113.223 | 126.284 | 71.904 | 13.2 | 1.05 | 53.969 | 7.527 | 0.385 | 0.407 | 0.001 |
| memory|conc8|explain-off | 396.08 | 1544.642 | 1954.328 | 582.054 | 11.7 | 6.677 | 354.511 | 20.676 | 0.376 | 0.399 | 0.001 |
| memory|conc1|explain-on | 59.638 | 103.495 | 120.055 | 66.536 | 14.2 | 0.956 | 44.871 | 6.257 | 0.334 | 0.322 | 6.472 |
| memory|conc8|explain-on | 414.25 | 1440.201 | 1772.78 | 575.652 | 12.0 | 6.805 | 349.308 | 19.972 | 0.359 | 0.372 | 19.736 |

Component columns are p50 ms. `explain` is 0.0 when explanations are off.

## Honest reading

- Best throughput is `memory|conc1|explain-off` at 13.2 req/s. Raising concurrency to 8 does **not** help: one Python
  process serialises the CPU-bound feature build on the GIL, so conc-8 p95 is ~15x conc-1 p95. Horizontal scaling needs uvicorn workers.
- Headline `history` is 45.716 ms p50 (the PostgreSQL round-trip); in-memory answers
  history in ~1 ms, so the postgres numbers are the conservative, deployment-like choice.
- Warm SHAP explanation adds ~6 ms (memory) to ~10 ms (postgres) at p50.
- History provider start-up: postgres 0.0 s, memory 0.504 s. In-memory loads all dev transactions at
  start-up; PostgreSQL queries per request instead.

In-process replay (no network stack, no uvicorn worker); numbers are a lower bound on real HTTP latency and exclude TLS/serialisation overhead. Components are the server-side timers from the API response.
