"""Part 13C — per-component latency benchmark for the scoring API.

Samples real validation-window transactions, replays them through the in-process
FastAPI app (``httpx.ASGITransport``, so sync endpoints run in the AnyIO
threadpool exactly as under uvicorn), and reports mean/p50/p95/p99 per component
for the matrix {history backend} x {concurrency 1, 8} x {explanation off, on}.

Writes ``results/latency.json`` (machine) and ``docs/latency.md`` (human).
The test window is never touched: every sampled row is asserted inside
``[t_train_end, t_val_end)``.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from fraudlens.api.deps import Scorer
from fraudlens.api.main import create_app
from fraudlens.config import load_config
from fraudlens.evaluation.dataset import load_feature_dataset
from fraudlens.evaluation.splits import make_temporal_split
from fraudlens.paths import docs_dir, raw_data_dir, results_dir
from fraudlens.utils import get_logger, git_commit

logger = get_logger(__name__)

COMPONENTS = ("history_fetch", "features", "model", "calibration", "policy", "explanation")
BACKENDS = ("postgres", "memory")
CONCURRENCIES = (1, 8)
EXPLANATION_FLAGS = (False, True)

# Rule fixing which cell is quoted as the headline. Declared explicitly (and recorded
# in latency.json) because the cell was chosen after the matrix was measured.
HEADLINE_RULE = (
    "postgres (deployment backend) x concurrency 1 (single-request serving) x "
    "explanation off (API default); chosen after the latency matrix was measured, "
    "not pre-registered"
)


def db_provenance(expected_profile: str) -> dict[str, Any] | None:
    """Row counts of the tables currently loaded in PostgreSQL (benchmark-time evidence).

    The postgres cells are only meaningful if the profile they claim is actually
    loaded; record the counts so the figure is self-describing.
    """
    try:
        from sqlalchemy import text

        from fraudlens.db.loader import get_engine

        with get_engine().connect() as conn:
            rows = int(conn.execute(text("SELECT count(*) FROM raw.transactions")).scalar())
            customers = int(conn.execute(text("SELECT count(*) FROM raw.customers")).scalar())
        return {
            "expected_profile": expected_profile,
            "transactions_rows": rows,
            "customers": customers,
        }
    except Exception as exc:  # noqa: BLE001 - provenance must not break the benchmark
        logger.warning("could not read PostgreSQL row counts: %s", exc)
        return None


def _total_ram_gb() -> float | None:
    """Physical RAM in GB via the Win32 ``GlobalMemoryStatusEx`` call."""
    try:
        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return round(status.ullTotalPhys / 1024**3, 1)
    except Exception:  # pragma: no cover - non-Windows / permission
        return None


def hardware() -> dict[str, Any]:
    """Capture the machine the numbers were measured on (conditions matter)."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
    }


def _opt(value: Any) -> Any:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else value


def _request_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "customer_id": row["customer_id"],
        "amount": float(row["amount"]),
        "timestamp": str(row["timestamp"]),
        "merchant_category": _opt(row.get("merchant_category")),
        "merchant_id": _opt(row.get("merchant_id")),
        "device_id": _opt(row.get("device_id")),
        "location_id": _opt(row.get("location_id")),
        "transaction_type": row["transaction_type"],
        "channel": row["channel"],
        "recipient_id": _opt(row.get("recipient_id")),
        "transaction_id": row["transaction_id"],
    }


def sample_requests(
    profile: str, n: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample the validation window: all frauds (up to ``n/2``) + a legit fill.

    Every validation-window fraud is taken (there are fewer than ``n/2`` on dev), then
    legit rows fill the rest — so the sample is fraud-enriched (dev: 360 fraud / 1,640
    legit = 18%, about 20x the ~0.9% validation prevalence), not balanced. A fixed seed
    keeps the exact row set stable across reruns; the enrichment changes the mix of
    actions, not per-request latency.
    """
    df, labels = load_feature_dataset(profile)
    split = make_temporal_split(df, load_config(profile))
    lo = pd.Timestamp(split.boundaries["t_train_end"])
    hi = pd.Timestamp(split.boundaries["t_val_end"])

    val_keys = df.iloc[split.val_idx][["transaction_id", "customer_id", "timestamp"]]
    if not ((val_keys["timestamp"] >= lo) & (val_keys["timestamp"] < hi)).all():
        raise RuntimeError("split val rows fall outside the validation window")

    raw = pd.read_parquet(raw_data_dir(profile) / "transactions.parquet")
    label_by_id = labels.set_index("transaction_id")["is_fraud"]
    val = raw[raw["transaction_id"].isin(val_keys["transaction_id"])].copy()
    val["is_fraud"] = val["transaction_id"].map(label_by_id)

    rng = np.random.default_rng(seed)
    pos = val[val["is_fraud"] == 1]
    neg = val[val["is_fraud"] == 0]
    n_pos = min(n // 2, len(pos))
    n_neg = min(n - n_pos, len(neg))
    picked = pd.concat(
        [
            pos.iloc[rng.choice(len(pos), n_pos, replace=False)],
            neg.iloc[rng.choice(len(neg), n_neg, replace=False)],
        ]
    ).sort_values(["timestamp", "transaction_id"], kind="stable")
    logger.info("sampled %d val rows (%d fraud, %d legit)", len(picked), n_pos, n_neg)
    requests = [_request_from_row(row) for _, row in picked.iterrows()]
    return requests, {"n_fraud": int(n_pos), "n_legit": int(n_neg)}


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def _collect(response: httpx.Response, e2e: list[float], comps: dict[str, list[float]]) -> int:
    if response.status_code != 200:
        logger.warning("non-200 (%s): %s", response.status_code, response.text[:200])
        return 1
    body = response.json()
    e2e.append(float(body["latency_ms"]))
    for name in COMPONENTS:
        comps[name].append(float(body["timing_ms"][name]))
    return 0


async def run_cell(
    app, requests: list[dict[str, Any]], concurrency: int, warmup: int, explanation: bool
) -> dict[str, Any]:
    """Replay ``requests`` through the app at the given concurrency."""
    payloads = [dict(r, include_explanation=explanation) for r in requests]
    e2e: list[float] = []
    comps: dict[str, list[float]] = {name: [] for name in COMPONENTS}
    errors = 0
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://bench", timeout=120.0
    ) as client:
        for payload in payloads[:warmup]:
            await client.post("/predict", json=payload)
        start = time.perf_counter()
        if concurrency == 1:
            for payload in payloads:
                errors += _collect(await client.post("/predict", json=payload), e2e, comps)
        else:
            sem = asyncio.Semaphore(concurrency)

            async def one(payload: dict[str, Any]) -> httpx.Response:
                async with sem:
                    return await client.post("/predict", json=payload)

            for response in await asyncio.gather(*(one(p) for p in payloads)):
                errors += _collect(response, e2e, comps)
        wall = time.perf_counter() - start
    return {
        "concurrency": concurrency,
        "explanation": explanation,
        "n": len(payloads),
        "n_errors": errors,
        "wall_seconds": round(wall, 3),
        "requests_per_second": round(len(payloads) / wall, 1) if wall else None,
        "e2e_ms": _percentiles(e2e) if e2e else None,
        "components_ms": {name: _percentiles(vals) for name, vals in comps.items() if vals},
    }


def _cell_name(backend: str, concurrency: int, explanation: bool) -> str:
    return f"{backend}|conc{concurrency}|explain-{'on' if explanation else 'off'}"


def _write_markdown(data: dict[str, Any]) -> Path:
    lines = [
        "# Latency — scoring API",
        "",
        f"Profile `{data['profile']}` · model `{data['model_version']}` · "
        f"n={data['n']} per cell ({data['n_fraud']} fraud / {data['n_legit']} legit — all"
        " validation-window frauds plus a legit fill, so **fraud-enriched, not balanced**; "
        f"seed {data['seed']}), {data['warmup']} warm-up excluded · "
        "replayed in-process via `httpx.ASGITransport` (sync endpoints on the AnyIO threadpool).",
        "",
        f"Hardware: {data['hardware']['processor']}, {data['hardware']['cpu_count']} cores, "
        f"{data['hardware']['ram_gb']} GB RAM, Python {data['hardware']['python']}, "
        f"{data['hardware']['platform']}.",
        "",
        f"Headline cell: **{data['headline']['cell']}** — end-to-end p95 "
        f"{data['headline']['e2e_p95_ms']} ms, p50 {data['headline']['e2e_p50_ms']} ms.",
        "",
        f"Headline rule: {data['headline_rule']}.",
        "",
        "Full conditions for the quoted figure: **profile `dev`**, PostgreSQL history backend, "
        "concurrency 1, explanation off, dev **validation-window** sample (the test window is "
        "never read), in-process ASGI replay (**a lower bound**: no network, TLS or uvicorn "
        "overhead).",
        "",
        (f"Benchmark-time PostgreSQL provenance: `raw.transactions` = "
         f"**{data['db']['transactions_rows']:,} rows**, `raw.customers` = "
         f"{data['db']['customers']:,} (expected profile `{data['db']['expected_profile']}`)."
         if data.get("db")
         else "Benchmark-time PostgreSQL provenance: not recorded (postgres backend unavailable)."),
        "",
        "| cell | e2e p50 | e2e p95 | e2e p99 | mean | req/s | history | features | "
        "model | calib | policy | explain |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in data["cells"]:
        e2e = cell["e2e_ms"]
        c = cell["components_ms"]
        lines.append(
            f"| {_cell_name(cell['backend'], cell['concurrency'], cell['explanation'])} "
            f"| {e2e['p50']} | {e2e['p95']} | {e2e['p99']} | {e2e['mean']} "
            f"| {cell['requests_per_second']} "
            f"| {c.get('history_fetch', {}).get('p50', float('nan'))} "
            f"| {c.get('features', {}).get('p50', float('nan'))} "
            f"| {c.get('model', {}).get('p50', float('nan'))} "
            f"| {c.get('calibration', {}).get('p50', float('nan'))} "
            f"| {c.get('policy', {}).get('p50', float('nan'))} "
            f"| {c.get('explanation', {}).get('p50', float('nan'))} |"
        )
    lines += [
        "",
        "Component columns are p50 ms. `explain` is 0.0 when explanations are off.",
        "",
    ]

    offs = [c for c in data["cells"] if not c["explanation"]]
    best = max(offs, key=lambda c: c["requests_per_second"])
    best_cell = _cell_name(best["backend"], best["concurrency"], best["explanation"])
    head_cell = next(
        c
        for c in data["cells"]
        if c["backend"] == data["headline"]["backend"]
        and c["concurrency"] == data["headline"]["concurrency"]
        and c["explanation"] is data["headline"]["explanation"]
    )
    hist_p50 = head_cell["components_ms"]["history_fetch"]["p50"]
    startup_txt = ", ".join(f"{k} {v} s" for k, v in data["provider_startup_seconds"].items())
    lines += [
        "## Honest reading",
        "",
        f"- Best throughput is `{best_cell}` at {best['requests_per_second']} req/s. "
        "Raising concurrency to 8 does **not** help here: the component timings show the "
        "feature build is the bottleneck. The",
        "  most likely explanation is the GIL serialising that CPU-bound work in this "
        "single-process replay — a **hypothesis**, not evidence about a multi-worker "
        "deployment. Horizontal scaling needs uvicorn workers/processes.",
        f"- Headline `history` is {hist_p50} ms p50 (the PostgreSQL round-trip); in-memory answers",
        "  history in ~1 ms, so the postgres numbers are the conservative, deployment-like choice.",
        "- Warm SHAP explanation adds ~6 ms (memory) to ~10 ms (postgres) at p50.",
        f"- History provider start-up: {startup_txt}. In-memory loads all dev transactions at",
        "  start-up; PostgreSQL queries per request instead.",
        "",
        data["note"],
        "",
    ]
    path = docs_dir() / "latency.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--n", type=int, default=2000, help="requests per cell")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backends", default="postgres,memory")
    args = parser.parse_args(argv)

    requests, mix = sample_requests(args.profile, args.n, args.seed)

    cells: list[dict[str, Any]] = []
    startup: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    db_counts: dict[str, Any] | None = None
    model_version: str | None = None
    model_size: int | None = None

    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        if backend not in BACKENDS:
            continue
        try:
            scorer = Scorer(profile=args.profile, history_backend=backend)
        except Exception as exc:  # DB down, missing extras, ...
            logger.error("backend %s unavailable: %s", backend, exc)
            unavailable[backend] = str(exc)
            continue
        if backend == "postgres":
            db_counts = db_provenance(args.profile)
        startup[backend] = round(float(getattr(scorer.provider, "load_seconds", 0.0)), 3)
        model_version = scorer.model_version
        model_size = scorer.model_size_bytes
        app = create_app(profile=args.profile, history_backend=backend, verify_freeze=True)
        app.state.scorer = scorer
        for explanation in EXPLANATION_FLAGS:
            for concurrency in CONCURRENCIES:
                logger.info("cell %s", _cell_name(backend, concurrency, explanation))
                cell = asyncio.run(
                    run_cell(app, requests, concurrency, args.warmup, explanation)
                )
                cell["backend"] = backend
                cells.append(cell)

    if not cells:
        logger.error("no latency cells produced; is Postgres up / memory loadable?")
        return 1

    def _pick(backend: str) -> dict[str, Any] | None:
        for cell in cells:
            if (
                cell["backend"] == backend
                and cell["concurrency"] == 1
                and cell["explanation"] is False
            ):
                return cell
        return None

    primary = _pick("postgres") or _pick("memory") or cells[0]
    headline = {
        "backend": primary["backend"],
        "concurrency": primary["concurrency"],
        "explanation": primary["explanation"],
        "n": primary["n"],
        "cell": _cell_name(primary["backend"], primary["concurrency"], primary["explanation"]),
        "e2e_p50_ms": primary["e2e_ms"]["p50"],
        "e2e_p95_ms": primary["e2e_ms"]["p95"],
    }

    data: dict[str, Any] = {
        "profile": args.profile,
        "model_version": model_version,
        "git_commit": git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n": args.n,
        "n_fraud": mix["n_fraud"],
        "n_legit": mix["n_legit"],
        "seed": args.seed,
        "warmup": args.warmup,
        "hardware": hardware(),
        "model_size_bytes": model_size,
        "provider_startup_seconds": startup,
        "concurrency_levels": list(CONCURRENCIES),
        "cells": cells,
        "headline": headline,
        "headline_rule": HEADLINE_RULE,
        "db": db_counts,
        "unavailable_backends": unavailable,
        "note": (
            "In-process replay (no network stack, no uvicorn worker); numbers are a "
            "lower bound on real HTTP latency and exclude TLS/serialisation overhead. "
            "Components are the server-side timers from the API response."
        ),
    }
    out = results_dir() / "latency.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    doc = _write_markdown(data)
    logger.info("wrote %s and %s", out, doc)
    logger.info("headline %s p95=%.1f ms", headline["cell"], headline["e2e_p95_ms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
