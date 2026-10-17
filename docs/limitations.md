# Limitations

Candid list. This is a **portfolio / research system, not a production banking
system**; nothing here is a claim of production readiness. Numbers are cited from
the committed results, not estimated.

## Synthetic data and simulation bias

- All primary results are on **synthetic data** produced by the Part 02 simulator.
  Every subgroup, archetype and fairness statement describes the **simulator's
  assumptions**, not real demographics or real customers
  (`docs/fairness_audit.md`, `docs/methodology.md`).
- The simulator's fraud archetypes and hard negatives are *chosen*. A different
  adversary mix would move every ranking metric. The experiments are honest about
  the simulator, not about the world.
- Fraud labels are simulated with a configured `label_delay_days`
  (`[2, 7]` on `dev`); real chargebacks can take far longer and can reverse.

## Sim-to-real gap

- The only real data in the project is the IEEE-CIS benchmark, used for a
  secondary validation (`docs/external_validation.md`,
  `results/sim_to_real.json`). The gap is measured, not hand-waved.
- The IEEE-CIS adapter relies on a **UID heuristic** to build customer-like ids;
  that heuristic is an approximation, and transferable features are a subset of
  the FraudLens registry. Benchmark transfer is evidence of direction, not parity.

## Evaluation power

- The locked test window holds **393 frauds** (0.987% prevalence), so every
  PR-AUC confidence interval is wide (XGBoost `0.9340–0.9798`). The point estimate
  is reported with the interval for exactly this reason.
- The cost-reduction headline is **83.23% as a point estimate**; the interval is
  reported in **INR** (`361,480–1,281,632`). A percentage CI is not quoted because
  the ratio's denominator is itself resampled.

## Decisions and signals a real institution would add

- **No network / device / graph signals.** Real fraud systems lean on device
  fingerprints, IP / geo velocity, email and phone graphs, and consortium data.
  FraudLens has behavioral aggregates only.
- **No chargebacks or investigator feedback loop.** The policy is trained on
  simulated labels; a real system would learn from analyst dispositions and
  chargebacks, which arrive late and biased (only reviewed cases get feedback).
- **No model-risk governance.** A real deployment needs independent validation,
  challenger models, documented approval and monitoring sign-off — out of scope
  here.
- **Fraud loss rate `α = 0.85`** and the friction/review costs are *assumptions*
  (`results/freeze.json:cost_params`). The policy is only as honest as those
  numbers; a sensitivity sweep is reported, not a guarantee.

## Honest gaps in this build (deviations from the plan)

- **Profile `main` was never generated.** The plan's acceptance command names
  `--profile main`; every published result is on `dev`. We did not generate
  `main`: it would mint new data and a new test window after the freeze, breaking
  the frozen-hash and test-window discipline. Recorded in
  `docs/progress/part_15.md`.
- **`ruff format` is not the repository style and is not enforced.** `ruff check`
  is clean, but `ruff format --check .` would reformat the existing tree; a
  bulk reformat was deliberately deferred to its own commit so it would not bury
  the Part 15 diff.
- **Latency is a lower bound.** The published p95 is an in-process ASGI replay
  (backend `postgres`, concurrency 1, explanation off, `n=2000`, fraud-enriched);
  it excludes the network hop a real deployment would have
  (`results/latency.json`, `docs/latency.md`).
- **Capacity never bound on test.** The amount-aware policy flagged `5.9/day`
  against a `23/day` cap with `0` demotions, so `enforce_capacity` was never
  exercised — the capacity-aware result is *not* evidence the cap works
  (`results/final_results.json:decision.meta.n_demoted`).
- **The anomaly experiment is null.** The adopted Isolation Forest was re-run
  after the cold-start fix and **not adopted** (bagged CIs include 0)
  (`results/final_results.json:drift`, `docs/experiments/09_unseen_fraud.md`).
- **Synthetic reasoning only for served paths.** API-context (`dev`) checks load
  the frozen dev bundle, which is gitignored and built locally; CI covers the
  tiny-built suite and the invariants, not dev-specific performance
  (`.github/workflows/ci.yml`).
- **Two real-data accesses and one test-window access are logged**
  (`results/test_access_log.csv`, `results/test_access_log_external.csv`); this is
  the disclosed and intended count, not a leak.

## What I would change before real deployment

1. Replace simulated labels with a real, delayed-label pipeline and add
   chargeback / investigator-feedback features.
2. Add device-network-graph signals and a proper entity-resolution UID.
3. Re-fit `α` and friction/review costs from finance and operations, and run the
   policy as a challenger under model-risk governance.
4. Re-baseline the temporal split on live data and stress-test the capacity cap
   under a real review volume (the cap never bound on this test window).
5. Re-measure latency end-to-end through the real network path and add load
   testing beyond concurrency 1.
