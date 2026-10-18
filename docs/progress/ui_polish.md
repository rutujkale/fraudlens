# UI polish — Streamlit analyst console (branch `ui-polish`)

Off-part visual/structural pass on `app/` against the "Technical Precision Slate"
mockups. **No** change under `src/fraudlens/`, `results/`, `artifacts/`, `config/`;
no merge to `main`.

## What was built

- **Theme.** `.streamlit/config.toml` translates
  `design/stitch/technical_precision_slate/DESIGN.md` into Streamlit theme tokens
  (cream canvas `#EDE7DA`, navy text `#0F1626`, steel-blue primary `#4A6FA5`,
  outline-variant `#D5CDBD`, `0.25rem` radii, Inter / JetBrains Mono, chart
  palettes). Font *names* only — no CDN, no bundled binaries; the browser falls
  back if the families are absent.
- **`app/components/theme.py`** (new). The tokens Streamlit's theme cannot express:
  one additive CSS block (tabular figures on metrics, mono for machine data, chips,
  hairline rules) plus `chip()` / `chips()` / `rule()`. No data, no logic.
- **`app/components/context.py`.** Now also applies the stylesheet, shows
  profile/model/policy chips, and exposes the Part 15 engineering strings —
  `engineering_latency()`, `engineering_latency_note()`, `engineering_parity()` —
  read **directly** from `fraudlens.reporting.render.build_context`.
- **Pages.** Entry, 1, 2, 3 restyled: preset as `segmented_control`, two-pane
  "Transaction Analyzer" workspace, dense KPI row + invariant banner on the queue
  page, and page 3 gained a *pre-registered outcomes* H1–H4 card row plus an
  *Engineering* strip. Chart palette (`charts.py`) remapped to DESIGN tokens.
- **Tests** (`tests/test_app_smoke.py`): locked-wording checks (banner, capacity
  note, SHAP caption, drift line verbatim), engineering-strip equality against
  `reporting.render`, and a **money sweep** with a negative canary.

## Engineering strip — path taken (decision point resolved)

**On-screen, direct render, zero duplication.** `components/context.py` imports
`build_context(test_counts={"n_tests": None, "n_leakage_tests": None})` (the stub
bypasses `collect_test_counts`, which scans the suite). The cross-check
`test_page3_engineering_strip_matches_render_context` asserts the latency, latency
conditions and parity strings on the page equal the accessor output. The
`build_context` call is cheap here and stable under `AppTest`, so the on-screen
path was kept — no docs-only fallback was needed.

## Vendored reference

`stitch_fraudlens_ui_architecture.zip` extracted to `design/stitch/{transaction_analyzer,
review_queue_cost_optimization,model_performance_drift,technical_precision_slate}/`
(read-only, never imported). 7/7 files SHA256-match the source zip.

## Commands run (actual output summaries)

| Command | Result |
|---|---|
| `python -m pytest tests/test_app_smoke.py -q` | `13 passed, 22 warnings in 11.77s` |
| `python -m ruff check app tests` | `All checks passed!` |
| `git diff --stat main` | 7 files changed, 403 insertions(+), 128 deletions(-) |
| `git status --short` | modified `app/*`, `tests/*`; untracked `.streamlit/`, `app/components/theme.py`, `design/` |

## Deliverables requested at hand-off

1. **Engineering-strip path:** on-screen direct render from `reporting.render`
   (see above); chosen because it is a single source of truth and the call is cheap
   in `AppTest`.
2. **Negative test:** `test_money_sweep_fails_on_injected_amount` appends
   `" Unaccounted demo figure ₹500."` to the rendered page-3 text and asserts the
   sweep returns `["500"]`; it is not in `_known_money()` (tiny costs are
   review 5 / friction 10 / block 30…), so it is flagged → observed failing as
   required. The clean page sweeps to `[]`.
3. **pytest:** `13 passed, 22 warnings in 11.77s`.
4. **ruff:** `All checks passed!`.
5. **Diff scope:** only `app/`, `tests/`, `.streamlit/`, `design/` (docs added).
6. **git log:** nothing committed yet on `ui-polish`; branch is `main` + working
   tree (see repo state). No merge to `main`.

## Deviations / notes

- **Money sweep shape.** The mockups put `₹` amounts everywhere; the existing app
  already renders INR amounts as bare tabular numbers with an `(INR)` label and has
  **no** `₹` symbol. The sweep therefore matches `₹…` and INR-adjacent amounts and
  is proven meaningful by the canary, rather than by incidental page values.
- **Fonts.** Inter / JetBrains Mono are referenced by name only; no binary or CDN
  was added (falls back to system UI / monospace).
- **`use_container_width` deprecation warnings** in the test run are pre-existing
  (present on `main`); not touched to keep the diff scoped.
- No fabricated mockup values, fake ids or fake system chrome (uptime, KAFKA,
  session ids, named analyst) were introduced.

## Open issues

- None blocking. Visual review on a live `streamlit run app/streamlit_app.py`
  against the mockups is the remaining human check.
