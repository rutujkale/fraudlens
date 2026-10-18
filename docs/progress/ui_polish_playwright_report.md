# Playwright audit — FraudLens Streamlit console (ui-polish)

**Scope:** home/entry (`/`), Page 1 Transaction Analyzer, Page 2 Review Queue, Page 3 Model Performance.
**Branch/commit:** `ui-polish` @ `2dd2d8f` (audit is analysis-only — no app code changed, nothing committed).
**Date:** 2026-10-18. **Driver:** `.../Temp/opencode/ui_polish_playwright/drive_ui_polish.py`, raw findings in `findings.json` (115 rows), screenshots in `docs/progress/ui_polish_playwright_screenshots/` (8 PNGs).

**Environment:** Streamlit 1.65 headless on `http://localhost:8501`; FastAPI on `127.0.0.1:8000` with `--history-backend memory` (Docker/Postgres not running). Playwright 1.64.0 + `axe-playwright-python` 0.1.8 + Chromium headless shell 156 — **installed into `.venv` only (dev-only; deliberately NOT added to `requirements.txt`/`pyproject.toml`)**. axe uses its bundled `axe.min.js` (no CDN).

**Method per page:** console errors/warnings + failed/external network requests (any non-localhost request = finding); locked-verbatim DOM text checks; interactive flows (details below); axe-core scan at load and after interactions, **all severities, unfiltered**; responsive sanity at 1440px and 1024px (horizontal scroll, sidebar overlap in both toggle states); screenshots. Navigation via sidebar links (see deviations).

## Findings summary

| Page | PASS | [ACCESSIBILITY] | [COSMETIC] | [BUG] | Total |
|---|---|---|---|---|---|
| Entry `/` | 16 | 5 | 0 | 0 | 21 |
| Page 1 Transaction Analyzer | 24 | 8 | 2 | 0 | 34 |
| Page 2 Review Queue | 20 | 4 | 2 | 0 | 26 |
| Page 3 Model Performance | 17 | 14 | 1+1 | 1 | 34 |
| **Total** | **77** | **31** | **6** | **1** | **115** |

Console and network are clean on every page in the final run: zero JS errors/warnings across all flows, zero external/CDN requests, zero failed/4xx/5xx requests.

## Locked-verbatim wording — all PASS (identical text on every page where rendered)

- Banner: `Synthetic data demo — not a production fraud system.` (entry, 1, 2, 3)
- Capacity: `Capacity caps REVIEW only (per IST day); BLOCK and STEP_UP are chosen by expected cost and are never bounded.` (entry, 1, 2, 3)
- SHAP caption (page 1, after submit): `Contribution to the model's log-odds (not points on the risk score)`
- Drift line (page 3): `Drift: no degradation observed (0.9291 val → 0.9619 test)`
- INR CI (page 3): `361,480–1,281,632`; 83.23% headline present
- Parity (page 3): `postgres and memory providers return identical feature rows on a tiny sample; calibrated-probability parity (1e-6, n=220) is proven on dev/memory.`
- Latency (page 3): `127.031 ms p95` + full conditions sentence verbatim

## Page 1 — Transaction Analyzer (24 PASS / 8 A11Y / 2 COS)

Interactive flow all green: segmented `role=radiogroup aria-label="Preset"` found; preset 1 pre-selected on load (amount `972.09`); `Manual entry` → `2500.00`; all four presets map to distinct expected amounts (`972.09 / 23471.08 / 2246.89 / 27500.00`); submit → probability `0.9697`, action `BLOCK`, risk score `97.0`, 1 Plotly chart, SHAP caption verbatim; console clean across preset clicks + submit.

- [ACCESSIBILITY] axe `color-contrast` **serious** — steel chip text (`.fl-chip--steel`, ~11px on 12% steel tint over cream ≈ 4.1–4.4:1 < 4.5); 3 nodes at load, 7 after submit. **App CSS** (`theme.py`).
- [ACCESSIBILITY] axe `heading-order` **moderate** — `#decision`: `st.title` (h1) → `st.subheader` (h3) skips h2. **App code.**
- [ACCESSIBILITY] axe `region` **moderate** (17/28 nodes) — content outside landmarks. **Streamlit framework layout.**
- [ACCESSIBILITY] axe `aria-allowed-attr` **critical** on `.stSidebar` — **Streamlit framework DOM** (identical on all four pages; app never touches sidebar markup).
- [ACCESSIBILITY] axe `empty-table-header` **minor** — empty `th` = dataframe index column (post-submit). **Streamlit/pandas index rendering**; app could opt out with `hide_index=True`.
- [COSMETIC] offscreen SVG at `left:-10000` (Plotly hidden sprite) at 1440px and 1024px — benign, causes no scroll (scrollWidth checks PASS).

## Page 2 — Review Queue & Cost Optimization (20 PASS / 4 A11Y / 2 COS)

Flow all green: baseline `Expected loss (INR) = 109,844`, `Flagged = 534`; toggle default **off**; `is_fraud` absent from queue headers before toggle; `Realized` metric absent before toggle; slider α `0.85 → 0.7` changes KPI `109,844 → 96,440`; **Reset to frozen assumptions** restores `109,844 / α=0.85`; toggle on → `Realized loss avoided = 799,525` (server rerun observed) and `is_fraud` + `fraud_scenario` appear as queue column headers; console clean across slider/reset/toggle.

Note: the queue grid is **horizontally virtualized** — `is_fraud`/`fraud_scenario` header cells exist only after scrolling the inner scroller right (`scrollWidth` 1110→1257 proves the server re-rendered 12 columns). Driver scrolls before reading; the columns are genuinely present.

- [ACCESSIBILITY] axe `color-contrast` **serious**, 11 nodes (steel chips/text) — **app CSS**.
- [ACCESSIBILITY] axe `heading-order` **moderate** — `#queue` (h1→h3). **App code.**
- [ACCESSIBILITY] axe `region` **moderate** (21 nodes) + `aria-allowed-attr` **critical** `.stSidebar` — **framework**.
- [COSMETIC] offscreen Plotly SVG (both widths).

## Page 3 — Model Performance & Drift (17 PASS / 14 A11Y / 2+1 COS / 1 BUG)

Flow all green: threshold `0.19 → 0.6` updates the **validation** confusion matrix (pixel-diff) while the **test-window matrix stays frozen** (expected — Part 12 window is evaluated once); `Limitations` expander renders non-empty (4885 chars, starts `Candid list. This is a portfolio / research system…`); console clean.

- **[BUG] `point estimate` literal not in DOM.** Spec asked to report exact text; the headline reads `…INR CI of 361,480–1,281,632 (point 773,067 INR)`. The number and CI are present; only the phrase differs — **wording-contract mismatch, no data missing.**
- [ACCESSIBILITY] 4 × **image without `alt`** — `st.image(..., caption=)` does not emit an `alt` attribute (4 report figures). **App code.**
- [ACCESSIBILITY] axe `empty-table-header` **minor**, 5 nodes (`table[aria-rowcount="4"]` index column) — **framework/index rendering**, `hide_index` would remove.
- [ACCESSIBILITY] axe `color-contrast` **serious** (9 nodes at load, 3 after interactions) — steel chips. **App CSS.**
- [ACCESSIBILITY] axe `heading-order` **moderate** — `#pre-registered-outcomes-part-12-test-window` (h1→h3). **App code.**
- [ACCESSIBILITY] axe `region` (37/38 nodes) + `aria-allowed-attr` `.stSidebar` — **framework**.
- [COSMETIC] offscreen Plotly SVG (both widths).

## Entry page (16 PASS / 5 A11Y)

Banner + capacity verbatim; no horizontal scroll at 1440/1024; sidebar 0–263 vs main 263–1024, no overlap in either toggle state; zero external requests; zero failed requests; console clean.

- [ACCESSIBILITY] palette pair computed by driver: **steel link `#4A6FA5` on cream = 4.15:1** (AA-large-only) and **gold `#B08D57` on white = 3.09:1** (AA-large-only) — normal-size text would fail AA. **App theme tokens.**
- [ACCESSIBILITY] `region` (7 nodes) + `aria-allowed-attr` `.stSidebar` — **framework**.

## Environment findings (reported, not fixed)

1. **Screenshots directory is not gitignored** — `docs/progress/ui_polish_playwright_screenshots/` shows as `??` in `git status`; `git check-ignore` exits 1. Left untouched per instructions.
2. **Playwright toolchain is dev-only** — `playwright`, `axe-playwright-python`, `greenlet`, `pyee` live in `.venv` only; not in `requirements.txt`/`pyproject.toml`.
3. Direct deep-links to `/Transaction_Analyzer` etc. make the Streamlit framework issue two `_stcore/health` + `_stcore/host-config` 404s; **sidebar (client-side) navigation is clean** — final runs used sidebar links only, hence zero failed requests.

## Deviations & driver caveats

- Audit-only: no app code, config, or `.gitignore` modified; this report is intentionally uncommitted.
- Streamlit 1.65 DOM specifics required driver adaptations (case-insensitive metric reads — theme CSS uppercases labels; sliders = `input[type=range]` moved by track click; `st.toggle` = `role=switch` input clicked via JS; segmented = `div[role=radiogroup]` — clicking the already-selected preset deselects, so the pre-selected default is read on load; download-button `href`s are lazy/empty so header/column checks use the grid DOM).
- Toggle/label assertions use dataframe column headers, not page text (the toggle's own label contains the string `is_fraud`).
- SHAP caption is only rendered after submit by design — checked in the post-submit pass.
- Validation matrix compared via pixel-diff because cells render off-DOM.

## Open items (for a follow-up, not done here)

Fixable in app code: steel chip contrast (`theme.py`), h1→h3 heading order (`st.header`/markdown), `alt` on the 4 figures, `hide_index=True` on index-column tables, and the `point estimate` wording (or accept `(point …)` as final). Framework-owned: `.stSidebar` aria-allowed-attr, `region`.
