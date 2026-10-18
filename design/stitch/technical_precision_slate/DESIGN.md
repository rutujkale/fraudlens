---
name: Technical Precision Slate
colors:
  surface: '#fff9ec'
  surface-dim: '#dfdacd'
  surface-bright: '#fff9ec'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f9f3e6'
  surface-container: '#f3ede0'
  surface-container-high: '#eee8db'
  surface-container-highest: '#e8e2d5'
  on-surface: '#1d1c14'
  on-surface-variant: '#45464c'
  inverse-surface: '#333028'
  inverse-on-surface: '#f6f0e3'
  outline: '#76777d'
  outline-variant: '#c6c6cd'
  surface-tint: '#575e71'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#141b2c'
  on-primary-container: '#7d8398'
  inverse-primary: '#c0c6dc'
  secondary: '#625e54'
  on-secondary: '#ffffff'
  secondary-container: '#e8e2d5'
  on-secondary-container: '#68645a'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#001c3b'
  on-tertiary-container: '#6185bc'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dce2f9'
  primary-fixed-dim: '#c0c6dc'
  on-primary-fixed: '#141b2c'
  on-primary-fixed-variant: '#404659'
  secondary-fixed: '#e8e2d5'
  secondary-fixed-dim: '#cbc6ba'
  on-secondary-fixed: '#1d1c14'
  on-secondary-fixed-variant: '#4a473d'
  tertiary-fixed: '#d5e3ff'
  tertiary-fixed-dim: '#a7c8ff'
  on-tertiary-fixed: '#001c3b'
  on-tertiary-fixed-variant: '#1e477b'
  background: '#fff9ec'
  on-background: '#1d1c14'
  surface-variant: '#e8e2d5'
  dark-navy: '#0F1626'
  warm-cream: '#EDE7DA'
  golden-brown: '#B08D57'
  steel-blue: '#4A6FA5'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: -0.01em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '400'
    lineHeight: 12px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 0.75rem
  gutter-compact: 0.5rem
  margin: 1rem
  margin-panel: 0.75rem
  space-xs: 0.25rem
  space-sm: 0.375rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
---

## Brand & Style

This design system is engineered for high-throughput fraud operations, risk intelligence, and quantitative review. The target users are forensic analysts, trust & safety investigators, and machine learning auditors operating in high-stress, decision-critical environments. The experience prioritizes cognitive stamina, zero optical latency, and uncompromising analytical precision over ornamentation.

The visual style blends **Technical Minimalism** with **Structured Data Densification**:
- **Information Architecture First:** Layouts favor extreme data density, flat architectural hierarchy, and disciplined alignment over decorative whitespace.
- **Architectural Contrast:** Visual hierarchy is anchored by a high-contrast relationship between a primary dark navy foundation and warm cream light surfaces, enriched with steel blue utility actions and muted golden brown highlights.
- **Calm Authority:** The balanced dual-surface palette eliminates eye fatigue during long triage shifts while retaining razor-sharp definition for micro-typography, tabular figures, and decision tags.
- **Unambiguous Telemetry:** Every color accent serves a functional verdict. Actions, states, and telemetry markers must remain clear, intentional, and readable without relying solely on color indicators.

## Colors

The color system strictly adheres to an authoritative 4-color palette designed for high contrast and analytical clarity:

### Core Palette
- **Primary Dark Navy (`#0F1626`):** The grounding deep tone. Governs top-level application navigation bars, dark toolbars, dark operational surfaces, high-contrast text on cream/light backgrounds, and dark structural outlines.
- **Warm Off-White / Cream (`#EDE7DA`):** The primary light baseline. Used for main content backgrounds, subtle light surfaces, investigative work containers, and card panels. On dark navy surfaces, it provides soft, high-legibility foreground text.
- **Muted Golden Brown (`#B08D57`):** The specialized accent. Reserved for gold accents, forensic highlights, selected decorative indicators, active investigative filters, key metrics, and special badges.
- **Steel Blue (`#4A6FA5`):** The functional interactive driver. Deployed across primary buttons, interactive links, interactive accents, selection states, and operational highlights.

### Contrast & Surface Application Rules
- **Light & Container Surfaces (`#EDE7DA`):** Text and primary graphic icons must strictly render in Primary Dark Navy (`#0F1626`) to preserve WCAG AAA readability. Secondary body copy utilizes `#0F1626` at 75% opacity or paired steel blue accents.
- **Dark Navy Surfaces (`#0F1626`):** Headers and body copy render in Warm Off-White (`#EDE7DA`) or clean white (`#FFFFFF`). Selected states and subtle framing use Muted Golden Brown (`#B08D57`) or Steel Blue (`#4A6FA5`).
- **Interactive Focus:** Interactive controls maintain clear affordances via Steel Blue (`#4A6FA5`) with Warm Off-White (`#EDE7DA`) or white foreground typography.

## Typography

The typography architecture uses a two-family system: **Inter** handles narrative structure, UI shell labels, and narrative analysis; **JetBrains Mono** controls all machine-generated data, cryptographic hashes, model identifiers, log-odds values, IP subnets, and tabular currency values.

- **Tabular Figures:** All numeric rendering in Inter must enforce `font-feature-settings: "tnum" 1, "cv05" 1, "cv11" 1` to guarantee vertical column alignment across rapidly updating queues.
- **Code & Metric Sizing:** Metric readouts (e.g., Risk Score `94.2`, Velocity count `18/1h`) utilize `label-lg` with medium weight. Column headers in dense queues use `label-md` uppercase with a `+0.04em` tracking offset for rapid horizontal scanning.
- **Suppressed Text:** System-suppressed values must display with monospaced italics at `label-sm` (`font-style: italic`).

## Layout & Spacing

This layout system operates on an ultracompact base increment. High-volume review workstations require dense data packed cleanly without sacrificing operational clarity.

- **Triage Paneling:** The primary workspace follows a 3-pane split:
  1. *Queue Navigation & Filter Strip:* 280px fixed width, structured over `#0F1626` navigation or `#EDE7DA` container panels.
  2. *Primary Table / Feed:* Fluid column dynamic width with 12px structural gutters.
  3. *Forensic Inspector Drawer / Deep Dive Panel:* 440px to 520px dockable side-sheet framed by crisp borders.
- **Table Row Metrics:** Default table rows measure strictly 32px or 36px in height (`space-sm` / `space-md` distribution) to maximize simultaneous alert visibility. Expanded investigative rows ceiling at 48px.
- **Responsive Adaptations:**
  - *Desktop (>1440px):* Full 3-pane layout visible simultaneously.
  - *Compact Desktop (1024px–1439px):* Side-sheet switches to an overlay dock or collapsible tab to maintain tabular column readability.
  - *Mobile/Tablet (<1023px):* Linear stacked triage cards; tabular multi-column views convert to labeled metric tuples.

## Elevation & Depth

To avoid visual smearing and preserve legibility under variable workstation lighting, this system rejects soft blur drop shadows in favor of **Crisp Structural Borders** and **Tonal Layer Stacking**.

- **Primary Nav & Shell Foundation (`#0F1626`):** The structural anchor for headers, sidebars, and high-impact framing.
- **Base Canvas (`#EDE7DA`):** The primary warm background surface for light views.
- **Contained Surface Level 1 (`#FFFFFF` or elevated `#EDE7DA`):** Work panels, tables, toolbars, and inspection cards, outlined with a 1px solid border in `#0F1626` (at 15% opacity) or `#B08D57` for priority elements.
- **Popovers & Overlays:** Tooltips, metric breakdown menus, filter popovers, and model explanations use a strict 1px border in `#0F1626` paired with an ultra-short occlusion shadow: `0 2px 4px rgba(15, 22, 38, 0.12), 0 1px 2px rgba(15, 22, 38, 0.08)`.
- **Active Focus & Row Selection:** Selected rows do not cast elevation; they display an inset 2px left border accent in Steel Blue (`#4A6FA5`) or Muted Golden Brown (`#B08D57`) paired with a subtle surface tint.

## Shapes

The design system enforces crisp, utilitarian geometry (`roundedness: 1`):

- **Buttons, Inputs, and Badges:** Standard corner radius is fixed at `0.25rem` (4px). This preserves vertical rhythm in condensed tables and prevents visual bloating.
- **Cards, Drawers, and Inspector Panels:** Encapsulated using `0.375rem` (6px) corners or `0.5rem` (8px). Never exceed 8px on any structural element.
- **Pill Badges & Tags:** Status chips, triage tokens, and tags use `0.25rem` (4px) with structural borders, avoiding overly rounded circular pills to preserve an analytical, instrument-grade appearance.

## Components

### Buttons & Interactive Controls
- **Primary Action (Execute / Commit):** Solid Steel Blue (`#4A6FA5`) background, Warm Off-White (`#EDE7DA`) or `#FFFFFF` text, 4px radius, compact padding (`space-xs` vertical, `space-md` horizontal), font weight 500. Focus outline: 2px offset in Steel Blue (`#4A6FA5`).
- **Accent / Special Action:** Muted Golden Brown (`#B08D57`) background with Primary Dark Navy (`#0F1626`) text for high-value audit authorizations.
- **Secondary / Ghost Actions:** Transparent surface, 1px border in `#0F1626` (20% opacity) on cream, or 1px border in `#EDE7DA` (20% opacity) on navy. Text matches corresponding surface high-contrast tone.

### Data Tables
- **Grid Architecture:** Explicit horizontal row dividers in 1px muted `#0F1626` (12% opacity) on cream backgrounds. Explicit vertical divisions between functional column groups.
- **Header Cells:** Background `#EDE7DA` or `#0F1626`, uppercase `JetBrains Mono` at 10px (`label-sm`), letter spacing `0.04em`. Contrast rules: `#0F1626` text on cream headers; `#EDE7DA` text on navy headers.
- **Hover & Interaction:** Row hover shifts instantly to a subtle overlay. Active/selected rows display a `#4A6FA5` 2px indicator bar.

### Status Badges & Selected Elements
- **Badges & Chips:** Height 20px. Monospaced font at 11px (`label-md`). 4px border radius.
- **Highlight Badges:** Muted Golden Brown (`#B08D57`) background or 1px border with `#0F1626` text.
- **Operational / Interactive Chips:** Steel Blue (`#4A6FA5`) outline or fill with high-contrast text.

### Inputs & Queue Filters
- **Search & Param Fields:** Height 32px (`body-sm`), 1px solid `#0F1626` (25% opacity) or `#4A6FA5` when focused, background `#FFFFFF` or elevated `#EDE7DA`. Text rendered in `#0F1626`.
- **Monospace Input Masks:** IP addresses, CIDR blocks, card BINs, and transaction UUIDs require `JetBrains Mono` at 12px.

### Cards & Container Panels
- **Standard Card:** Warm Off-White (`#EDE7DA`) background, 1px border in `#0F1626` (15% opacity), typography in Primary Dark Navy (`#0F1626`).
- **Dark Structural Card / Panel:** Primary Dark Navy (`#0F1626`) background, 1px border in `#B08D57` (30% opacity) or `#4A6FA5`, typography in Warm Off-White (`#EDE7DA`).