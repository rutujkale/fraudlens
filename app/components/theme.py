"""Shared visual chrome for the analyst console (UI polish).

One place applies the DESIGN.md tokens that Streamlit's theme cannot express:
a single additive CSS block (tabular figures, mono machine data, chips) and two
tiny HTML helpers. This module holds **no** data, numbers or business logic.

Palette (from ``design/stitch/technical_precision_slate/DESIGN.md``):
navy ``#0F1626`` · warm cream ``#EDE7DA`` · golden brown ``#B08D57`` ·
steel blue ``#4A6FA5`` · success green ``#1B6A4C`` · error ``#BA1A1A``.
"""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
  --fl-navy: #0F1626;
  --fl-cream: #EDE7DA;
  --fl-gold: #B08D57;
  --fl-steel: #4A6FA5;
  --fl-outline: #D5CDBD;
  --fl-green: #1B6A4C;
  --fl-error: #BA1A1A;
}
/* Tabular figures so metric columns stay aligned while the queue updates. */
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
  font-feature-settings: "tnum" 1, "cv05" 1, "cv11" 1;
}
[data-testid="stMetricValue"] { color: var(--fl-navy); }
[data-testid="stMetricLabel"] p {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.72rem;
}
/* Machine-generated data (versions, ids, log-odds) in the mono face. */
[data-testid="stMarkdownContainer"] code {
  color: var(--fl-navy);
  font-family: "JetBrains Mono", ui-monospace, "Cascadia Mono", monospace;
}
/* Hairline rule under panel headings (DESIGN.md card headers). */
.fl-rule { border-bottom: 1px solid var(--fl-outline); margin: 0.15rem 0 0.6rem 0; }
.fl-chip {
  display: inline-block; padding: 1px 7px; margin: 0 4px 4px 0;
  border-radius: 0.25rem; font-size: 0.70rem; font-weight: 600;
  letter-spacing: 0.02em; border: 1px solid var(--fl-outline);
  background: #FFFFFF; color: var(--fl-navy);
  font-family: "JetBrains Mono", ui-monospace, "Cascadia Mono", monospace;
}
.fl-chip--gold {
  background: rgba(176,141,87,0.15); border-color: rgba(176,141,87,0.40);
}
.fl-chip--steel {
  background: rgba(74,111,165,0.12); border-color: rgba(74,111,165,0.40);
  color: var(--fl-steel);
}
.fl-chip--ok {
  background: rgba(27,106,76,0.12); border-color: rgba(27,106,76,0.40);
  color: var(--fl-green);
}
.fl-chip--warn {
  background: rgba(186,26,26,0.10); border-color: rgba(186,26,26,0.40);
  color: var(--fl-error);
}
</style>
"""


def inject() -> None:
    """Inject the shared stylesheet once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def chip(text: object, tone: str = "") -> str:
    """One status chip as HTML. ``tone`` is gold/steel/ok/warn (or ``""``)."""
    cls = f"fl-chip fl-chip--{tone}" if tone else "fl-chip"
    return f'<span class="{cls}">{text}</span>'


def chips(items: list[tuple[object, str]]) -> None:
    """A row of status chips: ``[(text, tone), ...]``."""
    st.markdown("".join(chip(text, tone) for text, tone in items), unsafe_allow_html=True)


def rule() -> None:
    """A hairline rule, used under custom panel headings."""
    st.markdown('<div class="fl-rule"></div>', unsafe_allow_html=True)
