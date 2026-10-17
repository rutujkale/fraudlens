"""Part 15D reporting helpers: render docs whose numbers come from results."""

from fraudlens.reporting.render import (
    TEMPLATES,
    UnresolvedPlaceholder,
    build_context,
    render,
    render_templates,
)

__all__ = [
    "TEMPLATES",
    "UnresolvedPlaceholder",
    "build_context",
    "render",
    "render_templates",
]
