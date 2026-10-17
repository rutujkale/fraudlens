"""Part 15D tests: the doc renderer fails on unresolved keys and resolves every template."""

from __future__ import annotations

import subprocess
import sys

import pytest

from fraudlens.paths import REPO_ROOT
from fraudlens.reporting.render import (
    PLACEHOLDER,
    TEMPLATES,
    UnresolvedPlaceholder,
    build_context,
    render,
    resolve,
)

# Deterministic test counts keep the suite fast; the real counts are produced by
# pytest collection when the renderer runs for real.
COUNTS = {"n_tests": 999, "n_leakage_tests": 12}


def test_unresolved_placeholder_raises():
    with pytest.raises(UnresolvedPlaceholder, match="no value"):
        render("hello {{nowhere.at.all}}", {})


def test_none_value_raises():
    with pytest.raises(UnresolvedPlaceholder, match="None"):
        render("{{k}}", {"k": None})


def test_format_spec_applied():
    assert resolve("meta.n_features:03d", {"meta.n_features": 53}) == "053"


def test_derived_keys_are_built_from_results():
    ctx = build_context(test_counts=COUNTS)
    assert ctx["headline.model.xgb_ci"].startswith("0.9")
    assert "CI" in ctx["headline.decision.reduction"]
    assert ctx["headline.drift.line"].startswith("no degradation observed")


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_real_template_fully_resolves(name):
    template, _out = TEMPLATES[name]
    ctx = build_context(test_counts=COUNTS)
    rendered, used = render(template.read_text(encoding="utf-8"), ctx)
    assert used, "template has no placeholders"
    assert not PLACEHOLDER.findall(rendered), "unresolved placeholder left in output"


def test_cli_exits_nonzero_on_unresolved_placeholder(tmp_path):
    template = tmp_path / "t.md"
    template.write_text("value = {{nope}}", encoding="utf-8")
    out = tmp_path / "out.md"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/render_readme.py",
            "--template",
            str(template),
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "unresolved placeholder" in proc.stderr
    assert not out.exists()
