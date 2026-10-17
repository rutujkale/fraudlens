"""Precomputed data for the Streamlit analyst app (Part 14).

Public helpers:

* :mod:`fraudlens.appdata.artifacts` — build/load ``artifacts/app/<profile>/``.
* :mod:`fraudlens.appdata.recompute` — thin wrappers over the Part 07 cost/policy
  code so the app (and its tests) reuse the package, never re-implement it.
"""

from __future__ import annotations
