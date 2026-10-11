"""Generic point-in-time window helpers (Part 05A).

The strict window is ``[T - w, T)``: the current row and every row sharing its
timestamp are excluded.

``window_bounds`` is the **single place `np.searchsorted` is called anywhere in
the feature code**. Every aggregate funnels through it, so replacing
``window_bounds`` (see the leakage canary in ``tests/test_leakage.py``)
exercises the entire feature set at once. ``tests/test_leakage.py`` additionally
asserts that no other module imports/calls ``searchsorted``.
"""

from __future__ import annotations

import numpy as np

# Effectively-unbounded window for "lifetime" aggregates. Kept well inside the
# datetime64[ns] representable range (1678-2262) so ``T - w`` never overflows.
LIFETIME_WINDOW = np.timedelta64(36500, "D")

_DAY = np.timedelta64(1, "D")
_MINUTE = np.timedelta64(1, "m")


def window_bounds(ts: np.ndarray, t: np.datetime64, w: np.timedelta64) -> tuple[int, int]:
    """Return ``(start, end)`` slicing sorted ``ts`` to rows in ``[t - w, t)``.

    ``ts`` must be sorted ascending. The returned indices are half-open, so
    ``ts[start:end]`` contains exactly the rows strictly earlier than ``t`` that
    fall inside the window and never the current row or a same-timestamp peer.
    """
    start = int(np.searchsorted(ts, t - w, side="left"))
    end = int(np.searchsorted(ts, t, side="left"))
    return start, end


def count_in_window(ts: np.ndarray, t: np.datetime64, w: np.timedelta64) -> int:
    """Number of strictly-prior rows of ``ts`` in ``[t - w, t)``."""
    start, end = window_bounds(ts, t, w)
    return end - start


def sum_in_window(
    cumsum: np.ndarray, ts: np.ndarray, t: np.datetime64, w: np.timedelta64
) -> float:
    """Sum of a value whose length-``len(ts)+1`` cumulative array is ``cumsum``."""
    start, end = window_bounds(ts, t, w)
    return float(cumsum[end] - cumsum[start])


def lifetime_bounds(ts: np.ndarray, t: np.datetime64) -> tuple[int, int]:
    """``(0, end)`` for the full strictly-prior history (see ``LIFETIME_WINDOW``)."""
    return window_bounds(ts, t, LIFETIME_WINDOW)
