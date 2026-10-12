"""Probability calibration on raw scores (Part 07A).

We deliberately avoid ``CalibratedClassifierCV(cv="prefit")`` (deprecated) and
operate on scores directly. Two calibrators share the interface
``fit(raw, y)`` / ``predict(raw)`` and are joblib-serializable:

* ``SigmoidCalibrator`` — Platt scaling: logistic regression on ``logit(raw)``.
* ``IsotonicCalibrator`` — ``IsotonicRegression(out_of_bounds="clip")``.

A model trained with ``scale_pos_weight`` produces *distorted* probabilities
(its mean predicted probability is far above the base rate); calibration is what
repairs that, so raw-vs-calibrated metrics are reported side by side.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from fraudlens.evaluation.metrics import brier, ece, log_loss

_EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


class Calibrator(Protocol):
    """Common calibrator interface."""

    name: str

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Calibrator: ...
    def predict(self, raw: np.ndarray) -> np.ndarray: ...


class SigmoidCalibrator:
    """Platt scaling: logistic regression on the log-odds of the raw score."""

    name = "sigmoid"

    def __init__(self) -> None:
        self._lr = LogisticRegression(solver="lbfgs", max_iter=1000)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> SigmoidCalibrator:
        x = _logit(raw).reshape(-1, 1)
        self._lr.fit(x, np.asarray(y).astype(int))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        x = _logit(raw).reshape(-1, 1)
        return self._lr.predict_proba(x)[:, 1]


class IsotonicCalibrator:
    """Isotonic regression (piecewise-constant monotone map)."""

    name = "isotonic"

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        self._iso.fit(np.asarray(raw, dtype=float), np.asarray(y).astype(float))
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._iso.predict(np.asarray(raw, dtype=float))


def fit_calibrators(
    raw: np.ndarray, y: np.ndarray, names: tuple[str, ...] = ("sigmoid", "isotonic")
) -> dict[str, SigmoidCalibrator | IsotonicCalibrator]:
    """Fit each requested calibrator on ``(raw, y)`` and return them by name."""
    out: dict[str, SigmoidCalibrator | IsotonicCalibrator] = {}
    factories = {"sigmoid": SigmoidCalibrator, "isotonic": IsotonicCalibrator}
    for name in names:
        if name not in factories:
            raise ValueError(f"unknown calibrator {name!r}")
        out[name] = factories[name]().fit(raw, y)
    return out


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-frequency reliability bins: mean predicted vs observed per bin."""
    y = np.asarray(y).astype(float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        rows.append(
            {"mean_pred": float(p[b].mean()), "observed": float(y[b].mean()), "count": int(len(b))}
        )
    return rows


def calibration_table(
    y: np.ndarray,
    raw: np.ndarray,
    calibrators: dict[str, SigmoidCalibrator | IsotonicCalibrator],
    bins: int = 10,
) -> list[dict[str, float]]:
    """Brier/ECE/log-loss/mean-predicted for raw and each calibrator (before vs after)."""
    y = np.asarray(y).astype(int)
    rows = [
        {
            "calibrator": "raw",
            "n": int(len(y)),
            "observed_rate": float(y.mean()),
            "mean_pred": float(np.mean(raw)),
            "brier": brier(y, raw),
            "ece": ece(y, raw, bins),
            "log_loss": log_loss(y, raw),
        }
    ]
    for name, cal in calibrators.items():
        p = cal.predict(raw)
        rows.append(
            {
                "calibrator": name,
                "n": int(len(y)),
                "observed_rate": float(y.mean()),
                "mean_pred": float(np.mean(p)),
                "brier": brier(y, p),
                "ece": ece(y, p, bins),
                "log_loss": log_loss(y, p),
            }
        )
    return rows


def paired_cluster_bootstrap_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    n: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired cluster bootstrap CI of ``metric_fn(a) - metric_fn(b)``.

    The same customer resample is applied to both scores, so this tests whether the
    two calibrators differ rather than comparing two independent resamples.
    """
    y = np.asarray(y)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    groups = np.asarray(groups)
    point = float(metric_fn(y, score_a) - metric_fn(y, score_b))

    unique = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in unique]
    n_groups = len(unique)
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=float)
    for i in range(n):
        chosen = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([idx_by_group[c] for c in chosen])
        stats[i] = metric_fn(y[idx], score_a[idx]) - metric_fn(y[idx], score_b[idx])

    finite = stats[np.isfinite(stats)]
    if len(finite) < 2:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def choose_calibrator(
    y_thr: np.ndarray,
    raw_thr: np.ndarray,
    calibrators: dict[str, SigmoidCalibrator | IsotonicCalibrator],
    groups: np.ndarray,
    *,
    n_cal_positives: int,
    min_cal_positives: int = 150,
    n_boot: int = 500,
    seed: int = 42,
    metric_fn=brier,
) -> dict:
    """Pick the calibrator by Brier on ``val_thr``; ties/instability -> sigmoid.

    Decision rule: start from the lower point Brier, but fall back to the simpler
    sigmoid when (a) the paired bootstrap CI of the Brier difference contains 0,
    or (b) ``val_cal`` had fewer than ``min_cal_positives`` positives (isotonic is
    unstable on thin data). ``metric_fn`` names the selection metric (default Brier).
    """
    p_sig = calibrators["sigmoid"].predict(raw_thr)
    p_iso = calibrators["isotonic"].predict(raw_thr)
    diff, lo, hi = paired_cluster_bootstrap_diff(
        y_thr, p_sig, p_iso, groups, metric_fn, n=n_boot, seed=seed
    )
    unstable = n_cal_positives < min_cal_positives
    # diff = metric(sigmoid) - metric(isotonic); negative => sigmoid better (lower metric).
    if unstable:
        selected, reason = "sigmoid", f"val_cal positives {n_cal_positives} < {min_cal_positives}"
    elif np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi:
        selected, reason = "sigmoid", "Brier CI contains 0 (tie -> simpler)"
    elif diff <= 0.0:
        selected, reason = "sigmoid", "sigmoid lower Brier"
    else:
        selected, reason = "isotonic", "isotonic lower Brier (CI excludes 0)"
    return {
        "selected": selected,
        "reason": reason,
        "brier_sigmoid": brier(y_thr, p_sig),
        "brier_isotonic": brier(y_thr, p_iso),
        "diff_point": diff,
        "diff_lo": lo,
        "diff_hi": hi,
        "n_cal_positives": int(n_cal_positives),
    }
