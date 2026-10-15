"""Part 10C/D figures (SHAP bar, beeswarm, dependence, waterfall).

Every SHAP figure carries the same caveat: path-dependent mean|SHAP| measures
*usage* of a feature by the booster, not its held-out usefulness (Experiment 8D
showed planted noise ranks mid-pack under mean|SHAP| but last under permutation).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

CAVEAT = "path-dependent mean|SHAP| reflects usage, not held-out utility (see Experiment 8D)"
_CAVEAT_STYLE = {"fontsize": 7, "color": "0.35", "ha": "left", "va": "bottom"}


def _add_caveat(fig) -> None:
    fig.text(0.01, 0.01, CAVEAT, **_CAVEAT_STYLE)


def _save(fig, path: str | Path, *, note: str | None = None) -> Path:
    if note:
        fig.text(0.01, 0.035, note, fontsize=7, color="0.35", ha="left", va="bottom")
    out = Path(path)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_shap_bar(global_df: pd.DataFrame, path: str | Path, top_n: int = 20) -> Path:
    """Horizontal mean|SHAP| bars for the ``top_n`` features."""
    top = global_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="tab:blue")
    ax.set_xlabel("mean |SHAP| (margin units)")
    ax.set_title(f"Top {top_n} features by mean|SHAP| (dev validation)")
    fig.tight_layout()
    _add_caveat(fig)
    return _save(fig, path)


def plot_shap_beeswarm(
    shap_values: np.ndarray, x_sample: pd.DataFrame, path: str | Path, top_n: int = 20
) -> Path:
    """SHAP beeswarm summary; colour is the raw feature value."""
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, x_sample, max_display=top_n, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    fig.axes[0].set_title("SHAP beeswarm (margin units, dev validation)")
    fig.axes[0].set_xlabel("SHAP value (push on raw margin)")
    fig.tight_layout()
    _add_caveat(fig)
    return _save(fig, path)


def plot_shap_dependence(
    shap_values: np.ndarray,
    x_sample: pd.DataFrame,
    feature: str,
    color_feature: str,
    path: str | Path,
    *,
    binary: bool = False,
) -> Path:
    """Dependence plot for ``feature``, coloured by its strongest interactor.

    Binary features get a jittered strip plot (a two-point scatter carries no shape).
    """
    features = list(x_sample.columns)
    j = features.index(feature)
    values = x_sample[feature].to_numpy(dtype=float)
    color = x_sample[color_feature].to_numpy(dtype=float)
    sv = np.asarray(shap_values)[:, j]

    fig, ax = plt.subplots(figsize=(7, 5))
    if binary:
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        sc = ax.scatter(values + jitter, sv, c=color, cmap="coolwarm", s=6, alpha=0.6)
        ax.set_xticks(sorted(pd.unique(values[~np.isnan(values)])))
    else:
        sc = ax.scatter(values, sv, c=color, cmap="coolwarm", s=6, alpha=0.6)
    ax.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(f"{feature} (raw value)")
    ax.set_ylabel("SHAP value (push on raw margin)")
    ax.set_title(f"SHAP dependence: {feature}\ncoloured by {color_feature}")
    fig.colorbar(sc, ax=ax, label=color_feature)
    fig.tight_layout()
    _add_caveat(fig)
    return _save(fig, path)


def plot_waterfall(
    shap_values_row: np.ndarray,
    base_value: float,
    x_row: pd.DataFrame,
    path: str | Path,
    *,
    title: str,
    top_n: int = 12,
) -> Path:
    """SHAP waterfall for one transaction."""
    explanation = shap.Explanation(
        values=np.asarray(shap_values_row, dtype=float),
        base_values=float(base_value),
        data=x_row.iloc[0].to_numpy(dtype=float),
        feature_names=list(x_row.columns),
    )
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, max_display=top_n, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    _add_caveat(fig)
    return _save(fig, path)
