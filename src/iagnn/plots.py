"""Backtest figures, in the parent framework's house style.

Five figures, matching what that project's single-factor test emits so the
two can be read side by side:

* `plot_decile_bar`         — mean forward return per decile
* `plot_quintile_cumulative`— five quintile curves plus the long-short leg
* `plot_ic_series`          — daily RankIC with its rolling mean
* `plot_factor_summary`     — stat tiles over ten decile curves, long-short
                              and benchmark overlaid
* `plot_barra_industry`     — Barra style and industry exposure bars

Cumulative curves are **additive** (`cumsum`), which is the framework's
convention. Compounding here instead would make these plots disagree with
that project's reports on the same factor.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402  (backend must be set first)

QUINTILE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
QUINTILE_LABELS = ["Q1 (Bottom 20%)", "Q2", "Q3", "Q4", "Q5 (Top 20%)"]
LONG_SHORT_COLOR = "#C0392B"
BENCHMARK_COLOR = "#000000"
IC_ROLLING_WINDOW = 20


def use_cjk_font() -> None:
    """Prefer a CJK-capable font, for the Chinese industry labels.

    `axes.unicode_minus` is switched off because the CJK faces below have no
    U+2212 glyph, so a negative tick would render as a blank box.
    """
    matplotlib.rcParams["font.sans-serif"] = (
        ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
        + list(matplotlib.rcParams["font.sans-serif"])
    )
    matplotlib.rcParams["axes.unicode_minus"] = False


def _save(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decile_bar(quantiles: pd.DataFrame, factor: str, path: str | Path) -> Path:
    """Mean forward return per decile. A working factor climbs left to right."""
    means = quantiles.mean()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in means.values]
    ax.bar(range(len(means)), means.values, color=colors)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"{factor} — mean forward return by decile", fontsize=10)
    ax.set_xlabel("Decile (1 = lowest factor value)", fontsize=9)
    ax.set_ylabel("Mean daily return", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_quintile_cumulative(
    quintiles: pd.DataFrame, factor: str, path: str | Path, benchmark: pd.Series | None = None
) -> Path:
    """Five quintile cumulative curves plus the Q5-Q1 long-short leg."""
    cum = quintiles.fillna(0).cumsum()
    fig, ax = plt.subplots(figsize=(8, 4.2))

    for col, color, label in zip(quintiles.columns, QUINTILE_COLORS, QUINTILE_LABELS):
        ax.plot(cum.index, cum[col].values, color=color, label=label, linewidth=1.1)

    spread = (quintiles[quintiles.columns[-1]] - quintiles[quintiles.columns[0]]).fillna(0).cumsum()
    ax.plot(spread.index, spread.values, color=LONG_SHORT_COLOR, linewidth=1.6,
            linestyle="--", label="Long-short (Q5 - Q1)")

    if benchmark is not None and not benchmark.empty:
        bm = benchmark.reindex(cum.index).fillna(0).cumsum()
        ax.plot(bm.index, bm.values, color=BENCHMARK_COLOR, linewidth=1.0,
                linestyle=":", label="Benchmark (equal-weighted universe)")

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_title(f"{factor} — quintile cumulative return", fontsize=11)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Cumulative return (additive)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, path)


def plot_ic_series(ic: pd.Series, factor: str, path: str | Path,
                   window: int = IC_ROLLING_WINDOW) -> Path:
    """Daily RankIC (faint) under its rolling mean (bold)."""
    ic = ic.sort_index()
    rolling = ic.rolling(window, min_periods=max(5, window // 4)).mean()

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(ic.index, ic.values, color=QUINTILE_COLORS[0], linewidth=0.6, alpha=0.25,
            label="Daily RankIC")
    ax.plot(rolling.index, rolling.values, color="#111111", linewidth=1.6,
            label=f"{window}-day rolling mean")
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axhline(float(ic.mean()), color="#C0392B", linewidth=0.9, linestyle="-.",
               label=f"Mean = {ic.mean():.4f}")
    ax.set_title(f"{factor} — RankIC over time", fontsize=11)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("RankIC", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, path)


def plot_factor_summary(
    report: dict,
    factor: str,
    path: str | Path,
    title_suffix: str = "",
) -> Path:
    """The framework's summary card: stat tiles over ten decile curves.

    `report` is an `evaluate.evaluate()` result.
    """
    quantiles = report["quantile_returns"]
    metrics = report["metrics"]
    benchmark = report.get("benchmark")
    if quantiles.empty:
        raise ValueError("No quantile returns to plot.")

    cum = quantiles.fillna(0).cumsum()
    spread = (quantiles[quantiles.columns[-1]] - quantiles[quantiles.columns[0]]).fillna(0).cumsum()

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 4.3], hspace=0.42, wspace=0.22)

    tiles = [
        ("IC Mean", metrics.get("ic_mean"), "{:.4f}"),
        ("ICIR", metrics.get("icir"), "{:.3f}"),
        ("LS Sharpe", metrics.get("ls_sharpe"), "{:.2f}"),
        ("Long-only Sharpe", metrics.get("long_only_sharpe"), "{:.2f}"),
        ("LO excess Sharpe", metrics.get("long_only_excess_sharpe"), "{:.2f}"),
    ]
    for i, (label, value, fmt) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle(
            (0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
            facecolor="#F5F5F4", edgecolor="#E0E0DC", linewidth=1, zorder=0,
        ))
        ax.text(0.09, 0.66, label, fontsize=9, color="#6B6B68",
                transform=ax.transAxes, va="center")
        text = fmt.format(value) if value is not None and pd.notna(value) else "n/a"
        ax.text(0.09, 0.30, text, fontsize=17, color="#111111", fontweight="bold",
                transform=ax.transAxes, va="center")

    ax = fig.add_subplot(gs[1, :])
    n = cum.shape[1]
    cmap = plt.cm.Blues
    for i, col in enumerate(cum.columns):
        ax.plot(cum.index, cum[col].values, color=cmap(0.3 + 0.6 * i / max(n - 1, 1)),
                linewidth=1.0, label=col)
    ax.plot(spread.index, spread.values, color=LONG_SHORT_COLOR, linewidth=1.5,
            linestyle="--", label=f"Long-short ({cum.columns[-1]} - {cum.columns[0]})")
    if benchmark is not None and not benchmark.empty:
        bm = benchmark.reindex(cum.index).fillna(0).cumsum()
        ax.plot(bm.index, bm.values, color=BENCHMARK_COLOR, linewidth=1.0,
                linestyle=":", label="Benchmark")

    ax.set_xlim(cum.index.min(), cum.index.max())
    ax.set_title(f"Decile cumulative returns — {factor}", fontsize=12)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Cumulative return (additive)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    window = f"{metrics.get('start')} .. {metrics.get('end')}"
    fig.suptitle(f"{factor} — factor summary   ({window}){title_suffix}", fontsize=13, y=0.98)
    return _save(fig, path, dpi=170)


def plot_barra_industry(
    style_exposure: pd.Series,
    industry_exposure: pd.Series,
    alpha: float,
    factor: str,
    path: str | Path,
    n_industries: int = 20,
) -> Path:
    """Barra style exposure (with the unexplained share) and industry tilts."""
    use_cjk_font()
    fig, (ax_style, ax_ind) = plt.subplots(1, 2, figsize=(12, 5.5))

    labels = list(style_exposure.index) + ["Alpha (1 - R2)"]
    values = list(style_exposure.values) + [alpha]
    colors = ["#2a78d6"] * len(style_exposure) + ["#d03b3b"]
    ax_style.barh(labels, values, color=colors)
    ax_style.axvline(0, color="black", linewidth=0.6)
    ax_style.set_title("Barra style exposure + unexplained share", fontsize=11)
    ax_style.tick_params(labelsize=8)
    ax_style.invert_yaxis()
    ax_style.grid(axis="x", alpha=0.25, linewidth=0.5)

    ind = industry_exposure.dropna().sort_values()
    if len(ind) > n_industries:  # keep the strongest tilts at both ends
        half = n_industries // 2
        ind = pd.concat([ind.head(half), ind.tail(n_industries - half)])
    ax_ind.barh(ind.index, ind.values, color=np.where(ind.values < 0, "#d08a3b", "#1baf7a"))
    ax_ind.axvline(0, color="black", linewidth=0.6)
    ax_ind.set_title(f"Industry exposure (strongest {len(ind)} tilts)", fontsize=11)
    ax_ind.tick_params(labelsize=8)
    ax_ind.grid(axis="x", alpha=0.25, linewidth=0.5)

    fig.suptitle(f"{factor} — Barra style & industry attribution", fontsize=13)
    fig.tight_layout()
    return _save(fig, path)
