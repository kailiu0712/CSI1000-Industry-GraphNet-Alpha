"""Backtest figures, in the parent framework's house style.

Five figures, matching what that project's single-factor test emits so the
two can be read side by side:

* `plot_decile_bar`     — mean forward return per decile, gross beside net
* `plot_cost_impact`    — gross versus net cumulative return, both portfolios
* `plot_ic_series`      — daily RankIC with its rolling mean
* `plot_factor_summary` — stat tiles over ten net decile curves, with the
                          gross long-short leg and the benchmark overlaid
* `plot_barra_industry` — Barra style and industry exposure bars

Cumulative curves are **additive** (`cumsum`), which is the framework's
convention. Compounding here instead would make these plots disagree with
that project's reports on the same factor.

Where a figure can show gross and net, it shows both. Plotting only the net
series hides how much of the signal the cost stack is eating, and plotting
only the gross series is the more familiar way to be wrong.
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


def plot_decile_bar(
    quantiles: pd.DataFrame, factor: str, path: str | Path,
    net_quantiles: pd.DataFrame | None = None,
) -> Path:
    """Mean forward return per decile, gross beside net.

    A working factor climbs left to right. Costs shift every bar down by
    roughly the same amount, so what to look for in the net series is not the
    level but whether the *spread* still clears zero.
    """
    means = quantiles.mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(means))

    if net_quantiles is None or net_quantiles.empty:
        ax.bar(x, means.values, color=["#C44E52" if v < 0 else "#4C72B0" for v in means.values])
    else:
        net_means = net_quantiles.mean().reindex(means.index)
        width = 0.42
        ax.bar(x - width / 2, means.values, width, color="#9EC2E6", label="Gross")
        ax.bar(x + width / 2, net_means.values, width,
               color=["#C44E52" if v < 0 else "#26619C" for v in net_means.values],
               label="Net of costs")
        ax.legend(fontsize=8, frameon=False)

    ax.set_xticks(x)
    ax.set_xticklabels(means.index, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"{factor} — mean forward return by decile", fontsize=10)
    ax.set_xlabel("Decile (1 = lowest factor value)", fontsize=9)
    ax.set_ylabel("Mean daily return", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    return _save(fig, path)


def plot_cost_impact(
    report: dict, factor: str, path: str | Path
) -> Path:
    """Gross versus net cumulative return, for both portfolios.

    The single most useful chart for a high-turnover factor: the vertical gap
    between each pair of lines is what the cost stack takes.
    """
    quantiles = report["quantile_returns"]
    net_quantiles = report["net_quantile_returns"]
    net_ls = report["net_long_short"]
    top, bottom = quantiles.columns[-1], quantiles.columns[0]

    gross_ls = (quantiles[top] - quantiles[bottom]).fillna(0).cumsum()
    net_ls_cum = net_ls.reindex(quantiles.index).fillna(0).cumsum()
    gross_lo = quantiles[top].fillna(0).cumsum()
    net_lo = net_quantiles[top].reindex(quantiles.index).fillna(0).cumsum()

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(gross_ls.index, gross_ls.values, color=LONG_SHORT_COLOR, linewidth=1.2,
            linestyle="--", alpha=0.55, label="Long-short, gross")
    ax.plot(net_ls_cum.index, net_ls_cum.values, color=LONG_SHORT_COLOR, linewidth=1.8,
            label="Long-short, net of costs")
    ax.plot(gross_lo.index, gross_lo.values, color=QUINTILE_COLORS[0], linewidth=1.2,
            linestyle="--", alpha=0.55, label="Long-only top decile, gross")
    ax.plot(net_lo.index, net_lo.values, color=QUINTILE_COLORS[0], linewidth=1.8,
            label="Long-only top decile, net of costs")

    benchmark = report.get("benchmark")
    if benchmark is not None and not benchmark.empty:
        bm = benchmark.reindex(quantiles.index).fillna(0).cumsum()
        ax.plot(bm.index, bm.values, color=BENCHMARK_COLOR, linewidth=1.0,
                linestyle=":", label="Benchmark (equal-weighted universe)")

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_title(f"{factor} — the cost of turnover", fontsize=11)
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

    The decile curves are **net of transaction costs** — that is the series a
    reader should judge the factor on — with the gross long-short leg drawn
    faintly behind the net one so the cost drag is visible rather than
    quietly netted away. `report` is an `evaluate.evaluate()` result.
    """
    quantiles = report["quantile_returns"]
    net_quantiles = report.get("net_quantile_returns")
    net_ls = report.get("net_long_short")
    metrics = report["metrics"]
    benchmark = report.get("benchmark")
    if quantiles.empty:
        raise ValueError("No quantile returns to plot.")

    plotted = net_quantiles if net_quantiles is not None and not net_quantiles.empty else quantiles
    cum = plotted.fillna(0).cumsum()
    gross_spread = (
        quantiles[quantiles.columns[-1]] - quantiles[quantiles.columns[0]]
    ).fillna(0).cumsum()
    net_spread = (
        net_ls.reindex(quantiles.index).fillna(0).cumsum()
        if net_ls is not None and not net_ls.empty else None
    )

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 4.3], hspace=0.42, wspace=0.22)

    tiles = [
        ("IC Mean", metrics.get("ic_mean"), "{:.4f}"),
        ("ICIR", metrics.get("icir"), "{:.3f}"),
        ("LS Sharpe, net", metrics.get("ls_sharpe_net"), "{:.2f}"),
        ("Long-only Sharpe, net", metrics.get("long_only_sharpe_net"), "{:.2f}"),
        ("LO excess Sharpe, net", metrics.get("long_only_excess_sharpe_net"), "{:.2f}"),
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
    ax.plot(gross_spread.index, gross_spread.values, color=LONG_SHORT_COLOR, linewidth=1.2,
            linestyle="--", alpha=0.45,
            label=f"Long-short ({cum.columns[-1]} - {cum.columns[0]}), gross")
    if net_spread is not None:
        ax.plot(net_spread.index, net_spread.values, color=LONG_SHORT_COLOR, linewidth=1.7,
                label="Long-short, net of costs")
    if benchmark is not None and not benchmark.empty:
        bm = benchmark.reindex(cum.index).fillna(0).cumsum()
        ax.plot(bm.index, bm.values, color=BENCHMARK_COLOR, linewidth=1.0,
                linestyle=":", label="Benchmark")

    ax.set_xlim(cum.index.min(), cum.index.max())
    net_label = " (net of costs)" if plotted is net_quantiles else ""
    ax.set_title(f"Decile cumulative returns{net_label} — {factor}", fontsize=12)
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
