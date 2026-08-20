"""Out-of-sample factor evaluation.

Deliberately the same metric definitions the parent research framework's
single-factor test uses, so a number produced here can be compared directly
against that project's reports rather than only against itself. The
annualisation factor is 252 throughout.

* **RankIC** — daily Spearman correlation between the factor and the forward
  return. Rank-based, because a factor is used to *order* stocks; a few
  extreme values should not decide the score.
* **ICIR** — mean RankIC divided by its standard deviation. The stability of
  the edge, not just its size.
* **decile returns** — equal-weighted forward return per factor bucket,
  which is the closest thing to what a trading rule would earn.
* **long-short Sharpe** — top decile minus bottom decile. Market-neutral by
  construction, so it isolates the factor.
* **long-only Sharpe** — the top decile held outright. Reported both raw and
  in excess of the equal-weighted universe, because the raw number is mostly
  market beta: in a rising market a useless factor still posts a positive
  long-only Sharpe. The excess figure is the one that says whether the factor
  added anything.
* **annualised return** — geometric, `(prod(1+r))**(252/n) - 1`, because
  that is what the term means unqualified. Note the plotted curves stay
  additive (`cumsum`) to match the framework's charts: they show the shape
  of a track record, while this number sizes it. Reading the end of an
  additive curve as a compounded return overstates a result badly.
* **turnover** — fraction of the long bucket replaced day over day, the
  first-order check on whether the spread survives costs.

Every portfolio statistic is reported twice: gross, and net of China
A-share transaction costs (see `costs.py`), keyed with a `_net` suffix. For a
factor that replaces most of its book daily the gap between the two is not a
footnote, it is the result.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # imported lazily at call time to keep this module light
    from .costs import TransactionCosts

TRADING_DAYS_PER_YEAR = 252
IC_ROLLING_WINDOW = 20


def daily_rank_ic(
    df: pd.DataFrame, factor_col: str, return_col: str = "next_return", date_col: str = "TradingDay"
) -> pd.Series:
    """Spearman RankIC per date, indexed by date. Thin dates drop out."""
    def _ic(group: pd.DataFrame) -> float:
        valid = group[[factor_col, return_col]].dropna()
        if len(valid) < 2:
            return np.nan
        return valid[factor_col].rank().corr(valid[return_col].rank())

    return df.groupby(date_col, sort=True).apply(_ic, include_groups=False).dropna()


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean/std/ICIR plus the sign-stability and t-stat readings.

    `ic_t_stat` treats the daily ICs as an i.i.d. sample, which overstates
    significance to the extent that IC is autocorrelated. It is reported as a
    rough magnitude check, not as a test.
    """
    if ic.empty:
        return {"n_days": 0, "ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "ic_positive_rate": np.nan, "ic_t_stat": np.nan}
    mean, std = float(ic.mean()), float(ic.std())
    icir = mean / std if std else np.nan
    return {
        "n_days": int(len(ic)),
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_positive_rate": float((ic > 0).mean()),
        # Annualised-style scaling: ICIR * sqrt(n) is the classic t on the mean IC.
        "ic_t_stat": icir * np.sqrt(len(ic)) if std else np.nan,
    }


def quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    n_quantiles: int = 10,
    return_col: str = "next_return",
    date_col: str = "TradingDay",
) -> pd.DataFrame:
    """Mean forward return per factor quantile, per date.

    Quantiles are formed within each date (never pooled), with 0 = lowest
    factor value. Dates that cannot fill `n_quantiles` distinct buckets are
    skipped rather than collapsed into fewer, uneven ones.
    """
    frames = []
    for date, group in df.groupby(date_col, sort=True):
        valid = group[[factor_col, return_col]].dropna()
        if len(valid) < n_quantiles * 2:
            continue
        try:
            buckets = pd.qcut(valid[factor_col].rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        means = valid.groupby(buckets, observed=True)[return_col].mean()
        frames.append(means.rename(date))

    if not frames:
        return pd.DataFrame()
    out = pd.DataFrame(frames)
    out.index.name = date_col
    out.columns = [f"Q{int(c) + 1}" for c in out.columns]
    return out


def benchmark_returns(
    df: pd.DataFrame, return_col: str = "next_return", date_col: str = "TradingDay"
) -> pd.Series:
    """Equal-weighted mean forward return per date — the universe itself.

    This is the benchmark a long-only bucket has to beat. Without it, a
    long-only Sharpe is mostly a statement about the market, not the factor.
    """
    return df.groupby(date_col, sort=True)[return_col].mean().dropna()


def _sharpe(series: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    std = series.std(ddof=1)
    if series.empty or not np.isfinite(std) or std == 0:
        return np.nan
    return float(series.mean() / std * np.sqrt(periods_per_year))


def _max_drawdown_additive(series: pd.Series) -> float:
    """Deepest peak-to-trough fall of the additive cumulative curve."""
    if series.empty:
        return np.nan
    curve = series.fillna(0).cumsum()
    return float((curve - curve.cummax()).min())


def annualised_return(
    series: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Geometric annualised return (CAGR) of a daily return series.

        (prod(1 + r)) ** (periods_per_year / n) - 1

    Geometric rather than `mean * 252`, because "annualised return" without
    qualification means the rate that actually compounds to the observed
    result — the arithmetic version ignores volatility drag and reads high
    for a volatile series. This is a different convention from the plotted
    curves, which stay additive (`cumsum`) to match the parent framework's
    charts; the curves show the shape of the track record, this number sizes
    it.

    A daily return of -100% or worse leaves nothing to compound, so the
    result is floored at -1.0 rather than going complex.
    """
    s = series.dropna()
    if s.empty:
        return np.nan
    growth = float((1.0 + s).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(s)) - 1.0


def long_short_summary(
    quantiles: pd.DataFrame,
    benchmark: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Bucket-portfolio statistics: long-short, long-only, and benchmark.

    Return figures are annualised geometrically; see `annualised_return`.
    """
    if quantiles.empty:
        return {}
    top, bottom = quantiles.columns[-1], quantiles.columns[0]
    spread = (quantiles[top] - quantiles[bottom]).dropna()
    long_only = quantiles[top].dropna()

    out = {
        "ls_daily_mean": float(spread.mean()),
        "ls_daily_std": float(spread.std(ddof=1)),
        "ls_sharpe": _sharpe(spread, periods_per_year),
        "ls_annualised_return": annualised_return(spread, periods_per_year),
        "ls_max_drawdown": _max_drawdown_additive(spread),
        "long_only_daily_mean": float(long_only.mean()),
        "long_only_sharpe": _sharpe(long_only, periods_per_year),
        "long_only_annualised_return": annualised_return(long_only, periods_per_year),
        "top_decile_mean": float(quantiles[top].mean()),
        "bottom_decile_mean": float(quantiles[bottom].mean()),
        "monotonicity": float(pd.Series(quantiles.mean().to_numpy()).rank().corr(
            pd.Series(np.arange(quantiles.shape[1]) + 1.0))),
    }

    if benchmark is not None and not benchmark.empty:
        bm = benchmark.reindex(long_only.index).dropna()
        excess = (long_only - bm).dropna()
        out.update({
            "benchmark_daily_mean": float(bm.mean()),
            "benchmark_sharpe": _sharpe(bm, periods_per_year),
            "benchmark_annualised_return": annualised_return(bm, periods_per_year),
            "long_only_excess_daily_mean": float(excess.mean()),
            "long_only_excess_sharpe": _sharpe(excess, periods_per_year),
            "long_only_excess_annualised_return": annualised_return(excess, periods_per_year),
        })
    return out


def net_summary(
    net_quantiles: pd.DataFrame,
    net_ls: pd.Series,
    benchmark: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """The same portfolio statistics, after transaction costs.

    Keyed with a `_net` suffix so gross and net sit side by side in one
    metrics row: the gap between them *is* the finding for a factor that
    turns over most of its book daily.
    """
    if net_quantiles.empty:
        return {}
    top = net_quantiles.columns[-1]
    long_only = net_quantiles[top].dropna()
    spread = net_ls.dropna()

    out = {
        "ls_daily_mean_net": float(spread.mean()),
        "ls_sharpe_net": _sharpe(spread, periods_per_year),
        "ls_annualised_return_net": annualised_return(spread, periods_per_year),
        "ls_max_drawdown_net": _max_drawdown_additive(spread),
        "long_only_daily_mean_net": float(long_only.mean()),
        "long_only_sharpe_net": _sharpe(long_only, periods_per_year),
        "long_only_annualised_return_net": annualised_return(long_only, periods_per_year),
        "top_decile_mean_net": float(net_quantiles[top].mean()),
        "bottom_decile_mean_net": float(net_quantiles[net_quantiles.columns[0]].mean()),
        "monotonicity_net": float(pd.Series(net_quantiles.mean().to_numpy()).rank().corr(
            pd.Series(np.arange(net_quantiles.shape[1]) + 1.0))),
    }

    if benchmark is not None and not benchmark.empty:
        bm = benchmark.reindex(long_only.index).dropna()
        excess = (long_only - bm).dropna()
        out.update({
            "long_only_excess_daily_mean_net": float(excess.mean()),
            "long_only_excess_sharpe_net": _sharpe(excess, periods_per_year),
            "long_only_excess_annualised_return_net": annualised_return(excess, periods_per_year),
        })
    return out


def turnover(
    df: pd.DataFrame,
    factor_col: str,
    top_frac: float = 0.1,
    date_col: str = "TradingDay",
    id_col: str = "SecuCode",
) -> float:
    """Average daily one-sided turnover of the top-`top_frac` bucket."""
    holdings = []
    for _, group in df.groupby(date_col, sort=True):
        valid = group[[id_col, factor_col]].dropna()
        if valid.empty:
            continue
        n = max(1, int(len(valid) * top_frac))
        holdings.append(set(valid.nlargest(n, factor_col)[id_col]))

    if len(holdings) < 2:
        return np.nan
    changes = [
        len(curr - prev) / max(len(curr), 1)
        for prev, curr in zip(holdings[:-1], holdings[1:])
    ]
    return float(np.mean(changes))


def evaluate(
    df: pd.DataFrame,
    factor_col: str,
    n_quantiles: int = 10,
    return_col: str = "next_return",
    costs: "TransactionCosts | None" = None,
) -> dict[str, object]:
    """Full report for one factor over one window.

    Returns the metrics plus the series the plots are drawn from, so a caller
    never has to recompute them: the daily IC, the per-date decile returns,
    the quintile returns the framework-style curves use, and the
    equal-weighted benchmark.
    """
    from .costs import (
        TransactionCosts,
        average_daily_cost_bps,
        net_bucket_returns,
        net_long_short,
        portfolio_turnover,
    )

    costs = costs if costs is not None else TransactionCosts()

    ic = daily_rank_ic(df, factor_col, return_col=return_col)
    quantiles = quantile_returns(df, factor_col, n_quantiles=n_quantiles, return_col=return_col)
    benchmark = benchmark_returns(df, return_col=return_col)

    bucket_turnover = portfolio_turnover(
        df, factor_col, n_quantiles=n_quantiles, return_col=return_col
    )
    net_quantiles = net_bucket_returns(quantiles, bucket_turnover, costs)
    net_ls = net_long_short(quantiles, bucket_turnover, costs)

    metrics = {
        "factor": factor_col,
        "start": str(df["TradingDay"].min().date()) if len(df) else None,
        "end": str(df["TradingDay"].max().date()) if len(df) else None,
        "n_rows": int(len(df)),
        **ic_summary(ic),
        **long_short_summary(quantiles, benchmark),
        **net_summary(net_quantiles, net_ls, benchmark),
        "top_decile_turnover": turnover(df, factor_col),
        "top_decile_daily_cost_bps": average_daily_cost_bps(bucket_turnover, costs),
    }
    return {
        "metrics": metrics,
        "ic_series": ic,
        "quantile_returns": quantiles,
        "net_quantile_returns": net_quantiles,
        "net_long_short": net_ls,
        "bucket_turnover": bucket_turnover,
        "benchmark": benchmark,
        "costs": costs,
    }


def format_report(metrics: dict[str, object]) -> str:
    """Human-readable one-screen summary."""
    def fmt(key, spec=".4f"):
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return format(value, spec) if isinstance(value, float) else str(value)

    return "\n".join([
        f"Factor            : {metrics.get('factor')}",
        f"Window            : {metrics.get('start')} .. {metrics.get('end')}"
        f"  ({fmt('n_days', 'd')} days, {metrics.get('n_rows'):,} rows)",
        f"RankIC mean       : {fmt('ic_mean')}",
        f"RankIC std        : {fmt('ic_std')}",
        f"ICIR              : {fmt('icir')}",
        f"IC > 0 frequency  : {fmt('ic_positive_rate', '.2%')}",
        f"IC t-stat         : {fmt('ic_t_stat', '.2f')}",
        f"Top decile / day  : {fmt('top_decile_mean', '.5f')}",
        f"Bottom decile/day : {fmt('bottom_decile_mean', '.5f')}",
        f"Benchmark / day   : {fmt('benchmark_daily_mean', '.5f')}",
        f"Long-short Sharpe : {fmt('ls_sharpe', '.2f')} gross"
        f"  ->  {fmt('ls_sharpe_net', '.2f')} net",
        f"Long-only Sharpe  : {fmt('long_only_sharpe', '.2f')} gross"
        f"  ->  {fmt('long_only_sharpe_net', '.2f')} net"
        f"   (net excess of benchmark: {fmt('long_only_excess_sharpe_net', '.2f')})",
        f"Benchmark Sharpe  : {fmt('benchmark_sharpe', '.2f')}",
        f"Long-short ann.   : {fmt('ls_annualised_return', '.1%')} gross"
        f"  ->  {fmt('ls_annualised_return_net', '.1%')} net",
        f"Long-only ann.    : {fmt('long_only_annualised_return', '.1%')} gross"
        f"  ->  {fmt('long_only_annualised_return_net', '.1%')} net",
        f"Benchmark ann.    : {fmt('benchmark_annualised_return', '.1%')}",
        f"Long-short max DD : {fmt('ls_max_drawdown_net', '.2%')} net",
        f"Decile monotonic. : {fmt('monotonicity', '.3f')} gross"
        f"  ->  {fmt('monotonicity_net', '.3f')} net",
        f"Top-decile turnov.: {fmt('top_decile_turnover', '.2%')} per day",
        f"Daily cost drag   : {fmt('top_decile_daily_cost_bps', '.1f')} bps",
    ])
