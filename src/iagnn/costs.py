"""Transaction costs for China A-share equities.

A factor with ~78% daily turnover lives or dies on this, so the cost stack is
itemised rather than folded into one round-trip number:

| component | rate | side |
| --- | --- | --- |
| Stamp duty (印花税) | 5 bps, 10 bps before 2023-08-28 | sell only |
| Brokerage commission (佣金) | 2.5 bps | both |
| Transfer fee (过户费) | 0.1 bps | both |
| Slippage / market impact | 5 bps | both |

Notes on each, because the defaults are assumptions and should be argued
with rather than inherited:

* **Stamp duty is sell-side only and it changed inside the test window.**
  China halved it from 0.10% to 0.05% effective 2023-08-28, so a backtest
  spanning 2023 pays two different rates. `stamp_duty_bps_for` returns the
  rate in force on a given date rather than a single blended figure.
* **Commission** at 2.5 bps is an institutional-ish rate. Retail schedules
  run 2.5-3 bps with a ¥5 minimum; the minimum is ignored here, which is
  safe for institutional order sizes and optimistic for tiny ones.
* **Transfer fee** has applied to both Shanghai and Shenzhen since 2022.
* **Slippage** is the softest number in the table and the one most worth
  overriding. 5 bps per side is a moderate assumption for CSI 1000 names at
  modest size; it will be too low for a large book in small caps.

`short_borrow_annual_bps` is charged only on the short leg of the long-short
portfolio. A-share securities lending (融券) is expensive and, for small
caps, often simply unavailable — so treat the long-short line as an upper
bound on what a real short book could capture, and the long-only line as the
implementable one.

All rates are in basis points of traded notional (1 bp = 0.01%).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 1e-4


@dataclass(frozen=True)
class TransactionCosts:
    commission_bps: float = 2.5
    transfer_fee_bps: float = 0.1
    stamp_duty_bps: float = 5.0
    #: Rate in force before `stamp_duty_change_date`; China halved it on
    #: 2023-08-28, which falls inside this project's test window.
    stamp_duty_bps_before: float = 10.0
    stamp_duty_change_date: str = "2023-08-28"
    slippage_bps: float = 5.0
    short_borrow_annual_bps: float = 800.0
    trading_days_per_year: int = 252

    def stamp_duty_bps_for(self, dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        """Sell-side stamp duty in bps, per date."""
        dates = pd.to_datetime(pd.Series(dates).to_numpy())
        cutover = pd.Timestamp(self.stamp_duty_change_date)
        return np.where(dates < cutover, self.stamp_duty_bps_before, self.stamp_duty_bps)

    @property
    def buy_bps(self) -> float:
        """Cost of buying, in bps of notional. No stamp duty on purchases."""
        return self.commission_bps + self.transfer_fee_bps + self.slippage_bps

    def sell_bps(self, dates) -> np.ndarray:
        """Cost of selling, in bps of notional, per date."""
        return (
            self.commission_bps + self.transfer_fee_bps + self.slippage_bps
            + self.stamp_duty_bps_for(dates)
        )

    def round_trip_bps(self, dates) -> np.ndarray:
        """Buy plus sell, per date — what one unit of turnover costs."""
        return self.buy_bps + self.sell_bps(dates)

    @property
    def daily_borrow(self) -> float:
        """Short-leg financing, as a daily fraction of short notional."""
        return self.short_borrow_annual_bps * BPS / self.trading_days_per_year

    def describe(self) -> pd.DataFrame:
        """The cost stack as a table, for the run manifest and the README."""
        return pd.DataFrame(
            [
                ("Stamp duty (印花税)", self.stamp_duty_bps, "sell only",
                 f"{self.stamp_duty_bps_before} bps before {self.stamp_duty_change_date}"),
                ("Commission (佣金)", self.commission_bps, "both sides", "¥5 minimum ignored"),
                ("Transfer fee (过户费)", self.transfer_fee_bps, "both sides", "SH and SZ since 2022"),
                ("Slippage / impact", self.slippage_bps, "both sides", "assumption, override for size"),
                ("Short borrow (融券)", self.short_borrow_annual_bps, "short leg", "annualised, financing"),
            ],
            columns=["component", "bps", "side", "note"],
        )


def portfolio_turnover(
    df: pd.DataFrame,
    factor_col: str,
    n_quantiles: int = 10,
    return_col: str = "next_return",
    date_col: str = "TradingDay",
    id_col: str = "SecuCode",
) -> pd.DataFrame:
    """One-sided turnover per bucket per date: ``index=date, columns=Q1..Qn``.

    Turnover is the weight-based definition, ``0.5 * sum_i |w_i,t -
    w_i,t-1|`` over equal-weighted bucket holdings. That is the fraction of
    notional traded on one side, and unlike counting replaced names it stays
    correct when the bucket's size changes between days — which it does, as
    the eligible universe moves.

    The first date shows turnover 1.0: the book is built from cash, and that
    entry cost is real.
    """
    buckets: dict[pd.Timestamp, dict[int, set]] = {}
    for date, group in df.groupby(date_col, sort=True):
        valid = group[[id_col, factor_col, return_col]].dropna()
        if len(valid) < n_quantiles * 2:
            continue
        try:
            labels = pd.qcut(valid[factor_col].rank(method="first"), n_quantiles, labels=False)
        except ValueError:
            continue
        buckets[date] = {
            q: set(valid.loc[labels == q, id_col]) for q in range(n_quantiles)
        }

    dates = sorted(buckets)
    rows = []
    previous: dict[int, set] | None = None
    for date in dates:
        current = buckets[date]
        row = {}
        for q, names in current.items():
            if previous is None or not previous.get(q):
                row[q] = 1.0
                continue
            old = previous[q]
            w_new, w_old = 1.0 / len(names), 1.0 / len(old)
            union = names | old
            total = sum(
                abs((w_new if n in names else 0.0) - (w_old if n in old else 0.0))
                for n in union
            )
            row[q] = 0.5 * total
        rows.append(pd.Series(row, name=date))
        previous = current

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.index.name = date_col
    out = out.reindex(columns=range(n_quantiles))
    out.columns = [f"Q{q + 1}" for q in out.columns]
    return out


def net_bucket_returns(
    quantiles: pd.DataFrame, turnover: pd.DataFrame, costs: TransactionCosts
) -> pd.DataFrame:
    """Gross bucket returns minus the cost of that day's rebalance.

    Each bucket is treated as a long portfolio: turnover `t` means selling
    `t` of the book and buying `t` back, so the charge is
    ``t * (buy_bps + sell_bps)``.
    """
    if quantiles.empty or turnover.empty:
        return pd.DataFrame()
    turnover = turnover.reindex(quantiles.index).reindex(columns=quantiles.columns)
    round_trip = pd.Series(costs.round_trip_bps(quantiles.index) * BPS, index=quantiles.index)
    return quantiles - turnover.mul(round_trip, axis=0)


def net_long_short(
    quantiles: pd.DataFrame,
    turnover: pd.DataFrame,
    costs: TransactionCosts,
    include_borrow: bool = True,
) -> pd.Series:
    """Long-short return net of both legs' costs and the short borrow.

        net = (top_gross - top_cost) - (bottom_gross + bottom_cost) - borrow

    The short leg's trading cost *adds* to what the position owes, which is
    why it carries a plus sign here.
    """
    if quantiles.empty or turnover.empty:
        return pd.Series(dtype=float)

    top, bottom = quantiles.columns[-1], quantiles.columns[0]
    turnover = turnover.reindex(quantiles.index)
    round_trip = pd.Series(costs.round_trip_bps(quantiles.index) * BPS, index=quantiles.index)

    gross = quantiles[top] - quantiles[bottom]
    trading = (turnover[top] + turnover[bottom]) * round_trip
    borrow = costs.daily_borrow if include_borrow else 0.0
    return gross - trading - borrow


def average_daily_cost_bps(
    turnover: pd.DataFrame, costs: TransactionCosts, column: str | None = None
) -> float:
    """Mean daily cost of holding one bucket, in bps — the drag to beat."""
    if turnover.empty:
        return float("nan")
    column = column or turnover.columns[-1]
    round_trip = pd.Series(costs.round_trip_bps(turnover.index), index=turnover.index)
    return float((turnover[column] * round_trip).mean())
