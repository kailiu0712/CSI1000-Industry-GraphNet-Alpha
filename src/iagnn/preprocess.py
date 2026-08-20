"""Cross-sectional (same-day) preprocessing primitives.

Every transform here is computed *within a single trading day* and uses no
information from any other day. That is what makes it safe to apply the same
code to the training window and to the out-of-sample window: there is no
fitted state to leak forward, so "fit on train, apply to test" and "apply
everywhere" are the same operation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_N = 5
MIN_VALID_FOR_STATS = 5


def winsorize_by_group(s: pd.Series, group: pd.Series, n: int = MAD_N) -> pd.Series:
    """MAD(n) winsorization within each group (one group per trading day).

    Implemented with pandas' C-level groupby reductions rather than a
    per-group Python callback: with one group per day there are thousands of
    groups, and the per-call overhead -- not the arithmetic -- dominates.

    Groups where the MAD is zero or undefined, or that have fewer than
    `MIN_VALID_FOR_STATS` observations, are passed through untouched instead
    of being clipped to a degenerate interval.
    """
    counts = s.notna().groupby(group).transform("sum")
    median = s.groupby(group).transform("median")
    mad = (s - median).abs().groupby(group).transform("median")

    clipped = s.clip(median - n * mad, median + n * mad)
    skip = (mad == 0) | ~np.isfinite(mad) | (counts < MIN_VALID_FOR_STATS)
    return clipped.where(~skip, s)


def zscore_by_group(s: pd.Series, group: pd.Series) -> pd.Series:
    """Z-score within each group. Zero-variance groups are only centred."""
    counts = s.notna().groupby(group).transform("sum")
    mean = s.groupby(group).transform("mean")
    std = s.groupby(group).transform("std")

    centered = s - mean
    scaled = centered / std
    result = scaled.where((std != 0) & np.isfinite(std), centered)
    return result.where(counts >= MIN_VALID_FOR_STATS, s)


def standardize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    date_col: str = "TradingDay",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """inf -> NaN, winsorize, z-score, then fill what is still missing.

    Filling with 0.0 *after* z-scoring means a missing input is treated as
    "this stock is average on this factor today", which keeps the row in the
    graph instead of dropping a node and tearing a hole in its industry's
    peer structure.
    """
    out = df.copy()
    dates = out[date_col]
    for col in feature_cols:
        s = out[col].replace([np.inf, -np.inf], np.nan)
        s = winsorize_by_group(s, dates)
        s = zscore_by_group(s, dates)
        out[col] = s.fillna(fill_value)
    return out


def forward_return(
    df: pd.DataFrame,
    price_col: str,
    id_col: str = "SecuCode",
    date_col: str = "TradingDay",
    horizon: int = 1,
) -> pd.Series:
    """Next-`horizon`-day return per stock, aligned to `df`'s index.

    Label only -- it is never fed to the model as an input. Computed on the
    full listed panel before any universe filter, so it is the return over
    the stock's genuine next trading day rather than over the next day on
    which it happened to be in the index.
    """
    ordered = df.sort_values([id_col, date_col])
    pct = ordered.groupby(id_col, sort=False)[price_col].pct_change(fill_method=None)
    return pct.groupby(ordered[id_col], sort=False).shift(-horizon)
