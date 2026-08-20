"""Panel assembly: features + prices + index weights + industry.

The pipeline needs three things joined on ``(TradingDay, SecuCode)``:

1. the 22 factor values, from a prepared feature panel (see `ingest.py`);
2. daily base columns -- close price, and the CSI1000 index weight that
   defines both the universe and the size proxy -- from the research
   framework's ``factors/<year>/`` tree;
3. a stock -> industry mapping, which supplies the graph's partition.

`build_panel` returns the model-ready frame; `split_panel` cuts it into the
train and out-of-sample windows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .preprocess import forward_return, standardize_features

BASE_COLUMNS = [
    "TradingDay",
    "SecuCode",
    "ClosePrice",
    "Open5TWAP",
    "IndexW300",
    "IndexW500",
    "IndexW1000",
    "ID3000",
]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Make the join keys comparable: datetime dates, 6-digit string codes."""
    out = df.copy()
    if "TradingDay" in out.columns:
        out["TradingDay"] = pd.to_datetime(out["TradingDay"])
    if "SecuCode" in out.columns:
        out["SecuCode"] = out["SecuCode"].astype(str).str.zfill(6)
    return out


def load_feature_panel(path: str | Path, feature_cols: list[str]) -> pd.DataFrame:
    """Prepared feature panel -> ``TradingDay, SecuCode, <feature_cols>``."""
    panel = pd.read_parquet(path)
    rename = {"date": "TradingDay", "instrument": "SecuCode"}
    panel = panel.rename(columns={k: v for k, v in rename.items() if k in panel.columns})

    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Feature panel {path} is missing {len(missing)} columns: {missing}")

    panel = normalize_keys(panel[["TradingDay", "SecuCode"] + feature_cols])
    return panel.drop_duplicates(subset=["TradingDay", "SecuCode"], keep="last")


def load_base_columns(factor_dir: str | Path, years: list[int]) -> pd.DataFrame:
    """Full listed universe with prices and index weights, for `years`.

    This is the same table every other factor category in the research
    framework is merged against, so a factor exported from here lines up
    row-for-row with the framework's own files.
    """
    factor_dir = Path(factor_dir)
    frames = []
    for year in years:
        weight_path = factor_dir / str(year) / "stock_index_weight.parquet"
        price_path = factor_dir / str(year) / "daily_mean_price.parquet"
        if not weight_path.exists():
            continue
        weights = normalize_keys(pd.read_parquet(weight_path))
        if price_path.exists():
            prices = normalize_keys(
                pd.read_parquet(price_path, columns=["TradingDay", "SecuCode", "Open5TWAP"])
            )
            weights = weights.merge(prices, on=["TradingDay", "SecuCode"], how="inner")
        elif "Open5TWAP" not in weights.columns:
            weights["Open5TWAP"] = np.nan
        weights["year"] = year
        frames.append(weights)

    if not frames:
        raise FileNotFoundError(f"No stock_index_weight.parquet found under {factor_dir} for {years}")

    base = pd.concat(frames, ignore_index=True)
    keep = [c for c in BASE_COLUMNS if c in base.columns] + ["year"]
    return base[keep].sort_values(["TradingDay", "SecuCode"]).reset_index(drop=True)


def load_industry_map(path: str | Path) -> pd.Series:
    """SecuCode -> industry label.

    The mapping is a *current* classification, not point-in-time: a stock
    carries the same industry on every date. Reclassifications are rare
    enough that the graph topology barely moves, but this is a real
    limitation rather than an assumption worth hiding -- a stock that changed
    sector inside the sample is grouped by where it ended up.
    """
    mapping = pd.read_csv(path, dtype={"SecuCode": str})
    mapping["SecuCode"] = mapping["SecuCode"].astype(str).str.zfill(6)
    industry = mapping.set_index("SecuCode")["industry"]
    return industry.replace("", np.nan)


def years_between(start: str, end: str) -> list[int]:
    return list(range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1))


def build_panel(cfg: PipelineConfig, verbose: bool = True) -> pd.DataFrame:
    """Model-ready panel over the full train+test span.

    Order of operations matters for correctness:

    * forward returns are computed on the **full listed panel**, before any
      universe filter, so they measure the stock's genuine next trading days
      rather than the next days it happened to be in the index;
    * standardization runs **after** the universe filter, so each day's
      z-scores describe the population the model actually ranks;
    * both steps are same-day-only, so applying them across train and test at
      once cannot leak anything backward in time.

    `next_return` is the training label. Each configured evaluation return
    lands in its own ``ret_<name>`` column and is never shown to the model.
    """
    data = cfg.data
    years = years_between(data.train_start, data.test_end)

    base = load_base_columns(data.factor_dir, years)
    base["next_return"] = forward_return(
        base, data.label_price_col, horizon=data.label_horizon
    )
    for name, price_col, horizon in data.eval_returns:
        if price_col not in base.columns:
            continue
        base[f"ret_{name}"] = forward_return(base, price_col, horizon=horizon)

    features = load_feature_panel(data.feature_panel_path, cfg.feature_cols)
    panel = base.merge(features, on=["TradingDay", "SecuCode"], how="left")

    industry = load_industry_map(data.industry_map_path)
    panel["industry"] = panel["SecuCode"].map(industry)

    n_all = len(panel)
    eligible = panel[
        (panel[data.universe_weight_col] > 0)
        & panel["industry"].notna()
        & panel[cfg.feature_cols].notna().any(axis=1)
    ].copy()

    eligible = standardize_features(eligible, cfg.feature_cols)
    eligible["log_size"] = np.log(eligible[data.universe_weight_col].clip(lower=1e-6))
    eligible = eligible.sort_values(["TradingDay", "SecuCode"]).reset_index(drop=True)

    if verbose:
        span = f"{eligible['TradingDay'].min():%Y-%m-%d}..{eligible['TradingDay'].max():%Y-%m-%d}"
        print(
            f"Panel: {len(eligible):,} eligible rows of {n_all:,} listed "
            f"({eligible['TradingDay'].nunique():,} dates, "
            f"{eligible['SecuCode'].nunique():,} stocks, {span})"
        )
    return eligible


def split_panel(panel: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(train, test)`` slices of `panel`, by the configured date windows."""
    d = cfg.data
    day = panel["TradingDay"]
    train = panel[day.between(pd.Timestamp(d.train_start), pd.Timestamp(d.train_end))]
    test = panel[day.between(pd.Timestamp(d.test_start), pd.Timestamp(d.test_end))]
    return train.copy(), test.copy()
