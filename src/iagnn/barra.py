"""Barra style and industry attribution of the factor.

Answers "what is this factor actually betting on?". Each trading day, the
cross-section of standardized factor values is regressed on that day's Barra
style exposures and industry dummies, with **no intercept** — the industry
dummies partition the universe and already act as their own intercept, so
adding one would make the design matrix rank-deficient:

    factor_z[i,t] = sum_k beta[k,t] * style[k,i,t] + sum_j gamma[j,t] * industry[j,i,t]

Averaging `beta[k,t]` over t gives the factor's mean loading on style k;
averaging `gamma[j,t]` gives its mean level inside industry j. `1 - R^2`
averaged over t is the share of the factor that no known style or industry
explains — for an alpha factor, a large unexplained share is the point.

The exposure data is not distributed with this repository. `load_exposures`
reads the parent project's ``data/rq_csi1000_barra_exposure/*.parquet``, and
everything here degrades to `None` when that directory is absent.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_PATTERN = re.compile(r"barra_exposure_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet")

STYLE_COLUMNS = [
    "size", "non_linear_size", "momentum", "liquidity", "book_to_price",
    "leverage", "growth", "earnings_yield", "beta", "residual_volatility", "comovement",
]
MIN_OBS_PER_DATE = 50


def discover_industry_columns(exposure_dir: str | Path) -> list[str]:
    """Whatever dummy columns the exposure files carry beyond the styles."""
    import pyarrow.parquet as pq

    exposure_dir = Path(exposure_dir)
    files = sorted(p for p in exposure_dir.glob("*.parquet") if CHUNK_PATTERN.search(p.name))
    if not files:
        raise FileNotFoundError(f"No Barra exposure files under {exposure_dir}")
    names = pq.ParquetFile(files[0]).schema.names
    exclude = {"date", "instrument", "instrument_id"} | set(STYLE_COLUMNS)
    return [c for c in names if c not in exclude]


def load_exposures(exposure_dir: str | Path, start_date: str, end_date: str) -> pd.DataFrame:
    """``TradingDay, SecuCode, <styles>, <industry dummies>`` for the window."""
    exposure_dir = Path(exposure_dir)
    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)

    frames = []
    for path in sorted(exposure_dir.glob("*.parquet")):
        match = CHUNK_PATTERN.search(path.name)
        if not match:
            continue
        if pd.Timestamp(match.group(2)) < start_ts or pd.Timestamp(match.group(1)) > end_ts:
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].rename(columns={"date": "TradingDay"})
    df["SecuCode"] = df["instrument_id"].astype(int).astype(str).str.zfill(6)
    return (
        df.drop(columns=["instrument", "instrument_id"])
        .drop_duplicates(subset=["TradingDay", "SecuCode"])
    )


def _daily_regression(day_df: pd.DataFrame, style_cols: list[str], industry_cols: list[str]):
    """One date's cross-section -> (style betas, {industry: gamma}, R^2)."""
    valid = day_df.dropna(subset=["factor_z"] + style_cols)
    present = [c for c in industry_cols if valid[c].sum() > 0]
    if len(valid) < MIN_OBS_PER_DATE or not present:
        return None

    X = valid[style_cols + present].to_numpy(dtype=float)
    y = valid["factor_z"].to_numpy(dtype=float)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)

    residual = y - X @ coefs
    total_ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residual ** 2).sum()) / total_ss if total_ss > 0 else np.nan

    return coefs[:len(style_cols)], dict(zip(present, coefs[len(style_cols):])), r2


def attribute(
    factor_df: pd.DataFrame,
    exposures: pd.DataFrame,
    factor_col: str,
    style_cols: list[str] | None = None,
    industry_cols: list[str] | None = None,
) -> dict | None:
    """Average style/industry exposure of `factor_col`, plus its alpha share.

    `factor_df` needs ``TradingDay``, ``SecuCode`` and `factor_col`. The
    factor is re-standardized per date first, so the betas are comparable
    across dates regardless of the factor's own scale.
    """
    if exposures.empty:
        return None

    style_cols = style_cols or [c for c in STYLE_COLUMNS if c in exposures.columns]
    if industry_cols is None:
        industry_cols = [
            c for c in exposures.columns
            if c not in {"TradingDay", "SecuCode"} and c not in style_cols
        ]

    from .preprocess import zscore_by_group

    merged = factor_df[["TradingDay", "SecuCode", factor_col]].merge(
        exposures, on=["TradingDay", "SecuCode"], how="inner"
    )
    if merged.empty:
        return None
    merged["factor_z"] = zscore_by_group(merged[factor_col], merged["TradingDay"])

    style_records, industry_records, r2s = [], [], []
    for _, day_df in merged.groupby("TradingDay", sort=True):
        result = _daily_regression(day_df, style_cols, industry_cols)
        if result is None:
            continue
        betas, gammas, r2 = result
        style_records.append(betas)
        industry_records.append(gammas)
        r2s.append(r2)

    if not style_records:
        return None

    return {
        "style_exposure": pd.Series(np.mean(style_records, axis=0), index=style_cols),
        "industry_exposure": pd.DataFrame(industry_records).mean(),
        "alpha": float(1.0 - np.nanmean(r2s)),
        "n_dates": len(style_records),
    }
