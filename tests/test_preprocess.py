import numpy as np
import pandas as pd

from iagnn.preprocess import (
    forward_return,
    standardize_features,
    winsorize_by_group,
    zscore_by_group,
)


def test_winsorize_clips_only_the_outlier():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    group = pd.Series(["d"] * 6)
    out = winsorize_by_group(values, group)

    assert out.iloc[-1] < 1000.0
    assert out.iloc[:5].tolist() == values.iloc[:5].tolist()


def test_winsorize_leaves_tiny_groups_untouched():
    values = pd.Series([1.0, 1000.0])
    out = winsorize_by_group(values, pd.Series(["d", "d"]))
    pd.testing.assert_series_equal(out, values)


def test_zscore_is_per_day_not_pooled():
    df = pd.DataFrame({
        "day": ["a"] * 6 + ["b"] * 6,
        # Day b is shifted by +100 but has identical dispersion.
        "x": [1.0, 2, 3, 4, 5, 6] + [101.0, 102, 103, 104, 105, 106],
    })
    out = zscore_by_group(df["x"], df["day"])
    np.testing.assert_allclose(out[:6].to_numpy(), out[6:].to_numpy())
    assert abs(out.groupby(df["day"]).mean()).max() < 1e-12


def test_standardize_fills_missing_with_the_daily_mean():
    df = pd.DataFrame({
        "TradingDay": pd.Timestamp("2020-01-01"),
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.inf],
    })
    out = standardize_features(df, ["f1"])
    assert out["f1"].notna().all()
    assert np.isfinite(out["f1"]).all()
    # After z-scoring, the fill value 0.0 is exactly "average today".
    assert out["f1"].iloc[5] == 0.0


def test_forward_return_is_next_day_and_per_stock():
    df = pd.DataFrame({
        "SecuCode": ["A"] * 3 + ["B"] * 3,
        "TradingDay": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"] * 2),
        "ClosePrice": [10.0, 11.0, 12.0, 100.0, 90.0, 90.0],
    })
    fwd = forward_return(df, "ClosePrice")

    # A's first row looks forward to the 10% move into day 2.
    assert abs(fwd.loc[0] - 0.1) < 1e-12
    # The last observation of each stock has no next day.
    assert np.isnan(fwd.loc[2]) and np.isnan(fwd.loc[5])
    # B's return must not be contaminated by A's prices.
    assert abs(fwd.loc[3] - (-0.1)) < 1e-12
