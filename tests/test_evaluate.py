import numpy as np
import pandas as pd

from iagnn.evaluate import (
    daily_rank_ic,
    evaluate,
    ic_summary,
    quantile_returns,
    turnover,
)


def _panel(seed=0, signal=1.0, n_dates=20, n_stocks=200):
    rng = np.random.default_rng(seed)
    frames = []
    for date in pd.bdate_range("2023-01-02", periods=n_dates):
        factor = rng.normal(size=n_stocks)
        frames.append(pd.DataFrame({
            "TradingDay": date,
            "SecuCode": [f"{i:06d}" for i in range(n_stocks)],
            "factor": factor,
            "next_return": signal * factor + rng.normal(scale=1.0, size=n_stocks),
        }))
    return pd.concat(frames, ignore_index=True)


def test_rank_ic_detects_a_planted_signal():
    ic = daily_rank_ic(_panel(signal=1.0), "factor")
    assert len(ic) == 20
    assert ic.mean() > 0.3


def test_rank_ic_is_flat_on_pure_noise():
    ic = daily_rank_ic(_panel(signal=0.0), "factor")
    assert abs(ic.mean()) < 0.1


def test_ic_summary_handles_an_empty_series():
    summary = ic_summary(pd.Series(dtype=float))
    assert summary["n_days"] == 0
    assert np.isnan(summary["icir"])


def test_quantile_returns_are_monotone_for_a_clean_signal():
    quantiles = quantile_returns(_panel(signal=1.0), "factor", n_quantiles=5)
    means = quantiles.mean()
    assert list(means.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert means.is_monotonic_increasing


def test_turnover_is_zero_for_a_static_ranking():
    """A factor constant through time never trades."""
    frames = []
    scores = np.arange(200, dtype=float)
    for date in pd.bdate_range("2023-01-02", periods=10):
        frames.append(pd.DataFrame({
            "TradingDay": date,
            "SecuCode": [f"{i:06d}" for i in range(200)],
            "factor": scores,
            "next_return": 0.0,
        }))
    assert turnover(pd.concat(frames, ignore_index=True), "factor") == 0.0


def test_evaluate_returns_a_complete_report():
    report = evaluate(_panel(signal=1.0), "factor")
    for key in ("ic_mean", "icir", "ls_sharpe", "top_decile_turnover", "monotonicity"):
        assert key in report["metrics"]
    assert not report["ic_series"].empty
    assert not report["quantile_returns"].empty
