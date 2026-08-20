import numpy as np
import pandas as pd
import pytest

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


def test_benchmark_is_the_equal_weighted_universe():
    from iagnn.evaluate import benchmark_returns

    panel = _panel(signal=1.0, n_dates=5, n_stocks=50)
    bm = benchmark_returns(panel, return_col="next_return")
    expected = panel.groupby("TradingDay")["next_return"].mean()
    pd.testing.assert_series_equal(bm, expected, check_names=False)


def test_long_only_sharpe_is_reported_raw_and_in_excess():
    metrics = evaluate(_panel(signal=1.0), "factor")["metrics"]
    for key in ("long_only_sharpe", "long_only_excess_sharpe",
                "benchmark_sharpe", "ls_sharpe", "ls_max_drawdown"):
        assert key in metrics and not np.isnan(metrics[key])


def test_long_only_excess_strips_the_market_move():
    """A useless factor in a rising market: long-only Sharpe is positive on
    the market alone, but the excess Sharpe must not be."""
    rng = np.random.default_rng(7)
    frames = []
    for date in pd.bdate_range("2023-01-02", periods=60):
        drift = 0.01  # every stock up 1% that day, plus noise
        frames.append(pd.DataFrame({
            "TradingDay": date,
            "SecuCode": [f"{i:06d}" for i in range(200)],
            "factor": rng.normal(size=200),          # pure noise, no signal
            "next_return": drift + rng.normal(scale=0.01, size=200),
        }))
    metrics = evaluate(pd.concat(frames, ignore_index=True), "factor")["metrics"]

    assert metrics["long_only_sharpe"] > 1.0, "market drift alone should lift the raw figure"
    assert abs(metrics["long_only_excess_sharpe"]) < 1.0, "excess must not inherit the drift"


def test_annualised_return_is_geometric_not_arithmetic():
    from iagnn.evaluate import annualised_return

    # A steady +0.1% a day for exactly one year compounds to (1.001)^252 - 1.
    steady = pd.Series([0.001] * 252)
    assert annualised_return(steady) == pytest.approx(1.001 ** 252 - 1)
    # Which is strictly above the arithmetic 252 * 0.001 = 25.2%.
    assert annualised_return(steady) > 252 * 0.001


def test_annualised_return_scales_a_partial_year_up():
    from iagnn.evaluate import annualised_return

    half_year = pd.Series([0.001] * 126)
    full_year = pd.Series([0.001] * 252)
    assert annualised_return(half_year) == pytest.approx(annualised_return(full_year))


def test_annualised_return_is_floored_at_total_loss():
    from iagnn.evaluate import annualised_return

    assert annualised_return(pd.Series([-1.0, 0.05, 0.05])) == -1.0
    assert np.isnan(annualised_return(pd.Series(dtype=float)))


def test_gross_and_net_annualised_returns_are_both_reported():
    metrics = evaluate(_panel(signal=1.0), "factor")["metrics"]
    for key in ("ls_annualised_return", "ls_annualised_return_net",
                "long_only_annualised_return", "long_only_annualised_return_net",
                "benchmark_annualised_return"):
        assert key in metrics
    assert metrics["ls_annualised_return_net"] < metrics["ls_annualised_return"]


def test_max_drawdown_is_zero_for_a_monotonically_rising_curve():
    from iagnn.evaluate import _max_drawdown_additive

    assert _max_drawdown_additive(pd.Series([0.01] * 20)) == 0.0
    assert _max_drawdown_additive(pd.Series([0.1, -0.3, 0.05])) < 0
