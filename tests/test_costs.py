"""Transaction cost tests.

The arithmetic here is simple enough that the risk is not a wrong formula but
a wrong *sign* or a wrong *side* — charging stamp duty on purchases, or
subtracting the short leg's cost instead of adding it. These tests pin the
directions down.
"""
import numpy as np
import pandas as pd
import pytest

from iagnn.costs import (
    BPS,
    TransactionCosts,
    average_daily_cost_bps,
    net_bucket_returns,
    net_long_short,
    portfolio_turnover,
)


@pytest.fixture
def costs():
    return TransactionCosts()


def test_stamp_duty_is_sell_side_only(costs):
    dates = pd.DatetimeIndex(["2024-01-02"])
    assert costs.buy_bps == costs.commission_bps + costs.transfer_fee_bps + costs.slippage_bps
    assert costs.sell_bps(dates)[0] - costs.buy_bps == pytest.approx(costs.stamp_duty_bps)


def test_stamp_duty_halves_at_the_2023_cutover(costs):
    dates = pd.DatetimeIndex(["2023-08-25", "2023-08-28", "2023-08-29"])
    rates = costs.stamp_duty_bps_for(dates)
    assert rates[0] == costs.stamp_duty_bps_before   # 10 bps before
    assert rates[1] == costs.stamp_duty_bps          # 5 bps from the cutover date
    assert rates[2] == costs.stamp_duty_bps


def test_round_trip_is_buy_plus_sell(costs):
    dates = pd.DatetimeIndex(["2024-01-02"])
    assert costs.round_trip_bps(dates)[0] == pytest.approx(costs.buy_bps + costs.sell_bps(dates)[0])
    # Sanity on magnitude: ~20 bps round trip under the defaults.
    assert 15 < costs.round_trip_bps(dates)[0] < 25


def test_daily_borrow_annualises_back(costs):
    assert costs.daily_borrow * costs.trading_days_per_year == pytest.approx(
        costs.short_borrow_annual_bps * BPS
    )


def _two_day_panel(overlap: int, bucket_size: int = 10):
    """Two dates, deciles built so the top bucket keeps `overlap` names."""
    rows = []
    n = bucket_size * 10
    for day, offset in [("2024-01-02", 0), ("2024-01-03", bucket_size - overlap)]:
        # Rotate the ranking by `offset` so the top decile sheds that many names.
        for i in range(n):
            rows.append({
                "TradingDay": pd.Timestamp(day),
                "SecuCode": f"{i:06d}",
                "factor": float((i + offset) % n),
                "next_return": 0.0,
            })
    return pd.DataFrame(rows)


def test_turnover_is_zero_for_an_unchanged_bucket():
    turnover = portfolio_turnover(_two_day_panel(overlap=10), "factor")
    assert turnover.iloc[0]["Q10"] == 1.0        # day one builds the book from cash
    assert turnover.iloc[1]["Q10"] == pytest.approx(0.0)


def test_turnover_is_one_for_a_fully_replaced_bucket():
    turnover = portfolio_turnover(_two_day_panel(overlap=0), "factor")
    assert turnover.iloc[1]["Q10"] == pytest.approx(1.0)


def test_turnover_is_the_replaced_fraction():
    turnover = portfolio_turnover(_two_day_panel(overlap=7), "factor")
    assert turnover.iloc[1]["Q10"] == pytest.approx(0.3)


def test_net_bucket_return_is_gross_minus_turnover_times_round_trip(costs):
    index = pd.DatetimeIndex(["2024-01-02"])
    quantiles = pd.DataFrame({"Q1": [0.01], "Q10": [0.02]}, index=index)
    turnover = pd.DataFrame({"Q1": [0.5], "Q10": [1.0]}, index=index)

    net = net_bucket_returns(quantiles, turnover, costs)
    round_trip = costs.round_trip_bps(index)[0] * BPS
    assert net["Q10"].iloc[0] == pytest.approx(0.02 - 1.0 * round_trip)
    assert net["Q1"].iloc[0] == pytest.approx(0.01 - 0.5 * round_trip)


def test_short_leg_cost_adds_to_what_the_position_owes(costs):
    """Both legs' trading costs must reduce the long-short return."""
    index = pd.DatetimeIndex(["2024-01-02"])
    quantiles = pd.DataFrame({"Q1": [0.0], "Q10": [0.02]}, index=index)
    turnover = pd.DataFrame({"Q1": [1.0], "Q10": [1.0]}, index=index)

    net = net_long_short(quantiles, turnover, costs, include_borrow=False)
    round_trip = costs.round_trip_bps(index)[0] * BPS
    assert net.iloc[0] == pytest.approx(0.02 - 2 * round_trip)
    assert net.iloc[0] < 0.02


def test_borrow_further_reduces_the_long_short(costs):
    index = pd.DatetimeIndex(["2024-01-02"])
    quantiles = pd.DataFrame({"Q1": [0.0], "Q10": [0.02]}, index=index)
    turnover = pd.DataFrame({"Q1": [0.0], "Q10": [0.0]}, index=index)

    with_borrow = net_long_short(quantiles, turnover, costs, include_borrow=True).iloc[0]
    without = net_long_short(quantiles, turnover, costs, include_borrow=False).iloc[0]
    assert without - with_borrow == pytest.approx(costs.daily_borrow)


def test_zero_cost_settings_leave_returns_untouched():
    free = TransactionCosts(commission_bps=0, transfer_fee_bps=0, stamp_duty_bps=0,
                            stamp_duty_bps_before=0, slippage_bps=0, short_borrow_annual_bps=0)
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    quantiles = pd.DataFrame({"Q1": [0.01, 0.01], "Q10": [0.02, 0.02]}, index=index)
    turnover = pd.DataFrame({"Q1": [1.0, 1.0], "Q10": [1.0, 1.0]}, index=index)

    pd.testing.assert_frame_equal(net_bucket_returns(quantiles, turnover, free), quantiles)


def test_average_daily_cost_scales_with_turnover(costs):
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    turnover = pd.DataFrame({"Q10": [1.0, 1.0]}, index=index)
    half = pd.DataFrame({"Q10": [0.5, 0.5]}, index=index)

    assert average_daily_cost_bps(turnover, costs) == pytest.approx(
        2 * average_daily_cost_bps(half, costs)
    )


def test_cost_table_lists_every_component(costs):
    table = costs.describe()
    assert len(table) == 5
    assert {"component", "bps", "side", "note"} <= set(table.columns)
    assert (table["bps"] >= 0).all()


def test_evaluate_reports_gross_and_net_side_by_side():
    from iagnn.evaluate import evaluate

    rng = np.random.default_rng(0)
    frames = []
    for date in pd.bdate_range("2023-01-02", periods=40):
        factor = rng.normal(size=200)
        frames.append(pd.DataFrame({
            "TradingDay": date,
            "SecuCode": [f"{i:06d}" for i in range(200)],
            "factor": factor,
            "next_return": 0.01 * factor + rng.normal(scale=0.01, size=200),
        }))
    metrics = evaluate(pd.concat(frames, ignore_index=True), "factor")["metrics"]

    for key in ("ls_sharpe", "ls_sharpe_net", "long_only_sharpe", "long_only_sharpe_net",
                "top_decile_daily_cost_bps"):
        assert key in metrics
    # Costs can only reduce a positive return, never improve it.
    assert metrics["ls_daily_mean_net"] < metrics["ls_daily_mean"]
    assert metrics["long_only_daily_mean_net"] < metrics["long_only_daily_mean"]
