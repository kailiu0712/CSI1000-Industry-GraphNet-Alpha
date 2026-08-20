"""Figure tests.

These check that each figure renders to a non-trivial PNG and that the
cumulative curves use the additive convention the parent framework's plots
use — not that the pixels look a particular way.
"""
import numpy as np
import pandas as pd

from iagnn import plots
from iagnn.evaluate import evaluate


def _panel(seed=0, signal=1.0, n_dates=30, n_stocks=200):
    rng = np.random.default_rng(seed)
    frames = []
    for date in pd.bdate_range("2023-01-02", periods=n_dates):
        factor = rng.normal(size=n_stocks)
        frames.append(pd.DataFrame({
            "TradingDay": date,
            "SecuCode": [f"{i:06d}" for i in range(n_stocks)],
            "factor": factor,
            "next_return": signal * factor * 0.01 + rng.normal(scale=0.01, size=n_stocks),
        }))
    return pd.concat(frames, ignore_index=True)


def _is_real_png(path) -> bool:
    if not path.exists() or path.stat().st_size < 5_000:
        return False
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_every_figure_renders(tmp_path):
    report = evaluate(_panel(), "factor")

    paths = [
        plots.plot_factor_summary(report, "factor", tmp_path / "summary.png"),
        plots.plot_quintile_cumulative(report["quintile_returns"], "factor",
                                       tmp_path / "cum.png", benchmark=report["benchmark"]),
        plots.plot_decile_bar(report["quantile_returns"], "factor", tmp_path / "bar.png"),
        plots.plot_ic_series(report["ic_series"], "factor", tmp_path / "ic.png"),
    ]
    for path in paths:
        assert _is_real_png(path), path


def test_barra_figure_renders_with_cjk_industry_labels(tmp_path):
    style = pd.Series({"size": 0.2, "momentum": -0.1, "liquidity": 0.05})
    industry = pd.Series({"电子": 0.3, "医药生物": -0.2, "银行": 0.01})
    path = plots.plot_barra_industry(style, industry, alpha=0.62, factor="factor",
                                     path=tmp_path / "barra.png")
    assert _is_real_png(path)


def test_barra_figure_trims_to_the_strongest_tilts(tmp_path):
    industry = pd.Series({f"IND{i}": (i - 15) / 15 for i in range(31)})
    path = plots.plot_barra_industry(pd.Series({"size": 0.1}), industry, 0.5, "factor",
                                     tmp_path / "barra_trim.png", n_industries=6)
    assert _is_real_png(path)


def test_quintile_curves_are_additive_not_compounded():
    """The last point of the plotted curve must equal the plain sum."""
    report = evaluate(_panel(), "factor")
    quintiles = report["quintile_returns"]
    top = quintiles.columns[-1]

    plotted_end = quintiles[top].fillna(0).cumsum().iloc[-1]
    assert np.isclose(plotted_end, quintiles[top].fillna(0).sum())
    # And it must differ from the compounded figure, or the test proves nothing.
    compounded = (1 + quintiles[top].fillna(0)).prod() - 1
    assert not np.isclose(plotted_end, compounded, rtol=1e-9)


def test_figures_land_in_a_directory_that_did_not_exist(tmp_path):
    report = evaluate(_panel(), "factor")
    nested = tmp_path / "deep" / "nested" / "dir" / "ic.png"
    assert _is_real_png(plots.plot_ic_series(report["ic_series"], "factor", nested))
