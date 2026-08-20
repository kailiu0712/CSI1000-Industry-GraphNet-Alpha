"""Barra attribution tests.

The regression is checked against a cross-section built with known loadings,
so a wrong design matrix (an accidental intercept, styles and industries
swapped) shows up as recovered coefficients that miss the planted values.
"""
import numpy as np
import pandas as pd

from iagnn import barra


def _exposures_with_known_loadings(n_dates=8, n_stocks=300, seed=0):
    """factor = 0.5*size - 0.3*momentum + industry offsets + small noise."""
    rng = np.random.default_rng(seed)
    industries = ["IND_A", "IND_B", "IND_C"]
    offsets = {"IND_A": 0.4, "IND_B": -0.2, "IND_C": 0.0}

    frames = []
    for date in pd.bdate_range("2023-01-02", periods=n_dates):
        size = rng.normal(size=n_stocks)
        momentum = rng.normal(size=n_stocks)
        which = rng.choice(industries, size=n_stocks)
        factor = (0.5 * size - 0.3 * momentum
                  + np.array([offsets[w] for w in which])
                  + rng.normal(scale=0.05, size=n_stocks))
        frame = pd.DataFrame({
            "TradingDay": date,
            "SecuCode": [f"{i:06d}" for i in range(n_stocks)],
            "size": size,
            "momentum": momentum,
            "factor": factor,
        })
        for ind in industries:
            frame[ind] = (which == ind).astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_recovers_planted_style_loadings():
    df = _exposures_with_known_loadings()
    exposures = df.drop(columns=["factor"])
    result = barra.attribute(df[["TradingDay", "SecuCode", "factor"]], exposures, "factor",
                             style_cols=["size", "momentum"],
                             industry_cols=["IND_A", "IND_B", "IND_C"])

    assert result is not None
    # attribute() re-standardizes the factor per date, so the loadings are
    # recovered up to that common positive scale -- their ratio is preserved.
    style = result["style_exposure"]
    assert style["size"] > 0 and style["momentum"] < 0
    assert np.isclose(style["size"] / abs(style["momentum"]), 0.5 / 0.3, rtol=0.15)


def test_recovers_industry_ordering():
    df = _exposures_with_known_loadings()
    result = barra.attribute(df[["TradingDay", "SecuCode", "factor"]], df.drop(columns=["factor"]),
                             "factor", style_cols=["size", "momentum"],
                             industry_cols=["IND_A", "IND_B", "IND_C"])
    ind = result["industry_exposure"]
    assert ind["IND_A"] > ind["IND_C"] > ind["IND_B"]


def test_alpha_is_small_when_the_factor_is_fully_explained():
    df = _exposures_with_known_loadings()
    result = barra.attribute(df[["TradingDay", "SecuCode", "factor"]], df.drop(columns=["factor"]),
                             "factor", style_cols=["size", "momentum"],
                             industry_cols=["IND_A", "IND_B", "IND_C"])
    assert result["alpha"] < 0.05


def test_alpha_is_near_one_for_a_factor_unrelated_to_any_exposure():
    df = _exposures_with_known_loadings()
    rng = np.random.default_rng(99)
    df["factor"] = rng.normal(size=len(df))
    result = barra.attribute(df[["TradingDay", "SecuCode", "factor"]], df.drop(columns=["factor"]),
                             "factor", style_cols=["size", "momentum"],
                             industry_cols=["IND_A", "IND_B", "IND_C"])
    assert result["alpha"] > 0.9


def test_returns_none_when_there_are_no_exposures():
    assert barra.attribute(pd.DataFrame(), pd.DataFrame(), "factor") is None


def test_returns_none_when_nothing_overlaps():
    df = _exposures_with_known_loadings()
    scores = df[["TradingDay", "SecuCode", "factor"]].copy()
    scores["SecuCode"] = "999999"  # no shared names
    assert barra.attribute(scores, df.drop(columns=["factor"]), "factor",
                           style_cols=["size", "momentum"],
                           industry_cols=["IND_A", "IND_B", "IND_C"]) is None
