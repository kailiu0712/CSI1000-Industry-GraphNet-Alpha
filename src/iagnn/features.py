"""Feature-set definitions.

`IC4NET_FEATURE_COLS` is the current best-to-date input set, carried over
verbatim from ``backtest_framework/factor_pool_definitions.py`` (the 22
factors kept after the elastic-net / IC screening documented in
``backtest_framework/ic4net_deployment/README.md``).

The list is duplicated here on purpose: this package is meant to be pushed
as a standalone repository, so it must not import from the research
framework it was extracted from. `verify_feature_parity` re-checks the copy
against the framework when the framework happens to be importable, so the
duplication cannot drift silently.
"""
from __future__ import annotations

# --- the 22 kept factors -------------------------------------------------
# Twelve intraday/minute-bar structure factors ("1.x ..."), six daily
# technical factors, and four point-in-time fundamental ratios.
IC4NET_FEATURE_COLS: list[str] = [
    "1.11 VWAP Close Ratio Last30Min",
    "1.9 ExtremeHighReversal Last30Min",
    "1.7 DealRecovery Slope 30min",
    "1.2 M4",
    "1.4 lz 10",
    "1.1 Q3",
    "AmountMA20",
    "1.10 UpsideVolRatio Vol5DivVol20",
    "ATR5",
    "1.8 EarlyAfternoonRecovery VolumeWeighted VolatilityAdjusted",
    "PMCloseRange",
    "1.12 VolumeConcentration Last30Min PriceWeighted",
    "HF_GapII_SUM3D",
    "VRSI60",
    "FC_AssetImpairmentToRevenue_TTM",
    "AmountIR5",
    "Dratio",
    "VR_N",
    "EP_TTM",
    "CurrentAssetsTRate",
    "VEMA5",
    "SP1_TTM",
]

FUNDAMENTAL_FEATURE_COLS: list[str] = [
    "FC_AssetImpairmentToRevenue_TTM",
    "EP_TTM",
    "CurrentAssetsTRate",
    "SP1_TTM",
]

TECHNICAL_FEATURE_COLS: list[str] = [
    col for col in IC4NET_FEATURE_COLS if col not in FUNDAMENTAL_FEATURE_COLS
]


def verify_feature_parity() -> bool | None:
    """Has this file's copy drifted from the research framework's?

    ``True`` = identical, ``False`` = drifted, ``None`` = the framework is not
    importable so there is nothing to compare against -- the normal case for a
    standalone checkout. `None` rather than `True` so a caller cannot mistake
    "not checked" for "checked and fine".
    """
    try:
        from factor_pool_definitions import IC4NET_FEATURE_COLS as upstream
    except ImportError:
        try:
            from backtest_framework.factor_pool_definitions import (  # type: ignore
                IC4NET_FEATURE_COLS as upstream,
            )
        except ImportError:
            return None
    return list(upstream) == IC4NET_FEATURE_COLS
