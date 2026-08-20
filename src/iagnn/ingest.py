"""Optional bridge to the research framework that produced the feature panel.

The 22 inputs are not raw fields -- they are engineered from CSI1000 1-minute
bars and point-in-time financial statements by
``backtest_framework/local_factor_pool.py``. That builder, and the ~16 GB of
minute bars it reads, are not part of this repository.

So the pipeline's contract is a **prepared feature panel**: a parquet with
``date``, ``instrument`` and the 22 feature columns. This module builds that
parquet when the framework happens to be importable, and otherwise tells you
what shape the file has to be so you can produce it however you like.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .features import IC4NET_FEATURE_COLS

REQUIRED_COLUMNS = ["date", "instrument"] + IC4NET_FEATURE_COLS


def describe_panel_contract() -> str:
    lines = [
        "Prepared feature panel contract (parquet):",
        "  date        : datetime64, one row per stock per trading day",
        "  instrument  : str, 6-digit ticker (zero-padded, no exchange suffix)",
        f"  + {len(IC4NET_FEATURE_COLS)} feature columns, named exactly:",
    ]
    lines += [f"      {col}" for col in IC4NET_FEATURE_COLS]
    return "\n".join(lines)


def build_feature_panel(
    output_path: str | Path,
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    framework_root: str | Path | None = None,
    rebuild_cache: bool = False,
) -> Path:
    """Build the feature panel via the research framework and cache it.

    Raises `ModuleNotFoundError` with the panel contract attached when the
    framework is not on the path -- that is the expected outcome for a
    standalone checkout, and the message says what to supply instead.
    """
    if framework_root is not None:
        root = str(Path(framework_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)

    try:
        from local_factor_pool import build_ic4net_training_panel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The feature builder (backtest_framework/local_factor_pool.py) is not "
            "importable, so the panel cannot be rebuilt here.\n\n"
            + describe_panel_contract()
        ) from exc

    panel, diagnostics = build_ic4net_training_panel(
        start_date=start_date, end_date=end_date, rebuild_cache=rebuild_cache
    )
    keep = [c for c in REQUIRED_COLUMNS if c in panel.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
    if missing:
        raise KeyError(f"Framework panel is missing expected columns: {missing}")

    panel = panel[keep].copy()
    panel["instrument"] = panel["instrument"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    print(
        f"Feature panel: {len(panel):,} rows, {panel['date'].nunique():,} dates, "
        f"{panel['instrument'].nunique():,} instruments -> {output_path}"
    )
    print(f"  builder diagnostics: {diagnostics}")
    return output_path
