"""Turn a finished run into figures.

Separated from `pipeline.py` so the plots can be regenerated from saved
scores without retraining — restyling a chart should not cost a training
run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from . import plots


def figure_paths(cfg: PipelineConfig, window: str, returns: str) -> dict[str, Path]:
    base = Path(cfg.figure_dir)
    stem = f"{cfg.factor_name}_{window}_{returns}"
    return {
        "summary": base / f"{stem}_summary.png",
        "cost_impact": base / f"{stem}_cost_impact.png",
        "decile_bar": base / f"{stem}_decile_bar.png",
        "ic_series": base / f"{stem}_ic.png",
    }


def build_figures(
    cfg: PipelineConfig,
    reports: dict[str, dict],
    verbose: bool = True,
) -> dict[str, Path]:
    """One set of four figures per (window, return definition)."""
    written: dict[str, Path] = {}
    for key, report in reports.items():
        window, _, returns = key.partition("/")
        if report["quantile_returns"].empty:
            continue

        paths = figure_paths(cfg, window, returns)
        label = f"{cfg.factor_name} [{window} / {returns}]"

        plots.plot_factor_summary(report, label, paths["summary"])
        plots.plot_cost_impact(report, label, paths["cost_impact"])
        plots.plot_decile_bar(
            report["quantile_returns"], label, paths["decile_bar"],
            net_quantiles=report.get("net_quantile_returns"),
        )
        plots.plot_ic_series(report["ic_series"], label, paths["ic_series"])

        for name, path in paths.items():
            written[f"{name}_{window}_{returns}"] = path
        if verbose:
            print(f"  figures for {key} -> {paths['summary'].parent}")
    return written


def write_cost_table(cfg: PipelineConfig) -> Path:
    """Persist the cost stack next to the figures, so the README can cite it."""
    path = Path(cfg.figure_dir).parent / f"{cfg.factor_name}_transaction_costs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg.costs.describe().to_csv(path, index=False, encoding="utf-8")
    return path


def build_barra_figure(
    cfg: PipelineConfig,
    predictions: pd.DataFrame,
    verbose: bool = True,
) -> Path | None:
    """Barra/industry attribution, or None when the exposures aren't present."""
    if cfg.data.barra_exposure_dir is None:
        if verbose:
            print("  Barra attribution skipped (no exposure directory configured)")
        return None

    from . import barra

    exposure_dir = Path(cfg.data.barra_exposure_dir)
    if not exposure_dir.exists():
        if verbose:
            print(f"  Barra attribution skipped ({exposure_dir} not found)")
        return None

    exposures = barra.load_exposures(exposure_dir, cfg.data.test_start, cfg.data.test_end)
    if exposures.empty:
        if verbose:
            print("  Barra attribution skipped (no exposures overlap the test window)")
        return None

    test = predictions[
        pd.to_datetime(predictions["TradingDay"]).between(
            pd.Timestamp(cfg.data.test_start), pd.Timestamp(cfg.data.test_end)
        )
    ]
    result = barra.attribute(test, exposures, cfg.factor_name)
    if result is None:
        if verbose:
            print("  Barra attribution skipped (no date had enough overlapping names)")
        return None

    path = Path(cfg.figure_dir) / f"{cfg.factor_name}_test_barra_industry.png"
    plots.plot_barra_industry(
        result["style_exposure"], result["industry_exposure"], result["alpha"],
        f"{cfg.factor_name} [test]", path,
    )

    summary_path = Path(cfg.figure_dir).parent / f"{cfg.factor_name}_barra_exposure.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([
        result["style_exposure"].rename("exposure").to_frame().assign(kind="style"),
        result["industry_exposure"].rename("exposure").to_frame().assign(kind="industry"),
        pd.DataFrame({"exposure": [result["alpha"]], "kind": ["alpha"]}, index=["alpha_1_minus_r2"]),
    ]).to_csv(summary_path)

    if verbose:
        print(f"  Barra attribution over {result['n_dates']} dates, "
              f"unexplained share {result['alpha']:.3f} -> {path}")
    return path
