"""Industry-aware GNN for CSI1000 stock picking.

A residual GraphSAGE factor model. Each stock is scored from its own factor
values, then corrected by what its nearest same-industry, similar-size peers
look like that day. The correction is weighted by a learned scalar bounded
into a narrow band, so industry context refines the ranking instead of
replacing it.

Quick start::

    from iagnn import default_config, run

    cfg = default_config(
        feature_panel_path="artifacts/feature_panel.parquet",
        factor_dir="/path/to/backtest_framework/factors",
    )
    result = run(cfg)
    print(result.summary())
"""
from . import _bootstrap  # noqa: F401 -- must precede numpy/pandas; see _bootstrap

from .config import (
    DataConfig,
    GraphConfig,
    LossConfig,
    ModelConfig,
    PipelineConfig,
    TrainConfig,
    default_config,
)
from .evaluate import evaluate, format_report
from .features import IC4NET_FEATURE_COLS
from .graph import DateGraph, build_date_graph, build_graph_cache
from .model import IndustryGraphFactorModel, SparseMeanSAGE
from .pipeline import RunResult, run
from .trainer import predict, train

__version__ = "1.0.0"

__all__ = [
    "DataConfig",
    "DateGraph",
    "GraphConfig",
    "IC4NET_FEATURE_COLS",
    "IndustryGraphFactorModel",
    "LossConfig",
    "ModelConfig",
    "PipelineConfig",
    "RunResult",
    "SparseMeanSAGE",
    "TrainConfig",
    "build_date_graph",
    "build_graph_cache",
    "default_config",
    "evaluate",
    "format_report",
    "predict",
    "run",
    "train",
    "__version__",
]
