"""Every knob the pipeline has, in one place.

`PipelineConfig` is a plain dataclass so it can be constructed in a notebook,
overridden from the CLI, or serialised into a run manifest next to the model
weights. Nothing else in the package reads a module-level constant that is
not reachable from here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .features import IC4NET_FEATURE_COLS

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GraphConfig:
    """Per-date industry graph topology."""

    k_neighbors: int = 10
    #: Column used to rank same-industry peers by size. `IndexW1000` (the
    #: CSI1000 index weight) is a market-cap proxy: index weights are
    #: free-float-cap weighted, so ordering by weight orders by float cap.
    size_proxy_col: str = "IndexW1000"
    #: A date with fewer eligible names than this is dropped entirely --
    #: a cross-sectional IC on a thin cross-section is noise, not signal.
    min_eligible_per_date: int = 300


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the residual industry-GraphSAGE scorer."""

    hidden_dim: int = 32
    dropout: float = 0.1
    #: `alpha`, the weight on the graph correction, is learned but clamped to
    #: [alpha_min, alpha_min + alpha_range] so the graph branch stays a
    #: correction on the own-stock score rather than replacing it.
    alpha_min: float = 0.1
    alpha_range: float = 0.2


@dataclass(frozen=True)
class LossConfig:
    """Per-date loss: -IC + huber_weight * Huber + var_weight * collapse."""

    huber_weight: float = 0.05
    var_collapse_weight: float = 0.01
    #: Floor on the raw cross-sectional score std, below which the collapse
    #: penalty turns on. Guards the degenerate "predict a constant" optimum.
    min_score_std: float = 0.3


@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 15
    batch_dates: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 4
    early_stop_min_delta: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    seed: int = 0
    device: str | None = None  # None -> cuda when available, else cpu


@dataclass(frozen=True)
class DataConfig:
    """Where the inputs live and how the panel is cut.

    `feature_panel_path` is a parquet with columns ``date``, ``instrument``
    and the 22 feature columns. `factor_dir` is the research framework's
    ``factors/`` tree, which supplies the daily base columns (prices and
    index weights) and the industry map.
    """

    feature_panel_path: Path
    factor_dir: Path
    industry_map_path: Path
    train_start: str = "2019-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"
    #: Price column and horizon the training label is built from. The forward
    #: return is label-only and never enters the feature matrix.
    label_price_col: str = "ClosePrice"
    label_horizon: int = 1
    #: Return definitions the factor is *scored* against, as
    #: ``(name, price_column, horizon)``. The default matches the strategy
    #: this model is built for: enter in the day-T closing auction on that
    #: session's signal and exit at the T+1 close, so the position spans the
    #: overnight gap and the following session. Add further entries to score
    #: the same factor against other holding conventions; each one is
    #: reported separately.
    eval_returns: tuple[tuple[str, str, int], ...] = (
        ("close_t1", "ClosePrice", 1),
    )
    #: Universe filter: keep rows whose index weight in this column is > 0.
    universe_weight_col: str = "IndexW1000"
    #: Barra style/industry exposures, for the attribution figure. Not part
    #: of this repository; None disables that one plot and nothing else.
    barra_exposure_dir: Path | None = None


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    feature_cols: list[str] = field(default_factory=lambda: list(IC4NET_FEATURE_COLS))
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: Name the scored factor is written under, in the exported parquet and
    #: in every metrics table.
    factor_name: str = "GNN_IC4Net"
    output_dir: Path = REPO_ROOT / "artifacts"
    #: Figures land here rather than in `output_dir`, because they are the
    #: one run output meant to be committed and shown in the README.
    figure_dir: Path = REPO_ROOT / "docs" / "figures"

    @property
    def n_features(self) -> int:
        return len(self.feature_cols)

    def to_dict(self) -> dict:
        payload = asdict(self)

        def _stringify(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _stringify(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_stringify(v) for v in obj]
            return obj

        return _stringify(payload)

    def write_manifest(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def with_(self, **kwargs) -> "PipelineConfig":
        return replace(self, **kwargs)


def default_config(
    feature_panel_path: str | Path,
    factor_dir: str | Path,
    industry_map_path: str | Path | None = None,
    **overrides,
) -> PipelineConfig:
    """Config for the standard 2019-2022 train / 2023-2024 test run."""
    factor_dir = Path(factor_dir)
    if industry_map_path is None:
        industry_map_path = factor_dir / "secucode_industry_map.csv"
    data = DataConfig(
        feature_panel_path=Path(feature_panel_path),
        factor_dir=factor_dir,
        industry_map_path=Path(industry_map_path),
    )
    return PipelineConfig(data=data, **overrides)
