"""End-to-end orchestration: panel -> graphs -> train -> score -> evaluate.

`run` is the whole pipeline in one call. The important structural property is
that the model only ever sees training-window graphs during `train`, while
`predict` scores every date in the panel -- so the test-window scores come
from a model that never saw a test-window label.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .config import PipelineConfig
from .data import build_panel, split_panel
from .evaluate import evaluate, format_report
from .export import export_model_json, save_checkpoint, save_run_outputs, write_framework_factor
from .graph import build_graph_cache
from .model import IndustryGraphFactorModel
from .trainer import TrainHistory, predict, train


@dataclass
class RunResult:
    config: PipelineConfig
    model: IndustryGraphFactorModel
    device: torch.device
    history: TrainHistory
    panel: pd.DataFrame
    predictions: pd.DataFrame
    reports: dict[str, dict]
    artifacts: dict[str, Path]

    def metrics(self, window: str = "test", returns: str | None = None) -> dict:
        """Metrics for one window, defaulting to the first configured return."""
        if returns is None:
            returns = self.config.data.eval_returns[0][0]
        return self.reports[f"{window}/{returns}"]["metrics"]

    @property
    def test_metrics(self) -> dict:
        return self.metrics("test")

    def summary(self) -> str:
        blocks = [
            f"alpha (learned graph weight): {float(self.model.alpha.detach()):.4f}",
            "",
        ]
        for key, report in self.reports.items():
            blocks.append(f"--- {key.upper()} ---")
            blocks.append(format_report(report["metrics"]))
            blocks.append("")
        return "\n".join(blocks)


def run(
    cfg: PipelineConfig,
    write_framework_output: bool = True,
    make_figures: bool = True,
    verbose: bool = True,
) -> RunResult:
    if verbose:
        print(f"=== {cfg.factor_name}: industry-aware GNN ===")
        print(f"Features : {cfg.n_features}")
        print(f"Train    : {cfg.data.train_start} .. {cfg.data.train_end}")
        print(f"Test     : {cfg.data.test_start} .. {cfg.data.test_end} (never seen during fitting)")

    panel = build_panel(cfg, verbose=verbose)
    train_panel, test_panel = split_panel(panel, cfg)

    graphs = build_graph_cache(
        panel,
        cfg.feature_cols,
        k=cfg.graph.k_neighbors,
        min_nodes=cfg.graph.min_eligible_per_date,
        progress=verbose,
    )
    train_dates = set(train_panel["TradingDay"].unique())
    train_graphs = {d: g for d, g in graphs.items() if d in train_dates}

    if verbose:
        print(f"Graphs   : {len(graphs):,} dates total, {len(train_graphs):,} in the training window")

    model, device, history = train(train_graphs, cfg, verbose=verbose)
    predictions = predict(model, device, graphs, cfg, verbose=verbose)

    return_cols = [f"ret_{name}" for name, _, _ in cfg.data.eval_returns]
    return_cols = [c for c in return_cols if c in panel.columns]
    scored = panel[["TradingDay", "SecuCode", "next_return"] + return_cols].merge(
        predictions, on=["TradingDay", "SecuCode"], how="inner"
    )
    scored_train, scored_test = split_panel(scored, cfg)

    # One report per (window, return definition). Keys read "test/open5_t2".
    reports = {}
    for window, sliced in (("train", scored_train), ("test", scored_test)):
        for name, _, _ in cfg.data.eval_returns:
            col = f"ret_{name}"
            if col not in sliced.columns:
                continue
            reports[f"{window}/{name}"] = evaluate(sliced, cfg.factor_name, return_col=col)

    artifacts = save_run_outputs(cfg.output_dir, cfg, predictions, reports, history.to_frame())
    artifacts["checkpoint"] = save_checkpoint(model, cfg, Path(cfg.output_dir) / f"{cfg.factor_name}.pt")
    artifacts["model_json"] = export_model_json(
        model, cfg, Path(cfg.output_dir) / f"{cfg.factor_name}_model.json",
        diagnostics={
            "alpha": float(model.alpha.detach()),
            "train_dates": len(train_graphs),
            "scored_dates": len(graphs),
            "test_icir": {k: v["metrics"].get("icir") for k, v in reports.items()
                          if k.startswith("test/")},
        },
    )

    if make_figures:
        if verbose:
            print("\nBuilding figures ...")
        from .report import build_barra_figure, build_figures

        artifacts.update(build_figures(cfg, reports, verbose=verbose))
        barra_path = build_barra_figure(cfg, predictions, verbose=verbose)
        if barra_path is not None:
            artifacts["barra_industry"] = barra_path

    if write_framework_output:
        if verbose:
            print("\nWriting factor files for the research framework ...")
        for path in write_framework_factor(predictions, cfg, verbose=verbose):
            artifacts[f"factor_{path.parent.name}"] = path

    result = RunResult(cfg, model, device, history, panel, predictions, reports, artifacts)
    if verbose:
        print("\n" + result.summary())
    return result
