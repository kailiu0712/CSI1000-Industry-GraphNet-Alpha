"""Persisting a run: model weights, scored factor, and framework hand-off.

Two export targets, for two different consumers:

* `save_checkpoint` / `export_model_json` keep the fitted model. The JSON
  form is plain nested lists rather than a pickle, so the weights can be
  read by a runtime that has no torch and no access to this package -- which
  is what a notebook-only submission environment looks like.
* `write_framework_factor` writes the scored factor back into the research
  framework's ``factors/<year>/Factors_<name>_银行&非银_all.parquet`` layout
  so the existing single-factor test can pick it up with no changes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .config import PipelineConfig
from .data import BASE_COLUMNS, load_base_columns, normalize_keys, years_between

FRAMEWORK_FILE_TEMPLATE = "Factors_{name}_银行&非银_all.parquet"


def save_checkpoint(model: torch.nn.Module, cfg: PipelineConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg.to_dict(), "feature_cols": cfg.feature_cols},
        path,
    )
    return path


def load_checkpoint(path: str | Path, n_features: int, cfg: PipelineConfig):
    from .model import IndustryGraphFactorModel

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = IndustryGraphFactorModel(n_features, cfg.model)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def export_model_json(model: torch.nn.Module, cfg: PipelineConfig, path: str | Path,
                      diagnostics: dict | None = None) -> Path:
    """Torch-free weight dump: every tensor as dtype + shape + flat list."""
    state = {}
    for key, tensor in model.state_dict().items():
        t = tensor.detach().cpu()
        state[key] = {
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "data": t.reshape(-1).tolist(),
        }

    payload = {
        "model_type": "industry_residual_graphsage",
        "feature_cols": cfg.feature_cols,
        "config": cfg.to_dict(),
        "state_dict": state,
        "diagnostics": diagnostics or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_framework_factor(
    predictions: pd.DataFrame,
    cfg: PipelineConfig,
    factor_dir: str | Path | None = None,
    verbose: bool = True,
) -> list[Path]:
    """Write the scored factor into the framework's per-year factor files.

    The output carries the framework's full base-column set and one row per
    listed stock-day -- not just the rows that were scored. The framework's
    reader pairs base columns and factor columns *by position* within a file,
    so a file holding only the scored subset would misalign against the
    universe masks it applies. Stocks outside the model's universe get NaN,
    which the IC computation already treats as absent.
    """
    factor_dir = Path(factor_dir or cfg.data.factor_dir)
    years = years_between(cfg.data.train_start, cfg.data.test_end)
    base = load_base_columns(factor_dir, years)

    scored = normalize_keys(predictions)
    merged = base.merge(scored, on=["TradingDay", "SecuCode"], how="left")

    # Cross-sectional standardization of the final score, matching how every
    # other composite in the framework is stored, so decile cut-points and
    # report scales are comparable across factors.
    from .preprocess import winsorize_by_group, zscore_by_group

    values = merged[cfg.factor_name]
    values = winsorize_by_group(values, merged["TradingDay"])
    merged[cfg.factor_name] = zscore_by_group(values, merged["TradingDay"]).astype("float32")

    output_cols = ["TradingDay", "SecuCode", cfg.factor_name] + [
        c for c in BASE_COLUMNS if c not in ("TradingDay", "SecuCode") and c in merged.columns
    ]

    written = []
    for year in years:
        year_df = merged.loc[merged["year"] == year, output_cols].copy()
        if year_df.empty:
            continue
        year_df["SecuCode"] = year_df["SecuCode"].astype(int)
        year_df["TradingDay"] = pd.to_datetime(year_df["TradingDay"]).dt.strftime("%Y-%m-%d")

        out_dir = factor_dir / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / FRAMEWORK_FILE_TEMPLATE.format(name=cfg.factor_name)
        year_df.to_parquet(out_path, index=False)
        written.append(out_path)
        if verbose:
            n_scored = int(year_df[cfg.factor_name].notna().sum())
            print(f"  wrote {out_path} ({len(year_df):,} rows, {n_scored:,} scored)")
    return written


def save_run_outputs(
    output_dir: str | Path,
    cfg: PipelineConfig,
    predictions: pd.DataFrame,
    reports: dict[str, dict],
    history_frame: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write predictions, metrics, IC series and the run manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    pred_path = output_dir / f"{cfg.factor_name}_scores.parquet"
    predictions.to_parquet(pred_path, index=False)
    paths["predictions"] = pred_path

    metrics_rows = []
    for key, report in reports.items():
        # Keys read "test/open5_t2": window, then return definition.
        window, _, returns = key.partition("/")
        metrics_rows.append({"window": window, "returns": returns, **report["metrics"]})

        slug = key.replace("/", "_")  # keys carry a "/" that filenames cannot
        ic_path = output_dir / f"{cfg.factor_name}_ic_{slug}.csv"
        report["ic_series"].rename("rank_ic").to_csv(ic_path)
        paths[f"ic_{slug}"] = ic_path

        quantiles = report["quantile_returns"]
        if not quantiles.empty:
            q_path = output_dir / f"{cfg.factor_name}_quantiles_{slug}.csv"
            quantiles.to_csv(q_path)
            paths[f"quantiles_{slug}"] = q_path

    metrics_path = output_dir / f"{cfg.factor_name}_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    paths["metrics"] = metrics_path

    if history_frame is not None:
        history_path = output_dir / f"{cfg.factor_name}_train_history.csv"
        history_frame.to_csv(history_path, index=False)
        paths["history"] = history_path

    paths["manifest"] = cfg.write_manifest(output_dir / f"{cfg.factor_name}_manifest.json")
    return paths
