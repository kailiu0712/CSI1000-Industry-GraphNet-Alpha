"""Command-line entry points.

    python -m iagnn prepare  --output artifacts/feature_panel.parquet --framework-root ...
    python -m iagnn run      --feature-panel ... --factor-dir ...
    python -m iagnn evaluate --scores ... --factor-dir ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import REPO_ROOT, default_config
from .evaluate import evaluate, format_report
from .features import verify_feature_parity


def _add_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-start", default="2019-01-01")
    parser.add_argument("--train-end", default="2022-12-31")
    parser.add_argument("--test-start", default="2023-01-01")
    parser.add_argument("--test-end", default="2024-12-31")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iagnn", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Build the 22-feature panel via the research framework.")
    prepare.add_argument("--output", default=str(REPO_ROOT / "artifacts" / "feature_panel.parquet"))
    prepare.add_argument("--framework-root", default=None,
                         help="Path to backtest_framework/ (added to sys.path).")
    prepare.add_argument("--start-date", default="2019-01-01")
    prepare.add_argument("--end-date", default="2024-12-31")
    prepare.add_argument("--rebuild-cache", action="store_true")

    run_cmd = sub.add_parser("run", help="Train, score and evaluate end to end.")
    run_cmd.add_argument("--feature-panel", required=True)
    run_cmd.add_argument("--factor-dir", required=True)
    run_cmd.add_argument("--industry-map", default=None)
    run_cmd.add_argument("--factor-name", default="GNN_IC4Net")
    run_cmd.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts"))
    run_cmd.add_argument("--epochs", type=int, default=None)
    run_cmd.add_argument("--k-neighbors", type=int, default=None)
    run_cmd.add_argument("--seed", type=int, default=None)
    run_cmd.add_argument("--device", default=None)
    run_cmd.add_argument("--no-framework-output", action="store_true",
                         help="Skip writing Factors_<name>_*.parquet into the framework tree.")
    run_cmd.add_argument("--figure-dir", default=None, help="Where the PNG figures go.")
    run_cmd.add_argument("--barra-dir", default=None,
                         help="Barra exposure parquet directory, for the attribution figure.")
    run_cmd.add_argument("--no-figures", action="store_true", help="Skip figure generation.")
    run_cmd.add_argument("--slippage-bps", type=float, default=None,
                         help="Per-side slippage assumption (default 5 bps).")
    run_cmd.add_argument("--commission-bps", type=float, default=None,
                         help="Per-side brokerage commission (default 2.5 bps).")
    run_cmd.add_argument("--no-costs", action="store_true",
                         help="Report gross figures only (sets every cost rate to zero).")
    _add_window_args(run_cmd)

    eval_cmd = sub.add_parser("evaluate", help="Re-score metrics from a saved predictions file.")
    eval_cmd.add_argument("--scores", required=True)
    eval_cmd.add_argument("--factor-dir", required=True)
    eval_cmd.add_argument("--factor-name", default="GNN_IC4Net")
    _add_window_args(eval_cmd)

    sub.add_parser("features", help="Print the feature-set contract and exit.")
    return parser


def _config_from_args(args) -> "object":
    from dataclasses import replace

    cfg = default_config(
        feature_panel_path=args.feature_panel,
        factor_dir=args.factor_dir,
        industry_map_path=args.industry_map,
        factor_name=args.factor_name,
        output_dir=Path(args.output_dir),
    )
    cfg = cfg.with_(data=replace(
        cfg.data,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start, test_end=args.test_end,
    ))
    if args.epochs is not None:
        cfg = cfg.with_(train=replace(cfg.train, max_epochs=args.epochs))
    if args.seed is not None:
        cfg = cfg.with_(train=replace(cfg.train, seed=args.seed))
    if args.device is not None:
        cfg = cfg.with_(train=replace(cfg.train, device=args.device))
    if args.k_neighbors is not None:
        cfg = cfg.with_(graph=replace(cfg.graph, k_neighbors=args.k_neighbors))
    if getattr(args, "figure_dir", None):
        cfg = cfg.with_(figure_dir=Path(args.figure_dir))
    if getattr(args, "barra_dir", None):
        cfg = cfg.with_(data=replace(cfg.data, barra_exposure_dir=Path(args.barra_dir)))

    if getattr(args, "no_costs", False):
        from .costs import TransactionCosts
        cfg = cfg.with_(costs=TransactionCosts(
            commission_bps=0.0, transfer_fee_bps=0.0, stamp_duty_bps=0.0,
            stamp_duty_bps_before=0.0, slippage_bps=0.0, short_borrow_annual_bps=0.0,
        ))
    else:
        overrides = {}
        if getattr(args, "slippage_bps", None) is not None:
            overrides["slippage_bps"] = args.slippage_bps
        if getattr(args, "commission_bps", None) is not None:
            overrides["commission_bps"] = args.commission_bps
        if overrides:
            cfg = cfg.with_(costs=replace(cfg.costs, **overrides))
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "features":
        from .ingest import describe_panel_contract
        print(describe_panel_contract())
        status = {
            True: "matches",
            False: "MISMATCH -- this copy has drifted",
            None: "not checked (parent framework not importable)",
        }
        print(f"\nParity with the research framework: {status[verify_feature_parity()]}")
        return 0

    if args.command == "prepare":
        from .ingest import build_feature_panel
        build_feature_panel(
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            framework_root=args.framework_root,
            rebuild_cache=args.rebuild_cache,
        )
        return 0

    if args.command == "run":
        from .pipeline import run as run_pipeline
        cfg = _config_from_args(args)
        run_pipeline(
            cfg,
            write_framework_output=not args.no_framework_output,
            make_figures=not args.no_figures,
        )
        print(f"\nArtifacts written to {cfg.output_dir}")
        return 0

    if args.command == "evaluate":
        from .config import DataConfig
        from .data import load_base_columns, normalize_keys, years_between
        from .preprocess import forward_return

        scores = normalize_keys(pd.read_parquet(args.scores))
        years = years_between(args.train_start, args.test_end)
        base = load_base_columns(args.factor_dir, years)

        eval_returns = DataConfig.__dataclass_fields__["eval_returns"].default
        cols = []
        for name, price_col, horizon in eval_returns:
            if price_col not in base.columns:
                continue
            base[f"ret_{name}"] = forward_return(base, price_col, horizon=horizon)
            cols.append((name, f"ret_{name}"))

        merged = base[["TradingDay", "SecuCode"] + [c for _, c in cols]].merge(
            scores, on=["TradingDay", "SecuCode"], how="inner"
        )
        for window, (start, end) in {
            "train": (args.train_start, args.train_end),
            "test": (args.test_start, args.test_end),
        }.items():
            sliced = merged[merged["TradingDay"].between(pd.Timestamp(start), pd.Timestamp(end))]
            for name, col in cols:
                print(f"--- {window.upper()} / {name} ---")
                print(format_report(evaluate(sliced, args.factor_name, return_col=col)["metrics"]))
                print()
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
