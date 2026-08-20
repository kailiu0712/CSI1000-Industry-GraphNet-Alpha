"""Reproduce the headline run: train 2019-2022, evaluate 2023-2024.

Assumes this repository still sits inside the research framework it was
extracted from (``backtest_framework/industry_aware_gnn``), so it can locate
``factors/`` and rebuild the feature panel automatically. Point --factor-dir
and --feature-panel elsewhere for a standalone checkout.

    python scripts/reproduce_run.py
    python scripts/reproduce_run.py --skip-prepare      # reuse a cached panel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import iagnn  # noqa: E402  (torch-before-numpy; see iagnn/_bootstrap.py)

FRAMEWORK_ROOT = REPO_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework-root", default=str(FRAMEWORK_ROOT))
    parser.add_argument("--factor-dir", default=str(FRAMEWORK_ROOT / "factors"))
    parser.add_argument("--feature-panel", default=str(REPO_ROOT / "artifacts" / "feature_panel.parquet"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts"))
    parser.add_argument("--factor-name", default="GNN_IC4Net")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Reuse an existing feature panel instead of rebuilding it.")
    parser.add_argument("--no-framework-output", action="store_true",
                        help="Do not write Factors_<name>_*.parquet back into factors/.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    panel_path = Path(args.feature_panel)

    if not args.skip_prepare or not panel_path.exists():
        from iagnn.ingest import build_feature_panel
        build_feature_panel(
            output_path=panel_path,
            start_date="2019-01-01",
            end_date="2024-12-31",
            framework_root=args.framework_root,
        )

    cfg = iagnn.default_config(
        feature_panel_path=panel_path,
        factor_dir=args.factor_dir,
        factor_name=args.factor_name,
        output_dir=Path(args.output_dir),
    )
    result = iagnn.run(cfg, write_framework_output=not args.no_framework_output)

    print("\nArtifacts:")
    for key, path in sorted(result.artifacts.items()):
        print(f"  {key:24s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
