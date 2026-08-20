"""Render the README's results table from a run's metrics CSV.

The numbers in the README are generated, not typed, so they cannot drift from
the artifacts a run actually produced. Replaces whatever currently sits
between the RESULTS_TABLE markers.

    python scripts/update_readme_results.py
    python scripts/update_readme_results.py --metrics artifacts/GNN_IC4Net_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- RESULTS_TABLE -->"
END = "<!-- /RESULTS_TABLE -->"

ROWS = [
    ("Trading days", "n_days", "{:,.0f}"),
    ("Mean RankIC", "ic_mean", "{:.4f}"),
    ("RankIC std", "ic_std", "{:.4f}"),
    ("**ICIR**", "icir", "**{:.3f}**"),
    ("IC > 0 frequency", "ic_positive_rate", "{:.1%}"),
    ("IC t-stat", "ic_t_stat", "{:.1f}"),
    ("Top decile, mean daily return", "top_decile_mean", "{:.4%}"),
    ("Bottom decile, mean daily return", "bottom_decile_mean", "{:.4%}"),
    ("Benchmark (equal-weighted), mean daily", "benchmark_daily_mean", "{:.4%}"),
    ("**Long-short Sharpe** (Q10-Q1, ann.)", "ls_sharpe", "**{:.2f}**"),
    ("**Long-only Sharpe** (Q10, ann.)", "long_only_sharpe", "**{:.2f}**"),
    ("Long-only Sharpe, excess of benchmark", "long_only_excess_sharpe", "{:.2f}"),
    ("Benchmark Sharpe (ann.)", "benchmark_sharpe", "{:.2f}"),
    ("Long-short cumulative (additive)", "ls_cumulative_return", "{:.1%}"),
    ("Long-short max drawdown", "ls_max_drawdown", "{:.1%}"),
    ("Long-only cumulative (additive)", "long_only_cumulative_return", "{:.1%}"),
    ("Decile monotonicity (Spearman)", "monotonicity", "{:.3f}"),
    ("Top-decile daily turnover", "top_decile_turnover", "{:.1%}"),
]



RETURN_LABELS = {
    "close_t1": "Close &rarr; close, T+1",
    "open5_t2": "Open5TWAP T+1 &rarr; T+2 (tradable)",
}


def render(metrics: pd.DataFrame) -> str:
    """One table per return convention, train and test side by side."""
    lines = [BEGIN, ""]
    returns = list(dict.fromkeys(metrics["returns"]))

    for ret in returns:
        block = metrics[metrics["returns"] == ret].set_index("window")
        windows = [w for w in ("train", "test") if w in block.index]
        headers = {"train": "Train (in-sample)", "test": "**Test (out-of-sample)**"}

        if len(returns) > 1:  # a single convention needs no heading to disambiguate
            lines.append(f"### {RETURN_LABELS.get(ret, ret)}")
            lines.append("")
        spans = " | ".join(
            f"{block.loc[w, 'start']} .. {block.loc[w, 'end']}" for w in windows
        )
        lines.append(f"Windows: {spans}")
        lines.append("")
        lines.append("| metric | " + " | ".join(headers[w] for w in windows) + " |")
        lines.append("| --- |" + " --- |" * len(windows))

        for label, key, fmt in ROWS:
            cells = []
            for window in windows:
                value = block.loc[window].get(key)
                cells.append("n/a" if pd.isna(value) else fmt.format(value))
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append(END)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", default=str(REPO_ROOT / "artifacts" / "GNN_IC4Net_metrics.csv"))
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    args = parser.parse_args()

    table = render(pd.read_csv(args.metrics))
    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")

    if BEGIN not in text:
        raise SystemExit(f"{readme} has no {BEGIN} marker to fill in.")
    head, _, rest = text.partition(BEGIN)
    tail = rest.partition(END)[2] if END in rest else rest.split("\n", 1)[1]

    readme.write_text(head + table + tail, encoding="utf-8")
    print(f"Updated {readme}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
