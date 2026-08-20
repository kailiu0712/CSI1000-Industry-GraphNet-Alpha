# Industry-aware GNN for CSI1000 stock picking

A cross-sectional alpha model for the CSI 1000 universe. Each stock is scored
from its own factor values, then **corrected by what its nearest
same-industry, similar-size peers look like that day**.

The correction is the point. A stock's raw factor values say how it looks in
absolute terms; they don't say whether it looks that way because of something
specific to the company or because its whole industry does. A high-turnover
reading means something different for a semiconductor name in a week when
every semiconductor name is turning over than it does in a quiet week. The
model builds a graph of same-industry peers each day, aggregates over it, and
learns a correction term from the gap.

Trained on **2019-2022**, evaluated on **2023-2024** with no refitting.

---

## The model

For stock *i* on date *t*, with standardized features `x`:

```
h_self  = MLP_self(x)                              own-stock representation
base    = Linear(h_self)                           own-stock score
p       = mean of x over i's k nearest same-industry peers
d       = x - p                                    deviation from those peers
h_graph = GraphSAGE(x, edges)                      one hop of mean aggregation
delta   = MLP_graph([x, h_self, p, d, h_graph])    the industry correction
score   = base + alpha * delta
```

`alpha` is learned but squashed into a narrow band (default `[0.1, 0.3]`).
That bound is a design commitment, not a hyperparameter that happened to work:
if the graph branch were unbounded, the cheapest early loss reduction is to
lean entirely on peer means, and the result is a factor that ranks *industries*
rather than stocks. Bounding `alpha` keeps the graph a refinement of `base`.

### The graph

One graph per trading day, rebuilt from that day's cross-section:

- **Nodes** — every CSI 1000 constituent with a known industry and at least
  one non-missing feature.
- **Edges** — within each industry, connect each stock to its `k = 10`
  nearest peers by size. Size is proxied by the CSI 1000 index weight, which
  is free-float-cap weighted, so ordering by weight orders by float cap.
- Two edge sets come out of the same k-NN step, and the distinction matters.
  The **directed** set (i to its own k peers) defines the peer mean `p`; it
  stays directed and exactly-k so a crowded mid-cap's peer mean doesn't
  silently absorb every small-cap that picked *it*. The **symmetrized** set
  with self-loops is what GraphSAGE aggregates over, because message passing
  wants an undirected neighbourhood.
- A stock alone in its industry that day gets `p = x`, so its deviation is
  exactly zero — "no peer information" rather than a zero vector that would
  read downstream as "maximally cheap versus peers".

Graphs depend on no model parameter, so they are built once and reused across
every epoch.

### The objective

Everything is computed **per date**, then averaged over the dates in a batch:

```
loss = -IC + 0.05 * Huber(z(score), z(label)) + 0.01 * relu(0.3 - std(score))^2
```

- **-IC** is the term that matters: the daily cross-sectional correlation
  between score and forward return.
- **Huber** is mild shaping. IC alone is scale- and shift-invariant, which
  leaves the score's level unconstrained and makes early optimisation wander.
- **variance collapse** penalises a cross-section whose score spread has
  shrunk below a floor. Since -IC doesn't care about scale, nothing else
  stops the network from drifting toward near-constant output and letting
  float noise carry the correlation.

Pooling returns across dates instead would let a few high-volatility days
dominate the gradient and would reward predicting the market's direction —
not what a cross-sectional factor is for.

---

## Results

Trained on 2019-01-01 .. 2022-12-31; the test numbers come from a model that
never saw a label from the test window.

**The holding period is close to close.** The signal is formed from day T's
session and the position is entered in that day's closing auction, held
through the overnight gap, and exited at the T+1 close. That is the strategy
this model is built for, and `close_t1` — today's close to tomorrow's close —
is its P&L. It is also exactly the label the model is trained on, so the
objective and the evaluation measure the same thing.

Returns are computed on the full listed panel before the universe filter, so
they measure the stock's genuine next trading day rather than its next day in
the index.

<!-- RESULTS_TABLE -->

Windows: 2019-01-02 .. 2022-12-29 | 2023-01-03 .. 2024-11-01

| metric | Train (in-sample) | **Test (out-of-sample)** |
| --- | --- | --- |
| Trading days | 876 | 396 |
| Mean RankIC | 0.1104 | 0.0891 |
| RankIC std | 0.0637 | 0.0837 |
| **ICIR** | **1.734** | **1.066** |
| IC > 0 frequency | 95.9% | 86.1% |
| IC t-stat | 51.3 | 21.2 |
| Top decile, mean daily return | 0.4620% | 0.2474% |
| Bottom decile, mean daily return | -0.5188% | -0.3580% |
| Benchmark (equal-weighted), mean daily | 0.0356% | -0.0073% |
| **Long-short Sharpe** (Q10-Q1, ann.) | **21.30** | **10.25** |
| **Long-only Sharpe** (Q10, ann.) | **4.68** | **2.43** |
| Long-only Sharpe, excess of benchmark | 13.86 | 7.78 |
| Benchmark Sharpe (ann.) | 0.39 | -0.07 |
| Long-short cumulative (additive) | 859.2% | 239.8% |
| Long-short max drawdown | -1.7% | -3.1% |
| Long-only cumulative (additive) | 404.7% | 98.0% |
| Decile monotonicity (Spearman) | 1.000 | 1.000 |
| Top-decile daily turnover | 77.4% | 78.4% |

<!-- /RESULTS_TABLE -->

### Out-of-sample backtest, 2023-01-03 .. 2024-11-01

Every figure below is regenerated by `python scripts/reproduce_run.py`; the
full set, including the training window, is in [`docs/figures/`](docs/figures).
Cumulative curves are additive (`cumsum`), matching the parent framework's
plots.

The decile fan opens cleanly and never crosses — Q10 on top, Q1 at the
bottom, every rank in between where it belongs, over 396 out-of-sample days:

![Factor summary, out-of-sample](docs/figures/GNN_IC4Net_test_close_t1_summary.png)

The decile bar chart shows the same thing as levels. The spread is monotone
across all ten buckets rather than driven by one extreme: Q10 earns +0.247% a
day and Q1 loses -0.358%, and every step in between moves the right way
(Spearman monotonicity 1.000).

![Decile returns, out-of-sample](docs/figures/GNN_IC4Net_test_close_t1_decile_bar.png)

Worth noting what the benchmark did over this window: the equal-weighted
CSI 1000 returned an annualised Sharpe of **-0.07**, essentially flat to
slightly down. The long-only top decile posted 2.43 over the same dates, so
that figure is not market drift in disguise — the excess-of-benchmark Sharpe
is 7.78.

![Quintile cumulative return](docs/figures/GNN_IC4Net_test_close_t1_quintile_cumret.png)

The RankIC series shows where the stability comes from: IC is positive on 86%
of days rather than being carried by a few large ones.

![RankIC over time](docs/figures/GNN_IC4Net_test_close_t1_ic.png)

### Barra style and industry attribution

Each day the standardized factor is regressed on that day's Barra style
exposures and industry dummies with no intercept — the dummies partition the
universe and already act as one. Averaging the daily coefficients gives the
factor's mean tilt; `1 - R²` is the share no known style or industry explains.

![Barra style and industry attribution](docs/figures/GNN_IC4Net_test_barra_industry.png)

**84% of the factor is unexplained by any Barra style or industry** — which,
for a signal meant to be alpha rather than a repackaged risk premium, is the
result you want. Of the 16% that is explained:

- **Low residual volatility (-0.29)** is by far the strongest style tilt. The
  factor systematically prefers calmer names. This is the one exposure large
  enough to matter for risk budgeting, and it is a known compensated factor —
  some of the measured edge is being paid for bearing it in reverse.
- **Small negative size / non-linear size / liquidity** (-0.05 to -0.10): a
  mild tilt toward smaller, less liquid names, unsurprising in a CSI 1000
  universe and worth watching because it compounds the turnover problem.
- **Growth, leverage and earnings yield are essentially zero** — the four
  fundamental inputs are not driving the signal.
- **Industry tilts are small** — every one of the 31 sits within ±0.11 except
  the two financials. Long agriculture (+0.10), steel (+0.10), pharma (+0.09);
  short banks (-0.43) and non-bank financials (-0.15). The bank tilt is the
  one concentrated bet in the book and is four times the next largest.

That the industry tilts stay small is a direct check on the architecture's
central claim: bounding `alpha` was supposed to keep the graph branch from
turning the factor into an industry ranking, and the attribution says it did.

### Against the previous model

Same universe, same out-of-sample dates, same metric — out-of-sample RankICIR
for this model against the industry-GNN it replaces (`Composite10`: the same
architecture on the older 28-factor input set) and two earlier variants:

| out-of-sample, 2023-01-03 .. 2024-10-31 | Composite5 | Composite9 | Composite10 | **GNN_IC4Net** |
| --- | --- | --- | --- | --- |
| Mean RankIC | 0.042 | 0.076 | 0.082 | **0.089** |
| **RankICIR** | 0.490 | 0.919 | 0.921 | **1.065** |
| IC > 0 frequency | 69.7% | 84.6% | 83.3% | **86.1%** |

The new input set is a clear step up on every column, and the gain is in
consistency as much as in size: mean RankIC improves 8% over Composite10
while ICIR improves 16%, because the IC is steadier day to day rather than
larger on its good days.

The jump from Composite5 to the rest is the input set, not the architecture —
Composite5 runs the same industry-GNN over only 8 raw factors. The jump from
Composite9 to Composite10 is the architecture at fixed inputs, and it is
small (0.919 to 0.921). What separates this model is the 22-factor screen
feeding it, with the graph contributing a consistent but secondary refinement.
That ordering is worth keeping in mind before attributing the result to the
GNN alone.

### Other caveats

**The test window stops at 2024-11-01, not 2024-12-31.** Not a modelling
choice — the daily price and index-weight tables this project is evaluated
against end there.

**Every Sharpe and cumulative figure is gross of costs, and turnover is the
binding constraint.** Top-decile turnover is ~78% per day — the book is
almost entirely replaced at every close. Over 396 days that is on the order
of 300 round trips, so the out-of-sample long-short Sharpe of 10.25 and the
+239.8% cumulative figure describe a frictionless version of the signal, not
a P&L. At a few basis points of round-trip cost most of it is gone. Sizing
this into something tradable means a holding-period or turnover penalty in
the objective, which this model does not have.

**Execution has to happen at the close, and that is a real assumption.**
Twelve of the 22 inputs are last-30-minute intraday readings, so the signal is
only fully formed minutes before the auction it must be traded in. A
close-to-close backtest charges nothing for that. Any slippage between the
observed close and the achievable fill comes straight out of a +0.247%/day
top decile.

**The long-only Sharpe is the number most likely to be misread** — though
here it holds up. It is 2.43, against an equal-weighted universe that
returned -0.07 over the same window, giving an excess of 7.78. Both are
reported because a long-only Sharpe quoted without its benchmark usually says
more about the market than about the model; in this window the market
contributed nothing, so the figure is the factor's.

**Train-vs-test decay is real but modest.** ICIR falls from 1.73 to 1.07 and
mean RankIC from 0.110 to 0.089 — roughly a third of the in-sample edge given
back — while decile monotonicity stays at 1.000 and IC stays positive on 86%
of out-of-sample days.

Reproduce with `python scripts/reproduce_run.py`. The tables above the caveats
are rendered from the run's own metrics file by
`scripts/update_readme_results.py`, so they cannot drift from the artifacts.

---

## Inputs

22 factors, the set kept after the elastic-net / IC screening in the parent
research project: twelve intraday minute-bar structure factors, six daily
technical factors, and four point-in-time fundamental ratios.

| group | factors |
| --- | --- |
| Intraday structure (12) | `1.1 Q3`, `1.2 M4`, `1.4 lz 10`, `1.7 DealRecovery Slope 30min`, `1.8 EarlyAfternoonRecovery VolumeWeighted VolatilityAdjusted`, `1.9 ExtremeHighReversal Last30Min`, `1.10 UpsideVolRatio Vol5DivVol20`, `1.11 VWAP Close Ratio Last30Min`, `1.12 VolumeConcentration Last30Min PriceWeighted`, `HF_GapII_SUM3D`, `PMCloseRange`, `Dratio` |
| Daily technical (6) | `AmountMA20`, `AmountIR5`, `ATR5`, `VEMA5`, `VRSI60`, `VR_N` |
| Fundamental, point-in-time (4) | `EP_TTM`, `SP1_TTM`, `CurrentAssetsTRate`, `FC_AssetImpairmentToRevenue_TTM` |

`python -m iagnn features` prints the exact column contract.

### Data contract

The raw inputs — ~16 GB of CSI 1000 1-minute bars plus point-in-time financial
statements — are **not distributed with this repository**. The pipeline's
input is a *prepared feature panel*: a parquet with

| column | type | meaning |
| --- | --- | --- |
| `date` | `datetime64` | one row per stock per trading day |
| `instrument` | `str` | 6-digit ticker, zero-padded, no exchange suffix |
| *22 feature columns* | `float` | named exactly as listed above |

plus the research framework's `factors/<year>/` tree for daily prices, index
weights, and `secucode_industry_map.csv`.

`python -m iagnn prepare` builds the panel when the parent framework is
importable; otherwise supply the parquet yourself in the shape above.

---

## Usage

```bash
pip install -e ".[dev]"

# 1. build the 22-feature panel (needs the parent research framework)
python -m iagnn prepare \
    --framework-root /path/to/backtest_framework \
    --output artifacts/feature_panel.parquet

# 2. train on 2019-2022, score everything, evaluate 2023-2024
python -m iagnn run \
    --feature-panel artifacts/feature_panel.parquet \
    --factor-dir /path/to/backtest_framework/factors

# 3. re-report metrics from saved scores, without retraining
python -m iagnn evaluate \
    --scores artifacts/GNN_IC4Net_scores.parquet \
    --factor-dir /path/to/backtest_framework/factors
```

Or from Python:

```python
import iagnn

cfg = iagnn.default_config(
    feature_panel_path="artifacts/feature_panel.parquet",
    factor_dir="/path/to/backtest_framework/factors",
)
result = iagnn.run(cfg)
print(result.summary())
print(result.test_metrics["icir"])
```

`iagnn run` writes, into `artifacts/`: the scored factor, per-window RankIC
series and decile returns, a metrics table, the training history, a torch
checkpoint, a torch-free JSON weight dump, and a config manifest. Figures go
to `docs/figures/` — four per (window, holding period) plus the Barra
attribution, which needs `--barra-dir` pointing at the exposure parquets and
is skipped without complaint when they are absent. `--no-figures` turns the
whole figure step off. It also
writes the scored factor into the parent framework's `factors/<year>/` tree in
that project's own file layout, so its single-factor test picks the factor up
unchanged (`--no-framework-output` disables this).

---

## Layout

```
src/iagnn/
  config.py       every knob, as dataclasses; serialises to a run manifest
  features.py     the 22-factor contract (+ a drift check against the parent project)
  ingest.py       optional bridge that builds the feature panel
  data.py         panel assembly: features + prices + index weights + industry
  preprocess.py   same-day winsorize / z-score / forward-return label
  graph.py        per-date industry k-NN graphs, and batched collation
  model.py        SparseMeanSAGE + the residual scorer
  losses.py       per-date IC objective
  trainer.py      training and scoring loops
  evaluate.py     RankIC, ICIR, decile returns, Sharpes, turnover
  barra.py        style/industry attribution by daily cross-sectional regression
  plots.py        the five figures, in the parent framework's house style
  report.py       turns a finished run into figures
  export.py       checkpoints, JSON weights, framework hand-off
  pipeline.py     end-to-end orchestration
  cli.py          `python -m iagnn ...`
scripts/          one-command reproduction; README table renderer
docs/figures/     the committed backtest figures the README shows
tests/            50 tests; run with `pytest`
```

---

## Design notes and limitations

**Batching across dates is exact, not an approximation.** A batch merges its
dates into one disjoint-union graph with per-date node offsets. No edge ever
crosses a date boundary, and mean aggregation only ever reads a node's own
edges, so the result is identical to running each date separately. The gain is
throughput: one large forward/backward pass instead of dozens of small ones,
which on CPU is dominated by dispatch overhead.

**Early stopping watches the training objective, not a held-out slice.** The
held-out years are the evaluation this project exists to produce; using them
to decide when to stop would quietly make the out-of-sample report partly
in-sample.

**The industry map is a current classification, not point-in-time.** A stock
carries the same industry on every date. Reclassifications are infrequent
enough that the graph topology barely moves, but a stock that changed sector
mid-sample is grouped by where it ended up. Fixing this needs a dated
reclassification table, which the parent project does not have.

**Index weight is a size proxy, not market cap.** No direct market-cap field
exists in the parent factor set. Index weights are free-float-cap weighted, so
the *ordering* is right, which is all the k-NN step uses.

**Missing features are filled with the daily mean.** Filling with `0.0` after
z-scoring means "average on this factor today", which keeps the row in the
graph rather than deleting a node and tearing a hole in its industry's peer
structure.

**The training label and the evaluation returns are configured separately.**
`DataConfig.label_price_col` / `label_horizon` set what the model fits;
`DataConfig.eval_returns` sets what it is scored against. Both default to the
close-to-close holding period, so the objective and the report measure the
same thing. They are kept as separate settings because a factor is often
worth scoring against holding periods it was not trained for — add entries to
`eval_returns` and each is reported, with its own metrics and figures, rather
than being averaged into one number.

**On Windows, torch must be imported before numpy/pandas** or its DLLs can
fail to load (`WinError 1114`). The package's `__init__` handles this; see
`src/iagnn/_bootstrap.py`.

---

## License

MIT — see [LICENSE](LICENSE).
