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

The factor is scored against **two return conventions**, because they answer
different questions. `close_t1` is today's close to tomorrow's close — it
matches the training label, but assumes you can trade at a price you only
observe after the fact. `open5_t2` is the first-5-minute TWAP one day ahead to
the next: a full day of implementation lag, and the convention the parent
project's own single-factor test reports. Both are computed on the full listed
panel before the universe filter, so each measures the stock's genuine next
trading days rather than its next days in the index.

<!-- RESULTS_TABLE -->

### Close &rarr; close, T+1

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
| Long-short Sharpe (annualised) | 20.96 | 10.09 |
| Long-short cumulative return | 504952.8% | 973.5% |
| Decile monotonicity (Spearman) | 1.000 | 1.000 |
| Top-decile daily turnover | 77.4% | 78.4% |

### Open5TWAP T+1 &rarr; T+2 (tradable)

Windows: 2019-01-02 .. 2022-12-29 | 2023-01-03 .. 2024-11-01

| metric | Train (in-sample) | **Test (out-of-sample)** |
| --- | --- | --- |
| Trading days | 876 | 395 |
| Mean RankIC | 0.0342 | 0.0266 |
| RankIC std | 0.0601 | 0.0784 |
| **ICIR** | **0.569** | **0.340** |
| IC > 0 frequency | 71.9% | 64.1% |
| IC t-stat | 16.8 | 6.8 |
| Top decile, mean daily return | 0.0344% | 0.0320% |
| Bottom decile, mean daily return | -0.1509% | -0.0440% |
| Long-short Sharpe (annualised) | 4.30 | 1.62 |
| Long-short cumulative return | 396.4% | 33.6% |
| Decile monotonicity (Spearman) | 0.818 | 0.612 |
| Top-decile daily turnover | 77.4% | 78.4% |

<!-- /RESULTS_TABLE -->

### Against the previous model

Same universe, same out-of-sample dates, same metric — out-of-sample RankICIR
for this model against the industry-GNN it replaces (`Composite10`: the same
architecture on the older 28-factor input set) and two earlier variants:

| out-of-sample ICIR, 2023-01-03 .. 2024-10-31 | Composite5 | Composite9 | Composite10 | **GNN_IC4Net** |
| --- | --- | --- | --- | --- |
| Close &rarr; close, T+1 | 0.490 | 0.919 | 0.921 | **1.065** |
| Open5TWAP T+1 &rarr; T+2 (tradable) | 0.250 | 0.322 | **0.350** | 0.340 |

**This is the most important thing to understand about the model.** On the
horizon it is trained for, the new input set is a clear step up — RankIC 0.089
against 0.082, ICIR 1.07 against 0.92. Give the signal one day of
implementation lag and that advantage disappears: 0.340 against 0.350, a tie
inside noise.

The explanation is in the inputs. Twelve of the 22 factors are last-30-minute
intraday microstructure readings — closing VWAP ratios, volume concentration,
late-session reversals. They describe where a stock ends the day relative to
its own session, which is exactly the kind of information that predicts the
*immediately* following move and then decays. A return that only starts the
next morning has already given most of it away. The older, more
fundamental-weighted input set gives up less to the lag.

So: treat the close-to-close column as evidence the architecture and the
feature screen work, and the tradable column as the number to beat. Closing
that gap means training on the lagged label (`label_price_col="Open5TWAP"`,
`label_horizon=2`), not tuning the model.

### Other caveats

**The test window stops at 2024-11-01, not 2024-12-31.** Not a modelling
choice — the daily price and index-weight tables this project is evaluated
against end there.

**Every long-short line is gross of costs, and none of them survives them.**
Top-decile turnover is ~78% per day — the strategy replaces almost its whole
book daily. On the tradable convention the out-of-sample long-short leg earns
+33.6% gross over ~1.6 years; ~0.78 of the book turning daily over ~395 days
is on the order of 300 round trips, so even a few basis points per trade
erases it. These rows belong here as evidence that the ranking separates
returns, not as a P&L claim. The IC, its stability, and the decile
monotonicity are the load-bearing results — and note that on the tradable
convention monotonicity itself falls to 0.61 out of sample, against 1.00 on
the training horizon.

**Train-vs-test decay is real but modest.** On the training horizon, ICIR
falls from 1.73 to 1.07 and mean RankIC from 0.110 to 0.089 — roughly a third
of the in-sample edge given back — while decile monotonicity stays at 1.000
and IC stays positive on 86% of out-of-sample days.

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
checkpoint, a torch-free JSON weight dump, and a config manifest. It also
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
  evaluate.py     RankIC, ICIR, decile returns, turnover
  export.py       checkpoints, JSON weights, framework hand-off
  pipeline.py     end-to-end orchestration
  cli.py          `python -m iagnn ...`
scripts/          one-command reproduction
tests/            34 tests; run with `pytest`
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
`DataConfig.eval_returns` sets what it is scored against, and every entry is
reported. Keeping them apart is what makes the lag sensitivity in the results
section visible instead of hidden behind a single number.

**On Windows, torch must be imported before numpy/pandas** or its DLLs can
fail to load (`WinError 1114`). The package's `__init__` handles this; see
`src/iagnn/_bootstrap.py`.

---

## License

MIT — see [LICENSE](LICENSE).
