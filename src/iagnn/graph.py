"""Per-date industry graph construction.

One graph per trading day. Nodes are the stocks eligible that day; edges
connect each stock to the `k` same-industry peers closest to it in size.
Nothing here depends on model parameters, so graphs are built once and
reused across every epoch.

Two different edge sets come out of the same k-NN step, and the distinction
matters:

* the **directed peer set** -- stock i -> its own k nearest peers -- defines
  the peer mean ``p[i]``. It has to stay directed and exactly-k, otherwise a
  crowded mid-cap's peer mean would silently absorb every small-cap that
  picked *it* as a neighbour.
* the **symmetrized set with self-loops** is what the GraphSAGE layer
  aggregates over. Message passing wants an undirected neighbourhood, and
  the self-loop keeps a node's own state in its aggregate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class DateGraph:
    """One trading day's graph, ready to hand to the model."""

    date: pd.Timestamp
    x: torch.Tensor            # (n, n_features) standardized features
    p: torch.Tensor            # (n, n_features) same-industry peer means
    edge_index: torch.Tensor   # (2, n_edges) symmetrized + self-loops
    ids: np.ndarray            # (n,) SecuCode per node
    label: torch.Tensor | None = None  # (n,) forward return, when known

    def __len__(self) -> int:
        return self.x.shape[0]


def knn_within_group(
    local_idx: np.ndarray, size: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest peers by |size| distance, inside one industry.

    `local_idx` are positions in the day's node array; `size` is the matching
    size-proxy value for each. Returns ``(src, dst)`` where ``src[e]`` picked
    ``dst[e]`` as one of its neighbours. Self is excluded, and a group of
    size m yields exactly ``min(k, m-1)`` neighbours per node.
    """
    m = len(local_idx)
    if m <= 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    diffs = np.abs(size[:, None] - size[None, :])
    np.fill_diagonal(diffs, np.inf)
    kk = min(k, m - 1)
    nearest = np.argpartition(diffs, kk - 1, axis=1)[:, :kk]

    src = np.repeat(np.arange(m), kk)
    dst = nearest.reshape(-1)
    return local_idx[src], local_idx[dst]


def build_date_graph(
    day_df: pd.DataFrame,
    feature_cols: list[str],
    k: int = 10,
    min_nodes: int = 300,
    industry_col: str = "industry",
    size_col: str = "log_size",
    id_col: str = "SecuCode",
    label_col: str = "next_return",
) -> DateGraph | None:
    """Build one date's graph, or None if the cross-section is too thin.

    `day_df` must already be standardized (see `preprocess.standardize_features`).
    """
    if len(day_df) < min_nodes:
        return None

    day_df = day_df.reset_index(drop=True)
    n = len(day_df)
    x_np = day_df[feature_cols].to_numpy(dtype=np.float32)
    size = day_df[size_col].to_numpy()

    src_parts, dst_parts = [], []
    for _, positions in day_df.groupby(industry_col, sort=False).groups.items():
        positions = np.asarray(positions, dtype=np.int64)
        s, d = knn_within_group(positions, size[positions], k)
        src_parts.append(s)
        dst_parts.append(d)

    peer_src = np.concatenate(src_parts) if src_parts else np.empty(0, dtype=np.int64)
    peer_dst = np.concatenate(dst_parts) if dst_parts else np.empty(0, dtype=np.int64)

    # p[i] = mean of x[j] over i's own k nearest same-industry peers.
    peer_sum = np.zeros((n, len(feature_cols)), dtype=np.float32)
    peer_count = np.zeros(n, dtype=np.float32)
    np.add.at(peer_sum, peer_src, x_np[peer_dst])
    np.add.at(peer_count, peer_src, 1.0)

    isolated = peer_count == 0
    p_np = peer_sum / np.where(isolated, 1.0, peer_count)[:, None]
    # A stock alone in its industry today has no peer mean. Setting p = x
    # makes its deviation d = x - p exactly zero, i.e. "no peer information",
    # instead of an all-zeros vector that would read as "maximally cheap
    # relative to peers" downstream.
    p_np[isolated] = x_np[isolated]

    self_loop = np.arange(n, dtype=np.int64)
    sage_src = np.concatenate([peer_src, peer_dst, self_loop])
    sage_dst = np.concatenate([peer_dst, peer_src, self_loop])
    edge_index = torch.as_tensor(np.stack([sage_src, sage_dst]), dtype=torch.long)

    label = None
    if label_col in day_df.columns:
        label = torch.as_tensor(day_df[label_col].to_numpy(dtype=np.float32))

    return DateGraph(
        date=day_df["TradingDay"].iloc[0],
        x=torch.as_tensor(x_np),
        p=torch.as_tensor(p_np),
        edge_index=edge_index,
        ids=day_df[id_col].to_numpy(),
        label=label,
    )


def build_graph_cache(
    panel: pd.DataFrame,
    feature_cols: list[str],
    k: int = 10,
    min_nodes: int = 300,
    date_col: str = "TradingDay",
    progress: bool = True,
    **kwargs,
) -> dict[pd.Timestamp, DateGraph]:
    """One `DateGraph` per date that clears `min_nodes`, keyed by date."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is a soft dependency
        def tqdm(x, **_):
            return x

    cache: dict[pd.Timestamp, DateGraph] = {}
    groups = panel.groupby(date_col, sort=True)
    iterator = tqdm(groups, desc="Building per-date industry graphs", total=len(groups)) if progress else groups
    for _, day_df in iterator:
        graph = build_date_graph(day_df, feature_cols, k=k, min_nodes=min_nodes, **kwargs)
        if graph is not None:
            cache[graph.date] = graph
    return cache


def collate(
    graphs: list[DateGraph], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Merge a batch of dates into one disjoint-union graph.

    Node indices are offset per date and no edge ever crosses a date
    boundary, so this is mathematically identical to running each date on its
    own -- mean aggregation only ever reads a node's own edges. The point is
    purely throughput: one large forward/backward pass instead of `len(graphs)`
    small ones, which on CPU is dominated by dispatch overhead.

    Returns ``(x, p, edge_index, label, date_idx)``; `label` is None when any
    graph in the batch is unlabelled.
    """
    xs, ps, labels, srcs, dsts, date_idx = [], [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = len(g)
        xs.append(g.x)
        ps.append(g.p)
        labels.append(g.label)
        srcs.append(g.edge_index[0] + offset)
        dsts.append(g.edge_index[1] + offset)
        date_idx.append(torch.full((n,), i, dtype=torch.long))
        offset += n

    label = torch.cat(labels).to(device) if all(l is not None for l in labels) else None
    return (
        torch.cat(xs).to(device),
        torch.cat(ps).to(device),
        torch.stack([torch.cat(srcs), torch.cat(dsts)]).to(device),
        label,
        torch.cat(date_idx).to(device),
    )
