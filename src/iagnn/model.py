"""The residual industry-GraphSAGE scorer.

The design commitment is that the graph is a *correction*, not the signal.
For stock i on date t:

    h_self = MLP_self(x)                     own-stock representation
    base   = Linear(h_self)                  own-stock score
    p      = mean of x over i's k nearest same-industry peers
    d      = x - p                           deviation from those peers
    h_graph= GraphSAGE(x, edges)             one hop of mean aggregation
    delta  = MLP_graph([x, h_self, p, d, h_graph])
    score  = base + alpha * delta

`alpha` is a learned scalar squashed into a narrow band (default 0.1-0.3), so
the optimiser can decide how much the industry context is worth but cannot
drown out `base` with it. If the graph branch were unbounded, the cheapest
way to cut the loss early in training is to lean entirely on peer means,
which produces a factor that ranks industries rather than stocks.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SparseMeanSAGE(nn.Module):
    """One hop of GraphSAGE with mean aggregation.

    Written against plain ``index_add_`` instead of torch_geometric: the
    operation is four lines, and the dependency is otherwise a large native
    build to install and pin for no functional gain.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        neighbor_sum = torch.zeros_like(x).index_add_(0, dst, x[src])
        counts = (
            torch.zeros(x.size(0), device=x.device)
            .index_add_(0, dst, torch.ones(src.size(0), device=x.device))
            .clamp(min=1)
            .unsqueeze(1)
        )
        return F.relu(self.lin_self(x) + self.lin_neigh(neighbor_sum / counts))


class IndustryGraphFactorModel(nn.Module):
    def __init__(self, n_features: int, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        hidden = cfg.hidden_dim

        self.self_mlp = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(cfg.dropout)
        )
        self.base_head = nn.Linear(hidden, 1)
        self.sage = SparseMeanSAGE(n_features, hidden)

        concat_dim = n_features * 3 + hidden * 2  # x, p, d | h_self, h_graph
        self.graph_mlp = nn.Sequential(
            nn.Linear(concat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, 1),
        )
        # sigmoid(0) = 0.5, so alpha starts at the middle of its band.
        self._alpha_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self) -> torch.Tensor:
        return self.cfg.alpha_min + self.cfg.alpha_range * torch.sigmoid(self._alpha_raw)

    def forward(self, x: torch.Tensor, p: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_self = self.self_mlp(x)
        base = self.base_head(h_self).squeeze(-1)
        h_graph = self.sage(x, edge_index)
        z = torch.cat([x, h_self, p, x - p, h_graph], dim=1)
        delta = self.graph_mlp(z).squeeze(-1)
        return base + self.alpha * delta

    def components(self, x, p, edge_index) -> dict[str, torch.Tensor]:
        """Same forward pass, but returns `base`, `delta` and `alpha` too.

        Useful for auditing how much of the final score the graph branch is
        actually responsible for on a given date.
        """
        h_self = self.self_mlp(x)
        base = self.base_head(h_self).squeeze(-1)
        h_graph = self.sage(x, edge_index)
        z = torch.cat([x, h_self, p, x - p, h_graph], dim=1)
        delta = self.graph_mlp(z).squeeze(-1)
        alpha = self.alpha
        return {"base": base, "delta": delta, "alpha": alpha, "score": base + alpha * delta}
