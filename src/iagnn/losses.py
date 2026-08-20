"""Per-date loss.

The objective is rank quality *inside a single cross-section*, so every term
is computed per date and then averaged over the dates in a batch. Pooling
returns across dates would let a few high-volatility days dominate the
gradient and would reward predicting the market's direction rather than the
relative ordering of stocks, which is not what a cross-sectional factor is
for.

    loss = -IC + huber_weight * Huber(z(score), z(label))
              + var_weight * relu(min_std - std(score))^2

* **-IC** is the term that matters: the Pearson correlation between the
  standardized score and the standardized forward return, i.e. the factor's
  information coefficient that day.
* **Huber** on the standardized pair is a mild shaping term. IC alone is
  scale- and shift-invariant, which leaves the score's level unconstrained
  and makes early optimisation wander.
* **variance collapse** penalises a cross-section whose raw score spread has
  shrunk below a floor. -IC is invariant to score scale, so nothing else in
  the objective stops the network from drifting toward near-constant output
  and letting float noise carry the correlation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig


def _standardize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    std = t.std(unbiased=True) + 1e-8
    return (t - t.mean()) / std, std


def date_loss(
    score: torch.Tensor,
    label: torch.Tensor,
    cfg: LossConfig,
    min_valid: int = 300,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(loss, ic)`` for one date, or None if too few labelled names.

    Individual stocks can lack a forward return on an otherwise healthy date
    (a trading halt, a delisting), so validity is checked after masking
    rather than assumed from the graph's node count.
    """
    valid = ~torch.isnan(label)
    if int(valid.sum()) < min_valid:
        return None

    score, label = score[valid], label[valid]
    score_z, raw_std = _standardize(score)
    label_z, _ = _standardize(label)

    ic = (score_z * label_z).mean()
    huber = F.huber_loss(score_z, label_z, delta=1.0)
    collapse = torch.relu(cfg.min_score_std - score.std(unbiased=True)) ** 2

    total = -ic + cfg.huber_weight * huber + cfg.var_collapse_weight * collapse
    return total, ic.detach()


def batch_loss(
    score: torch.Tensor,
    label: torch.Tensor,
    date_idx: torch.Tensor,
    cfg: LossConfig,
    min_valid: int = 300,
) -> tuple[torch.Tensor | None, float, int]:
    """Mean per-date loss over a collated batch.

    Returns ``(loss, summed_ic, n_dates_used)``; `loss` is None when no date
    in the batch had enough labelled names to score.
    """
    total = None
    ic_sum = 0.0
    n_used = 0
    for i in range(int(date_idx.max().item()) + 1 if date_idx.numel() else 0):
        mask = date_idx == i
        result = date_loss(score[mask], label[mask], cfg, min_valid=min_valid)
        if result is None:
            continue
        loss, ic = result
        total = loss if total is None else total + loss
        ic_sum += float(ic)
        n_used += 1

    if n_used == 0:
        return None, 0.0, 0
    return total / n_used, ic_sum, n_used
