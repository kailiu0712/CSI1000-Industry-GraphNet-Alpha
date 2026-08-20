"""Training and scoring loops."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .config import PipelineConfig
from .graph import DateGraph, collate
from .losses import batch_loss
from .model import IndustryGraphFactorModel


@dataclass
class TrainHistory:
    loss: list[float] = field(default_factory=list)
    ic: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    stopped_epoch: int | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"epoch": range(1, len(self.loss) + 1),
                             "loss": self.loss, "train_ic": self.ic, "lr": self.lr})


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    graphs: dict[pd.Timestamp, DateGraph],
    cfg: PipelineConfig,
    verbose: bool = True,
) -> tuple[IndustryGraphFactorModel, torch.device, TrainHistory]:
    """Fit on `graphs` (the training window only) and return the fitted model.

    Early stopping watches the training objective rather than a held-out
    slice on purpose. The held-out years are the evaluation this whole
    exercise exists to produce, so touching them here -- even only to decide
    when to stop -- would quietly turn the out-of-sample report into a
    partially in-sample one.
    """
    tcfg = cfg.train
    device = resolve_device(tcfg.device)
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    model = IndustryGraphFactorModel(cfg.n_features, cfg.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg.lr_scheduler_factor, patience=tcfg.lr_scheduler_patience
    )

    dates = sorted(graphs)
    if not dates:
        raise ValueError("No training dates survived graph construction.")

    history = TrainHistory()
    best_loss, stale = float("inf"), 0

    try:
        from tqdm import tqdm
        epochs = tqdm(range(tcfg.max_epochs), desc="Training industry-GNN")
        write = tqdm.write
    except ImportError:  # pragma: no cover
        epochs, write = range(tcfg.max_epochs), print

    model.train()
    for epoch in epochs:
        epoch_loss, epoch_ic, n_dates = 0.0, 0.0, 0

        for start in range(0, len(dates), tcfg.batch_dates):
            batch = [graphs[d] for d in dates[start:start + tcfg.batch_dates]]
            x, p, edge_index, label, date_idx = collate(batch, device)
            if label is None:
                raise ValueError("Training graphs must carry labels.")

            score = model(x, p, edge_index)
            loss, ic_sum, n_used = batch_loss(
                score, label, date_idx, cfg.loss, min_valid=cfg.graph.min_eligible_per_date
            )
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.detach()) * n_used
            epoch_ic += ic_sum
            n_dates += n_used

        avg_loss = epoch_loss / max(n_dates, 1)
        avg_ic = epoch_ic / max(n_dates, 1)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.loss.append(avg_loss)
        history.ic.append(avg_ic)
        history.lr.append(current_lr)

        if hasattr(epochs, "set_postfix"):
            epochs.set_postfix(loss=f"{avg_loss:+.5f}", ic=f"{avg_ic:+.5f}",
                               alpha=f"{float(model.alpha.detach()):.3f}", lr=f"{current_lr:.1e}")

        if avg_loss < best_loss - tcfg.early_stop_min_delta:
            best_loss, stale = avg_loss, 0
        else:
            stale += 1
            if stale >= tcfg.early_stop_patience:
                history.stopped_epoch = epoch + 1
                if verbose:
                    write(f"  early stop at epoch {epoch + 1}/{tcfg.max_epochs} "
                          f"(no improvement for {tcfg.early_stop_patience} epochs)")
                break

    return model, device, history


@torch.no_grad()
def predict(
    model: IndustryGraphFactorModel,
    device: torch.device,
    graphs: dict[pd.Timestamp, DateGraph],
    cfg: PipelineConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every date in `graphs`. Returns ``TradingDay, SecuCode, <factor>``."""
    model.eval()
    dates = sorted(graphs)
    step = cfg.train.batch_dates
    rows = []

    try:
        from tqdm import tqdm
        chunks = tqdm(range(0, len(dates), step), desc=f"Scoring {cfg.factor_name}",
                      total=(len(dates) + step - 1) // step, disable=not verbose)
    except ImportError:  # pragma: no cover
        chunks = range(0, len(dates), step)

    for start in chunks:
        batch_dates = dates[start:start + step]
        batch = [graphs[d] for d in batch_dates]
        x, p, edge_index, _, date_idx = collate(batch, device)
        score = model(x, p, edge_index).cpu().numpy()
        idx = date_idx.cpu().numpy()
        for i, date in enumerate(batch_dates):
            mask = idx == i
            rows.append(pd.DataFrame({
                "TradingDay": date,
                "SecuCode": graphs[date].ids,
                cfg.factor_name: score[mask],
            }))

    return pd.concat(rows, ignore_index=True)
