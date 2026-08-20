import torch

from iagnn.config import LossConfig
from iagnn.losses import batch_loss, date_loss


def test_perfect_ranking_beats_inverted_ranking():
    cfg = LossConfig()
    label = torch.randn(400)
    good = date_loss(label.clone(), label, cfg, min_valid=10)
    bad = date_loss(-label.clone(), label, cfg, min_valid=10)

    assert good is not None and bad is not None
    assert good[0] < bad[0]
    assert float(good[1]) > 0.99 and float(bad[1]) < -0.99


def test_returns_none_when_too_few_labelled_names():
    label = torch.full((400,), float("nan"))
    label[:5] = torch.randn(5)
    assert date_loss(torch.randn(400), label, LossConfig(), min_valid=300) is None


def test_nan_labels_are_masked_not_propagated():
    label = torch.randn(400)
    label[::10] = float("nan")
    result = date_loss(torch.randn(400), label, LossConfig(), min_valid=10)
    assert result is not None
    assert torch.isfinite(result[0])


def test_collapse_penalty_activates_below_the_std_floor():
    cfg = LossConfig(min_score_std=1.0, huber_weight=0.0)
    label = torch.randn(400)
    # Same ranking (identical IC), different score scale.
    wide = date_loss(label * 5.0, label, cfg, min_valid=10)
    narrow = date_loss(label * 0.01, label, cfg, min_valid=10)
    assert narrow[0] > wide[0]


def test_batch_loss_averages_over_dates():
    cfg = LossConfig()
    label = torch.randn(800)
    date_idx = torch.cat([torch.zeros(400, dtype=torch.long), torch.ones(400, dtype=torch.long)])
    loss, ic_sum, n_used = batch_loss(label.clone(), label, date_idx, cfg, min_valid=10)

    assert n_used == 2
    assert ic_sum > 1.9  # both dates near IC = 1
    single = date_loss(label[:400], label[:400], cfg, min_valid=10)[0]
    assert torch.allclose(loss, single, atol=1e-4)
