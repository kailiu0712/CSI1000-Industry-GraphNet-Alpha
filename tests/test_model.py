import torch

from iagnn.config import ModelConfig
from iagnn.graph import build_date_graph
from iagnn.model import IndustryGraphFactorModel, SparseMeanSAGE


def test_sage_mean_aggregation_is_the_actual_mean():
    """With identity-ish weights, one node's output must be the mean of its
    neighbourhood -- the property the whole graph branch rests on."""
    layer = SparseMeanSAGE(2, 2)
    with torch.no_grad():
        layer.lin_self.weight.zero_()
        layer.lin_self.bias.zero_()
        layer.lin_neigh.weight.copy_(torch.eye(2))
        layer.lin_neigh.bias.zero_()

    x = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    # node 0 aggregates from nodes 1 and 2 -> mean = 4.0
    edge_index = torch.tensor([[1, 2], [0, 0]])
    out = layer(x, edge_index)
    assert torch.allclose(out[0], torch.tensor([4.0, 4.0]))


def test_alpha_stays_inside_its_band():
    cfg = ModelConfig(alpha_min=0.1, alpha_range=0.2)
    model = IndustryGraphFactorModel(5, cfg)
    tol = 1e-6  # sigmoid saturates in float32, so the bounds are hit exactly
    for raw in (-50.0, -1.0, 0.0, 1.0, 50.0):
        with torch.no_grad():
            model._alpha_raw.fill_(raw)
            alpha = float(model.alpha)
        assert cfg.alpha_min - tol <= alpha <= cfg.alpha_min + cfg.alpha_range + tol


def test_score_decomposes_as_base_plus_alpha_delta(synthetic_panel, feature_cols):
    day = synthetic_panel[synthetic_panel["TradingDay"] == synthetic_panel["TradingDay"].min()]
    graph = build_date_graph(day, feature_cols, k=5, min_nodes=10)

    model = IndustryGraphFactorModel(len(feature_cols)).eval()
    with torch.no_grad():
        score = model(graph.x, graph.p, graph.edge_index)
        parts = model.components(graph.x, graph.p, graph.edge_index)

    assert torch.allclose(score, parts["base"] + parts["alpha"] * parts["delta"], atol=1e-6)
    assert score.shape == (len(graph),)


def test_forward_is_permutation_equivariant(synthetic_panel, feature_cols):
    """Relabelling the nodes must permute the scores, not change them --
    otherwise the factor would depend on row order in the input file."""
    day = synthetic_panel[synthetic_panel["TradingDay"] == synthetic_panel["TradingDay"].min()]
    graph = build_date_graph(day, feature_cols, k=5, min_nodes=10)
    model = IndustryGraphFactorModel(len(feature_cols)).eval()

    perm = torch.randperm(len(graph))
    inverse = torch.argsort(perm)
    with torch.no_grad():
        original = model(graph.x, graph.p, graph.edge_index)
        shuffled = model(
            graph.x[perm], graph.p[perm], inverse[graph.edge_index]
        )
    assert torch.allclose(original, shuffled[inverse], atol=1e-5)
