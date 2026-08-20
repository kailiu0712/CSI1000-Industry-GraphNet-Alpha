import numpy as np
import pandas as pd
import torch

from iagnn.graph import build_date_graph, build_graph_cache, collate, knn_within_group


def test_knn_excludes_self_and_returns_exactly_k():
    idx = np.arange(6)
    size = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    src, dst = knn_within_group(idx, size, k=2)

    assert len(src) == len(dst) == 6 * 2
    assert not (src == dst).any(), "a node must never be its own neighbour"
    # The extreme value 1.0 must pick the two next-smallest, 2.0 and 3.0.
    assert set(dst[src == 0]) == {1, 2}


def test_knn_caps_k_at_group_size_minus_one():
    idx = np.arange(3)
    size = np.array([1.0, 2.0, 3.0])
    src, dst = knn_within_group(idx, size, k=10)
    assert len(src) == 3 * 2  # min(k, m-1) = 2


def test_knn_singleton_group_has_no_edges():
    src, dst = knn_within_group(np.array([4]), np.array([1.0]), k=5)
    assert len(src) == 0 and len(dst) == 0


def test_graph_edges_never_cross_industries(synthetic_panel, feature_cols):
    day = synthetic_panel[synthetic_panel["TradingDay"] == synthetic_panel["TradingDay"].min()]
    graph = build_date_graph(day, feature_cols, k=5, min_nodes=10)

    industry = day["industry"].to_numpy()
    src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    assert (industry[src] == industry[dst]).all()


def test_graph_has_self_loops_and_is_symmetric(synthetic_panel, feature_cols):
    day = synthetic_panel[synthetic_panel["TradingDay"] == synthetic_panel["TradingDay"].min()]
    graph = build_date_graph(day, feature_cols, k=5, min_nodes=10)

    edges = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    for node in range(len(graph)):
        assert (node, node) in edges
    for a, b in edges:
        assert (b, a) in edges


def test_thin_cross_section_is_dropped(synthetic_panel, feature_cols):
    day = synthetic_panel[synthetic_panel["TradingDay"] == synthetic_panel["TradingDay"].min()]
    assert build_date_graph(day.head(20), feature_cols, k=5, min_nodes=300) is None


def test_isolated_node_gets_zero_deviation(feature_cols):
    """A stock alone in its industry must get p = x, hence d = x - p = 0."""
    df = pd.DataFrame({
        "TradingDay": pd.Timestamp("2020-01-01"),
        "SecuCode": ["000001", "000002", "000003"],
        "industry": ["A", "A", "LONELY"],
        "log_size": [1.0, 2.0, 3.0],
        "f1": [1.0, 2.0, 9.0],
        "f2": [0.0, 1.0, 8.0],
        "f3": [1.0, 1.0, 7.0],
        "next_return": [0.01, 0.02, 0.03],
    })
    graph = build_date_graph(df, feature_cols, k=2, min_nodes=1)
    assert torch.allclose(graph.p[2], graph.x[2])


def test_collate_keeps_dates_disjoint(synthetic_panel, feature_cols):
    cache = build_graph_cache(synthetic_panel, feature_cols, k=5, min_nodes=10, progress=False)
    graphs = [cache[d] for d in sorted(cache)[:3]]
    x, p, edge_index, label, date_idx = collate(graphs, torch.device("cpu"))

    assert x.shape[0] == sum(len(g) for g in graphs)
    assert label is not None
    # No edge may connect nodes belonging to two different dates.
    assert (date_idx[edge_index[0]] == date_idx[edge_index[1]]).all()
