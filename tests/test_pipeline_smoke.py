"""End-to-end smoke test on synthetic data.

Runs the real training loop, the real graph builder and the real evaluator on
a planted signal. It is the check that the pieces actually compose -- and
that the model can recover a signal it is given.
"""
import numpy as np
import pytest

from iagnn.config import GraphConfig, LossConfig, PipelineConfig, TrainConfig, DataConfig
from iagnn.evaluate import evaluate
from iagnn.graph import build_graph_cache
from iagnn.trainer import predict, train


@pytest.fixture
def small_cfg(tmp_path, feature_cols):
    data = DataConfig(
        feature_panel_path=tmp_path / "unused.parquet",
        factor_dir=tmp_path,
        industry_map_path=tmp_path / "unused.csv",
        train_start="2020-01-01",
        train_end="2020-01-10",
        test_start="2020-01-11",
        test_end="2020-01-20",
    )
    return PipelineConfig(
        data=data,
        feature_cols=list(feature_cols),
        graph=GraphConfig(k_neighbors=5, min_eligible_per_date=50),
        loss=LossConfig(min_score_std=0.1),
        train=TrainConfig(max_epochs=6, batch_dates=4, device="cpu", seed=0),
        factor_name="TestFactor",
        output_dir=tmp_path / "artifacts",
    )


def test_model_recovers_a_planted_signal(synthetic_panel, small_cfg):
    graphs = build_graph_cache(
        synthetic_panel,
        small_cfg.feature_cols,
        k=small_cfg.graph.k_neighbors,
        min_nodes=small_cfg.graph.min_eligible_per_date,
        progress=False,
    )
    assert len(graphs) == synthetic_panel["TradingDay"].nunique()

    model, device, history = train(graphs, small_cfg, verbose=False)
    assert len(history.loss) >= 1
    assert history.ic[-1] > history.ic[0], "training IC should improve over the first epochs"

    predictions = predict(model, device, graphs, small_cfg, verbose=False)
    assert len(predictions) == len(synthetic_panel)
    assert predictions[small_cfg.factor_name].notna().all()

    scored = synthetic_panel[["TradingDay", "SecuCode", "next_return"]].merge(
        predictions, on=["TradingDay", "SecuCode"]
    )
    metrics = evaluate(scored, small_cfg.factor_name)["metrics"]
    assert metrics["ic_mean"] > 0.15, f"planted signal not recovered: {metrics['ic_mean']}"


def test_training_is_reproducible_under_a_fixed_seed(synthetic_panel, small_cfg):
    graphs = build_graph_cache(
        synthetic_panel, small_cfg.feature_cols,
        k=5, min_nodes=50, progress=False,
    )
    first, device, _ = train(graphs, small_cfg, verbose=False)
    second, _, _ = train(graphs, small_cfg, verbose=False)

    p1 = predict(first, device, graphs, small_cfg, verbose=False)
    p2 = predict(second, device, graphs, small_cfg, verbose=False)
    np.testing.assert_allclose(
        p1[small_cfg.factor_name].to_numpy(), p2[small_cfg.factor_name].to_numpy(), rtol=1e-5
    )


def test_config_manifest_round_trips(small_cfg, tmp_path):
    import json

    path = small_cfg.write_manifest(tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["factor_name"] == "TestFactor"
    assert payload["graph"]["k_neighbors"] == 5
    assert payload["feature_cols"] == small_cfg.feature_cols
