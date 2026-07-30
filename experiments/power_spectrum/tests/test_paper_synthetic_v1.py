from __future__ import annotations

import json

import pytest
import torch

from experiments.power_spectrum.code import analyze_paper_synthetic_v1
from experiments.power_spectrum.code import run_paper_synthetic_v1 as runner


def test_focused_paper_grid_matches_expected_size_and_budget() -> None:
    config = runner.load_config(
        runner.POWER_ROOT / "configs" / "paper_synthetic_v1.json"
    )
    cells = runner.enumerate_cells(config)
    assert len(cells) == 30
    assert sum(cell["task"] == "denoising" for cell in cells) == 24
    assert sum(cell["task"] == "coupling" for cell in cells) == 6
    assert any(
        cell["task"] == "denoising"
        and cell["T"] == 2
        and cell["k_pos"] == 17
        for cell in cells
    )
    assert runner.build_plan(config)["within_cost_plan"]


def test_paper_generators_are_deterministic_and_np10_is_rank_one() -> None:
    first = runner.paper_markov_data(n_seqs=4, seed=0)
    second = runner.paper_markov_data(n_seqs=4, seed=0)
    assert torch.equal(first.x, second.x)
    assert first.x.shape == (4, 64, 40)
    assert first.support is not None
    assert first.hidden_support is not None
    assert not torch.equal(first.support, first.hidden_support)

    coupling = runner.paper_coupling_data(seed=0, n_seqs=4)
    assert coupling.x.shape == (4, 64, 256)
    assert coupling.hidden_features is not None
    assert torch.linalg.matrix_rank(
        coupling.hidden_features,
        tol=1e-6,
    ).item() == 1


def test_feature_recovery_recovers_exact_dictionary() -> None:
    directions = torch.eye(3)
    metrics = runner._feature_recovery_auc(directions, directions)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["mean_max_cos"] == pytest.approx(1.0)


def test_comparison_rejects_incomplete_spectral_summary() -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        analyze_paper_synthetic_v1.build_comparison(
            {"tasks": {}},
            {"complete": False},
        )


def test_config_is_json_serializable() -> None:
    config = runner.load_config(
        runner.POWER_ROOT / "configs" / "paper_synthetic_v1.json"
    )
    assert json.loads(json.dumps(config)) == config
