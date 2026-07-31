from __future__ import annotations

import torch

from experiments.power_spectrum.trajectory_bottleneck.model import (
    TrajectoryBottleneck,
)


def _model(*, rank: int = 0) -> TrajectoryBottleneck:
    generator = torch.Generator().manual_seed(7)
    decoder = torch.randn(12, 8, generator=generator)
    return TrajectoryBottleneck(
        base_decoder=decoder,
        base_decoder_bias=torch.zeros(8),
        window=3,
        k_window=4,
        rank=rank,
        dead_threshold_tokens=3,
        aux_k=4,
    )


def test_sparse_trajectory_shapes_and_gradients() -> None:
    model = _model(rank=3)
    indices = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]
    )
    values = torch.ones_like(indices, dtype=torch.float32)
    target = torch.randn(2, 3, 8)
    result = model.loss(indices, values, target)
    assert result["loss"].ndim == 0
    assert 0 <= result["l0"].item() <= 4
    result["loss"].backward()
    assert model.temporal_encoder.grad is not None
    assert model.temporal_decoder.grad is not None
    assert model.cross_out.grad is not None


def test_dead_tracker_marks_features_that_really_fired() -> None:
    model = _model()
    with torch.no_grad():
        model.code_bias.fill_(-10)
        model.code_bias[3] = 10
    indices = torch.zeros((2, 3, 2), dtype=torch.long)
    values = torch.ones_like(indices, dtype=torch.float32)
    target = torch.zeros((2, 3, 8))
    model.loss(indices, values, target)
    assert model.num_tokens_since_fired[3].item() == 0
    other = torch.cat(
        (
            model.num_tokens_since_fired[:3],
            model.num_tokens_since_fired[4:],
        )
    )
    assert torch.all(other == 6)


def test_decoder_temporal_profiles_stay_normalized() -> None:
    model = _model()
    with torch.no_grad():
        model.temporal_decoder.mul_(4)
    model.normalize_decoder_profiles()
    norms = model.temporal_decoder.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
