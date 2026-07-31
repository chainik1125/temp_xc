from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from experiments.power_spectrum.backtracking_sae_pooling.steering_baselines import (
    run_baselines,
    summarize,
)


def test_successful_judge_keys_deduplicates_retries() -> None:
    rows = [
        {"transcript_id": "q1", "magnitude": -1, "arch": "topk_sae", "seed": 42, "label": -1},
        {"transcript_id": "q1", "magnitude": -1, "arch": "topk_sae", "seed": 42, "label": 1},
        {"transcript_id": "q1", "magnitude": -1, "arch": "topk_sae", "seed": 42, "label": 1},
        {"transcript_id": "q2", "magnitude": 0, "arch": "txc_base", "seed": 42, "label": 0},
    ]
    assert run_baselines.successful_judge_keys(rows, arch="topk_sae", seed=42) == {
        ("q1", -1.0, "topk_sae", 42)
    }


def test_atomic_write_json_replaces_complete_file(tmp_path) -> None:
    target = tmp_path / "result.json"
    run_baselines.atomic_write_json(target, {"complete": True, "value": 3})
    assert json.loads(target.read_text()) == {"complete": True, "value": 3}
    assert not target.with_suffix(".json.tmp").exists()


def test_metric_curve_uses_signed_canonical_keys() -> None:
    metrics = {
        "delta_gc_mag_-1.0": 0.25,
        "delta_gc_mag_+0.0": 0.0,
        "delta_gc_mag_+1.0": -0.5,
    }
    assert summarize.metric_curve(metrics, [-1, 0, 1]) == [0.25, 0.0, -0.5]


@dataclass
class _DummyConfig:
    name: str = "topk_sae"
    T: int = 1
    d_sae: int = 2


class _DummySAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = _DummyConfig()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def decoder_directions(self) -> torch.Tensor:
        return torch.eye(2)


def test_pooled_sae_adapter_returns_one_aligned_window_code() -> None:
    x = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]])
    mean = run_baselines.PooledSAEAdapter(_DummySAE(), pool="mean", window=2)
    maximum = run_baselines.PooledSAEAdapter(_DummySAE(), pool="max", window=2)

    assert mean.config.T == 2
    assert mean.config.name == "pooled_sae_mean"
    assert torch.equal(mean.encode(x), torch.tensor([[[2.0, 3.0]]]))
    assert torch.equal(maximum.encode(x), torch.tensor([[[3.0, 4.0]]]))
    assert torch.equal(maximum.decoder_directions(), torch.eye(2))
