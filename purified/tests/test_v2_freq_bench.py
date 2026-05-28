"""FreqBench port — generator shapes + e2e order-control contract."""

from __future__ import annotations

import numpy as np
import pytest

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig
from temp_bench.data.freq_bench_data import (
    freq_bench_ac, freq_bench_dc, freq_bench_mixed,
)


@pytest.fixture(autouse=True)
def _allow_dirty(monkeypatch):
    monkeypatch.setenv("TEMP_BENCH_ALLOW_DIRTY", "1")


def test_generators_shapes_and_oracles():
    dc = freq_bench_dc(W=8, p=0.7, sigma=0.1, d_in=32, n_seqs=64, seed=0)
    assert dc.x.shape == (64, 8, 32) and dc.A_loc == 0.7
    assert dc.A_oracle > dc.A_loc                      # majority vote beats one token

    ac = freq_bench_ac(W=8, sigma=0.1, M=16, d_in=32, n_seqs=64, seed=0)
    assert ac.x.shape == (64, 8, 32)
    assert ac.A_loc == 0.5 and ac.A_oracle == 1.0
    assert set(np.unique(ac.y.numpy())) <= {0, 1}

    mx = freq_bench_mixed(W=8, sigma=0.1, M=64, n_classes=10, variant="unsigned",
                          d_in=128, n_seqs=64, seed=0)
    assert mx.n_classes == 10 and mx.velocities is not None
    assert abs(mx.A_loc - 0.1) < 1e-9


def test_per_token_arch_flat_on_ac():
    """Per-token SAE cannot encode signed direction: controls stay flat."""
    r = run_experiment(
        experiment="freq_bench", arch_name="topk_sae", seed=0,
        datasource_name="fb_ac_smoke",
        training_cfg=TrainingConfig(n_steps=30, batch_size=64, buffer_tokens=8192,
                                    warmup_steps=0),
        eval_cfg={"smoke": True},
    )
    m = r.row.metrics
    assert abs(m["order_gap"]) < 0.12        # A ≈ A_shuffle
    assert abs(m["reverse_drop"]) < 0.12     # reversing does not fool it


def test_window_arch_encodes_direction_on_ac():
    """T=2 crosscoder encodes the transition: order_gap>0, reverse below chance."""
    r = run_experiment(
        experiment="freq_bench", arch_name="txc_base", seed=0,
        datasource_name="fb_ac_smoke",
        training_cfg=TrainingConfig(n_steps=60, batch_size=64, buffer_tokens=8192,
                                    warmup_steps=0,
                                    arch_hparams_override={"T": 2, "k_pos": 1}),
        eval_cfg={"smoke": True, "T": 2, "k_pos": 1},
    )
    m = r.row.metrics
    assert m["A"] > 0.8                      # recovers the sign
    assert m["order_gap"] > 0.3              # shuffling destroys it
    assert m["A_reverse"] < 0.3              # reversing fools the forward probe
