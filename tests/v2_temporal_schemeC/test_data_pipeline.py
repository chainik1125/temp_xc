"""Tests for data_pipeline.py — verify equivalence with old copy-pasted functions."""

import math

import pytest
import torch

from src.utils.seed import set_seed
from src.data.toy.toy_model import ToyModel
from src.data.toy.markov import generate_markov_activations

# Will be implemented:
from src.pipeline.toy_data import (
    DataConfig,
    DataPipeline,
    build_data_pipeline,
    compute_scaling_factor,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiny config for fast tests
TINY_CFG = DataConfig(
    num_features=4,
    hidden_dim=8,
    seq_len=16,
    pi=[0.5] * 4,
    rho=[0.0, 0.3, 0.7, 0.9],
    dict_width=8,
    seed=42,
    eval_n_seq=10,
)


# ── Old code (copy-pasted from run_auc_and_crosscoder.py) for reference ──


def _old_compute_scaling_factor(model, pi_t, rho_t, device, hidden_dim, seq_len):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, seq_len, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(hidden_dim) / hidden.reshape(-1, hidden_dim).norm(dim=-1).mean().item()


def _old_make_flat_gen(model, pi_t, rho_t, device, sf, hidden_dim, seq_len):
    def gen(batch_size):
        n_seq = max(1, batch_size // seq_len)
        acts, _ = generate_markov_activations(n_seq, seq_len, pi_t, rho_t, device=device)
        return (model(acts) * sf).reshape(-1, hidden_dim)[:batch_size]
    return gen


def _old_make_seq_gen(model, pi_t, rho_t, device, sf, seq_len, shuffle=False):
    def gen(n_seq):
        acts, _ = generate_markov_activations(n_seq, seq_len, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                acts[i] = acts[i, torch.randperm(seq_len, device=device)]
        return model(acts) * sf
    return gen


def _old_make_window_gen(model, pi_t, rho_t, device, sf, seq_len, T):
    def gen(batch_size):
        n_seq = max(1, batch_size // (seq_len - T + 1)) + 1
        acts, _ = generate_markov_activations(n_seq, seq_len, pi_t, rho_t, device=device)
        hidden = model(acts) * sf
        windows = []
        for t in range(seq_len - T + 1):
            windows.append(hidden[:, t:t + T, :])
        all_w = torch.cat(windows, dim=0)
        idx = torch.randperm(all_w.shape[0], device=device)[:batch_size]
        return all_w[idx]
    return gen


# ── Tests ────────────────────────────────────────────────────────────


class TestComputeScalingFactor:
    def test_matches_old_implementation(self):
        set_seed(42)
        model = ToyModel(num_features=4, hidden_dim=8).to(DEVICE)
        model.eval()
        pi_t = torch.tensor(TINY_CFG.pi)
        rho_t = torch.tensor(TINY_CFG.rho)

        set_seed(99)
        old_sf = _old_compute_scaling_factor(model, pi_t, rho_t, DEVICE, 8, 16)
        set_seed(99)
        new_sf = compute_scaling_factor(model, TINY_CFG, DEVICE)

        assert abs(old_sf - new_sf) < 1e-10, f"old={old_sf}, new={new_sf}"

    def test_returns_positive_float(self):
        set_seed(42)
        model = ToyModel(num_features=4, hidden_dim=8).to(DEVICE)
        model.eval()
        sf = compute_scaling_factor(model, TINY_CFG, DEVICE)
        assert isinstance(sf, float)
        assert sf > 0


class TestBuildDataPipeline:
    def test_builds_without_error(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        assert isinstance(pipeline, DataPipeline)

    def test_eval_hidden_shape(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        assert pipeline.eval_hidden.shape == (10, 16, 8)

    def test_true_features_shape(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        assert pipeline.true_features.shape[0] == 4  # num_features
        assert pipeline.true_features.shape[1] == 8  # hidden_dim

    def test_flat_gen_shape(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        batch = pipeline.gen_flat(32)
        assert batch.shape == (32, 8)

    def test_seq_gen_shape(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        batch = pipeline.gen_seq(5)
        assert batch.shape == (5, 16, 8)

    def test_seq_shuffled_gen_shape(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        batch = pipeline.gen_seq_shuffled(5)
        assert batch.shape == (5, 16, 8)

    def test_window_gen_shape(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE, window_sizes=[2, 5])
        batch_t2 = pipeline.gen_windows[2](32)
        batch_t5 = pipeline.gen_windows[5](32)
        assert batch_t2.shape == (32, 2, 8)
        assert batch_t5.shape == (32, 5, 8)

    def test_flat_gen_deterministic(self):
        p1 = build_data_pipeline(TINY_CFG, DEVICE)
        p2 = build_data_pipeline(TINY_CFG, DEVICE)
        set_seed(77)
        b1 = p1.gen_flat(16)
        set_seed(77)
        b2 = p2.gen_flat(16)
        assert torch.allclose(b1, b2), "Same seed should give same output"

    def test_shuffled_gen_destroys_temporal_order(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        set_seed(77)
        temporal = pipeline.gen_seq(5)
        set_seed(77)
        shuffled = pipeline.gen_seq_shuffled(5)
        # Same marginal stats but different order — at least one position should differ
        assert not torch.allclose(temporal, shuffled), "Shuffled should differ from temporal"

    def test_scaling_factor_stored(self):
        pipeline = build_data_pipeline(TINY_CFG, DEVICE)
        assert isinstance(pipeline.scaling_factor, float)
        assert pipeline.scaling_factor > 0
