"""Tests for the data pipeline and evaluation module."""

import pytest
import torch

from src.training.config import (
    CouplingConfig,
    DataConfig,
    MarkovConfig,
    ToyModelConfig,
)
from src.data.nlp.loader import build_data_pipeline
from src.eval.runner import evaluate_model, feature_recovery_auc
from src.architectures.topk_sae import TopKSAESpec
from src.architectures.crosscoder import CrosscoderSpec

DEVICE = torch.device("cpu")


@pytest.fixture
def small_config():
    return DataConfig(
        toy_model=ToyModelConfig(num_features=16, hidden_dim=32),
        markov=MarkovConfig(pi=0.1, rho=0.0),
        seq_len=16,
        d_sae=16,
        seed=42,
        eval_n_seq=10,
    )


@pytest.fixture
def leaky_config():
    return DataConfig(
        toy_model=ToyModelConfig(num_features=16, hidden_dim=32),
        markov=MarkovConfig(pi=0.1, rho=0.6, delta=0.5),
        seq_len=16,
        d_sae=16,
        seed=42,
        eval_n_seq=10,
    )


@pytest.fixture
def coupled_config():
    return DataConfig(
        toy_model=ToyModelConfig(num_features=16, hidden_dim=32),
        markov=MarkovConfig(pi=0.1, rho=0.6),
        coupling=CouplingConfig(K_hidden=5, M_emission=10, n_parents=2),
        seq_len=16,
        d_sae=10,
        seed=42,
        eval_n_seq=10,
    )


class TestDataPipeline:
    def test_build_pipeline(self, small_config):
        pipeline = build_data_pipeline(small_config, DEVICE, window_sizes=[2])
        assert pipeline.true_features.shape == (32, 16)
        assert pipeline.eval_hidden.shape == (10, 16, 32)
        assert pipeline.scaling_factor > 0
        assert pipeline.global_features is None

    def test_gen_flat(self, small_config):
        pipeline = build_data_pipeline(small_config, DEVICE)
        x = pipeline.gen_flat(64)
        assert x.shape == (64, 32)

    def test_gen_seq(self, small_config):
        pipeline = build_data_pipeline(small_config, DEVICE)
        x = pipeline.gen_seq(8)
        assert x.shape == (8, 16, 32)

    def test_gen_window(self, small_config):
        pipeline = build_data_pipeline(small_config, DEVICE, window_sizes=[2, 5])
        w2 = pipeline.gen_windows[2](32)
        assert w2.shape[1] == 2
        assert w2.shape[2] == 32
        w5 = pipeline.gen_windows[5](32)
        assert w5.shape[1] == 5


class TestLeakyResetPipeline:
    def test_build_pipeline(self, leaky_config):
        pipeline = build_data_pipeline(leaky_config, DEVICE)
        assert pipeline.true_features.shape == (32, 16)
        assert pipeline.scaling_factor > 0
        assert pipeline.global_features is None

    def test_gen_flat(self, leaky_config):
        pipeline = build_data_pipeline(leaky_config, DEVICE)
        x = pipeline.gen_flat(64)
        assert x.shape == (64, 32)
        assert not x.isnan().any()

    def test_rho_eff_increases_with_delta(self):
        m0 = MarkovConfig(pi=0.1, rho=0.6, delta=0.0)
        m1 = MarkovConfig(pi=0.1, rho=0.6, delta=0.5)
        m2 = MarkovConfig(pi=0.1, rho=0.6, delta=0.75)
        assert m0.rho_eff < m1.rho_eff < m2.rho_eff


class TestCoupledPipeline:
    def test_build_pipeline(self, coupled_config):
        pipeline = build_data_pipeline(coupled_config, DEVICE, window_sizes=[2])
        # true_features are emission features (M=10)
        assert pipeline.true_features.shape == (32, 10)
        # global_features are hidden features (K=5)
        assert pipeline.global_features is not None
        assert pipeline.global_features.shape == (32, 5)
        assert pipeline.eval_hidden.shape == (10, 16, 32)

    def test_gen_flat(self, coupled_config):
        pipeline = build_data_pipeline(coupled_config, DEVICE)
        x = pipeline.gen_flat(64)
        assert x.shape == (64, 32)

    def test_gen_seq(self, coupled_config):
        pipeline = build_data_pipeline(coupled_config, DEVICE)
        x = pipeline.gen_seq(8)
        assert x.shape == (8, 16, 32)

    def test_gen_window(self, coupled_config):
        pipeline = build_data_pipeline(coupled_config, DEVICE, window_sizes=[2])
        w = pipeline.gen_windows[2](32)
        assert w.shape[1] == 2
        assert w.shape[2] == 32


class TestFeatureRecovery:
    def test_perfect_recovery(self):
        features = torch.randn(32, 16)
        features = features / features.norm(dim=0, keepdim=True)
        result = feature_recovery_auc(features, features)
        assert result["auc"] > 0.99
        assert result["frac_recovered_90"] > 0.99

    def test_random_decoder(self):
        features = torch.randn(32, 16)
        features = features / features.norm(dim=0, keepdim=True)
        random_dec = torch.randn(32, 16)
        result = feature_recovery_auc(random_dec, features)
        assert result["auc"] < 0.5


class TestEvaluateModel:
    def test_eval_topk_sae(self, small_config):
        pipeline = build_data_pipeline(small_config, DEVICE)
        spec = TopKSAESpec()
        model = spec.create(32, 16, k=4, device=DEVICE)
        result = evaluate_model(
            spec, model, pipeline.eval_hidden, DEVICE,
            true_features=pipeline.true_features,
            seq_len=small_config.seq_len,
        )
        assert result.nmse > 0
        assert result.auc is not None
        assert 0 <= result.auc <= 1
        assert result.global_auc is None

    def test_eval_crosscoder(self, small_config):
        pipeline = build_data_pipeline(small_config, DEVICE, window_sizes=[2])
        spec = CrosscoderSpec(T=2)
        model = spec.create(32, 16, k=4, device=DEVICE)
        result = evaluate_model(
            spec, model, pipeline.eval_hidden, DEVICE,
            true_features=pipeline.true_features,
            seq_len=small_config.seq_len,
        )
        assert result.nmse > 0
        assert result.auc is not None

    def test_eval_dual_auc_coupled(self, coupled_config):
        pipeline = build_data_pipeline(coupled_config, DEVICE)
        spec = TopKSAESpec()
        model = spec.create(32, 10, k=4, device=DEVICE)
        result = evaluate_model(
            spec, model, pipeline.eval_hidden, DEVICE,
            true_features=pipeline.true_features,
            global_features=pipeline.global_features,
            seq_len=coupled_config.seq_len,
        )
        assert result.nmse > 0
        assert result.auc is not None
        assert result.global_auc is not None
        assert 0 <= result.auc <= 1
        assert 0 <= result.global_auc <= 1

    def test_eval_leaky_reset(self, leaky_config):
        pipeline = build_data_pipeline(leaky_config, DEVICE)
        spec = TopKSAESpec()
        model = spec.create(32, 16, k=4, device=DEVICE)
        result = evaluate_model(
            spec, model, pipeline.eval_hidden, DEVICE,
            true_features=pipeline.true_features,
            seq_len=leaky_config.seq_len,
        )
        assert result.nmse > 0
        assert result.auc is not None
