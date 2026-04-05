"""Tests for the data pipeline and evaluation module."""

import pytest
import torch

from src.bench.config import DataConfig, ToyModelConfig, MarkovConfig
from src.bench.data import build_data_pipeline
from src.bench.eval import evaluate_model, feature_recovery_auc
from src.bench.architectures.topk_sae import TopKSAESpec
from src.bench.architectures.crosscoder import CrosscoderSpec

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


class TestDataPipeline:
    def test_build_pipeline(self, small_config):
        pipeline = build_data_pipeline(small_config, DEVICE, window_sizes=[2])
        assert pipeline.true_features.shape == (32, 16)
        assert pipeline.eval_hidden.shape == (10, 16, 32)
        assert pipeline.scaling_factor > 0

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


class TestFeatureRecovery:
    def test_perfect_recovery(self):
        # If decoder == true features, AUC should be ~1
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
