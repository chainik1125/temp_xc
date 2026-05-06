"""Tests for evaluation metrics."""

import numpy as np
import torch
import pytest

from temporal_bench.metrics import (
    _auc_columns,
    compute_nmse,
    compute_l0,
    feature_recovery,
    feature_recovery_activation_global_local,
    feature_recovery_decoder_global_local,
)
from temporal_bench.config import DataConfig
from temporal_bench.data.pipeline import DataPipeline
from temporal_bench.sweep import _create_model


class TestNMSE:
    def test_perfect_reconstruction(self):
        x = torch.randn(10, 5, 40)
        assert compute_nmse(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_zero_reconstruction(self):
        x = torch.randn(10, 5, 40)
        x_hat = torch.zeros_like(x)
        assert compute_nmse(x, x_hat) == pytest.approx(1.0, rel=0.1)

    def test_positive(self):
        x = torch.randn(10, 5, 40)
        x_hat = x + 0.1 * torch.randn_like(x)
        assert compute_nmse(x, x_hat) > 0


class TestL0:
    def test_all_zero(self):
        latents = torch.zeros(4, 5, 20)
        assert compute_l0(latents) == 0.0

    def test_all_nonzero(self):
        latents = torch.ones(4, 5, 20)
        assert compute_l0(latents) == 20.0

    def test_sparse(self):
        latents = torch.zeros(4, 5, 20)
        latents[:, :, :3] = 1.0  # 3 nonzero per token
        assert compute_l0(latents) == pytest.approx(3.0)


class TestFeatureRecovery:
    def test_perfect_recovery(self):
        """Decoder columns = true features -> AUC should be 1.0."""
        true_features = torch.eye(10, 40)[:10]  # (10, 40)
        decoder_dirs = true_features.T  # (40, 10)
        result = feature_recovery(decoder_dirs, true_features)
        assert result["auc"] == pytest.approx(1.0, abs=0.02)
        assert result["r_at_90"] == 1.0
        assert result["mean_max_cos"] == pytest.approx(1.0, abs=1e-5)

    def test_random_recovery(self):
        """Random decoder should have low AUC."""
        torch.manual_seed(42)
        true_features = torch.randn(10, 40)
        decoder_dirs = torch.randn(40, 10)
        result = feature_recovery(decoder_dirs, true_features)
        assert result["auc"] < 0.5
        assert result["r_at_90"] < 0.3

    def test_partial_recovery(self):
        """Some features aligned, some not."""
        true_features = torch.eye(10, 40)[:10]
        decoder_dirs = torch.zeros(40, 10)
        # First 5 decoder columns match first 5 features
        for i in range(5):
            decoder_dirs[i, i] = 1.0
        # Last 5 are random
        decoder_dirs[:, 5:] = torch.randn(40, 5)
        result = feature_recovery(decoder_dirs, true_features)
        assert result["r_at_90"] == pytest.approx(0.5, abs=0.1)


class TestAucColumns:
    def test_perfect_separator(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]])
        assert _auc_columns(y, scores)[0] == pytest.approx(1.0)

    def test_anti_separator(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([[0.9], [0.8], [0.7], [0.3], [0.2], [0.1]])
        assert _auc_columns(y, scores)[0] == pytest.approx(0.0)

    def test_single_class_returns_nan(self):
        y = np.array([1, 1, 1, 1])
        scores = np.random.randn(4, 3)
        assert np.isnan(_auc_columns(y, scores)).all()


class TestGlobalLocalRecovery:
    """Smoke tests against an untrained model — the metric pipeline must run
    end-to-end and produce values in plausible ranges. Numerical magnitudes
    on untrained models are not asserted."""

    @pytest.fixture(scope="class")
    def setup(self):
        torch.manual_seed(0)
        data_cfg = DataConfig(n_features=12, d_model=24, pi=0.1, seed=0)
        pipe = DataPipeline(data_cfg, device=torch.device("cpu"))
        x, s, _h = pipe.eval_data_with_support(n_sequences=20, T=4, rho=0.6, seed=1)
        return pipe, x, s

    @pytest.mark.parametrize("model_name", ["regular_sae", "regular_sae_kT"])
    def test_decoder_local_equals_global_for_position_independent(self, setup, model_name):
        pipe, _x, _s = setup
        model = _create_model(model_name, d_in=24, d_sae=12, T=4, k=2, device=torch.device("cpu"))
        out = feature_recovery_decoder_global_local(model, pipe.true_features)
        assert out["auc_local"] == pytest.approx(out["auc_global"])

    @pytest.mark.parametrize("model_name", ["txcdr", "stacked_sae"])
    def test_decoder_local_global_well_formed_for_temporal(self, setup, model_name):
        pipe, _x, _s = setup
        model = _create_model(model_name, d_in=24, d_sae=12, T=4, k=2, device=torch.device("cpu"))
        out = feature_recovery_decoder_global_local(model, pipe.true_features)
        # AUCs are in [0, 1].
        for key in ("auc_local", "auc_global"):
            assert 0.0 <= out[key] <= 1.0

    @pytest.mark.parametrize(
        "model_name", ["regular_sae", "regular_sae_kT", "txcdr", "stacked_sae"]
    )
    def test_activation_metrics_in_unit_interval(self, setup, model_name):
        pipe, x, s = setup
        model = _create_model(model_name, d_in=24, d_sae=12, T=4, k=2, device=torch.device("cpu"))
        out = feature_recovery_activation_global_local(model, x, s)
        # Best-of-orientation pick guarantees AUC >= 0.5.
        for key in ("auc_local", "auc_global"):
            assert 0.5 <= out[key] <= 1.0


class TestCoupledMode:
    """Smoke tests for coupled-features data + gAUC metric."""

    def test_coupling_matrix_has_exactly_n_parents_per_row(self):
        from temporal_bench.data.coupled import generate_coupling_matrix

        rng = torch.Generator().manual_seed(0)
        C = generate_coupling_matrix(K=10, M=20, n_parents=3, generator=rng)
        assert C.shape == (20, 10)
        assert (C.sum(dim=1) == 3).all()

    def test_apply_coupling_or_matches_oracle_for_known_chain(self):
        from temporal_bench.data.coupled import apply_coupling_or

        # Two emissions, three hidden chains. Em 0 parented by {0,1}; em 1 by {2}.
        C = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        h = torch.tensor([[[1, 0], [0, 0], [0, 1]]], dtype=torch.float32)  # (1, 3, 2)
        s = apply_coupling_or(h, C)
        # Em 0 fires when h0 OR h1 is on -> (t=0: yes, t=1: no)
        # Em 1 fires when h2 is on -> (t=0: no, t=1: yes)
        expected = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
        assert torch.equal(s, expected)

    def test_compute_hidden_features_unit_norm(self):
        from temporal_bench.data.coupled import (
            compute_hidden_features,
            generate_coupling_matrix,
        )

        rng = torch.Generator().manual_seed(0)
        C = generate_coupling_matrix(K=8, M=16, n_parents=2, generator=rng)
        emission_features = torch.randn(16, 32, generator=rng)
        hidden_features = compute_hidden_features(emission_features, C)
        assert hidden_features.shape == (8, 32)
        norms = hidden_features.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_pipeline_end_to_end_coupled_mode(self):
        from temporal_bench.config import DataConfig
        from temporal_bench.data.pipeline import DataPipeline

        cfg = DataConfig(n_features=12, d_model=24, pi=0.15, n_hidden=6, n_parents=2, seed=0)
        pipe = DataPipeline(cfg, device=torch.device("cpu"))
        assert pipe.is_coupled
        assert pipe.true_features.shape == (12, 24)
        assert pipe.hidden_features.shape == (6, 24)
        x, s, h = pipe.eval_data_with_support(n_sequences=10, T=3, rho=0.5, seed=1)
        assert x.shape == (10, 3, 24)
        assert s.shape == (10, 12, 3)  # M emissions
        assert h.shape == (10, 6, 3)  # K hidden chains
        # Emissions are derived from h by OR-gate; whenever any parent fires,
        # the emission must fire too.
        from temporal_bench.data.coupled import apply_coupling_or

        s_recomputed = apply_coupling_or(h, pipe.coupling_matrix)
        assert torch.equal(s, s_recomputed)

    def test_evaluate_populates_gauc_when_hidden_features_supplied(self):
        from temporal_bench.config import DataConfig
        from temporal_bench.data.pipeline import DataPipeline
        from temporal_bench.metrics import evaluate

        cfg = DataConfig(n_features=12, d_model=24, pi=0.15, n_hidden=6, n_parents=2, seed=0)
        pipe = DataPipeline(cfg, device=torch.device("cpu"))
        x, s, _h = pipe.eval_data_with_support(n_sequences=10, T=3, rho=0.5, seed=1)
        model = _create_model("txcdr", d_in=24, d_sae=12, T=3, k=2, device=torch.device("cpu"))
        em = evaluate(model, x, pipe.true_features, eval_s=s, hidden_features=pipe.hidden_features)
        assert 0.0 <= em.auc <= 1.0
        assert 0.0 <= em.auc_hidden <= 1.0
        # Untrained model has random decoders; both AUCs should be small.
        assert em.auc_hidden < 0.7
        # Without hidden_features, gAUC stays NaN.
        em2 = evaluate(model, x, pipe.true_features)
        import math
        assert math.isnan(em2.auc_hidden)
