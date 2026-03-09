"""Unit tests for the data generation pipeline."""

import pytest
import torch

from src.data_generation.configs import (
    DataGenerationConfig,
    FeatureConfig,
    MagnitudeConfig,
    SequenceConfig,
    TransitionConfig,
)
from src.data_generation.dataset import generate_dataset
from src.data_generation.transition import (
    build_transition_matrix,
    expected_holding_times,
    stationary_distribution,
    theoretical_autocorrelation,
    validate_transition_matrix,
)


# ============================================================================
# TransitionConfig tests
# ============================================================================


class TestTransitionConfig:
    def test_from_reset_process_iid(self):
        """lam=1.0 gives i.i.d. Bernoulli (reset matrix)."""
        cfg = TransitionConfig.from_reset_process(lam=1.0, p=0.05)
        # P(off -> on) = p, P(on -> on) = p (every row is [1-p, p])
        assert torch.allclose(cfg.matrix[0, 1], torch.tensor(0.05), atol=1e-6)
        assert torch.allclose(cfg.matrix[1, 1], torch.tensor(0.05), atol=1e-6)

    def test_from_reset_process_perfect_memory(self):
        """lam=0.0 gives identity matrix (perfect memory)."""
        cfg = TransitionConfig.from_reset_process(lam=0.0, p=0.05)
        assert torch.allclose(cfg.matrix, torch.eye(2), atol=1e-6)

    def test_arbitrary_valid_matrix(self):
        """Arbitrary valid transition matrix is accepted."""
        P = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        cfg = TransitionConfig(matrix=P, stationary_on_prob=0.3 / (0.3 + 0.4))
        assert torch.allclose(cfg.matrix, P)

    def test_negative_entries_rejected(self):
        """Matrix with negative entries raises ValueError."""
        P = torch.tensor([[1.1, -0.1], [0.4, 0.6]])
        with pytest.raises(ValueError, match="negative"):
            TransitionConfig(matrix=P)

    def test_rows_not_summing_to_one_rejected(self):
        """Matrix with rows not summing to 1 raises ValueError."""
        P = torch.tensor([[0.5, 0.3], [0.4, 0.6]])
        with pytest.raises(ValueError, match="sum to 1"):
            TransitionConfig(matrix=P)

    def test_wrong_shape_rejected(self):
        """Non-2x2 matrix raises ValueError."""
        P = torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        with pytest.raises(ValueError, match="2x2"):
            TransitionConfig(matrix=P)


# ============================================================================
# Transition utility tests
# ============================================================================


class TestTransitionUtils:
    def test_build_transition_matrix_shape(self):
        P = build_transition_matrix(0.5, 0.05)
        assert P.shape == (2, 2)
        assert torch.allclose(P.sum(dim=1), torch.ones(2), atol=1e-6)

    def test_stationary_distribution_reset(self):
        """Stationary distribution of reset process is [1-p, p]."""
        P = build_transition_matrix(0.5, 0.1)
        pi = stationary_distribution(P)
        assert torch.allclose(pi, torch.tensor([0.9, 0.1]), atol=1e-5)

    def test_stationary_distribution_arbitrary(self):
        P = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        pi = stationary_distribution(P)
        # pi_on = 0.3 / (0.3 + 0.4) = 3/7
        assert abs(pi[1].item() - 3 / 7) < 1e-5

    def test_theoretical_autocorrelation_iid(self):
        """lam=1.0 -> zero autocorrelation at all lags > 0."""
        P = build_transition_matrix(1.0, 0.05)
        acf = theoretical_autocorrelation(P, 10)
        assert abs(acf[0].item() - 1.0) < 1e-6
        assert all(abs(acf[i].item()) < 1e-6 for i in range(1, 11))

    def test_theoretical_autocorrelation_perfect_memory(self):
        """lam=0.0 -> autocorrelation = 1 at all lags."""
        P = build_transition_matrix(0.0, 0.05)
        acf = theoretical_autocorrelation(P, 10)
        assert torch.allclose(acf, torch.ones(11), atol=1e-6)

    def test_expected_holding_times(self):
        P = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        times = expected_holding_times(P)
        assert abs(times["off"] - 5.0) < 1e-6  # 1 / 0.2
        assert abs(times["on"] - 1.0 / 0.3) < 1e-4

    def test_validate_transition_matrix_valid(self):
        validate_transition_matrix(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))

    def test_validate_transition_matrix_invalid(self):
        with pytest.raises(ValueError):
            validate_transition_matrix(torch.tensor([[0.5, 0.3], [0.4, 0.6]]))


# ============================================================================
# Output shape tests
# ============================================================================


class TestOutputShapes:
    def test_generate_dataset_shapes(self):
        """All output tensors have the correct shapes."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.5, p=0.05),
            features=FeatureConfig(k=8, d=32),
            sequence=SequenceConfig(T=64, n_sequences=3),
        )
        result = generate_dataset(cfg)

        assert result["features"].shape == (8, 32)
        assert result["support"].shape == (3, 8, 64)
        assert result["magnitudes"].shape == (3, 8, 64)
        assert result["activations"].shape == (3, 8, 64)
        assert result["x"].shape == (3, 64, 32)

    def test_support_is_binary(self):
        """Support tensor contains only 0s and 1s."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
            features=FeatureConfig(k=5, d=16),
            sequence=SequenceConfig(T=32, n_sequences=2),
        )
        result = generate_dataset(cfg)
        unique_vals = result["support"].unique()
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())

    def test_magnitudes_nonnegative(self):
        """Magnitude tensor is non-negative."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
            features=FeatureConfig(k=5, d=16),
            sequence=SequenceConfig(T=32, n_sequences=2),
        )
        result = generate_dataset(cfg)
        assert (result["magnitudes"] >= 0).all()


# ============================================================================
# Correctness tests
# ============================================================================


class TestCorrectness:
    def test_x_is_linear_combination_of_features(self):
        """x_t = sum_i a_{i,t} * f_i for all t and sequences."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
            features=FeatureConfig(k=5, d=16),
            sequence=SequenceConfig(T=32, n_sequences=3),
        )
        result = generate_dataset(cfg)
        features = result["features"]  # (k, d)
        activations = result["activations"]  # (n_seq, k, T)
        x = result["x"]  # (n_seq, T, d)

        for seq_idx in range(3):
            expected_x = activations[seq_idx].T @ features  # (T, d)
            assert torch.allclose(x[seq_idx], expected_x, atol=1e-5)

    def test_activations_equal_support_times_magnitudes(self):
        """a_{i,t} = s_{i,t} * m_{i,t}."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
            features=FeatureConfig(k=5, d=16),
            sequence=SequenceConfig(T=32, n_sequences=2),
        )
        result = generate_dataset(cfg)
        expected = result["support"] * result["magnitudes"]
        assert torch.allclose(result["activations"], expected)

    def test_stationary_probability_empirical(self):
        """Empirical firing probability matches p within tolerance."""
        p = 0.1
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.5, p=p),
            features=FeatureConfig(k=20, d=8),
            sequence=SequenceConfig(T=500, n_sequences=50),
        )
        result = generate_dataset(cfg)
        empirical_p = result["support"].mean().item()
        assert abs(empirical_p - p) < 0.02, f"Empirical p={empirical_p}, expected ~{p}"

    def test_iid_has_no_autocorrelation(self):
        """lam=1.0 produces empirically uncorrelated support."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=1.0, p=0.1),
            features=FeatureConfig(k=1, d=4),
            sequence=SequenceConfig(T=1000, n_sequences=100),
        )
        result = generate_dataset(cfg)
        support = result["support"][:, 0, :]  # (100, 1000)
        mean_s = support.mean()
        var_s = support.var()
        # Lag-1 autocorrelation
        cov = ((support[:, :-1] - mean_s) * (support[:, 1:] - mean_s)).mean()
        acf_1 = (cov / var_s).item()
        assert abs(acf_1) < 0.05, f"Lag-1 autocorrelation = {acf_1}, expected ~0"

    def test_perfect_memory_preserves_initial_state(self):
        """lam=0.0 means state never changes after initialization."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.0, p=0.5),
            features=FeatureConfig(k=10, d=8),
            sequence=SequenceConfig(T=100, n_sequences=1),
        )
        result = generate_dataset(cfg)
        support = result["support"][0]  # (10, 100)
        # Every feature should have the same value at all timesteps
        for i in range(10):
            assert (support[i] == support[i, 0]).all()

    def test_deterministic_with_same_seed(self):
        """Same config and seed produce identical results."""
        cfg = DataGenerationConfig(
            transition=TransitionConfig.from_reset_process(lam=0.5, p=0.1),
            features=FeatureConfig(k=5, d=16),
            sequence=SequenceConfig(T=32, n_sequences=2),
            seed=123,
        )
        r1 = generate_dataset(cfg)
        r2 = generate_dataset(cfg)
        assert torch.allclose(r1["x"], r2["x"])
        assert torch.allclose(r1["support"], r2["support"])

    def test_custom_transition_matrix(self):
        """Pipeline works with a non-reset-process transition matrix."""
        P = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        cfg = DataGenerationConfig(
            transition=TransitionConfig(matrix=P, stationary_on_prob=3.0 / 7.0),
            features=FeatureConfig(k=5, d=16),
            sequence=SequenceConfig(T=64, n_sequences=2),
        )
        result = generate_dataset(cfg)
        assert result["x"].shape == (2, 64, 16)
