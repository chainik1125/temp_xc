import numpy as np
import pytest

from bill.temporal_autocorrelation.autocorrelation import (
    VARIANCE_FLOOR,
    compute_autocorrelations_vectorized,
)
from bill.temporal_autocorrelation.statistics import (
    FeatureStatistics,
    compute_shuffled_baseline,
    ljung_box_test,
)


def make_ar1(rho: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an AR(1) process: x_t = rho * x_{t-1} + eps_t."""
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.standard_normal()
    return x


class TestAutocorrelationVectorized:
    def test_ar1_lag1(self):
        """AR(1) with rho=0.8 should have lag-1 AC near 0.8."""
        rng = np.random.default_rng(42)
        signal = make_ar1(0.8, 10_000, rng)
        acts = signal.reshape(-1, 1)  # [T, 1]

        ac = compute_autocorrelations_vectorized(acts, max_lag=2, min_activations=0)
        assert ac.shape == (1, 2)
        assert abs(ac[0, 0] - 0.8) < 0.05

    def test_ar1_lag2(self):
        """AR(1) with rho=0.8 should have lag-2 AC near 0.64."""
        rng = np.random.default_rng(42)
        signal = make_ar1(0.8, 10_000, rng)
        acts = signal.reshape(-1, 1)

        ac = compute_autocorrelations_vectorized(acts, max_lag=2, min_activations=0)
        assert abs(ac[0, 1] - 0.64) < 0.05

    def test_all_zeros_returns_nan(self):
        """All-zero signal should return NaN (no activations)."""
        acts = np.zeros((100, 3))
        ac = compute_autocorrelations_vectorized(acts, max_lag=5, min_activations=1)
        assert np.isnan(ac).all()

    def test_constant_returns_nan(self):
        """Constant nonzero signal has zero variance, should return NaN."""
        acts = np.ones((100, 2)) * 5.0
        ac = compute_autocorrelations_vectorized(acts, max_lag=5, min_activations=0)
        assert np.isnan(ac).all()

    def test_min_activations_filter(self):
        """Features below the min_activations threshold should get NaN."""
        rng = np.random.default_rng(0)
        acts = np.zeros((200, 2))
        # Feature 0: 5 nonzero values (below threshold of 20)
        acts[rng.choice(200, 5, replace=False), 0] = 1.0
        # Feature 1: 50 nonzero values (above threshold)
        acts[rng.choice(200, 50, replace=False), 1] = 1.0

        ac = compute_autocorrelations_vectorized(acts, max_lag=3, min_activations=20)
        assert np.isnan(ac[0]).all()
        assert not np.isnan(ac[1]).all()

    def test_vectorized_matches_scalar(self):
        """Multi-feature result matches computing each feature separately."""
        rng = np.random.default_rng(123)
        T, D = 500, 5
        acts = np.abs(rng.standard_normal((T, D)))

        ac_vec = compute_autocorrelations_vectorized(acts, max_lag=4, min_activations=0)

        for d in range(D):
            ac_scalar = compute_autocorrelations_vectorized(
                acts[:, d : d + 1], max_lag=4, min_activations=0
            )
            np.testing.assert_allclose(ac_vec[d], ac_scalar[0], atol=1e-10)

    def test_white_noise_near_zero(self):
        """White noise should have autocorrelation near zero at all lags."""
        rng = np.random.default_rng(99)
        signal = rng.standard_normal((10_000, 1))

        ac = compute_autocorrelations_vectorized(signal, max_lag=10, min_activations=0)
        assert np.all(np.abs(ac) < 0.05)


class TestShuffledBaseline:
    def test_shuffled_collapses_autocorrelation(self):
        """Shuffling an AR(1) process should collapse autocorrelation to ~0."""
        rng = np.random.default_rng(42)
        signal = make_ar1(0.9, 5_000, rng)
        acts = signal.reshape(-1, 1)

        shuffled_ac = compute_shuffled_baseline(
            acts, max_lag=5, min_activations=0, rng=np.random.default_rng(0)
        )
        assert np.all(np.abs(shuffled_ac) < 0.1)


class TestLjungBox:
    def test_rejects_ar1(self):
        """Ljung-Box should reject the null for an AR(1) process."""
        rng = np.random.default_rng(42)
        signal = make_ar1(0.8, 1_000, rng)
        acts = signal.reshape(-1, 1)

        pvals = ljung_box_test(acts, max_lag=10)
        assert pvals[0] < 0.01

    def test_fails_to_reject_white_noise(self):
        """Ljung-Box should not reject the null for white noise."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((1_000, 1))

        pvals = ljung_box_test(signal, max_lag=10)
        assert pvals[0] > 0.05

    def test_all_zero_returns_nan(self):
        """All-zero feature should return NaN p-value."""
        acts = np.zeros((100, 1))
        pvals = ljung_box_test(acts, max_lag=5)
        assert np.isnan(pvals[0])


class TestFeatureStatistics:
    def test_incremental_accumulation(self):
        """Statistics accumulated over two sequences should be consistent."""
        rng = np.random.default_rng(42)
        D = 10
        acts1 = np.abs(rng.standard_normal((100, D)))
        acts2 = np.abs(rng.standard_normal((100, D)))

        stats = FeatureStatistics(num_features=D, max_lag=3)
        stats.update(acts1, min_activations=0)
        stats.update(acts2, min_activations=0)
        stats.finalize()

        assert stats.total_positions == 200
        assert stats.mean_magnitude_when_active.shape == (D,)
        assert stats.activation_frequency.shape == (D,)
        assert stats.mean_autocorrelation.shape == (D, 3)
        # All features should have valid stats (nonzero abs normal values)
        assert not np.isnan(stats.mean_magnitude_when_active).any()
        assert not np.isnan(stats.mean_autocorrelation).any()
