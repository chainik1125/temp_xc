from dataclasses import dataclass, field

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

from bill.temporal_autocorrelation.autocorrelation import compute_autocorrelations_vectorized


@dataclass
class FeatureStatistics:
    """Incrementally accumulated statistics across sequences.

    All arrays have shape [D] or [D, max_lag]. Call update() per sequence,
    then finalize() once to compute means.
    """

    num_features: int
    max_lag: int

    # Accumulators (summed across sequences)
    magnitude_sum: np.ndarray = field(init=False)
    active_count: np.ndarray = field(init=False)
    total_positions: int = field(init=False, default=0)
    autocorrelation_sum: np.ndarray = field(init=False)
    autocorrelation_count: np.ndarray = field(init=False)

    # Final results (populated by finalize())
    mean_magnitude_when_active: np.ndarray | None = field(init=False, default=None)
    activation_frequency: np.ndarray | None = field(init=False, default=None)
    mean_autocorrelation: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self):
        D, K = self.num_features, self.max_lag
        self.magnitude_sum = np.zeros(D)
        self.active_count = np.zeros(D, dtype=np.int64)
        self.autocorrelation_sum = np.zeros((D, K))
        self.autocorrelation_count = np.zeros((D, K), dtype=np.int64)

    def update(self, feature_acts: np.ndarray, min_activations: int) -> None:
        """Accumulate statistics from one sequence.

        Args:
            feature_acts: Array of shape [T, D] for a single sequence.
            min_activations: Passed through to autocorrelation computation.
        """
        T, D = feature_acts.shape

        # Magnitude and frequency
        active_mask = feature_acts > 0  # [T, D]
        self.magnitude_sum += feature_acts.sum(axis=0)
        self.active_count += active_mask.sum(axis=0)
        self.total_positions += T

        # Autocorrelation
        ac = compute_autocorrelations_vectorized(
            feature_acts, self.max_lag, min_activations
        )  # [D, max_lag]
        valid = ~np.isnan(ac)
        safe_ac = np.where(valid, ac, 0.0)
        self.autocorrelation_sum += safe_ac
        self.autocorrelation_count += valid.astype(np.int64)

    def finalize(self) -> None:
        """Compute means from accumulated sums."""
        with np.errstate(divide="ignore", invalid="ignore"):
            self.mean_magnitude_when_active = np.where(
                self.active_count > 0,
                self.magnitude_sum / self.active_count,
                np.nan,
            )
            self.activation_frequency = self.active_count / self.total_positions
            self.mean_autocorrelation = np.where(
                self.autocorrelation_count > 0,
                self.autocorrelation_sum / self.autocorrelation_count,
                np.nan,
            )


def compute_shuffled_baseline(
    feature_acts: np.ndarray,
    max_lag: int,
    min_activations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute autocorrelation on a token-position-shuffled sequence.

    Randomly permutes token positions independently for each feature,
    destroying temporal structure while preserving marginal statistics.

    Args:
        feature_acts: Array of shape [T, D] for a single sequence.
        max_lag: Maximum lag.
        min_activations: Passed through to autocorrelation computation.
        rng: NumPy random generator.

    Returns:
        Array of shape [D, max_lag] — shuffled autocorrelations.
    """
    T, D = feature_acts.shape
    shuffled = feature_acts.copy()
    # Shuffle each feature's time series independently
    for d in range(D):
        rng.shuffle(shuffled[:, d])
    return compute_autocorrelations_vectorized(shuffled, max_lag, min_activations)


def ljung_box_test(
    feature_acts: np.ndarray,
    max_lag: int,
) -> np.ndarray:
    """Run Ljung-Box test per feature on a single sequence.

    Tests the joint null hypothesis that autocorrelations at lags 1..max_lag
    are all zero.

    Args:
        feature_acts: Array of shape [T, D] for a single sequence.
        max_lag: Number of lags to test.

    Returns:
        Array of shape [D] — p-values. NaN for features that are all-zero
        or constant.
    """
    T, D = feature_acts.shape
    p_values = np.full(D, np.nan)

    for d in range(D):
        series = feature_acts[:, d]
        if np.count_nonzero(series) == 0 or np.std(series) == 0:
            continue
        result = acorr_ljungbox(series, lags=max_lag, return_df=True)
        # Take p-value at the maximum lag (joint test over all lags)
        p_values[d] = result["lb_pvalue"].iloc[-1]

    return p_values
