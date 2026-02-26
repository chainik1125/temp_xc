import numpy as np

# Minimum variance to consider a feature's autocorrelation well-defined.
# Features with variance below this are effectively constant across the
# sequence, so autocorrelation is meaningless.
VARIANCE_FLOOR = 1e-10


def compute_autocorrelations_vectorized(
    feature_acts: np.ndarray,
    max_lag: int,
    min_activations: int,
) -> np.ndarray:
    """Compute temporal autocorrelation for all features in a single sequence.

    For each feature, computes the standard autocorrelation at lags 1..max_lag:

        rho(k) = sum((x_t - mean) * (x_{t+k} - mean)) / sum((x_t - mean)^2)

    Computed on the raw signal including zeros (inactive positions). Features
    with fewer than min_activations nonzero positions, or with near-zero
    variance, get NaN.

    Args:
        feature_acts: Array of shape [T, D] — one sequence, all features.
        max_lag: Maximum lag to compute (e.g. 10).
        min_activations: Minimum nonzero positions required per feature.

    Returns:
        Array of shape [D, max_lag] — autocorrelation at lags 1..max_lag.
        NaN for features that don't meet the activation threshold or have
        near-zero variance.
    """
    T, D = feature_acts.shape
    result = np.full((D, max_lag), np.nan)

    # Mask out features with too few activations
    num_active = np.count_nonzero(feature_acts, axis=0)  # [D]
    valid = num_active >= min_activations

    if not valid.any():
        return result

    acts = feature_acts[:, valid]  # [T, D_valid]

    # Mean-center
    mean = acts.mean(axis=0, keepdims=True)  # [1, D_valid]
    centered = acts - mean  # [T, D_valid]

    # Variance (denominator for all lags)
    variance = np.sum(centered**2, axis=0)  # [D_valid]

    # Mask out near-zero variance
    var_ok = variance > VARIANCE_FLOOR
    if not var_ok.any():
        return result

    for lag in range(1, max_lag + 1):
        covariance = np.sum(centered[: T - lag] * centered[lag:], axis=0)  # [D_valid]
        ac = np.where(var_ok, covariance / variance, np.nan)
        result[valid, lag - 1] = ac

    return result
