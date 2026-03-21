"""Evaluation metrics for the HMM SAE baseline experiment.

The feature_recovery_score function matches Andre's implementation in
temporal_crosscoders/metrics.py: it sweeps a threshold from 0 to 1,
computes the fraction of ground-truth features whose best decoder match
exceeds each threshold (a survival curve), then takes the area under
that curve via trapezoidal integration.
"""

import numpy as np
import torch


@torch.no_grad()
def feature_recovery_score(
    W_dec: torch.Tensor,
    ground_truth_features: torch.Tensor,
) -> dict:
    """Compute AUC, mean-max cos-sim, and recovery fractions.

    Follows the threshold-sweep AUC from Andre's temporal_crosscoders/metrics.py.
    For each ground-truth feature, finds the decoder direction with highest
    absolute cosine similarity. Then sweeps a threshold t from 0 to 1 and
    computes the fraction of features recovered at each threshold. The AUC
    of this survival curve distinguishes bimodal recovery from uniform partial
    recovery.

    Args:
        W_dec: Decoder weight matrix of shape (n_latents, d_input), rows are
            learned feature directions.
        ground_truth_features: Ground-truth features of shape (k, d_input),
            rows are unit-norm directions.

    Returns:
        Dict with keys:
            mean_max_cos_sim: mean of per-feature max |cos sim|
            frac_recovered_90: fraction of features with max |cos sim| >= 0.9
            frac_recovered_80: fraction of features with max |cos sim| >= 0.8
            per_feature: numpy array of per-feature max |cos sim| values
            auc: area under the threshold-sweep survival curve
    """
    # Normalize to unit norm
    W_dec_norm = W_dec / W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    gt_norm = ground_truth_features / ground_truth_features.norm(
        dim=1, keepdim=True
    ).clamp(min=1e-8)

    # Cosine similarity matrix: (k, n_latents)
    cos_sim = gt_norm @ W_dec_norm.T
    # Max absolute cosine similarity per ground-truth feature
    max_per_true = cos_sim.abs().max(dim=1).values  # (k,)

    k = ground_truth_features.shape[0]
    thresholds = np.linspace(0, 1, k)
    max_np = max_per_true.cpu().numpy()
    curve = np.array([(max_np >= t).mean() for t in thresholds])
    auc_val = float(np.trapz(curve, thresholds) / (thresholds[-1] - thresholds[0]))

    return {
        "mean_max_cos_sim": max_per_true.mean().item(),
        "frac_recovered_90": (max_per_true >= 0.9).float().mean().item(),
        "frac_recovered_80": (max_per_true >= 0.8).float().mean().item(),
        "per_feature": max_np,
        "auc": auc_val,
    }


def compute_empirical_autocorrelation(
    support: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """Compute empirical autocorrelation of binary support sequences (per-chain).

    Computes autocorrelation within each chain, then averages. This is correct
    when chains mix (lambda > 0) but underestimates the marginal autocorrelation
    at lambda=0 because the per-chain estimator conditions on the (frozen) hidden
    state, making emissions appear i.i.d. within each chain. See
    compute_pooled_autocorrelation for the marginal estimator.

    Args:
        support: Binary support tensor of shape (n_seq, k, T).
        max_lag: Maximum lag to compute.

    Returns:
        Tensor of shape (max_lag + 1,) with empirical autocorrelation at each lag.
    """
    n_seq, k, T = support.shape
    # Flatten sequences and features: (n_seq * k, T)
    flat = support.reshape(-1, T)

    mean = flat.mean(dim=1, keepdim=True)  # (n_seq*k, 1)
    centered = flat - mean
    var = (centered**2).mean(dim=1)  # (n_seq*k,)

    autocorr = torch.zeros(max_lag + 1)
    for tau in range(max_lag + 1):
        if tau == 0:
            autocorr[tau] = 1.0
            continue
        # Covariance at lag tau
        cov = (centered[:, :T - tau] * centered[:, tau:]).mean(dim=1)  # (n_seq*k,)
        # Normalize by variance, handle zero-variance chains
        valid = var > 1e-12
        if valid.any():
            autocorr[tau] = (cov[valid] / var[valid]).mean()
        else:
            autocorr[tau] = 0.0

    return autocorr


def compute_pooled_autocorrelation(
    support: torch.Tensor, max_lag: int
) -> torch.Tensor:
    """Compute marginal autocorrelation by pooling across all chains.

    Treats all (n_seq * k) chains as draws from the same process and computes
    a single global mean and variance. This correctly captures the marginal
    autocorrelation Corr(s_t, s_{t+tau}) = rho^tau * gamma even at lambda=0,
    where the per-chain estimator fails because it conditions on the frozen
    hidden state.

    Args:
        support: Binary support tensor of shape (n_seq, k, T).
        max_lag: Maximum lag to compute.

    Returns:
        Tensor of shape (max_lag + 1,) with pooled autocorrelation at each lag.
    """
    n_seq, k, T = support.shape
    flat = support.reshape(-1, T)  # (n_seq*k, T)

    # Global mean and variance across all chains and positions
    global_mean = flat.mean()
    centered = flat - global_mean
    global_var = centered.pow(2).mean()

    if global_var < 1e-12:
        return torch.zeros(max_lag + 1)

    autocorr = torch.zeros(max_lag + 1)
    autocorr[0] = 1.0
    for tau in range(1, max_lag + 1):
        cov = (centered[:, :T - tau] * centered[:, tau:]).mean()
        autocorr[tau] = (cov / global_var).item()

    return autocorr
