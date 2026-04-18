"""Feature recovery metrics: AUC and cosine-similarity-based evaluation.

Given a decoder matrix and ground-truth feature directions, measures how
well the learned dictionary recovers the true features.

Adapted from Andre Shportko's Aniket's original metrics module.
"""

import numpy as np
import torch


def cos_sims(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarities between columns of mat1 and mat2.

    Args:
        mat1: (d, h1) — e.g., decoder columns.
        mat2: (d, h2) — e.g., true feature directions.

    Returns:
        (h1, h2) cosine similarity matrix.
    """
    n1 = mat1 / mat1.norm(dim=0, keepdim=True).clamp(min=1e-8)
    n2 = mat2 / mat2.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return n1.T @ n2


@torch.no_grad()
def feature_recovery_score(
    decoder_directions: torch.Tensor,
    true_features: torch.Tensor,
    n_thresholds: int = 50,
) -> dict:
    """Compute feature recovery AUC and related metrics.

    For each true feature, finds the best-matching decoder column (by absolute
    cosine similarity). Then sweeps a threshold τ from 0 to 1 and computes the
    fraction of true features recovered at each threshold. AUC is the area under
    this curve.

    Args:
        decoder_directions: (d, h) decoder weight matrix (columns = atoms).
        true_features: (d, n_features) ground-truth feature directions.
        n_thresholds: Number of threshold points for the AUC curve.

    Returns:
        Dict with keys:
            auc: Area under the recovery curve (0 to 1).
            mean_max_cos: Mean of the best-match cosine per true feature.
            frac_recovered_90: Fraction of features with best-match cos ≥ 0.9.
            frac_recovered_80: Fraction with cos ≥ 0.8.
            per_feature: (n_features,) best-match cosine per true feature.
    """
    sims = cos_sims(decoder_directions, true_features).abs()  # (h, n_features)
    max_per_true = sims.max(dim=0).values  # (n_features,)

    thresholds = np.linspace(0, 1, n_thresholds)
    curve = np.array([
        (max_per_true.cpu().numpy() >= t).mean() for t in thresholds
    ])
    auc_val = float(np.trapezoid(curve, thresholds))

    return {
        "auc": auc_val,
        "mean_max_cos": max_per_true.mean().item(),
        "frac_recovered_90": (max_per_true >= 0.9).float().mean().item(),
        "frac_recovered_80": (max_per_true >= 0.8).float().mean().item(),
        "per_feature": max_per_true.cpu().numpy(),
    }


def sae_decoder_directions(sae) -> torch.Tensor:
    """Extract (d, h) decoder columns from a ReLUSAE."""
    return sae.W_dec.data.T  # W_dec is (h, d), we want (d, h)


def tfa_decoder_directions(tfa) -> torch.Tensor:
    """Extract (d, h) decoder columns from a TemporalSAE.

    TFA uses a shared dictionary D of shape (width, dimin).
    Decoder columns are D.T: (dimin, width).
    """
    return tfa.D.data.T  # D is (width, dimin), we want (dimin, width)
