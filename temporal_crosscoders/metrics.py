"""
metrics.py — Feature recovery evaluation metrics.
"""

import torch
import numpy as np
from config import NUM_FEATS

THRESHOLDS = np.linspace(0, 1, NUM_FEATS)


def cos_sims(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Cosine similarities between columns of mat1 and mat2. Returns (h1, h2)."""
    n1 = mat1 / mat1.norm(dim=0, keepdim=True).clamp(min=1e-8)
    n2 = mat2 / mat2.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return n1.T @ n2


@torch.no_grad()
def feature_recovery_score(
    decoder_directions: torch.Tensor,  # (d, h)
    true_features: torch.Tensor,       # (d, k)
) -> dict:
    """Compute AUC, mean-max cos-sim, and recovery fractions."""
    sims = cos_sims(decoder_directions, true_features).abs()  # (h, k)
    max_per_true = sims.max(dim=0).values                     # (k,)
    curve = np.array([(max_per_true.cpu().numpy() >= t).mean() for t in THRESHOLDS])
    auc_val = float(np.trapezoid(curve, THRESHOLDS) / (THRESHOLDS[-1] - THRESHOLDS[0]))
    return {
        "mean_max_cos_sim": max_per_true.mean().item(),
        "frac_recovered_90": (max_per_true >= 0.9).float().mean().item(),
        "frac_recovered_80": (max_per_true >= 0.8).float().mean().item(),
        "per_feature": max_per_true.cpu().numpy(),
        "auc": auc_val,
    }
