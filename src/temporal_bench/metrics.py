"""Evaluation metrics computed identically across all models.

Metrics:
  - NMSE: normalized mean squared error
  - L0: mean number of nonzero latents per token
  - Feature recovery AUC: area under the recovery curve
  - R@tau: fraction of features recovered above cosine threshold tau
  - Mean max cosine: average best-match cosine similarity
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .models.base import TemporalAE


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single model run."""

    nmse: float
    l0: float
    auc: float
    r_at_90: float
    r_at_80: float
    mean_max_cos: float


def compute_nmse(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Normalized mean squared error: sum(||x - x_hat||^2) / sum(||x||^2)."""
    return (x - x_hat).pow(2).sum().item() / x.pow(2).sum().item()


def compute_l0(latents: torch.Tensor) -> float:
    """Mean number of nonzero latent activations per token.

    Args:
        latents: (B, T, m) or (B, m) latent activations.
    """
    if latents.dim() == 3:
        B, T, m = latents.shape
        flat = latents.reshape(B * T, m)
    else:
        flat = latents
    return (flat != 0).float().sum(dim=-1).mean().item()


def feature_recovery(
    decoder_dirs: torch.Tensor,
    true_features: torch.Tensor,
    n_thresholds: int = 50,
) -> dict[str, float]:
    """Compute feature recovery metrics.

    For each true feature, find the best-matching decoder column by absolute
    cosine similarity. Then compute AUC, R@tau, and mean max cosine.

    Args:
        decoder_dirs: (d, m) decoder weight matrix (columns are atoms).
        true_features: (n_features, d) true feature directions.
        n_thresholds: Number of threshold values for AUC integration.

    Returns:
        Dict with keys: auc, r_at_90, r_at_80, mean_max_cos.
    """
    # Normalize
    D = F.normalize(decoder_dirs, dim=0)  # (d, m) unit-norm columns
    F_true = F.normalize(true_features, dim=1)  # (n_features, d) unit-norm rows

    # Cosine similarity: (n_features, m)
    cos_sim = (F_true @ D).abs()

    # Best match for each true feature
    max_cos, _ = cos_sim.max(dim=1)  # (n_features,)

    n_features = true_features.shape[0]
    mean_max_cos = max_cos.mean().item()

    # R@tau
    r_at_90 = (max_cos >= 0.9).float().mean().item()
    r_at_80 = (max_cos >= 0.8).float().mean().item()

    # AUC: integrate fraction recovered across thresholds
    thresholds = torch.linspace(0.0, 1.0, n_thresholds + 1, device=max_cos.device)
    fracs = torch.tensor(
        [(max_cos >= tau).float().mean().item() for tau in thresholds]
    )
    # Trapezoidal integration
    auc = torch.trapezoid(fracs, thresholds).item()

    return {
        "auc": auc,
        "r_at_90": r_at_90,
        "r_at_80": r_at_80,
        "mean_max_cos": mean_max_cos,
    }


@torch.no_grad()
def evaluate(
    model: TemporalAE,
    eval_data: torch.Tensor,
    true_features: torch.Tensor,
) -> EvalMetrics:
    """Run model on eval data and compute all metrics.

    Args:
        model: Any TemporalAE model.
        eval_data: (n_seq, T, d) evaluation data.
        true_features: (n_features, d) ground truth feature directions.

    Returns:
        EvalMetrics with all standard metrics.
    """
    model.eval()
    out = model(eval_data)

    nmse = compute_nmse(eval_data, out.x_hat)
    l0 = compute_l0(out.latents)

    decoder_dirs = model.decoder_directions()
    recovery = feature_recovery(decoder_dirs, true_features)

    model.train()
    return EvalMetrics(
        nmse=nmse,
        l0=l0,
        auc=recovery["auc"],
        r_at_90=recovery["r_at_90"],
        r_at_80=recovery["r_at_80"],
        mean_max_cos=recovery["mean_max_cos"],
    )
