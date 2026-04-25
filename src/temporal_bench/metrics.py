"""Evaluation metrics computed identically across all models.

Metrics:
  - NMSE: normalized mean squared error
  - L0: mean number of nonzero latents per token
  - Feature recovery AUC: area under the recovery curve
  - R@tau: fraction of features recovered above cosine threshold tau
  - Mean max cosine: average best-match cosine similarity

Global/local recovery metrics for the noisy-emission setting (Fig 8/9):
  - Single-latent Pearson correlation with observed support s (local) and
    hidden state h (global), using the best-match latent per true feature.
  - Linear-probe R^2 from the full latent vector to s and h (Ridge).
  - Denoising ratio = global / local correlation; > per-token floor (0.77)
    indicates the model is tracking the hidden state rather than the noisy
    emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

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
    # Move to CPU for metric computation
    D = F.normalize(decoder_dirs.cpu(), dim=0)  # (d, m) unit-norm columns
    F_true = F.normalize(true_features.cpu(), dim=1)  # (n_features, d) unit-norm rows

    # Cosine similarity: (n_features, m)
    cos_sim = (F_true @ D).abs()

    # Best match for each true feature
    max_cos, _ = cos_sim.max(dim=1)  # (n_features,)

    n_features = true_features.shape[0]
    mean_max_cos = max_cos.mean().item()

    # R@tau
    r_at_90 = (max_cos >= 0.9).float().mean().item()
    r_at_80 = (max_cos >= 0.8).float().mean().item()

    # AUC: integrate fraction recovered across thresholds (all on CPU for simplicity)
    max_cos_cpu = max_cos.cpu()
    thresholds = torch.linspace(0.0, 1.0, n_thresholds + 1)
    fracs = torch.tensor(
        [(max_cos_cpu >= tau).float().mean().item() for tau in thresholds]
    )
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
    l0 = compute_l0(model.latents_for_metrics(out))

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


# --- Global / local recovery for noisy HMM emissions (Fig 8/9) ----------------


@dataclass
class DenoisingMetrics:
    """Per-feature and summary global/local recovery statistics."""

    # Per-feature Pearson correlation of best-match latent with s (local) / h (global).
    corr_local_per_feature: list[float] = field(default_factory=list)
    corr_global_per_feature: list[float] = field(default_factory=list)
    # Per-feature Ridge-probe R^2 from full z to s (local) / h (global).
    r2_local_per_feature: list[float] = field(default_factory=list)
    r2_global_per_feature: list[float] = field(default_factory=list)
    # Group labels (e.g. per-feature rho used in heterogeneous-rho experiments).
    feature_rho: list[float] = field(default_factory=list)

    @property
    def corr_local(self) -> float:
        return float(np.mean(self.corr_local_per_feature)) if self.corr_local_per_feature else float("nan")

    @property
    def corr_global(self) -> float:
        return float(np.mean(self.corr_global_per_feature)) if self.corr_global_per_feature else float("nan")

    @property
    def r2_local(self) -> float:
        return float(np.mean(self.r2_local_per_feature)) if self.r2_local_per_feature else float("nan")

    @property
    def r2_global(self) -> float:
        return float(np.mean(self.r2_global_per_feature)) if self.r2_global_per_feature else float("nan")

    @property
    def denoising_ratio_corr(self) -> float:
        """Mean global / local correlation ratio; > 0.77 (per-token floor) indicates denoising."""
        local = self.corr_local
        if local == 0.0 or not np.isfinite(local):
            return float("nan")
        return self.corr_global / local

    @property
    def denoising_ratio_r2(self) -> float:
        local = self.r2_local
        if local == 0.0 or not np.isfinite(local):
            return float("nan")
        return self.r2_global / local


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


@torch.no_grad()
def evaluate_denoising(
    model: TemporalAE,
    eval_x: torch.Tensor,
    eval_s: torch.Tensor,
    eval_h: torch.Tensor,
    true_features: torch.Tensor,
    *,
    feature_rho: list[float] | None = None,
    ridge_alpha: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
) -> DenoisingMetrics:
    """Compute single-latent correlations and Ridge-probe R^2 against s and h.

    Args:
        model: Any TemporalAE.
        eval_x: (n_seq, T, d) observations (embedded from s).
        eval_s: (n_seq, n_features, T) observed support.
        eval_h: (n_seq, n_features, T) hidden state.
        true_features: (n_features, d) ground-truth feature directions,
            used to pick the best-match latent per feature by decoder cosine.
        feature_rho: optional (n_features,) list of per-feature rhos for
            per-group reporting in plots.
        ridge_alpha: Ridge regularization strength.
        test_size: Held-out fraction for the probe R^2.
        seed: Shuffle seed for the probe split.

    Returns:
        DenoisingMetrics (per-feature and summary statistics).
    """
    model.eval()
    out = model(eval_x)
    model.train()

    z = out.latents  # (n_seq, T, m) expected for all SAE-family models
    assert z.dim() == 3, f"expected (n_seq, T, m) latents, got {z.shape}"
    n_seq, T, m = z.shape

    # Align s and h to (n_seq, T, n_features) to match z's layout.
    s_tT = eval_s.permute(0, 2, 1).contiguous()  # (n_seq, T, n_features)
    h_tT = eval_h.permute(0, 2, 1).contiguous()
    n_features = true_features.shape[0]

    # Best-match latent per true feature: max abs cosine against decoder cols.
    decoder_dirs = model.decoder_directions()  # (d, m)
    D = F.normalize(decoder_dirs.cpu(), dim=0)
    F_true = F.normalize(true_features.cpu(), dim=1)
    cos_sim = (F_true @ D).abs()  # (n_features, m)
    best_latent = cos_sim.argmax(dim=1).numpy()  # (n_features,)

    # Flatten to (N, *) for easy correlation / probe fits.
    z_flat = z.reshape(n_seq * T, m).cpu().numpy()  # (N, m)
    s_flat = s_tT.reshape(n_seq * T, n_features).cpu().numpy()  # (N, n_feat)
    h_flat = h_tT.reshape(n_seq * T, n_features).cpu().numpy()  # (N, n_feat)

    idx_train, idx_test = train_test_split(
        np.arange(z_flat.shape[0]), test_size=test_size, random_state=seed
    )

    metrics = DenoisingMetrics(
        feature_rho=list(feature_rho) if feature_rho is not None else []
    )

    for i in range(n_features):
        j = int(best_latent[i])
        z_j = z_flat[:, j]
        metrics.corr_local_per_feature.append(_pearson(z_j, s_flat[:, i]))
        metrics.corr_global_per_feature.append(_pearson(z_j, h_flat[:, i]))

        # Ridge probe uses the full latent vector.
        ridge_s = Ridge(alpha=ridge_alpha)
        ridge_s.fit(z_flat[idx_train], s_flat[idx_train, i])
        metrics.r2_local_per_feature.append(
            float(ridge_s.score(z_flat[idx_test], s_flat[idx_test, i]))
        )
        ridge_h = Ridge(alpha=ridge_alpha)
        ridge_h.fit(z_flat[idx_train], h_flat[idx_train, i])
        metrics.r2_global_per_feature.append(
            float(ridge_h.score(z_flat[idx_test], h_flat[idx_test, i]))
        )

    return metrics
