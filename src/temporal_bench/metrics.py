"""Evaluation metrics computed identically across all models.

Metrics:
  - NMSE: normalized mean squared error
  - L0: mean number of nonzero latents per token
  - Feature recovery AUC: area under the recovery curve
  - R@tau: fraction of features recovered above cosine threshold tau
  - Mean max cosine: average best-match cosine similarity

Global vs local feature recovery (clean-setting variants, Fig 10):
  - Decoder-based: local = mean of per-position decoder AUCs; global = AUC
    of the position-averaged decoder. For models without distinct positional
    decoders (regular SAE), local == global.
  - Activation-trace: local = best per-token classification AUC of latent
    activation vs feature presence s_jt; global = best window-pooled
    (max-over-t) classification AUC vs feature presence anywhere in the
    window.

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
    # Decoder-based global vs local feature recovery (Option A).
    # local = mean across positions of per-position decoder AUC;
    # global = AUC of position-averaged decoder.
    # For non-temporal models these collapse to the standard `auc` field.
    auc_decoder_local: float = float("nan")
    auc_decoder_global: float = float("nan")
    # Activation-trace global vs local recovery (Option B), computed only
    # when the eval call provides per-feature support `s`.
    # local = mean over features of best-latent per-token classification AUC
    #         (latent activation vs s_jt > 0);
    # global = mean over features of best-latent window-pooled (max-over-t)
    #          classification AUC (latent vs feature-active-in-window).
    auc_activation_local: float = float("nan")
    auc_activation_global: float = float("nan")
    # Coupled-features global recovery (Aniket Level 3 / Dmitry exp1c3).
    # Standard `auc` (computed against emission feature directions) is
    # eAUC (local). `auc_hidden` is gAUC: standard feature_recovery AUC
    # but with the K hidden-direction features as the comparison set.
    # Populated only when evaluate() is called with hidden_features.
    auc_hidden: float = float("nan")
    r_at_90_hidden: float = float("nan")
    r_at_80_hidden: float = float("nan")
    mean_max_cos_hidden: float = float("nan")


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


def feature_recovery_decoder_global_local(
    model: TemporalAE,
    true_features: torch.Tensor,
    n_thresholds: int = 50,
) -> dict[str, float]:
    """Decoder-based global vs local feature recovery (Option A).

    local = mean over positions of per-position decoder feature-recovery AUC.
    global = feature-recovery AUC of the position-averaged decoder.

    For models where ``n_positions`` is None (regular SAE), no per-position
    decoder exists, so local and global both reduce to the standard
    pooled-decoder AUC.

    Args:
        model: Any TemporalAE.
        true_features: (n_features, d) ground-truth directions.
        n_thresholds: Forwarded to feature_recovery.

    Returns:
        Dict with keys: auc_local, auc_global.
    """
    n_pos = model.n_positions
    pooled = feature_recovery(
        model.decoder_directions(), true_features, n_thresholds=n_thresholds
    )
    if n_pos is None or n_pos <= 1:
        return {"auc_local": pooled["auc"], "auc_global": pooled["auc"]}

    per_pos_aucs = []
    for t in range(n_pos):
        rec_t = feature_recovery(
            model.decoder_directions(pos=t),
            true_features,
            n_thresholds=n_thresholds,
        )
        per_pos_aucs.append(rec_t["auc"])
    return {
        "auc_local": float(np.mean(per_pos_aucs)),
        "auc_global": pooled["auc"],
    }


def _auc_columns(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-column ROC AUC for binary y against each column of scores.

    Implements the rank-based Mann-Whitney U formula, vectorized over the
    score columns. Returns an array of shape (M,) with the AUC of each
    column. NaN if y has only one class.
    """
    n_pos = int(y.sum())
    n_neg = y.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], np.nan)
    # Column-wise ranks (average ranks for ties).
    from scipy.stats import rankdata

    ranks = rankdata(scores, axis=0)  # (N, M)
    sum_pos_ranks = ranks[y.astype(bool)].sum(axis=0)  # (M,)
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@torch.no_grad()
def feature_recovery_activation_global_local(
    model: TemporalAE,
    eval_x: torch.Tensor,
    eval_s: torch.Tensor,
) -> dict[str, float]:
    """Activation-trace global vs local recovery (Option B).

    For each true feature j, score every latent m by:
      - local AUC: classification of (s_{j,t} > 0) using latent activation
        z_{n,t,m} as the score, evaluated at the per-token level.
      - global AUC: classification of (any s_{j,t} > 0 in the window) using
        max_t z_{n,t,m} as the score, at the window level.
    Best latent per feature is the one with highest AUC (orientation-invariant
    via max(auc, 1-auc)). Aggregate by mean over features.

    Args:
        model: TemporalAE returning latents shape (B, T, m).
        eval_x: (n_seq, T, d) input observations.
        eval_s: (n_seq, n_features, T) per-feature support.

    Returns:
        Dict with keys: auc_local, auc_global.
    """
    model.eval()
    out = model(eval_x)
    model.train()
    z = out.latents
    assert z.dim() == 3, f"expected (n_seq, T, m), got {z.shape}"
    n_seq, T, m = z.shape

    # (n_seq, T, n_features) feature-presence support.
    s_tT = eval_s.permute(0, 2, 1).contiguous()
    n_features = s_tT.shape[-1]

    z_np = z.detach().cpu().numpy()
    s_np = s_tT.detach().cpu().numpy()

    # Local: token-level. y is per (n_seq, T) flattened.
    z_flat = z_np.reshape(n_seq * T, m)
    s_flat = s_np.reshape(n_seq * T, n_features)

    # Global: window-level. score = max over T of z; y = any s>0 in window.
    z_max = z_np.max(axis=1)  # (n_seq, m)
    s_any = (s_np > 0).any(axis=1).astype(np.float32)  # (n_seq, n_features)

    local_aucs = []
    global_aucs = []
    for j in range(n_features):
        y_local = (s_flat[:, j] > 0).astype(np.float32)
        if 0 < y_local.sum() < y_local.shape[0]:
            aucs_l = _auc_columns(y_local, z_flat)
            # orientation-invariant: best latent = max(auc, 1-auc)
            best_l = np.nanmax(np.maximum(aucs_l, 1.0 - aucs_l))
            local_aucs.append(float(best_l))

        y_global = s_any[:, j]
        if 0 < y_global.sum() < y_global.shape[0]:
            aucs_g = _auc_columns(y_global, z_max)
            best_g = np.nanmax(np.maximum(aucs_g, 1.0 - aucs_g))
            global_aucs.append(float(best_g))

    return {
        "auc_local": float(np.mean(local_aucs)) if local_aucs else float("nan"),
        "auc_global": float(np.mean(global_aucs)) if global_aucs else float("nan"),
    }


@torch.no_grad()
def evaluate(
    model: TemporalAE,
    eval_data: torch.Tensor,
    true_features: torch.Tensor,
    eval_s: torch.Tensor | None = None,
    hidden_features: torch.Tensor | None = None,
) -> EvalMetrics:
    """Run model on eval data and compute all metrics.

    Args:
        model: Any TemporalAE model.
        eval_data: (n_seq, T, d) evaluation data.
        true_features: (n_features, d) ground truth feature directions
            (in coupled mode this is the M emission directions; standard
            `auc` is then eAUC, the local recovery metric).
        eval_s: Optional (n_seq, n_features, T) per-feature support. When
            provided, activation-trace global/local AUCs are also computed.
        hidden_features: Optional (K, d) hidden-direction features. When
            provided, the gAUC suite (auc_hidden, r_at_*, mean_max_cos)
            is also populated.

    Returns:
        EvalMetrics with all standard metrics. The decoder-based global/local
        AUCs are always populated; activation-trace fields are populated only
        when eval_s is provided; gAUC (auc_hidden) only when hidden_features
        is provided.
    """
    model.eval()
    out = model(eval_data)

    nmse = compute_nmse(eval_data, out.x_hat)
    l0 = compute_l0(model.latents_for_metrics(out))

    decoder_dirs = model.decoder_directions()
    recovery = feature_recovery(decoder_dirs, true_features)
    dec_gl = feature_recovery_decoder_global_local(model, true_features)

    model.train()
    em = EvalMetrics(
        nmse=nmse,
        l0=l0,
        auc=recovery["auc"],
        r_at_90=recovery["r_at_90"],
        r_at_80=recovery["r_at_80"],
        mean_max_cos=recovery["mean_max_cos"],
        auc_decoder_local=dec_gl["auc_local"],
        auc_decoder_global=dec_gl["auc_global"],
    )
    if eval_s is not None:
        act_gl = feature_recovery_activation_global_local(model, eval_data, eval_s)
        em.auc_activation_local = act_gl["auc_local"]
        em.auc_activation_global = act_gl["auc_global"]
    if hidden_features is not None:
        hidden_recovery = feature_recovery(decoder_dirs, hidden_features)
        em.auc_hidden = hidden_recovery["auc"]
        em.r_at_90_hidden = hidden_recovery["r_at_90"]
        em.r_at_80_hidden = hidden_recovery["r_at_80"]
        em.mean_max_cos_hidden = hidden_recovery["mean_max_cos"]
    return em


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
