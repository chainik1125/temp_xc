"""Temporal analysis metrics for real-LM SAE / crosscoder comparison.

These are the metrics the sprint plan (`exploration.md`) calls for to
distinguish "genuinely temporal" features from "features that just happen to
live in a temporal architecture":

- temporal_mi           Mutual information between f_t and f_{t+k}. A feature
                        whose firing pattern predicts its own future at
                        lag > 0 is carrying temporal structure.
- activation_span_stats Per-feature activation geometry across a sequence.
                        Mean contiguous span, number of bursts, duty cycle.
                        SAE-like features fire on single tokens; temporal
                        features should have spans > 1.
- cluster_features      UMAP + HDBSCAN (with a KMeans fallback) over decoder
                        directions. Feeds the Dmitry unsupervised→supervised
                        pipeline: cluster learned features, auto-interp the
                        cluster centroids, compare clusters across archs.

Kept in its own module so nothing that imports these has to pull in sae_lens
(which breaks on Trillium). Safe to `from src.shared.temporal_metrics import *`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch


# ─────────────────────────────────────────────────────────── temporal MI ────
def _binarize(acts: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Binarize (B, T, F) activations. Threshold = per-feature 90th pct by default."""
    if threshold is None:
        # Per-feature threshold = median of nonzero activations (robust to long tails).
        nonzero_mask = acts > 0
        out = np.zeros_like(acts, dtype=np.uint8)
        for f in range(acts.shape[-1]):
            col = acts[..., f]
            nz = col[nonzero_mask[..., f]]
            thr = float(np.median(nz)) if nz.size > 0 else 0.0
            out[..., f] = (col > thr).astype(np.uint8)
        return out
    return (acts > threshold).astype(np.uint8)


def _mi_binary(a: np.ndarray, b: np.ndarray) -> float:
    """MI between two binary vectors of equal length. Returns nats."""
    # Joint distribution over {0,1}x{0,1}
    n = a.size
    if n == 0:
        return 0.0
    p11 = float(np.logical_and(a == 1, b == 1).sum()) / n
    p10 = float(np.logical_and(a == 1, b == 0).sum()) / n
    p01 = float(np.logical_and(a == 0, b == 1).sum()) / n
    p00 = 1.0 - p11 - p10 - p01
    pa1 = p11 + p10
    pb1 = p11 + p01

    def _term(pxy, px, py):
        if pxy <= 0 or px <= 0 or py <= 0:
            return 0.0
        return pxy * np.log(pxy / (px * py))

    return (
        _term(p11, pa1, pb1)
        + _term(p10, pa1, 1 - pb1)
        + _term(p01, 1 - pa1, pb1)
        + _term(p00, 1 - pa1, 1 - pb1)
    )


@dataclass
class TemporalMIResult:
    lags: list[int]
    mean_mi_per_lag: list[float]          # averaged across features
    frac_features_above_threshold: list[float]  # fraction with MI > 0.01 nats
    per_feature_mi: np.ndarray | None = None    # (F, K) if return_per_feature
    threshold: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        d = {
            "lags": self.lags,
            "mean_mi_per_lag": self.mean_mi_per_lag,
            "frac_features_above_threshold": self.frac_features_above_threshold,
            "threshold": self.threshold,
        }
        if self.per_feature_mi is not None:
            d["per_feature_mi_shape"] = list(self.per_feature_mi.shape)
        return d


def temporal_mi(
    activations: torch.Tensor | np.ndarray,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    threshold: float = 0.01,
    max_features: int | None = 2048,
    return_per_feature: bool = False,
) -> TemporalMIResult:
    """Compute temporal mutual information between f_t and f_{t+k}.

    Args:
        activations: shape (B, T, F) real-valued feature activations.
        lags: lag values k to evaluate.
        threshold: MI (nats) cutoff for "has temporal structure".
        max_features: if F is huge (e.g. 32k for 8B × 8x expansion), subsample
            this many randomly to keep the compute bounded.
        return_per_feature: if True, also return the full (F, K) MI matrix.

    Returns: TemporalMIResult.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().float().cpu().numpy()
    assert activations.ndim == 3, "expected (B, T, F)"
    B, T, F = activations.shape

    if max_features is not None and F > max_features:
        rng = np.random.default_rng(0)
        feat_idx = rng.choice(F, size=max_features, replace=False)
        activations = activations[:, :, feat_idx]
        F = max_features

    binary = _binarize(activations)  # (B, T, F)

    per_feat = np.zeros((F, len(lags)), dtype=np.float32)
    for li, k in enumerate(lags):
        if k >= T:
            continue
        a = binary[:, :-k, :].reshape(-1, F)
        b = binary[:, k:, :].reshape(-1, F)
        for f in range(F):
            per_feat[f, li] = _mi_binary(a[:, f], b[:, f])

    mean_per_lag = per_feat.mean(axis=0).tolist()
    frac_per_lag = (per_feat > threshold).mean(axis=0).tolist()

    return TemporalMIResult(
        lags=list(lags),
        mean_mi_per_lag=mean_per_lag,
        frac_features_above_threshold=frac_per_lag,
        per_feature_mi=per_feat if return_per_feature else None,
        threshold=threshold,
    )


# ─────────────────────────────────────────────────── activation span stats ──
@dataclass
class ActivationSpanStats:
    mean_span: float            # mean contiguous run length (tokens)
    mean_bursts_per_seq: float  # mean number of on-runs per sequence per feature
    mean_duty_cycle: float      # fraction of tokens active
    fraction_single_token: float  # fraction of features whose mean span == 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def activation_span_stats(
    activations: torch.Tensor | np.ndarray,
    threshold: float | None = None,
    max_features: int | None = 2048,
) -> ActivationSpanStats:
    """Per-feature contiguous-span statistics over sequences.

    A "burst" = a maximal run of consecutive tokens where the feature is on.
    SAE-like features tend to have mean_span ≈ 1; temporal features should
    have > 1.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().float().cpu().numpy()
    assert activations.ndim == 3
    B, T, F = activations.shape

    if max_features is not None and F > max_features:
        rng = np.random.default_rng(0)
        activations = activations[:, :, rng.choice(F, size=max_features, replace=False)]
        F = max_features

    binary = _binarize(activations, threshold=threshold)  # (B, T, F) uint8

    per_feature_mean_span: list[float] = []
    per_feature_bursts: list[float] = []
    per_feature_duty: list[float] = []

    for f in range(F):
        col = binary[:, :, f]  # (B, T)
        duty = float(col.mean())
        per_feature_duty.append(duty)
        span_total = 0
        burst_count = 0
        for b in range(B):
            row = col[b]
            in_burst = False
            cur_len = 0
            for v in row:
                if v:
                    if not in_burst:
                        burst_count += 1
                        in_burst = True
                    cur_len += 1
                    span_total += 1
                else:
                    in_burst = False
        mean_span = span_total / max(burst_count, 1)
        per_feature_mean_span.append(mean_span)
        per_feature_bursts.append(burst_count / B)

    arr_span = np.asarray(per_feature_mean_span)
    return ActivationSpanStats(
        mean_span=float(arr_span.mean()),
        mean_bursts_per_seq=float(np.mean(per_feature_bursts)),
        mean_duty_cycle=float(np.mean(per_feature_duty)),
        fraction_single_token=float((arr_span <= 1.0 + 1e-9).mean()),
    )


# ───────────────────────────────────────────────────── feature clustering ──
@dataclass
class ClusterResult:
    coords: np.ndarray          # (F, 2) 2D projection
    labels: np.ndarray          # (F,) cluster assignment, -1 = noise
    n_clusters: int
    method: str
    umap_params: dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "coords_shape": list(self.coords.shape),
            "labels_unique": int(len(set(self.labels.tolist()))),
            "n_clusters": self.n_clusters,
            "method": self.method,
            "umap_params": self.umap_params,
        }


def cluster_features(
    decoder_directions: torch.Tensor | np.ndarray,
    n_clusters: int = 20,
    random_state: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    use_hdbscan: bool = True,
) -> ClusterResult:
    """UMAP decoder directions → 2D; cluster with HDBSCAN or KMeans fallback.

    decoder_directions: (d_model, d_sae) or (d_sae, d_model) — the function
    normalizes on the d_sae axis regardless.
    """
    if isinstance(decoder_directions, torch.Tensor):
        decoder_directions = decoder_directions.detach().float().cpu().numpy()

    # Ensure rows = features, cols = d_model
    D = decoder_directions
    if D.shape[0] < D.shape[1]:
        # (d_model, d_sae) — transpose
        D = D.T
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)

    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist,
            n_components=2, random_state=random_state, metric="cosine",
        )
        coords = reducer.fit_transform(D)
        umap_params = {"n_neighbors": n_neighbors, "min_dist": min_dist, "metric": "cosine"}
    except Exception as e:
        # Fallback: PCA to 2D if UMAP isn't installed.
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=random_state).fit_transform(D)
        umap_params = {"fallback": "pca", "reason": str(e)}

    labels: np.ndarray
    method: str
    if use_hdbscan:
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(10, D.shape[0] // 200))
            labels = clusterer.fit_predict(coords)
            method = "hdbscan"
        except Exception:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(coords)
            method = "kmeans_fallback"
    else:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(coords)
        method = "kmeans"

    n_clust = int(len(set(int(x) for x in labels if x != -1)))
    return ClusterResult(
        coords=coords.astype(np.float32),
        labels=labels.astype(np.int32),
        n_clusters=n_clust,
        method=method,
        umap_params=umap_params,
    )
