#!/usr/bin/env python3
"""compute_temporal_metrics.py — backfill the pre-registered sprint metrics.

For a trained checkpoint + cached activation slice, computes the three
metrics defined in docs/aniket/sprint_coding_dataset_plan.md § Metric
definitions:

  - silhouette score in pre-UMAP PCA-50 space (KMeans k=20, cosine)
  - cluster size entropy
  - mean auto-MI across lags {1, 2, 4, 8} (pre-registered headline
    scalar = mean of mean_mi_per_lag)

Writes one JSON per checkpoint to:
  <output-dir>/metrics_<label>.json

Usage:
    python scripts/compute_temporal_metrics.py \\
        --checkpoint results/nlp/step2-unshuffled/ckpts/crosscoder__...pt \\
        --arch crosscoder --subject-model deepseek-r1-distill-llama-8b \\
        --cached-dataset gsm8k --layer-key resid_L12 \\
        --k 100 --T 5 \\
        --label step2-unshuffled-crosscoder \\
        --output-dir reports/step2-deepseek-reasoning
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.bench.model_registry import get_model_config, list_models
from src.bench.architectures.base import ArchSpec
from src.bench.architectures.topk_sae import TopKSAESpec
from src.bench.architectures.stacked_sae import StackedSAESpec
from src.bench.architectures.crosscoder import CrosscoderSpec
from src.shared.temporal_metrics import temporal_mi as _temporal_mi
from temporal_crosscoders.NLP.config import cache_dir_for


ARCH_SPECS: dict[str, callable] = {
    "topk_sae": lambda T: TopKSAESpec(),
    "stacked_sae": lambda T: StackedSAESpec(T=T),
    "crosscoder": lambda T: CrosscoderSpec(T=T),
}


def load_model(
    checkpoint: str,
    arch: str,
    subject_model: str,
    k: int,
    T: int,
    expansion_factor: int,
    device: torch.device,
) -> tuple[ArchSpec, torch.nn.Module, int, int]:
    """Load a trained checkpoint via the spec-based architecture classes.

    Returns (spec, model, d_in, d_sae).
    """
    cfg = get_model_config(subject_model)
    d_in = cfg.d_model
    d_sae = d_in * expansion_factor
    spec = ARCH_SPECS[arch](T)
    model = spec.create(d_in=d_in, d_sae=d_sae, k=k, device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return spec, model, d_in, d_sae


def load_activation_windows(
    cache_dir: str,
    layer_key: str,
    T: int,
    max_sequences: int,
    max_windows: int,
    seed: int,
) -> np.ndarray:
    """Load a subsample of cached activations and build sliding windows of size T.

    Returns (N, T, d) float32 — N is clipped to max_windows.
    """
    act_path = os.path.join(cache_dir, f"{layer_key}.npy")
    data = np.load(act_path, mmap_mode="r")  # (n_seq, seq_len, d)
    n_seq, seq_len, d = data.shape

    rng = np.random.default_rng(seed)
    seq_idx = rng.choice(n_seq, size=min(max_sequences, n_seq), replace=False)
    seqs = np.stack([data[i].copy() for i in seq_idx])  # (S, seq_len, d)

    n_windows_per_seq = seq_len - T + 1
    if n_windows_per_seq <= 0:
        raise ValueError(f"seq_len={seq_len} < T={T}; no windows possible")

    # Build sliding windows: (S, n_windows_per_seq, T, d)
    windows = np.lib.stride_tricks.sliding_window_view(
        seqs, window_shape=T, axis=1,
    ).transpose(0, 1, 3, 2)  # (S, n_windows_per_seq, T, d)
    flat_windows = windows.reshape(-1, T, d)

    if flat_windows.shape[0] > max_windows:
        pick = rng.choice(flat_windows.shape[0], size=max_windows, replace=False)
        flat_windows = flat_windows[pick]

    return flat_windows.astype(np.float32)


def encode_batched(
    spec: ArchSpec,
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Forward-pass windows through spec.encode in mini-batches.

    Returns (N, T, d_sae) float32 on CPU.
    """
    N, T, d = windows.shape
    out_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, N, batch_size):
            x = torch.from_numpy(windows[s : s + batch_size]).to(device)
            z = spec.encode(model, x)  # (B, T, d_sae)
            out_chunks.append(z.detach().float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def decoder_geometry_metrics(
    spec: ArchSpec,
    model: torch.nn.Module,
    device: torch.device,
    n_pca: int = 50,
    n_clusters: int = 20,
    seed: int = 42,
) -> dict:
    """Silhouette + cluster-size entropy on the pre-UMAP PCA-50 decoder space.

    Pipeline matches feature_map.py: position-averaged decoder, L2-normalize
    per feature, PCA to n_pca, KMeans(n_clusters), then silhouette(cosine)
    and entropy of the resulting cluster-size distribution.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Position-averaged decoder directions, shape (d_in, d_sae).
    dd = spec.decoder_directions(model, pos=None).to(device).detach().cpu().numpy()
    # Rows = features, cols = d_in
    D = dd.T  # (d_sae, d_in)
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    D = D / norms

    n_pca = min(n_pca, D.shape[0] - 1, D.shape[1])
    pca = PCA(n_components=n_pca, random_state=seed)
    X = pca.fit_transform(D)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
    labels = km.labels_

    sil = float(silhouette_score(X, labels, metric="cosine"))
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum())

    return {
        "silhouette": sil,
        "cluster_entropy": entropy,
        "n_clusters_found": int(len(counts)),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "cluster_sizes": counts.tolist(),
    }


def auto_mi_metric(
    feats: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    max_features: int = 2048,
) -> dict:
    """Mean auto-MI across lags — the pre-registered third metric.

    `mean_mi_per_lag` and `frac_features_above_threshold` are reported for
    transparency; the *pre-registered headline scalar* is the mean over lags
    of `mean_mi_per_lag`, unambiguous.
    """
    res = _temporal_mi(feats, lags=lags, max_features=max_features)
    # res.mean_mi_per_lag is a list of floats, one per lag that fit in T.
    nonzero = [v for v in res.mean_mi_per_lag if v > 0.0]
    scalar = float(np.mean(res.mean_mi_per_lag))
    return {
        "mean_auto_mi_scalar": scalar,
        "mean_mi_per_lag": list(res.mean_mi_per_lag),
        "frac_features_above_threshold": list(res.frac_features_above_threshold),
        "lags": list(res.lags),
        "threshold": res.threshold,
        "n_features_sampled": int(min(feats.shape[-1], max_features)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--arch", type=str, required=True,
        choices=list(ARCH_SPECS.keys()),
    )
    parser.add_argument(
        "--subject-model", type=str, required=True, choices=list_models(),
    )
    parser.add_argument("--cached-dataset", type=str, required=True)
    parser.add_argument("--layer-key", type=str, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--max-sequences", type=int, default=256,
        help="Max cached chains to sample for the activation slice.",
    )
    parser.add_argument(
        "--max-windows", type=int, default=16384,
        help="Max sliding windows fed through encode() for auto-MI.",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=20,
        help="KMeans cluster count for silhouette/entropy.",
    )
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto",
        help="'auto' picks cuda if available else cpu.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.checkpoint):
        print(f"FATAL: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    cache_dir = cache_dir_for(args.subject_model, args.cached_dataset)
    if not os.path.exists(cache_dir):
        print(f"FATAL: cache dir not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== compute_temporal_metrics: {args.label} ===")
    print(f"  arch         : {args.arch}")
    print(f"  checkpoint   : {args.checkpoint}")
    print(f"  subject model: {args.subject_model}")
    print(f"  cache dir    : {cache_dir}")
    print(f"  layer key    : {args.layer_key}")
    print(f"  device       : {device}")

    t0 = time.time()
    spec, model, d_in, d_sae = load_model(
        args.checkpoint, args.arch, args.subject_model,
        args.k, args.T, args.expansion_factor, device,
    )
    print(f"  loaded model in {time.time() - t0:.1f}s  (d_in={d_in}, d_sae={d_sae})")

    # ---- Decoder-geometry metrics ----
    t0 = time.time()
    geom = decoder_geometry_metrics(
        spec, model, device,
        n_pca=args.n_pca, n_clusters=args.n_clusters, seed=args.seed,
    )
    print(f"  geometry metrics in {time.time() - t0:.1f}s: "
          f"silhouette={geom['silhouette']:.4f}  "
          f"entropy={geom['cluster_entropy']:.4f}")

    # ---- Activation-based auto-MI ----
    t0 = time.time()
    windows = load_activation_windows(
        cache_dir, args.layer_key, args.T,
        max_sequences=args.max_sequences,
        max_windows=args.max_windows,
        seed=args.seed,
    )
    print(f"  loaded {windows.shape[0]} windows  ({time.time() - t0:.1f}s)")

    t0 = time.time()
    feats = encode_batched(spec, model, windows, device)
    print(f"  encoded in {time.time() - t0:.1f}s  -> {tuple(feats.shape)}")

    t0 = time.time()
    mi = auto_mi_metric(feats)
    print(f"  auto-MI in {time.time() - t0:.1f}s: "
          f"scalar={mi['mean_auto_mi_scalar']:.5f}  "
          f"per_lag={[round(v, 5) for v in mi['mean_mi_per_lag']]}")

    # ---- Assemble + write JSON ----
    out = {
        "label": args.label,
        "arch": args.arch,
        "subject_model": args.subject_model,
        "cached_dataset": args.cached_dataset,
        "layer_key": args.layer_key,
        "k": args.k,
        "T": args.T,
        "expansion_factor": args.expansion_factor,
        "checkpoint": args.checkpoint,
        "n_windows": int(windows.shape[0]),
        "n_features": int(feats.shape[-1]),
        "decoder_geometry": geom,
        "auto_mi": mi,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"metrics_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
