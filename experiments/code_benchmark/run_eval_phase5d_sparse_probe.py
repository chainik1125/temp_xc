"""Phase 5d — SAE windowed probe restricted to non-zero contributions.

User's concern: does the 81 920-d SAE-window probe win come from the probe
exploiting the zero-pattern (via per-column mean centering making absence
informative), rather than from the actually-active feature values?

Sanity tests:

  (A) No-center ridge on SAE-window latents. Zero stays literally zero;
      prediction y_hat = sum_alpha β_alpha * z_alpha; no signal from "feature
      α was NOT active".

  (B) Sparsity-filtered ridge: keep only columns (feature, position) that
      fire in at least P % of training samples. Strictly reduces the column
      pool to commonly-active features so rare features can't overfit.

  (C) Bag-of-features probe: collapse position by summing (or max-pooling)
      feature activations across the 5-position window. Shape becomes
      (n_samples, 16 384). Tests whether position matters at all.

Compares to the baseline sae_window5 (which we already have at R²=0.873 on
bracket_depth from phase 5c).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_pipeline.python_code import SubjectModelConfig, load_cache  # noqa: E402
from code_pipeline.eval_utils import (  # noqa: E402
    build_model_from_checkpoint,
    encode_sae_per_token,
    labels_for_sources,
)


FIELD = "bracket_depth"


def sweep_dual_ridge(X, y, seed, n_train, lams, center_X=True):
    mask = y >= 0
    X = X[mask]; y = y[mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_train = min(n_train, int(0.8 * X.shape[0]))
    tr = perm[:n_train]; te = perm[n_train:]
    Xtr = X[tr].astype(np.float64); ytr = y[tr].astype(np.float64)
    Xte = X[te].astype(np.float64); yte = y[te].astype(np.float64)
    n, p = Xtr.shape
    if center_X:
        x_mean = Xtr.mean(axis=0, keepdims=True)
        xc = Xtr - x_mean
        Xte_c = Xte - x_mean
    else:
        xc = Xtr
        Xte_c = Xte
    y_mean = ytr.mean()
    yc = ytr - y_mean
    sst = ((yte - yte.mean()) ** 2).sum() + 1e-12

    curve = []
    if p <= n:
        xtx = xc.T @ xc
        xty = xc.T @ yc
        I_diag = np.eye(p)
        for lam in lams:
            beta = np.linalg.solve(xtx + lam * I_diag, xty)
            pred = Xte_c @ beta + y_mean
            sse = ((yte - pred) ** 2).sum()
            curve.append({"lambda": float(lam),
                           "r2": float(1.0 - sse / sst)})
    else:
        K = xc @ xc.T
        K_test = Xte_c @ xc.T
        I_diag = np.eye(n)
        for lam in lams:
            alpha = np.linalg.solve(K + lam * I_diag, yc)
            pred = K_test @ alpha + y_mean
            sse = ((yte - pred) ** 2).sum()
            curve.append({"lambda": float(lam),
                           "r2": float(1.0 - sse / sst)})
    best = max(curve, key=lambda c: c["r2"])
    return {"best": best, "curve": curve,
            "n_train": int(tr.size), "n_test": int(te.size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--n-chunks", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=18000)
    parser.add_argument("--density-thresholds", type=float, nargs="+",
                        default=[0.001, 0.005, 0.01, 0.05])
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg.get("seed", 42))
    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])
    layers = subject_cfg.required_layers()
    cache_root = HERE / cfg.get("cache_root", "cache")
    checkpoint_root = HERE / cfg.get("checkpoint_root", "checkpoints")
    results_root = HERE / cfg.get("output_root", "results")

    tokens, sources, acts_by_layer, _ = load_cache(cache_root, layers)
    split = torch.load(cache_root / "split.pt")
    idx = split["eval_idx"][: args.n_chunks]
    sources_sub = [sources[i] for i in idx.tolist()]
    labels_nt = labels_for_sources(sources_sub)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchor_acts = acts_by_layer[subject_cfg.anchor_layer][idx]
    sae_ckpt = torch.load(checkpoint_root / "topk_sae.pt",
                           map_location="cpu", weights_only=False)
    sae_model, _ = build_model_from_checkpoint(sae_ckpt, subject_cfg.d_model)
    sae_model = sae_model.to(device).eval()
    sae_lat, _, _, _ = encode_sae_per_token(sae_model, anchor_acts, device)
    N = anchor_acts.shape[0]; T_chunk = anchor_acts.shape[1]
    d_sae = sae_lat.shape[1]
    sae_lat = sae_lat.reshape(N, T_chunk, d_sae)
    del sae_model

    T_window = 5
    chunk_idx_list, tok_idx_list = [], []
    for c in range(N):
        for t in range(T_window - 1, T_chunk):
            chunk_idx_list.append(c); tok_idx_list.append(t)
    chunk_idx = np.asarray(chunk_idx_list)
    tok_idx = np.asarray(tok_idx_list)
    y = labels_nt[FIELD][chunk_idx, tok_idx].astype(np.float32)

    per_k = [sae_lat[chunk_idx, tok_idx - (T_window - 1 - k), :]
             for k in range(T_window)]
    X_window = np.concatenate(per_k, axis=1).astype(np.float32)   # (N, 5*d_sae)
    print(f"[phase5d] X_window shape: {X_window.shape}", flush=True)

    active = X_window != 0
    col_density = active.mean(axis=0)
    print(f"[phase5d] column density stats: "
          f"mean={col_density.mean():.4f}  median={np.median(col_density):.4f}  "
          f"max={col_density.max():.4f}  #cols>0.01={(col_density > 0.01).sum()}",
          flush=True)

    lams = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
    summary = {}

    # baseline: full window, centered (matches phase 5c 'sae_window5')
    print(f"\n[phase5d] BASELINE centered full window", flush=True)
    r = sweep_dual_ridge(X_window, y, seed=seed, n_train=args.n_train,
                          lams=lams, center_X=True)
    print(f"[phase5d]   centered best λ={r['best']['lambda']:.1e}  "
          f"R²={r['best']['r2']:+.4f}", flush=True)
    summary["full_centered"] = r

    # (A) no-center ridge
    print(f"\n[phase5d] (A) no-center ridge — zero entries stay zero", flush=True)
    r = sweep_dual_ridge(X_window, y, seed=seed, n_train=args.n_train,
                          lams=lams, center_X=False)
    print(f"[phase5d]   no_center best λ={r['best']['lambda']:.1e}  "
          f"R²={r['best']['r2']:+.4f}", flush=True)
    summary["no_center"] = r

    # (B) density-filtered: keep only cols with density >= threshold
    for thr in args.density_thresholds:
        keep = col_density >= thr
        X_sub = X_window[:, keep]
        print(f"\n[phase5d] (B) density>=%.4f: {keep.sum()} cols kept out of "
              f"{col_density.size}" % thr, flush=True)
        r = sweep_dual_ridge(X_sub, y, seed=seed, n_train=args.n_train,
                              lams=lams, center_X=True)
        print(f"[phase5d]   density>=%.4f best λ={r['best']['lambda']:.1e}  "
              f"R²={r['best']['r2']:+.4f}" % thr, flush=True)
        summary[f"density_ge_{thr}"] = {
            "n_cols_kept": int(keep.sum()),
            "threshold": float(thr),
            **r,
        }

    # (C) bag-of-features: sum across positions — shape (N, d_sae)
    X_bag_sum = np.zeros((X_window.shape[0], d_sae), dtype=np.float32)
    for k in range(T_window):
        X_bag_sum += per_k[k].astype(np.float32)
    print(f"\n[phase5d] (C) bag-of-features (sum pool) shape={X_bag_sum.shape}",
          flush=True)
    r = sweep_dual_ridge(X_bag_sum, y, seed=seed, n_train=args.n_train,
                          lams=lams, center_X=True)
    print(f"[phase5d]   bag_sum best λ={r['best']['lambda']:.1e}  "
          f"R²={r['best']['r2']:+.4f}", flush=True)
    summary["bag_sum_5pos"] = r

    with (results_root / "phase5d_sparse_probe.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase5d] wrote {results_root / 'phase5d_sparse_probe.json'}")


if __name__ == "__main__":
    main()
