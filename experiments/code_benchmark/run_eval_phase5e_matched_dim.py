"""Phase 5e — SAE-window probe dim-matched to raw-window probe.

Raw temporal-window probe is 11 520-dim (5 × d_model = 5 × 2304). The full
SAE-window probe is 81 920-dim (5 × d_sae = 5 × 16 384). This script prunes
the SAE-window columns to exactly 11 520, keeping the top by training-set
density, and re-runs the ridge sweep per field.

If SAE-window-11520 still beats raw-temporal-window at ~same dimensionality,
the SAE's basis genuinely decodes bracket-depth better than raw residuals.
If they tie, the SAE win was dimensionality.

Output: results/phase5e_matched_dim.json
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
    encode_mlc_per_token,
    labels_for_sources,
)


CONTINUOUS = ["bracket_depth", "indent_spaces", "scope_nesting", "distance_to_header"]
RAW_DIM_TARGET = 11_520


def sweep_dual_ridge(X, y, seed, n_train, lams):
    mask = y >= 0
    X = X[mask]; y = y[mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_train = min(n_train, int(0.8 * X.shape[0]))
    tr = perm[:n_train]; te = perm[n_train:]
    Xtr = X[tr].astype(np.float64); ytr = y[tr].astype(np.float64)
    Xte = X[te].astype(np.float64); yte = y[te].astype(np.float64)
    n, p = Xtr.shape
    x_mean = Xtr.mean(axis=0, keepdims=True)
    y_mean = ytr.mean()
    xc = Xtr - x_mean
    yc = ytr - y_mean
    Xte_c = Xte - x_mean
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
            curve.append({"lambda": float(lam), "r2": float(1.0 - sse / sst)})
    else:
        K = xc @ xc.T
        K_test = Xte_c @ xc.T
        I_diag = np.eye(n)
        for lam in lams:
            alpha = np.linalg.solve(K + lam * I_diag, yc)
            pred = K_test @ alpha + y_mean
            sse = ((yte - pred) ** 2).sum()
            curve.append({"lambda": float(lam), "r2": float(1.0 - sse / sst)})
    best = max(curve, key=lambda c: c["r2"])
    return {"best": best, "curve": curve,
            "n_train": int(tr.size), "n_test": int(te.size)}


def build_windowed(lat_arr: np.ndarray, chunk_idx, tok_idx, T_window) -> np.ndarray:
    parts = [lat_arr[chunk_idx, tok_idx - (T_window - 1 - k), :]
             for k in range(T_window)]
    return np.concatenate(parts, axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--n-chunks", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=18000)
    parser.add_argument("--target-dim", type=int, default=RAW_DIM_TARGET)
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
    mlc_acts = {L: acts_by_layer[L][idx] for L in subject_cfg.mlc_layers}
    sae_ckpt = torch.load(checkpoint_root / "topk_sae.pt",
                           map_location="cpu", weights_only=False)
    mlc_ckpt = torch.load(checkpoint_root / "mlc_l5.pt",
                           map_location="cpu", weights_only=False)
    sae_model, _ = build_model_from_checkpoint(sae_ckpt, subject_cfg.d_model)
    mlc_model, _ = build_model_from_checkpoint(mlc_ckpt, subject_cfg.d_model)
    sae_model = sae_model.to(device).eval()
    mlc_model = mlc_model.to(device).eval()

    sae_lat, _, _, _ = encode_sae_per_token(sae_model, anchor_acts, device)
    N = anchor_acts.shape[0]; T_chunk = anchor_acts.shape[1]
    d_sae = sae_lat.shape[1]
    sae_lat = sae_lat.reshape(N, T_chunk, d_sae)

    mlc_lat, _, _, _ = encode_mlc_per_token(
        mlc_model, mlc_acts, subject_cfg.mlc_layers, device)
    mlc_lat = mlc_lat.reshape(N, T_chunk, d_sae)
    del sae_model, mlc_model

    T_window = 5
    chunk_idx_list, tok_idx_list = [], []
    for c in range(N):
        for t in range(T_window - 1, T_chunk):
            chunk_idx_list.append(c); tok_idx_list.append(t)
    chunk_idx = np.asarray(chunk_idx_list)
    tok_idx = np.asarray(tok_idx_list)

    X_sae_window = build_windowed(sae_lat, chunk_idx, tok_idx, T_window)
    X_mlc_window = build_windowed(mlc_lat, chunk_idx, tok_idx, T_window)
    print(f"[phase5e] SAE window {X_sae_window.shape}  "
          f"MLC window {X_mlc_window.shape}", flush=True)

    # Density ranking (computed once across full set of samples)
    dens_sae = (X_sae_window != 0).mean(axis=0)
    dens_mlc = (X_mlc_window != 0).mean(axis=0)

    K = args.target_dim
    sae_topk_idx = np.argsort(-dens_sae)[:K]
    mlc_topk_idx = np.argsort(-dens_mlc)[:K]
    X_sae_matched = X_sae_window[:, sae_topk_idx]
    X_mlc_matched = X_mlc_window[:, mlc_topk_idx]
    print(f"[phase5e] SAE matched to {K} cols (top-density), dims {X_sae_matched.shape}",
          flush=True)
    print(f"[phase5e] MLC matched to {K} cols (top-density), dims {X_mlc_matched.shape}",
          flush=True)

    lams = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
    summary = {}
    for field in CONTINUOUS:
        y = labels_nt[field][chunk_idx, tok_idx].astype(np.float32)
        by_input = {}
        for name, X in [("sae_window_top" + str(K), X_sae_matched),
                         ("mlc_window_top" + str(K), X_mlc_matched)]:
            print(f"\n[phase5e] field={field}  input={name}  shape={X.shape}",
                  flush=True)
            r = sweep_dual_ridge(X, y, seed=seed, n_train=args.n_train, lams=lams)
            by_input[name] = r
            print(f"[phase5e]   best λ={r['best']['lambda']:.1e}  "
                  f"R²={r['best']['r2']:+.4f}", flush=True)
        summary[field] = by_input

    with (results_root / "phase5e_matched_dim.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase5e] wrote {results_root / 'phase5e_matched_dim.json'}")


if __name__ == "__main__":
    main()
