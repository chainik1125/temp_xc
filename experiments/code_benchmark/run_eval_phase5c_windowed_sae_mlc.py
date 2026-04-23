"""Phase 5c — SAE and MLC probes trained on a temporal window of latents.

Phase 2 probed the SAE at a single position and the TXC on a 5-position
window's shared latent. The fair question the user asked:

    If I give SAE/MLC a temporal window at the *probe* level
    (concatenate per-token latents across t-4..t into a 5×d_sae vector),
    does the SAE/MLC probe match or beat TXC?

If yes → TXC's Phase-2 advantage is "the encoder sees 5 positions" and a
downstream probe can compose single-position encodings to recover it.
If no → TXC's encoder is doing something probe-level concatenation cannot.

Output: results/phase5c_windowed_sae_mlc.json
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


def ridge_r2_solve(xtx, xty, x_mean, y_mean, Xte, yte, lam):
    I = np.eye(xtx.shape[0])
    beta = np.linalg.solve(xtx + lam * I, xty)
    pred = (Xte - x_mean) @ beta + y_mean
    sse = ((yte - pred) ** 2).sum()
    sst = ((yte - yte.mean()) ** 2).sum() + 1e-12
    return float(1.0 - sse / sst)


def sweep_lambdas(X, y, seed, n_train, lams):
    mask = y >= 0
    X = X[mask]; y = y[mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_train = min(n_train, int(0.8 * X.shape[0]))
    tr = perm[:n_train]; te = perm[n_train:]
    Xtr = X[tr].astype(np.float64); ytr = y[tr].astype(np.float64)
    Xte = X[te].astype(np.float64); yte = y[te].astype(np.float64)
    x_mean = Xtr.mean(axis=0, keepdims=True)
    y_mean = ytr.mean()
    xc = Xtr - x_mean
    yc = ytr - y_mean
    xtx = xc.T @ xc
    xty = xc.T @ yc
    curve = [{"lambda": float(l),
               "r2": ridge_r2_solve(xtx, xty, x_mean, y_mean, Xte, yte, l)}
             for l in lams]
    best = max(curve, key=lambda c: c["r2"])
    return {"best": best, "curve": curve,
            "n_train": int(tr.size), "n_test": int(te.size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--n-chunks", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=20000)
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

    # ---- encode SAE and MLC per-token latents across the eval chunks ----
    sae_ckpt = torch.load(checkpoint_root / "topk_sae.pt",
                           map_location="cpu", weights_only=False)
    mlc_ckpt = torch.load(checkpoint_root / "mlc_l5.pt",
                           map_location="cpu", weights_only=False)
    sae_model, _ = build_model_from_checkpoint(sae_ckpt, subject_cfg.d_model)
    mlc_model, _ = build_model_from_checkpoint(mlc_ckpt, subject_cfg.d_model)
    sae_model = sae_model.to(device).eval()
    mlc_model = mlc_model.to(device).eval()

    # SAE: (N*T, d_sae) then reshape to (N, T, d_sae)
    sae_lat, _, sae_c, sae_t = encode_sae_per_token(sae_model, anchor_acts, device)
    N = anchor_acts.shape[0]
    T_chunk = anchor_acts.shape[1]
    d_sae = sae_lat.shape[1]
    sae_lat = sae_lat.reshape(N, T_chunk, d_sae)
    print(f"[phase5c] SAE latents shape: {sae_lat.shape}", flush=True)

    mlc_lat, _, mlc_c, mlc_t = encode_mlc_per_token(
        mlc_model, mlc_acts, subject_cfg.mlc_layers, device)
    mlc_lat = mlc_lat.reshape(N, T_chunk, d_sae)
    print(f"[phase5c] MLC latents shape: {mlc_lat.shape}", flush=True)

    del sae_model, mlc_model

    # ---- build probe index: (chunk, t) for t >= T_window-1 ----
    T_window = 5
    chunk_idx_list, tok_idx_list = [], []
    for c in range(N):
        for t in range(T_window - 1, T_chunk):
            chunk_idx_list.append(c)
            tok_idx_list.append(t)
    chunk_idx = np.asarray(chunk_idx_list)
    tok_idx = np.asarray(tok_idx_list)
    print(f"[phase5c] {chunk_idx.size} probe samples", flush=True)

    # ---- assemble per-sample windowed-latent concatenations ----
    # X_sae_window: (n_samples, 5 * d_sae) by concatenating sae_lat over 5 positions.
    # Large: 5 * 16384 = 81920 dims. That's a lot — be careful with dtype.
    def make_windowed(lat_arr: np.ndarray) -> np.ndarray:
        per_k = [lat_arr[chunk_idx, tok_idx - (T_window - 1 - k), :]
                 for k in range(T_window)]
        return np.concatenate(per_k, axis=1).astype(np.float32)

    X_sae_single = sae_lat[chunk_idx, tok_idx, :].astype(np.float32)
    X_mlc_single = mlc_lat[chunk_idx, tok_idx, :].astype(np.float32)
    X_sae_window = make_windowed(sae_lat)
    X_mlc_window = make_windowed(mlc_lat)

    print(f"[phase5c] X_sae_single {X_sae_single.shape}  "
          f"X_sae_window {X_sae_window.shape}", flush=True)
    print(f"[phase5c] X_mlc_single {X_mlc_single.shape}  "
          f"X_mlc_window {X_mlc_window.shape}", flush=True)

    lams = [1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
    inputs = [
        ("sae_single", X_sae_single),
        ("sae_window5", X_sae_window),
        ("mlc_single", X_mlc_single),
        ("mlc_window5", X_mlc_window),
    ]
    summary: dict = {}
    for field in CONTINUOUS:
        y = labels_nt[field][chunk_idx, tok_idx].astype(np.float32)
        by_input = {}
        for name, X in inputs:
            print(f"[phase5c] field={field}  input={name}  shape={X.shape}",
                  flush=True)
            r = sweep_lambdas(X, y, seed=seed, n_train=args.n_train, lams=lams)
            by_input[name] = r
            best = r["best"]
            print(f"[phase5c]   best λ={best['lambda']:.1e}  R²={best['r2']:+.4f}",
                  flush=True)
        summary[field] = by_input

    with (results_root / "phase5c_windowed_sae_mlc.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase5c] wrote {results_root / 'phase5c_windowed_sae_mlc.json'}")


if __name__ == "__main__":
    main()
