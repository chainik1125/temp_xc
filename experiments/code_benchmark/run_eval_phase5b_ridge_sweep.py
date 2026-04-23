"""Phase 5b — ridge-λ sweep for the raw-residual baseline probes.

Phase 5 v1 used ridge=1.0 for all three raw inputs, which is almost certainly
under-regularized for the 11 520-dim temporal-window and layer-stack inputs
(n_train=20 000, so input:train ratio ≈ 0.58 — anything past λ=1 helps a lot).

This script sweeps λ ∈ {10^-3 .. 10^4} across the three raw inputs on the same
four continuous fields as Phase 5, and reports best-λ R² per (input, field).
Compare these to the SAE / TXC / MLC probe R²s from Phase 2.

Output: results/phase5b_ridge_sweep.json
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
from code_pipeline.eval_utils import labels_for_sources  # noqa: E402


CONTINUOUS = ["bracket_depth", "indent_spaces", "scope_nesting", "distance_to_header"]


def ridge_r2(X_train, y_train, X_test, y_test, ridge=1e-3):
    x_mean = X_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean()
    xc = (X_train - x_mean).astype(np.float64)
    yc = (y_train - y_mean).astype(np.float64)
    xtx = xc.T @ xc
    xty = xc.T @ yc
    xtx.flat[:: xtx.shape[0] + 1] += ridge
    beta = np.linalg.solve(xtx, xty)
    pred = (X_test - x_mean) @ beta + y_mean
    sse = ((y_test - pred) ** 2).sum()
    sst = ((y_test - y_test.mean()) ** 2).sum() + 1e-12
    return float(1.0 - sse / sst)


def sweep_one(X: np.ndarray, y: np.ndarray, seed: int, n_train: int,
              lams: list[float]) -> dict:
    mask = y >= 0
    X = X[mask]; y = y[mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_train = min(n_train, int(0.8 * X.shape[0]))
    tr = perm[:n_train]; te = perm[n_train:]
    # Precompute shared quantities so we aren't re-centering per λ.
    Xtr = X[tr].astype(np.float64)
    ytr = y[tr].astype(np.float64)
    Xte = X[te].astype(np.float64)
    yte = y[te].astype(np.float64)
    x_mean = Xtr.mean(axis=0, keepdims=True)
    y_mean = ytr.mean()
    xc = Xtr - x_mean
    yc = ytr - y_mean
    xtx = xc.T @ xc
    xty = xc.T @ yc
    I_diag = np.eye(xtx.shape[0])
    sst = ((yte - yte.mean()) ** 2).sum() + 1e-12
    results = []
    for lam in lams:
        beta = np.linalg.solve(xtx + lam * I_diag, xty)
        pred = (Xte - x_mean) @ beta + y_mean
        sse = ((yte - pred) ** 2).sum()
        results.append({"lambda": float(lam), "r2": float(1.0 - sse / sst)})
    best = max(results, key=lambda r: r["r2"])
    return {"best": best, "curve": results,
            "n_train": int(tr.size), "n_test": int(te.size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--n-chunks", type=int, default=300)
    parser.add_argument("--n-train", type=int, default=20000)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = int(cfg.get("seed", 42))
    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])
    layers = subject_cfg.required_layers()

    cache_root = HERE / cfg.get("cache_root", "cache")
    results_root = HERE / cfg.get("output_root", "results")
    tokens, sources, acts_by_layer, _ = load_cache(cache_root, layers)
    split = torch.load(cache_root / "split.pt")
    idx = split["eval_idx"][: args.n_chunks]
    sources_sub = [sources[i] for i in idx.tolist()]
    labels_nt = labels_for_sources(sources_sub)

    anchor_L = subject_cfg.anchor_layer
    T_window = 5
    T_chunk = tokens.shape[1]
    chunk_idx_list, tok_idx_list = [], []
    for c in range(len(idx)):
        for t in range(T_window - 1, T_chunk):
            chunk_idx_list.append(c); tok_idx_list.append(t)
    chunk_idx = np.asarray(chunk_idx_list)
    tok_idx = np.asarray(tok_idx_list)
    print(f"[phase5b] {chunk_idx.size} samples", flush=True)

    d = subject_cfg.d_model
    anchor = acts_by_layer[anchor_L][idx].float().numpy()
    X_single = anchor[chunk_idx, tok_idx, :]
    X_temp = np.stack(
        [anchor[chunk_idx, tok_idx - (T_window - 1 - k), :] for k in range(T_window)],
        axis=1,
    ).reshape(chunk_idx.size, T_window * d)
    X_layer = np.stack(
        [acts_by_layer[L][idx].float().numpy()[chunk_idx, tok_idx, :]
         for L in subject_cfg.mlc_layers],
        axis=1,
    ).reshape(chunk_idx.size, len(subject_cfg.mlc_layers) * d)

    # Reasonable grid spanning 7 orders of magnitude
    lams = [10 ** p for p in range(-3, 5)]            # 1e-3 .. 1e4
    inputs = [("single_pos", X_single),
              ("temporal_window", X_temp),
              ("layer_stack", X_layer)]
    summary: dict = {}
    for field in CONTINUOUS:
        y = labels_nt[field][chunk_idx, tok_idx].astype(np.float32)
        by_input = {}
        for name, X in inputs:
            print(f"[phase5b] field={field} input={name} "
                  f"shape={X.shape}", flush=True)
            by_input[name] = sweep_one(X, y, seed=seed,
                                         n_train=args.n_train, lams=lams)
            best = by_input[name]["best"]
            print(f"[phase5b]   best λ={best['lambda']:.1e}  R²={best['r2']:+.4f}",
                  flush=True)
        summary[field] = by_input

    with (results_root / "phase5b_ridge_sweep.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase5b] wrote {results_root / 'phase5b_ridge_sweep.json'}")


if __name__ == "__main__":
    main()
