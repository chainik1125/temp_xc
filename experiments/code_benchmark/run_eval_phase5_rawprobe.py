"""Phase 5 — raw-residual dense-probe control for Phase 2.

Question: is TXC's probe win on bracket_depth etc. "just" that its encoder
sees 5×d input while the SAE sees 1×d? A dense ridge probe on the raw
concatenated residuals tests this directly — no SAE in the loop.

For each label field we fit three probes:

    A. single-position:  x_t           → y       (2304-dim input)  -- SAE baseline
    B. temporal window:  concat_t(x_{t-4..t})    (11520-dim input) -- TXC baseline
    C. layer stack:      concat_L(x_t,L∈{10..14}) (11520-dim input) -- MLC baseline

Reports R² (continuous) and AUC (categorical) for each. If B ≈ TXC's SAE
probe and C ≈ MLC's SAE probe, the SAE basis contributes nothing above raw
window access — TXC's Phase-2 win was "large-window".

Output: results/phase5_rawprobe_summary.json
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
    FIELD_NAMES,
    gather_labels,
    labels_for_sources,
)


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


def run_probe(X: np.ndarray, y: np.ndarray, seed: int, ridge: float, n_train: int) -> dict:
    mask = y >= 0
    X = X[mask]; y = y[mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    n_train = min(n_train, int(0.8 * X.shape[0]))
    tr = perm[:n_train]; te = perm[n_train:]
    r2 = ridge_r2(X[tr], y[tr], X[te], y[te], ridge=ridge)
    return {"metric": "r2", "value": r2,
            "n_train": int(tr.size), "n_test": int(te.size)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--n-chunks", type=int, default=500,
                        help="chunks to assemble into probe data")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--ridge", type=float, default=1.0)
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
    T_window = 5                                               # match TXC T=5
    T_chunk = tokens.shape[1]

    # Position grid for probe samples — each (chunk, t) where t >= T_window-1
    # so that both temporal-window and single-position probes see valid data.
    chunk_idx_list = []
    tok_idx_list = []
    for c in range(len(idx)):
        for t in range(T_window - 1, T_chunk):
            chunk_idx_list.append(c)
            tok_idx_list.append(t)
    chunk_idx = np.asarray(chunk_idx_list)
    tok_idx = np.asarray(tok_idx_list)
    print(f"[phase5] {chunk_idx.size} probe samples across {len(idx)} chunks",
          flush=True)

    # ---- assemble the three input matrices ----
    d = subject_cfg.d_model

    # A. single-position (anchor layer)
    anchor = acts_by_layer[anchor_L][idx].float().numpy()      # (C, T, d)
    X_single = anchor[chunk_idx, tok_idx, :]                   # (N, d)

    # B. temporal window at anchor layer
    # X_temp[i] = concat(anchor[chunk_i, t_i-4:t_i+1]) flattened (5*d,)
    X_temp = np.stack(
        [anchor[chunk_idx, tok_idx - (T_window - 1 - k), :] for k in range(T_window)],
        axis=1,
    ).reshape(chunk_idx.size, T_window * d)                    # (N, 5*d)

    # C. layer stack at one position
    # X_layer[i] = concat(x_t across layers 10..14) flattened (5*d,)
    X_layer = np.stack(
        [acts_by_layer[L][idx].float().numpy()[chunk_idx, tok_idx, :]
         for L in subject_cfg.mlc_layers],
        axis=1,
    ).reshape(chunk_idx.size, len(subject_cfg.mlc_layers) * d)   # (N, 5*d)

    print(f"[phase5] X_single {X_single.shape}  X_temp {X_temp.shape}  "
          f"X_layer {X_layer.shape}", flush=True)

    # ---- probe per field per input ----
    summary: dict = {}
    for field in CONTINUOUS:
        y = labels_nt[field][chunk_idx, tok_idx].astype(np.float32)
        res = {}
        for name, X in [("single_pos", X_single),
                        ("temporal_window", X_temp),
                        ("layer_stack", X_layer)]:
            r = run_probe(X, y, seed=seed, ridge=args.ridge,
                          n_train=args.n_train)
            res[name] = r
            print(f"[phase5] {field:>20s}  input={name:>16s}  R²={r['value']:+.4f}  "
                  f"n_train={r['n_train']}  n_test={r['n_test']}", flush=True)
        summary[field] = res

    with (results_root / "phase5_rawprobe_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[phase5] wrote {results_root / 'phase5_rawprobe_summary.json'}")


if __name__ == "__main__":
    main()
