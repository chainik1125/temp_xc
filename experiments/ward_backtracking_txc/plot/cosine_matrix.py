"""Cosine matrix of top-K TXC features × Stage A DoM vectors."""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.ward_backtracking_txc.plot._common import (
    load_cfg, plots_dir, features_npz_path, iter_arch_hookpoint,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8: return 0.0
    return float((a @ b) / (na * nb))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    out_dir = plots_dir(cfg)

    dom_path = Path(cfg["paths"]["stageA_dom"])
    if not dom_path.exists():
        print(f"[skip] no DoM at {dom_path}"); return
    dom = torch.load(dom_path, weights_only=False)
    dom_vecs = {
        "DoM_base_union": dom["base"]["union"].numpy(),
        "DoM_reasoning_union": dom["reasoning"]["union"].numpy(),
    }

    rows: list[tuple[str, np.ndarray]] = []
    for arch, hp in iter_arch_hookpoint(cfg):
        path = features_npz_path(cfg, arch, hp["key"])
        if not path.exists(): continue
        z = np.load(path, allow_pickle=True)
        for fid, vec in zip(z["top_features"][:8], z["decoder_at_pos0"][:8]):
            rows.append((f"{arch}/{hp['key']}/f{int(fid)}@pos0", vec))
        # Skip "@union" duplicate when arch has no T axis (pos0 == union).
        if arch in ("topk_sae", "tsae"):
            continue
        for fid, vec in zip(z["top_features"][:8], z["decoder_union"][:8]):
            rows.append((f"{arch}/{hp['key']}/f{int(fid)}@union", vec))

    if not rows:
        print("[skip] no feature files yet"); return

    M = np.zeros((len(rows), len(dom_vecs)))
    for i, (_, v) in enumerate(rows):
        for j, (_, dv) in enumerate(dom_vecs.items()):
            M[i, j] = _cos(v, dv)

    fig, ax = plt.subplots(figsize=(4.5, max(4, len(rows) * 0.18)))
    v = np.abs(M).max() or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(dom_vecs)))
    ax.set_xticklabels(list(dom_vecs.keys()), rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    fig.colorbar(im, ax=ax, label="cosine")
    ax.set_title("Cos(dictionary features, Stage A DoM) — across architectures")
    fig.tight_layout()
    out = out_dir / "cosine_matrix.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
