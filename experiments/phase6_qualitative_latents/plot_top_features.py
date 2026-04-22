"""Phase 6: top-k feature activation plots on concat-set A and B.

Mirrors Ye et al. 2025 Figure 1 (concat A) and Figure 4 (concat B):
for each arch, plot the top-8 features (ranked by activation variance)
as horizontal activation curves across the concat sequence, with
passage-boundary dashed lines.

Output under `experiments/phase6_qualitative_latents/results/top_features/`:

    <concat>__<arch>__top8.png
    <concat>__<arch>__top8.thumb.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
Z_DIR = REPO / "experiments/phase6_qualitative_latents/z_cache"
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/results/top_features"

N_TOP = 8


def plot_one(concat_name: str, arch: str, autointerp_labels: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(Z_DIR / concat_name / f"{arch}__z.npy").astype(np.float32)
    prov = json.loads((Z_DIR / concat_name / "provenance.json").read_text())
    n_tokens, d_sae = z.shape

    var = z.var(axis=0)
    top_idx = np.argsort(-var)[:N_TOP]

    fig, axes = plt.subplots(N_TOP, 1, figsize=(10, 1.2 * N_TOP),
                             sharex=True, dpi=120)
    for row, fi in enumerate(top_idx):
        ax = axes[row]
        ax.plot(z[:, fi], color="tab:blue", linewidth=0.9)
        ax.set_ylabel(f"#{fi}", rotation=0, ha="right", va="center",
                      fontsize=7, labelpad=22)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlim(0, n_tokens)

        for seg in prov["provenance"][1:]:  # skip first boundary (t=0)
            ax.axvline(seg["start"], color="gray", linestyle="--",
                       alpha=0.5, linewidth=0.7)

        if autointerp_labels and str(fi) in autointerp_labels:
            ax.set_title(autointerp_labels[str(fi)],
                         fontsize=7, loc="left", color="dimgray")

    # annotate passage names under bottom axis
    axes[-1].set_xticks(
        [(p["start"] + p["end"]) // 2 for p in prov["provenance"]],
        [p["label"] for p in prov["provenance"]], fontsize=7,
    )
    fig.suptitle(f"{arch} — top-{N_TOP} features by variance — {concat_name}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUT_DIR / f"{concat_name}__{arch}__top{N_TOP}.png"
    fig.savefig(dst, dpi=120, bbox_inches="tight")
    fig.savefig(dst.with_name(dst.stem + ".thumb.png"),
                dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {dst}")


def load_autointerp_labels(arch: str) -> dict:
    """Return dict of feature_idx_str -> label, if autointerp ran first."""
    p = REPO / f"experiments/phase6_qualitative_latents/results/autointerp/{arch}__labels.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())
    return {str(r["feature_idx"]): r["label"] for r in rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+",
                   default=["agentic_txc_02", "agentic_mlc_08",
                            "tsae_paper", "tsae_ours", "tfa_big"])
    p.add_argument("--concats", type=str, nargs="+",
                   default=["concat_A", "concat_B"])
    args = p.parse_args()

    for concat in args.concats:
        for arch in args.archs:
            z_path = Z_DIR / concat / f"{arch}__z.npy"
            if not z_path.exists():
                print(f"skip {arch}/{concat}: no z cache")
                continue
            labels = load_autointerp_labels(arch)
            plot_one(concat, arch, autointerp_labels=labels)


if __name__ == "__main__":
    main()
