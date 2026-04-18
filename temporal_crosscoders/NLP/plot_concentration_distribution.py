"""Per-feature temporal concentration distribution across architectures.

Shows the full distribution of per-feature concentration (max / sum over
T positions) for every arch at every layer. Concentration = 1/T (=0.2
for T=5) means the feature spreads perfectly across all window positions;
concentration = 1 means the feature fires at exactly one position.

Metric is architecture-faithful:
- TXCDR: decoder-based concentration from ||W_dec[f, t, :]||. Defined
  for every feature, dead or alive.
- TFA novel / TFA pred / Stacked: activation-based concentration from
  the per-position mean of |feat_acts[f, t]| across the eval sample.
  Defined only for "alive" features (mass > 0).

The apples-to-apples thing these metrics share: "across the T window
positions, where does this feature place its reconstruction
contribution?" Both forms answer that in the architecture's native
representation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from temporal_crosscoders.NLP.autointerp import load_model


ARCHS_ORDER = ["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"]
ARCH_LABELS = {
    "stacked_sae": "Stacked SAE\n(per-pos SAE)",
    "crosscoder": "TXCDR\n(per-pos decoder)",
    "tfa_pos": "TFA novel\n(topk per token)",
    "tfa_pos_pred": "TFA pred\n(dense pred_codes)",
}
ARCH_SHORT = {
    "stacked_sae": "Stacked",
    "crosscoder": "TXCDR",
    "tfa_pos": "TFA novel",
    "tfa_pos_pred": "TFA pred",
}
ARCH_COLORS = {
    "stacked_sae": "#1f77b4",
    "crosscoder": "#2ca02c",
    "tfa_pos": "#d62728",
    "tfa_pos_pred": "#ff7f0e",
}


def activation_concentration(scan_dir: Path, arch: str, layer: str, k: int):
    """Per-feature activation-based concentration for every alive feature."""
    p = scan_dir / f"span_all__{arch}__{layer}__k{k}.json"
    d = json.load(open(p))
    return np.array([r["concentration"] for r in d["features"].values()])


def decoder_concentration(ckpt_dir: Path, layer: str, k: int):
    """TXCDR decoder-based concentration for all 18432 features."""
    ckpt = ckpt_dir / f"crosscoder__gemma-2-2b-it__fineweb__{layer}__k{k}__seed42.pt"
    m = load_model(ckpt_path=str(ckpt), model_type="crosscoder",
                   subject_model="gemma-2-2b-it", k=k, T=5).cpu()
    with torch.no_grad():
        W = m.W_dec.data.float()
        e = W.norm(dim=-1)
        total = e.sum(dim=-1).clamp(min=1e-12)
        conc = (e.max(dim=-1).values / total).numpy()
    del m
    return conc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default="results/nlp_sweep/gemma/scans")
    ap.add_argument("--ckpt-dir", default="results/nlp_sweep/gemma/ckpts")
    ap.add_argument("--out-dir", default="results/nlp_sweep/gemma/figures")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    ckpt_dir = Path(args.ckpt_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    data: dict[str, dict[str, np.ndarray]] = {}
    for layer in ["resid_L25", "resid_L13"]:
        data[layer] = {}
        for arch in ARCHS_ORDER:
            if arch == "crosscoder":
                concs = decoder_concentration(ckpt_dir, layer, args.k)
            else:
                concs = activation_concentration(scan_dir, arch, layer, args.k)
            data[layer][arch] = concs
            print(f"  {layer} {arch}: n={len(concs)} mean={concs.mean():.3f} "
                  f"median={np.median(concs):.3f} <0.35: {(concs < 0.35).mean()*100:.1f}%")

    # ── Figure ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.22,
                          width_ratios=[1.4, 1.0])

    for row, layer in enumerate(["resid_L25", "resid_L13"]):
        # Left panel: overlaid histograms
        ax = fig.add_subplot(gs[row, 0])
        bins = np.linspace(0.2, 1.0, 41)
        for arch in ARCHS_ORDER:
            concs = data[layer][arch]
            ax.hist(concs, bins=bins, color=ARCH_COLORS[arch], alpha=0.45,
                    label=f"{ARCH_SHORT[arch]} (mean={concs.mean():.2f}, "
                          f"n={len(concs):,})",
                    density=True, edgecolor="white", linewidth=0.3)
        ax.axvline(0.2, color="gray", linestyle=":", alpha=0.8, lw=1)
        ax.text(0.202, ax.get_ylim()[1] * 0.93, "1/T=0.2\n(uniform)",
                fontsize=8, color="gray", va="top")
        ax.axvline(0.35, color="black", linestyle="--", alpha=0.6, lw=1)
        ax.text(0.352, ax.get_ylim()[1] * 0.83, "high-span\nthreshold",
                fontsize=8, color="black", va="top")
        ax.set_xlabel("Per-feature temporal concentration\n"
                      "(max position / sum of positions)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Concentration distribution — {layer}   "
                     "(lower = more spread = more 'temporal-span' behaviour)",
                     fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.19, 1.01)

        # Right panel: mean + fraction-below-0.35 bar chart
        ax = fig.add_subplot(gs[row, 1])
        names = [ARCH_SHORT[a] for a in ARCHS_ORDER]
        means = [data[layer][a].mean() for a in ARCHS_ORDER]
        medians = [np.median(data[layer][a]) for a in ARCHS_ORDER]
        colors = [ARCH_COLORS[a] for a in ARCHS_ORDER]

        xs = np.arange(len(names))
        w = 0.38
        bars_mean = ax.bar(xs - w/2, means, w, color=colors, alpha=0.9,
                           edgecolor="black", label="mean")
        bars_med = ax.bar(xs + w/2, medians, w, color=colors, alpha=0.55,
                          edgecolor="black", label="median", hatch="///")
        for b, v in zip(bars_mean, means):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=8)
        for b, v in zip(bars_med, medians):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(0.2, color="gray", linestyle=":", alpha=0.8, lw=1)
        ax.axhline(0.35, color="black", linestyle="--", alpha=0.6, lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=9, rotation=10, ha="right")
        ax.set_ylabel("Concentration  (1/T=0.2 uniform)", fontsize=11)
        ax.set_title(f"Mean / median concentration — {layer}", fontsize=11)
        ax.set_ylim(0, 1.1)
        if row == 0:
            ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Temporal concentration per feature, per architecture — "
        "TXCDR has the broadest per-position spread (near-uniform at 1/T); "
        "Stacked SAE is highly concentrated at one position by construction",
        fontsize=13, y=0.995,
    )

    base = os.path.join(args.out_dir, "concentration_distribution")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    im = Image.open(base + ".png")
    if im.width > 1400:
        ratio = 1400 / im.width
        im.resize((1400, int(im.height * ratio)), Image.LANCZOS).save(
            base + ".doc.png", optimize=True)
    else:
        im.save(base + ".doc.png", optimize=True)
    if im.width > 480:
        ratio = 480 / im.width
        im.resize((480, int(im.height * ratio)), Image.LANCZOS).save(
            base + ".thumb.png", optimize=True)
    else:
        im.save(base + ".thumb.png", optimize=True)
    plt.close(fig)
    print(f"wrote {base}.png (+.doc.png +.thumb.png)")


if __name__ == "__main__":
    main()
