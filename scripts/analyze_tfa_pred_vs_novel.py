"""Phase 1c: TFA pred vs novel mass split.

For each TFA feature i, compute:
  - pred_mass[i]  = sum of |pred_codes[:,:,i]|  over eval data
  - novel_mass[i] = sum of novel_codes[:,:,i]   (already non-neg after ReLU) over eval
  - pred_ratio[i] = pred_mass[i] / (pred_mass[i] + novel_mass[i])

Features with high pred_ratio are "pred-dominated" — their activity comes
mostly from context prediction, and should read as persistent contextual features.
Features with low pred_ratio are "novel-dominated" — sparse transient features.

Outputs:
    results/analysis/tfa_pred_vs_novel/
        pred_ratio_hist.png
        top_pred_dominated_features.json
        top_novel_dominated_features.json
        summary.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/elysium/temp_xc")

CKPT = "/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts/tfa_pos__gemma-2-2b-it__fineweb__resid_L25__k100__seed42.pt"
ACT = "/home/elysium/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb/resid_L25.npy"
OUT = "/home/elysium/temp_xc/results/analysis/tfa_pred_vs_novel"
D_IN = 2304
D_SAE = 18432
K = 100
N_EVAL = 1000


@torch.no_grad()
def compute_masses():
    from src.bench.architectures._tfa_module import TemporalSAE
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = TemporalSAE(
        dimin=D_IN, width=D_SAE, n_heads=4, sae_diff_type="topk", kval_topk=K,
        tied_weights=True, n_attn_layers=1, bottleneck_factor=8,
        use_pos_encoding=True, max_seq_len=512,
    ).cuda()
    model.load_state_dict(state)
    model.eval()

    arr = np.load(ACT, mmap_mode="r")
    eval_x = torch.from_numpy(np.array(arr[-N_EVAL:])).float()
    scale = math.sqrt(D_IN) / eval_x[:16].norm(dim=-1).mean().item()

    pred_mass = torch.zeros(D_SAE)
    novel_mass = torch.zeros(D_SAE)
    pred_max = torch.zeros(D_SAE)
    novel_max = torch.zeros(D_SAE)
    for s in range(0, N_EVAL, 16):
        x = (eval_x[s : s + 16].cuda() * scale)
        _, inter = model(x)
        pc = inter["pred_codes"].abs().reshape(-1, D_SAE).cpu()
        nc = inter["novel_codes"].reshape(-1, D_SAE).cpu()
        pred_mass += pc.sum(dim=0)
        novel_mass += nc.sum(dim=0)
        pred_max = torch.maximum(pred_max, pc.max(dim=0).values)
        novel_max = torch.maximum(novel_max, nc.max(dim=0).values)
    return pred_mass, novel_mass, pred_max, novel_max


def main():
    os.makedirs(OUT, exist_ok=True)

    print(f"Computing pred/novel mass on {N_EVAL} eval sequences...")
    pred_mass, novel_mass, pred_max, novel_max = compute_masses()

    # Avoid div-by-zero; features with both masses tiny are dead
    total_mass = pred_mass + novel_mass
    pred_ratio = torch.where(total_mass > 1e-6, pred_mass / total_mass, torch.tensor(-1.0))

    alive_mask = total_mass > 1e-6
    n_alive = int(alive_mask.sum())
    alive_ratios = pred_ratio[alive_mask].numpy()

    print(f"  alive features: {n_alive} / {D_SAE}")
    print(f"  pred_ratio stats (among alive):")
    print(f"    mean   = {alive_ratios.mean():.3f}")
    print(f"    median = {np.median(alive_ratios):.3f}")
    print(f"    frac pred-dominated (ratio > 0.5): {(alive_ratios > 0.5).mean():.3f}")
    print(f"    frac novel-dominated (ratio < 0.5): {(alive_ratios < 0.5).mean():.3f}")
    print(f"    frac strongly pred (> 0.8):  {(alive_ratios > 0.8).mean():.3f}")
    print(f"    frac strongly novel (< 0.2): {(alive_ratios < 0.2).mean():.3f}")

    # Top pred-dominated: features with high pred_ratio AND nontrivial mass
    # Sort by pred_ratio (descending) but filter to features with enough mass
    mass_threshold = torch.quantile(total_mass[alive_mask], 0.5)  # above-median mass
    candidates_pred = (pred_ratio > 0.5) & (total_mass > mass_threshold)
    candidates_novel = (pred_ratio < 0.5) & (total_mass > mass_threshold) & (pred_ratio >= 0)

    # Top pred-dominated by pred_mass
    pred_dom_idx = torch.where(candidates_pred)[0]
    pred_sorted = pred_dom_idx[torch.argsort(pred_mass[pred_dom_idx], descending=True)]
    # Top novel-dominated by novel_mass
    novel_dom_idx = torch.where(candidates_novel)[0]
    novel_sorted = novel_dom_idx[torch.argsort(novel_mass[novel_dom_idx], descending=True)]

    N_TOP = 50
    top_pred = []
    for i in pred_sorted[:N_TOP].tolist():
        top_pred.append({
            "feat_idx": int(i),
            "pred_ratio": float(pred_ratio[i]),
            "pred_mass": float(pred_mass[i]),
            "novel_mass": float(novel_mass[i]),
            "pred_max": float(pred_max[i]),
            "novel_max": float(novel_max[i]),
        })
    top_novel = []
    for i in novel_sorted[:N_TOP].tolist():
        top_novel.append({
            "feat_idx": int(i),
            "pred_ratio": float(pred_ratio[i]),
            "pred_mass": float(pred_mass[i]),
            "novel_mass": float(novel_mass[i]),
            "pred_max": float(pred_max[i]),
            "novel_max": float(novel_max[i]),
        })

    with open(f"{OUT}/top_pred_dominated_features.json", "w") as f:
        json.dump(top_pred, f, indent=2)
    with open(f"{OUT}/top_novel_dominated_features.json", "w") as f:
        json.dump(top_novel, f, indent=2)

    summary = {
        "n_alive": n_alive,
        "pred_ratio_mean": float(alive_ratios.mean()),
        "pred_ratio_median": float(np.median(alive_ratios)),
        "frac_pred_dominated": float((alive_ratios > 0.5).mean()),
        "frac_novel_dominated": float((alive_ratios < 0.5).mean()),
        "frac_strongly_pred_0.8": float((alive_ratios > 0.8).mean()),
        "frac_strongly_novel_0.2": float((alive_ratios < 0.2).mean()),
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTop 5 pred-dominated features:")
    for t in top_pred[:5]:
        print(f"  feat_{t['feat_idx']:5d}  pred_ratio={t['pred_ratio']:.3f}  "
              f"pred_mass={t['pred_mass']:.1f}  novel_mass={t['novel_mass']:.1f}")

    print("\nTop 5 novel-dominated features:")
    for t in top_novel[:5]:
        print(f"  feat_{t['feat_idx']:5d}  pred_ratio={t['pred_ratio']:.3f}  "
              f"pred_mass={t['pred_mass']:.1f}  novel_mass={t['novel_mass']:.1f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(alive_ratios, bins=50, color="steelblue")
    axes[0].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="50/50")
    axes[0].set_xlabel("pred_ratio = pred_mass / (pred_mass + novel_mass)")
    axes[0].set_ylabel("# features")
    axes[0].set_title(f"TFA feature pred-vs-novel mass split  (n_alive={n_alive})")
    axes[0].legend()

    # Scatter: pred_mass vs novel_mass per feature (log-log)
    pm = pred_mass[alive_mask].numpy()
    nm = novel_mass[alive_mask].numpy()
    ratios = alive_ratios
    # Clip zeros for log plot
    pm = np.maximum(pm, 1e-3)
    nm = np.maximum(nm, 1e-3)
    sc = axes[1].scatter(nm, pm, c=ratios, cmap="coolwarm", s=3, alpha=0.5, vmin=0, vmax=1)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("novel_mass")
    axes[1].set_ylabel("pred_mass")
    axes[1].set_title("Per-feature pred vs novel mass")
    plt.colorbar(sc, ax=axes[1], label="pred_ratio")
    # Diagonal y = x
    lims = [1e-2, max(pm.max(), nm.max())]
    axes[1].plot(lims, lims, "k--", alpha=0.3, label="y = x")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT}/pred_ratio_hist.png", dpi=120)
    print(f"\n  -> {OUT}/pred_ratio_hist.png")


if __name__ == "__main__":
    main()
