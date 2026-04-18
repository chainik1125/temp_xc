"""Phase 1a: Decoder-direction alignment across Stacked / TXCDR / TFA-pos.

For the same (layer, k) trio, extract decoder directions from each
architecture, compute pairwise cosine similarity, and identify features
that are "unique" to each architecture (low max cosine with the others).

Inputs: Gemma-2-2B-IT resid_L25 k=100 checkpoints from the main sweep.
Outputs:
    results/analysis/decoder_alignment/
        best_match_scatter.png
        best_match_hist.png
        per_arch_top_unique.json
        summary.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/elysium/temp_xc")

# Direct state_dict inspection; no model-class imports needed.


CKPT_DIR = "/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts"
OUT_DIR = "/home/elysium/temp_xc/results/analysis/decoder_alignment"
D_IN = 2304
D_SAE = 18432
T = 5
K = 100
LAYER = "resid_L25"


def load_stacked() -> torch.Tensor:
    """Return (d_sae, d_in) decoder directions, averaged over T positions."""
    state = torch.load(
        f"{CKPT_DIR}/stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    # Stacked: 5 per-position SAEs, each with W_dec (d_in, d_sae)
    W_dec_list = [state[f"saes.{t}.W_dec"] for t in range(T)]
    W_dec = torch.stack(W_dec_list).mean(dim=0)  # (d_in, d_sae)
    return W_dec.T  # -> (d_sae, d_in)


def load_txcdr() -> torch.Tensor:
    """Return (d_sae, d_in) decoder directions, averaged over T positions."""
    state = torch.load(
        f"{CKPT_DIR}/crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    # W_dec: (d_sae, T, d_in)
    W_dec = state["W_dec"].mean(dim=1)  # -> (d_sae, d_in)
    return W_dec


def load_tfa() -> torch.Tensor:
    """Return (d_sae, d_in) decoder directions."""
    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    # D: (width, dimin) = (d_sae, d_in)
    return state["D"]


def unit_norm(x: torch.Tensor) -> torch.Tensor:
    """Normalize rows to unit norm."""
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norms


def best_match(a: torch.Tensor, b: torch.Tensor, batch: int = 512) -> torch.Tensor:
    """For each row in a, return max |cos(a_i, b_j)| over all j.

    Computes in chunks to avoid materializing the full (d_sae, d_sae) matrix.
    """
    a = unit_norm(a).float().cuda()
    b = unit_norm(b).float().cuda()
    n = a.shape[0]
    out = torch.zeros(n, device="cuda")
    for s in range(0, n, batch):
        chunk = a[s : s + batch]
        sims = chunk @ b.T  # (batch, d_sae)
        out[s : s + batch] = sims.abs().max(dim=-1).values
    return out.cpu()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading decoder directions...")
    D_stacked = load_stacked()
    D_txcdr = load_txcdr()
    D_tfa = load_tfa()

    print(f"  Stacked:  {list(D_stacked.shape)}   mean_norm={D_stacked.norm(dim=-1).mean():.3f}")
    print(f"  TXCDR:    {list(D_txcdr.shape)}   mean_norm={D_txcdr.norm(dim=-1).mean():.3f}")
    print(f"  TFA-pos:  {list(D_tfa.shape)}   mean_norm={D_tfa.norm(dim=-1).mean():.3f}")

    # Fail fast if shapes don't match
    assert D_stacked.shape == D_txcdr.shape == D_tfa.shape == (D_SAE, D_IN), "shape mismatch"

    print("\nComputing cross-architecture best-match cosine sim...")
    # For each feature in arch X, find its best match in arch Y
    best = {
        "tfa_vs_txcdr":    best_match(D_tfa, D_txcdr),
        "tfa_vs_stacked":  best_match(D_tfa, D_stacked),
        "txcdr_vs_tfa":    best_match(D_txcdr, D_tfa),
        "txcdr_vs_stacked":best_match(D_txcdr, D_stacked),
        "stacked_vs_tfa":  best_match(D_stacked, D_tfa),
        "stacked_vs_txcdr":best_match(D_stacked, D_txcdr),
    }

    # Also: best match of TFA against ITSELF (excluding self) for diagonal-check
    # (should be close to 1 for many features due to decoder over-parameterization)

    print("\nSummary statistics:")
    summary = {}
    for key, vals in best.items():
        stats = {
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p10": float(vals.quantile(0.10)),
            "p25": float(vals.quantile(0.25)),
            "p75": float(vals.quantile(0.75)),
            "p90": float(vals.quantile(0.90)),
            "frac_above_0.9": float((vals >= 0.9).float().mean()),
            "frac_below_0.3": float((vals <= 0.3).float().mean()),
        }
        summary[key] = stats
        print(f"  {key:25s}: median={stats['median']:.3f}  "
              f"frac>=0.9 = {stats['frac_above_0.9']:.3f}  "
              f"frac<=0.3 = {stats['frac_below_0.3']:.3f}")

    # Identify top-N "unique" features per architecture
    # (lowest max-cos vs EITHER of the other two archs)
    print("\nIdentifying top-30 unique features per architecture...")

    tfa_uniqueness   = torch.max(best["tfa_vs_txcdr"],   best["tfa_vs_stacked"])
    txcdr_uniqueness = torch.max(best["txcdr_vs_tfa"],   best["txcdr_vs_stacked"])
    stacked_uniqueness = torch.max(best["stacked_vs_tfa"], best["stacked_vs_txcdr"])

    unique_features = {
        "tfa_pos":     torch.argsort(tfa_uniqueness)[:30].tolist(),
        "crosscoder":  torch.argsort(txcdr_uniqueness)[:30].tolist(),
        "stacked_sae": torch.argsort(stacked_uniqueness)[:30].tolist(),
    }
    unique_scores = {
        "tfa_pos":     tfa_uniqueness[unique_features["tfa_pos"]].tolist(),
        "crosscoder":  txcdr_uniqueness[unique_features["crosscoder"]].tolist(),
        "stacked_sae": stacked_uniqueness[unique_features["stacked_sae"]].tolist(),
    }

    # Save per-arch top-unique features
    with open(f"{OUT_DIR}/per_arch_top_unique.json", "w") as f:
        json.dump({
            "features": unique_features,
            "max_cos_with_other_archs": unique_scores,
        }, f, indent=2)

    # Save summary
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot: histograms of best-match cos sim per pair ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        ("TFA-pos vs TXCDR", best["tfa_vs_txcdr"], best["txcdr_vs_tfa"]),
        ("TFA-pos vs Stacked", best["tfa_vs_stacked"], best["stacked_vs_tfa"]),
        ("TXCDR vs Stacked", best["txcdr_vs_stacked"], best["stacked_vs_txcdr"]),
    ]
    for ax, (title, a, b) in zip(axes, pairs):
        ax.hist(a.numpy(), bins=50, alpha=0.6, label=title.split(" vs ")[0] + " → " + title.split(" vs ")[1])
        ax.hist(b.numpy(), bins=50, alpha=0.6, label=title.split(" vs ")[1] + " → " + title.split(" vs ")[0])
        ax.set_title(title)
        ax.set_xlabel("best |cos sim|")
        ax.set_ylabel("# features")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/best_match_hist.png", dpi=120)
    print(f"  -> {OUT_DIR}/best_match_hist.png")

    # ── Plot: scatter per feature, one panel per arch ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    triples = [
        ("TFA-pos features", best["tfa_vs_txcdr"], best["tfa_vs_stacked"], "TXCDR", "Stacked"),
        ("TXCDR features", best["txcdr_vs_tfa"], best["txcdr_vs_stacked"], "TFA-pos", "Stacked"),
        ("Stacked features", best["stacked_vs_tfa"], best["stacked_vs_txcdr"], "TFA-pos", "TXCDR"),
    ]
    for ax, (title, x, y, xlabel, ylabel) in zip(axes, triples):
        ax.scatter(x.numpy(), y.numpy(), s=2, alpha=0.3)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"best cos with {xlabel}")
        ax.set_ylabel(f"best cos with {ylabel}")
        ax.set_title(title)
        # Highlight the "unique" quadrant (low on both axes)
        ax.axvspan(0, 0.3, alpha=0.1, color="red")
        ax.axhspan(0, 0.3, alpha=0.1, color="red")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/best_match_scatter.png", dpi=120)
    print(f"  -> {OUT_DIR}/best_match_scatter.png")

    # ── Print top-5 most-unique per arch for sanity ─────────────────────────
    print("\nTop 5 most-unique features per architecture:")
    for arch, feats in unique_features.items():
        print(f"\n  {arch}:")
        for i, (idx, score) in enumerate(zip(feats[:5], unique_scores[arch][:5])):
            print(f"    feat_{idx:5d}  max-cos-with-other = {score:.3f}")


if __name__ == "__main__":
    main()
