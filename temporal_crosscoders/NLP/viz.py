#!/usr/bin/env python3
"""
viz.py — Post-sweep visualization and analysis for NLP experiments.

Produces:
  1. Loss comparison heatmaps: StackedSAE vs TXCDR per layer
  2. FVU comparison across layers
  3. Loss vs k curves per layer and T
  4. Convergence curves
  5. Entropy comparison: StackedSAE vs TXCDR
  6. HMM rho fitting: fit feature activation timeseries to lag-1 autocorrelation
  7. Feature activation contrast: sentence-level activation comparison
  8. Summary table

Usage:
    python viz.py
    python viz.py --fit-rho
    python viz.py --contrast
    python viz.py --all
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_all_logs(log_dir: str) -> dict:
    """
    Load all JSON logs.

    Returns:
        {(model, layer, k, T): [{"step": ..., "loss": ..., ...}, ...]}
    """
    results = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.json"))):
        basename = os.path.basename(path).replace(".json", "")
        if basename == "sweep_summary":
            continue
        # Naming: {model}__{layer}__k{k}__T{T}
        parts = basename.split("__")
        if len(parts) != 4:
            continue
        model = parts[0]
        layer = parts[1]
        k_match = re.match(r"k(\d+)", parts[2])
        T_match = re.match(r"T(\d+)", parts[3])
        if not k_match or not T_match:
            continue
        k = int(k_match.group(1))
        T = int(T_match.group(1))

        with open(path) as f:
            history = json.load(f)
        results[(model, layer, k, T)] = history
    return results


def get_final(history: list[dict], key: str, default: float = 0.0) -> float:
    if not history:
        return default
    return history[-1].get(key, default)


# ─── 1. Loss comparison heatmaps ──────────────────────────────────────────────

def plot_loss_heatmaps(results: dict, viz_dir: str) -> None:
    """For each layer: k x T heatmap of Loss(TXCDR) - Loss(StackedSAE)."""
    layers = sorted(set(l for _, l, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), squeeze=False)

    for col, layer in enumerate(layers):
        ax = axes[0, col]
        mat = np.full((len(ks), len(Ts)), np.nan)
        for i, k in enumerate(ks):
            for j, T in enumerate(Ts):
                tx_key = ("txcdr", layer, k, T)
                sae_key = ("stacked_sae", layer, k, T)
                if tx_key in results and sae_key in results:
                    diff = get_final(results[tx_key], "loss") - get_final(results[sae_key], "loss")
                    mat[i, j] = diff

        vabs = max(
            abs(np.nanmin(mat)) if not np.all(np.isnan(mat)) else 0.01,
            abs(np.nanmax(mat)) if not np.all(np.isnan(mat)) else 0.01,
            0.01,
        )
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(mat, cmap="RdBu", norm=norm, aspect="auto", origin="lower")

        for i in range(len(ks)):
            for j in range(len(Ts)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vabs * 0.6 else "black"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold", color=color)

        ax.set_xticks(range(len(Ts)))
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([str(k) for k in ks])
        ax.set_xlabel("T")
        ax.set_ylabel("k")
        ax.set_title(f"DeltaLoss — {layer}")

    fig.colorbar(im, ax=axes.ravel().tolist(),
                 label="DeltaLoss (TXCDR - StackedSAE)", shrink=0.8, pad=0.08)
    fig.suptitle("Reconstruction Loss Difference (negative = TXCDR wins)", fontsize=12, y=1.02)
    path = os.path.join(viz_dir, "delta_loss_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 2. FVU comparison across layers ─────────────────────────────────────────

def plot_fvu_comparison(results: dict, viz_dir: str) -> None:
    """Bar chart comparing FVU across layers for both architectures at each (k, T)."""
    layers = sorted(set(l for _, l, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    fig, axes = plt.subplots(len(Ts), 1, figsize=(12, 4 * len(Ts)), squeeze=False)

    for row, T in enumerate(Ts):
        ax = axes[row, 0]
        x = np.arange(len(layers))
        width = 0.35

        for ki, k in enumerate(ks):
            offset = (ki - len(ks) / 2 + 0.5) * width / len(ks) * 2
            sae_fvus = [get_final(results[("stacked_sae", l, k, T)], "fvu")
                        if ("stacked_sae", l, k, T) in results else np.nan for l in layers]
            tx_fvus = [get_final(results[("txcdr", l, k, T)], "fvu")
                       if ("txcdr", l, k, T) in results else np.nan for l in layers]

            ax.bar(x + offset - width / 4, sae_fvus, width / len(ks),
                   label=f"SAE k={k}", alpha=0.7)
            ax.bar(x + offset + width / 4, tx_fvus, width / len(ks),
                   label=f"TXCDR k={k}", alpha=0.7, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=15)
        ax.set_ylabel("FVU (lower is better)")
        ax.set_title(f"FVU Comparison — T={T}")
        ax.legend(fontsize=6, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(viz_dir, "fvu_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 3. Loss vs k curves ─────────────────────────────────────────────────────

def plot_loss_vs_k(results: dict, viz_dir: str) -> None:
    """Loss vs k for each (layer, T), comparing architectures."""
    layers = sorted(set(l for _, l, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    fig, axes = plt.subplots(
        len(Ts), len(layers), figsize=(5 * len(layers), 4 * len(Ts)), squeeze=False,
    )

    for row, T in enumerate(Ts):
        for col, layer in enumerate(layers):
            ax = axes[row, col]
            for model_type, style, label in [
                ("stacked_sae", "--o", "StackedSAE"),
                ("txcdr", "-^", "TXCDR"),
            ]:
                losses = []
                valid_ks = []
                for k in ks:
                    key = (model_type, layer, k, T)
                    if key in results:
                        losses.append(get_final(results[key], "loss"))
                        valid_ks.append(k)
                if valid_ks:
                    ax.plot(valid_ks, losses, style, label=label, lw=1.5, ms=5)

            ax.set_xlabel("k")
            ax.set_ylabel("Loss")
            ax.set_title(f"{layer} — T={T}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Reconstruction Loss vs k", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(viz_dir, "loss_vs_k.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 4. Convergence curves ───────────────────────────────────────────────────

def plot_convergence(results: dict, viz_dir: str) -> None:
    """Training loss convergence per layer, overlaying SAE vs TXCDR."""
    layers = sorted(set(l for _, l, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    colors = {"stacked_sae": "steelblue", "txcdr": "firebrick"}

    for layer in layers:
        fig, axes = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
        for col, k in enumerate(ks):
            ax = axes[0, col]
            for T in Ts:
                for model_type in ["stacked_sae", "txcdr"]:
                    key = (model_type, layer, k, T)
                    if key not in results:
                        continue
                    hist = results[key]
                    steps = [h["step"] for h in hist]
                    losses = [h["loss"] for h in hist]
                    ls = "--" if model_type == "stacked_sae" else "-"
                    name = "SAE" if model_type == "stacked_sae" else "TXCDR"
                    alpha = 0.4 + 0.6 * (Ts.index(T) / max(len(Ts) - 1, 1))
                    ax.plot(steps, losses, color=colors[model_type],
                            ls=ls, lw=1.5, alpha=alpha, label=f"{name} T={T}")

            ax.set_title(f"k={k}")
            ax.set_xlabel("Step")
            if col == 0:
                ax.set_ylabel("Loss")
            ax.legend(fontsize=6, loc="upper right")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Convergence — {layer}", fontsize=12)
        plt.tight_layout()
        path = os.path.join(viz_dir, f"convergence_{layer}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


# ─── 5. Entropy comparison ───────────────────────────────────────────────────

def plot_entropy_comparison(results: dict, viz_dir: str) -> None:
    """Bar chart: final entropy for StackedSAE vs TXCDR, grouped by (layer, k, T).

    Prediction: StackedSAE should have higher entropy (more uniform feature
    activation) since it lacks temporal sharing.
    """
    layers = sorted(set(l for _, l, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    # Per-layer panel, x-axis = (k, T) combos
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5), squeeze=False)

    for col, layer in enumerate(layers):
        ax = axes[0, col]
        # Only include combos where at least one architecture has results
        combos = [(k, T) for k in ks for T in Ts
                  if ("stacked_sae", layer, k, T) in results or ("txcdr", layer, k, T) in results]
        x = np.arange(len(combos))

        sae_entropies = []
        tx_entropies = []
        for k, T in combos:
            sae_entropies.append(get_final(results[("stacked_sae", layer, k, T)], "entropy")
                                 if ("stacked_sae", layer, k, T) in results else np.nan)
            tx_entropies.append(get_final(results[("txcdr", layer, k, T)], "entropy")
                                if ("txcdr", layer, k, T) in results else np.nan)

        width = 0.35
        ax.bar(x - width / 2, sae_entropies, width, label="StackedSAE", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, tx_entropies, width, label="TXCDR", color="firebrick", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"k{k}\nT{T}" for k, T in combos], fontsize=6, rotation=45)
        ax.set_ylabel("Entropy (bits)")
        ax.set_title(f"Feature Activation Entropy — {layer}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Activation Entropy: StackedSAE vs TXCDR", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(viz_dir, "entropy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Also save delta entropy summary
    lines = ["Entropy Difference (StackedSAE - TXCDR):", ""]
    for layer in layers:
        for k in ks:
            for T in Ts:
                sae_key = ("stacked_sae", layer, k, T)
                tx_key = ("txcdr", layer, k, T)
                if sae_key not in results and tx_key not in results:
                    continue
                sae_h = get_final(results.get(sae_key, []), "entropy")
                tx_h = get_final(results.get(tx_key, []), "entropy")
                diff = sae_h - tx_h
                lines.append(f"  {layer:14s} k={k:>3d} T={T:>2d}  SAE={sae_h:.4f}  TXCDR={tx_h:.4f}  delta={diff:+.4f}")
    text = "\n".join(lines)
    path = os.path.join(viz_dir, "entropy_summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"  Saved: {path}")


# ─── 6. HMM rho fitting ─────────────────────────────────────────────────────

def fit_rho_from_activations(
    checkpoint_path: str,
    layer_key: str,
    model_type: str,
    k: int,
    T: int,
    cache_dir: str,
    n_samples: int = 2000,
) -> dict:
    """
    Fit the lag-1 autocorrelation (rho) of learned feature activations.

    For each latent dimension, compute the binary support timeseries over
    consecutive windows, then estimate rho = corr(s_t, s_{t+1}).

    This mirrors the HMM simulation rho from the toy model experiments —
    if the model learns temporally structured features, rho should be > 0.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import StackedSAE, TemporalCrosscoder
    from config import D_SAE, LAYER_SPECS

    spec = LAYER_SPECS[layer_key]
    d_act = spec["d_act"]

    # Load model
    if model_type == "stacked_sae":
        model = StackedSAE(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    else:
        model = TemporalCrosscoder(d_in=d_act, d_sae=D_SAE, T=T, k=k)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load cached activations
    act_path = os.path.join(cache_dir, f"{layer_key}.npy")
    data = np.load(act_path, mmap_mode="r")
    num_chains = data.shape[0]
    seq_length = data.shape[1]

    # Compute binary support over consecutive non-overlapping windows
    n_windows = seq_length // T
    rho_per_latent: list[float] = []
    firing_prob_per_latent: list[float] = []

    with torch.no_grad():
        # Sample a subset of chains
        chain_indices = np.random.choice(num_chains, size=min(n_samples, num_chains), replace=False)

        all_supports: list[np.ndarray] = []
        for ci in chain_indices:
            chain_data = torch.from_numpy(data[ci].copy()).float().unsqueeze(0)  # (1, L, d)
            window_supports: list[np.ndarray] = []
            for w in range(n_windows):
                start = w * T
                window = chain_data[:, start : start + T, :]  # (1, T, d)
                if model_type == "stacked_sae":
                    _, _, u = model(window)
                    # u: (1, T, h) — binary support per position
                    support = (u > 0).float().mean(dim=1).squeeze(0).numpy()  # (h,)
                else:
                    _, _, z = model(window)
                    support = (z > 0).float().squeeze(0).numpy()  # (h,)
                window_supports.append(support)

            if len(window_supports) >= 2:
                all_supports.append(np.stack(window_supports))  # (n_windows, h)

    if not all_supports:
        return {"mean_rho": 0.0, "std_rho": 0.0, "mean_firing_prob": 0.0}

    # Stack all chains: (n_chains, n_windows, h)
    supports = np.stack(all_supports)

    # Compute lag-1 autocorrelation per latent
    h = supports.shape[2]
    rhos = np.zeros(h)
    for j in range(h):
        series = supports[:, :, j].flatten()  # all chains concatenated
        if series.std() < 1e-8:
            rhos[j] = 0.0
            continue
        # Lag-1 autocorrelation
        x = series[:-1]
        y = series[1:]
        x_c = x - x.mean()
        y_c = y - y.mean()
        denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())
        if denom < 1e-8:
            rhos[j] = 0.0
        else:
            rhos[j] = (x_c * y_c).sum() / denom

    firing_probs = supports.mean(axis=(0, 1))  # (h,)

    return {
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos)),
        "median_rho": float(np.median(rhos)),
        "mean_firing_prob": float(np.mean(firing_probs)),
        "rho_per_latent": rhos.tolist(),
        "firing_prob_per_latent": firing_probs.tolist(),
    }


def plot_rho_distribution(
    rho_results: dict[str, dict], viz_dir: str,
) -> None:
    """Plot the distribution of fitted rho values across latents."""
    n = len(rho_results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, min(n, 4), figsize=(5 * min(n, 4), 4), squeeze=False)

    for i, (label, res) in enumerate(rho_results.items()):
        if i >= 4:
            break
        ax = axes[0, i]
        rhos = np.array(res["rho_per_latent"])
        ax.hist(rhos, bins=50, alpha=0.7, color="steelblue", edgecolor="black", lw=0.5)
        ax.axvline(res["mean_rho"], color="red", ls="--", lw=2,
                   label=f"mean={res['mean_rho']:.3f}")
        ax.axvline(res["median_rho"], color="orange", ls="--", lw=2,
                   label=f"median={res['median_rho']:.3f}")
        ax.set_xlabel("rho (lag-1 autocorrelation)")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Activation rho Distribution (HMM fit)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(viz_dir, "rho_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 7. Feature activation contrast ─────────────────────────────────────────

def plot_feature_activation_contrast(
    checkpoint_dir: str,
    cache_dir: str,
    viz_dir: str,
    layer_key: str = "mid_res",
    k: int = 50,
    T: int = 25,
    n_features_to_show: int = 30,
) -> None:
    """
    Contrast how features activate within a single sentence (T=25 tokens)
    for StackedSAE vs TXCDR.

    Shows a heatmap of feature activations (top-n by magnitude) across token
    positions for one example sentence. Prediction: StackedSAE should show
    higher entropy (more spread out, independent activations) while TXCDR
    should show more structured, temporally coherent patterns.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import StackedSAE, TemporalCrosscoder
    from config import D_SAE, LAYER_SPECS, run_name as rn_fn

    spec = LAYER_SPECS[layer_key]
    d_act = spec["d_act"]

    # Load both models
    sae_ckpt = os.path.join(checkpoint_dir, f"{rn_fn('stacked_sae', layer_key, k, T)}.pt")
    tx_ckpt = os.path.join(checkpoint_dir, f"{rn_fn('txcdr', layer_key, k, T)}.pt")

    if not os.path.exists(sae_ckpt) or not os.path.exists(tx_ckpt):
        print(f"  Skipping contrast: missing checkpoints for {layer_key} k={k} T={T}")
        return

    sae_model = StackedSAE(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    sae_model.load_state_dict(torch.load(sae_ckpt, map_location="cpu", weights_only=True))
    sae_model.eval()

    tx_model = TemporalCrosscoder(d_in=d_act, d_sae=D_SAE, T=T, k=k)
    tx_model.load_state_dict(torch.load(tx_ckpt, map_location="cpu", weights_only=True))
    tx_model.eval()

    # Load a single window from cache
    act_path = os.path.join(cache_dir, f"{layer_key}.npy")
    data = np.load(act_path, mmap_mode="r")
    chain_idx = 42  # deterministic example
    x_np = data[chain_idx, :T, :]  # (T, d)
    x = torch.from_numpy(x_np.copy()).float().unsqueeze(0)  # (1, T, d)

    # Try to load token IDs for labeling
    token_path = os.path.join(cache_dir, "token_ids.npy")
    token_labels = None
    if os.path.exists(token_path):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
            token_ids = np.load(token_path, mmap_mode="r")
            tids = token_ids[chain_idx, :T]
            token_labels = [tokenizer.decode([int(tid)]) for tid in tids]
        except Exception:
            token_labels = None

    if token_labels is None:
        token_labels = [f"t{i}" for i in range(T)]

    with torch.no_grad():
        # StackedSAE: (1, T, h) activations
        _, _, u_sae = sae_model(x)
        u_sae = u_sae.squeeze(0)  # (T, h)

        # TXCDR: (1, h) shared latent → we expand to (T, h) via decoder
        _, x_hat_tx, z_tx = tx_model(x)
        z_tx = z_tx.squeeze(0)  # (h,)
        # For TXCDR, the "per-position contribution" is z * W_dec[:, t, :]
        # but since z is shared, we tile it for visualization
        u_tx = z_tx.unsqueeze(0).expand(T, -1)  # (T, h) — same z at each position

    # Select top features by total activation magnitude
    sae_mag = u_sae.abs().sum(dim=0)  # (h,)
    tx_mag = u_tx.abs().sum(dim=0)    # (h,)
    combined = sae_mag + tx_mag

    top_indices = combined.topk(n_features_to_show).indices.numpy()

    sae_heatmap = u_sae[:, top_indices].numpy()   # (T, n_feat)
    tx_heatmap = u_tx[:, top_indices].numpy()      # (T, n_feat)

    # Compute per-position entropy for each model
    import math
    def position_entropy(acts: np.ndarray) -> np.ndarray:
        """acts: (T, n_feat) → entropy per position (T,)"""
        binary = (acts > 0).astype(float)
        p = binary.mean(axis=1)  # (T,) - fraction of features active
        eps = 1e-8
        p = np.clip(p, eps, 1 - eps)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    sae_pos_ent = position_entropy(u_sae.numpy())
    tx_pos_ent = position_entropy(u_tx.numpy())

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [5, 5, 2]})

    # SAE heatmap
    ax = axes[0]
    im0 = ax.imshow(sae_heatmap.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(n_features_to_show))
    ax.set_yticklabels([f"f{i}" for i in top_indices], fontsize=5)
    ax.set_xticks(range(T))
    ax.set_xticklabels(token_labels, fontsize=6, rotation=60, ha="right")
    ax.set_title(f"StackedSAE Feature Activations — {layer_key} k={k} T={T}")
    ax.set_ylabel("Feature")
    fig.colorbar(im0, ax=ax, shrink=0.5, label="Activation")

    # TXCDR heatmap
    ax = axes[1]
    im1 = ax.imshow(tx_heatmap.T, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(n_features_to_show))
    ax.set_yticklabels([f"f{i}" for i in top_indices], fontsize=5)
    ax.set_xticks(range(T))
    ax.set_xticklabels(token_labels, fontsize=6, rotation=60, ha="right")
    ax.set_title(f"TXCDR Feature Activations (shared latent) — {layer_key} k={k} T={T}")
    ax.set_ylabel("Feature")
    fig.colorbar(im1, ax=ax, shrink=0.5, label="Activation")

    # Position entropy comparison
    ax = axes[2]
    x_pos = np.arange(T)
    ax.plot(x_pos, sae_pos_ent, "o--", color="steelblue", lw=1.5, ms=4, label=f"SAE (mean={sae_pos_ent.mean():.3f})")
    ax.plot(x_pos, tx_pos_ent, "^-", color="firebrick", lw=1.5, ms=4, label=f"TXCDR (mean={tx_pos_ent.mean():.3f})")
    ax.set_xticks(range(T))
    ax.set_xticklabels(token_labels, fontsize=6, rotation=60, ha="right")
    ax.set_ylabel("Position Entropy (bits)")
    ax.set_title("Per-Position Entropy: SAE (predicted higher) vs TXCDR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(viz_dir, f"feature_contrast_{layer_key}_k{k}_T{T}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─── 8. Summary table ───────────────────────────────────────────────────────

def print_summary_table(results: dict, viz_dir: str) -> None:
    """Print and save a text summary table."""
    layers = sorted(set(l for _, l, _, _ in results.keys()))
    ks = sorted(set(k for _, _, k, _ in results.keys()))
    Ts = sorted(set(T for _, _, _, T in results.keys()))

    lines: list[str] = []
    header = f"{'Layer':<14} {'Model':<14} {'k':>4} {'T':>3} {'L0':>6} {'Loss':>10} {'FVU':>8} {'Entropy':>8}"
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for layer in layers:
        for k in ks:
            for T in Ts:
                for model in ["stacked_sae", "txcdr"]:
                    key = (model, layer, k, T)
                    if key not in results:
                        continue
                    h = results[key][-1] if results[key] else {}
                    lines.append(
                        f"{layer:<14} {model:<14} {k:>4} {T:>3} "
                        f"{h.get('window_l0', 0):>6.0f} "
                        f"{h.get('loss', 0):>10.4f} {h.get('fvu', 0):>8.4f} "
                        f"{h.get('entropy', 0):>8.4f}"
                    )

    lines.append("=" * len(header))

    # TXCDR advantage summary
    lines.append("\n-- TXCDR ADVANTAGE SUMMARY (DeltaLoss = TXCDR - SAE) --")
    for layer in layers:
        deltas = []
        for k in ks:
            for T in Ts:
                tx = ("txcdr", layer, k, T)
                sae = ("stacked_sae", layer, k, T)
                if tx in results and sae in results:
                    diff = get_final(results[tx], "loss") - get_final(results[sae], "loss")
                    deltas.append(diff)
        if deltas:
            mean_d = np.mean(deltas)
            lines.append(
                f"  [{layer}] mean DeltaLoss={mean_d:+.4f}  "
                f"range=[{min(deltas):+.4f}, {max(deltas):+.4f}]"
            )

    text = "\n".join(lines)
    print(text)

    path = os.path.join(viz_dir, "summary_table.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  Saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NLP post-sweep visualization")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument("--viz-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cached activations directory")
    parser.add_argument("--fit-rho", action="store_true", help="Run HMM rho fitting on checkpoints")
    parser.add_argument("--contrast", action="store_true", help="Run feature activation contrast analysis")
    parser.add_argument("--all", action="store_true", help="Run all analyses including rho and contrast")
    parser.add_argument("--contrast-layer", type=str, default="mid_res", help="Layer for contrast analysis")
    parser.add_argument("--contrast-k", type=int, default=50, help="k for contrast analysis")
    parser.add_argument("--contrast-T", type=int, default=25, help="T for contrast analysis")
    args = parser.parse_args()

    from config import LOG_DIR, VIZ_DIR, CHECKPOINT_DIR, CACHE_DIR

    log_dir = args.log_dir or LOG_DIR
    viz_dir = args.viz_dir or VIZ_DIR
    checkpoint_dir = args.checkpoint_dir or CHECKPOINT_DIR
    cache_dir = args.cache_dir or CACHE_DIR

    os.makedirs(viz_dir, exist_ok=True)

    print("Loading logs...")
    results = load_all_logs(log_dir)
    if not results:
        print(f"  No logs found in {log_dir}/. Run sweep.py first.")
        sys.exit(1)

    print(f"  Loaded {len(results)} run histories\n")

    print("-- Generating visualizations --")
    plot_loss_heatmaps(results, viz_dir)
    plot_fvu_comparison(results, viz_dir)
    plot_loss_vs_k(results, viz_dir)
    plot_convergence(results, viz_dir)
    plot_entropy_comparison(results, viz_dir)
    print()
    print_summary_table(results, viz_dir)

    # HMM rho fitting (optional, requires checkpoints)
    if args.fit_rho or args.all:
        print("\n-- HMM rho Fitting --")
        rho_results: dict[str, dict] = {}

        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if not checkpoint_files:
            print(f"  No checkpoints in {checkpoint_dir}/. "
                  f"Run sweep.py with --save-checkpoints.")
        else:
            for ckpt_path in sorted(checkpoint_files):
                basename = os.path.basename(ckpt_path).replace(".pt", "")
                parts = basename.split("__")
                if len(parts) != 4:
                    continue
                model_type = parts[0]
                layer_key = parts[1]
                k = int(parts[2].replace("k", ""))
                T = int(parts[3].replace("T", ""))

                label = f"{model_type}_{layer_key}_k{k}_T{T}"
                print(f"  Fitting rho for {label}...")
                try:
                    res = fit_rho_from_activations(
                        ckpt_path, layer_key, model_type, k, T, cache_dir,
                    )
                    rho_results[label] = res
                    print(f"    mean rho = {res['mean_rho']:.4f} +/- {res['std_rho']:.4f}")
                except Exception as e:
                    print(f"    ERROR: {e}")

            if rho_results:
                plot_rho_distribution(rho_results, viz_dir)

                # Save rho results
                rho_path = os.path.join(viz_dir, "rho_fitting.json")
                serializable = {}
                for label, res in rho_results.items():
                    serializable[label] = {
                        k: v for k, v in res.items()
                        if k not in ("rho_per_latent", "firing_prob_per_latent")
                    }
                with open(rho_path, "w") as f:
                    json.dump(serializable, f, indent=2)
                print(f"  Saved: {rho_path}")

    # Feature activation contrast (optional)
    if args.contrast or args.all:
        print("\n-- Feature Activation Contrast --")
        plot_feature_activation_contrast(
            checkpoint_dir, cache_dir, viz_dir,
            layer_key=args.contrast_layer,
            k=args.contrast_k,
            T=args.contrast_T,
        )

    print(f"\n  All plots saved to {viz_dir}/")


if __name__ == "__main__":
    main()
