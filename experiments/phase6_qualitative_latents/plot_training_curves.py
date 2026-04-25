"""Quick training-curve plot: loss + L0 vs step for a handful of archs.

Reads `experiments/phase5_downstream_utility/results/training_logs/{run_id}.json`,
plots loss and L0 timeseries side-by-side. Intended for slack-shareable
snapshots — not a paper figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "experiments/phase5_downstream_utility/results/training_logs"

# (run_id, display label, color)
DEFAULT_ARCHS = [
    ("tsae_paper__seed42",         "T-SAE (paper)",            "#d62728"),
    ("agentic_txc_10_bare__seed42", "Track 2 (TXC+anti-dead)", "#08519c"),
    ("phase63_track2_t20__seed42", "Track 2 T=20",             "#08306b"),
    ("txcdr_t5__seed42",           "TXCDR T=5 (no anti-dead)", "#6baed6"),
    ("mlc__seed42",                "MLC baseline",             "#74c476"),
    ("agentic_mlc_08__seed42",     "MLC (multi-layer)",        "#2ca02c"),
    ("topk_sae__seed42",           "TopK SAE (plain)",         "#ff7f0e"),
]


CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"


def _final_dead_fraction(run_id: str, dead_threshold: int = 10_000_000) -> float | None:
    """Read the ckpt and return the fraction of features that are 'dead'
    at end of training, defined as: num_tokens_since_fired ≥ dead_threshold.

    For archs that don't track this buffer (e.g., plain TopK SAE), returns
    None.
    """
    arch = run_id.rsplit("__seed", 1)[0]
    p = CKPT_DIR / f"{run_id}.pt"
    if not p.exists():
        return None
    import torch
    state = torch.load(p, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", {})
    if "num_tokens_since_fired" not in sd:
        return None
    n = sd["num_tokens_since_fired"]
    thr = state.get("meta", {}).get("dead_threshold_tokens", dead_threshold)
    return float((n.float() >= float(thr)).sum() / n.numel())


def _dead_fraction_offline(run_id: str, n_tokens: int = 50_000) -> float | None:
    """Fallback when ckpt doesn't track num_tokens_since_fired:
    load the model, run encode on a sample of L13 activations from the
    cached anchor, count features that never fire (z=0 across all
    sampled tokens). Returns dead fraction in [0, 1].

    Requires the FineWeb L13 cache and load_arch dispatch.
    """
    arch = run_id.rsplit("__seed", 1)[0]
    seed = int(run_id.rsplit("__seed", 1)[1])
    try:
        import torch
        import numpy as np
        # Load model via encode_archs.load_arch
        import sys as _sys
        _sys.path.insert(0, str(REPO / "experiments/phase6_qualitative_latents"))
        from encode_archs import load_arch  # noqa: E402
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, meta = load_arch(arch, device, seed=seed)
        # Sample anchor activations (FineWeb L13)
        anchor_path = REPO / "data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy"
        if not anchor_path.exists():
            return None
        x = np.load(anchor_path, mmap_mode="r")
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        idx = np.random.RandomState(0).choice(x.shape[0], min(n_tokens, x.shape[0]), replace=False)
        x_sample = torch.from_numpy(x[idx].copy()).float().to(device)
        # Encode in batches; track alive features
        T = meta.get("T", 1)
        # Determine d_sae as the largest dim across W_enc/W_dec — robust
        # to per-arch tensor layout differences.
        if hasattr(model, "W_enc"):
            d_sae = max(model.W_enc.shape)
        elif hasattr(model, "W_dec"):
            d_sae = max(model.W_dec.shape)
        else:
            return None
        alive = torch.zeros(d_sae, dtype=torch.bool, device=device)
        BATCH = 256
        with torch.no_grad():
            if T and T > 1:
                # TXC-style: form windows from sample
                n_full = (x_sample.shape[0] // T) * T
                x_win = x_sample[:n_full].reshape(-1, T, x_sample.shape[-1])
                for b0 in range(0, x_win.shape[0], BATCH):
                    z = model.encode(x_win[b0:b0 + BATCH])
                    if isinstance(z, tuple): z = z[0]
                    alive |= (z.abs().sum(dim=tuple(range(z.dim() - 1))) > 0)
            else:
                # Per-token archs (T-SAE / TopK / MLC variants — but MLC needs multi-layer)
                if "mlc" in arch.lower():
                    return None  # MLC needs multi-layer stack — skip offline
                for b0 in range(0, x_sample.shape[0], BATCH):
                    z = model.encode(x_sample[b0:b0 + BATCH])
                    if isinstance(z, tuple): z = z[0]
                    alive |= (z.abs().sum(dim=0) > 0)
        del model
        torch.cuda.empty_cache()
        return float(1.0 - alive.float().mean().item())
    except Exception as e:
        print(f"[dead-offline] {run_id}: {e.__class__.__name__}: {str(e)[:80]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(REPO / "experiments/phase6_qualitative_latents/results/training_curves.png"))
    ap.add_argument("--logy", action="store_true", help="log scale on loss axis")
    args = ap.parse_args()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    for run_id, label, color in DEFAULT_ARCHS:
        p = LOGS / f"{run_id}.json"
        if not p.exists():
            print(f"[skip] {run_id}: log missing")
            continue
        d = json.loads(p.read_text())
        steps = d.get("steps_logged", [])
        loss = d.get("loss", [])
        l0 = d.get("l0", [])
        if not steps or not loss:
            print(f"[skip] {run_id}: empty log")
            continue
        ax1.plot(steps, loss, "-", color=color, linewidth=1.8, label=label, alpha=0.85)
        ax2.plot(steps, l0, "-", color=color, linewidth=1.8, label=label, alpha=0.85)
        # Mark convergence point
        if d.get("converged") and d.get("final_step"):
            fs = d["final_step"]
            ax1.axvline(fs, color=color, linestyle=":", alpha=0.3, linewidth=0.8)

    ax1.set_xlabel("training step", fontsize=11)
    ax1.set_ylabel("loss (MSE recon, scale of model latent units)", fontsize=11)
    ax1.set_title("Training loss vs step", fontsize=12)
    ax1.grid(alpha=0.3)
    if args.logy:
        ax1.set_yscale("log")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.95)

    ax2.set_xlabel("training step", fontsize=11)
    ax2.set_ylabel("mean L0 (active features per token)", fontsize=11)
    ax2.set_title("Sparsity (L0) vs step", fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.95)

    # Third panel: final dead-feature fraction at end of training.
    # Computed from each ckpt's num_tokens_since_fired buffer (only
    # populated for anti-dead archs; plain TopK SAE doesn't track it).
    bar_labels, bar_values, bar_colors = [], [], []
    for run_id, label, color in DEFAULT_ARCHS:
        frac = _final_dead_fraction(run_id)
        source = "buffer"
        if frac is None:
            frac = _dead_fraction_offline(run_id)
            source = "offline (50k tokens)"
        if frac is None:
            bar_labels.append(label + "†")
            bar_values.append(None)
        else:
            bar_labels.append(label + (" *" if source == "offline (50k tokens)" else ""))
            bar_values.append(frac * 100)
        bar_colors.append(color)
    # Plot only archs with data; show others as N/A
    plotted_idx = [i for i, v in enumerate(bar_values) if v is not None]
    xs = list(range(len(plotted_idx)))
    ax3.bar(xs, [bar_values[i] for i in plotted_idx],
            color=[bar_colors[i] for i in plotted_idx],
            edgecolor="black", linewidth=0.8)
    ax3.set_xticks(xs)
    ax3.set_xticklabels([bar_labels[i] for i in plotted_idx],
                        rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("% dead features at end of training", fontsize=11)
    ax3.set_title("Final dead-feature fraction\n"
                  "(* = computed offline from 50k FineWeb tokens; "
                  "† = not measurable here)",
                  fontsize=11)
    ax3.grid(alpha=0.3, axis="y")
    ax3.set_axisbelow(True)
    # Annotate bars with values
    for i, idx in enumerate(plotted_idx):
        v = bar_values[idx]
        ax3.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    fig.suptitle(
        "Training dynamics + final dead features: archs at seed 42  "
        "(dotted vertical = early-stop convergence)",
        fontsize=12,
    )
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    thumb = Path(args.out).with_suffix(".thumb.png")
    fig.savefig(thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} + {thumb}")


if __name__ == "__main__":
    main()
