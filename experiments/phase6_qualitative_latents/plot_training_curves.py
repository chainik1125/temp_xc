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
    ("agentic_txc_02_batchtopk__seed42", "Cycle F (BatchTopK)","#9ecae1"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(REPO / "experiments/phase6_qualitative_latents/results/training_curves.png"))
    ap.add_argument("--logy", action="store_true", help="log scale on loss axis")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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

    fig.suptitle(
        "Training dynamics: 7 archs at seed 42  "
        "(dotted line = early-stop convergence)",
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
