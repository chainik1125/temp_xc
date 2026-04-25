"""Phase 6.3 Priority 2b: plot cumulative SEMANTIC count vs N (top-N
cut) for 3 archs on concat_random.

Tells us whether TXC's "qualitative gap" is a top-32 artefact (gap
closes at larger N) or a real structural property (gap persists at all
N).

Reads `{arch}__seed42__concatrandom__top256.json` emitted by
run_topN_sweep.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO / "experiments/phase6_qualitative_latents/results/autointerp"

ARCHS = [
    ("tsae_paper",               "T-SAE (paper)",       "#d62728", "*"),
    ("agentic_txc_02_batchtopk", "Cycle F",             "#9ecae1", "v"),
    ("agentic_txc_10_bare",      "Track 2",             "#08519c", "D"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concat", type=str, default="random")
    ap.add_argument("--out", type=str,
                    default=str(REPO / "experiments/phase6_qualitative_latents/results/phase63_topN_sweep.png"))
    args = ap.parse_args()

    Ns = [32, 64, 128, 256]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for arch, label, color, marker in ARCHS:
        path = LABELS_DIR / f"{arch}__seed{args.seed}__concat{args.concat}__top256.json"
        if not path.exists():
            print(f"[skip] {arch}: no top256 JSON")
            continue
        d = json.loads(path.read_text())
        cums = d["cumulative_semantic_counts"]
        y = [cums[str(n)] for n in Ns]
        ax.plot(Ns, y, "-" + marker, color=color, linewidth=2,
                markersize=10, markeredgecolor="black", markeredgewidth=0.8,
                label=label)
        # Annotate the top-256 endpoint with the value
        ax.annotate(f"{y[-1]}", (Ns[-1], y[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=10, color=color, fontweight="bold")

    # Diagonal "perfect saturation" reference (SEMANTIC frac = 1)
    ax.plot(Ns, Ns, ":", color="#888888", alpha=0.5,
            label="perfect saturation (all sem)")

    ax.set_xlabel(f"Top-N features by per-token variance  (concat_{args.concat})",
                  fontsize=11)
    ax.set_ylabel("Cumulative SEMANTIC count", fontsize=11)
    ax.set_title(
        "Priority 2b: SEMANTIC count vs N — 3 archs, concat_random, seed 42\n"
        "Tests whether TXC's top-32 gap is a ranking artefact or structural",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xticks(Ns)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    thumb = Path(args.out).with_suffix(".thumb.png")
    fig.savefig(thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} + {thumb}")


if __name__ == "__main__":
    main()
