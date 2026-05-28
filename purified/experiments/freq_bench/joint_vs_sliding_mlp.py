"""Linear vs MLP probe across joint T=W and sliding-T architectures.

Tests whether the joint-T=W ceiling is a probe-class artefact. If it were,
an unrestricted nonlinear (MLP) probe should close the gap to sliding-T.
If the ceiling is architectural, the MLP probe should give only modest
lift on joint cells while the joint-vs-sliding gap stays open.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "freq_bench" / "v2_sweep"
PROTO = "1.2.0"
W, K, DSAE = 16, 1, 1024

CELLS = [
    ("regular_sae", "per-token", "#888888"),
    ("txc_base_perpos_TW", "txc_base perpos\nT=W", "#ff7f0e"),
    ("txc_base_TW", "txc_base joint\nT=W", "#1f77b4"),
    ("txcdr_t2", "sliding T=2", "#9467bd"),
    ("txcdr_t5", "sliding T=5", "#2ca02c"),
]


def pick(label):
    for line in open(LEADERBOARD):
        r = json.loads(line)
        if (r.get("experiment") == "freq_bench"
                and r.get("evaluator_protocol_version") == PROTO):
            ec = r["eval_cfg"]
            if (ec.get("label") == label and ec.get("W") == W
                    and ec.get("k_pos") == K and ec.get("d_sae") == DSAE):
                return r["metrics"]
    return None


def main():
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = np.arange(len(CELLS))
    w = 0.36
    lin_vals, mlp_vals = [], []
    for label, _, _ in CELLS:
        m = pick(label) or {}
        lin_vals.append(m.get("NTPS", 0))
        mlp_vals.append(m.get("NTPS_mlp", 0))

    for i, (_, _, color) in enumerate(CELLS):
        ax.bar(xs[i] - w/2, lin_vals[i], w, color=color, edgecolor="k",
               label="linear probe" if i == 0 else None)
        ax.bar(xs[i] + w/2, mlp_vals[i], w, color=color, edgecolor="k",
               alpha=0.55, hatch="//",
               label="MLP probe" if i == 0 else None)

    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n, _ in CELLS], fontsize=9)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("NTPS")
    ax.set_ylim(-0.05, 0.9)
    ax.set_title(f"Linear vs MLP probe @ W={W}, raw_k={K}, d_sae={DSAE}\n"
                 "joint-vs-sliding gap is architectural, not probe-class")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.3, axis="y")

    # annotate each bar
    for i, (l, m) in enumerate(zip(lin_vals, mlp_vals)):
        ax.text(xs[i] - w/2, l + 0.015, f"{l:.2f}", ha="center", fontsize=8)
        ax.text(xs[i] + w/2, m + 0.015, f"{m:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    p = OUT / "joint_vs_sliding_mlp.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)
    print("\n=== summary ===")
    for (label, _, _), l, m in zip(CELLS, lin_vals, mlp_vals):
        print(f"  {label:25s}  linear={l:+.3f}  mlp={m:+.3f}  "
              f"lift={(m-l):+.3f}")


if __name__ == "__main__":
    main()
