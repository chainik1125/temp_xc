"""Headline figure for the dictionary benchmark.

A — the decisive control. Permuting a learned TXC atom's rows in time preserves every
    per-slot direction and every norm and destroys only the temporal arrangement. If the
    shuffled arm matches the intact one, the crosscoder's advantage is not temporal.
B — the same numbers against the honest budget: scalars the operator sets, not latents.

    uv run python scripts/plot_dictbench.py
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
C = json.load(open(ROOT / "results/dict_bench/controls.json"))
OUT = ROOT / "plots/2026-07-25_dictbench"
OUT.mkdir(parents=True, exist_ok=True)

A = C["arms"]
MS = sorted(A["txc"], key=int)
x = [int(m) for m in MS]
get = lambda arm, key="fidelity": [A[arm][m][key] for m in MS]

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=150)

# ---------------- A: the control that answers the question ----------------
ax = axes[0]
ax.plot(x, get("txc"), color="#0072B2", marker="o", ms=7, lw=2.2,
        label="temporal crosscoder (learned)")
ax.plot(x, get("txc_shuffled"), color="#CC79A7", marker="s", ms=7, lw=2.2, ls="--",
        label="same atoms, rows shuffled in time")
ax.plot(x, get("txc_random"), color="0.55", marker="^", ms=6, lw=1.8,
        label="random slabs (same coverage & norm)")
ax.plot(x, get("sae_broadcast"), color="#E69F00", marker="d", ms=6, lw=1.8,
        label="SAE, one direction held constant")
ax.axhline(0, color="0.85", lw=1)
ax.set_xscale("log", base=2)
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in x])
ax.set_xlabel("m — latents the operator may use")
ax.set_ylabel("fidelity  =  Δmargin(write) / Δmargin(ideal schedule)")
ax.set_title("A. Destroying the learned temporal order costs nothing\n"
             "(learned content matters; its arrangement in time does not)",
             fontsize=10.5)
ax.legend(fontsize=8.5, frameon=True, edgecolor="0.85", loc="upper left")
ax.grid(color="0.93", lw=0.8)
ax.annotate("shuffled ≥ intact at\n4 of the 5 budgets", (4, 0.458),
            xytext=(5.2, 0.20), fontsize=8.5, color="#8f4c74", ha="left",
            arrowprops=dict(arrowstyle="->", color="#8f4c74", lw=1))
ax.annotate("random slabs fail — so the\nlearned content is real", (8, 0.059),
            xytext=(1.35, 0.135), fontsize=8, color="0.4",
            arrowprops=dict(arrowstyle="->", color="0.55", lw=0.9))

# ---------------- B: honest budget — scalars, not latents ----------------
ax2 = axes[1]
ax2.plot(get("txc", "scalars"), get("txc"), color="#0072B2", marker="o", ms=7,
         lw=2.2, label="temporal crosscoder (1 scalar per latent)")
ax2.plot(get("sae_fullsupp", "scalars"), get("sae_fullsupp"), color="#009E73",
         marker="s", ms=7, lw=2.2,
         label="SAE with per-slot coefficients (k per latent)")
ax2.plot(get("sae_perpos", "scalars"), get("sae_perpos"), color="#E69F00",
         marker="^", ms=6, lw=1.8, label="SAE, one slot per scalar")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("scalars the operator actually sets")
ax2.set_ylabel("fidelity")
ax2.set_title("B. Counted in scalars rather than latents,\nthe SAE is not behind",
              fontsize=10.5)
ax2.legend(fontsize=8.5, frameon=True, edgecolor="0.85", loc="lower right")
ax2.grid(color="0.93", lw=0.8)
ax2.annotate("12 scalars:\nSAE 0.63  ·  TXC ≈ 0.59", (12, 0.627),
             xytext=(1.6, 0.80), fontsize=8.5, color="#0a6b52",
             arrowprops=dict(arrowstyle="->", color="#0a6b52", lw=1))
for a in (ax, ax2):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

fig.suptitle("Does a temporal crosscoder steer trajectories better than a TopK SAE? "
             "Not for temporal reasons.\n"
             "Qwen-2.5-1.5B layer 14 · both dictionaries trained on one activation cache "
             "at matched token-activations/step · d_sae 4096 · n=20 pairs",
             fontsize=10, y=1.03)
fig.tight_layout(rect=(0, 0, 1, 0.94))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"dictbench_headline.{ext}", bbox_inches="tight")
print("saved", OUT / "dictbench_headline.png")
