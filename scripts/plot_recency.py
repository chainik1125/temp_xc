"""A crosscoder latent can reverse the model's instruction-recency bias; a constant write
cannot move it at all.

The task: twelve segments of filler with two conflicting instructions at fixed positions 2
and 9. The two classes are exact reorderings -- same filler, same instructions, same
positions -- differing only in which instruction is early and which is late. Language models
resolve conflicting instructions largely by recency, which is what a prompt injection
exploits, and here that bias is worth -2.42 nats at baseline.

The metric is a difference of differences,

    [ logP(obey instr 1 | doc A) - logP(obey instr 2 | doc A) ]
  - [ logP(obey instr 1 | doc B) - logP(obey instr 2 | doc B) ] ,   steered minus unsteered,

so a write that simply makes the model prefer instruction 1 pushes both documents equally and
contributes exactly zero. Only a write whose effect depends on POSITION can move this number.

Left: dose response. The crosscoder's slab is antisymmetric in the dose -- push one way and
the model obeys the early instruction, push the other way and it obeys the late one -- which
is what a directed intervention looks like. Every constant-write arm is positive at BOTH
extreme doses, which is what a magnitude artefact looks like.

Right: where each write puts its mass, and it corrects the obvious guess. The SUPERVISED
rank-1 write puts almost all of its mass exactly on the two segments carrying the
instructions. The crosscoder does NOT: its profile is much flatter, peaks in the wrong places,
and still reaches 76% of the supervised effect. So the crosscoder is not solving this by
locating the instructions -- whatever it found is something else that works nearly as well,
and `txc_profile_random` (the same profile with random directions, flat at 0.00) shows the
profile alone is not what carries it.

Reads results/txc_wins/recency*.json.
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "recency.png"

C_TXC = "#E69F00"
C_SAE = "#0072B2"
C_FLAT = "#D55E00"
C_TSAE = "#009E73"
C_DOM = "#000000"
C_RND = "#999999"

ARMS = [("rank1_best", "#7F7F7F", "--", "best rank-1 write (per-token ceiling, supervised)"),
        ("sae_schedule", "#56B4E9", "--", "SAE direction on a supervised schedule"),
        ("txc_slab", C_TXC, "-", "crosscoder slab (unsupervised)"),
        ("sae_broadcast", C_SAE, "-", "TopK SAE direction"),
        ("tsae_broadcast", C_TSAE, "-", "attention tSAE direction"),
        ("txc_flat", C_FLAT, "-", "crosscoder slab, profile removed"),
        ("txc_profile_random", "#8C564B", ":", "crosscoder profile, random directions")]


def main(name: str = "recency") -> int:
    src = ROOT / "results" / "txc_wins" / f"{name}.json"
    if not src.exists():
        print(f"[skip] {src} not written yet")
        return 1
    r = json.loads(src.read_text())
    arms, alphas = r["arms"], r["alphas"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))

    ax = axes[0]
    for key, colour, ls, label in ARMS:
        if key not in arms:
            continue
        a = arms[key]
        ax.errorbar(alphas, a["delta_margin"], yerr=a["sem"], fmt="o" + ls,
                    color=colour, lw=2.0, ms=5, capsize=3, label=label)
    ax.axhline(0.0, color="#888888", lw=1.2)
    base = r.get("baseline_contrast", {}).get("mean")
    if base is not None:
        # Crossing this line means the recency bias has not merely been suppressed but
        # reversed: the model now obeys the EARLY instruction.
        ax.axhline(-base, ls=":", color="#B03060", lw=1.6)
        ax.text(alphas[0], -base + 0.25, "recency bias fully reversed above this line",
                fontsize=8.5, color="#B03060")
    ax.set_xlabel(r"steering dose $\alpha$  (all writes at identical injected norm)")
    ax.set_ylabel(r"$\Delta$  [obey early $-$ obey late],  A minus B")
    ax.set_title("Reversing which instruction the model obeys")
    ax.set_ylim(-5.6, 10.2)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.95, ncol=1)

    ax = axes[1]
    prof = r.get("write_profile")
    if prof:
        T = len(prof["txc_slab"])
        for key, colour, ls, label in ARMS:
            if key in ("rank1_best", "txc_slab", "txc_flat") and key in prof:
                ax.plot(range(T), prof[key], "o" + ls, color=colour, lw=2.0, ms=5,
                        label=label)
        for p in (2, T - 3):
            ax.axvline(p, color="#B03060", lw=1.4, alpha=0.55)
        ax.text(2, ax.get_ylim()[1] * 0.98, " instruction positions", fontsize=8.5,
                color="#B03060", va="top")
        ax.set_xlabel("segment position within the document")
        ax.set_ylabel("norm of the write at that position")
        ax.set_title("Where each write puts its mass")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    else:
        ax.text(0.5, 0.5, "write_profile not recorded in this run",
                ha="center", va="center", fontsize=10, color="#888888")
        ax.set_axis_off()

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "recency"))
