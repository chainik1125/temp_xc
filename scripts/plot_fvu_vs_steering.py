"""Reconstruction quality does not predict steering quality -- the ordering is inverted.

Each architecture is trained at ITS OWN best recipe, taken from a full lr x steps sweep, and
matched at 8.0 realised coefficients per segment on held-out data. The best reconstructor of the
three steers worst and the worst reconstructor steers best. At the matched dose the rank
inversion is strict and complete: FVU orders tSAE < SAE < crosscoder, steering orders them
exactly in reverse, and the spread is 28x rather than the 3.4x visible at best dose.

This matters beyond temporal codes: benchmarks rank dictionaries by FVU, and on this task that
ranking is exactly backwards for the use a crosscoder is being proposed for.

Reads results/txc_wins/recipe_recency.json (FVU per arm at its own best recipe) and
results/txc_wins/recency_fair.json (steering with every arm at that recipe).
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECIPE = ROOT / "results" / "txc_wins" / "recipe_recency.json"
STEER = ROOT / "results" / "txc_wins" / "recency_fair.json"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "fvu_vs_steering.png"

# Matched dose: the smallest magnitude at which the crosscoder is significant on this task.
DOSE = 0.5

# (recipe key, steering arm, label, colour)
ARMS = [
    ("tsae_topk", "tsae_broadcast", "attention temporal SAE", "#009E73"),
    ("sae", "sae_broadcast", "TopK SAE", "#0072B2"),
    ("txc", "txc_slab", "temporal crosscoder", "#E69F00"),
]


def at_dose(arm, mag):
    """Best of the two SIGNS at this dose magnitude.

    The magnitude is matched across arms; the sign is not, because which class you want to
    steer toward is something the experimenter knows and a steering vector's sign is a free
    parameter. Taking the signed +mag instead would score an arm whose correct direction is
    negative as a failure -- the exact bug that made a one-sided grid hide the SAE's ability
    on the order task, and here it would bias in the crosscoder's favour by reporting the
    two per-token arms as negative.
    """
    best = None
    for a, v, e in zip(arm["alphas"], arm["delta_margin"], arm.get("sem", [0] * 99)):
        if abs(abs(a) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e)
    return best or (max(arm["delta_margin"]), 0.0)


def main() -> int:
    if not (RECIPE.exists() and STEER.exists()):
        print("[skip] inputs not written yet")
        return 1
    best = json.loads(RECIPE.read_text())["best"]
    arms = json.loads(STEER.read_text())["arms"]

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for rk, sk, label, col in ARMS:
        fvu = best[rk]["fvu"]
        # MATCHED DOSE MAGNITUDE, not each arm's own best. Best-dose reporting puts every
        # arm at its own saturation point, outside the linear regime the comparison is
        # about, and flatters flat arms by letting them pick a maximum over the whole grid.
        # At the matched dose the inversion is STRICTER: FVU ranks tSAE < SAE < crosscoder
        # and steering ranks them exactly in reverse, which is NOT true at best dose where
        # the SAE and tSAE are within 0.03 of each other.
        val, sem = at_dose(arms[sk], DOSE)
        ax.errorbar(fvu, val, yerr=sem, fmt="o", ms=13, color=col, capsize=4,
                    elinewidth=1.4, zorder=3)
        lr, steps = best[rk]["lr"], int(best[rk]["steps"])
        ax.annotate(f"{label}\nlr {lr:g}, {steps} steps\nFVU {fvu:.4f}",
                    (fvu, val), textcoords="offset points", xytext=(14, -6),
                    fontsize=9, va="center")

    xs = [best[rk]["fvu"] for rk, _, _, _ in ARMS]
    ys = [at_dose(arms[sk], DOSE)[0] for _, sk, _, _ in ARMS]
    ax.plot(xs, ys, ls=":", lw=1.4, color="#888888", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("FVU at 8.0 realised coefficients per segment, held out\n"
                  "(lower is a better reconstructor)")
    ax.set_ylabel(f"steering Δ margin at matched dose |α| = {DOSE}\n(higher is a better intervention)")
    ax.set_title("Reconstruction quality does not predict steering quality\n"
                 "each architecture at its own best recipe — the ordering is inverted",
                 fontsize=11.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(0.010, 0.30)
    ax.set_ylim(0, max(ys) * 1.30)
    ax.annotate("best reconstructor,\nworst steerer", (xs[0], ys[0]),
                textcoords="offset points", xytext=(10, 52), fontsize=9,
                style="italic", color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))
    ax.annotate("worst reconstructor,\nbest steerer", (xs[2], ys[2]),
                textcoords="offset points", xytext=(-30, -62), fontsize=9,
                style="italic", color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    for (rk, sk, label, _), x, y in zip(ARMS, xs, ys):
        print(f"  {label:<24} FVU {x:.4f}   steering {y:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
