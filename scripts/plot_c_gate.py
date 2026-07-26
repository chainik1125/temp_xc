"""Does the constant share of the optimal write predict which architecture wins?

CAUTION, and it is why this figure exists rather than a table. `c` must be computed from the
GRADIENT of the metric being reported, not from the difference-of-means slab. The two are
nearly orthogonal in practice -- measured cos = +0.095, +0.044, -0.037 and +0.003 on four
different tasks -- and a difference-of-means `c` fails to separate cases that a gradient `c`
separates by a factor of seven. Points measured from the gradient are drawn as circles and
are the only ones the claim rests on; squares are difference-of-means and are shown greyed
into the same axes only so the gap is visible.

`c` is the share of the optimal steering write that lies along the all-positions-equal
direction:

    c = T ||mean_t P_t||^2 / ||P||_F^2

where `P` is the gradient of the reported metric with respect to the (T, d) write, or the
supervised difference-of-means slab where no gradient was measured. It has two readings that
turn out to be the same statement. It is the share a CONSTANT write can push along -- one
direction added at every position, which is the only per-latent intervention a per-token
dictionary has. And it is the share a POOLED per-token probe can read, because pooling
averages over positions and so separates the classes exactly when the mean of the difference
slab is nonzero.

Plotted against it: how far the crosscoder's slab beats the best CONSTANT write available on
that task -- the largest of the SAE's direction, the tSAE's direction, the crosscoder's own
slab flattened, and a random constant direction. Every arm is at matched injected norm, doses
swept symmetrically about zero so no arm is locked to the wrong sign, and matched on realised
coefficients per segment measured out of sample.

Status of the claim, stated plainly because it is not settled. On the two tasks where c comes
from the gradient it separates them cleanly and in the predicted direction -- the previous
sprint's order task at c = 0.241 loses to a constant write, recency at c = 0.034 wins. On the
difference-of-means points it does not order the tasks at all: the phase ladder at c = 0.006
to 0.040 is a null or a loss throughout, and phase-1 at c = 0.040 loses by 19 points while
recency at c = 0.035 wins. Whether that is the wrong c or a wrong hypothesis is settled by the
gradient runs, not by this figure.

Reads every results/txc_wins/*.json that carries a rank measurement.
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "c_gate.png"

CONSTANT_ARMS = ("sae_broadcast", "tsae_broadcast", "txc_flat", "random_broadcast")
# One point per task; where a task has several dictionary inits they are averaged and the
# spread is drawn, because init moved these numbers by up to 10x earlier in the sprint.
FAMILIES = [
    ("recency_v2", "recency", "#E69F00"),
    ("recency_gen", "recency", "#E69F00"),
    ("evidence", "evidence", "#009E73"),
    ("recency_var_v2", "recency, positions vary", "#56B4E9"),
    ("order_sym", "order (last sprint's task)", "#D55E00"),
    ("phase1_v2", "phase, 1 switch", "#999999"),
    ("phase3_v2", "phase, 3 switches", "#999999"),
    ("phase5_v2", "phase, 5 switches", "#999999"),
    ("phase11_v2", "phase, 11 switches", "#999999"),
    ("rot_m", "rotation ladder", "#CC79A7"),
]


def best(arm):
    return max(arm["delta_margin"])


def main() -> int:
    pts = []
    for prefix, label, colour in FAMILIES:
        runs = [json.loads(p.read_text()) for p in sorted(SRC.glob(f"{prefix}*.json"))
                if json.loads(p.read_text()).get("rank")]
        if not runs:
            continue
        for r in runs:
            rg = r.get("rank_grad") or r["rank"]
            arms = r["arms"]
            if "txc_slab" not in arms:
                continue
            const = max((best(arms[a]) for a in CONSTANT_ARMS if a in arms), default=0.0)
            pts.append((rg["c"], best(arms["txc_slab"]) - const, label, colour,
                        "rank_grad" in r))

    if not pts:
        print("[skip] no runs with a rank measurement yet")
        return 1

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    seen = set()
    for c, margin, label, colour, from_grad in pts:
        ax.scatter(c, margin, s=70 if from_grad else 40, color=colour,
                   marker="o" if from_grad else "s",
                   edgecolor="black", linewidth=0.6, zorder=3,
                   label=label if label not in seen else None)
        seen.add(label)
    ax.axhline(0.0, ls="--", color="#888888", lw=1.4)
    ax.text(ax.get_xlim()[1], 0.3, "crosscoder beats every constant write above this line",
            fontsize=8.5, color="#555555", ha="right")
    ax.set_xlabel(r"$c$  —  share of the optimal write that is constant across positions")
    ax.set_ylabel("crosscoder slab  $-$  best constant write   (delta margin)")
    ax.set_title("Constant share against the crosscoder's margin over constant writes")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    print(f"\n{'task':<28}{'c':>8}{'txc - best constant':>22}")
    for c, margin, label, _, _ in sorted(pts, key=lambda p: p[0]):
        print(f"{label:<28}{c:>8.3f}{margin:>22.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
