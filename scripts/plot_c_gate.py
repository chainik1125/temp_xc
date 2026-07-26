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

Only gradient-measured points are plotted, because the difference-of-means version of `c` does
not order these tasks at all and the gradient version does. The clearest single case: the
phase-1 task has c = 0.040 by difference of means and c = 0.227 by gradient, and it loses to a
constant write by 19 points. The difference-of-means number would have predicted a crosscoder
win; the gradient number predicts the loss that happened.

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
    ("recency", "recency", "#E69F00"),
    ("evidence", "evidence", "#009E73"),
    ("recency_var_v2", "recency, positions vary", "#56B4E9"),
    ("order_sym", "order (last sprint's task)", "#D55E00"),
    ("phase1_g", "phase ladder", "#999999"),
    ("phase3_g", "phase ladder", "#999999"),
    ("phase5_g", "phase ladder", "#999999"),
    ("phase11_g", "phase ladder", "#999999"),
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
            rg = r.get("rank_grad")
            if rg is None:
                continue   # difference-of-means c is not the constant share of the metric
            arms = r["arms"]
            if "txc_slab" not in arms or r.get("n_train", 0) < 500:
                continue   # drop the reduced-size validation config
            const = max((best(arms[a]) for a in CONSTANT_ARMS if a in arms), default=0.0)
            pts.append((rg["c"], best(arms["txc_slab"]) - const, label, colour,
                        "rank_grad" in r))

    if not pts:
        print("[skip] no runs with a rank measurement yet")
        return 1

    # Rank correlation, reported because the relationship is a tendency and not a rule and
    # the number is the honest way to say so.
    import itertools
    n = len(pts)
    conc = dis = 0
    for i, j in itertools.combinations(range(n), 2):
        sgn = (pts[i][0] - pts[j][0]) * (pts[i][1] - pts[j][1])
        conc += sgn > 0
        dis += sgn < 0
    tau = (conc - dis) / max(conc + dis, 1)

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
    ax.set_title(f"Constant share vs the crosscoder's margin  "
                 f"(Kendall $\\tau$ = {tau:+.2f}, n = {n})")
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
