"""How much of the optimal intervention is out of a per-token dictionary's reach, and what
that buys the crosscoder.

Steering a window of T segments means adding a T x d MATRIX to the residual stream. A
per-token dictionary has one direction per latent, so even steered perfectly -- with a
different dose at every position, which is what practitioners do and what the attention tSAE
does automatically -- it can only produce a RANK-1 matrix. A crosscoder latent is
unconstrained in rank. So the share of the optimal write that lies at rank 1,

    r1 = sigma_1^2 / ||P||_F^2 ,

says before any dictionary is trained how much room there is for a window code to win at all.

The task family varies exactly that. `rotate{m}` cuts a document into m blocks of DISTINCT
topics and builds the foil by rotating the blocks by one, so the two classes are the same
sentences read from different starting points. The optimal write then has rank m - 1 and
r1 falls as roughly 2/m.

Left: measured r1 against the analytic prediction. Right: what each arm achieves, with
`rank1_best` -- the best rank-1 approximation of the optimal write, i.e. the ceiling for ANY
per-token dictionary handed a perfect schedule -- as the line the crosscoder has to clear for
the result to be about expressiveness rather than about discovery.

Reads results/txc_wins/rotate{2,4,6,12}_v1.json.
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "rank_ladder.png"
MS = [2, 4, 6, 12]

C_TXC = "#E69F00"
C_SAE = "#0072B2"
C_R1 = "#CC79A7"
C_SCHED = "#56B4E9"
C_FLAT = "#D55E00"
C_GRAD = "#000000"


def best(arm):
    j = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
    return arm["delta_margin"][j], arm["sem"][j]


def main() -> int:
    runs = {}
    for m in MS:
        p = SRC / f"rotate{m}_v1.json"
        if p.exists():
            runs[m] = json.loads(p.read_text())
    if not runs:
        print(f"[skip] no rotate runs in {SRC}")
        return 1
    ms = sorted(runs)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    for key, colour, label in (("rank", C_R1, "measured (difference of means)"),
                               ("rank_grad", C_GRAD, "measured (gradient of the metric)")):
        xs = [m for m in ms if key in runs[m]]
        ys = [runs[m][key]["r1"] for m in xs]
        if xs:
            ax.plot(xs, ys, "o-", color=colour, lw=2.0, ms=6, label=label)
    ax.plot(ms, [2.0 / m if m > 2 else 1.0 for m in ms], "s--", color="#888888",
            lw=1.6, ms=5, label="analytic, rank $m-1$")
    ax.set_xscale("log"); ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("$m$, number of distinct blocks rotated")
    ax.set_ylabel(r"$r_1$: share of the optimal write reachable at rank 1")
    ax.set_title("How much a per-token dictionary can express")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    ax = axes[1]
    for key, colour, ls, label in (
            ("grad_slab", C_GRAD, "--", "optimal write (metric gradient)"),
            ("txc_slab", C_TXC, "-", "crosscoder slab"),
            ("grad_rank1", C_R1, "-", "best rank-1 write (per-token ceiling)"),
            ("sae_schedule_grad", C_SCHED, "-", "SAE direction on the best schedule"),
            ("sae_broadcast", C_SAE, "-", "SAE direction, constant"),
            ("txc_flat", C_FLAT, ":", "crosscoder slab, profile removed")):
        xs, ys, es = [], [], []
        for m in ms:
            if key in runs[m]["arms"]:
                v, e = best(runs[m]["arms"][key])
                xs.append(m); ys.append(v); es.append(e)
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt="o" + ls, color=colour, lw=2.0, ms=5,
                        capsize=3, label=label)
    ax.axhline(0.0, color="#888888", lw=1.2)
    ax.set_xscale("log"); ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("$m$, number of distinct blocks rotated")
    ax.set_ylabel(r"$\Delta$ margin at each arm's best dose")
    ax.set_title("Clearing the per-token ceiling is the expressiveness claim")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
