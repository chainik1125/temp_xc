"""The per-token dictionary's reading advantage is specific to SLOW temporal structure.

The task is fixed except for one thing: how fast the factor alternates. Documents are cut
into equal calm/tense blocks and the foil is a cyclic rotation by one block, so the two
classes contain literally the same sentences and differ only in the phase of a square wave.
Raising the switch count holds the multiset, the block structure, the dictionary budget and
the injected norm constant, and asks the temporal profile to resolve faster alternation.

Left: what each architecture can READ. At one switch the pooled per-token SAE latent is at
AUC 0.997 and the crosscoder at 0.746 -- the result that retired reading comparisons last
sprint. By eleven switches the ordering has reversed. A causal transformer smears its history
into every token, which is how a pooled per-token code recovers order at all; that smearing
cannot resolve alternation at period two.

Right: what each architecture can STEER, at matched injected norm and matched realised
coefficients per segment. The crosscoder's slab is the only arm that is positive across the
ladder. `txc_flat` -- the same slab time-averaged and rebroadcast, same latent, same norm,
profile removed -- is the control that makes this a claim about the temporal profile.

Reads results/txc_wins/phase{1,3,5,11}.json.
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "phase_ladder.png"
SWITCHES = [1, 3, 5, 11]

C_SAE = "#0072B2"     # blue           -- per-token TopK SAE
C_TXC = "#E69F00"     # orange         -- crosscoder, temporal profile intact
C_FLAT = "#D55E00"    # vermillion     -- crosscoder with the profile removed
C_TSAE = "#009E73"    # bluish green   -- attention tSAE (TopK)
C_RND = "#999999"     # grey           -- random temporal profile


def best(arm):
    """Best dose for this arm, with the per-document standard error at that dose."""
    j = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
    return arm["delta_margin"][j], arm["sem"][j]


def load_seeds(s):
    """Every dictionary init available for this cell.

    Init matters more than it has any right to: at five switches the crosscoder's best
    delta ranged over 1.56, 15.70 and 11.48 across three inits of the same configuration.
    The sign was stable in every one, the magnitude was not, so a single init is not a
    verdict and the figure shows the spread rather than one draw.
    """
    out = []
    for suffix in ("", "_ds1", "_ds2"):
        p = SRC / f"phase{s}{suffix}.json"
        if p.exists():
            out.append(json.loads(p.read_text()))
    return out


def main() -> int:
    runs, seeds = {}, {}
    for s in SWITCHES:
        p = SRC / f"phase{s}.json"
        if not p.exists():
            print(f"[skip] {p} not written yet")
            return 1
        runs[s] = json.loads(p.read_text())
        seeds[s] = load_seeds(s)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    for key, colour, label in ((("sae"), C_SAE, "TopK SAE (codes pooled over window)"),
                               (("txc"), C_TXC, "crosscoder (window code)"),
                               (("tsae"), C_TSAE, "attention tSAE (codes pooled)")):
        ys = [runs[s]["reading"][key]["auc"] for s in SWITCHES if key in runs[s]["reading"]]
        xs = [s for s in SWITCHES if key in runs[s]["reading"]]
        ax.plot(xs, ys, "o-", color=colour, lw=2.0, ms=6, label=label)
    ax.axhline(0.5, ls="--", color="#888888", lw=1.3)
    ax.text(11, 0.515, "chance", fontsize=8.5, color="#888888", ha="right")
    ax.set_xscale("log"); ax.set_xticks(SWITCHES)
    ax.set_xticklabels([str(s) for s in SWITCHES])
    ax.set_xlabel("switches per document  (faster alternation to the right)")
    ax.set_ylabel("best-single-latent AUC")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("Reading: the SAE's advantage is specific to slow structure")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    ax = axes[1]
    for key, colour, label in (("txc_slab", C_TXC, "crosscoder slab"),
                               ("sae_broadcast", C_SAE, "TopK SAE direction"),
                               ("tsae_broadcast", C_TSAE, "attention tSAE direction"),
                               ("txc_flat", C_FLAT, "crosscoder slab, profile removed"),
                               ("random_slab", C_RND, "random temporal profile")):
        xs, ys, lo, hi = [], [], [], []
        for s in SWITCHES:
            vals = [best(r["arms"][key])[0] for r in seeds[s] if key in r["arms"]]
            if not vals:
                continue
            xs.append(s); ys.append(sum(vals) / len(vals))
            lo.append(min(vals)); hi.append(max(vals))
        ax.plot(xs, ys, "o-", color=colour, lw=2.0, ms=6, label=label)
        # Range across dictionary inits, not a standard error: with three draws the honest
        # display is the spread itself.
        ax.fill_between(xs, lo, hi, color=colour, alpha=0.16, lw=0)
    ax.axhline(0.0, ls="--", color="#888888", lw=1.3)
    ax.set_xscale("log"); ax.set_xticks(SWITCHES)
    ax.set_xticklabels([str(s) for s in SWITCHES])
    ax.set_xlabel("switches per document")
    ax.set_ylabel(r"$\Delta$ margin at each arm's best dose")
    n_seed = min(len(seeds[s]) for s in SWITCHES)
    ax.set_title(f"Steering: mean over {n_seed} dictionary inits, band = range")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
