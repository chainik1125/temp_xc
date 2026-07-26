"""Why the previous sprint's headline is withdrawn: half the dose axis was never sampled.

That sprint reported a crosscoder advantage on an order-only task and rested it on one
control. `txc_flat` is the crosscoder's own decoder slab with the temporal profile averaged
away and rebroadcast — same latent, same mean direction, same norm, only the time structure
removed. It was measured at -8.02, read as the effect INVERTING when the profile was
destroyed, and reported as proof that the temporal profile was doing the work.

The grid it was measured on swept POSITIVE doses only. Swept symmetrically, `txc_flat` is
large and negative on the right and large and positive on the left — an ordinary signed
effect, of which a one-sided sweep can only ever see one branch. Since the sign of a steering
vector is a free parameter, the honest reading is the opposite of the published one:
`txc_flat` is not a control that failed, it is a BETTER CONSTANT WRITE than the SAE's, and it
beats the crosscoder's own full slab.

The shaded half is the region the original grid sampled. Everything that overturns the result
is in the unshaded half.

Two dictionary inits are shown because the conclusion does not depend on one draw: `txc_flat`
peaks at +12.10 and +18.47 against the crosscoder's +6.34 and +3.41.

A second thing is visible and worth naming. The CROSSCODER'S OWN CURVE FLIPS SIGN BETWEEN THE
TWO INITS — rising to the right at init 0 and to the left at init 1 — while `txc_flat` and the
SAE keep their orientation in both. So on this task the crosscoder's selected latent does not
even have a stable direction across training runs, which is the same instability that made
the phase-ladder cells flip with init, and it is a further reason the original result should
not have rested on a single draw.

Reads results/txc_wins/order_sym_ds{0,1}.json.
"""
import json
import pathlib

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-26_txcwins" / "withdrawal.png"

ARMS = [("txc_slab", "#E69F00", "-", "crosscoder slab"),
        ("txc_flat", "#D55E00", "-", "crosscoder slab, profile removed"),
        ("sae_broadcast", "#0072B2", "-", "TopK SAE direction"),
        ("tsae_broadcast", "#009E73", ":", "attention tSAE direction")]


def main() -> int:
    runs = [(ds, json.loads(p.read_text()))
            for ds in (0, 1)
            if (p := SRC / f"order_sym_ds{ds}.json").exists()]
    if not runs:
        print(f"[skip] no order_sym runs in {SRC}")
        return 1

    fig, axes = plt.subplots(1, len(runs), figsize=(5.9 * len(runs), 4.6), sharey=True)
    axes = [axes] if len(runs) == 1 else list(axes)

    for ax, (ds, r) in zip(axes, runs):
        arms, alphas = r["arms"], r["arms"]["txc_slab"]["alphas"]
        ax.axvspan(0, max(alphas) * 1.08, color="#bbbbbb", alpha=0.22, lw=0, zorder=0)
        for key, colour, ls, label in ARMS:
            if key not in arms:
                continue
            ax.errorbar(alphas, arms[key]["delta_margin"], yerr=arms[key]["sem"],
                        fmt="o" + ls, color=colour, lw=2.0, ms=5, capsize=3,
                        label=label, zorder=3)
        ax.axhline(0.0, color="#888888", lw=1.2)
        ax.axvline(0.0, color="#888888", lw=1.0, ls="--")
        ax.set_xlabel(r"steering dose $\alpha$")
        ax.set_title(f"dictionary init {ds}", fontsize=10.5)
        ax.grid(alpha=0.25, lw=0.6)
        if ax is axes[0]:
            ax.set_ylabel(r"$\Delta$ margin  (positive steers toward class A)")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
        ax.text(max(alphas) * 0.54, ax.get_ylim()[1] * 0.93,
                "the only half\nthe original\ngrid sampled", fontsize=8.5,
                color="#555555", ha="center", va="top")

    fig.suptitle("The withdrawn result: the control that was supposed to prove the "
                 "temporal profile\nis simply a better constant write, and its winning "
                 "branch was never measured", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)
    for ds, r in runs:
        a = r["arms"]
        print(f"  init {ds}: txc_flat peaks {max(a['txc_flat']['delta_margin']):+.2f}, "
              f"crosscoder {max(a['txc_slab']['delta_margin']):+.2f}, "
              f"SAE {max(a['sae_broadcast']['delta_margin']):+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
