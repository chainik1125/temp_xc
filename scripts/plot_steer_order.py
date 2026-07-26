"""Reading and steering come apart: the sprint's headline figure.

Left: on a window factor whose label is pure ORDER (identical multiset, identical switch
count, blocks swapped), a pooled per-token SAE latent reads the label almost perfectly while
the crosscoder reads it worse.

Right: on the SAME task, steering reverses. A per-token dictionary's only per-latent write is
one direction applied at every position -- constant in time -- and against two orderings of
one multiset a constant write has nothing to push on. The crosscoder's (T, d) slab varies
across positions and steers. `txc_flat` is the control that makes this a claim about the
temporal profile rather than about the direction: it is the crosscoder's own slab averaged
over time and rebroadcast, same latent, same norm.

Reads results/dict_bench/steer_order.json (written by steer_order_modal.py).
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "dict_bench" / "steer_order.json"
OUT = ROOT / "plots" / "2026-07-25_dictbench" / "steer_order.png"

C_SAE = "#0072B2"    # blue   -- per-token dictionary
C_TXC = "#E69F00"    # orange -- crosscoder, temporal profile intact
C_FLAT = "#D55E00"   # vermillion -- crosscoder with the profile removed
C_DOM = "#000000"    # supervised ceiling


def main() -> int:
    if not SRC.exists():
        print(f"[skip] {SRC} not written yet")
        return 1
    r = json.loads(SRC.read_text())
    arms = r["arms"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # ---------------- left: what each architecture can READ ----------------
    ax = axes[0]
    names = ["TopK SAE\n(pooled latent)", "crosscoder\n(window latent)"]
    vals = [r["sae_pooled_auc"], r["txc_window_auc"]]
    bars = ax.bar(names, vals, color=[C_SAE, C_TXC], width=0.55)
    ax.axhline(0.5, ls="--", color="#888888", lw=1.4)
    ax.text(1.45, 0.515, "chance", fontsize=8.5, color="#888888", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.45, 1.08)
    ax.set_ylabel("window-AUC  (best single latent)")
    ax.set_title("Reading an order-only factor")
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    # ---------------- right: what each architecture can STEER ----------------
    ax = axes[1]
    style = [("dom_slab", C_DOM, "--", "difference-of-means (supervised)"),
             ("txc_slab", C_TXC, "-", "crosscoder slab"),
             ("sae_broadcast", C_SAE, "-", "SAE direction, broadcast"),
             ("txc_flat", C_FLAT, ":", "crosscoder slab, time-averaged")]
    for key, col, ls, lab in style:
        if key not in arms:
            continue
        a = arms[key]
        ax.plot(a["alphas"], a["delta_margin"], ls, color=col, lw=2.2,
                marker="o", ms=5, label=lab)
    ax.axhline(0, color="#444444", lw=1.0)
    ax.set_xlabel("steering dose α  (matched injected norm)")
    ax.set_ylabel("Δ margin   logP(order A) − logP(order B)")
    ax.set_title("Steering the same factor")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    # The supervised arm is an order of magnitude larger; log-ish scaling would hide the
    # sign changes that matter, so clip the view to the dictionary arms and say so.
    lo = min(min(arms[k]["delta_margin"]) for k in ("txc_flat", "sae_broadcast", "txc_slab")
             if k in arms)
    hi = max(arms["txc_slab"]["delta_margin"])
    ax.set_ylim(lo * 1.4, hi * 1.6)
    ax.text(0.98, 0.03, "supervised arm continues off-scale",
            transform=ax.transAxes, ha="right", fontsize=8, style="italic",
            color="#444444")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)

    print(f"\n  reads:  SAE {r['sae_pooled_auc']:.3f}   crosscoder {r['txc_window_auc']:.3f}")
    print("  steers: " + "   ".join(
        f"{k} {max(v['delta_margin']):+.2f}" for k, v in arms.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
