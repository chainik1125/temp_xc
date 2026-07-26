"""Frontier figure: how to compare a temporal crosscoder to a TopK SAE.

Left panel is the comparison itself -- reconstruction error against the only
sparsity axis the two architectures share, *coefficients spent per segment*.
Nominal k is not that axis: a TXC's k counts latents per window, an SAE's counts
latents per token, and the TXC additionally fails to spend the k it is given.

Right panel is why the naive axis fails: realised coefficients saturate well
below the nominal budget, so raising k past the saturation point buys nothing.

Reads results/dict_bench/frontier.json (written by frontier_modal.py).
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "dict_bench" / "frontier.json"
OUT = ROOT / "plots" / "2026-07-25_dictbench" / "frontier.png"

# Wong colourblind-safe palette.
C_SAE = "#0072B2"   # blue
C_TXC = "#E69F00"   # orange
C_REF = "#000000"


def main() -> int:
    if not SRC.exists():
        print(f"[skip] {SRC} not written yet")
        return 1
    r = json.loads(SRC.read_text())
    sae, txc = r["sae"], r["txc"]
    if not sae or not txc:
        print("[skip] frontier.json has an empty arm")
        return 1

    # One LR arm per architecture: the one that reconstructs best at its own
    # largest budget, so neither side is judged on an under-trained run.
    lrs = sorted({t["lr"] for t in txc})
    best_lr = min(lrs, key=lambda L: min(t["fvu"] for t in txc if t["lr"] == L))
    txc_b = sorted((t for t in txc if t["lr"] == best_lr),
                   key=lambda t: t["coeff_per_segment"])
    sae_b = sorted(sae, key=lambda s: s["coeff_per_segment"])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    # ---------------- left: the shared-axis frontier ----------------
    ax = axes[0]
    ax.plot([s["coeff_per_segment"] for s in sae_b], [s["fvu"] for s in sae_b],
            "o-", color=C_SAE, lw=2, ms=6, label="TopK SAE")
    ax.plot([t["coeff_per_segment"] for t in txc_b], [t["fvu"] for t in txc_b],
            "s-", color=C_TXC, lw=2, ms=6,
            label=f"temporal crosscoder (lr {best_lr:g})")
    ax.set_xscale("log")
    ax.set_xlabel("coefficients spent per segment")
    ax.set_ylabel("FVU  (held-out, per-segment)")
    ax.set_title("Matched on what both architectures spend")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # Annotate the gap at the tightest budget both reach.
    lo = max(sae_b[0]["coeff_per_segment"], txc_b[0]["coeff_per_segment"])
    s_at = min(sae_b, key=lambda s: abs(s["coeff_per_segment"] - lo))
    t_at = min(txc_b, key=lambda t: abs(t["coeff_per_segment"] - lo))
    ax.annotate("", xy=(lo, s_at["fvu"]), xytext=(lo, t_at["fvu"]),
                arrowprops=dict(arrowstyle="<->", color=C_REF, lw=1.2))
    ax.text(lo * 1.15, 0.5 * (s_at["fvu"] + t_at["fvu"]),
            f"{t_at['fvu'] - s_at['fvu']:+.2f} FVU\nat {lo:.0f}/segment",
            fontsize=8.5, va="center")

    # ---------------- right: the budget the TXC declines to spend ----------------
    ax = axes[1]
    ax.plot([s["nominal_k"] for s in sae_b],
            [s["coeff_per_segment"] for s in sae_b],
            "o-", color=C_SAE, lw=2, ms=6, label="TopK SAE")
    # A TXC's nominal budget is k_per_pos coefficients for every segment.
    ax.plot([t["nominal_k_per_pos"] for t in txc_b],
            [t["coeff_per_segment"] for t in txc_b],
            "s-", color=C_TXC, lw=2, ms=6, label="temporal crosscoder")
    hi = max(sae_b[-1]["nominal_k"], txc_b[-1]["nominal_k_per_pos"])
    ax.plot([1, hi], [1, hi], "--", color=C_REF, lw=1.2, label="spends its budget")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal k per segment")
    ax.set_ylabel("realised coefficients per segment")
    ax.set_title("Nominal k is not a budget the TXC spends")
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)

    # The one-line headline, printed so it lands in the log too.
    print("\nmatched-budget comparison (TXC vs nearest SAE by coeff/segment):")
    for t in txc_b:
        n = min(sae_b, key=lambda s: abs(s["coeff_per_segment"]
                                         - t["coeff_per_segment"]))
        print(f"  {t['coeff_per_segment']:6.2f}/seg  TXC {t['fvu']:.3f}  "
              f"SAE {n['fvu']:.3f}  -> "
              f"{'TXC' if t['fvu'] < n['fvu'] else 'SAE'} better")
    return 0


if __name__ == "__main__":
    sys.exit(main())
