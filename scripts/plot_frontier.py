"""How to compare a temporal crosscoder to a TopK SAE, and why nominal k cannot do it.

Right panel is the sprint's central finding. Realised sparsity is min(k, #{pre > 0}): TopK
selects k latents and ReLU zeroes any whose pre-activation was negative. Which term binds
is set by the optimiser, not the architecture -- at lr=1e-3 the crosscoder's realised spend
*falls* as k rises, at lr=3e-4 the same model at the same nominal budgets tracks its budget
to kper=8 and only then saturates. Nothing in the nominal configuration distinguishes them.

Left panel is the comparison that survives, on the only axis both architectures spend in the
same units: coefficients per segment, measured rather than assumed.

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
C_GOOD = "#E69F00"  # orange -- crosscoder at the learning rate that works
C_BAD = "#D55E00"   # vermillion -- crosscoder at the one that collapses
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

    lrs = sorted({t["lr"] for t in txc})
    # The arm that reconstructs best at its largest budget is the one worth comparing;
    # the other is kept in the figure because the gap between them is the finding.
    good_lr = min(lrs, key=lambda L: min(t["fvu"] for t in txc if t["lr"] == L))
    arms = {}
    for L in lrs:
        arms[L] = sorted((t for t in txc if t["lr"] == L),
                         key=lambda t: t["nominal_k_per_pos"])
    sae_b = sorted(sae, key=lambda s: s["coeff_per_segment"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    # ---------------- left: the shared-axis frontier ----------------
    ax = axes[0]
    ax.plot([s["coeff_per_segment"] for s in sae_b], [s["fvu"] for s in sae_b],
            "o-", color=C_SAE, lw=2.2, ms=6, label="TopK SAE", zorder=4)
    for L in lrs:
        a = sorted(arms[L], key=lambda t: t["coeff_per_segment"])
        col = C_GOOD if L == good_lr else C_BAD
        style = "s-" if L == good_lr else "s--"
        ax.plot([t["coeff_per_segment"] for t in a], [t["fvu"] for t in a],
                style, color=col, lw=2.2, ms=6, alpha=1.0 if L == good_lr else 0.75,
                label=f"crosscoder, lr {L:g}", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("coefficients actually spent per segment")
    ax.set_ylabel("FVU  (held-out, per-segment)")
    ax.set_title("Matched on what both architectures spend")
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="center right", frameon=True, fontsize=8.5)

    # Annotate the widest budget the crosscoder reaches -- the fairest single number.
    a = sorted(arms[good_lr], key=lambda t: t["coeff_per_segment"])
    t_at = a[-1]
    s_at = min(sae_b, key=lambda s: abs(s["coeff_per_segment"]
                                        - t_at["coeff_per_segment"]))
    x = t_at["coeff_per_segment"]
    ax.annotate("", xy=(x, s_at["fvu"]), xytext=(x, t_at["fvu"]),
                arrowprops=dict(arrowstyle="<->", color=C_REF, lw=1.3))
    ax.text(x * 0.62, 0.5 * (s_at["fvu"] + t_at["fvu"]),
            f"{t_at['fvu'] / s_at['fvu']:.0f}×\nat {x:.0f}/segment",
            fontsize=9, va="center", ha="right", fontweight="bold")

    # ---------------- right: what each model does with the budget it is given ----------
    ax = axes[1]
    hi = max(sae_b[-1]["nominal_k"], max(t["nominal_k_per_pos"] for t in txc))
    ax.plot([1, hi], [1, hi], "--", color=C_REF, lw=1.3, zorder=1,
            label="spends its budget")
    ax.plot([s["nominal_k"] for s in sae_b],
            [s["coeff_per_segment"] for s in sae_b],
            "o-", color=C_SAE, lw=2.2, ms=6, label="TopK SAE", zorder=4)
    for L in lrs:
        col = C_GOOD if L == good_lr else C_BAD
        style = "s-" if L == good_lr else "s--"
        ax.plot([t["nominal_k_per_pos"] for t in arms[L]],
                [t["coeff_per_segment"] for t in arms[L]],
                style, color=col, lw=2.2, ms=6,
                alpha=1.0 if L == good_lr else 0.75,
                label=f"crosscoder, lr {L:g}", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal k per segment")
    ax.set_ylabel("coefficients actually spent per segment")
    ax.set_title("Same architecture, same k — the optimiser decides")
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)

    print("\nmatched-budget comparison at the working learning rate:")
    for t in sorted(arms[good_lr], key=lambda t: t["coeff_per_segment"]):
        n = min(sae_b, key=lambda s: abs(s["coeff_per_segment"]
                                         - t["coeff_per_segment"]))
        print(f"  {t['coeff_per_segment']:6.2f}/seg  crosscoder {t['fvu']:.3f}  "
              f"SAE {n['fvu']:.3f}  -> {t['fvu'] / n['fvu']:.1f}x")

    print("\nwhat the learning rate alone does, at matched nominal k:")
    lo_lr = max(lrs)
    for t in arms[good_lr]:
        o = next((u for u in arms[lo_lr]
                  if u["nominal_k_per_pos"] == t["nominal_k_per_pos"]), None)
        if o:
            print(f"  kper={t['nominal_k_per_pos']:>3}  "
                  f"lr {lo_lr:g}: {o['coeff_per_segment']:5.2f}/seg  ->  "
                  f"lr {good_lr:g}: {t['coeff_per_segment']:5.2f}/seg   "
                  f"({t['coeff_per_segment'] / o['coeff_per_segment']:.1f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
