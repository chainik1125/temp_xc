"""Window length vs realised capacity, with T=1 as a built-in SAE control.

At T=1 a TemporalCrosscoder is a TopK SAE: the encoder einsum collapses to a matmul and the
decoder normalisation over dims (1, 2) is per-atom unit norm, exactly as TopKSAE normalises
its columns. So the T=1 point tells you whether the two implementations agree at all, and
the slope in T tells you what sharing one code across a window actually costs.

Left panel: realised coefficients per segment against T, with the nominal budget as a
reference line. Right panel: FVU against T, with the SAE at the same coefficients per
segment as a reference line.

Reads results/dict_bench/tsweep.json (written by tsweep_modal.py).
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "dict_bench" / "tsweep.json"
OUT = ROOT / "plots" / "2026-07-25_dictbench" / "tsweep.png"

C_SAE = "#0072B2"   # blue
C_TXC = "#E69F00"   # orange
C_ALT = "#009E73"   # green
C_REF = "#000000"


def main() -> int:
    if not SRC.exists():
        print(f"[skip] {SRC} not written yet")
        return 1
    r = json.loads(SRC.read_text())
    sae, rows, kper = r["sae"], r["txc"], r["kper"]
    if not rows:
        print("[skip] tsweep.json has no crosscoder rows")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    colours = {"base": C_TXC, "center": C_ALT}

    # ---------------- left: realised capacity against window length ----------------
    ax = axes[0]
    for arm in ["base", "center"]:
        a = sorted((x for x in rows if x["arm"] == arm), key=lambda x: x["T"])
        if not a:
            continue
        ax.plot([x["T"] for x in a], [x["coeff_per_segment"] for x in a],
                "s-", color=colours[arm], lw=2, ms=6, label=f"crosscoder ({arm})")
    Ts = sorted({x["T"] for x in rows})
    ax.axhline(kper, ls="--", color=C_REF, lw=1.2,
               label=f"nominal budget ({kper}/segment)")
    ax.plot([1], [sae["coeff_per_segment"]], "o", color=C_SAE, ms=9,
            label="TopK SAE", zorder=5)
    ax.set_xlabel("window length T (segments sharing one code)")
    ax.set_ylabel("realised coefficients per segment")
    ax.set_title("What the shared code actually spends")
    ax.set_xticks(Ts)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    # ---------------- right: reconstruction against window length ----------------
    ax = axes[1]
    for arm in ["base", "center"]:
        a = sorted((x for x in rows if x["arm"] == arm), key=lambda x: x["T"])
        if not a:
            continue
        ax.plot([x["T"] for x in a], [x["fvu"] for x in a],
                "s-", color=colours[arm], lw=2, ms=6, label=f"crosscoder ({arm})")
    ax.axhline(sae["fvu"], ls="--", color=C_SAE, lw=1.6,
               label=f"TopK SAE at {kper}/segment")
    ax.set_xlabel("window length T (segments sharing one code)")
    ax.set_ylabel("FVU  (held-out, per-segment)")
    ax.set_title("Cost of sharing one code across the window")
    ax.set_xticks(Ts)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)

    # The control, printed because it is the number the figure hinges on.
    base1 = [x for x in rows if x["T"] == 1 and x["arm"] == "base"]
    print(f"\nSAE                       coeff/seg {sae['coeff_per_segment']:.2f}  "
          f"FVU {sae['fvu']:.4f}  #pre>0/seg {sae['n_pos_preact_per_segment']:.0f}")
    if base1:
        b = base1[0]
        print(f"crosscoder at T=1 (base)  coeff/seg {b['coeff_per_segment']:.2f}  "
              f"FVU {b['fvu']:.4f}  #pre>0/seg {b['n_pos_preact_per_segment']:.0f}  "
              f"-> {b['fvu_ratio_to_sae']:.2f}x the SAE")
        verdict = ("implementations AGREE at T=1, so the T-slope is the real cost of "
                   "window sharing" if b["fvu_ratio_to_sae"] < 1.15 else
                   "implementations DISAGREE at T=1, so part of every gap in this sprint "
                   "is implementation, not architecture")
        print(f"\nverdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
