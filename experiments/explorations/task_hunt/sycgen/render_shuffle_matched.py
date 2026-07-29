"""Render the rebuttal figure for the sparsity-matched sycgen shuffle ablation.

Han (2026-07-29): *"for the rebuttal we care only about the gap between the
trained TXC on order preserved vs the trained TXC on shuffled; untrained twins
are NOT of the same level."*

He is right and the point is not cosmetic. The twin reaches **0.058** ordered
recovery at T=16 against the trained model's **0.578** — its gap is a
difference between two near-chance numbers, so it is not commensurable with
the trained gap and does not belong on the same axis. mac-d raised exactly
this as their own qualifier; this figure acts on it.

The matched run shipped a TABLE only (`tab_sycgen_shuffle_matched.md`) — the
acceptance gate named one and never asked for a plot, so the
rebuttal-relevant figure did not exist. This is that figure.

WHAT IS PLOTTED: trained TXC, ordered vs shuffled, mean +- s.d. over 3 seeds,
at every T, **at matched sparsity**. Nothing else. No twin, no baselines.

DRAW: `redraw` is primary here because this is a CROSS-T reading, and the
`plain` draw leaves `1/T!` of rows unshuffled (50% at T=2), which attenuates
the small-T gap by construction. The card fixed that rule before the run.
`plain` is drawn faintly for honesty, not for comparison.

    .venv/bin/python -m experiments.explorations.task_hunt.sycgen.render_shuffle_matched
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
RES = Path(__file__).resolve().parent / "results"
OUT = ROOT / "figs_writeup" / "fig_sycgen_shuffle_matched"
ANCHOR = 0.4819          # per-token BatchTopK SAE at T=1 (sycgen anchor trio)
PRIMARY = "redraw"       # cross-T reading; see module docstring


def load():
    rows = []
    for f in sorted(glob.glob(str(RES / "shuffle_matched.shard*.json"))):
        rows += json.loads(Path(f).read_text())["rows"]
    if not rows:
        raise SystemExit(f"no shards in {RES}")
    return [r for r in rows if r["arm"] == "txc" and r["weights"] == "trained"]


def series(rows, draw):
    """Per-T means, and the PAIRED s.d. of the gap.

    ⚑ ordered and shuffled are measured on the SAME seed and the SAME model,
    so they are paired. Propagating the gap's error as
    `sqrt(sd_o**2 + sd_s**2)` treats them as independent and materially
    OVERSTATES the uncertainty — the seed-to-seed swing in *level* is common
    to both arms and cancels in the difference. The gap's s.d. is the s.d. of
    the per-seed gaps, which is what is plotted.
    """
    o, s, g = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        if r["draw"] != draw:
            continue
        o[r["T"]].append(r["recovery_ordered"])
        s[r["T"]].append(r["recovery_shuffled_fixedprobe"])
        g[r["T"]].append(r["recovery_ordered"] - r["recovery_shuffled_fixedprobe"])
    Ts = sorted(o)
    return (Ts,
            [mean(o[t]) for t in Ts], [pstdev(o[t]) for t in Ts],
            [mean(s[t]) for t in Ts], [pstdev(s[t]) for t in Ts],
            [mean(g[t]) for t in Ts], [pstdev(g[t]) for t in Ts])


def main() -> int:
    rows = load()
    Ts, om, osd, sm, ssd, gm, gsd = series(rows, PRIMARY)
    *_, sm2, _, _, _ = series(rows, "plain")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.35, 1]})

    ax.errorbar(Ts, om, yerr=osd, marker="o", lw=2, capsize=3,
                color="#1f4e79", label="TXC, order preserved")
    ax.errorbar(Ts, sm, yerr=ssd, marker="s", lw=2, capsize=3,
                color="#c0392b", label="TXC, shuffled within window")
    ax.plot(Ts, sm2, ls=":", lw=1.2, color="#c0392b", alpha=0.55,
            label="shuffled (plain draw — attenuated at small T)")
    ax.axhline(ANCHOR, ls="--", lw=1.2, color="#666")
    ax.annotate(f"per-token SAE  {ANCHOR:.3f}", (Ts[0], ANCHOR),
                textcoords="offset points", xytext=(4, -13),
                fontsize=8.5, color="#666")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel("window size $T$")
    ax.set_ylabel("recovery")
    ax.set_title("sycgen — trained TXC, ordered vs shuffled\n"
                 "at matched sparsity (3 seeds, mean ± s.d.)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)

    gaps = gm          # paired per-seed gaps, not a difference of means
    ax2.bar([str(t) for t in Ts], gaps, yerr=gsd, capsize=3,
            color="#1f4e79", alpha=0.85)
    ax2.axhline(0, lw=1, color="k")
    ax2.set_xlabel("window size $T$")
    ax2.set_ylabel("ordered − shuffled  (paired s.d.)")
    ax2.set_title("the gap itself", fontsize=10.5)
    ax2.grid(axis="y", alpha=0.25)
    for i, g in enumerate(gaps):
        ax2.annotate(f"{g:+.3f}", (i, g), ha="center",
                     textcoords="offset points",
                     xytext=(0, 4 if g >= 0 else -12), fontsize=8.5)

    fig.suptitle("Trained TXC only — untrained twins are NOT on this axis "
                 "(they reach 0.058 ordered at T=16 vs 0.578 trained)",
                 fontsize=9, y=1.0, color="#444")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=170, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}.png / .pdf")
    print(f"draw={PRIMARY}  " + "  ".join(
        f"T{t}: {a:.4f}->{b:.4f} ({a-b:+.4f})" for t, a, b in zip(Ts, om, sm)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
