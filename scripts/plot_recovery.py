"""True-feature recovery against window length, per feature extent.

Ground-truth features have a direction u and a profile p spanning L consecutive segments.
A T-window can cover at most T contiguous entries of p, so the best cosine any T-slab can
reach against the full atom p (x) u is fixed by geometry:

    ceiling(T, p) = || largest contiguous T-chunk of p || / ||p||

which for a flat profile is sqrt(min(T, L) / L). Dashed lines are that ceiling; solid lines
are what training actually recovers. The distance between them is optimisation; the ceiling
itself is not something better training can beat.

Reads results/dict_bench/recovery.json (written by recovery_local.py).
"""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "dict_bench" / "recovery.json"
OUT = ROOT / "plots" / "2026-07-25_dictbench" / "recovery.png"

# Wong palette, one hue per feature extent.
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def main() -> int:
    if not SRC.exists():
        print(f"[skip] {SRC} not written yet")
        return 1
    r = json.loads(SRC.read_text())
    rows, extents, Ts = r["rows"], r["extents"], r["Ts"]
    rows = sorted(rows, key=lambda x: x["T"])

    # Windows far longer than any feature present degrade training rather than measure
    # geometry; those points are shown but excluded from the saturation read-out.
    fvu = {x["T"]: x["fvu"] for x in rows}
    ok = [x["T"] for x in rows if fvu[x["T"]] < 3 * min(fvu.values())]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    for i, L in enumerate(extents):
        c = COLOURS[i % len(COLOURS)]
        ax.plot([x["T"] for x in rows], [x["recovery"][str(L)] for x in rows],
                "o-", color=c, lw=2.2, ms=6, label=f"extent L={L}")
        ax.plot([x["T"] for x in rows], [x["ceiling"][str(L)] for x in rows],
                "--", color=c, lw=1.4, alpha=0.75)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel("window length T")
    ax.set_ylabel("true-feature recovery  (max cosine)")
    ax.set_title("Recovery rises with T, along a geometric ceiling")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=8.5, ncol=2)
    ax.text(0.97, 0.05, "dashed = analytic ceiling", transform=ax.transAxes,
            ha="right", fontsize=8.5, style="italic")

    # ---------------- right: the per-token limit, which is the transferable point -------
    ax = axes[1]
    t1 = rows[0]
    xs = list(range(len(extents)))
    ax.bar([x - 0.2 for x in xs], [t1["recovery"][str(L)] for L in extents],
           width=0.38, color="#999999", label=f"per-token dictionary (T={t1['T']})")
    best_T = max(ok)
    tb = next(x for x in rows if x["T"] == best_T)
    ax.bar([x + 0.2 for x in xs], [tb["recovery"][str(L)] for L in extents],
           width=0.38, color="#0072B2", label=f"window dictionary (T={best_T})")
    for i, L in enumerate(extents):
        ax.plot([i - 0.39, i - 0.01], [t1["ceiling"][str(L)]] * 2, "k--", lw=1.4)
    ax.plot([], [], "k--", lw=1.4, label=f"ceiling at T={t1['T']}")
    ax.set_xticks(xs); ax.set_xticklabels([f"L={L}" for L in extents])
    ax.set_xlabel("temporal extent of the true feature")
    ax.set_ylabel("true-feature recovery  (max cosine)")
    ax.set_title("A per-token dictionary is capped by the feature's extent")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, lw=0.6, axis="y")
    ax.legend(loc="lower left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    print("[saved]", OUT)

    print("\nrecovery / ceiling, by extent:")
    header = "".join(f"T={x['T']}".rjust(16) for x in rows)
    print("  extent" + header)
    for L in extents:
        cells = ""
        for x in rows:
            cell = f"{x['recovery'][str(L)]:.3f} / {x['ceiling'][str(L)]:.3f}"
            cells += cell.rjust(16)
        print(f"  L={L:<5}" + cells)
    excluded = [x["T"] for x in rows if x["T"] not in ok]
    if excluded:
        print(f"\nexcluded from the saturation read-out (training degraded, FVU > 3x best): "
              f"T={excluded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
