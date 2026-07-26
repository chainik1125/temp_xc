"""Re-report every arm at a MATCHED dose magnitude instead of at its own best dose.

Two independent reasons to prefer this, and they arrived from different directions.

SELECTION. `at_best` takes an argmax over the dose grid on the same documents used to report
the delta and its standard error. Simulation of that setup (4 doses, 200 documents, per-dose
SEM 0.64, doc-level noise correlated 0.85 across doses) puts the inflation at ~0.00 for a
well-separated peak and 0.19-0.26 for flat or null arms. So the bias flatters the CONTROLS,
not the crosscoder, and the reported gaps are if anything understated -- but the z is
computed from a SEM that does not know a maximum was taken, so it is overstated wherever two
arms are both flat.

LINEARITY. Every ratio the rank framework predicts -- sqrt(r1), sqrt(c), the rank law -- is a
first-order statement. Each arm's own best dose is by construction where that arm saturates,
which is exactly outside the regime the prediction is about. Ratios have to be read at the
SMALLEST dose showing a significant effect, not the largest.

WHAT IS MATCHED AND WHAT IS NOT. The dose MAGNITUDE is matched across arms; the SIGN is still
free per arm, because which class you want to steer toward is something the experimenter
knows and a steering vector's sign is a free parameter. That keeps the fairness fix that
found phase1's inverted arm while removing the magnitude-selection advantage: the choice is
now over two options rather than over the whole grid.

    python scripts/matched_dose_table.py            # all runs
    python scripts/matched_dose_table.py recency    # runs whose name starts with `recency`
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins"

ARMS = ["txc_slab", "sae_broadcast", "tsae_broadcast", "txc_flat", "txc_profile_random",
        "sae_schedule", "sae_schedule_grad", "rank1_best", "grad_rank1", "grad_slab",
        "dom_slab", "random_slab", "random_broadcast"]


def at_dose(arm, mag):
    """Best of the two signs at this dose magnitude, with its standard error."""
    best = None
    for a, v, e in zip(arm["alphas"], arm["delta_margin"], arm["sem"]):
        if abs(abs(a) - mag) > 1e-9:
            continue
        if best is None or v > best[0]:
            best = (v, e)
    return best


def smallest_significant(arm, mags, n_sem=2.0):
    """Smallest dose magnitude at which this arm is significant at n_sem standard errors."""
    for m in mags:
        got = at_dose(arm, m)
        if got and got[0] > n_sem * got[1]:
            return m
    return None


def main(prefix: str = "") -> int:
    files = sorted(p for p in SRC.glob("*.json")
                   if p.stem.startswith(prefix) and "arms" in p.read_text()[:4000])
    if not files:
        print(f"[none] no runs matching {prefix!r}")
        return 1

    print(f"{'run':<24}{'dose':>6}  " + "".join(f"{a[:13]:>15}" for a in ARMS[:6]))
    print("-" * (30 + 15 * 6))
    for p in files:
        r = json.loads(p.read_text())
        arms = r.get("arms") or {}
        if "txc_slab" not in arms:
            continue
        mags = sorted({abs(a) for a in arms["txc_slab"]["alphas"]})
        mag = smallest_significant(arms["txc_slab"], mags)
        if mag is None:
            print(f"{p.stem:<24}{'  n/s':>6}   crosscoder not significant at any dose")
            continue
        cells = []
        for a in ARMS[:6]:
            got = at_dose(arms[a], mag) if a in arms else None
            cells.append(f"{got[0]:+.2f}" if got else "-")
        print(f"{p.stem:<24}{mag:>6.2f}  " + "".join(f"{c:>15}" for c in cells))

    print("\nz-separations recomputed at the matched dose (crosscoder vs each arm):")
    for p in files:
        r = json.loads(p.read_text())
        arms = r.get("arms") or {}
        if "txc_slab" not in arms:
            continue
        mags = sorted({abs(a) for a in arms["txc_slab"]["alphas"]})
        mag = smallest_significant(arms["txc_slab"], mags)
        if mag is None:
            continue
        t = at_dose(arms["txc_slab"], mag)
        bits = []
        for a in ("sae_broadcast", "tsae_broadcast", "txc_flat", "grad_rank1",
                  "sae_schedule_grad", "rank1_best", "sae_schedule"):
            if a not in arms:
                continue
            o = at_dose(arms[a], mag)
            if o is None:
                continue
            bits.append(f"{a}={((t[0] - o[0]) / math.sqrt(t[1] ** 2 + o[1] ** 2)):+.1f}")
        print(f"  {p.stem:<24}a={mag:.2f}  " + "  ".join(bits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
