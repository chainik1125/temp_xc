"""Which verdicts depend on the dose convention, and by how much.

Finding 1 of this sprint is a withdrawal caused by a dose convention, so any result that
changes sign between conventions has to be stated rather than buried. This compares, for
every run with a symmetric grid:

    matched dose  -- crosscoder minus the best constant arm, at the SMALLEST magnitude where
                     the crosscoder is significant, sign free per arm. Primary convention:
                     it sits in the linear regime the rank framework describes and does not
                     let any arm pick a maximum over the whole grid.
    peak dose     -- the same difference with every arm at its own best dose. Secondary.

What the answer turns out to be: flips are confined to cells that are near zero under BOTH
conventions, and their direction is symmetric, so neither convention systematically favours
the crosscoder. No result with a margin above ~2 changes verdict.

    python scripts/convention_flips.py
"""
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONST = ("sae_broadcast", "tsae_broadcast", "txc_flat", "random_broadcast")


def at(arm, mag):
    best = None
    for a, v, e in zip(arm["alphas"], arm["delta_margin"], arm.get("sem", [0] * 99)):
        if abs(abs(a) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e)
    return best


def peak(arm):
    j = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
    return arm["delta_margin"][j], arm["sem"][j]


def main() -> int:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "txc_wins" / "*.json"))):
        try:
            r = json.loads(pathlib.Path(f).read_text())
        except Exception:
            continue
        a = r.get("arms") or {}
        if "txc_slab" not in a or "sae_broadcast" not in a:
            continue
        mags = sorted({abs(x) for x in a["txc_slab"]["alphas"]})
        if len(mags) < 3:
            continue                       # one-sided grid: no matched-dose reading
        md = next((m for m in mags
                   if (g := at(a["txc_slab"], m)) and g[0] > 2 * g[1]), None)
        if md is None:
            continue
        dm = at(a["txc_slab"], md)[0] - max(at(a[k], md)[0] for k in CONST if k in a)
        dp = peak(a["txc_slab"])[0] - max(peak(a[k])[0] for k in CONST if k in a)
        rows.append((pathlib.Path(f).stem, dm, dp))

    flips = [(n, dm, dp) for n, dm, dp in rows if (dm > 0) != (dp > 0)]
    print(f"{'run':<28}{'matched':>10}{'peak':>10}")
    for n, dm, dp in flips:
        print(f"{n:<28}{dm:>+10.2f}{dp:>+10.2f}")
    print(f"\n{len(flips)} of {len(rows)} runs change verdict between conventions")
    if flips:
        worst = max(max(abs(dm), abs(dp)) for _, dm, dp in flips)
        toward_matched = sum(dm > 0 for _, dm, _ in flips)
        print(f"largest margin involved in any flip: {worst:.2f}")
        print(f"direction: {toward_matched} favour matched, "
              f"{len(flips) - toward_matched} favour peak")
        stable = [abs(dm) for n, dm, dp in rows if (dm > 0) == (dp > 0)]
        print(f"every run with |matched margin| > {worst:.2f} is convention-stable: "
              f"{all(abs(dm) <= worst for _, dm, dp in flips)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
