"""Which arms did we measure, store, and then never look at?

This sprint's recurring failure is a quantity that is present, correct, and unread. Eight
instances were found by reading our own code; the ninth and tenth were found by joining a
claim in the summary back to the file it cites. This is the mechanical version of that
second check, and unlike the others it can be run before anyone writes a claim.

An arm that is in `arms` but has no `txc_slab_vs_<arm>` entry in `z` was steered, scored and
stored, and then took no part in any verdict. `win` is computed from a fixed subset, so an
uncompared arm cannot make `win` false no matter what it says. That is exactly how the
crosscoder came to be described as beating `sae_schedule` in a document while losing to it,
at z = -20.6, in the file the sentence cites.

Prints the coverage table, then -- for arms present in most files and compared in none --
computes the comparison that was never made, at both dose magnitudes, max over signs.

    python scripts/unread_arms.py
"""
import collections
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
DOSES = (0.5, 1.0)


def at(arm, mag):
    """Max over signs at a dose magnitude; None if the grid lacks it."""
    best = None
    for al, v, e in zip(arm["alphas"], arm["delta_margin"],
                        arm.get("sem", [0.0] * len(arm["alphas"]))):
        if abs(abs(al) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e)
    return best


def main() -> int:
    files, present, compared = [], collections.Counter(), collections.Counter()
    for f in sorted(glob.glob(str(RES / "*.json"))):
        try:
            d = json.loads(pathlib.Path(f).read_text())
        except Exception:
            continue
        z, arms = d.get("z") or d.get("z_peak_dose") or {}, d.get("arms") or {}
        if not z or "txc_slab" not in arms:
            continue
        files.append((f, d))
        for a in arms:
            if a == "txc_slab":
                continue
            present[a] += 1
            if f"txc_slab_vs_{a}" in z:
                compared[a] += 1

    print(f"{len(files)} result files carrying both `arms` and `z`\n")
    print(f"  {'arm':<28}{'measured':>10}{'compared':>10}{'unread':>9}")
    gaps = []
    for a in sorted(present):
        miss = present[a] - compared[a]
        flag = "  <-- never compared" if compared[a] == 0 else ""
        print(f"  {a:<28}{present[a]:>10}{compared[a]:>10}{miss:>9}{flag}")
        if compared[a] == 0 and present[a] >= 10:
            gaps.append(a)

    for a in gaps:
        print(f"\nTHE COMPARISON NOBODY MADE: txc_slab vs {a}, z, max over signs")
        print(f"  {'cell':<34}" + "".join(f"{'|a|=' + str(m):>11}" for m in DOSES)
              + f"{'txc / ' + a:>26}")
        lose = tot = 0
        for f, d in files:
            arms = d["arms"]
            if a not in arms:
                continue
            zs = []
            for m in DOSES:
                t, q = at(arms["txc_slab"], m), at(arms[a], m)
                zs.append(None if not t or not q else
                          (t[0] - q[0]) / ((t[1] ** 2 + q[1] ** 2) ** 0.5 + 1e-12))
            t, q = at(arms["txc_slab"], DOSES[0]), at(arms[a], DOSES[0])
            if zs[0] is None:
                continue
            tot += 1
            lose += zs[0] < 0
            cells = "".join(f"{v:>11.2f}" if v is not None else f"{'--':>11}" for v in zs)
            print(f"  {pathlib.Path(f).name:<34}{cells}{t[0]:>+14.2f} /{q[0]:>+8.2f}")
        print(f"  -> crosscoder is behind {a} in {lose} of {tot} cells at |a|={DOSES[0]}")
    if not gaps:
        print("\nNo arm is measured widely and compared nowhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
