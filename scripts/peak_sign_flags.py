"""The sign detector the summary describes, as code that runs.

The summary's methodology section proposes this and states what it would have caught:

    Print, beside every reported number, the SIGN of the dose at which each arm's |delta|
    is maximal. If that sign is not constant across arms, signed-positive indexing is
    silently comparing arms measured on opposite branches.

It was described but never implemented, and a detector that exists only in prose has not
actually been run on anything -- which is the same gap this sprint spent the night
documenting in other places. This is it as code, over every result file with a symmetric
dose grid.

WHAT "FLAGGED" MEANS. For each arm, take the dose maximising |delta| and record its sign.
A cell is flagged when those signs are not all equal, because then reading every arm at
`+alpha` compares arms sitting on opposite branches of a signed effect -- the error that
withdrew the previous sprint's headline, appeared in a figure script at hour eight, and
appeared twice more in a red-team pass.

The output also reports how much the error would have cost: `signed_gap` is the crosscoder
minus the best constant arm read at signed +alpha, `absmax_gap` is the same with the sign
free per arm. Where those disagree in sign, the convention alone decides the verdict.

    python scripts/peak_sign_flags.py
"""
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
CONST = ("sae_broadcast", "tsae_broadcast", "txc_flat", "random_broadcast")


def peak_sign(arm):
    """Sign of the dose maximising |delta|, and that delta."""
    i = max(range(len(arm["delta_margin"])), key=lambda i: abs(arm["delta_margin"][i]))
    a = arm["alphas"][i]
    return (1 if a > 0 else -1), arm["delta_margin"][i], a


def best_at_signed(arm, mag):
    """Delta at +mag only -- the buggy convention, reproduced deliberately."""
    for a, v in zip(arm["alphas"], arm["delta_margin"]):
        if abs(a - mag) < 1e-9:
            return v
    return None


def best_at_absmax(arm, mag):
    """Delta at |alpha| = mag, sign free -- the correct convention."""
    vs = [v for a, v in zip(arm["alphas"], arm["delta_margin"])
          if abs(abs(a) - mag) < 1e-9]
    return max(vs) if vs else None


def main() -> int:
    flagged, total = [], 0
    for f in sorted(glob.glob(str(RES / "*.json"))):
        try:
            d = json.loads(pathlib.Path(f).read_text())
        except Exception:
            continue
        arms = d.get("arms") or {}
        if "txc_slab" not in arms:
            continue
        al = arms["txc_slab"]["alphas"]
        # Only meaningful on a symmetric grid; a one-sided grid cannot be checked at all,
        # which is itself the point of Finding 1.
        if not (any(a < 0 for a in al) and any(a > 0 for a in al)):
            continue
        total += 1
        signs = {n: peak_sign(a)[0] for n, a in arms.items()
                 if n == "txc_slab" or n in CONST}
        if len(set(signs.values())) == 1:
            continue
        mag = min(abs(a) for a in al if a != 0)
        t_s, t_a = best_at_signed(arms["txc_slab"], mag), best_at_absmax(arms["txc_slab"], mag)
        cs = [best_at_signed(arms[n], mag) for n in CONST if n in arms]
        ca = [best_at_absmax(arms[n], mag) for n in CONST if n in arms]
        cs = [v for v in cs if v is not None]
        ca = [v for v in ca if v is not None]
        if t_s is None or not cs:
            continue
        flagged.append((pathlib.Path(f).name, d.get("task"), signs,
                        t_s - max(cs), t_a - max(ca)))

    print(f"{total} result files with a symmetric dose grid; "
          f"{len(flagged)} flagged\n")
    print(f"  {'file':<34}{'signed gap':>12}{'sign-free gap':>15}  arms on the minus branch")
    verdict_changes = 0
    for name, task, signs, gs, ga in flagged:
        minus = [n for n, s in signs.items() if s < 0]
        if (gs > 0) != (ga > 0):
            verdict_changes += 1
        mark = "  <-- VERDICT FLIPS" if (gs > 0) != (ga > 0) else ""
        print(f"  {name[:33]:<34}{gs:>+12.2f}{ga:>+15.2f}  {', '.join(minus)}{mark}")
    print(f"\n  {verdict_changes} of {len(flagged)} flagged cells change verdict between the two")
    print("  conventions at the smallest dose. Every flagged cell is one where reading all")
    print("  arms at +alpha compares arms measured on opposite branches of a signed effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
