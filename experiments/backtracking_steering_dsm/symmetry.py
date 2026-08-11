"""Split each steering curve into its nonspecific and directional components.

    uv run --no-sync python experiments/backtracking_steering_dsm/symmetry.py \
        --curves <curves.json> [--control control_random]

Several arms raise the genuine-event count in BOTH steering directions. That is
what perturbing the residual stream by a given norm does regardless of which way
you push, so a raw peak |Delta-gc| mixes two different things together. Around
alpha = 0 every curve decomposes uniquely into

    sym(a)  = (gc(+a) + gc(-a)) / 2 - gc(0)     even part, magnitude effect
    anti(a) = (gc(-a) - gc(+a)) / 2             odd part, direction effect

`sym` is the part a norm-matched random direction can also produce; `anti` is
the part that requires the direction to mean something. The sign convention puts
`anti > 0` when NEGATIVE alpha induces more backtracking, which is the direction
the mined decoders actually steer.

Only |alpha| values whose BOTH cells clear the Sonnet floor are averaged, so the
decomposition never mixes a coherent cell with a degenerate one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


def decompose(curve: list[dict], require_coherent: bool = True) -> dict:
    """-> per-|alpha| sym/anti plus their averages over the usable range."""
    by = {x["magnitude"]: x for x in curve}
    gc0 = by.get(0.0, {}).get("gc", float("nan"))
    mags = sorted({abs(m) for m in by if m != 0.0})
    rows, sym_v, anti_v = [], [], []
    for a in mags:
        hi, lo = by.get(a), by.get(-a)
        if hi is None or lo is None:
            continue
        ok = (hi["cell_coh_sonnet"] and lo["cell_coh_sonnet"])
        if require_coherent and not ok:
            continue
        sym = (hi["gc"] + lo["gc"]) / 2 - gc0
        anti = (lo["gc"] - hi["gc"]) / 2
        rows.append({"abs_magnitude": a, "sym": sym, "anti": anti,
                     "gc_pos": hi["gc"], "gc_neg": lo["gc"],
                     "both_coherent": ok,
                     "words_pos": hi["mean_words"], "words_neg": lo["mean_words"]})
        sym_v.append(sym)
        anti_v.append(anti)
    return {"gc_at_zero": gc0, "per_magnitude": rows,
            "mean_sym": _mean(sym_v), "mean_anti": _mean(anti_v),
            "max_abs_sym": max((abs(v) for v in sym_v), default=float("nan")),
            "max_abs_anti": max((abs(v) for v in anti_v), default=float("nan")),
            "n_paired_magnitudes": len(rows)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--curves", type=Path, required=True)
    p.add_argument("--control", default="control_random")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    curves = json.loads(a.curves.read_text())
    # Both views are printed. Restricting to pairs where BOTH signs clear the
    # Sonnet floor is the conservative reading, but it keeps only the smallest
    # |alpha| for the asymmetric arms -- precisely where the effect has not
    # started -- because their coherent range is itself lopsided. The
    # all-magnitudes view is therefore shown next to it, with the count of
    # pairs that were fully coherent so the reader can see how much of it rests
    # on degenerate cells.
    dec = {tag: decompose(c, require_coherent=False) for tag, c in curves.items()}
    dec_coh = {tag: decompose(c, require_coherent=True) for tag, c in curves.items()}
    ctl = dec.get(a.control)

    print("all paired magnitudes:")
    print(f"{'source':38s} {'gc(0)':>6} {'mean sym':>9} {'mean anti':>10} "
          f"{'max|anti|':>10} {'pairs':>6} {'coh pairs':>10}")
    for tag, d in sorted(dec.items(), key=lambda kv: -abs(kv[1]["mean_anti"])):
        print(f"{tag:38s} {d['gc_at_zero']:>6.2f} {d['mean_sym']:>+9.3f} "
              f"{d['mean_anti']:>+10.3f} {d['max_abs_anti']:>10.3f} "
              f"{d['n_paired_magnitudes']:>6d} "
              f"{dec_coh[tag]['n_paired_magnitudes']:>10d}")

    print("\nrestricted to pairs where both signs clear the Sonnet floor:")
    print(f"{'source':38s} {'mean sym':>9} {'mean anti':>10} {'pairs':>6}")
    for tag, d in sorted(dec_coh.items(), key=lambda kv: -abs(kv[1]["mean_anti"])):
        print(f"{tag:38s} {d['mean_sym']:>+9.3f} {d['mean_anti']:>+10.3f} "
              f"{d['n_paired_magnitudes']:>6d}")

    if ctl:
        print(f"\nexcess over {a.control} (this arm minus the control, "
              f"matched |alpha| only):")
        print(f"{'source':38s} {'excess sym':>11} {'excess anti':>12}")
        cs = {r["abs_magnitude"]: r for r in ctl["per_magnitude"]}
        for tag, d in sorted(dec.items(), key=lambda kv: -abs(kv[1]["mean_anti"])):
            if tag == a.control:
                continue
            es = [r["sym"] - cs[r["abs_magnitude"]]["sym"]
                  for r in d["per_magnitude"] if r["abs_magnitude"] in cs]
            ea = [r["anti"] - cs[r["abs_magnitude"]]["anti"]
                  for r in d["per_magnitude"] if r["abs_magnitude"] in cs]
            print(f"{tag:38s} {_mean(es):>+11.3f} {_mean(ea):>+12.3f}")
    else:
        print(f"\n[warn] control '{a.control}' not in curves; "
              f"no excess computed", flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
