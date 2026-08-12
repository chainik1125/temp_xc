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
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


# --------------------------------------------------------------------------
# bootstrap CI on the excess directional component
# --------------------------------------------------------------------------

def _per_prompt(wave_dir: Path) -> dict:
    """-> {tag: {(prompt_id, magnitude): genuine_count}}, judged rows only."""
    out = {}
    for rp in sorted(wave_dir.glob("rows__*.json")):
        tag = rp.name[len("rows__"):-len(".json")]
        jp = wave_dir / f"judged__{tag}.json"
        if not jp.exists():
            continue
        try:
            rows = json.loads(rp.read_text())["rows"]
            judged = json.loads(jp.read_text())
        except ValueError:
            continue
        d = {}
        for i, r in enumerate(rows):
            g = int(judged.get(str(i), {}).get("genuine_count", -1))
            if g >= 0:
                d[(r["prompt_id"], r["magnitude"])] = g
        out[tag] = d
    return out


def _mean_anti(d: dict, pids: list[str], mags: list[float]) -> float:
    """Mean odd component over |mags|, evaluated on the given prompt multiset."""
    vals = []
    for a in mags:
        hi = [d[(p, a)] for p in pids if (p, a) in d]
        lo = [d[(p, -a)] for p in pids if (p, -a) in d]
        if hi and lo:
            vals.append((_mean(lo) - _mean(hi)) / 2)
    return _mean(vals)


def bootstrap_excess_anti(wave_dir: Path, control: str, mags: list[float],
                          n_resamples: int = 2000, seed: int = 42,
                          ci: float = 0.95) -> dict:
    """Percentile CI on (arm mean_anti - control mean_anti).

    Prompts are resampled with replacement and the SAME resampled prompt set is
    used for the arm and for the control, so the pairing that makes the
    subtraction meaningful is preserved -- resampling them independently would
    inflate the interval with variance the comparison does not actually carry.
    """
    per = _per_prompt(wave_dir)
    if control not in per:
        return {}
    pids = sorted({p for p, _ in per[control]})
    rng = random.Random(seed)
    draws = [[pids[rng.randrange(len(pids))] for _ in pids]
             for _ in range(n_resamples)]
    out = {}
    for tag, d in per.items():
        if tag == control:
            continue
        point = _mean_anti(d, pids, mags) - _mean_anti(per[control], pids, mags)
        vals = sorted(_mean_anti(d, s, mags) - _mean_anti(per[control], s, mags)
                      for s in draws)
        lo = vals[int((1 - ci) / 2 * n_resamples)]
        hi = vals[int((1 + ci) / 2 * n_resamples) - 1]
        out[tag] = {"excess_anti": point, "ci95": [lo, hi],
                    "excludes_zero": (lo > 0) or (hi < 0)}
    return out


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
    p.add_argument("--wave-dir", type=Path, default=None,
                   help="rows+judged dir; enables the bootstrap CI on excess anti")
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

    if ctl and a.wave_dir:
        mags = sorted({r["abs_magnitude"] for r in ctl["per_magnitude"]})
        boot = bootstrap_excess_anti(a.wave_dir, a.control, mags)
        print(f"\nexcess anti with prompt-bootstrap 95% CI "
              f"(|alpha| in {[f'{m:g}' for m in mags]}):")
        print(f"{'source':38s} {'excess anti':>12} {'95% CI':>20} {'sig':>5}")
        for tag, b in sorted(boot.items(), key=lambda kv: -kv[1]["excess_anti"]):
            print(f"{tag:38s} {b['excess_anti']:>+12.3f} "
                  f"[{b['ci95'][0]:>+7.3f}, {b['ci95'][1]:>+7.3f}] "
                  f"{'yes' if b['excludes_zero'] else 'no':>5}")
        for tag, d in dec.items():
            if tag in boot:
                d["bootstrap_excess_anti"] = boot[tag]

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
