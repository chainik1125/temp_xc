"""Does the crosscoder's advantage over the BEST POSSIBLE constant write depend on task rank?

The selection runs showed that on instruction position the crosscoder exceeds the entire
broadcast form by only 0.95-1.20x. That is one task, and one task cannot distinguish two very
different readings:

  A  the crosscoder's temporal freedom is worth little IN GENERAL, and the headline margin
     over `sae_broadcast` is a fact about the SAE's dictionary and our selector;
  B  instruction position happens to be a task a constant write nearly solves, and on tasks
     that genuinely need rank >= 2 the form advantage appears.

`r1` — the share of the metric gradient captured by its best rank-1 approximation — separates
them, and it is measured before any dictionary is trained. Under reading B the ratio

    txc_slab / broadcast_optimal

should rise as `r1` falls. Under reading A it should be flat near 1.

REGISTERED, before the runs land: I expect B, i.e. a rising relationship, with `rot_m12`
(r1 = 0.177) the largest. It is the one cell in the set built so that rank-1 writes are poor
while its CONSTANT share is the same as instruction position's (c = 0.033 against 0.034), so
it isolates rank from the constant-write bound rather than confounding them.

FALSIFIER, and it is the more consequential outcome: a flat relationship means the crosscoder
never meaningfully exceeds the best constant write on ANY task we have built, including one
designed to reward exactly that — which would make the whole headline a statement about
dictionaries and selectors and not about temporal expressiveness at all.

`broadcast_optimal` is built from the metric gradient, so it is a SUPERVISED reference and not
an arm a practitioner holds. The comparison bounds what the crosscoder's FORM buys; it is not
a claim that a per-token dictionary could reach that line.

    python scripts/form_advantage_vs_r1.py
"""
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"


def at(arm, mag):
    best = None
    for al, v, e in zip(arm["alphas"], arm["delta_margin"],
                        arm.get("sem", [0.0] * len(arm["alphas"]))):
        if abs(abs(al) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e, al)
    return best


def main() -> int:
    rows = []
    for f in sorted(glob.glob(str(RES / "*_sel_ds*.json"))):
        d = json.loads(pathlib.Path(f).read_text())
        a = d.get("arms", {})
        if "broadcast_optimal" not in a or "rank_grad" not in d:
            continue
        mag = d.get("matched_dose_magnitude", 0.5)
        t, b = at(a["txc_slab"], mag), at(a["broadcast_optimal"], mag)
        g = at(a["grad_slab"], mag)
        if not t or not b:
            continue
        z = (t[0] - b[0]) / ((t[1] ** 2 + b[1] ** 2) ** 0.5 + 1e-12)
        rows.append({"task": d["task"], "seed": d["dict_seed"],
                     "r1": d["rank_grad"]["r1"], "c": d["rank_grad"]["c"],
                     "txc": t[0], "bo": b[0], "ratio": t[0] / b[0] if b[0] else float("nan"),
                     "z": z, "grad": g[0] if g else float("nan"), "dose": mag})
    if not rows:
        print("[skip] no selection runs with broadcast_optimal yet")
        return 1

    rows.sort(key=lambda r: (r["r1"], r["task"], r["seed"]))
    print("crosscoder against the best constant write that exists, ordered by r1\n")
    print(f"  {'task':<18}{'seed':>5}{'r1':>7}{'c':>7}{'dose':>6}"
          f"{'txc':>9}{'bcast_opt':>11}{'ratio':>8}{'z':>7}{'txc/grad':>10}")
    for r in rows:
        print(f"  {r['task']:<18}{r['seed']:>5}{r['r1']:>7.3f}{r['c']:>7.3f}{r['dose']:>6.2g}"
              f"{r['txc']:>+9.2f}{r['bo']:>+11.2f}{r['ratio']:>8.2f}{r['z']:>+7.1f}"
              f"{r['txc'] / r['grad']:>10.3f}")

    # Per task, then the relationship across tasks.
    tasks = {}
    for r in rows:
        tasks.setdefault(r["task"], []).append(r)
    print("\n  per-task median ratio, ordered by r1 (rising = the form advantage is real)")
    pts = []
    for t, rs in sorted(tasks.items(), key=lambda kv: kv[1][0]["r1"]):
        med = sorted(x["ratio"] for x in rs)[len(rs) // 2]
        pts.append((rs[0]["r1"], med))
        print(f"    r1 {rs[0]['r1']:.3f}  {t:<18} ratio {med:.2f}  ({len(rs)} init"
              f"{'s' if len(rs) > 1 else ''})")
    if len(pts) >= 3:
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        syy = sum((p[1] - my) ** 2 for p in pts)
        rr = sxy / ((sxx * syy) ** 0.5 + 1e-12)
        print(f"\n    slope of ratio on r1 = {sxy / (sxx + 1e-12):+.2f}, "
              f"Pearson r = {rr:+.3f} over {n} tasks")
        print("    negative slope supports the rank reading; a flat or positive one says the")
        print("    crosscoder's form buys nothing even where the task is built to reward it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
