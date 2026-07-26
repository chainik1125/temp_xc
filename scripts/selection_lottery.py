"""The selection lottery: does the gap survive a selector that cannot favour the crosscoder?

THE OBJECTION THIS ANSWERS. Every headline comparison in this sprint takes best-of-4096 by
READING AUC, and that selector is saturated for the SAE -- `auc_selection` = 1.000 in 9 of 9
cells. When hundreds of latents tie at ceiling, "the SAE's best latent" is an arbitrary draw
from a tied pool, and since this sprint's own finding is that reading and steering
dissociate, an arbitrary draw is the worst possible way to pick a steering vector. The
comparison could be crosscoder-versus-arbitrary rather than crosscoder-versus-SAE.

THE FIX, AND WHY IT IS NOT CIRCULAR. Each arm's latent is chosen by MEASURED steering delta
on a document split drawn after the gradient documents and before the test set, so selection
and reporting never share a document. The candidate shortlist is the union of the top 16 by
first-order gradient alignment and the top 16 by reading AUC -- symmetric across arms, and
deliberately covering both hypotheses about where a good latent hides. The reading half is
the branch that can surface a latent whose effect is curvature-dominated and therefore
invisible to a first-order ranking; without it a null would be undecidable between "no good
SAE latent exists" and "we ranked candidates on the wrong quantity".

WHAT WOULD OVERTURN THE HEADLINE. If the SAE's steering-selected latent closes the gap, the
headline is an artefact of the selector. Registered before the run: I expect it to WIDEN,
because the smoke put the SAE's reading pick at rank 332/4096 and the crosscoder's at
1297/4096 -- proportionally the crosscoder was the one further from its own gradient-best, so
the saturated selector was penalising it more.

REPORTED AT MATCHED DOSE, max over signs. Peak-dose reporting puts every arm at its own
saturation point; signed-positive indexing scores any arm whose correct direction is negative
as a failure. Both errors are in this sprint's own audit trail, so the sign of each arm's
maximising dose is printed beside every number -- if that sign is not constant across arms,
signed indexing would have been comparing arms measured on opposite branches.

    python scripts/selection_lottery.py
"""
import glob
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAT = str(ROOT / "results" / "txc_wins" / "recency_tr_sel_ds*.json")

# Steering-selected vs reading-selected, for each architecture.
PAIRS = [("sae_broadcast", "sae_broadcast_readingsel", "SAE broadcast"),
         ("txc_slab", "txc_slab_readingsel", "crosscoder slab")]


def at_dose(arm, mag):
    """Best of the two SIGNS at this dose MAGNITUDE, with the maximising sign returned.

    The magnitude is matched across arms; the sign is free, because which class you steer
    toward is something the experimenter knows.
    """
    best = None
    for a, v, e in zip(arm["alphas"], arm["delta_margin"],
                       arm.get("sem", [0.0] * len(arm["alphas"]))):
        if abs(abs(a) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e, a)
    if best is None:
        i = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
        return arm["delta_margin"][i], arm.get("sem", [0.0])[0], arm["alphas"][i]
    return best


def peak(arm):
    i = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
    return arm["delta_margin"][i], arm.get("sem", [0.0] * 99)[i], arm["alphas"][i]


def z(a, b):
    va, ea, _ = a
    vb, eb, _ = b
    return (va - vb) / ((ea ** 2 + eb ** 2) ** 0.5 + 1e-12)


def main() -> int:
    files = sorted(glob.glob(PAT))
    if not files:
        print(f"[skip] no files matching {PAT}")
        return 1
    runs = [json.loads(pathlib.Path(f).read_text()) for f in files]
    print(f"{len(runs)} inits, task {runs[0]['task']} -> {runs[0].get('task_test')}, "
          f"{runs[0]['model']} L{runs[0]['layer']}, held-out content "
          f"{runs[0].get('held_out_content')}\n")

    # ---- 1. how bad is the reading selector, and is that stable across inits? -----------
    print("SELECTOR QUALITY -- rank of the READING pick among all 4096 by first-order")
    print("alignment with the metric gradient, and its score as a fraction of the best.\n")
    print(f"  {'init':>5}  {'arm':<12}{'read latent':>12}{'rank/4096':>11}"
          f"{'score':>9}{'best':>9}{'ratio':>8}")
    ratios = {"sae": [], "txc": []}
    ranks = {"sae": [], "txc": []}
    for r in runs:
        sel = r.get("selection") or {}
        for nm in ("sae", "txc"):
            s = sel.get(nm)
            if not s:
                continue
            ratio = s["score_of_reading_pick"] / (s["score_of_gradient_pick"] + 1e-12)
            ratios[nm].append(ratio)
            ranks[nm].append(s["reading_latent_rank_by_gradient"])
            print(f"  {r['dict_seed']:>5}  {nm:<12}{s['reading_latent']:>12}"
                  f"{s['reading_latent_rank_by_gradient']:>11}"
                  f"{s['score_of_reading_pick']:>9.4f}"
                  f"{s['score_of_gradient_pick']:>9.4f}{ratio:>8.3f}")
    for nm in ("sae", "txc"):
        if ranks[nm]:
            print(f"  -> {nm}: rank median {statistics.median(ranks[nm]):.0f} "
                  f"(range {min(ranks[nm])}-{max(ranks[nm])}), "
                  f"alignment ratio median {statistics.median(ratios[nm]):.3f} "
                  f"(range {min(ratios[nm]):.3f}-{max(ratios[nm]):.3f})")
    print("  A rank near 2048 is what an arbitrary draw gives; rank 1 means the reading")
    print("  selector and the gradient selector agree and the selector costs that arm nothing.\n")

    # ---- 2. does measured selection change the verdict? ---------------------------------
    for label, dose_of, dose_name in (
            ("MATCHED DOSE", lambda r: r.get("matched_dose_magnitude", 0.5), "matched"),
            ("PEAK DOSE", None, "peak")):
        print(f"{label} -- steering-selected against reading-selected, "
              f"with the maximising sign in brackets\n")
        print(f"  {'init':>5}  {'arm':<16}{'reading-sel':>14}{'steering-sel':>16}"
              f"{'gain':>9}")
        gaps = []
        for r in runs:
            arms = r["arms"]
            mag = dose_of(r) if dose_of else None
            vals = {}
            for new, old, nm in PAIRS:
                a_new = at_dose(arms[new], mag) if mag else peak(arms[new])
                a_old = at_dose(arms[old], mag) if mag else peak(arms[old])
                vals[new] = a_new
                print(f"  {r['dict_seed']:>5}  {nm:<16}"
                      f"{a_old[0]:>+10.2f}[{a_old[2]:+g}]{a_new[0]:>+12.2f}[{a_new[2]:+g}]"
                      f"{a_new[0] - a_old[0]:>+9.2f}")
            zz = z(vals["txc_slab"], vals["sae_broadcast"])
            gaps.append((vals["txc_slab"][0], vals["sae_broadcast"][0], zz))
            print(f"  {'':>5}  {'-> crosscoder vs SAE, both steering-selected':<48}"
                  f"z = {zz:+.1f}")
        if gaps:
            print(f"\n  {dose_name} dose, all inits: crosscoder "
                  f"{min(g[0] for g in gaps):+.2f} to {max(g[0] for g in gaps):+.2f}, "
                  f"SAE {min(g[1] for g in gaps):+.2f} to {max(g[1] for g in gaps):+.2f}, "
                  f"z {min(g[2] for g in gaps):+.1f} to {max(g[2] for g in gaps):+.1f}\n")

    # ---- 3. the sqrt(c) consistency check ------------------------------------------------
    # sqrt(c) is the exact first-order reach of the BEST CONCEIVABLE broadcast direction,
    # relative to the optimal write. `broadcast_optimal` is that direction measured rather
    # than predicted, so its delta should track sqrt(c) x delta(grad_slab) to first order.
    # Agreement validates the bound empirically; disagreement bounds how far into the
    # nonlinear regime the matched dose already is, which is worth knowing either way.
    print("SQRT(C) CHECK -- the best conceivable broadcast write, measured against its "
          "first-order prediction\n")
    print(f"  {'init':>5}{'sqrt(c)':>9}{'grad_slab':>11}{'predicted':>11}"
          f"{'broadcast_optimal':>19}{'ratio':>8}")
    for r in runs:
        arms, sc = r["arms"], r.get("sqrt_c_grad")
        if sc is None or "broadcast_optimal" not in arms:
            print(f"  {r['dict_seed']:>5}  [absent -- run predates the arm]")
            continue
        mag = r.get("matched_dose_magnitude", 0.5)
        gs = at_dose(arms["grad_slab"], mag)[0]
        bo = at_dose(arms["broadcast_optimal"], mag)[0]
        pred = sc * gs
        print(f"  {r['dict_seed']:>5}{sc:>9.4f}{gs:>11.2f}{pred:>11.2f}"
              f"{bo:>19.2f}{bo / (pred + 1e-12):>8.2f}")
    print("\n  A ratio near 1 says the first-order account holds at the matched dose. Far")
    print("  from 1 says the dose is already outside the linear regime the bound describes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
