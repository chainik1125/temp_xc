"""Compact table over every task run in results/txc_wins/.

One row per run: realised coefficients per segment for each dictionary (the comparison axis
-- never nominal k), best-single-latent reading AUC on the HELD-OUT split, best steering
delta for the crosscoder and for each baseline, and the z-separations that decide the cell.

    python scripts/summarise_txc_wins.py            # all runs
    python scripts/summarise_txc_wins.py recency    # runs whose name starts with `recency`
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "txc_wins"


def best(arm):
    j = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
    return arm["delta_margin"][j], arm["sem"][j], arm["alphas"][j]


def main(prefix: str = "") -> int:
    files = sorted(p for p in SRC.glob("*.json") if p.stem.startswith(prefix))
    if not files:
        print(f"[none] no runs matching {prefix!r} in {SRC}")
        return 1

    hdr = (f"{'run':<22}{'coeff/seg sae|txc|tsae':>24}"
           f"{'read auc sae|txc|tsae':>24}{'txc_slab':>12}{'sae':>9}"
           f"{'flat':>9}{'prof_rnd':>10}{'dom':>9}  win")
    print(hdr)
    print("-" * len(hdr))
    for p in files:
        r = json.loads(p.read_text())
        sp, rd, arms = r.get("sparsity", {}), r.get("reading", {}), r.get("arms", {})

        def s(key):
            v = sp.get(key)
            return f"{v:.1f}" if v is not None else "  -"

        def a(key):
            v = rd.get(key, {}).get("auc")
            return f"{v:.3f}" if v is not None else "  -  "

        def b(key):
            if key not in arms:
                return "     -"
            v, e, _ = best(arms[key])
            return f"{v:+.2f}"

        z = r.get("z", {})
        zs = z.get("txc_slab_vs_sae_broadcast")
        win = ("WIN" if r.get("win") else "no") + (f" z={zs:.1f}" if zs else "")
        print(f"{p.stem:<22}"
              f"{s('sae') + '|' + s('txc') + '|' + s('tsae'):>24}"
              f"{a('sae') + '|' + a('txc') + '|' + a('tsae'):>24}"
              f"{b('txc_slab'):>12}{b('sae_broadcast'):>9}{b('txc_flat'):>9}"
              f"{b('txc_profile_random'):>10}{b('dom_slab'):>9}  {win}")

    print("\nBaselines where the task uses probe mode (score = logP(cont1) - logP(cont2)):")
    for p in files:
        r = json.loads(p.read_text())
        if r.get("probe_mode") and r.get("baseline_contrast"):
            bc = r["baseline_contrast"]
            print(f"  {p.stem:<22} unsteered score(A) - score(B) = "
                  f"{bc['mean']:+.2f} +- {bc['sem']:.2f}")

    # r1 bounds what a per-token dictionary can express before any training happens; c is
    # both the constant-write share and the share a pooled probe can read. `grad` variants
    # are taken from the gradient of the metric actually reported rather than from
    # difference of means, and are the ones to quote where both exist.
    rows = [(p.stem, json.loads(p.read_text())) for p in files]
    if any(r.get("rank") for _, r in rows):
        print(f"\n{'run':<22}{'r1(dom)':>9}{'c(dom)':>9}{'r1(grad)':>10}"
              f"{'c(grad)':>9}{'txc_slab':>10}{'rank1_best':>12}{'sae_sched':>11}")
        for stem, r in rows:
            if not r.get("rank"):
                continue
            rk, rg = r["rank"], r.get("rank_grad") or {}
            arms = r["arms"]

            def bb(key):
                return f"{best(arms[key])[0]:+.2f}" if key in arms else "    -"
            print(f"{stem:<22}{rk['r1']:>9.3f}{rk['c']:>9.3f}"
                  f"{(('%.3f' % rg['r1']) if rg else '-'):>10}"
                  f"{(('%.3f' % rg['c']) if rg else '-'):>9}"
                  f"{bb('txc_slab'):>10}"
                  f"{bb('grad_rank1') if 'grad_rank1' in arms else bb('rank1_best'):>12}"
                  f"{bb('sae_schedule_grad') if 'sae_schedule_grad' in arms else bb('sae_schedule'):>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
