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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
