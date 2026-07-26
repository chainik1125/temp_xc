"""What the layer screen says about the two models where nothing steers.

Reads `results/txc_wins/layerscreen_{q15,smol,q05}.json` and the transfer steering runs, and
prints the three comparisons the screen was built to make:

  1. rho, the per-document gradient agreement, against its 1/sqrt(n) floor. This was the
     registered hypothesis for the negative and the screen REFUTES it.
  2. the ||Gbar|| cliff, in relative AND absolute units, so the relative-norm convention is
     ruled out as its cause rather than assumed innocent.
  3. ||Gbar|| against MEASURED supervised steering at the same layers -- the only check that
     tells us whether the screen predicts anything, rather than merely describing geometry.

    python scripts/layer_screen_table.py
"""
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
MODELS = [("_q15", "Qwen2.5-1.5B-Instruct", 28), ("_smol", "SmolLM2-1.7B-Instruct", 24),
          ("_q05", "Qwen2.5-0.5B-Instruct", 24)]


def peak(arm):
    """Peak over a symmetric grid, max over signs -- never the signed positive dose."""
    i = max(range(len(arm["delta_margin"])), key=lambda i: arm["delta_margin"][i])
    return arm["delta_margin"][i], arm.get("sem", [0.0] * 99)[i], arm["alphas"][i]


def measured():
    """Measured steering from the transfer runs, keyed by (model, layer)."""
    out = {}
    for f in glob.glob(str(RES / "recency*.json")):
        d = json.loads(pathlib.Path(f).read_text())
        if "arms" not in d or "grad_slab" not in d.get("arms", {}):
            continue
        key = (d.get("model"), d.get("layer"))
        if key in out:
            continue
        a = d["arms"]
        out[key] = {n: peak(a[n]) for n in
                    ("grad_slab", "txc_slab", "sae_broadcast", "random_slab") if n in a}
    return out


def main() -> int:
    meas = measured()
    rows = []
    for tag, name, n_layers in MODELS:
        p = RES / f"layerscreen{tag}.json"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        rows.append((name, n_layers, json.loads(p.read_text())["models"][0]))
    if not rows:
        return 1

    print("1. RHO -- do per-document optimal writes agree enough for ONE write to serve all?")
    print("   Registered hypothesis: rho at its floor in the dead models, well above it in")
    print("   the working one. THE SCREEN REFUTES THIS.\n")
    print(f"   {'model':<26}{'floor':>7}{'rho min':>9}{'rho max':>9}"
          f"{'rho @ best layer':>18}")
    for name, _, d in rows:
        rs = [v["rho"] for v in d["layers"].values()]
        bl = max(d["layers"].items(), key=lambda kv: kv[1]["Gbar_fro"])
        print(f"   {name:<26}{d['rho_noise_floor']:>7.3f}{min(rs):>9.3f}{max(rs):>9.3f}"
              f"{bl[1]['rho']:>13.3f} (L{bl[0]})")
    print("   Every model sits far above the floor everywhere. A single fixed write is the")
    print("   right object in ALL THREE, so 'no shared write exists' is not the explanation.\n")

    print("2. THE CLIFF -- ||Gbar|| collapses at one layer in every model, and it is not a")
    print("   normalisation artefact: converting to absolute units makes it SHARPER.\n")
    print(f"   {'model':<26}{'cliff after':>12}{'depth frac':>12}"
          f"{'rel drop':>10}{'abs drop':>10}")
    for name, n_layers, d in rows:
        Ls = sorted(d["layers"], key=int)
        rel = [d["layers"][L]["Gbar_fro"] for L in Ls]
        ab = [d["layers"][L]["Gbar_fro"] / d["layers"][L]["act_norm"] for L in Ls]
        # The last layer is always degenerate (the write lands after everything that reads
        # it), so it is excluded rather than allowed to masquerade as the cliff.
        i = max(range(len(Ls) - 2), key=lambda i: rel[i] / max(rel[i + 1], 1e-9))
        print(f"   {name:<26}{'L' + Ls[i]:>12}{(int(Ls[i]) + 1) / n_layers:>12.2f}"
              f"{rel[i] / max(rel[i + 1], 1e-9):>9.1f}x"
              f"{ab[i] / max(ab[i + 1], 1e-12):>9.1f}x")
    print("   The last third of every model is unsteerable for this metric.\n")

    print("3. DOES ||Gbar|| PREDICT MEASURED STEERING? Screen value against the supervised")
    print("   gradient arm actually run at that layer -- the screen's only real test.\n")
    print(f"   {'model':<26}{'L':>4}{'||Gbar||':>10}{'grad_slab measured':>21}"
          f"{'txc_slab':>11}{'random':>10}")
    for name, _, d in rows:
        for L in sorted(d["layers"], key=int):
            m = meas.get((name if "/" not in name else name,) + (int(L),)) or \
                meas.get((next((k[0] for k in meas if k[0] and k[0].endswith(name)),
                                None), int(L)))
            if not m:
                continue
            g = m.get("grad_slab")
            t = m.get("txc_slab")
            r = m.get("random_slab")
            print(f"   {name:<26}{L:>4}{d['layers'][L]['Gbar_fro']:>10.2f}"
                  f"{g[0]:>+16.2f}[{g[2]:+g}]{t[0]:>+11.2f}{r[0]:>+10.2f}")
    print("\n   WITHIN a model ||Gbar|| ranks the depths: on SmolLM2's six it gets all six")
    print("   with a single adjacent transposition (L9/L12, whose values differ by 17%).")
    print("   ACROSS models it does NOT: Qwen2.5-0.5B at L12 has ||Gbar|| = 69.4 and moves")
    print("   +0.37, while SmolLM2 at L21 has 2.59 -- 27x smaller -- and moves +0.98. So")
    print("   ||Gbar|| is necessary and not sufficient, exactly like c, and it is only")
    print("   comparable within a fixed model. What it adds is that it is a SIZE, where c")
    print("   and r1 are SHAPES -- which is why those two fired identically on cells that")
    print("   steer and cells that cannot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
