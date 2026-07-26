"""Is the crosscoder's advantage architectural, or does it just hold more lottery tickets?

The sprint's central claim is that the crosscoder finds unsupervised a write a per-token
dictionary COULD express but does not learn. That is a claim about a SEARCH, not about
representation -- and every cell in the sprint gave each architecture ONE dictionary init and
one best-of-4096 latent selection. We already know that lottery is high-variance from our own
data: the crosscoder's margin moved 10x across three inits on the phase ladder, and on the
order task its selected latent does not have a stable SIGN across inits. The SAE was never
given the same number of tickets.

So this reports the DISTRIBUTION of each arm's best latent across dictionary inits on the
held-out instruction-position task, and asks one question:

    does the SAE's BEST-OF-N reach the crosscoder's TYPICAL draw?

If it does, the headline weakens to "the crosscoder finds this write more often", which is a
claim about optimisation rather than expressiveness. If it does not, the discovery claim
survives the most obvious attack available to a reader, on held-out content.

Reported at the matched dose, sign free per arm, consistent with the rest of the sprint.

    python scripts/seed_lottery.py
"""
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATTERN = "recency_tr_ho_ds*.json"


def at(arm, mag):
    best = None
    for a, v, e in zip(arm["alphas"], arm["delta_margin"], arm.get("sem", [0] * 99)):
        if abs(abs(a) - mag) < 1e-9 and (best is None or v > best[0]):
            best = (v, e)
    return best


def main() -> int:
    runs = []
    for f in sorted(glob.glob(str(ROOT / "results" / "txc_wins" / PATTERN))):
        r = json.loads(pathlib.Path(f).read_text())
        a = r.get("arms") or {}
        if "txc_slab" not in a or "sae_broadcast" not in a:
            continue
        mags = sorted({abs(x) for x in a["txc_slab"]["alphas"]})
        md = next((m for m in mags
                   if (g := at(a["txc_slab"], m)) and g[0] > 2 * g[1]), mags[0])
        runs.append((r.get("dict_seed", "?"),
                     at(a["txc_slab"], md)[0], at(a["sae_broadcast"], md)[0], md))
    if not runs:
        print("[skip] no runs yet")
        return 1

    txc = sorted(v for _, v, _, _ in runs)
    sae = sorted(v for _, _, v, _ in runs)
    print(f"held-out instruction position, {len(runs)} dictionary inits, "
          f"matched dose |a| = {runs[0][3]:g}\n")
    print(f"{'init':>6}{'crosscoder':>13}{'SAE':>10}")
    for s, t, e, _ in runs:
        print(f"{str(s):>6}{t:>13.2f}{e:>10.2f}")

    def med(x):
        n = len(x)
        return x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2

    print(f"\n{'':>6}{'crosscoder':>13}{'SAE':>10}")
    print(f"{'min':>6}{txc[0]:>13.2f}{sae[0]:>10.2f}")
    print(f"{'median':>6}{med(txc):>13.2f}{med(sae):>10.2f}")
    print(f"{'max':>6}{txc[-1]:>13.2f}{sae[-1]:>10.2f}")
    print(f"\nSAE best-of-{len(runs)}          {sae[-1]:+.2f}")
    print(f"crosscoder median      {med(txc):+.2f}")
    print(f"crosscoder WORST draw  {txc[0]:+.2f}")
    verdict = ("SAE best-of-N REACHES the crosscoder's worst draw -- the gap is partly a "
               "lottery" if sae[-1] >= txc[0] else
               "SAE best-of-N does NOT reach even the crosscoder's WORST draw")
    print(f"\n=> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
