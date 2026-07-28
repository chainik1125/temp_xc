"""Receipt for GENERATION_CARD.md s2 — `floor_excess` == in-window fraction.

The card's density target rests on one identity: for a balanced 3-class
AGE face whose `visible_evidence_floor` is fit on
`(censored_age, in_window_event_count)`, the floor's excess over chance
equals `f = P(event inside the T window)`.

    floor_acc = f*1 + (1-f)*[(1/3)/(1-f)] = f + 1/3
    => floor_excess = f

The floor knows the EXACT age whenever the event is in-window, so it
classifies those rows correctly whatever class they land in. The
identity therefore survives until T passes the UPPER tercile edge e2
(f = 2/3), NOT the lower one.

⚠ WHY THIS SCRIPT EXISTS. The first draft of the card claimed the
identity held only while T <= e1, and that a pure age face was
therefore "capped near floor_excess = 1/3". Both are false, and this
simulation is what caught it before the card was frozen. The false
version would have justified a much denser corpus on the grounds that
the floor could not run away -- when in fact the floor is computed from
GROUND TRUTH and climbs toward 1.0, which is precisely why the band's
upper edge is real (3 of 5 record cells above +0.25 lose to their own
floor).

⚠⚠ SECOND ERROR, CAUGHT THE SAME DAY -- see `--real`. The simulation
below is an UNNECESSARY INTERMEDIATE. `elicit_lib.claim_zone()` already
measures `f` directly on a built stream, and against evalage's MEASURED
floor_excess it gives K = 0.96 -- i.e. the identity holds on real data
with no correction at all. My earlier K = 0.63 came from comparing the
measured value against an f simulated from EXPONENTIAL gaps, when
evalage's gaps are LOG-UNIFORM. The 1.6x discrepancy was my gap model
being wrong, and I gave it a spurious mechanism ("eligibility removes
positions nearest an event").

That error was material: it put the planning centre at g = 170 tok,
which the real age CDF says lands f ~ 0.36 -- deep into the band where
3 of 5 record cells lose to their own floor. Corrected centre is
g ~ 385. Kept here rather than deleted, because the exponential route
is still the only thing available BEFORE any text exists, and a reader
needs to know how far it can be off.

Usage:
    python -m experiments.explorations.task_hunt.retryesc_gen.verify_floor_identity
    ... --solve      solve for the mean gap g at each target f (SIMULATED)
    ... --real       validate the identity against evalage's real corpus
                     and re-solve g from its measured age CDF  <- USE THIS
"""
from __future__ import annotations

import sys

import numpy as np

T = 64
N = 400_000
GAPS = (64, 100, 150, 200, 256, 300, 427, 600, 862, 886, 2000)
# evalage: simulated f at its measured gap g=862 vs the floor_excess its
# screen actually reported. Probe eligibility is restricted to assistant
# tokens while events sit in environment turns, so positions nearest an
# event are disproportionately INELIGIBLE and drop out of f.
EVALAGE_G, EVALAGE_MEASURED = 862, 0.045


def floor_excess(g: float, rng: np.random.Generator) -> tuple[float, float]:
    """Return (f, floor_excess) for exponential gaps of mean `g`."""
    age = rng.exponential(g, N)
    e1, e2 = np.quantile(age, [1 / 3, 2 / 3])
    y = np.digitize(age, [e1, e2])
    inw = age < T
    # in-window rows -> exact age -> exact class; censored -> best single guess
    acc = inw.sum() * 1.0
    rem = y[~inw]
    if rem.size:
        acc += np.bincount(rem, minlength=3).max()
    return float(inw.mean()), acc / N - 1 / 3


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"T = {T}, N = {N:,}, exponential gaps\n")
    print(f"{'mean gap g':>11}{'f = P(age<T)':>14}{'floor_excess':>14}"
          f"{'|err|':>9}")
    worst = 0.0
    for g in GAPS:
        f, fe = floor_excess(g, rng)
        worst = max(worst, abs(fe - f))
        print(f"{g:>11}{f:>14.4f}{fe:>14.4f}{abs(fe - f):>9.4f}")
    print(f"\nworst |floor_excess - f| = {worst:.6f}  => identity holds, "
          f"and note it holds past f = 1/3 at the dense end")

    if "--solve" not in sys.argv:
        return

    from scipy.optimize import brentq

    rng2 = np.random.default_rng(1)
    f_at = lambda g: floor_excess(g, rng2)[0]          # noqa: E731
    k = EVALAGE_MEASURED / floor_excess(EVALAGE_G, rng2)[0]
    print(f"\neligibility correction K = {k:.3f}  "
          f"(evalage measured {EVALAGE_MEASURED:+.3f} at g={EVALAGE_G})")
    print("\nmean gap g needed for each target floor_excess:")
    print(f"{'target f':>10}{'naive g':>11}{'calibrated g':>15}")
    for target in (0.25, 0.20, 0.15):
        gn = brentq(lambda g: f_at(g) - target, 20, 3000, xtol=0.5)
        gc = brentq(lambda g: f_at(g) * k - target, 20, 3000, xtol=0.5)
        print(f"{target:>10.2f}{gn:>11.0f}{gc:>15.0f}")
    print("\nQuoted as a BRACKET in the card, not averaged: the "
          "calibration rests on a single point (evalage).")


HUNT = __import__("pathlib").Path(__file__).resolve().parent.parent
EVALAGE_GAP_MEDIAN = 862.0          # evalage realised gap median (receipt)
BEST_T = {"gpt2": 64, "gemma2_2b": 64, "llama31_8b": 32}
LEG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}


def real() -> None:
    """Validate `floor_excess == f` on evalage, then re-solve g."""
    import json
    import math

    pm = json.loads(
        (HUNT / "labels/evalage_premeasure_3leg.json").read_text())
    print("IDENTITY ON REAL DATA (evalage) — claim_zone f vs MEASURED "
          "floor_excess\n")
    print(f"{'leg':<12}{'claim_zone f':>14}{'measured':>12}{'diff':>9}")
    pred, meas = [], []
    for model, key in LEG.items():
        f = pm["legs"][key]["claim_zone"]["frac_in_window"]["T64"]
        cells = json.loads(
            (HUNT / f"evalage/results/screen_evalage_{model}.json").read_text()
        )["cells"]
        fl = cells[f"evalage_age/T{BEST_T[model]}/visible_evidence_floor"][
            "acc_test"] - 1 / 3
        pred.append(f)
        meas.append(fl)
        print(f"{model:<12}{f:>14.4f}{fl:>+12.4f}{fl - f:>+9.4f}")
    mp, mm = sum(pred) / len(pred), sum(meas) / len(meas)
    print(f"{'MEAN':<12}{mp:>14.4f}{mm:>+12.4f}{mm - mp:>+9.4f}")
    print(f"\nK = {mm / mp:.3f}  <- NOT the 0.63 the frozen card first "
          f"claimed; per-leg scatter is probe noise")

    cz = pm["legs"]["gpt2"]["claim_zone"]
    fw, med = cz["frac_in_window"], cz["median"]
    lo_a, lo_c, hi_a, hi_c = 64.0, fw["T64"], med, 0.5

    def age_at(cdf):
        fr = (cdf - lo_c) / (hi_c - lo_c)
        return math.exp(math.log(lo_a) + fr * (math.log(hi_a) - math.log(lo_a)))

    def f_at(g):
        x = 64.0 * (EVALAGE_GAP_MEDIAN / g)
        fr = (math.log(x) - math.log(lo_a)) / (math.log(hi_a) - math.log(lo_a))
        return lo_c + fr * (hi_c - lo_c)

    print(f"\nevalage real age CDF anchors: P(<=16)={fw['T16']:.4f} "
          f"P(<=32)={fw['T32']:.4f} P(<=64)={fw['T64']:.4f} median={med:.0f}")
    print(f"\n{'target f':>9}{'age at CDF':>12}{'scale':>8}{'gap median g':>15}")
    for t in (0.25, 0.20, 0.15):
        x = age_at(t)
        print(f"{t:>9.2f}{x:>12.0f}{x / 64:>8.2f}"
              f"{EVALAGE_GAP_MEDIAN / (x / 64):>15.0f}")
    print("\nwhat the frozen card's original routes would have given:")
    for g, label in ((170, '"calibrated" centre'), (286, '"naive" centre'),
                     (385, "CORRECTED centre")):
        f = f_at(g)
        flag = "  <-- past the +0.25 edge" if f > 0.25 else ""
        print(f"  g={g:>4} tok  ->  f ~ {f:.3f}   {label}{flag}")
    print("\nLog-interpolation between the 64 and median anchors; crude, "
          "and it assumes evalage's age SHAPE carries over. It is anchored "
          "on real data, which the exponential route was not.")


if __name__ == "__main__":
    if "--real" in sys.argv:
        real()
    else:
        main()
