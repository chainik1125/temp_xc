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

Usage:
    python -m experiments.explorations.task_hunt.retryesc_gen.verify_floor_identity
    ... --solve      also solve for the mean gap g at each target f
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


if __name__ == "__main__":
    main()
