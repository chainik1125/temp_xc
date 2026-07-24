# Mini-card — factory candidate B4: sentence-length SLOPE (Stage-1 screen)

**Status: FROZEN at commit (commit-then-run; NO cell executed when this
card is committed — git order is the evidence).** Agent: runpod-d.
Briefing: `briefings/task-hunt-r2-d.md` § 3. Bundle:
`../labels/build_verbosity.py` → `../labels/verbosity.npz`. Draft:
`CARD_DRAFT.md`. Driver: `../factory_screen.py`, target `vslope`.

## Why this one is structurally different from every other Ward bundle

Every other candidate screened so far (`ward_lambda`, `sc_lambda`,
`oprate`, `qrate`) is a trailing **RATE** — a level. This one is a
**SLOPE**: the trend in sentence length, a *change* quantity. That
matters for the hunt because a level is exactly the kind of thing a
model can maintain as a running per-position state (and, per the round-1
conversion lesson, evidently does). A slope requires *comparing* early
and late parts of the window, which is the one thing a single position's
marginal is least able to carry.

The bundle's own numbers say the same thing from the label side: the
visible-evidence line for `vslope` is **below chance and falls with T**
(T8 0.483, T16 0.425, T32 0.303) — the in-window raw evidence is
anti-informative about the slope class, so a window probe that scores
well cannot be reading a simple in-window count. `vlevel` (the level
cousin, NOT shipped as primary) behaves the same way (0.420/0.351/0.236),
and the two are near-independent (corr −0.038).

**Consequence for the reading, stated up front:** this is the candidate
where a positive result would be the most interesting of the batch and a
NEGATIVE result is the most likely. I am screening it because it is the
one structurally anti-ambient candidate available, not because I expect
it to win.

## What is consumed, unmodified

`man_vslope_{doc,pos,cls}` + `man_vslope_null_*`, tercile binning,
the bundle's `trace_split`, 20,000 rows/class.

## Frozen protocol

`factory_screen.py` defaults (models {base, distill} × layers {hs13,
hs11} × T ∈ {2,4,8,16,32}; per-token / flatten / MEAN / shuffle;
permutation null seed 99; NULL-label arm; per-token-first triage).

## Frozen predictions

- **P1 (per-token):** **0.55–0.70** — markedly LOWER than the rate
  candidates (`sc_lambda` 0.87), because a slope is not a maintainable
  scalar state in the way a rate is. This is the prediction that
  distinguishes this candidate; if per-token lands at 0.85+ like the
  rates, the "slope is different" premise is wrong and I say so.
- **P2 (order MATTERS — the one place I predict it):** unlike every
  previous candidate, **g_order = flat − mean > 0.02 at T ≥ 16** and
  the within-window shuffle COSTS accuracy (shuffle_gap > 0.02). A slope
  is not order-free; window-mean should be near-blind to it by
  construction.
- **P3 (money pattern):** g clears 3 σ_null by T = 16 and grows to
  T = 32 (a slope needs a long enough baseline to be estimable).
- **P4 (model axis):** base ≈ distill, |Δ| ≤ 0.03.

## Falsifier / kill rule (pre-registered)

KILL if ANY of:
1. g ≤ 3 σ_null at every T.
2. g flat or non-growing across the ladder.
3. NULL-label recovery within 0.02 AUC of the real label's at the best T.
4. per-token AUC ≥ 0.90.

**Not a kill, but scored honestly:** if P2 fails — i.e. the slope turns
out ORDER-FREE too (g_order ≤ 0.02, shuffle-immune) — the candidate may
still pass the kill rules, but then it is *another* regime-2 aggregation
result and the "slope is structurally different" motivation is
FALSIFIED. That must be written as a falsified prediction, not quietly
folded into a KEEP.
