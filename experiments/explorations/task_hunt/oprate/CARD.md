# Mini-card — factory candidate B2: operation-class run-rates (Stage-1 screen)

**Status: FROZEN at commit (commit-then-run; NO cell executed when this
card is committed — git order is the evidence).** Agent: runpod-d.
Briefing: `briefings/task-hunt-r2-d.md` § 3 (quantity mode). Bundle:
runpod-b's `../labels/build_oprate.py` → `../labels/oprate.npz` +
`oprate_stats.json`. Draft: `CARD_DRAFT.md`. Screen driver:
`../factory_screen.py` (the same frozen protocol `sc_lambda` ran).

## Why this one is the most valuable of the remaining Ward bundles

`sc_lambda` (KEEP, qualified) turned out to be a **`ward_lambda` cousin**
— corr 0.473 — so it is not an independent case study. This candidate is
**independent of both**: the bundle reports
`corr(rate_ver, λ̂_sc) = 0.026`, and its own two targets are mutually
independent (`corr(rate_ver, rate_case) = −0.032`). A window result here
would be a genuinely separate datapoint, which is exactly what the hunt
lacks.

## What is consumed, unmodified

`man_ver_{doc,pos,cls}` / `man_case_{doc,pos,cls}` and their
`*_null_*` counterparts, the bundle's binning (`zero_split`), and its
`trace_split`. 20,000 rows per class per target. **Two targets are
screened, both PRIMARY in their own right and reported separately** —
`ver` (verification-check sentence rate) and `case` (case-enumeration
sentence rate); neither is promoted over the other after the fact.
Kernel-smoothed trailing rates, so the current sentence's own class is
not an input.

## Label-side facts known before the screen

Visible-evidence line (label-side AUC of the in-window class count
alone) — **the highest of any bundle so far**, which makes this the
hardest candidate to earn a maintained-state reading:

| target | T8 | T16 | T32 |
|---|---|---|---|
| `ver` | 0.585 | 0.682 | **0.830** |
| `case` | 0.572 | 0.648 | **0.783** |

## Frozen protocol

`factory_screen.py` defaults: models {base, distill} × layers {hs13
primary, hs11 confirmatory} × T ∈ {2,4,8,16,32}; per-token /
window-flatten / window-MEAN / within-window-shuffle (seed 23);
permutation null (seed 99) → σ_null; the whole stack repeated on the
bundle's NULL labels. Per-token-first triage is executed literally (the
per-token arm is flushed before any window arm of that cell).

## Frozen predictions

- **P1 (per-token):** 0.70–0.88 for both targets — high, because the
  round-1 conversion lesson and `sc_lambda`'s 0.87 both say this model
  linearises trailing-rate latents into the current token.
- **P2 (money pattern):** g = flat − per-token clears 3 σ_null by T = 8
  and **grows** through T = 32, for at least one of the two targets.
- **P3 (order-free):** shuffle-IMMUNITY — |shuffle_gap| ≤ 0.02 and
  |g_order| ≤ 0.02 at T ≥ 8 (regime-2 aggregation, the `sc_lambda` and
  `ward_lambda` precedent).
- **P4 (capacity control):** g_agg ≈ g (within 0.02) at T ≥ 8 — the
  equal-dimension MEAN arm reproduces the gain, so it is not probe
  capacity.
- **P5 (model axis):** base ≈ distill, |Δ| ≤ 0.03.
- **P6 (the two targets differ):** `ver` shows a larger g than `case` at
  T = 32 (verification checks cluster in bursts; case enumeration is
  more uniformly spread). Stated so it can be scored, not hedged.

## Falsifier / kill rule (pre-registered, PER TARGET)

KILL a target if ANY of:
1. g ≤ 3 σ_null at every T (no window access).
2. g flat or non-growing across the whole ladder (no T-story).
3. **g_agg < ½·g at T = 16 and T = 32** — i.e. the gain needs the
   flatten arm's extra T·d_in features and does NOT survive at equal
   dimensionality ⇒ probe capacity, not aggregation. (This replaces
   `sc_lambda`'s visible-evidence kill rule, which I froze badly there:
   per-token alone already exceeded that line, so no arm could fail it.
   The equal-dimension test is the discriminating one and it is stated
   here BEFORE seeing any cell.)
4. NULL-label recovery comes within 0.02 AUC of the real label's at the
   best T (the "recovery" is trace-ambient rate, not local history).
5. per-token AUC ≥ 0.90 (saturated; no headroom to win).

A KEEP must state the per-token level prominently (the conversion
caveat) and report both targets. Verdict → one paragraph in `../LOG.md`.
