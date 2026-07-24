# Mini-card — factory candidate B3: question-rate intensity (Stage-1 screen)

**Status: FROZEN at commit (commit-then-run; NO cell executed when this
card is committed — git order is the evidence).** Agent: runpod-d.
Briefing: `briefings/task-hunt-r2-d.md` § 3. Bundle:
`../labels/build_qrate.py` → `../labels/qrate.npz`. Draft:
`CARD_DRAFT.md`. Driver: `../factory_screen.py`, target `""` is NOT used
— this bundle ships the unprefixed `man_*` layout like `sc_lambda`.

## What it is, and its honest standing in the batch

Events = "?"-terminated sentences; the same exponential-kernel trailing
rate machinery as `ward_lambda` / `sc_lambda`. It is therefore the
**third member of the same family**, and its main value is as a
REPLICATION of the `sc_lambda` pattern on a different, cleaner event
stream (question marks are exact and unambiguous, where the
self-correction marker list is a frozen judgement call). It is not a new
phenomenon and will not be presented as one.

Bundle triage (label-side, shipped): current-token-identity AUC
**0.610**, position AUC **0.586** — both PASS, and both lower than
`sc_lambda`'s (0.636 / 0.625), so the label is slightly less
token-readable than the self-correction one.

Visible-evidence line (label-side, in-window "?" count alone):
T8 **0.560**, T16 **0.623**, T32 **0.742**.

## What is consumed, unmodified

`man_{doc,pos,cls}` + `man_null_*`, the bundle's binning and
`trace_split`. Marker (`?`) tokens masked from manifest rows by the
builder, same discipline as `sc_lambda`.

## Frozen protocol

`factory_screen.py` defaults (models {base, distill} × layers {hs13,
hs11} × T ∈ {2,4,8,16,32}; per-token / flatten / MEAN / shuffle;
permutation null seed 99; NULL-label arm; per-token-first triage).

## Frozen predictions

- **P1 (replication):** the `sc_lambda` shape reproduces — per-token
  **0.80–0.90**, g negative at T = 2, crossing 3 σ_null by T = 8, and
  growing monotonically to **+0.04…+0.09** at T = 32.
- **P2 (order-free):** shuffle-IMMUNITY, |g_order| ≤ 0.02 at T ≥ 8.
- **P3 (capacity control):** g_agg ≈ g within 0.02 at T ≥ 8.
- **P4 (model axis):** base ≈ distill, |Δ| ≤ 0.03.

## Falsifier / kill rule (pre-registered)

KILL if ANY of:
1. g ≤ 3 σ_null at every T.
2. g flat or non-growing across the ladder.
3. **g_agg < ½·g at T = 16 and T = 32** (the gain needs the flatten
   arm's extra features ⇒ probe capacity, not aggregation).
4. NULL-label recovery within 0.02 AUC of the real label's at the best T.
5. per-token AUC ≥ 0.90.

Because this is an explicit replication, a KEEP here adds confidence to
the `sc_lambda` reading but **does not** count as an independent
candidate; that must be stated in the verdict.
