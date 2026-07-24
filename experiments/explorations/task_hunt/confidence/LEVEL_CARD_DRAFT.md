# DRAFT mini-card — hedging LEVEL Stage-2 (aggregation-framed regime-2)

**Status: DRAFT (runpod-b, item 4 of `briefings/hunt-support-stats.md`).
NOT operative. runpod-e sharpens and freezes its OWN card before any
run — the running agent's card governs (protocol per round-1 review
note 4). This draft exists so freezing is an edit, not a from-scratch
write.**

Provenance: round-2 decision (b) — the program ACCEPTS an
aggregation-framed regime-2 win; the killed `confidence/CARD.md` is the
TREND card and is NOT this candidate's confirmation (its recorded-seed
paragraph in `../LOG.md` is the provenance trail). Fresh prereg,
fresh predictions.

## The claim being tested (say it exactly)

A window architecture recovers the **hedge LEVEL carried by the
trailing window** better than per-token decoders at matched realized
l0, with the advantage **growing in T** — and the mechanism claimed is
**order-free aggregation** (window-mean pooling), NOT order. The
mechanism receipt is therefore **shuffle IMMUNITY**: within-window
shuffle must NOT destroy the window recovery (disclosed up front — the
opposite receipt to round 1's order requirement, and the honest one
for this latent: the screen showed slope ≈ anchor − window-mean under
anchor matching, an order-free functional).

## Label definition (draft — runpod-e freezes the exact variant)

Source artifacts (committed, label-side only, alignment contract in
`../labels/README.md`): `../labels/confidence.npz` — per-token 3-class
hedge state `hedge` (lexically stamped; state counts 0/1/2 =
94,140 / 319,181 / 53,763; `valid` masks round-trip-verified positions,
valid rate 0.909) on the canonical 4044 × 128 Ward grid; class-balanced
manifests 20,000 rows/class, `pos ≥ 32`, split by trace.

- **PRIMARY (draft): continuous window-mean hedge level** — the mean of
  the per-token hedge state over the trailing T-token window at the
  anchor tile (the aggregation latent itself; regression head, Pearson
  r on held-out traces, mirroring `lambda_recovery`).
- **Secondary (draft): anchor-sentence hedge level** (the 3-class state
  at the anchor sentence) — the discrete face; interpret against its
  known regime-1 behavior (the round-1 screen's state CONTROL was
  per-token ≈ window on both readers), i.e. this face alone is NOT the
  win condition.
- **DECISION POINT for runpod-e:** primary = window-mean level vs
  anchor level read through pooling. The draft prior says window-mean
  (it is the quantity the screen showed to be window-carried); if the
  freeze picks anchor level, state why the regime-1 control result
  does not apply.

## Ambient/conversion guards (carried over, they bind here too)

- Exact-histogram matching on anchor hedge state × position bucket for
  any probe row set (the round-1 ambient-route guard) — without it,
  window-mean level is partially readable from the anchor's own
  lexical stamp.
- Per-token-first triage (hunt convention, adopted at review): run the
  per-token probe on the primary target alone before the panel; a HIGH
  per-token ceiling is presumptively converted — escalate only with a
  card-stated reason.
- Position floor: report the position-only readout of window-mean
  level next to every cell (hedge level drifts with doc position —
  state mean by doc third 0.819 / 0.955 / 0.976 in
  `../labels/confidence_stats.json` — the ramp is ambient and must not
  be sold as window signal).

## Stage-2 design (mirror candidate 1's panel exactly)

Plugin datasource (no core edits), single scarce anchor d_sae = 2048,
nominal k_pos = 8, 5 archs (per-token BatchTopK, T-SAE, Stacked,
TXC-pre, TXC-post) × T ladder × seeds {1, 2, 42} + untrained controls,
through the canonical runner; headline metric = held-out Pearson r of a
per-tile linear probe. **Matched REALIZED l0 is the fairness check** —
after runpod-d's round-2 amendment, state per-T nominal k so realized
l0 stays ≈ 7–8 for every arch (round 1's TXC-post collapse is the
cautionary case; the variance-aware renderer flags it automatically).

Reader/layer (draft): distill hs15 (resid_post L14) — the screen's
best cell. T ladder (draft): {2, 4, 8, 16, **32**} — the clock bridge
(median 16 tokens/sentence, mean 19.2, p10 6, p90 37;
`../labels/proofops_stats.json`) puts sentence-scale pooling at
T ≥ 16, and the screen's window-mean gap grew through T = 64
(0.521 → 0.545 → 0.565 at T 16/32/64 vs per-token 0.468); T = 32 is
the largest T the eval window (L = 32) admits. DECISION POINT: drop
T = 32 only if the cache/runner cost says so, and record that the
tested range then sits at the BOTTOM of the effect's T-range.

**Readout convention sentence (carry into the record verbatim):** the
evaluation reads ONE tile's code per prediction — under the
code-readout convention, per-token archs are read at single positions
by construction, and pooling T-SAE codes across T positions would
spend T× the code bandwidth a window arch uses.

## Frozen predictions (draft — runpod-e commits its own numbers)

- P1: TXC-pre recovery of the primary target rises with T at matched
  realized l0 and exceeds both per-token references at T ≥ 16.
- P2: the trained−untrained margin grows with T (the architecture
  learns a T-dependent code, not an init artifact).
- P3 (mechanism receipt, the disclosed one): within-window shuffle
  leaves the window recovery intact (immunity) — order-free pooling is
  the claim. A LARGE shuffle degradation would actually FALSIFY the
  aggregation framing.
- P4: the per-token probe on the primary target stays near the
  regime-2 floor after matching (per-token-first triage passes).

## Kill rule (draft)

KILL if ANY of: (1) per-token-first triage shows a high per-token
ceiling on the primary target after anchor matching (converted ⇒ run
the depth sweep as the WHY-diagnostic, then stop); (2) no T-rise at
matched realized l0 (flat panel); (3) the window advantage does not
clear the position floor; (4) the trained−untrained margin does not
grow with T. Shuffle immunity alone is NOT a kill — it is the
mechanism receipt this card claims.

## Label-side numbers already in hand (nothing to rebuild)

`confidence.npz` (hedge state + slope4/slope8 + valid + manifests) and
the clock bridge are committed and reviewed; the LEVEL target needs
only a window-mean reduction over the existing `hedge` field —
CPU-trivial, no new labeling pass. slope8 tercile edges
(−0.0238, +0.0357) remain available if a trend face is ever wanted for
context (NOT as a primary — that candidate is killed).
