# Mini-card — factory candidate B1: self-correction marker intensity λ̂_sc (Stage-1 screen)

**Status: FROZEN at commit (commit-then-run; the screen has NOT been
executed when this card is committed — git order is the evidence).**
Agent: runpod-d. Briefing: `briefings/task-hunt-r2-d.md` § 3 (quantity
mode). Bundle: runpod-b's `../labels/build_sc_lambda.py` →
`../labels/sc_lambda.npz` + `sc_lambda_stats.json`; draft card
`CARD_DRAFT.md` (this card sharpens it and governs the run).

## What is consumed, unmodified

The bundle ships screen-ready manifests and I use them **as shipped** —
no re-binning, no re-splitting, no re-masking. Verified before freezing
(label-side only, no activations touched):

- `man_doc/man_pos/man_cls`: 3 classes × 20,000 rows; `man_pos ≥ 32`
  (so every T in the ladder fits inside the stream); `is_marker_tok`
  sums to **0** over manifest rows (the masking rule holds — the label
  is never read off the marker token itself); `valid` all True.
- `trace_split`: 240 train / 60 test traces → 48,025 train / 11,975 test
  manifest rows, near class-balanced in both.
- Null arm `man_null_*` + `lam_null_bin`: the within-trace event shuffle
  (seed 101) that preserves each trace's marker RATE but destroys local
  clustering.

**PRIMARY target: top vs bottom bin** (`man_cls ∈ {0, 2}`, binary), the
`lam` zero-split scheme the bundle froze. Secondary: the same stack on
the NULL labels (`man_null_*`), reported beside every primary number.

## Why this candidate is worth GPU time (the escalation reason)

Required by the new per-token-first convention. λ̂_sc is the **same
kernel machinery as `ward_lambda`** — the program's one Stage-2
QUALIFIED POSITIVE — with the Sonnet-judged event stream swapped for a
frozen, zero-API lexical one. `ward_lambda` had a HIGH per-token ceiling
(0.776–0.795) and was still a genuine window result (gap growing to
+0.054 at T=32), so on this family a high per-token alone is **not**
disqualifying: the question is the window gap and its T-growth. Disclosed
family resemblance: `corr(λ̂_sc, ward λ̂_hist) = 0.473` — this is a
COUSIN of candidate 1, not an independent candidate, and any KEEP must
say so.

## Label-side facts, known before the screen (from the bundle's stats)

- Triage PASS but **close on the token axis**: current-token-identity
  AUC **0.636** (kill threshold 0.65), position AUC **0.625** (kill 0.70).
  The token-identity number is the ambience floor any activation
  per-token probe must be read against.
- **Visible-evidence line** — label-side AUC of the in-window marker
  COUNT alone: **T8 0.525, T16 0.578, T32 0.701**. This is the sharp
  bar: a window probe that merely counts the marker tokens the window
  already displays is not a maintained state.
- Event rate 0.136 of sentences; marker-token rate 0.010.

## Frozen screen protocol

Models {base, distill} × layer hs13 (`resid_post` L12, primary) and hs11
(L10, confirmatory) × **T ∈ {2, 4, 8, 16, 32}**, right-edge windows
ending at the manifest position. Arms per cell, on the frozen `problib`
stack (`conversion_depth.problib.fit_probe`, class-weighted, 2 classes):
per-token, window-flatten, window-MEAN (→ g_agg / g_order), and
within-window-SHUFFLED (seed 23). Permutation null (seed 99) on the
per-token and flatten arms → σ_null. Identical rows for every model,
layer and T, so every difference is attributable to the representation.

**Per-token-first triage (binding convention, executed as ordered):**
the per-token arm runs FIRST and is written to disk before any window
arm starts, so the ordering is auditable in the results file.

## Frozen predictions

- **P1 (per-token level):** per-token AUC lands **0.65–0.82** — above the
  0.636 token-identity floor (activations carry more than the current
  token's identity) but short of saturation.
- **P2 (the money pattern):** g(T) = flat − per-token is positive beyond
  3 σ_null for some T ≥ 8, and **grows** across the ladder (the kernel's
  τ = 3-sentence mass sits ~48 tokens back, so each doubling of T adds
  visible kernel mass through T = 32).
- **P3 (order-free):** shuffle-IMMUNITY — |flat − shuf| ≤ 0.02 and
  g_order = flat − mean ≤ 0.02 at every T (the `ward_lambda` precedent:
  g_order ≤ 0 in 17 of 20 primary cells). The mechanism is aggregation,
  not order.
- **P4 (model axis):** base ≈ distill (|Δ| ≤ 0.03), as for `ward_lambda`.
- **P5 (the hard one — visible evidence):** the flatten arm **exceeds**
  the label-side visible-evidence line at matched T (> 0.525 at T8,
  > 0.578 at T16, > 0.701 at T32). I flag this as the prediction most
  likely to fail at T = 32, where the bar is highest.

## Falsifier / kill rule (pre-registered, on the PRIMARY target)

KILL if ANY of:
1. **No window access:** g(T) ≤ 3 σ_null at every T.
2. **No T-story:** g(T) flat or non-growing across the whole ladder.
3. **Not a maintained state:** the flatten arm fails to beat the
   visible-evidence line at T = 16 (0.578) — i.e. the window probe is
   only counting the markers the window already shows.
4. **Ambient rate, not history:** recovery on the NULL labels
   (within-trace event shuffle) comes within 0.02 AUC of the real
   labels at the best T — the "recovery" is trace-ambient marker rate.
5. **Saturated per-token:** per-token AUC ≥ 0.90 (no headroom; fully
   linearized).

A KEEP additionally requires the disclosure that this is a `ward_lambda`
cousin (r = 0.473), not an independent second win. Verdict → one
paragraph in `../LOG.md`.
