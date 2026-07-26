---
status: active
created: 2026-07-26 ~11:30 London
for: mac-a (executor) — W2: dialogue-native candidate screen + gated mini-panel
read-first: briefings/day2-dialogue-shared.md
---

# W2 — a second order-carried case-study candidate, on the substrate where order lives

**The bet:** dialogue is the only substrate with measured
order-carriage (R11). A case-study-shaped task there needs a
**trailing STATE whose value requires comparing positions** — not a
level readable from any bag of tokens. Two faces, one screen, then a
mini-panel strictly through the shared doc's gate.

## The two faces (one bundle, one card, corr disclosed)

On the existing DailyDialog substrate (`dialevel/` builders; rebuild
caches in-container; gpt2 + llama31, gemma if a secret appears):

1. **`ttrend` — trailing turn-length TREND** (the Δ/slope face of
   what dialevel screened as LEVEL): kernel-weighted slope of
   turn lengths over the trailing window, terciled
   (falling / flat / rising). A slope needs at least two levels at
   different distances — the regime-3-shaped face, the λ̂-analog on
   dialogue. dialevel's level result (window-readable, R11
   order-sensitive) makes this the single best-motivated candidate
   in the program.
2. **`dqgap` — turns-since-last-question** (the qgap clock ported to
   dialogue, where "?" is DENSE — quote the measured per-turn "?"
   rate in the card; on fineweb it was 0.038/sentence and P7 was
   parked for exactly that sparsity). Distance-to-anchor in its
   purest form, on the one substrate where distance seems to matter.

Build labels from the bundle's turn segmentation (builder committed
before outputs, the family rule). Position-matched manifests, family
eligibility/caps/seeds, zero_split check for dqgap (many windows may
contain a question — disclose the class balance).

## Screen (convention-of-record; freeze card + executor FIRST)

The slen/refmark/quotedens grid, verbatim shape: tok linear+MLP
first; position floor; **visible-evidence floor per T** (for dqgap:
count of "?" tokens in window; for ttrend: label-side OLS from the
window's VISIBLE turn-boundary positions + turn lengths — if the
floor beats every activation arm, the face is boundary-counting, KILL
clause); actxmean ± foreign nulls at T ∈ {4,8,16,32,64}; ORDER arms
(flatten/shuffle/foreign) linear at T ∈ {16,32}, MLP triple at T32;
permutation nulls at T16; **within-dialogue arms BINDING** (the
0.98 identity trap is this substrate's own history). Pre-register
Q1–Q5 incl. the order prediction: **if these faces are real
TXC-candidates, sc > 0 where wc > 0** — an order-free KEEP here goes
to the breadth table, not to a panel (the gate's clause (ii)).
KEEP/KILL bars: the family's (+0.05 vs tok, width ≥ +0.02, beat the
visible floor, wd same-direction).

Est: labels + caches + 2-model screen ≤ $15, ≤ 2 h wall. **Freeze
and push the card, then start caches while mac-local's freeze review
lands in parallel.**

## The gated mini-panel (prep NOW, launch ONLY through the gate)

While screens run, prepare — commit-then-run, but do NOT launch:
- Datasource plugin for the winning face (the `real_lambda` /
  `real_oprate` SyntheticData pattern; reference-basis
  emission_features, doc-identity trace_ids from conversation index).
- Panel CARD draft, the λ̂ Stage-2 pattern REDUCED for the clock:
  one model (the stronger screen model), anchor layer from the
  screen, 5 archs, **T ∈ {2, 4, 8, 16}**, seeds {1, 2, 42},
  trained + untrained, post at k = 8·T from cell one, buffer sized
  to the corpus (state the fill argument), paired v1+v2 probe
  columns per PROBE_V2_SPEC § 2 (claim on v1), realized-l0 band
  stated, evidence-line analog per T, doc-identity floor + demeaned
  receipt, P1–P5 + KEEP/KILL, **tsae cells first, one per container**
  (the scheduling lesson; on Modal high-CPU they ran 62–77 min).
- Cost envelope in the card: 3 tsae containers + main pool on
  A100-40, ≈ 2 h wall, est ≤ $60. Panel cells are the ONLY thing
  A100s are authorized for.

**The gate is the shared doc's § (all five clauses, mac-local's
written LOG approval).** If the gate does not fire: the frozen panel
card + registered datasource IS the deliverable — "panel-ready with
an order-carried screen prior" hands the post-deadline queue its
first day-one launch. Write it up that way, no regret language.

## Falsifier honesty

ttrend may screen order-FREE (slope readable from visible turn
boundaries — the floor will say so) or dqgap may be visible-count —
those are sound kills that sharpen R11's mechanism from the task
side; W1's ladder is the other side. A no-KEEP day with the ladder
verdict is a good day.
