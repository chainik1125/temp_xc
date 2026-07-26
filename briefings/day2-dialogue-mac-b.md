---
status: active
created: 2026-07-26 ~11:30 London
for: mac-b (executor) — W1: the order-mechanism ladder on dialevel
read-first: briefings/day2-dialogue-shared.md
---

# W1 — decompose dialevel's order signal (the R11 mechanism ladder)

**The question, exactly:** dialevel's within-dialogue readout loses
+0.057/+0.063/+0.035 AUC (gpt2/gemma/llama, T = 32, R11) when the
anchor-fixed context is shuffled. slen killed "generic recency" as
the mechanism (R20). What in the dialogue context carries it —
**turn ORDER, within-turn token order, or distance-to-anchor
weighting that only expresses on dialogue?** Convert R11 from
counterexample into mechanism. This is the paper-facing deliverable
even if W2 KEEPs nothing: it tells the team exactly where TXC-style
position-mixing has something real to encode.

## Substrate and assets (all exist; nothing new to design)

`dialevel/` — DailyDialog, the frozen bundle + committed builders
(`cache_acts.py`), the screen with its wd arms (`screen.py`), the
anchor-fixed shuffle + foreign-null machinery (`actxmean_null.py`,
`capacity_check.py`). The R11 cells are `wd/T32: win_linear −
win_shuf_linear` in `results/screen_{model}.json`. Rebuild caches
in-container from the committed builders (identity re-asserted, the
family convention). Models: gpt2 + llama31 (gemma if a secret
appears — it is the LARGEST R11 cost, +0.063, so say so in coverage).

## The ladder (freeze your card FIRST — arms, predictions, verdict rule)

All arms at matched width, anchor slot fixed, same rows as the R11
cells (wd readout, T ∈ {16, 32}; T32 is the R11 anchor, T16 the
robustness point), seeded, permutation nulls beside:

- **L0 full context shuffle** — MUST reproduce the R11 cost on the
  rebuilt caches (positive control; if it does not reproduce within
  its null band, STOP and report — nothing downstream is
  interpretable).
- **L1 within-turn shuffle, turn order preserved** (shuffle tokens
  inside each turn's span; turn sequence intact).
- **L2 turn-block permutation, within-turn order preserved** (permute
  whole turns; token order inside each turn intact). Use the bundle's
  turn segmentation; disclose mean turns-per-window per T.
- **L3 near/far half swap** (runpod-e's recency probe: shuffle only
  the far half vs only the near half of the context) — the residual
  recency check on the substrate where order DOES matter.
- **L4 foreign context** (width null, unchanged from the screen).

**Pre-register in the card:** cost(L1) + cost(L2) ≈ cost(L0) is the
clean-decomposition expectation; verdict = whichever of L1/L2 carries
**≥ half of L0's cost on 2/2 models** is THE mechanism
(TURN-STRUCTURE if L2, WITHIN-TURN if L1, MIXED if both ≥ a third,
RECENCY-RESIDUAL if L3-near ≫ L3-far while L1/L2 split, UNRESOLVED
otherwise — state all five outcomes before running). Quote every cost
beside its permutation-null band. Within-dialogue control is already
the readout here (wd arms) — the identity route is controlled by
construction; say so.

## Economics and order of work

Caches ≈ dialevel's existing sizes (small; DailyDialog): L40S,
build-in-container. Ladder = probe fits on cached activations —
**≤ 45 min/model, est ≤ $8 total** of your $60 cap. Freeze card →
push → launch (detached) → verdict LOG entry + receipts row(s) for
the mechanism claim → STATUS. If time remains after the verdict, ask
mac-local before starting anything new (the shared doc's gates
apply).

## Falsifier honesty

If L0 fails to reproduce R11 on rebuilt caches, that IS the result
(report as REPRODUCTION FAILURE with both numbers; do not proceed).
If no single arm carries half, UNRESOLVED is a publishable outcome —
do not narrative-upgrade the largest fraction into "the mechanism".
