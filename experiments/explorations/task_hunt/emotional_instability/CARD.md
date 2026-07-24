# Mini-card (FROZEN pre-run) — emotional-instability onset

**Candidate 3, task-hunt arm B** (`briefings/task-hunt-b.md`; paper
summary: `docs/papers/gemma_needs_help.md`). Drafted by `runpod-b`
(prep deliverable 5); **frozen by `runpod-e` at this commit** after
candidates 1 (replag) and 2 (confidence) closed KILL (`../LOG.md`).
Each pipeline stage below is committed before it runs; screen-stage
constants are frozen here.

## Frozen run plan (runpod-e)

- **Elicitation:** ~300 conversations = 30 verified-impossible numeric
  puzzles (paper's two verbatim App-B prompts + programmatic
  Countdown/fraction variants, brute-force-verified unsolvable under
  the stated constraints, prompt template verbatim) × 10 rollouts,
  **8 turns, temperature 1.0**, 7 neutral rejections (paper's Extended
  wording), max 512 new tokens/turn. Substrate **gemma-3-12b-it**;
  escalate to 27b-it ONLY if < 30 % of 12b conversations reach a
  judged score ≥ 5 anywhere (the paper's elicitation-rate bar,
  recorded before escalating).
- **Labels:** per-response frustration 0–10 (App B.2 judge prompt
  VERBATIM) + within-turn onset token (App C.1 labeler VERBATIM,
  mapped to a token position by exact string match of emotional_word
  with preceding_context). **κ prereg gate:** 30 conversations
  dual-judged (sonnet-class primary + haiku-class second), quadratic-
  weighted κ ≥ 0.3 to proceed (em_onset convention; ε̂ and within-1
  rate recorded). Judge budget ≤ $40 total.
- **Grid:** conversations tokenized with the model's chat template;
  probe rows only at ASSISTANT-response tokens; screen layer
  resid_post **L24 = hs25** (mid-depth of 48; alternates hs13, hs37
  cached, not screened); frozen probe stack = `conversion_depth/
  problib.py`; T ∈ {16, 32, 64}; windows may span turn boundaries
  (the escalation loop IS cross-turn context); tokens-per-turn clock
  stats recorded BEFORE the screen; the honest "trend timescale
  unreachable at panel-feasible T" kill applies.
- **Readout (a) — pre-onset anticipation (D-ladder, PRIMARY):**
  positives at offset ∈ {[1,4], [5,8], [9,16]} before the
  conversation's FIRST onset token; negatives ≥ 64 tokens from any
  onset (guard band (16, 64) excluded), drawn from pre-onset regions
  and no-onset conversations only; anchor-token-identity × position
  matching per the replag convention; split by puzzle (all rollouts of
  a puzzle in one fold).
- **Readout (b) — escalation intensity (SECONDARY):** tercile of the
  CURRENT response's judged score, probed at pre-onset positions
  within the response, classes exact-matched on (turn index ×
  position-in-turn bucket) — turn number cannot carry the label.
- **Sanity anchor (validates labels, NOT a target):** post-onset
  detection (positions after onset vs matched negatives) must be
  per-token-readable (linear AUC ≥ 0.75 expected — lexically stamped);
  a card claiming detection as the finding dies at the gate.

## Frozen predictions + KEEP/KILL

- P1: anticipation gap (window − per-token, same pair across D bands)
  turns on with T and is largest for the nearest band; per-token flat
  in T.
- P2: escalation-intensity shows a window gap that grows T16 → T64
  (the loop's evidence accumulates over turns).
- P3: the anchor is regime-1 (per-token ≈ window, both high).
- **KEEP** iff (a) or (b) shows window − per-token ≥ +0.05 (AUC or
  acc) at some screened T with T-growth AND the context-shuffle
  removes ≥ half of that gap (the order/structure receipt the hunt
  deliverable requires) AND the anchor validates (P3).
- **KILL** otherwise — including the timescale kill and the
  "aggregation-carried, shuffle-immune" outcome (recorded as a
  regime-2 seed, as with candidate 2).

## Substrate + elicitation (fully specified in the paper)

**gemma-3-12b-it** (27b-it only if 12b's elicitation rate is too low) —
the IT model is CORRECT here: the phenomenon is created by Gemma's
post-training (the paper's base-vs-instruct prefill result), the same
fit-where-the-phenomenon-lives rule as the EM organism. Elicitation:
impossible-numeric puzzles + neutral rejections, 8 turns, temperature 1;
the paper reports per-turn mean frustration rising ≈ 1.5 → 5.5 across
turns — a **graded escalation trend**, not a rollout boolean.

## Label spec (judge + onset labeler, prereg'd)

- **Per-response frustration 0–10** — the paper's judge prompt VERBATIM
  (in our doc, Appendix B protocol; their cross-judge reliability:
  Pearson r = 0.792, 78 % within one point).
- **Within-turn onset token** — their validated onset-labeler prompt
  (App C.1): the token where emotional language first appears.
- Budget ≤ $40 judge spend; **prereg + κ on 30 traces before scaling**
  (the em_onset convention): freeze judge/labeler prompts, score 30
  dual-judged traces, proceed only if κ clears the program's adequacy
  floor (κ ≥ 0.3), recording ε̂.

## Two readouts, in order

1. **Pre-onset anticipation** at frozen offsets before the labeled
   onset token (the em_onset D+ design — same ladder mechanics as the
   forbidden-word draft: within-D positives, far negatives, guard band,
   identity/position matching, doc-level splits).
2. **Escalation-intensity regression** — turn-indexed frustration trend
   (turn number × judged score), the graded target.

## The trap, named (regime-1 anchor, not target)

Post-onset *detection* is lexically stamped — "frustrated", "myself",
"[deep] breath" dominate the paper's differential-word tables ⇒
predicted regime-1 / per-token-readable. Detection is the SANITY ANCHOR
(it must work, per-token ≈ window); **a card claiming detection dies at
the gate**. The candidate is the anticipation/escalation state before
the lexical stamp appears.

## Predicted T-pattern (STORY.md § 7) + timescale risk

Anticipation: threshold ladder in the offset D, as forbidden-word.
Escalation trend: lives at TURN scale — screen T ∈ {16, 32, 64} and
treat an honest "window cannot reach the trend's timescale at
panel-feasible T" as a valid kill (same clock discipline as arm-B
candidate 2; measure tokens-per-turn on the actual rollouts first and
choose T from it).

## Falsifier / KEEP-KILL (draft)

- **KEEP** iff pre-onset anticipation shows the ordered-onset ladder
  with g_order > 0 while the detection anchor stays regime-1.
- **KILL — ambient** if per-token ≈ window at every offset (the
  emotional buildup is itself lexically stamped earlier than the
  labeled onset).
- **KILL — timescale unreachable** for the escalation readout if no
  screened T spans a meaningful fraction of a turn (clock measurement,
  not regime claim).
- **KILL — elicitation failure** if 12b's high-frustration rate is too
  low for label mass at the prereg'd budget (record the rate; only then
  consider 27b-it).
