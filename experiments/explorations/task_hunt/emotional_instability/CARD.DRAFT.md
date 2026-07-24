# DRAFT mini-card — emotional-instability onset (NOT frozen)

**Candidate 3, task-hunt arm B** (`briefings/task-hunt-b.md`; paper
summary: `docs/papers/gemma_needs_help.md`). Drafted by `runpod-b`
(prep deliverable 5). **The running agent (`runpod-e`) freezes its own
card** — edit freely, rename to `CARD.md` in the freezing commit.
Order vs arm-B candidate 2: per the briefing, decide on feasibility
after candidate 1's verdict — note the clock-bridge numbers now in
`../labels/proofops_stats.json` bear directly on candidate 2's
reachability (slope8 ≈ 128 tokens of support).

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
