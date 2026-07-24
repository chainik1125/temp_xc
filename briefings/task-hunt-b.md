---
status: active
created: 2026-07-24
for: runpod-e
venue: runpod (GPU, new pod)
---

# The task hunt, arm B — repetition-lag Δ (+ confidence trend)

**You are `runpod-e`** — a NEW GPU pod (`/workspace/.agent_id` =
`runpod-e`; see `agents/README.md` + `agents/runpod-e/STATUS.md`).
**Read `briefings/task-hunt.md` FIRST** — its goal statement, two-stage
fail-fast protocol, Stage-1/Stage-2 specs, and acceptance rules govern
you identically; this briefing only assigns your candidates. You have your
**own independent 700 GB volume** (no cross-pod mounts) — build all
caches locally and keep them on the persistent volume so a repurposed
pod inherits them. Shared hunt log:
`experiments/explorations/task_hunt/LOG.md` (append-only, pull-rebase).
Deadline: results by 2026-07-26 morning PT. Fail fast.

## Your candidates

1. **Repetition-lag Δ (exact labels, threshold T-scaling — top prior).**
   Latent: distance Δ to the previous occurrence of the current
   n-gram in natural text (fineweb slice; labels computed exactly from
   tokens — runpod-b is building them via `task-hunt-prep.md`, pull as
   they land or build inline from its committed script). Provably
   non-ambient (no single token knows Δ); recovery of lag-Δ structure
   needs T > Δ ⇒ **built-in threshold scaling: sweep Δ ∈ {4, 8, 16}
   and show each Δ-bucket turns on as T crosses it** — the money plot
   is per-Δ recovery vs T. **Screen across model SCALE** — induction
   heads convert repetition structure into per-token features, and
   conversion capacity grows with size, so the frozen prior is that
   the unconverted gap is LARGER in smaller models: screen gpt2-small,
   **gemma-2-2b BASE** (the substrate-audit convention for web text),
   and Llama-3.1-8B base (minutes each to cache one mid-depth layer);
   record the scale-ordering of the gap (a finding either way); Stage 2
   runs the single best (model) cell at T ∈ {2, 4, 8, 16, 32}.
2. **Confidence-trend (grounded backup, clock-mismatch risk).**
   Windowed hedging→commitment slope on the R1-Distill traces
   (runpod-b's targets + clock-bridge stats); token windows must span
   2+ sentences — screen at T ∈ {16, 32, 64} and treat an honest
   "window cannot reach the trend's timescale at panel-feasible T" as
   a valid kill. Screen only after candidate 1's Stage-1 verdict is
   committed.
3. **Emotional-instability onset (NEW, Han 2026-07-24 — "Gemma needs
   help", `docs/papers/gemma_needs_help.md`).** Substrate:
   **gemma-3-12b-it** (27b-it only if 12b's elicitation rate is too
   low) — the IT model is CORRECT here: the phenomenon is created by
   Gemma's post-training (the paper's base-vs-instruct prefill result),
   the same fit-where-the-phenomenon-lives rule as the EM organism.
   Elicitation is fully specified in the paper: impossible-numeric
   puzzles + neutral rejections, 8 turns; per-turn mean frustration
   rises 1.5 → 5.5 — a **graded escalation trend**, not a rollout
   boolean. Labels: per-response frustration 0–10 (the paper's judge
   prompt, verbatim in our doc) + within-turn onset token (their
   validated onset-labeler prompt, App C.1); ≤ $40 judge budget,
   prereg + κ on 30 traces before scaling (em_onset convention). Two
   readouts, in order: (a) pre-onset anticipation at frozen offsets
   before the labeled onset token (the em_onset D+ design); (b)
   escalation-intensity regression (turn-indexed trend). **The trap,
   named in the card:** post-onset *detection* is lexically stamped
   ("frustrated", "myself", "breath" — the paper's differential-word
   tables) ⇒ predicted regime-1 / per-token-readable; that is the
   sanity anchor, not the target — a card claiming detection dies at
   the gate. Timescale risk as candidate 2: the trend lives at turn
   scale, so screen T ∈ {16, 32, 64} with the same honest kill. Order
   vs candidate 2: decide on feasibility after candidate 1's verdict —
   if runpod-b's clock-bridge stats already kill candidate 2's
   timescale, come straight here.

## Acceptance gate — stop for review

Per the main briefing: mini-cards frozen pre-screen; every verdict in
the shared LOG; survivor(s) through Stage 2 with the T-scaling figure;
STATUS rewritten; canonical-runner hygiene; no reviewer/meeting quotes
in tracked files. Briefing stays until mac-local review.
