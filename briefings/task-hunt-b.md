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
you identically; this briefing only assigns your candidates. **The
shared 700 GB volume is mounted at `/shared`** (conventions in
`agents/README.md`): read anything; **write ONLY under
`/shared/task_hunt_caches/e/`** (so your caches survive repurposing);
never touch `/shared/temp_xc` or `/shared/.agent_id`; reuse weights
from `/shared/hf` by copy/symlink, new downloads to your LOCAL
`HF_HOME`. Shared hunt log:
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

## Acceptance gate — stop for review

Per the main briefing: mini-cards frozen pre-screen; every verdict in
the shared LOG; survivor(s) through Stage 2 with the T-scaling figure;
STATUS rewritten; canonical-runner hygiene; no reviewer/meeting quotes
in tracked files. Briefing stays until mac-local review.
