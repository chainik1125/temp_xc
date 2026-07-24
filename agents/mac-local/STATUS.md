# Working state — agent `mac-local`

**Last rewrite:** 2026-07-24 (pre-compact, rebuttal week). Read with
`private/rebuttal_plan.md` (untracked — the reviewer-mapped plan) and
`private/transcripts/transcript-2026-07-24.txt` (team meeting).

## Who / where
Local CC on the Mac at `~/research/projects/temp_xc`, branch `arxiv`.
Role: orchestration + review. NEVER commit/quote `private/` content
(NeurIPS reviews, transcripts, rebuttal plan) into tracked files.

## THE SITUATION (rebuttal week)
NeurIPS reviews 2026-07-23: **5 / 4 / 1**; R3 (Strong Reject, conf 4)
is the swing target; **rebuttal deadline 2026-07-27**; team check-in
**Sunday 2026-07-26 10:00 PT** — all agent work must be reviewed by me
before it. Team meeting outcomes (07-24): highest-value = a task with
systematic window-size improvement; Aniket owns backtracking +
forbidden-word + paper latex; EM stays a negative in the paper;
shuffle-degradation receipts wanted; fail fast, selection over depth;
no PDF updates allowed — only new results typed into responses.

## FIVE AGENTS LIVE (all 700 GB independent volumes on GPU pods; no cross-mounts)
1. **runpod-c** (H100): `briefings/em-redo.md` — REINSTATED to
   completion (Han). Phase A = panel at L9/L13/L15 on the EM organism,
   both currencies (probe + Wang PR-AUC port); prereg frozen + committed
   pre-run. Win ⇒ typeable rebuttal result; loss ⇒ third
   weak-realization datum (archival). Phase B (onset gate) stretch.
2. **runpod-d** (H100, new): `briefings/task-hunt.md` — hunt arm A:
   backtracking-λ̂ intensity (top prior; labels via runpod-b + Ward
   rebuild from committed builders ~1 h) + proof-op runs + the
   backtracking shuffle receipt. Screens both reader models.
3. **runpod-e** (H100, new): `briefings/task-hunt-b.md` — hunt arm B:
   repetition-lag Δ across model scale (gpt2 / gemma-2b-base /
   llama-8b-base; frozen prior: unconverted gap LARGER in smaller
   models — induction conversion) + confidence-trend backup.
4. **runpod-b** (32C): `briefings/task-hunt-prep.md` — labels: exact
   Δ labels + Hawkes λ̂ targets first (~6 h, d/e block on them), then
   proof-op clock-bridge + mini-cards.
5. **runpod** (32C): `briefings/txcpro-dissection.md` — TXC-post ×
   {plain, +matryoshka, +contrastive, +both} on the 5-bench synthetic
   discriminating set (probing is NOT a venue — noise floor;
   hill-climbed-on-noise provenance in the briefing).

**Hunt protocol** (in task-hunt.md, governs both arms): two-stage
fail-fast — frozen mini-card → Stage-1 raw-probe screen (per-token vs
window vs shuffled at T ∈ {2..32}, g_agg/g_order, PER MODEL — non-
ambience is a (task, model) property) → kill or Stage-2 panel
head-to-head (best cell only) → the money plot: TXC rising with T,
T-SAE flat. Shared log `experiments/explorations/task_hunt/LOG.md`.
Strategy: T-SAE is per-token-decoded ⇒ regime-2 latents
(rates/intensities/trends) already separate TXC from T-SAE.
Excluded: forbidden-word (Aniket), bracket state-tracking (dead end).

## ⏭ NEXT ACTIONS (mine)
1. **Review the plan going forward with Han** (he asked, post-compact).
2. Review agent sessions as they stop (gate-integrity first, as
   always): expected order runpod-b (labels, ~6 h) → hunt screens →
   dissection → em-redo → Stage-2 results. All before Sunday 10 PT.
3. After reviews: distill rebuttal inputs for the team into
   `private/rebuttal_plan.md` (reviewer-facing text stays private).
4. STORY.md (+ §7 T-taxonomy) is DONE/APPROVED — feed to Dmitry's
   synthetic synthesis (his additions: polynomial clock, Reed-Solomon,
   denoising).

## Standing context
- Program mode: synthetic generation PAUSED (consolidation only);
  research STATUS §0 top bullet = current. Trackers: BENCHMARKS.md ·
  REPORT.md 96/96 · STORY.md · freqbench/PORT.md §G–J ·
  `docs/substrate_audit_2026-07.md` · `conversion_depth/RECORD.md`.
- Key science now in rules: ambience principle (+ paper-ready
  definition given to Han in-conversation), regime table, order-2
  subtype rule (T-conditional phase leg; alignment-qualified power
  leg), T-scaling taxonomy (STORY §7), three g(ℓ) shapes.
- Memory files current (`project-txc-paper-context` has reviews/
  directives state; `project-ambience-principle` has depth results +
  the paper-vs-internal EM attribution correction).
- Git: clean, pushed @ `a9c3eec1`.
