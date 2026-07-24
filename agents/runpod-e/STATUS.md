# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 ~02:45, mid-session (candidates 1+2 CLOSED,
starting candidate 3).

## Who / where
H100 pod, `/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-e`.
Task-hunt arm B (`briefings/task-hunt-b.md`). Git `runpod-e-agent`,
creds `store --file=/workspace/.git-credentials`; `export
HF_TOKEN=$(cat /workspace/.tokens/hf_token)` per command (gemma gated);
`ANTHROPIC_API_KEY` in ~/.bashrc from `/workspace/.tokens/`.
Pull-rebase before EVERY push. Deadline 2026-07-26 morning PT.

## DONE (verdicts committed in `experiments/explorations/task_hunt/LOG.md`)
- **Candidate 1 (repetition-lag Δ): KILL** — detection converted
  (regime-1) at all scales gpt2/gemma2/llama8b; order-residue on lag
  VALUE only, big in gpt2 (+0.11), thin at 2B/8B (+0.02). Results:
  `task_hunt/replag/results/` + figs. Volume caches:
  `/workspace/replag_caches/`.
- **Candidate 2 (confidence trend): KILL** — window gap real, monotone
  in T (distill mean-probe 0.521→0.565, tok 0.468), state control
  regime-1, but AGGREGATION-carried (mean ≥ flatten, shuffle retains)
  ⇒ order receipt fails. Regime-2 re-card seed recorded in LOG.
  Results: `task_hunt/confidence/results/`. Ward stream + base/distill
  17-layer caches on `/workspace/conv_depth_caches/` (traces.json
  re-ported per stage_a/ATTRIBUTION.md, gitignored).

## NOW: candidate 3 — emotional-instability onset (gemma-3-12b-it)
Paper protocol in `docs/papers/gemma_needs_help.md` (prompts VERBATIM:
elicitation App B — impossible numeric + 8-turn neutral rejections,
temp 1; judge 0-10 App B.2; onset labeler App C.1). runpod-b draft:
`task_hunt/emotional_instability/CARD.DRAFT.md` — I freeze my own CARD.
Plan (freeze in CARD before each stage runs):
1. **CARD.md**: substrate gemma-3-12b-it; ~300 8-turn conversations
   (30 verified-impossible Countdown/fraction puzzles × 10 rollouts,
   temp 1); labels = per-response frustration 0-10 + within-turn onset
   token (string match from labeler JSON); **κ prereg gate on 30
   dual-judged traces (κ ≥ 0.3, em_onset convention) BEFORE scaling;
   ≤ $40 judge budget**; readouts (a) pre-onset anticipation D-ladder
   (Ward D+ shape: within-[D_lo,D_hi] positives, far negatives, guard
   band, identity+position matching, split by puzzle), (b) escalation
   trend (turn-indexed); post-onset detection = lexically-stamped
   SANITY ANCHOR only (a detection claim dies at the gate). Timescale:
   measure tokens/turn on real rollouts, pick T honestly (screen
   T ∈ {16,32,64}); honest "unreachable at panel-feasible T" kill
   allowed.
2. `emotional_instability/generate.py` — batched HF generation,
   resumable, rollouts to volume (`/workspace/emo_caches/rollouts/`).
3. `judge.py` — Anthropic API: κ pilot (30, dual sonnet+haiku) → gate
   → full per-response scores + onset labels; committed JSONs.
4. `build_labels.py` — token grid (chat template), onset token match,
   turn index, frustration per response; probe manifests.
5. `cache_acts.py` — mid-depth layer (~L24 of 48, d3840) over
   conversations (long sequences, NOT 128-chunks).
6. `screen.py` — frozen problib stack, same conventions as
   replag/confidence screens.
Verdict → LOG; if KEEP → Stage 2 single best cell (canonical runner).

## Sibling state (for context)
runpod-d: λ̂ card frozen, screen pending (its LOG entry). runpod-b:
labels+cards shipped. Aniket's forbidden-word work: SILOED — don't
consume.
