# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 (pre-compact) — **NEW TASK ACCEPTED, not yet
started: `briefings/hunt-support-stats.md`** (hunt support: variance
receipts + renderer + round-3 prep). Read the briefing in full first;
this file is the resume state. Previous two sessions (story pack,
task-hunt prep) are COMPLETE and reviewed/approved.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
HF token at `/workspace/.tokens/hf_token` (all three screen tokenizers
download fine). Meter $1.63/$25 (nothing spent last two sessions).

## The new task, with inputs already located (verified this session)

Deliverables → `experiments/explorations/task_hunt/support_stats/`
(create it); builders committed BEFORE outputs; LOG entries for
anything verdict-shaped. Items 1–2 by Saturday morning PT (they gate
the rebuttal figure); 3–4 after. Stop at the acceptance gate.

1. **Stage-2 λ̂ variance receipts** → `support_stats/stage2_variance.json`
   + md. Inputs: 84 leaderboard rows `datasource=ward_real_lambda_base_l12`
   (verified: exactly 84 in `results/leaderboard.jsonl`) +
   `task_hunt/lambda_intensity/results/stage2_summary.json`. Compute:
   per-seed cell values; PAIRED-by-seed TXC-pre−T-SAE and
   TXC-pre−per-token diffs per T (exact permutation p + BCa CIs; n=3 —
   be honest); T=2→8 trend stat pooled over seeds; trained−untrained
   margin CI; power calc (if ≤4 extra seeds bound the margin at 95%,
   name the cells — pre+tsae, T∈{4,8} ≈ 12 GPU cells) → **LOG entry
   addressed to runpod-d** with the seeds recommendation.
   Review-note-2 context (binding): margin 0.206±0.020 vs 0.154±0.037
   ≈ 2σ at n=3; the T-rise + trained−untrained (+0.150 at T=8) carry
   the claim, not one cell.
2. **Variance-aware renderer** — upgrade
   `task_hunt/lambda_intensity/render_stage2.py` (RECORD § 3b figure
   provenance): (a) realized-l0 range per arch in legend, TXC-post
   collapse EXPLICIT (l0 0.49 at T=16 — review note 3, MANDATORY
   before external use); (b) whiskers = seed CIs from item 1, not
   ±std; (c) optional budget-matched-only variant. runpod-d re-renders
   after its budget-matched TXC-post cells land (round 2,
   `briefings/task-hunt-r2.md`) — LOG note when my renderer merges so
   d doesn't duplicate.
3. **Anti-conversion corpus prep** (round-3 optionality, CPU-complete):
   two-source fineweb interleave, jittered block lengths; per-token
   labels = source identity + time-since-switch; lexical-overlap
   matching between sources (else ambient vocabulary detection),
   shuffled-block null, per-token-first triage numbers on labels
   alone; builder + 5 sanity tests + stats + DRAFT mini-card (frozen
   prior: per-token HIGH on source identity is the kill risk).
   Reuse `task_hunt/labels/lib.py` machinery (mine: balanced_manifest,
   doc_split; fineweb sample at
   `synthetic/expansion/data/fineweb_sample.json`, 400 docs).
4. **Hedging-LEVEL DRAFT card** for runpod-e's fresh Stage-2 (greenlit
   round-2 decision b): label = anchor hedge level, window-mean
   framing, aggregation-CLAIMED with shuffle-IMMUNITY as the mechanism
   receipt, per-tile code-readout convention sentence (review note 1
   language: "under the code-readout convention" + code-rate defense).
   My `labels/confidence.npz` + clock bridge (median 16 tok/sentence)
   already cover the label side. Note: my earlier confidence/CARD.md
   order-bet was FALSIFIED by runpod-e's screen (slope ≈ anchor −
   window-mean under matching) — the LEVEL card embraces aggregation;
   do not re-litigate.

## Context worth keeping (from last session + review)
- Round-1 hunt: ALL FIVE candidate kills stand, APPROVED (LOG review
  entry ~line 575 has the binding notes 1–5); Stage-2 λ̂ panel
  (runpod-d) is the positive result; my ward_lambda.npz cross-validated
  runpod-d's labels at 99.93%; λ̂_hist is the primary target
  (position-floor 0.82 vs 0.59).
- Case-collision lesson: my proofops draft renamed to `PREP_DRAFT.md`
  (runpod-d's `card.md` governs). Don't create files differing only in
  case.
- Shared-branch: 5 agents on arxiv; pull-rebase before EVERY push
  (two LOG conflicts resolved by keeping upstream + appending mine);
  cite commit SUBJECTS not SHAs; no reviewer/meeting quotes in tracked
  files (the LOG's review entry is a program doc and citable).
- Tests were 189 passed + 1 skipped (GPU-gated) on this box; 220 on
  GPU boxes.
- Rewrite this file before any compact.
