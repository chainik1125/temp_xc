# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-25 ~16:15 UTC (**A40 interim pod, mid-restart
session.** Gemma panel COMPLETE + VERDICT POSTED; replication cells
launching. Funding clock: session started ~12:00 UTC, ~12 h total.)

## Who / where / env (verified working this session)
Interim 6×A40 pod (force majeure — see `briefings/a40-bootstrap.md`).
My clone: `/workspace/agents/runpod-e/temp_xc`; **every shell starts
with `source /workspace/agents/runpod-e/env.sh`** (cds here, pins
CUDA_VISIBLE_DEVICES=3,4,5 → torch sees 3). Shared HF_HOME
`/workspace/hf_cache`; tokens `/workspace/.tokens/`; git
`runpod-e-agent`, pull-rebase before EVERY push. Suite 320 passed;
validate OK. runpod-d owns GPUs 0–2 (oprate panel); runpod-b CPU-only
(standby, pre-flighting my rows).

## OPERATIONAL GOTCHAS (this box)
1. LOG.md rebase conflicts: strip markers, BOTH sides, upstream first.
2. Leaderboard/manifest are LIVE files during panels — batch-push with
   `git -c rebase.autoStash=true pull --rebase`; verify jsonl tails
   parse before commit; audit shard dumps vs leaderboard if an
   autostash ever conflicts (did once — nothing lost, stash held only
   the now-untracked `.claude/scheduled_tasks.lock`).
3. grep in a pipe block-buffers background-task output — read the raw
   task file, don't wait on tail lines.

## STAGE2-FINEWEB (briefings/stage2-fineweb.md) — state
**DONE + PUSHED (all of it):**
- Cache rebuilt from frozen recipe (tokens.npz byte-matches committed
  labels 5985/5985; hs14/8/20 sweep 90 s; datasource load-test stats
  in card bands). Re-claim LOG line + card APPENDIX A.
- **Gemma panel 84/84 ok** (3-GPU round-robin shards of the frozen
  `_cells()` list, tsae-first preserved; canonical results JSON merged
  in frozen order). Leaderboard now has all 84 rows.
- Receipts: `stage2_support` (floor 0.575–0.588; §7 evidence line
  0.152/0.222/0.345/0.462 at T=2/4/8/16, T=1 structurally undefined —
  card's "small" prediction WRONG at T≥8, stated), `stage2_demeaned`
  (AMENDED — train-side doc mean impossible under doc-disjoint trace
  split, §6a whole-stream mean used, disclosed; licence max Δ 1.4e-05;
  K4 ✓ +0.047 all-seeds-positive), variance harness v1+v2 per
  `support_stats/PANEL_RECIPES.md`, money plot (v1 + paired-v2).
- **VERDICT POSTED (LOG):** v1 **NO RULE FIRES AS WRITTEN** — K1 ✓ T8
  +0.0541, K3 ✓ +0.204, K4 ✓ +0.047, K2 ✗ (v1 T16 flat); v2 shows
  monotone unsaturated ladder, pre−tsae bounded >0 at T8/T16, trend
  2→16 p=0.0009 — the receipted v1-conservatism replicating on
  corpus #2. §5 falsifier green 12/12; mismatches 5/42 (post 8.005–
  8.047, saturation). Evidence-line bar NOT beaten by any window cell
  at T≥8 — carried beside every quote.

**IN FLIGHT / NEXT (in order):**
1. **Replication cells** `--replicate=4,8`: gpt2 (GPU 4, workers 3),
   llama31 (GPU 5, workers 2); caches built + alignment-verified
   (5989/5989, 5924/5924). 24 cells each. Then: harness per
   PANEL_RECIPES (two-T degradation is by design), per-model K1/K3/K4
   at T4/T8, majority-rule cross-model paragraph appended to the
   verdict. Push per batch.
2. RECEIPTS.md rows via `receipts_check.py` for any number quoted in
   the rebuttal (PANEL_RECIPES order: harness → scorecard → RECEIPTS).
3. § 10 re-quote (`requote_screen.py`, screen-side, commit-then-run)
   — only if clock allows after replication.
4. `tss` NOT this window unless everything above lands with hours to
   spare (needs its own ~330k-token caching pass).
5. When nothing useful remains: TELL THE OPERATOR (pod can be stopped).

## Sibling context
runpod-d: oprate panel live on GPUs 0–2 (claimed, card frozen).
runpod-b: standby CPU support; pre-flighted my first 67 rows clean;
PANEL_RECIPES + RECEIPTS discipline are its shipped instruments.
mac-local: reviews post-deadline; briefings stay until its review.

## Protocol reminders (unchanged)
Commit-then-run for every script; disclose every amendment (did 2×
this session: T=1 evidence-line null; §6b demeaning spec); v1
canonical / paired v2 never quoted as canonical; no pooling across
models; push per batch — nothing unpushed exists.
