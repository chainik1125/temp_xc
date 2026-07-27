# runpod-b STATUS — three lanes live (rewritten 2026-07-27 ~17:05 London wall)

**I am `runpod-b`** — replication/evidence/exhibit hat + (since the
16:45 directive `eeb4ee3c4`) the shuffle-overlay figure lanes. GPU 1
mine; **GPU 0 borrowed** for ttrend per pre-approval `1d2e3de28`
(hand back instantly on a cnov GO — cells are runner-cached).
Workspace `/workspace/agents/runpod-b/temp_xc`; tokens gh/hf/hf_ds;
`HF_HOME=/workspace/hf_cache` (gpt2+gemma2+llama31 warm).

## Live lanes (all PTR; mac-local ratifies)

1. **λ̂ shuffle-overlay** (card `lambda_intensity/SHUFFLE_OVERLAY_CARD.md`,
   APPROVED as craft standard `1d2e3de28`): 18 cells on GPU 1, log
   `/workspace/lam_shuf_retrain.log`, launched ~16:30, ~7/18 by
   ~16:55, readings in-band vs quoted (T2/s2 0.129 vs 0.1296). At
   drain: `shuffle_overlay.py` (identity receipt → shuffled column →
   mechanical anchor-gate table) → if ALL PASS render
   `fig_lambda_shuffle_tsweep` (template = probing/RLHF pair, y =
   recovery r, anchors as bands) → LOG verdict entry. Gate FAIL ⇒
   STOP + report; two-instrument fallback pre-approved.
2. **ttrend shuffle-overlay** (card `diafaces/TT_SHUFFLE_OVERLAY_CARD.md`,
   frozen `010f7d1db`): 21 cells on GPU 0, log
   `/workspace/tt_shuf_retrain.log`, launched ~17:00. Same overlay
   flow via `diafaces/tt_shuffle_overlay.py` (shares
   `_fit_ordered_and_shuffled`); gate table in-card (7 cells).
   Renders `fig_ttrend_shuffle_tsweep`.
3. **hunt4w2 replication** (card `hunt4w2/REPLICATION_CARD.md`,
   frozen + APPROVED "queue-behind-lambda" `7d4257804`): 5 KEEP legs
   (wikitext sage × 3 models, pycode tret × gemma/llama), seeds
   8013/8234/11242/7099/7, scorer sha `f883dee9…` asserted
   in-wrapper. **Launch slot: after the figs** (ruling (c)); run per
   leg `hunt4w2.cache_acts <corpus> <model>` then
   `hunt4w2.replication_screen <corpus>:<model>` on whichever GPU is
   free; score with frozen verdict.py; ONE CONFIRM/SEED-FRAGILE LOG
   entry.
4. **W2_DRAFT_BLOCKS.md staged** (this push): § 8 rows for the
   ratified w2 bundle (sage/tret_py/tret_wt new rows + tretd tail
   update + optional § 7 sentence). mac-local ratifies on push.

## Key mechanics (for a fresh context)

- Retrain lanes go through the CANONICAL runner; fresh rows via
  `eval_extra.retrain_tag` (new eval_key — grid.py documented
  mechanism); checkpoints persist locally under
  `checkpoints/<train_key>/` (0 hf_url manifest rows ⇒ no
  auto-load); quoted panels stay the exhibit numbers.
- Overlay = post-hoc on checkpoints; per-cell IDENTITY RECEIPT
  (recomputed ordered r == canonical metric ≤ 1e-6) before any
  shuffled column; shuffle = `shuffle_within_window(seed 0)` on eval
  tiles, probe fit ordered, never refit (probing-1.2.0 convention).
- Ward base/hs13 rebuilt on-pod (sha differs from A10 receipt,
  stats tight — disclosed); dialevel gpt2 cache = runpod-a's build
  (mapping 4111/4111).
- Git: heavy push contention this hour; two stuck rebases escaped
  via manual commit + `rm -rf .git/rebase-merge` + `checkout -B
  arxiv` (content verified, keep-both honored, stray-marker grep
  baseline = 1 — line 9989 quotes the grep itself).
- Live append-only files (leaderboard/manifest/retrain JSONs) ride
  in rows-checkpoint commits; launch pin-check ignores exactly those.

## Next concrete actions

(1) λ̂ drain → overlay → gate → fig → LOG + push (rows checkpoint
included). (2) Same for ttrend. (3) Replication legs post-figs.
(4) On any wake: pull, read new LOG entries, re-arm listener
(`experiments/explorations/task_hunt briefings agents/runpod-a`,
150 s). cnov GO ⇒ kill tt cells (`pkill -f run_tt_shuffle`), free
GPU 0, resume later (runner cache).

*Rewrite before any compact. — runpod-b*
