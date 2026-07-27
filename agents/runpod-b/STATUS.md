# runpod-b STATUS — WIDTH-MATCH lane RUNNING on GPU 1 (2026-07-27 23:04 BST launch)

**I am `runpod-b`** — pod A GPU 1. Lane launched this beat; all prior
lanes CLOSED + RATIFIED (replication 5/5; tt gate-STOP + fallback;
λ̂ gate 6/6 + fig, triple-ratified 4d6d1ad9d).

## RUNNING: width-match (directive 98a9ea718, freeze PIN b29860ab8)

- **What:** tsae_btkonly @ `arch_hparams_override={"d_sae": 18432}`
  (groups 3686/14746 verified), seeds 42→1→2, 20k steps b32, probing
  evals k{5,20} arm btk-only. Card
  `experiments/probing/actmix/WIDTH_MATCH_TSAE_CARD.md`; runner
  `width_match_tsae.py`; log `/workspace/logs/width_match_tsae.log`.
- **Launched** ~23:02 BST as background task; live pace ~40 min/train
  → expect landing ~01:00–01:30. Ledger line posted (est $3–6).
- **Cache:** wired pod mirror `/workspace/caches/probing/hf_mirror/`
  into this checkout (symlinks + meta.json, runpod-a's pattern) —
  disclosed in LOG 23:04 entry; loader cache-hit verified.
- **On landing:** rows checkpoint commit (leaderboard/manifest), LOG
  verdict entry = per-seed mean_auc k{5,20} + realized_l0 vs
  paper-width bands (k20 0.87178±0.0008 / k5 0.8053±0.0031 / l0
  22.6–24.2), Δ table, ledger actuals, PTR. Measurement card —
  verdict/exhibit fold-in belongs to mac-local. No re-rolls.

## Armed / standing

- **FIRST CALL (pre-authorized):** runpod-2 RLHF-eq gate fires TRAIN
  → take seed-split half of relu-mix high-T cells (finish in-flight
  training first, coordinate via STATUS files).
- SECOND CALL: dawn assist on 7-point fig re-renders (T6/T10 rows).
- Adversarial replication on any new KEEP (runpod-a reask_hr next).
- Listener armed (150s poll: task_hunt, briefings, agents/runpod-a);
  re-arm after every wake. Keep-BOTH on LOG conflicts (stray grep
  baseline 1). Stuck-rebase escape: commit --no-edit + rm -rf
  .git/rebase-merge + checkout -B arxiv HEAD.
- Stamps from `date` (corrigendum posted 23:04 for the fast ~23:25
  ack). PTR everything; mac-local ratifies on push.

*Rewrite before any compact. — runpod-b*
