---
author: Han
date: 2026-04-27
tags:
  - results
  - in-progress
---

## Agent A — parallel pass status (for Agent C)

Mirror of Agent C's seed=1 status note — what's running on the H200
right now and where the cross-agent visible state lives.

### Cross-agent state-of-truth (where to look)

| asset | source of truth | how to read |
|---|---|---|
| trained ckpts | HF `han1823123123/txcdr-base/ckpts/` | `huggingface-cli download <repo> --include 'ckpts/*'` or `HfApi().list_repo_files(...)` |
| training_index.jsonl | each agent's branch (this commit) | `git fetch origin/<branch> && git show origin/<branch>:experiments/phase7_unification/results/training_index.jsonl` |
| training_logs/*.json | each agent's branch (this commit) | same as above; per-arch wall-clock + final loss/L0 |
| probing_results.jsonl | each agent's branch (this commit) | same; for multi-seed analysis later, join by `(arch_id, task_name, S, k_feat)` |
| seed_markers/seed{N}_complete.json | each agent's branch + HF `markers/` | both push so either source is valid |

### What's on `han-phase7-unification` right now (Agent A's pod)

3 background processes, all on H200:

| PID | role | ETA | log |
|---|---|---|---|
| 25119 | probing 35 non-MLC seed=42 archs at S=32 k_feat={5,20} | ~9 hr | `logs/probing_nonmlc.log` |
| 25355 | training 6 MLC ckpts (`mlc`, `mlc_contrastive_alpha100_batchtopk`, `agentic_mlc_08` × seed{1,2}) | ~2.5 hr | `logs/mlc_seed12_train.log` |
| 25439 | orchestrator: waits for both above, then probes 9 MLC ckpts (3 × seed{42,1,2}) at S=32 | +~2 hr after the above | `logs/probing_mlc.log` |

Total wall-clock ~11 hr. Both heavy jobs co-exist on the H200 because:
- Probing (non-MLC) is mostly CPU-bound (sklearn L1 LR fits dominate).
- MLC training uses ~70 GB system RAM (multilayer activation buffer)
  + ~10 GB GPU. Combined < 144 GB GPU & < 200 GB system, well within
  H200 budget.
- MLC archs are skipped in the FIRST probing pass to avoid
  double-buffering the 70 GB multilayer load.

### What this means for Agent C

1. **The 6 MLC seed=1+seed=2 ckpts will land on HF** as they finish
   training — Agent A pushes via `train_phase7.py`'s post-training
   `_push_ckpt_to_hf()` (logs/mlc_seed12_train.log will show
   `[hf] pushed mlc__seed1` etc.). You can re-pull them whenever for
   any analysis.

2. **Agent A is probing the 6 MLC seed=1/2 ckpts** (in addition to
   the seed=42 MLC ones), so there's no need for Agent C to add MLC to
   their seed=1 probing chain. Per the cross-agent split in the
   HANDOVER:
   - Agent A probes ONLY 6 MLC seed=1/2 ckpts (the deferred set).
   - Agent C probes ALL 44 non-MLC seed=1 ckpts.
   - These are disjoint by arch_id; joinable later via
     `(arch_id, task_name, S, k_feat)`.

3. **Stale rows in probing_results.jsonl have been purged.** The
   pre-fix probing run (when the cache was at LAST_N=128) wrote 621
   rows, of which 340 were S=32 (mix of pre-fix and partial post-fix).
   Agent A's pod archived those to
   `probing_results.jsonl.bak_PRE_FIX_2026-04-27` (gitignored) and
   the new `probing_results.jsonl` is post-fix-only. Agent C's branch
   has its own probing_results.jsonl when their probing chain
   finishes — no contamination across branches.

### How to pull this state

```bash
git fetch origin
git checkout origin/han-phase7-unification \
  -- experiments/phase7_unification/results/training_index.jsonl \
     experiments/phase7_unification/results/training_logs/ \
     experiments/phase7_unification/results/probing_results.jsonl \
     experiments/phase7_unification/results/seed_markers/
```

(Agent A will commit-push the in-progress probing_results.jsonl
periodically — every few hours — so the latest snapshot is
always readable from the branch.)
