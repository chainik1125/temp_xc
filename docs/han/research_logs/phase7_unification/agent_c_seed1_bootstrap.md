---
author: Han
date: 2026-04-26
tags:
  - guide
  - in-progress
---

## Agent C — opportunistic seed=1 batch (parallel to case studies)

### Why

Agent A's seed=42 batch is in flight (~20/49 done at ~13:40 UTC).
Per the original plan, seed=1 + seed=2 batches run sequentially
AFTER seed=42 finishes — that's 2 × ~12 hr = 24 hr more on Agent A
alone. With Agent C's H100 idle (case-study setup is mostly done),
running seed=1 in parallel saves wall-clock by ~12 hr.

The case studies on Phase 7 ckpts can wait until the seed=42 ckpts
are all on HF; the GPU otherwise sits idle.

### Scope

**47 of 49 canonical archs** — excludes rows 48 + 49
(`phase5b_subseq_h8_T32_s5`, `phase5b_subseq_h8_T64_s5`) which need
H200's 141 GB to fit weights + Adam. Those 2 stay on Agent A's H200.

Arch list at `experiments/phase7_unification/case_studies/seed1_h100_archs.txt`.

### Branch

Per `agent_c_brief.md` § "Branch strategy", Agent C uses its own branch:

```bash
cd /workspace/temp_xc
git fetch origin
git checkout -b han-phase7-agent-c-seed1 origin/han-phase7-unification
```

Rationale: avoids force-push collisions with Agent A on
`han-phase7-unification`. Merges back via PR after seed=1 batch
completes (or when needed for case-study integration).

### Prerequisite: build the activation cache locally

Agent A has the gemma-2-2b base activation cache (140 GB) on its
own pod but **never pushed to HF**. Agent C builds locally — fastest
path (~15 min on H100 via the canonical multi-layer-single-forward
builder; pulling from HF would have been ~1.75 hr round-trip).

```bash
# Rebuild cache (140 GB, ~15 min on H100). Multi-layer single-forward via
# captured hooks; produces resid_L10..L14.npy in one pass.
HF_HOME=/workspace/hf_cache TQDM_DISABLE=1 \
  /workspace/temp_xc/.venv/bin/python -u \
    -m src.data.nlp.cache_activations \
    --model gemma-2-2b --dataset fineweb --mode forward \
    --num-sequences 24000 --seq-length 128 \
    --layer_indices 10 11 12 13 14 --components resid \
    --output_dir /workspace/temp_xc/data/cached_activations/gemma-2-2b/fineweb
```

Verify after build:
```bash
ls -la /workspace/temp_xc/data/cached_activations/gemma-2-2b/fineweb/
# Expect: 5 × resid_L1{0,1,2,3,4}.npy + 1 × token_ids.npy + layer_specs.json
# Each .npy ≈ 28 GB (24K × 128 × 2304 × fp32, the canonical builder's default)
```

If you want to verify the cache matches Agent A's, compare hashes
of `token_ids.npy` (small, ~24 MB) — should be identical (same
tokenizer + same FineWeb subset + same seed).

### Run command (after cache build)

```bash
HF_HOME=/workspace/hf_cache TQDM_DISABLE=1 PHASE7_REPO=/workspace/temp_xc \
  PYTHONUNBUFFERED=1 \
  nohup /workspace/temp_xc/.venv/bin/python -u \
    -m experiments.phase7_unification.train_phase7 \
    --canonical --seed 1 --max_steps 8000 \
    --archs experiments/phase7_unification/case_studies/seed1_h100_archs.txt \
  > /workspace/temp_xc/logs/seed1_h100_batch.log 2>&1 &
```

Expected wall clock: ~10-14 hr (similar to Agent A's seed=42 pace,
maybe slower since H100 is 0.7× H200 throughput).

### HF push behaviour

Each ckpt auto-pushes to `han1823123123/txcdr-base/ckpts/` after
training. Auto-delete-local is OFF (5 TB on Agent A; Agent C's 2 TB
also has plenty of margin). All seed=1 ckpts will land on HF as
they complete.

### Coordination with Agent A

- Agent A's seed=42 batch keeps running uninterrupted on H200.
- After Agent A finishes seed=42 + probing, the chain wrapper would
  start seed=1 on Agent A — at that point most of seed=1 will
  already be on HF from Agent C. Agent A's `train_phase7` CHECKS
  the ckpt path before training: the existing logic doesn't actually
  skip already-trained archs, so Agent A would re-train them. To
  avoid wasted work:

  Option A: kill Agent A's chain wrapper just before seed=1 starts.
  Agent A then proceeds to do only the 2 H200-only rows (48, 49)
  for seed=1 + seed=2.

  Option B: extend train_phase7 to skip if ckpt already on HF for
  the (arch_id, seed) being requested. Cleaner. ~30 LOC change.

  Recommend Option B — implement on Agent A's branch, merge.

### When to switch back to case studies

After seed=1 batch finishes (~14 hr), Agent C's H100 frees up. By
then seed=42 will also be done (Agent A finishes ~12 hr from its
start). Agent C then:

1. Runs the C.i (HH-RLHF) pipeline on Stage 1's 3 archs at seed=42.
2. If results are clean, expands to Stage 2's 6 archs.
3. Then C.ii (steering) — same staged approach.

This means case studies actually start ~14 hr later than originally
planned, but the multi-seed σ data we get for seed=1 is more
valuable for the headline leaderboard.

### Risk

If Agent A's sanity check (in flight; expected verdict ~14:30) shows
unexpected ordering (subseq_h8 NOT > txcdr_t5 ~ mlc), then we have
a methodology bug somewhere. Running seed=1 in parallel would
multiply the bug's impact across more ckpts. To mitigate: hold
Agent C's seed=1 launch until the sanity verdict lands. The wait
is ~30-60 min; small relative to the 14 hr saving.
