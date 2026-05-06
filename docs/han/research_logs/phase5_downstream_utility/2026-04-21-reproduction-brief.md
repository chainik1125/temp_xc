---
author: Han
date: 2026-04-21
tags:
  - guide
  - reference
  - in-progress
---

## Phase 5 reproduction briefing (fresh-box local reproduction)

**Audience**: A new agent on a fresh local box (no activation cache, no
ckpts) tasked with reproducing Phase 5's **25-arch benchmark at
`last_position` + `mean_pool`** and the **TXCDR T-sweep** from scratch.

**Goal (per Han's ask)**: verify there are no critical oversights in
the pipeline by independently regenerating the cached activations,
training the SAEs, probing, and comparing your numbers against the
checked-in `summary.md`.

**Out of scope**: Phase 5.7 autoresearch (in flight), `full_window`
aggregation (deprecated, kept in JSONL only for the record).

> **Shortcut**: if your goal is just re-probing against the committed
> ckpts rather than verifying the activation/cache pipeline, skip
> Steps 1-4 below and download everything from HuggingFace instead —
> see [`docs/huggingface-artifacts.md`](../../../huggingface-artifacts.md).
> That's a ~40-min download vs the ~12-15 hr full rebuild.

### Target state

Reproduce the benchmark that's committed on branch `han`. When this
brief was written the head was `fcdfb95` (post-A2 v0 with the baseline
skip bug fixed). Any commit >= `883faa7` ("Phase 5 T-sweep: 3-way
aggregation comparison") already has the 25-arch + T-sweep results
locked in.

    git log --oneline | head -20

Ground truth to diff against:

- `docs/han/research_logs/phase5_downstream_utility/summary.md` — the
  25-arch leaderboard tables at both aggregations + the T-sweep
  section (headline numbers reproduced here in "Expected results").
- `experiments/phase5_downstream_utility/results/headline_summary_*.json`
  — raw aggregated AUC/acc per (arch, task-set).
- `experiments/phase5_downstream_utility/results/txcdr_t_sweep_summary_{auc,acc}.json`
  — T-sweep numbers.
- `experiments/phase5_downstream_utility/results/probing_results.jsonl`
  — row-level results.

### Hardware / software assumptions

- **GPU**: RTX 5090 (32 GB VRAM). Phase 5 was run on an A40 (48 GB).
  Most 25-arch steps will fit; see *"Memory pressure points"* below
  for the two that may OOM on 32 GB.
- **Python**: ≥ 3.12 via `uv sync` (uses `uv.lock`).
- **Disk**: ~120 GB free before starting. Budget:
  - Activation cache (5 layers × 24 k seqs × 128 tokens × 2304 d ×
    fp16) ≈ 17.7 GB. Plus `token_ids.npy` 6 MB.
  - Probe cache (36 tasks × anchor/MLC/MLC-tail arrays, fp16) ≈ 50-60
    GB.
  - Checkpoints (25 × ~850 MB fp16) ≈ 22 GB.
  - Plots + JSONL + logs ≈ 1 GB.
- **HuggingFace**: accept the Gemma license at
  <https://huggingface.co/google/gemma-2-2b-it> and export `HF_TOKEN`.
  FineWeb (HuggingFaceFW/fineweb, `sample-10BT`) has no gate.

### One-time setup

```bash
git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc
git checkout han

curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv missing
uv sync                                             # installs .venv/

export HF_TOKEN=<your_gemma_accepted_token>
export HF_HOME=~/hf_cache                           # or wherever
huggingface-cli login --token "$HF_TOKEN"
export TQDM_DISABLE=1
export PYTHONPATH=$(pwd)
```

Sanity check:

```bash
.venv/bin/python -c "
import torch
from transformers import AutoModelForCausalLM
print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))
m = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it', torch_dtype=torch.float16)
print('gemma loaded, d_model =', m.config.hidden_size)
"
```

If `gemma loaded` prints without a 401/gated error you're good.

### Pipeline (in order)

Each step writes to a deterministic location and is independently
resumable (each runner has a per-entry "already exists → skip" guard).
Estimated wall-clock is measured on the A40; 5090 is typically within
±20 %.

#### Step 1 — Tokenise FineWeb + cache L13 (~20 min)

`src.data.nlp.cache_activations` writes both `token_ids.npy` and a
per-layer residual-stream tensor in one pass. Run it once for L13; the
remaining 4 layers (L11/L12/L14/L15) are added by Step 2 via a
dedicated resumable single-layer builder.

```bash
.venv/bin/python -m src.data.nlp.cache_activations \
  --model gemma-2-2b-it --dataset fineweb --mode forward \
  --num-sequences 24000 --seq-length 128 \
  --layer_indices 13
```

Output:
  - `data/cached_activations/gemma-2-2b-it/fineweb/token_ids.npy`
    shape `(24000, 128)`
  - `data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy`
    shape `(24000, 128, 2304)` fp16 (~3.5 GB)

#### Step 2 — Cache the remaining MLC layers L11, L12, L14, L15

Phase 5's MLC uses a 5-layer window centred on L13. L13 was cached in
Step 1; add the other four via the single-layer-per-invocation
resumable builder (MooseFS on our pod didn't tolerate 4-memmap
concurrency; 5090's local NVMe is fine, but the script is already
single-layer so keep it that way). Each layer is ~3.5 GB fp16.
Wall-clock ~15 min per layer on A40.

```bash
for L in 11 12 14 15; do
  .venv/bin/python experiments/phase5_downstream_utility/build_multilayer_cache.py \
    --layer $L
done
```

Resumable: each invocation drops a `.resid_L{L}.progress.json`
sidecar. Crash → re-run the same command.

Verify:

```bash
ls -la data/cached_activations/gemma-2-2b-it/fineweb/
# expect: token_ids.npy + 5 × resid_L{11..15}.npy (each ~3.5 GB)
```

Each `.npy` should open as shape `(24000, 128, 2304)` fp16.

#### Step 3 — Build probe activation cache (36 tasks)

Loads each probing dataset, runs Gemma-2-2B-IT on 3040 train + 760 test
sentences per task, caches the tail-20-token L13 activations, the MLC
5-layer activation at the last real token, AND the MLC × tail-20
tensor (`acts_mlc_tail.npz`, needed for `mean_pool`). Wall-clock ~60
min; each task's cache is 500 MB-2 GB.

```bash
.venv/bin/python experiments/phase5_downstream_utility/probing/build_probe_cache.py \
  --include-crosstoken
```

`--include-crosstoken` adds WinoGrande + WSC so you end up with 36
tasks (= 34 + 2). Without it you get 34.

Verify:

```bash
ls experiments/phase5_downstream_utility/results/probe_cache/ | wc -l
# expect: 36
ls experiments/phase5_downstream_utility/results/probe_cache/ag_news_business/
# expect: acts_anchor.npz  acts_mlc.npz  acts_mlc_tail.npz  meta.json
```

`acts_mlc_tail.npz` is what unlocks `mean_pool` / `full_window` for MLC
+ `time_layer_crosscoder_t5`. If it's missing, those two archs will
skip with a warning in `run_probing.py`.

#### Step 4 — Train the 25 architectures at seed 42

Full set (from `summary.md` §*Architectures — 25 total*):

```
topk_sae                          # token SAE
mlc  mlc_contrastive              # layer crosscoder family (5 layers)
txcdr_t2  txcdr_t3  txcdr_t5      # T-sweep
txcdr_t8  txcdr_t10  txcdr_t15  txcdr_t20
stacked_t5  stacked_t20           # stacked per-position
matryoshka_t5                     # position-nested Matryoshka
txcdr_shared_dec_t5  txcdr_shared_enc_t5  txcdr_tied_t5
txcdr_pos_t5  txcdr_causal_t5                       # weight-sharing
txcdr_block_sparse_t5
txcdr_lowrank_dec_t5  txcdr_rank_k_dec_t5           # decoder rank
temporal_contrastive                                 # Ye et al. 2025 token SAE
time_layer_crosscoder_t5                             # joint (T, L) crosscoder
tfa_small  tfa_pos_small                             # TFA at seq_len=32
```

Use the trainer's CLI directly — it preloads activation buffers once,
scales well per-arch, and plateau-stops at <2 %/1k step loss drop
(max 25 000 steps). Runs serially on one GPU. Wall-clock on A40:

- Single-anchor archs (TXCDR, stacked, topk, matryoshka, TFA): ~20-35
  min each.
- MLC family + time_layer_crosscoder_t5 (5-layer buffer ~18 GB GPU):
  ~15-25 min.
- 25 archs total ≈ 10-12 h.

```bash
.venv/bin/python experiments/phase5_downstream_utility/train_primary_archs.py \
  --seeds 42 --max-steps 25000 \
  --archs topk_sae mlc mlc_contrastive \
          txcdr_t2 txcdr_t3 txcdr_t5 txcdr_t8 txcdr_t10 txcdr_t15 txcdr_t20 \
          stacked_t5 stacked_t20 matryoshka_t5 \
          txcdr_shared_dec_t5 txcdr_shared_enc_t5 txcdr_tied_t5 \
          txcdr_pos_t5 txcdr_causal_t5 txcdr_block_sparse_t5 \
          txcdr_lowrank_dec_t5 txcdr_rank_k_dec_t5 \
          temporal_contrastive time_layer_crosscoder_t5 \
          tfa_small tfa_pos_small
```

Expect one `.pt` per arch in `results/ckpts/` and one row per arch in
`results/training_index.jsonl`.

If the terminal drops: re-running is safe — every arch that already has
a `ckpts/<arch>__seed42.pt` will still be trained again (the CLI has no
"skip if exists" guard at the arch level; plateau-stop catches it near
the same step though). To skip already-trained archs, remove them from
`--archs` manually.

#### Step 5 — Probe at `last_position` + `mean_pool`

Two probing passes, same CLI:

```bash
# last_position (baselines + all 25 archs)
.venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation last_position

# mean_pool (same set)
.venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation mean_pool
```

Each pass ~1-2 h for all 25 archs × 36 tasks × 4 k-values. Streams one
task at a time to cap RAM at ~5 GB (this is important on a 32 GB
host).

Verify row count after both passes:

```bash
.venv/bin/python -c "
import json
from pathlib import Path
rows = [json.loads(l) for l in Path('experiments/phase5_downstream_utility/results/probing_results.jsonl').open()]
by_agg = {}
for r in rows:
  if r.get('error') or r.get('test_auc') is None: continue
  by_agg.setdefault(r['aggregation'], set()).add((r['run_id'], r['task_name'], r.get('k_feat')))
for a in sorted(by_agg):
  print(a, len(by_agg[a]))
"
# expect (minimum): last_position >= 3600, mean_pool >= 3600
# 25 archs × 36 tasks × 4 k + 2 baselines × 36 tasks × 1 k = 3672, give or take.
```

#### Step 6 — Headline plots

```bash
.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py
```

Produces the 8 headline bars + 8 per-task heatmaps + 8
`headline_summary_*.json` files. Diff the JSONs against the ones
checked into the repo as the smoke-test pass/fail signal.

```bash
# For each aggregation × metric × taskset, check mean AUC per arch
diff <(jq -S . experiments/phase5_downstream_utility/results/headline_summary_last_position_auc_full.json) \
     <(git show origin/han:experiments/phase5_downstream_utility/results/headline_summary_last_position_auc_full.json | jq -S .)
```

#### Step 7 — T-sweep plot

The 7 T values are already trained in Step 4 (T ∈ {2,3,5,8,10,15,20}).
All three aggregations have been run in Step 5. Regenerate the plot:

```bash
.venv/bin/python experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep.py
```

Writes `results/plots/txcdr_t_sweep_{auc,acc}.png` +
`results/txcdr_t_sweep_summary_{auc,acc}.json`. Diff the JSONs against
the committed ones.

### Expected results (ground truth to match)

Headline at `last_position` × k=5 × full task set (top of leaderboard):

| arch | mean AUC |
|---|---|
| mlc_contrastive | 0.8025 |
| mlc | 0.7943 |
| time_layer_crosscoder_t5 | 0.7928 |
| txcdr_rank_k_dec_t5 | 0.7852 |
| txcdr_t5 | 0.7822 |

Headline at `mean_pool` × k=5 × full task set:

| arch | mean AUC |
|---|---|
| txcdr_t5 | 0.8064 |
| txcdr_t3 | 0.8022 |
| txcdr_rank_k_dec_t5 | 0.7990 |
| txcdr_t15 | 0.7868 |
| mlc | 0.7848 |

Baselines in both cases: `baseline_attn_pool` ≈ 0.929,
`baseline_last_token_lr` ≈ 0.926. Full tables in `summary.md`.

T-sweep at `mean_pool` × k=5 (all T):

| T | 2 | 3 | 5 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|
| AUC | 0.7786 | 0.8022 | **0.8064** | 0.7711 | 0.7754 | 0.7868 | 0.7545 |

**Tolerance**: Phase-5 was a single-seed experiment (seed 42). Expect
your numbers to match within **±0.005 AUC** on mean-across-tasks. If
any single arch's mean AUC diverges by >0.01 from the committed
number, that's a yellow flag — inspect training log
(`training_logs/<arch>__seed42.json`) for convergence + final-loss
parity. >0.02 is red.

Note: probing is not bit-deterministic even at fixed seed — the
top-k-by-class-separation step ties break non-deterministically on
ties in the abs-mean-diff. The existing JSONL has ~300/13000 row
pairs that disagree by >0.005 AUC when the same run is executed
twice. The aggregated mean-across-tasks number should still match
within ±0.005.

### Memory pressure points (32 GB VRAM)

Phase-5 A40 peaks:

- MLC + MLC-contrastive training: ~32 GB GPU (multilayer buffer 18 GB
  + Adam state + forward/backward tensors).
- time_layer_crosscoder_t5 training: kept at `d_sae=8192` specifically
  because `d_sae=18432` wouldn't fit 48 GB A40. On 32 GB 5090, 8192
  should still fit; don't try raising it.
- txcdr_t20 training: full-T=20 window encoder has 20 × d_in × d_sae
  = 850 M param encoder tensor. Fits on A40 at batch_size=1024; if the
  5090 OOMs, drop `batch_size` in
  `train_primary_archs.py::TrainCfg` from 1024 → 512.
- mean_pool probing for MLC / mlc_contrastive / time_layer_crosscoder_t5:
  streams slide-by-slide; peak VRAM ~5 GB; RAM ~8 GB.

If you OOM on the multilayer-buffered runs, drop
`PRELOAD_SEQS` in `train_primary_archs.py` (default 6000) to 3000.

### Known gotchas

- **cgroup / memory limit**: on the RunPod A40 pod the cgroup limit is
  46 GB and page cache hugs the limit, failcnt=0. This is *not* a
  concern on a bare-metal 5090 — your OS is free to use all RAM. Don't
  be alarmed by `free -h` showing low "available" memory during probe
  cache streaming; it's evictable page cache.
- **TQDM_DISABLE=1**: without it the training/probing logs are
  unreadable. Always export it.
- **Non-writable-numpy warning**: the training script triggers a
  `UserWarning: The given NumPy array is not writable` when preloading
  the anchor buffer. Ignore — we read-only the mmap.
- **`train_primary_archs.py` has no per-arch skip**: re-running trains
  everything in `--archs` regardless of existing ckpts. List only the
  archs you need.
- **Probing is non-deterministic** (see "Tolerance" above). If you're
  comparing row-by-row against the committed JSONL, expect disagreement
  on ~2 % of rows at the ±0.005 AUC scale.
- **sklearn deprecation noise**: `LogisticRegression(penalty='l1',
  ...)` triggers deprecation warnings in sklearn ≥ 1.8. Harmless.
- **If `acts_mlc_tail.npz` is missing** on some task, MLC /
  mlc_contrastive / time_layer_crosscoder_t5 will skip at `mean_pool`
  and `full_window` with a message in the log. Re-run
  `build_probe_cache.py` — the missing tasks are usually the
  cross-token ones added late in Phase 5.

### Critical oversight checks

Things the re-run is *specifically* supposed to catch:

1. **Data leakage**. The committed `leakage_audit.json` reports 0/875
   probe-sentence signature hits in the FineWeb cache. Re-run this
   audit:
   ```bash
   .venv/bin/python experiments/phase5_downstream_utility/leakage_audit.py
   diff results/leakage_audit.json <(git show origin/han:.../leakage_audit.json)
   ```
2. **Probe-cache consistency**. Each task's `meta.json` should include
   `dataset_key`, `n_train`, `n_test`, `tail_mlc_n=20`. The split
   sizes must be train=3040 / test=760 on every task that isn't
   cross-token (cross-token are smaller).
3. **MLC layer window**. `mlc`, `mlc_contrastive`, and
   `time_layer_crosscoder_t5` must all use the L11-L15 window. Inspect
   the first batch's shape during training: should be `(B, 5, 2304)`
   for MLC, `(B, 5, 5, 2304)` for time_layer.
4. **`aggregation` tags in JSONL**. No `last_position` row should
   appear from `acts_mlc_tail.npz` (which is tail-20, not a single
   position). No `mean_pool` or `full_window` row should use a single
   last-token encoding. Grep:
   ```bash
   jq -r '.aggregation' results/probing_results.jsonl | sort -u
   # expect: last_position, mean_pool, full_window, last_position_val, mean_pool_val
   ```
   `*_val` rows come from Phase-5.7 autoresearch (train-split-only);
   ignore them for the reproduction of the 25-arch bench.
5. **Per-arch encode path**. The "fairness" principle is that every
   arch's probing encoder uses the same input tensor at the same
   token/window position. Spot-check by reading the relevant
   `_encode_*` in `probing/run_probing.py` and tracing back to each
   arch's `decoder_directions_at()` / `encode()`.

### What NOT to do

- **Do not** attempt to reproduce `full_window`. It's deprecated; new
  plots skip it. The committed `full_window` rows are retained for the
  historical record only.
- **Do not** train Phase-5.7 archs (`txcdr_contrastive_t5`,
  `txcdr_rotational_t5`, …). Those are autoresearch in-flight; their
  training + val/test split are live on branch `han` but are not part
  of the 25-arch reproduction.
- **Do not** run `run_overnight_phase5.sh`, `run_autoresearch.sh`, or
  `overnight_sprint.sh` blindly — those are RunPod-specific
  orchestrators with pgrep waits + commit-and-push loops. On a local
  5090 box just run the six steps above directly.

### If results diverge

1. Pick the single arch that's furthest from the committed number.
2. Read its `training_logs/<arch>__seed42.json`. Compare
   `final_step`, `converged`, `final_loss`, `final_l0`, `elapsed_s`
   against the committed row. A mismatch here usually points to either
   (a) activation cache built with the wrong `seq_len` or dtype, or
   (b) preload subset size (`PRELOAD_SEQS`) affecting sampling.
3. Probe just that one arch at just one task in isolation, then hand-
   diff the `decision_score` array against the per-example prediction
   files in `results/predictions/` (committed for the 7 top archs at
   last_position only).

### Open question for the reproducer agent

The activation cache is ~18 GB of fp16 floats. It's the single
biggest source of "did you really reproduce this" confidence. If
possible, also run `experiments/phase5_downstream_utility/leakage_audit.py`
on the freshly-built cache and verify 0/875 signature hits against the
probe set. A hash of `resid_L13.npy` can't match ours bit-for-bit
(Gemma is non-deterministic at fp16 ≥ batch dimension 1), but the
*derived statistics* (mean / L2-norm per position) should match within
1e-4.
