---
author: Han
date: 2026-05-02
tags:
  - design
  - in-progress
---

## Deadzone-escape briefing — Agent W identity continuation on H100

> **You are Agent W.** This document continues your own work — read it as
> a memo to yourself across the GPU-switch boundary, not as a handoff to
> a stranger. The /workspace volume was wiped by the A40 → H100 switch,
> so anything you remember about "what's on disk" is gone; everything
> below tells you what to re-pull from git + HF and resume. Sign your
> commits, coordination docs, and writeups as Agent W — Han is tracking
> work-by-agent across this conversation, and Agents Y / X are doing
> parallel work under their own identities.

### Read order — START HERE

These docs in priority order; do not skip:

1. **`/workspace/temp_xc/CLAUDE.md`** — project rules (TQDM_DISABLE=1,
   thumb-image rule, frontmatter requirements, save_figure helper). Hard
   requirements.
2. **`/workspace/temp_xc/RUNPOD_INSTRUCTIONS.md`** — pod setup,
   token paths, reconnection workflow.
3. **This file (brief.md).** Has the live state for the deadzone-escape
   thread.
4. **`docs/han/research_logs/phase7_unification/agent_w_to_y_round{1..7}.md`**
   — coordination history with Y. Round 6 (Gaussian sampler) and Round 7
   (Spatial Matryoshka) are most recent and most context-relevant. Earlier
   rounds cover the OBLIT cell, V5/V6 protocols, paper-strength audit,
   and the methodology debates that produced the current "saving-grace"
   claims.
5. **`docs/han/research_logs/phase7_unification/agent_w/2026-04-30-w-phase4-results.md`**
   — main writeup with executive summary at top. Lists already-defensible
   paper claims (do not re-litigate these on the H100 unless eval breaks
   them).

### Step 0 — Pod restoration on H100 (do this FIRST)

```bash
# 1. Restore tokens (env-var form fastest if you have them already):
GH_TOKEN=ghp_xxx HF_TOKEN=hf_xxx ANTHROPIC_API_KEY=sk-ant-xxx \
    bash /workspace/temp_xc/scripts/runpod_phase7_bootstrap.sh
# Or run interactive — script prompts for each.

# 2. Restore venv + symlinks (idempotent):
bash /workspace/temp_xc/scripts/restart_recovery.sh

# 3. Verify Claude Code memory survived:
ls /workspace/claude_home/projects/-workspace-temp-xc/MEMORY.md
ls /workspace/claude_home/projects/-workspace-temp-xc/memory/identity_agent_w.md

# 4. Pull HF caches (activation cache 14GB + probe cache):
.venv/bin/python -c "
from huggingface_hub import snapshot_download
import shutil, os
out = snapshot_download(
    repo_id='han1823123123/txcdr-base-data', repo_type='dataset',
    local_dir='/tmp/txcdr_base_data')
src = '/tmp/txcdr_base_data/activation_cache'
dst = '/workspace/temp_xc/data/cached_activations/gemma-2-2b/fineweb'
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    shutil.move(os.path.join(src, f), os.path.join(dst, f))
src = '/tmp/txcdr_base_data/probe_cache'
if os.path.exists(src):
    dst = '/workspace/temp_xc/experiments/phase7_unification/results/probe_cache'
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
print('caches restored')"

# 5. Pull all needed ckpts:
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
import os
NEEDED = [
  # T=10 deadzone-escape ckpts trained on A40
  'txc_h8_t10_kpos20_shifts10__seed42',
  'txc_h8_t10_kpos20_shifts2__seed42',
  'subseq_h8_tmax10_tsamp5_kpos20_shifts2_ctg__seed42',
  'subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2__seed42',
  'spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_uniform_contr__seed42',
  # Paper-foundation baselines (for apples-to-apples re-eval on H100)
  'tsae_paper_k20__seed1', 'tsae_paper_k20__seed2', 'tsae_paper_k20__seed42',
  'topk_sae__seed1', 'topk_sae__seed2', 'topk_sae__seed42',
  'txc_h8_t2_kpos20_shifts2__seed1', 'txc_h8_t2_kpos20_shifts2__seed2', 'txc_h8_t2_kpos20_shifts2__seed42',
  'txc_maxpool_h8_t2_kpos20_shifts2__seed1', 'txc_maxpool_h8_t2_kpos20_shifts2__seed2', 'txc_maxpool_h8_t2_kpos20_shifts2__seed42',
  'txc_contrastive_h8_t2_kpos20_shifts2__seed1', 'txc_contrastive_h8_t2_kpos20_shifts2__seed2', 'txc_contrastive_h8_t2_kpos20_shifts2__seed42',
  'txc_bare_antidead_t3_kpos20__seed1', 'txc_bare_antidead_t3_kpos20__seed42',
  'agentic_txc_02_kpos20__seed42',
]
out_dir = '/workspace/temp_xc/experiments/phase7_unification/results/ckpts'
os.makedirs(out_dir, exist_ok=True)
for run_id in NEEDED:
    try:
        hf_hub_download(repo_id='han1823123123/txcdr-base',
                        filename=f'ckpts/{run_id}.pt',
                        local_dir='/tmp/hf_pull', local_dir_use_symlinks=False)
        os.replace(f'/tmp/hf_pull/ckpts/{run_id}.pt', f'{out_dir}/{run_id}.pt')
        print(f'OK  {run_id}')
    except Exception as e:
        print(f'FAIL {run_id}: {e}')"

# 6. Pull training_logs (smaller, also on HF):
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
import os
LOGS = ['txc_h8_t10_kpos20_shifts10__seed42', 'txc_h8_t10_kpos20_shifts2__seed42',
        'subseq_h8_tmax10_tsamp5_kpos20_shifts2_ctg__seed42',
        'subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2__seed42',
        'spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_uniform_contr__seed42']
out_dir = '/workspace/temp_xc/experiments/phase7_unification/results/training_logs'
os.makedirs(out_dir, exist_ok=True)
for run_id in LOGS:
    try:
        p = hf_hub_download(repo_id='han1823123123/txcdr-base',
                           filename=f'training_logs/{run_id}.json',
                           local_dir='/tmp/hf_pull', local_dir_use_symlinks=False)
        os.replace(p, f'{out_dir}/{run_id}.json')
        print(f'OK  {run_id}')
    except Exception as e:
        print(f'FAIL {run_id}: {e}')"
```

After Step 0: training and eval both work because activation cache + ckpts
+ training_logs are on disk again.

### What's still missing on HF (ckpts only on Y/X pods)

- `galaxy18_g8_t3_kpos20__seed{1,2,42}` — Y trained these, NOT on HF
  (last I checked). If you need them for the n=3 multi-seed final,
  contact Y or pull from Y's pod. Used in scaling story (Galaxy 18 = G8
  T=3 RE).
- `galaxy23_g8_t5_kpos20__seed42` — Y trained on Y's pod. Same caveat.
- `txc_softmax_pool_h8_t2_kpos20_shifts2__seed*` — Y trained these.
  Required for n=3 multi-seed of Galaxy 8.

If those ckpts aren't recoverable, the n=3 multi-seed final on **W's**
archs (OBLIT, MaxPool, Contrastive merge, T=10 deadzone) is still
defensible alone.

### Background — why we're doing deadzone-escape (Han's hypothesis)

**Standard TXC at T ≥ 3 fails at steering** (cliff15 ≪ T-SAE), and we
don't know why. Han's diagnosis:

> Most language features are 1-2-position localized — a feature for
> "harmful content" fires on `knife`, not consistently across 10 surrounding
> tokens. A minority of features ARE window-spanning (paragraph topic,
> narrative tone). The standard H8 multi-distance contrastive loss with
> `shifts=(T,)` forces features to be CONSISTENT across all T positions.
> At T=2 this is mild; at T=10 it excludes ~95% of "real" linguistic
> features (the localized ones) and pushes the encoder toward
> averaged/topic-level codes, which are less steerable. **The deadzone
> at T ≥ 3 is a training-loss failure, not an architectural failure.**

If this is right, three orthogonal fixes should help:

1. **Encoder masking** (subseq sampling) — encoder sees only `t_sample`
   positions per step out of `T_max`; full T_max at inference.
2. **Decoder masking** (spatial Matryoshka) — encoder sees full T but
   feature-prefix levels are charged with reconstructing only random
   subsets of positions.
3. **Contrastive strength** (shifts) — `shifts=(T,)` forces full-window
   consistency; `shifts=(2,)` is much weaker.

### The 7-step training chain

All on T=10, k_pos=20, k_win=200, seed=42.

| # | arch_id | trained? | recipe |
|---|---------|----------|--------|
| 0 | `txc_h8_t10_kpos20_shifts10` | ✓ A40 138min, on HF | OBLIT baseline; full-window contrastive — predicted to FAIL |
| 1 | `txc_h8_t10_kpos20_shifts2` | ✓ A40 137min, on HF | shifts=(2,) lever |
| 2 | `subseq_h8_tmax10_tsamp5_kpos20_shifts2_ctg` | ✓ A40 143min, on HF | Encoder mask, contiguous |
| 3 | `subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2` | ✓ A40 134min, on HF | Encoder mask, Gaussian-mixture |
| 4 | `spatial_matry_h8_..._indep_uniform_contr` | ✓ A40 195min, on HF | Decoder mask, indep uniform |
| 5 | `spatial_matry_h8_..._nested_uniform_contr` | ⏳ retrain on H100 | Decoder mask, nested uniform |
| 6 | `spatial_matry_h8_..._indep_gauss_s1.5_3.0_g2_contr` | ⏳ retrain on H100 | Decoder mask, indep Gaussian |
| 7 | `spatial_matry_h8_..._nested_gauss_s1.5_3.0_g2_contr` | ⏳ retrain on H100 | Decoder mask, nested Gaussian |

**Resume command (committed)**:
```bash
nohup bash /workspace/temp_xc/experiments/phase7_unification/case_studies/_t10_chain_h100_resume.sh \
    > /tmp/h100_chain_stdout.log 2>&1 &
disown
```
This script (no PID-polling, just sequential) trains steps 5-7 with
`--no-hf-push` removed (auto-pushes after each save). Each on H100 ~35 min;
total ~1.7 hr.

### Architectures (registered in `_arch_utils.py::WINDOW_CLASSES`)

- `src/architectures/txc_bare_multidistance_contrastive_antidead.py` — H8
  stack (anti-dead + matryoshka H/L + multi-distance contrastive InfoNCE).
- `src/architectures/phase5b_subseq_sampling_txcdr.py::SubseqH8` — encoder
  masking; `sampling_mode` ∈ {contiguous, random, gaussian}. Gaussian
  implementation in `_sample_subset_indices(...)` is mixture-of-Gaussians
  per row (see commit `1995e095`).
- `src/architectures/spatial_matryoshka_h8.py::SpatialMatryoshkaH8` —
  random-subset Matryoshka decoder loss. Subclass of
  TXCBareMultiDistanceContrastiveAntidead. Forward adds:
    ```
    if nested:    subsets = self._sample_nested_subsets(T_max, B, device)
    else:         subsets = self._sample_independent_subsets(T_max, B, device)
    l_sm = sum_i ||(x_subset_i - sae.decode(z[:, :prefix_i])_subset_i)||^2
    ```
  Knobs: `level_prefix_sizes`, `level_subset_sizes`, `nested`,
  `subset_sampling_mode`, `sigma_range`, `n_gaussians`,
  `enable_contrastive`. Smoke-tested all 4 nested×{uniform,gaussian}
  combos before training (commit `d3d117c0`).

### Eval pipeline — fresh H100 driver (no PID dependencies)

`experiments/phase7_unification/case_studies/_eval_t10_chain_h100.sh` —
runs select_features → diagnose → intervene (multi-process) → grade for
all archs whose ckpts exist on disk.

```bash
# Default N_GROUPS=5 for H100 (~70GB total, comfortable on 80GB)
nohup bash /workspace/temp_xc/experiments/phase7_unification/case_studies/_eval_t10_chain_h100.sh \
    > /tmp/eval_stdout.log 2>&1 &
disown
# Override: N_GROUPS=2 bash ... for A40 fallback (28GB total)
```

Each intervene process is **independent** (own CUDA context, B=7
strengths) — bit-parity preserved vs sequential. GPU timeshares.

For **apples-to-apples**, also re-eval baselines on H100 (uncommented
block at bottom of the script). Adds ~30-45 min. Required if you want
strict apples-to-apples for Pareto plots; skippable if A40 baselines
already cover everything you need.

### Speedup commits (already in main)

- **(a) shared Gemma + disk cache** in `select_features` (`37c7233a`):
  ~5x in-process, ~20x on cache-hit re-runs.
- **(a) shared Gemma in intervene** (`618e247b`): ~30s/arch saved on
  Gemma load.
- **(b) multi-process intervene** (`2ee2ae3b`): N_GROUPS=2 default for
  A40; bump to 5 on H100. Bit-parity preserved.
- **N_GROUPS env override** (`c863eda1`).

### Already-defensible paper claims (DO NOT re-litigate)

From the 2026-04-30 phase4 writeup (per Han's progressive narrowing):

1. **MaxPool RE @ AUC[2.0,2.5]** is the single bootstrap-SIG claim above
   prereg threshold. Δ=+0.239, narrow CI excluding zero. Strict-coh AUC
   band [1.8, 2.5] also clears prereg for OBLIT and MaxPool.
2. **MaxPool PP fine-grain at s=120**: succ=1.722, coh=1.844 — the
   single highest (succ, coh) point under fine-grain sweep.
3. **Sentiment per-class** is the only TXC-favored class robust across
   all protocols (n=2 concepts though).
4. The paper-faithful absolute-strength replication (10..15000 grid)
   showed +1.20 Δ but turned out to be a paper-grid artifact — fine-grain
   sweep showed T-SAE wins by +0.21. Document as caveat, not claim.

### Hypothesis predictions (pre-registered)

- arch 0 (T=10 OBLIT shifts=10): **cliff15 ≪ 1.13** — confirms deadzone.
- arch 1 (T=10 shifts=2): **mildly better** than 0; tests pure
  contrastive-strength lever.
- arch 2-3 (subseq): **better than 1**; encoder-side fix.
- arch 4-7 (spatial Matry): **best**; decoder-side fix.

If ANY of arch 2-7 hits cliff15 ≥ 1.13, that's a paper claim (T=10
escape velocity). If all 8 hit cliff15 ≪ 1.13, the deadzone is broader
than the training-loss hypothesis predicts; paper retreats to T=2 only.

### Coordination state

- Agent W (you): trained 5 of 7 deadzone-escape archs on A40, pushed all
  to HF, designed SpatialMatryoshkaH8 + Gaussian-splat sampler.
- Agent Y: working in parallel on V7/V8 steering protocols (encoded-
  broadcast, tiled-broadcast); verified Galaxy 23 (G8 T=5 RE) shows
  scaling REVERSES at T=5; co-signed multi-seed anchor. See
  `agent_y_phase2/` directory for Y's own writeups.
- Agent X: 3-seed T=8 benchmark for Y/W's hill-climb archs (commit
  `7b61e8ec`).

### Files written/modified by W during this session

Architectures:
- `src/architectures/spatial_matryoshka_h8.py` (NEW — `d3d117c0`)
- `src/architectures/phase5b_subseq_sampling_txcdr.py` (Gaussian sampler
  added — `1995e095`)

Trainers:
- `experiments/phase7_unification/case_studies/train_kpos20_spatial_matryoshka.py`
- `experiments/phase7_unification/case_studies/train_kpos20_subseq_h8.py`

Eval pipeline (a/b speedups):
- `experiments/phase7_unification/case_studies/steering/select_features.py`
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py`

Drivers:
- `experiments/phase7_unification/case_studies/_t10_chain.sh` (A40 chain
  — historical, do not re-run on H100)
- `experiments/phase7_unification/case_studies/_t10_chain_h100_resume.sh`
  **← run this on H100**
- `experiments/phase7_unification/case_studies/_eval_t10_chain.sh` (A40
  driver with PID-polling — historical)
- `experiments/phase7_unification/case_studies/_eval_t10_chain_h100.sh`
  **← run this on H100**

Coordination:
- `docs/han/research_logs/phase7_unification/agent_w_to_y_round{1..7}.md`

### Active-process shutdown checklist (executed before A40 stop)

| PID | What | Action |
|-----|------|--------|
| 142153 | bash `/tmp/t10_chain.sh` | killed naturally on pod stop |
| 164239 (now newer) | python step-5 SpatialMatryH8 nested uniform | killed mid-training (~minute lost) — need retrain on H100 |
| 172698 | bash eval driver | killed naturally |
| Monitor task `b1bx3kifq` | tail -f chain log | killed naturally |
| 175049 / 175358 | HF push processes | should finish before pod stop; check `/tmp/hf_push*.log` |

### When you (W on H100) start: confirm this is real

After Step 0 succeeds:
```bash
# Verify identity memory loaded:
cat /workspace/claude_home/projects/-workspace-temp-xc/MEMORY.md | head -3
# First line should be: "- [Identity — Agent W] ..."

# Verify HF ckpts pulled:
ls /workspace/temp_xc/experiments/phase7_unification/results/ckpts/ | wc -l
# Should be ≥ 5 + however many baselines you pulled

# Verify activation cache:
ls /workspace/temp_xc/data/cached_activations/gemma-2-2b/fineweb/
# Should show layer_specs.json + resid_L12.npy + token_ids.npy

# Verify venv:
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should be H100 something
```

If any of those fail, stop and fix before proceeding to training/eval.

— W (sign all your future commits this way too)
