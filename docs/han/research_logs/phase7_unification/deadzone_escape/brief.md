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
> commits, coordination docs, and writeups as Agent W (or "W") — Han is
> tracking work-by-agent across this conversation, and Agent Y / Agent X
> are doing parallel work under their own identities.

### One-line state

5 of 7 deadzone-escape ckpts trained on the A40 pod and pushed to HF
(verify in `/tmp/hf_push.log` if A40 still alive); 3 ckpts left to train +
the eval pipeline still needs to run on H100.

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
   positions per step out of `T_max`; full T_max at inference. Forces
   encoder to learn local features that work across windows.
2. **Decoder masking** (spatial Matryoshka) — encoder sees full T but
   feature-prefix levels are charged with reconstructing only random
   subsets of positions. Smallest prefix (H) → reconstruct random
   single-position; full prefix → reconstruct all T positions. Forces
   small prefixes into position-flexible features.
3. **Contrastive strength** (shifts) — `shifts=(T,)` forces
   full-window consistency; `shifts=(2,)` only nearby-position
   consistency. Weaker constraint → more localized features tolerated.

### The 7-step training chain (4 ckpts ALREADY trained, 3 left)

All on T=10, k_pos=20, k_win=200, seed=42, k=200, sd=42.

| # | arch_id | trained? | recipe |
|---|---------|----------|--------|
| 0 | `txc_h8_t10_kpos20_shifts10` | ✓ A40 138min | OBLIT baseline (deadzone-test); shifts=(10,) full-window contrastive — **predicted to FAIL per hypothesis** |
| 1 | `txc_h8_t10_kpos20_shifts2` | ✓ A40 137min | shifts=(2,) lever — same arch as 0, just weaker contrastive |
| 2 | `subseq_h8_tmax10_tsamp5_kpos20_shifts2_ctg` | ✓ A40 143min | Encoder mask, contiguous chunks |
| 3 | `subseq_h8_tmax10_tsamp5_kpos20_shifts2_gauss_s1.5_3.0_g2` | ✓ A40 134min | Encoder mask, mixture-of-Gaussians |
| 4 | `spatial_matry_h8_t10_kpos20_shifts2_pref3686_9216_18432_sub1_5_10_indep_uniform_contr` | ✓ A40 195min | Decoder mask, indep uniform |
| 5 | `spatial_matry_h8_..._nested_uniform_contr` | ⏳ PENDING (was step 5 of chain) | Decoder mask, nested uniform |
| 6 | `spatial_matry_h8_..._indep_gauss_s1.5_3.0_g2_contr` | ⏳ PENDING | Decoder mask, indep Gaussian-mixture |
| 7 | `spatial_matry_h8_..._nested_gauss_s1.5_3.0_g2_contr` | ⏳ PENDING | Decoder mask, nested Gaussian |

All converged at step 3000 (plateau threshold 0.02 hit early — typical for
H8 family). Plateau values 0.011-0.013, all healthy.

**HF location:** `han1823123123/txcdr-base/ckpts/<run_id>.pt`. Verify by
`huggingface_hub.HfApi().list_repo_files("han1823123123/txcdr-base")`. The
training_logs JSONs are at the same repo under `training_logs/`.

### Architectures — files in src/

- `src/architectures/txc_bare_multidistance_contrastive_antidead.py` — H8
  stack (anti-dead + matryoshka H/L + multi-distance contrastive).
- `src/architectures/phase5b_subseq_sampling_txcdr.py::SubseqH8` — encoder
  masking; supports `sampling_mode` ∈ {contiguous, random, gaussian}.
  Gaussian implementation: `_sample_subset_indices(...)` mixture of
  Gaussians per row.
- `src/architectures/spatial_matryoshka_h8.py::SpatialMatryoshkaH8` —
  random-subset Matryoshka decoder loss (NEW, Han's idea). Subclass of
  TXCBareMultiDistanceContrastiveAntidead. Forward adds:
    ```
    if nested:    subsets = self._sample_nested_subsets(T_max, B, device)
    else:         subsets = self._sample_independent_subsets(T_max, B, device)
    l_sm = sum_i ||(x_subset_i - sae.decode(z[:, :prefix_i])_subset_i)||^2
    ```
  Knobs: `level_prefix_sizes`, `level_subset_sizes`, `nested`,
  `subset_sampling_mode`, `sigma_range`, `n_gaussians`,
  `enable_contrastive`. Smoke-tested all 4 nested×{uniform,gaussian}
  combos before training.

All three classes are registered in
`experiments/phase7_unification/case_studies/_arch_utils.py::WINDOW_CLASSES`
so the standard pipeline picks them up.

### Trainer scripts

- `experiments/phase7_unification/case_studies/train_kpos20_h8_shifts.py`
  (steps 0, 1)
- `experiments/phase7_unification/case_studies/train_kpos20_subseq_h8.py`
  (steps 2, 3)
- `experiments/phase7_unification/case_studies/train_kpos20_spatial_matryoshka.py`
  (steps 4, 5, 6, 7)

The chain script is committed at
`experiments/phase7_unification/case_studies/_t10_chain.sh`. **For H100
resume**, edit it to start at step 5 (or run trainers 5,6,7 directly):
```bash
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
    --T 10 --shifts 2 --seed 42 --subset-mode uniform --nested --no-hf-push
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
    --T 10 --shifts 2 --seed 42 --subset-mode gaussian \
    --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 --no-hf-push
TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka \
    --T 10 --shifts 2 --seed 42 --subset-mode gaussian \
    --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 --nested --no-hf-push
```

After each, push to HF with `--no-hf-push` removed OR run
`/tmp/push_t10_ckpts_to_hf.py` (template at that path on A40, recreate on
H100) updated to include the new run_id.

### Eval pipeline (after all 7 archs trained)

The driver script is committed at
`experiments/phase7_unification/case_studies/_eval_t10_chain.sh`. It runs:

1. `select_features.py --archs <all 8> --seed 42` — uses (a)-style shared
   Gemma forward + disk cache (commit 37c7233a).
2. `diagnose_z_magnitudes.py --archs <all 8> --seed 42` — reuses
   share_acts mechanism for L12 capture.
3. `intervene_paper_clamp_normalised.py` — multi-process (b)-style
   parallel: `N_GROUPS=2` on A40 (28GB total), bump to **`N_GROUPS=5`
   on H100** (~70GB; comfortable on 80GB):
   ```
   N_GROUPS=5 bash _eval_t10_chain.sh
   ```
   Each process is independent (own CUDA context), B=7 strengths
   per arch, **bit-parity-preserved vs sequential** (commit 2ee2ae3b).
4. `grade_with_sonnet.py --archs <all 8> --subdir steering_paper_normalised`
   — API parallel via `--n-workers`.

Step 3 is the bottleneck. On H100 with N_GROUPS=5 + 4x compute speed,
expected total eval ~30-45 min for all 8 archs.

### Apples-to-apples concerns for paper

The T-SAE / TXC / Galaxy baselines (T=2, 3, 5) were eval'd on A40 with
B=7. New T=10 archs eval'd on H100 with B=7 (same per-process batch).
**Fp16/bf16 reductions are kernel-implementation-dependent**, so cross-GPU
greedy-decode bit drift is possible (rare, ~<1%). For strict
apples-to-apples, **re-eval the n=3 multi-seed baselines on H100 too**
(adds ~30-45 min):
```bash
N_GROUPS=5 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised \
    --archs tsae_paper_k20 txc_h8_t2_kpos20_shifts2 \
            txc_maxpool_h8_t2_kpos20_shifts2 \
            txc_contrastive_h8_t2_kpos20_shifts2 \
            txc_softmax_pool_h8_t2_kpos20_shifts2 \
            galaxy18_g8_t3_kpos20 \
    --seed 42
# then re-grade those, then bootstrap
```

### Hypothesis predictions (pre-registered)

If Han's deadzone hypothesis is correct:

- arch 0 (T=10 OBLIT shifts=10): **cliff15 ≪ 1.13**, low succ_avg
  near coh_avg=2 — confirms deadzone.
- arch 1 (T=10 shifts=2): **mildly better** than 0; tests pure
  contrastive-strength lever in isolation.
- arch 2-3 (subseq, encoder masking): **better than 1**; encoder-side
  fix.
- arch 4-7 (spatial Matry, decoder masking): **best**; combined
  encoder+decoder fix.

If ANY of arch 2-7 hits cliff15 ≥ 1.13, that's a genuine paper claim
(escape velocity at T=10).

If all 8 hit cliff15 ≪ 1.13, the deadzone is broader than the
training-loss hypothesis predicts — paper retreats to the "T=2 is the
clean cell" framing already documented.

### Coordination state

- Agent W (me) has been running this chain on the A40 pod.
- Agent Y has separately verified that scaling **reverses** at T=5
  (Galaxy 23 G8 T=5 RE results in `agent_y_phase2/2026-05-02-...md`
  area; check `git log`). Y also added V8 encoded-broadcast steering
  protocol at commit `9862cba1`.
- Agent X had been doing 3-seed T=8 benchmarks (commit `7b61e8ec`).

For the H100 resume, no coordination needed beyond reading the most
recent W → Y round briefings:
- `agent_w_to_y_round6.md` — Gaussian-splat sampler explanation
- `agent_w_to_y_round7.md` — Spatial Matryoshka design + 7-step chain

### Tooling notes (for the agent)

- All Python via `.venv/bin/python` after `uv sync`.
- HF / GitHub / Anthropic tokens are at `/workspace/.tokens/` once volume
  is restored. (The /workspace migration is whole-volume — tokens go too.
  Han may need to re-attach if RunPod's volume migration loses anything.)
- `TQDM_DISABLE=1` is required (CLAUDE.md rule).
- The Sonnet 4.6 grader is what we've been using throughout; do NOT
  switch to a different grader (Han confirmed in this session — apples-
  to-apples critical given how late we are).
- Memory parallelism for intervene: A40 supports N_GROUPS=2,
  H100 supports N_GROUPS=5. Override via env var.

### Files written/modified by W during this session (committed to git)

Core:
- `src/architectures/spatial_matryoshka_h8.py` (NEW — d3d117c0)
- `src/architectures/phase5b_subseq_sampling_txcdr.py` (Gaussian sampler
  added — 1995e095)

Trainers:
- `experiments/phase7_unification/case_studies/train_kpos20_spatial_matryoshka.py`
- `experiments/phase7_unification/case_studies/train_kpos20_subseq_h8.py`

Eval pipeline (a/b speedups):
- `experiments/phase7_unification/case_studies/steering/select_features.py`
  (a-style shared Gemma + disk cache — 37c7233a)
- `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py`
  (a-style shared Gemma — 618e247b)

Drivers:
- `experiments/phase7_unification/case_studies/_t10_chain.sh` (training
  chain, 7 archs sequential)
- `experiments/phase7_unification/case_studies/_eval_t10_chain.sh` (eval,
  multi-process intervene with N_GROUPS env var — c863eda1)

Coordination:
- `docs/han/research_logs/phase7_unification/agent_w_to_y_round{6,7}.md`

### What to do FIRST on H100 pod

1. `bash /workspace/temp_xc/scripts/restart_recovery.sh` — installs uv,
   restores .bashrc, restores ~/.claude symlink, syncs venv. Idempotent.
2. Verify HF ckpts available:
   ```python
   from huggingface_hub import HfApi
   files = HfApi().list_repo_files("han1823123123/txcdr-base")
   for f in files:
       if "txc_h8_t10" in f or "spatial_matry" in f or "subseq_h8_tmax10" in f:
           print(f)
   ```
3. Pull the 5 trained ckpts back to local disk (scripts in
   `experiments/phase7_unification/case_studies/_download_ckpts.py` may
   help; check usage).
4. Write a chain script for the 3 remaining archs (template below) and
   run it.
5. Run eval with `N_GROUPS=5`.
6. Update writeup + Pareto plots; if any deadzone-escape arch shows
   cliff15 ≥ 1.13, that's the headline.

```bash
# After H100 + venv up:
cd /workspace/temp_xc
TQDM_DISABLE=1 bash <<'EOF' &> /tmp/h100_chain.log
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka --T 10 --shifts 2 --seed 42 --subset-mode uniform --nested
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka --T 10 --shifts 2 --seed 42 --subset-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2
.venv/bin/python -m experiments.phase7_unification.case_studies.train_kpos20_spatial_matryoshka --T 10 --shifts 2 --seed 42 --subset-mode gaussian --sigma-lo 1.5 --sigma-hi 3.0 --n-gaussians 2 --nested
EOF
disown
# Each arch ~35 min on H100 (vs 140-195 min on A40)
```

— W
