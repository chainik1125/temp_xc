---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Agent Z handover — pre-compaction state

> Author note: written by Z just before context compaction at 46%
> usage. Captures exact state of Z's hill-climb work so a post-compact
> agent (or a follow-up Z) can resume without re-deriving the
> failed-paths history.

### Current goal (post-update from Han)

Han pivoted the hill-climb target on **2026-04-29**:
- Hill-climb metric: **k_feat=20 seed=42 base side**, NOT k_feat=5
  (k=20 is less noisy; σ_seeds=0.0012 vs ~0.005 at k=5).
- Task set: **PAPER (16 tasks)**, NOT FULL-36. Defined in
  `experiments/phase7_unification/task_sets.py::PAPER`. Spec in
  `agent_x_paper/2026-04-29-paper-task-set.md`.
- **Bar to beat at seed=42**: `tsae_paper_k500` = 0.9151 (overall
  leader), `txc_bare_antidead_t5` = 0.9131 (TXC-family leader,
  Δ=+0.0036 over `topk_sae` at the 3-seed mean — described as
  "decisive" in the paper-task-set doc).

### Z's leaderboard cells (PAPER k_feat=20 seed=42)

| arch_id                                | mean AUC | rank | n  | verdict |
|----------------------------------------|----------|------|----|---------|
| tsae_paper_k500                        | 0.9151   | 1    | 16 | leader |
| txc_bare_antidead_t5                   | 0.9131   | 2    | 16 | TXC leader |
| **`hill_subseq_h8_T12_s5`** (V1, Z's)  | **0.9126** | 3 | 16 | ⚠️ tied — only 0.0005 below TXC leader |
| mlc                                    | 0.9116   | 4    | 16 | |
| `hill_h8_T5_shifts32only`  (R2 L2, Z's)| 0.9062   | 13   | 16 | LOSE |
| `hill_h8_T5_shifts1and32`  (R2 L1, Z's)| 0.9025   | 17   | 16 | LOSE |

**Headline finding** (1-seed): V1 is statistically indistinguishable
from `txc_bare_antidead_t5` at PAPER k=20 (Δ=−0.0005). Long-distance
shifts (R2 L1, L2) HURT k=20 vs the standard recipe. Need multi-seed
to confirm V1 robustness.

### Hill-climb directions tried + failed (5090 OOM)

The 5090 (32 GB VRAM) **cannot fit SubseqH8 + matryoshka + multi-
distance contrastive at any T_max ≥ 14** with paper-canonical
batch_size=4096, even with the L=32 cache trick. Memory bottleneck:
encoder forward activations are (B=4096, T_max, d_sae=18432) per
shift × number of shifts = ~1.2 GB per (token × shift) × 4 shifts ×
T_max → 25–50 GB just in encoder activations (cumulative across
shifts). Adam state + decoder + workspace add another ~10 GB.
Reducing `t_sample` doesn't help because the encoder forward stores
activations for ALL T_max positions regardless of which subset is
sampled at the loss path.

Configurations that 5090-OOM'd (don't retry without H200):
- V2: SubseqH8 T_max=16 t_sample=5
- V3: SubseqH8 T_max=20 t_sample=5
- R2 L3: H8 multidist T=8 shifts=[1, 2, 4, 32]
- R2 L4: H8 multidist T=8 shifts=[1, 2, 32]
- R3 M1: SubseqH8 T_max=14 t_sample=5
- R3 M2: SubseqH8 T_max=20 t_sample=2
- R3 M3: SubseqH8 T_max=16 t_sample=2
- R3 M4: SubseqH8 T_max=16 t_sample=5 (V2 retry)

### Configurations that DID fit + train successfully

- V1: SubseqH8 T_max=12 t_sample=5 — Agent A trained on H200
  (RunPod, since destroyed). Already had 72 probing rows in
  `probing_results.jsonl` when Z arrived (probed by X or Agent A
  before the V2 attempt — Z did NOT re-probe). Z fixed the
  `training_index.jsonl` ckpt path from RunPod `/workspace/...` to
  local `/home/elysium/...` and pulled the ckpt from HF.
- R2 L1: TXCBareMultiDistance T=5 shifts=[1, 32]   — Z trained on 5090, 43.8 min, Z probed
- R2 L2: TXCBareMultiDistance T=5 shifts=[32]      — Z trained on 5090, 29.2 min, Z probed

Pattern: 5090 fits TXCBareMultiDistance at T=5 with ≤ 2 shifts.

### Cache pitfall (CRITICAL — RE-READ BEFORE TOUCHING CACHE)

The L=128 raw activation cache `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` is **LEFT-padded** (pad_token id=0 at the
start, real text at the end). Token distribution check:

- min n_real = 37 (so all sequences have ≥37 real tokens)
- Sequences shorter than 96 real tokens have padding extending past
  position 32 (i.e., the first 32 positions are pure padding for
  ~4.4% of sequences).
- The LAST 32 positions are guaranteed all-real for every sequence.

**Z previously sliced `[:, :32, :]` (FIRST 32) — that's WRONG**:
~5% of training samples land on pure-padding activations. Z deleted
that bad cache.

If you slice the L=128 cache to L=32, take **`[:, -32:, :]`** (last
32 positions). Or — equivalently — re-tokenize from scratch at CTX=32
(causal LM truncation keeps first 32 tokens of long sequences, so
no padding issue).

### Current cache state (as of pre-compact)

- `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` —
  **DELETED**. Need rebuild before training resumes.
- `token_ids.npy` and `layer_specs.json` preserved.
- `experiments/phase7_unification/results/probe_cache_S32/` — intact
  (36 task dirs, 17 GB). Built with anchor-only via Z's
  `hill_climb/build_anchor_only_probe_cache.py`. Sufficient for
  probing all hill-climb arch families (no MLC).

### Files Z added under hill_climb/ (committed in 5e1f805)

| file | purpose | status |
|---|---|---|
| `_cpu_buf_gens.py` | Pinned-CPU buffer + GPU-stream gen functions | DEPRECATED — caused ~9 sec/step pathological perf, do not reuse |
| `_run_one_subseq.py` | Single-cell SubseqH8 launcher with --ctx flag | works for fresh-process spawn pattern; combine with `--ctx 32` once cache rebuilt with last-32-positions |
| `_summarize_results.py` | Aggregate hill_ probing rows + compare vs leaderboard | works; useful for synthesis |
| `build_anchor_only_probe_cache.py` | Build anchor-only probe cache (skip MLC tail to save disk) | already used; produced the current probe_cache_S32 |
| `round1_v2_T16_cache_off_gpu.py` | V2 with cache-off-GPU experiment | DEPRECATED (V2 OOMs anyway) |
| `round2_long_shifts_verbose.py` | R2 long-shift trainer | works for L1+L2; L3+L4 OOM |
| `round2_long_shifts.py` | Original (X-style) R2 trainer | superseded by verbose variant |
| `round3_subseq_lowsample.py` | R3 single-process attempt | DEPRECATED (CUDA poisoning across cells) |

### Files Z modified + committed (in 5e1f805)

- `experiments/phase7_unification/results/training_index.jsonl` —
  fixed V1's ckpt path from `/workspace/temp_xc/...` (RunPod) to
  `/home/elysium/temp_xc/...` (local). Adds R2 L1 + L2 rows
  (`_save_run` appended these during training).
- `experiments/phase7_unification/results/probing_results.jsonl` —
  144 new rows for R2 L1 + L2 across 36 tasks × {k_feat=5, k_feat=20}.

### Recommended next moves for post-compact agent

1. **Decide bound on what 5090 can train.**
   Confirmed feasible: TXCBareMultiDistance T=5, ≤2 shifts.
   Confirmed infeasible: any SubseqH8 with T_max≥14, any H8
   multidist with T=8 + ≥3 shifts at batch=4096.

2. **Rebuild the activation cache CORRECTLY.**
   - Option A: re-run `build_act_cache_phase7.py --layer 12` (5090,
     ~5 min if undisturbed at default b=16; slower if user is using
     the machine). Produces L=128 cache.
   - Option B: modify the build script to produce L=32 directly via
     truncation at tokenization. No padding issue (long fineweb
     sequences truncate, no left-padding).
   - **NEVER `arr[:, :32, :]`** — that's the bug Z fell into.
     Use `arr[:, -32:, :]` if slicing the L=128 cache.

3. **Hill-climb directions for k=20 PAPER (the metric that matters)**

   The headline metric is **k_feat=20 mean AUC on the PAPER 16-task
   set at seed=42 base**. Bar to beat:
   - **0.9131** (`txc_bare_antidead_t5`, TXC-family leader at seed=42)
   - **0.9151** (`tsae_paper_k500`, overall leader at seed=42)

   Z's V1 sits at **0.9126** (1-seed), 0.0005 below the TXC leader.
   "Decisive" beat per Han's spec is +0.0036 (~3× σ_seeds).

   Directions, ranked by promise:

   **a. Long-distance contrastive shifts (TRIED at k=20 — HURTS).**
   Z's R2 cells were exactly this idea (single shift=32 mixed with
   short shifts at T=5):
   - L1 `hill_h8_T5_shifts1and32`: PAPER k=20 = 0.9025 (LOSE: −0.0106)
   - L2 `hill_h8_T5_shifts32only`: PAPER k=20 = 0.9062 (LOSE: −0.0069)
   Conclusion at Phase 7 k=20 (current methodology):
   long shifts (32) regress vs the standard short-shift recipe.
   **Don't pursue this further at k=20.** (Phase 5's "U-shape"
   finding was on -it at lp/mp probing — different methodology and
   different subject model. Worth a separate test at Phase 7 k=5
   on -it if anyone has cycles, but k=5 is too noisy at base for
   Z's headline.)

   **b. Large T + small `t_sample` (TRIED on 5090 — all OOM).**
   This was Han's earlier hint: SubseqH8 at T_max=20, t_sample=2
   would have the loss-path footprint of t_sample=2 while extending
   T-window range to 20. Z tried T_max=14/s5, T_max=16/s2, T_max=16/s5,
   T_max=20/s2 — all OOM on 32 GB. The encoder forward stores
   (B=4096, T_max, d_sae=18432) per shift × 4 shifts of activations
   for autograd, regardless of t_sample. **This direction is
   correct in principle, but blocked on 5090 hardware.** Defer to
   H200 — when X-H200 picks up these cells, run T_max ∈ {16, 20, 24}
   with t_sample ∈ {2, 3, 5}.

   **c. Multi-seed V1 (HIGH-VALUE, untried).**
   V1's 0.9126 is 1-seed only. The TXC-family leader is 3-seed mean.
   Run SubseqH8 T_max=12 t_sample=5 at seed=1 and seed=2 (V1
   replication). If 3-seed mean ≥ 0.9131, **V1 ties or beats the
   leader by sheer architectural fit**. ~110 min × 2 ≈ 4 hr on 5090
   (V1 fits the 5090 — Agent A trained it on H200, but T_max=12 is
   the only SubseqH8 size proven to fit 32 GB based on the earlier
   ctx-cropped test). Highest-value next move IF the 5090 cache
   rebuild succeeds.

   **d. TXCBareAntidead T-sweep filling (untried).**
   The k=20 leader is `txc_bare_antidead_t5`. Existing T values
   trained: T=5, 10, 20. Gaps: T=4, 6, 7, 8, 12. Each fits 5090
   easily (TXCBareAntidead has no matryoshka, no contrastive — just
   plain TopK + anti-dead reset). Per-cell ~25 min on 5090. If a
   non-canonical T value (e.g., T=6 or T=7) beats 0.9131 on PAPER
   k=20, that's a clean leaderboard win. Low-risk hill-climb.

   **e. SubseqH8 + T_max=12 + alternative shift sets (untried).**
   V1 used auto-shifts (1, 3, 6) for T_max=12. Try shift sets with
   the dead-zone shift removed: `(1, 6)` only (drop 3),
   `(1, 2, 6)`, `(2, 4, 6)`. Possible small gain at k=20 from
   removing the noisy mid-range shift. T_max=12 fits 5090.

   **f. Y's handoff (NOT for leaderboard).**
   `txc_bare_antidead_t5_kpos20` (k_pos=20, k_win=100) — Y's
   sparsity-decomposition case study, NOT a probe-AUC leaderboard
   contender (Y warns it may TRAIL on probe AUC). Case-study only.
   Fits 5090. Lower priority unless leaderboard hill-climb stalls.

   **DO NOT** retry SubseqH8 with T_max ≥ 14 on the 5090 — confirmed
   OOM, mark H200_required.

4. **Probe cells as you train.**
   `.venv/bin/python -m experiments.phase7_unification.run_probing_phase7 --run_ids <run_id> --S 32 --k_feat 5 20`
   Appends to `probing_results.jsonl`. Then run
   `.venv/bin/python -m experiments.phase7_unification.hill_climb._summarize_results`
   to see Z's cells vs leaderboard at PAPER set.

5. **Synthesis log goes to**:
   `docs/han/research_logs/phase7_unification/agent_z_paper/2026-04-29-z-synthesis.md`
   Document: what was tried, what worked, what OOM'd, hill-climb
   verdicts, recommendation for `paper_archs.json` additions (or
   "no winner — V1 ties at k=20, no architectural advance").

### Branch + commit state

- Branch: `han-phase7-unification`. Last pulled + pushed at
  `2026-04-29 ~22:50 UTC`.
- Z's first commit on origin: **`5e1f805` "Phase 7 Z: hill-climb on
  Gemma-2-2b base — round1+round2 + handover briefing"** — adds all
  hill_climb/*.py scripts, agent_z_paper/ writeups (blockers + V1
  baseline + this handover), the 2 R2 ckpt training_logs, and the
  R2 rows in training_index.jsonl + 144 R2 probing rows. Rebased
  cleanly onto `68ef145`. (NOTE: this handover doc was committed
  *before* the post-commit edits on this section; if you `git diff
  HEAD`, the current version of this doc may have minor textual
  improvements not yet committed.)
- HF model repo (`han1823123123/txcdr-base`) has all 3 hill ckpts +
  training logs: `hill_subseq_h8_T12_s5__seed42`,
  `hill_h8_T5_shifts1and32__seed42`,
  `hill_h8_T5_shifts32only__seed42`.

### Key references

- Brief: `agent_z_brief.md`
- Phase 7 paper-task-set doc: `agent_x_paper/2026-04-29-paper-task-set.md`
- Y handoff (sparse-TXC for case study): `agent_y_paper/2026-04-29-y-z-handoff.md`
- Z's prior writeups: `agent_z_paper/2026-04-28-z-blockers.md`,
  `agent_z_paper/2026-04-28-z-v1-baseline.md`
- Source code:
  - `experiments/phase7_unification/_train_utils.py:preload_single`
    (loads cache to GPU)
  - `experiments/phase7_unification/_paths.py` (cache + repo paths)
  - `experiments/phase7_unification/run_probing_phase7.py` (probing
    driver — uses S=32 left-aligned probe cache)
  - `experiments/phase7_unification/task_sets.py::PAPER` (16-task
    headline set)
- HF: `han1823123123/txcdr-base/ckpts/hill_*` (Z's ckpts).

### Active background processes at handover time

- All training/build processes killed.
- GPU at 1938 MiB (idle baseline).
- Disk: 408 GB free.
- Multiple stale Monitor tasks may have timed out by now; ignore.

### What Z should NOT do (lessons from this session)

- Don't restart V2 (T_max=16 SubseqH8) on 5090 — confirmed OOM.
- Don't run multiple training cells in one Python process — first
  OOM poisons CUDA context (manual_seed itself OOMs on next cell).
  Use spawn-per-cell pattern via shell wrapper.
- Don't slice the L=128 cache to L=32 with `[:, :32, :]` — left
  padding makes positions 0–31 useless for 4.4% of sequences.
- Don't lower `batch_size` below 4096 to fit a borderline arch —
  breaks paper apples-to-apples. Defer to H200 instead.
- Don't probe with the full-36 task set as the headline — paper
  uses PAPER-set (16 tasks). Filter post-hoc via `task_sets.PAPER`.
