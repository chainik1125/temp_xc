---
author: Dmitry
date: 2026-07-27
tags:
  - design
  - in-progress
---


> NOTE on location: per standing preference, copy this plan into
> `docs/dmitry/reviewer_responses/stacked_sae_plan.md` with frontmatter
> (author/date/tags) as step 0 of execution.

## Context

Reviewer 1 (ICML, borderline-accept) identified that the paper's claimed key
baseline — Stacked SAE, which isolates temporal aggregation from
cross-position weight sharing — has no reported results on any real-world case
study, despite App A claiming it was "used in C1, C2, and C7". Audit confirmed:
stacked results exist only for synthetic benches (paper + leaderboard), plus
two *pre-protocol* runs (phase-5 probing on gemma-2-2b-it L13; ward stage-B
backtracking pilot) whose artifacts are on HF
(`han1823123123/txcdr`, `aniketdesh/ward-stage-b-{dictionaries,cache}`).

Goal: train + evaluate Stacked SAE **matched to the headline TXC cell** in each
of the four real-world case studies (C7 backtracking, C3 sparse probing,
C6 emergent misalignment, HH-RLHF), under each experiment's locked paper
protocol, so the numbers can drop into Fig. 4 / Table 2 / the case-study
figures for the rebuttal + camera-ready. Synthetic already has stacked arms
(stretch: refresh if matching conventions changed).

## Locked-protocol targets (from paper appendix, worktree copy = arxiv branch)

| Case study | Subject/hookpoint | Headline TXC match target | Stacked config to train | Seeds |
|---|---|---|---|---|
| C7 backtracking | Llama-3.1-8B resid L10 (steer into R1-Distill-Llama-8B) | TXC-base T=5, k_pos=20, d=32,768, 300k steps, bs=1024 | T banks × d=32,768, k_pos=20 (T decision: 5 vs 6) | 42 |
| C3 probing | gemma-2-2b-it resid L13, FineWeb 24k×128 cache | TXC-base T=5, k_pos=20, d=18,432, 20k steps, L0=20 invariant | T=5 bank × d=18,432, k_pos=20 (+T=10/20 stretch) | 3 seeds |
| C6 EM | Qwen-2.5-7B-Instruct + bad-medical LoRA (hookpoint TBD from config) | TBD (T-SAE is C6 winner; TXC cell config from em experiment) | match headline TXC cell | per-paper |
| RLHF | HH-RLHF cache (TBD layer) | TBD from rlhf experiment config | match headline TXC cell | per-paper |

(TBD cells being filled in by exploration agents.)

## Deliverables

1. Table 2 + Fig 4 stacked rows (C7 detection PR-AUC/ROC-AUC at S∈{1..32}; Δgc
   inducement bar + contingency).
2. C3 sparse-probing figure line + per-task heatmap row (AUC-bar over k_feats grid).
3. C6 α-frontier + ΔAlign|coh≥70 + detection PR-AUC for stacked cell.
4. RLHF semantic/length-spurious feature table row.
5. App A text fix ("used in C1, C2, C7" sentence) + parameter-count table
   (reviewer also asked for per-arch params/inference cost).
6. Reviewer-response paragraph draft with the numbers.

## User decisions (2026-07-26)

- **Staging**: rebuttal gets C7 + C3 results now; C6/RLHF + extra seeds land for
  camera-ready.
- **Compute**: provision fresh RunPod pods (existing h100 roots are full).
- **Judge spend ceiling**: ~$200 total (Sonnet-4.6 C7 inducement + C6 stage-4).
- **C7 stacked window**: T=5, matched to headline TXC-base (detection protocol
  uses trailing 5 of the 6 offsets).

## Exploration findings (agent 2 — reviewer docs + infra)

- **Framing already exists**: `docs/dmitry/reviewer_responses/temporal_benchmark_screen.md`
  defines the rung ladder; Stacked SAE = R3, `H_txc = R4 − R3` is "the decisive
  rung". `notes.md` item 4: "They wanted stacked SAE, so we can add it".
  No deadline/owner/compute allocation exists anywhere — this plan originates them.
- **C7 pilot numbers already tabulated** on branch `aniket-ward-stage-b`
  (`docs/aniket/experiments/ward_backtracking/results_b.md`): best stacked cell
  `stacked_sae__resid_L10__k16` Sonnet-primary 0.0054 vs TXC 0.0114 (58% behind)
  — H_txc > 0 at pilot scale. Matches HF ckpt name exactly. The txcwins sprint
  flagged this as highest value-per-minute, then dropped it (rank-reframe pivot);
  recoverability never actually verified.
- **Synthetic stacked isolation already exists**: `window_length_theory.md`
  polynomial-clock table (local SAE / window-linear "Stacked" / TXC at W=1..4,
  TXC 0.923 vs stacked 0.198 at W=4) — usable for the rebuttal today.
- **C7 judge cost per paper-scale cell**: 61 questions × 25 magnitudes × 1
  rollout = 1,525 generations = 1,525 Sonnet-4.6 calls ≈ **$3/cell**
  (grade_backtracking ~$0.002/row). Extended ±24/±32 adds 244. Judges resumable;
  transcripts persisted to judge_outputs.jsonl.
- **C6 judge cost per (arch, seed) cell**: 3 finalists × 27-point α grid ×
  64 rollouts = 5,184 + 384 extended = **5,568 judged generations (Haiku 4.5)**,
  ~15–23 min/cell wall-clock (generation+judging, from stage4_frontier.json meta).
- **Bootstrap chain**: `scripts/runpod_setup.sh` → `runpod_activate.sh` →
  `runpod_verify_gpu.sh`; curl-pipe variant `runpod_venhoff_bootstrap.sh`.
  `scripts/download_models.sh` pulls the four subject models. **No fetch script
  exists for the HF artifact repos** (ward-stage-b-*, txcdr) — small new step needed.

### Pre-registered traps

1. **√T decoder-norm rescale** (app:c6-decoder-norm): applies to TXC (rows ≈ 1/√T);
   Stacked SAE rows are unit-norm → must NOT be rescaled, or the whole C6 α
   frontier is mis-dosed. Same check for C7 V0 write magnitude.
2. **Parameter-count honesty** (reviewer Q5): stacked = T independent dicts = T×
   params of a TopK SAE at same d_sae. The matched axis is window budget
   k_win = T·k_pos (what App A claims). State it before the reviewer does.
3. App A "used in C1, C2, and C7" sentence must be fixed regardless.

## Exploration findings (agent 1 — temp_bench pipelines, origin/arxiv)

- **Generic runner**: `run.py` → `src/temp_bench/core/runner.py::run_experiment`
  dispatches any arch in `configs/archs.yaml` to any of 5 evaluators. `stacked_sae`
  (T=5, k_pos=20, d=18432, consumes='sequence') and `stacked_batchtopk` (T=4,
  consumes='window') are **already registered** — but have 0 real-task rows.
- **Four blockers**:
  1. Encode-shape contract: real evaluators expect window archs to emit
     `(B, d_sae)`; stacked archs emit `(B, T, d_sae)`. `evals/em.py` already
     amax-pools 3-D codes (works out of box); `evals/probing.py` silently
     produces wrong-width features; `evals/rlhf.py` raises RuntimeError.
     Fix: new pooled adapter arch `stacked_pooled.py` subclassing
     `StackedBatchTopK`/`StackedSAE` with `encode → (B, d_sae)` (max over T),
     per docs/framework.md "never edit core" rule.
  2. `stacked_sae.consumes='sequence'` sends probing down the per-token path →
     ValueError. Use a `consumes='window'` variant.
  3. No `per_section_hparams` for em/backtracking (need `d_sae: 32768`).
  4. **`BacktrackingEval.eval` is a stub** — canonical C7 driver lives on
     `origin/final:experiments/c7_backtracking/run.py` + `analysis.py` +
     `src/temp_bench/data/nlp/ward.py` (per the stub's own message). C7 work
     should run the origin/final driver rather than port the evaluator (verify).
- **Headline TXC arms (registry-resolved)**: probing/rlhf d=18432, T=5, k_pos=20
  (k_win=100); em d=32768, k_pos=25 (registry) but actmix panel used
  20/token with per-window rescale (K_PER_TOKEN·T), no bricken, N_STEPS=25k,
  bs=1024 windows, seeds (42, 1); backtracking d=32768, k_pos=20. `txc_pro`
  deprecated — paper TXC-pro = `phase5b_subseq_h8` (probing/c7/em),
  `agentic_txc_02` (rlhf, matched at k_win=500 not L0=20).
- **Stacked sparsity note**: stacked `k_pos` is per-position natively → k_pos=20
  gives 20/token with no rescale (rescale only needed if matching
  `txc_batchtopk_post` whose k_pos is per-window).
- **Caches on HF `han1823123123/temp-bench-data`** (457 GB): probing anchor
  `act_cache/e4916bcae1881963` (gemma-IT L13), c6 Qwen `resid_post_L24`,
  c7 Llama `resid_post_L10`; link via `experiments/probing/actmix/prep_cache.py`
  after manual `hf download` mirror. RLHF eval cache rebuildable via
  `actmix_rlhf/build_cache.py` (integrity-gated); EM cohort via
  `conversion_depth/cache_em_cohort3.py`; env vars `$TEMP_BENCH_EM_COHORT_DIR`,
  `$TEMP_BENCH_HH_RLHF_DIR`, `$TEMP_BENCH_PROBE_CACHE`.
- **Branch**: current checkout has NO temp_bench — execution needs a fresh
  branch off `origin/arxiv` (never push to arxiv; pin discipline:
  `assert_pinned` requires HEAD==pin & ancestor-of-arxiv; runner refuses dirty
  trees without TEMP_BENCH_ALLOW_DIRTY=1).
- Campaign drivers to extend: `experiments/explorations/actmix_em/cells.py`,
  `actmix_rlhf/cells.py`, `experiments/probing/actmix/sweep.py`.
- Contract tests auto-cover new archs via
  `tests/test_v2_interfaces.py::parametrize(list_archs())`.

## Verified myself: origin/temp-bench C7 driver

- `origin/temp-bench:experiments/c7_backtracking/run.py` is the canonical C7
  locked-arch sweep (the arxiv stub's `origin/final` pointer is stale — branch
  was rewritten; content lives on origin/temp-bench). **DEFAULT_ARCHS includes
  `stacked_sae` as one of the 7 locked archs**, seeds (1,2,42), protocol
  2.0.0 = 41-mag extended grid.
- `experiments/c7_backtracking/results.json` (same branch) holds a complete
  7-arch Δgc sweep, `missing_archs: []`, 12,200 judge calls, 7 cells:
  `stacked_sae peak Δgc=0.328 @m=+12 (stability 18/24)` vs `txc_base 0.426
  @m=−8`, `txc_pro 0.377 @m=+12`, `tfa 0.344`, `topk 0.230`, `tsae_paper 0.246`,
  `mlc 0.164`. **TXC-base > stacked by ~30% rel** — the reviewer-facing
  comparison already exists in some generation.
- **Caveat**: peak magnitudes differ from the paper's magnitude-axis appendix
  (txc_base −8 here vs −12 in paper; tsae +16 vs +7) → different run
  generation/protocol than final Fig 4. Executor must reconcile: diff protocol
  version + training budget of these cells vs the paper cells (check
  `temp-bench-models` HF for the stacked train_key; check analysis.py +
  pr_auc_per_S.png provenance for the detection side).
- Paper's shipped C7 arms: d_sae=32768, k_pos=20 (COMPOSITION_AUDIT §5).
  Detection-side stacked (Table 2 row) may already be computable from the
  temp-bench leaderboard/checkpoints without retraining.

## Exploration findings (agent 3 — C6 EM + RLHF wiring)

- **Two pipeline generations**: paper-era originals on `origin/final:purified/`
  (full Wang steering + Haiku judge; C6 headline) vs `origin/arxiv` v2 ports
  (detection/decomposition only, zero API). RLHF paper-era lives on
  `origin/han-phase7-agent-c` phase7 case_studies.
- **C6 locked cell**: txc_base c6 = {d_sae 32768, k_pos 25} (k_win 125),
  bs=1024, 25k steps, bricken for txc only (sae_arditi plain); 2 organisms
  (7B medical L15 / 14B finance L24), seeds {42, 1}. LoRA adapters:
  `andyrdt/Qwen2.5-7B-Instruct_bad-medical`, `ModelOrganismsForEM/...finance`.
- **C6 costs per cell**: detection-only (arxiv evaluator) = 1 training run +
  encode passes, **zero API** — and `evals/em.py` already amax-pools 3-D stacked
  codes (works as-is). Full Wang steering = 14,816 generations + 14,816 Haiku
  calls ≈ **$10/cell**, 1–3 GPU-h (7B ≈ 1–1.5 h).
- **RLHF cell**: actmix convention D_SAE=18432, N_STEPS=25k, seed 42,
  matched at k_win=500 (txc k_pos=100·T); stacked equivalent k_pos=100
  per-position. Eval zero-API; optional Haiku autointerp = 20 calls ≈ $0.06.
- **Concrete code blockers + the in-repo fix**:
  - `evals/rlhf.py::aggregate_response_mean` and `evals/probing.py` both do
    `squeeze(1)` + reshape → break on (B,T,d_sae). Wang stage-1 `encode_mean`
    silently garbage-ranks; `decoder_row` AttributeError (stacked_sae has no
    top-level W_dec) or silently wrong slice (stacked_batchtopk (T,d_sae,d_in)).
  - Correct reductions already exist in
    `experiments/ward_backtracking_txc/architectures.py::arch_decoder_directions`
    / `arch_forward` (mean-over-T window code; per-pos decoder stack) — copy them.
- **Arch selection**: registry (configs/archs.yaml) + explicit panel lists in
  configs/experiments.yaml, actmix_{em,rlhf}/cells.py, analyze.py, and
  final:purified c6 run.py make_training_cfg (raises on unknown arch names).
- **Prefer `stacked_batchtopk_btkonly`** for ACTMIX arms (consumes='window',
  batch 1024, composition-harmonized); `stacked_sae` (consumes='sequence')
  needs small batch like tsae (32).
