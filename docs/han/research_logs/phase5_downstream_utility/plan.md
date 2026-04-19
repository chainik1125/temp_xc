---
author: Han
date: 2026-04-18
tags:
  - plan
  - in-progress
---

## Phase 5 pre-registration — downstream utility of temporal SAEs

This is the pre-registered methodology for Phase 5, written before any
Phase 5 checkpoint is trained. It is the first reviewable checkpoint
of the phase. Deviations mid-execution are recorded as dated addenda
at the bottom of this file; the git log is the review trail.

For the *why* and the full design-space vision, see [`brief.md`](brief.md).
This file concentrates on **what we will actually run in the 72 h
autonomous window**, given the budget constraints Han confirmed.

### North star

A single publishable claim that is **credible to an adversarial
reviewer**: whether a temporal SAE variant (any of: vanilla TXCDR,
weight-sharing ablations, Matryoshka-TXCDR, or an off-menu novel
architecture) is differentially useful for a concrete probing task —
measured against a strong baseline (attention-pooled activation
probe), not just against vanilla single-token SAEs.

### Compute envelope (binding)

A40 48 GB. Soft cap 48 h GPU time; target ≤ 36 h to leave headroom
for the write-up and rescue runs. If we are burning through the
budget on sub-phase 5.1 alone, we stop, edit this file with an
addendum, and cut. The order of cuts (first to go, last to go):

1. **First**: drop the off-menu novel architecture; keep Matryoshka.
2. **Then**: drop Protocol B entirely (headline becomes Protocol A only).
3. **Then**: drop the T=20 T-sweep point (T=5 only).
4. **Then**: drop TFA and TFA-pos baselines.
5. **Last**: drop the 3-seed headline re-run.

The replication (5.1) and the novel-architecture row (5.3) are the
contributions; we protect those before anything else.

### Subject model and layer

**gemma-2-2b-it, layer 13 anchor (MLC window L11–L15 centered on
L13).**

Rationale — this diverges from Aniket's setup (gemma-2-2b base,
L12, MLC L10–L14). Accepting the divergence because:

- `data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy` is
  already on disk (~2.5 GB), saving ~1 h cache build + 5 GB of model
  download. The budget is tight.
- Phase 4's entire story is on `gemma-2-2b-it` L13 and L25; keeping
  Phase 5 on the same subject model makes our Phase-4→Phase-5
  comparisons internal-consistent.
- The `-it` vs `base` distinction and the L12 vs L13 shift are both
  minor relative to the architectural claims we care about; we will
  replicate Aniket's *methodology*, not reproduce his *numbers* to
  the decimal.

This divergence is stated in every figure caption and the summary.

### Dataset: training corpus and probing corpora

- **Training corpus for all SAE variants**: the existing FineWeb
  cache at `data/cached_activations/gemma-2-2b-it/fineweb/` — 24 000
  sequences × 128 tokens × 2304 dims per layer. Only L13 and L25 are
  currently cached; L11, L12, L14, L15 must be added for MLC.
- **Probing corpora**: SAEBench's 8 sparse-probing datasets
  (ag_news, amazon_reviews, amazon_reviews_sentiment, bias_in_bios ×
  3, europarl, github-code). Plus one cross-token probing task in
  sub-phase 5.4 — defaulting to coreference resolution (cheapest).

### Leakage audit — complete, PASS

Ran `experiments/phase5_downstream_utility/leakage_audit.py` before
touching any training code. Results in
`results/leakage_audit.json`.

- **Corpus leakage**: 0 / 875 probe-text signatures appear as
  substrings in the FineWeb cache (7 of 8 SAEBench datasets tested;
  codeparrot/github-code's load script is deprecated on HF and was
  skipped — low overlap-probability corpus anyway). Well under the
  1 % severity threshold.
- **Split leakage**: SAEBench's `probe_training.py` applies its
  top-k-by-class-separation feature selection to the train split
  only, then masks both splits with the resulting indices. Verified
  by reading SAEBench `sparse_probing/probe_training.py` upstream.
  No split-leakage risk.

**Verdict**: proceed without retraining; no disjoint-corpus move
required. The signed-and-dated audit report lives at
`experiments/phase5_downstream_utility/results/leakage_audit.json`.

### Architectures in the sweep

All trained on `resid_L13` of `gemma-2-2b-it` unless otherwise
noted. `d_sae = 18 432` (8× expansion over 2304). Two seeds unless
noted as "headline" (three). Adam, LR = 3e-4, batch 1024 tokens,
`l1_coeff = 0`, TopK only.

Fixed sparsity anchor: **100 per-token activations**. This maps to
Aniket's Protocol A, which we run first; Protocol B is an optional
final row.

| Arch | T | Sparsity | Notes |
|---|---|---|---|
| TopKSAE | – | k = 100 | baseline, Aniket's SAE |
| MLC (layers L11–L15) | – | k = 100 | layer-axis crosscoder |
| TXCDR | 5 | k·T = 500 | vanilla, shared latent across 5 positions |
| TXCDR | 20 | k·T = 2000 | T-sweep upper point (protocol A) |
| Stacked SAE | 5 | k per pos = 100 | T independent SAEs |
| Stacked SAE | 20 | k per pos = 100 | |
| Shared per-position SAE | 5 | k per pos = 100 | one SAE applied at each of T positions — new arch |
| TFA | – | k = 100 novel, dense pred | Phase 4 config |
| TFA-pos | – | k = 100 novel, dense pred | Phase 4 config |
| **Matryoshka-TXCDR (position-nested)** | 5 | k·T = 500 | 5.3 primary |
| **Off-menu variant** (TBD: rotational decoder OR causal TXCDR) | 5 | k·T = 500 | 5.3 stretch |

Ten archs × Protocol A × 8 tasks × seed 1 = **80 checkpoints to
train + 80 probing evals + 8 attention-pooled baselines** at
minimum. Three-seed headline = TopKSAE, MLC, TXCDR T=5, +
Matryoshka-TXCDR = 4 archs × 3 seeds = 12 checkpoints. Protocol B
(if affordable): TXCDR T=5 + T=20 only, seed 1 = 2 checkpoints.

**Total checkpoints**: ≤ 10 + 8 + 2 = 20 SAE/crosscoder
checkpoints. Each ≤ 60 min on A40 (estimate; TFA may take longer).
Target 15 h of pure training compute.

### Convergence protocol (binding for 5.1)

- Train to **matched convergence**: the loss drop over the last
  1 000 logged steps is < 2 % of the mean over the preceding 1 000
  steps.
- Log loss every 200 steps.
- Max cap: **25 000 steps**. Past that we accept whatever we've got
  and log a "did-not-converge" flag in the checkpoint metadata.
- Every training run emits a JSON sidecar recording:
  `(arch, T, seed, k_pos, k_win, steps_trained,
    converged: bool, final_nmse, final_l0,
    loss_drop_last_1k_pct)`.
- Any row in the headline table that trains to cap without hitting
  the 2 %/1k plateau is flagged in the figure caption.

### Sparsity protocol — Protocol A first

**Protocol A: per-token rate matched at k_pos = 100.**

| Arch | k passed to `.create()` | k_pos | k_win at T=5 / T=20 |
|---|---|---|---|
| TopKSAE | 100 | 100 | – |
| MLC | 100 | 100 | – |
| TXCDR (via `CrosscoderSpec.create`, which silently does k·T) | 100 | 100 | 500 / 2000 |
| Stacked SAE | 100 | 100 | 500 / 2000 |
| Shared per-pos SAE | 100 | 100 | 500 / 2000 |
| TFA / TFA-pos | 100 novel | 100 novel | – |
| Matryoshka-TXCDR | 100 | 100 | 500 |

Every figure caption, table header, and JSONL record includes
`(arch, T, k_pos, k_win)` per the brief's rule.

**Protocol B (window-matched) is a stretch goal.** If budget
allows after 5.1 lands, we rerun TXCDR T=5 and T=20 with
`k_win = 500` held fixed (i.e. per-position k = 100, 25 — rescale
the `CrosscoderSpec.create` k·T multiplier manually). The
A-vs-B gap is a reporting axis, not a separate claim.

### Probing evaluation protocol

We do **not** fork SAEBench. SAEBench is installed into a sidecar
`uv`-managed env (pinned `numpy<2`, `datasets<4`, `sae_lens>=6.22`)
per Aniket's notes §7; the main `temp-xc` env is untouched. Our
adapter wraps `ArchSpec` checkpoints into SAEBench's `BaseSAE`
contract.

- **Aggregation strategy**: `last_position` only for the headline.
  Mean / max / full_window are Phase 4-style analysis, not part of
  the headline claim, and running all 4 multiplies probing cost by
  4×. If the ordering across archs is stable at `last_position`,
  adding the three other aggregations is cheap at write-up time.
- **k-values for feature selection**: `{1, 2, 5, 20}` — SAEBench
  default + 20 for matching the plan from Aniket's file. Headline
  cell uses `k = 5`.
- **Attention-pooled baseline**: implemented independently in
  `experiments/phase5_downstream_utility/attn_pooled_probe.py`
  from Eq. 2 of [`papers/are_saes_useful.md`](../../../../papers/are_saes_useful.md):
    - `score_t = X_t · q`, `a = softmax(score)`, `v_t = X_t · v`,
      probe_logit = `a · v`, where `q, v ∈ R^{d_model}`.
    - Trained end-to-end with BCE loss on the same train/test
      splits SAEBench uses (4000 / 1000). No feature selection
      (the attention is learned).
    - Reported alongside every SAE-probe row as "Δ vs attention
      pool".

### Success / failure criteria (pre-registered, binding)

- **Outcome A** (target): at least one temporal-SAE variant (any of
  TXCDR, Stacked, Matryoshka, off-menu) beats the attention-pooled
  baseline on ≥ 4 / 8 tasks by ≥ 1.5 pp AUC. This is the strong
  positive result.
- **Outcome B** (nuanced positive): MLC beats every temporal
  variant, but a temporal variant beats the attention-pooled
  baseline on the cross-token probing task (5.4). Paper reframes to
  "layer-axis crosscoding wins in general; temporal structure
  carries distinct information in cross-token regimes."
- **Outcome C** (negative): no temporal variant beats
  attention-pool anywhere, and MLC also fails to beat attention-pool
  on at least half the tasks. Paper becomes a
  `are_saes_useful.md`-style negative result, honestly reported.

A **replication check** also lands regardless of outcome: if our
convergence-corrected TXCDR T=5 still loses to MLC in head-to-head
probing, the Aniket claim stands. If it wins, that is the
publishable surprise.

### Execution order (binding)

1. **5.0 foundations** (≤ 4 h). This plan + infra gaps + multi-layer
   cache build-out. Specifically:
   a. Close infra gaps per brief's "Reference: Infrastructure audit"
      list: `l1_coeff/warmup_steps/weight_decay` into
      `CrosscoderSpec.train` + `StackedSAESpec.train`, L1 on
      `novel_codes` in `TFASpec.train`, `match_stacked_budget`
      flag on `CrosscoderSpec`, delete `relu_sae.py` if confirmed
      unused.
   b. Add MLC to `src/architectures/` (new file, new registry
      entry) — independent implementation.
   c. Extend the activation cache to include `resid_L11`,
      `resid_L12`, `resid_L14`, `resid_L15` (≈ 10 GB disk, ≈ 1 h
      forward passes on A40).
   d. Write the SAEBench-sidecar probing harness under
      `experiments/phase5_downstream_utility/probing/`: adapter,
      aggregation, runner, JSONL schema. Independent of Aniket's
      `src/bench/saebench/`.
   e. Write the attention-pooled baseline (≈ 50 LoC).
2. **5.1 replication** (≤ 20 h). Train the 10 primary archs under
   Protocol A, one seed each. Run the probing eval. Collect
   headline JSONL.
3. **5.3 novel architectures** (≤ 8 h, runs partially in parallel
   with 5.1). Matryoshka-TXCDR first, validated on Phase 3
   coupled-features toy data before spending NLP compute. Off-menu
   variant second (either *rotational decoder* — low-rank
   `exp(tA) W_base` on `W_dec` — or *causal TXCDR* — acausality
   hypothesis test).
4. **5.4 cross-token task** (≤ 6 h). Coreference resolution on a
   winogrande / coref-specific probe set. Run primary archs only.
5. **Three-seed headline re-run** (≤ 4 h). TopKSAE, MLC, TXCDR T=5,
   Matryoshka-TXCDR × 3 seeds → mean ± std for the final bar chart.
6. **5.5 writeup** (≤ 2 h). `summary.md` with honest outcome
   reporting, figures under `results/phase5/plots/`.

### What we are NOT committing to

The brief enumerates a broader menu: sharedDec/sharedEnc/tied
ablations (sub-phase 5.2), TXCDR-pos, factorized decoder,
Time×Layer, BatchTopK, dynamics-based encoder, sentence-windowed,
hybrid TFA×TXCDR. Any of those that survive after 5.1/5.3/5.4 land
move into sub-phase 5.6 (bonus). If the T=5 Matryoshka result is
exciting, sub-phase 5.2's ablation ladder becomes the natural
next-step; otherwise it's appendix material.

### Figures produced

1. **Figure 1 — Headline bar chart**. Test AUC averaged across 8
   SAEBench tasks, per arch, Protocol A, `last_position`, k = 5,
   with attention-pooled baseline as a horizontal line. Error bars
   from the 3-seed re-run (primary rows only).
2. **Figure 2 — Per-task breakdown**. Same data, one panel per
   task. Reveals which tasks MLC wins on vs. temporal-SAE variants.
3. **Figure 3 — T-sweep**. TXCDR + Matryoshka-TXCDR + Stacked SAE at
   T ∈ {5, 20} (Protocol A). Tests whether Matryoshka narrows
   TXCDR's negative-scaling slope.
4. **Figure 4 — Training curves**. Loss vs step, log-log, one line
   per arch. Diagnostic: confirms convergence.
5. **Figure 5 — Toy validation of novel archs**. Matryoshka-TXCDR
   and off-menu variant vs vanilla TXCDR on Phase 3
   coupled-features gAUC plot.
6. **Figure 6 (if 5.4 lands) — Cross-token task** bar chart.

### Repository layout for this phase

```
experiments/phase5_downstream_utility/
├── leakage_audit.py                 (done)
├── build_multilayer_cache.py        (5.0 new)
├── train_primary_archs.py           (5.1 new)
├── train_novel_archs.py             (5.3 new)
├── probing/
│   ├── adapter.py                   (SAEBench BaseSAE wrapper — independent)
│   ├── aggregation.py               (last-position / mean / max / full-window)
│   ├── attn_pooled_probe.py         (Eq. 2 baseline)
│   └── run_probing.py               (orchestrator)
├── plots/                           (output PNGs)
└── results/
    ├── leakage_audit.json           (done)
    ├── ckpts/                       (trained checkpoints — gitignored)
    ├── training_logs/               (per-run JSON sidecars)
    └── probing_results.jsonl        (headline table data)
```

New architectures go into `src/architectures/` per the project's
layer rule (`src/` never imports from `experiments/`):

```
src/architectures/
├── mlc.py                           (new, 5.0 step b)
├── matryoshka_txcdr.py              (new, 5.3)
└── <off_menu_variant>.py            (new, 5.3 stretch)
```

Each new arch registers itself in `src/architectures/__init__.py`'s
`REGISTRY` and is automatically picked up by `get_default_models`.

### Addenda

Addenda go below this line, dated, one entry per deviation from the
pre-registered plan.

### Addendum — 2026-04-19 (scope reduction for 5.1 compute budget)

**Change**: dropped TFA, TFA-pos, and SharedPerPositionSAE from the
default 5.1 sweep. Primary headline now covers 7 architectures:
TopKSAE, MLC, TXCDR T=5, TXCDR T=20, Stacked T=5, Stacked T=20,
MatryoshkaTXCDR T=5.

**Why**:
- TFA's self-attention at seq_len=128, d_sae=18 432 is ≈ 20 s per
  training step on A40 under my pipeline. 25 000 steps ≈ 5.8 days.
  Keeping TFA in 5.1 would consume the entire Phase-5 budget on a
  single architecture.
- SharedPerPositionSAE is mathematically identical to TopKSAE under
  the last_position probing aggregation we use for the headline cell
  (shared encoder + per-position data → shared dictionary + constant
  per-position bias is a no-op at read-out). Adding it to the headline
  bar adds no discriminating signal.

**How to apply**:
- TFA / TFA-pos move to sub-phase 5.6 (bonus, after 5.1–5.5 land). If
  5.6 is reached, train with a smaller d_sae (≤ 4 000) or shorter
  sequences (≤ 32 tokens) to keep wall-clock feasible.
- SharedPerPositionSAE is deferred to a targeted training-dynamics
  ablation in 5.2 (the full weight-sharing ladder) where its
  comparison to TopKSAE under matched data generators is informative.

Runnable via the train script's `--archs` override if a future
session wants to include them.

### Addendum — 2026-04-19 (probing-harness deviation)

**Change**: ditched the SAEBench sidecar plan. Instead, wrote an
independent sparse-probing runner in
`experiments/phase5_downstream_utility/probing/` that reproduces
SAEBench's sparse-probing semantics (train/test splits, top-k class-
separation feature selection from Kantamneni Eq. 1, sklearn L1
logistic regression) without depending on SAEBench at all.

**Why**:
- SAEBench pins `numpy < 2` and `datasets < 4`, incompatible with
  our main `uv` environment. A sidecar env is a lot of install pain
  on MooseFS (we already hit quota + null-byte write issues from
  the main env).
- A direct reimplementation is ~300 LoC vs ~2 days of debugging
  SAEBench + its dataset-name drift (bias_in_bios_class_set*
  subsets are not on HF; SAEBench builds them internally).
- The user's direction is to build Phase 5 independently — porting
  our own probing code is aligned with that.

**How to apply**: our `probing_results.jsonl` format is NOT byte-
identical to SAEBench's `eval_results.json` format. Any downstream
analysis should read our schema:
`{run_id, arch, task_name, dataset_key, aggregation, k_feat,
  test_auc, n_train, n_test, elapsed_s}`. The baselines (last-token
LR, attention-pooled Eq.2) are emitted with `run_id=BASELINE_*` and
`arch=baseline_*`.

**Caveat**: our 25 binary probing tasks span the SAEBench 7 dataset
families but are **not** 1:1 identical to SAEBench's tasks.
Differences:
- bias_in_bios: we select the top-15 professions by HF-test-set
  frequency and partition into 3 sets of 5; SAEBench's internal class
  sets may pick different professions.
- amazon_reviews: SAEBench has 5 categories; the HF dataset we access
  only exposes 2 category IDs in the first 20k rows, so we have 1
  task (category=0 vs category=1) on this pair.
- github-code: deprecated HF script. The substitute `bigcode/the-stack-smol`
  is gated; we have zero tasks for now. Skipped for 5.1.

Our sweep is therefore a 25-task benchmark, not the 30-ish-task
canonical SAEBench set. All architecture comparisons are
within-sweep (apples-to-apples), so the absolute AUC numbers will
differ from SAEBench-on-Gemma2B-base but the arch ordering and
attention-pool-baseline gap are the headline claim.
