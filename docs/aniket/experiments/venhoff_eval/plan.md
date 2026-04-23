---
author: Aniket Deshpande
date: 2026-04-18
tags:
  - proposal
  - venhoff-eval
  - temporal-crosscoder
---

## Venhoff reasoning-eval — experiment plan

**Purpose (primary, post-2026-04-20 pivot):** test whether Temporal
Crosscoder (TempXC), MLC, or SAE features produce *steering vectors*
that recover a larger fraction of the thinking-model-vs-base-model
accuracy gap on MATH500 than Venhoff's per-sentence steering does,
using Llama-3.1-8B (base) ↔ DeepSeek-R1-Distill-Llama-8B (thinking)
as the base/thinking pair. **The bar to clear is Venhoff's 3.5% Gap
Recovery on that cell** (their Table 2). Any architecture-Gap-Recovery
combo > 3.5% is a publishable positive signal.

**Purpose (secondary, kept as side-channel):** the taxonomy-quality
comparison from the original plan runs as a free byproduct since
Phase 1 (cluster + label + score) already produces the clusters the
steering phase consumes. Predictions P1′-P4′ against MLC on Haiku 4.5
judge scores are kept in Appendix A.

**Paper deadline**: NeurIPS abstract May 4 (14 days). ICML workshop May 8.

See [[integration_plan|integration_plan.md]] for the code-level
decisions. This document covers the experiment-level hypothesis,
metric, and predictions.

### Decisions locked

| date | # | decision | status |
|---|---|---|---|
| 2026-04-18 | Q1 | Path 1 (SAE) + Path 3 (TempXC) + Path MLC (MLC) | **locked** |
| 2026-04-18 | Q2 | Anchor layer 6 (SAE/TempXC/MLC training), steering layer 12 (Llama-8B base → Venhoff default) | **locked** |
| 2026-04-18 | Q3 | All 4 TempXC aggregations run; `full_window` is the headline | **locked** |
| 2026-04-18 | Q4 | Smoke 100 MATH500 problems, full 500 (the whole split) | **locked** |
| 2026-04-18 | Q5 | Haiku 4.5 judge for taxonomy side-channel; deterministic grader for primary Gap Recovery | **locked** |
| 2026-04-20 | Q6 | **Dataset: MATH500** (not MMLU-Pro, not GSM8K) — Dmitry's redirect | **locked** |
| 2026-04-20 | Q7 | **Model pair: Llama-3.1-8B base ↔ DeepSeek-R1-Distill-Llama-8B thinking** — the cell where Venhoff's method failed (3.5% Gap Recovery). | **locked** |
| 2026-04-20 | Q8 | **Phase 2/3 code approach: vendor Venhoff's `train-vectors/` + `hybrid/` scripts, drive via subprocess with our ckpts exported in their format.** Avoids porting 250k lines. | **locked** |

## 1. Hypothesis

**Primary (post-pivot).**

> Per-cluster steering vectors optimized from TempXC (or MLC) cluster
> assignments recover a larger fraction of the Llama-3.1-8B → DeepSeek-R1-
> Distill-Llama-8B accuracy gap on MATH500 than Venhoff's per-sentence
> steering vectors trained from their SAE. Their reported number is
> **3.5% Gap Recovery**; any architecture > 3.5% supports the paper.

Why we'd expect this: Venhoff's method fails on MATH500-Llama-8B
specifically because per-position steering vectors can only inject
single-token reasoning cues. If reasoning-model advantage is a
multi-position phenomenon (long backtracks, multi-sentence case
analysis), a temporal method that binds together structure across
token positions should transfer more of the signal.

**Secondary (kept as side-channel).**

> On reasoning traces, TempXC / MLC features cluster into more
> coherent reasoning categories than per-token SAE features, as
> measured by Venhoff's taxonomy-quality metrics (accuracy,
> completeness, semantic orthogonality) with Claude Haiku 4.5 as
> judge. Predictions in Appendix A.

## 2. What we're measuring

### Primary metric — Gap Recovery on MATH500

Define (exact quantity from Venhoff Table 2):

```
gap_recovery = (hybrid_accuracy - base_accuracy) / (thinking_accuracy - base_accuracy)
```

where:
- `base_accuracy` = Llama-3.1-8B base on MATH500 (Venhoff: 27.8%).
- `thinking_accuracy` = DeepSeek-R1-Distill-Llama-8B on MATH500 (Venhoff: 79.8%).
- `hybrid_accuracy` = base + per-cluster steering vectors trained from
  our SAE/TempXC/MLC cluster annotations, applied at layer 12 during
  base-model inference over MATH500.

**Venhoff's baseline to beat: 3.5%.**

All three arch-level scores (`gap_recovery_sae`, `gap_recovery_tempxc`,
`gap_recovery_mlc`) are reported as max over Venhoff's 10 × 5
hyperparameter grid:
- `coefficient ∈ {0.1, 0.2, ..., 1.0}` — steering vector scale
- `token_window ∈ {0, -1, -15, -50, -100}` — how far back in the
  generated stream we apply steering

500 MATH500 problems per cell. Grading is deterministic via
canonical-answer extraction from `\boxed{}` (no LLM judge).

### Secondary metric — taxonomy quality (side channel)

Kept from the original plan since Phase 1 clustering + labeling runs
anyway as input to steering-vector training. Same three Venhoff scores
with Haiku 4.5 judge (see Appendix A for predictions). Reported in the
supplement; not on the headline figure.

Plus our harness-native metrics:
- **Reconstruction NMSE** on reasoning activations (SAEBench-parallel)
- **Training loss curves** with plateau-early-stop
- **Ordered + shuffled pair** for each taxonomy eval cell — advantage must
  survive shuffling to count as "temporal"

## 3. Architectures compared

Per Dmitry's 4/18 simplification: `{SAE, TempXC, MLC}` only. No TFA.

- **SAE**: standard TopK per-token. Baseline. Venhoff's default slot.
- **TempXC**: `T = 5` only (single value, not swept). Trained on
  per-token activations with aggregation at annotation time (Path 3,
  preserves temporal axis) — see [[integration_plan#3. Axis-collapse decision (load-bearing)|integration_plan § 3]].
  Rationale for collapsing the T sweep: (a) SAEBench showed T=5 was
  the least-degraded of our T values, so it's the setting most likely
  to produce a positive signal at all — if T=5 nulls, T=10/20 is very
  unlikely to rescue; (b) we already have T=5 checkpoints from the
  SAEBench rerun, avoiding a full retrain; (c) 3× compute + judge
  savings, which keeps Phase 1 inside one pod-week. If T=5 shows
  signal, expanding to T ∈ {10, 20} moves to Phase 1c instead of the
  main run.
- **MLC**: `n_layers = 5` around the anchor layer. Venhoff uses layer
  6 for 8B, so MLC window is `{4, 5, 6, 7, 8}`.

## 4. Grid

Phase 1 + 2 + 3 full run (MATH500 × Llama-8B pair):

- **Phase 1** (cluster discovery — reused from original plan)
  - 3 architectures (SAE path1, TempXC-T5 path3, MLC path_mlc)
  - Pinned `n_clusters = 15` at first (Venhoff's default for their
    Llama-8B cell). Sweep 5..50 kept as a supplement run if signal.
  - Layer 6 for SAE / MLC / TempXC training.
  - **500 MATH500 problems** (full split; not 5k like MMLU-Pro).
  - All dictionary fits capped at 10k steps with plateau early-stop.

- **Phase 2** (steering-vector training — new)
  - For each architecture's trained SAE at n_clusters=15, run Venhoff's
    `optimize_steering_vectors.py` for each of 15 categories + 1 bias
    vector = 16 steering vectors per arch.
  - `steering_layer = 12` (Llama-8B; Venhoff's default, *not* our SAE
    training layer).
  - `n_training_examples = 2048, n_eval_examples = 512, max_iters = 50`.
  - **48 vectors total** (3 arches × 16 per arch).

- **Phase 3** (hybrid inference on MATH500 — new)
  - For each arch: run Venhoff's `hybrid_token.py` over the 10 × 5
    hyperparam grid (coefficients × token_windows). 50 hybrid runs
    per arch.
  - Each hybrid run: base model generates MATH500 with steering vectors
    gated per sentence-cluster assignment.
  - **150 hybrid runs total** (3 arches × 50 hyperparam cells).

**Training fit count** (what consumes GPU for Phase 1):
- SAE: 1 fit × 1 cluster_size = **1 fit** (smoke bar); 10 fits for sweep
- MLC: same
- TempXC: same
- **Headline: 3 fits** (one per arch at n_clusters=15). Each <5 min.

**Steering + hybrid compute (Phases 2 + 3):**
- Phase 2: ~15 min per vector × 48 vectors = ~12 H100-hours
- Phase 3: ~2 min per MATH500 hybrid run × 150 runs = ~5 H100-hours
  (vLLM-batched over the 500 problems per run; grading deterministic).
- **Total Phase 2 + 3: ~17 H100-hours.**

**Secondary (taxonomy-quality, side-channel):**
- 60 cells × (accuracy, completeness, orthogonality) = same as before,
  Haiku 4.5 judge, ~$15 fees. Runs in parallel with Phase 2 since they
  share no GPU.

## 5. Predictions

Pre-registered 2026-04-20 against Gap Recovery on MATH500-Llama-8B.
Venhoff's baseline on this cell is **3.5% Gap Recovery** (Table 2 of
arXiv:2510.07364). All P-labels are for the *best* Gap Recovery across
our (coefficient, token_window) grid, per architecture.

| prediction | arch-level Gap Recovery | interpretation |
|---|---|---|
| **P0 (sanity-floor)**: Our SAE reproduces Venhoff's ~3.5% on the Llama-8B cell | SAE ≈ 3.5 ± 2 pp | If we're far from this we've broken their method; stop and debug before reporting anything. |
| **P1 (null)**: No architecture > 5% Gap Recovery | all arches ≤ 5% | Temporal / layer crosscoding does not rescue the Llama-8B cell. Paper becomes a careful dual-null writeup, foregrounding the taxonomy side-channel + Han's SVD diagnostic as the mechanism story. |
| **P2 (weak)**: TempXC *or* MLC > 5% but < 15% Gap Recovery; SAE ≤ 5% | best arch ∈ (5%, 15%) | Crosscoders help but the improvement is modest. Publishable with caveats — the delta from 3.5% → ~10% is "2-3× the Venhoff number" which is a decent headline. Steering-vector Phase 2 is viable. |
| **P3 (medium)**: TempXC or MLC > 15% Gap Recovery | best arch > 15% | Strong positive signal; at this scale Phase 2 + contrastive-loss TempXC is clearly worth investment. Clean NeurIPS abstract. |
| **P4 (strong)**: TempXC > MLC on Gap Recovery AND both > 15% | TempXC − MLC > 2 pp, both > 15% | Best case — temporal axis specifically beats layer axis, motivating the "reasoning is multi-position" framing end-to-end. |

**Win criteria for the paper:**
- **NeurIPS abstract**: P2, P3, or P4 clear the bar; the delta from
  3.5% is the lede. P1 pivots to a careful dual-null writeup.
- **P0 is a go/no-go gate.** If our SAE Gap Recovery on the Llama-8B
  cell comes in wildly off from Venhoff's 3.5% (say < 1% or > 10%),
  the whole pipeline is suspect — either our ckpt export format is
  wrong or their scripts are misconfigured in our env. **Check P0
  *first* on the smoke run before launching full arch sweep.**
- **ICML workshop fallback**: P1 + clear diagnostic story (taxonomy
  differences, SVD spectrum, training curves) is publishable as a
  careful negative.

**Open with Dmitry**: Do we count a narrow win at 5-10% as "enough"
for the NeurIPS abstract, or do we want P3+ before we lead with it?
Not blocking the run — the same pipeline produces evidence either way.

### Sanity checks before declaring a result

- Gap Recovery must be positive — `hybrid_accuracy > base_accuracy`.
  If our steering makes the base model *worse*, the number is
  ill-defined (negative Gap Recovery); report as "regression" and
  investigate.
- `thinking_accuracy` we compute should come in at ~79.8% (Venhoff
  Table 2). Big deviation = we're generating on the wrong model /
  wrong dataset / wrong prompt template.
- Coefficient sweep: we expect best-gap-recovery to be at some
  *interior* coefficient (not 0.1 nor 1.0). If it's at the boundary,
  widen the grid.

## 6. Runtime expectations

Per `integration_plan § 6` (post-pivot) and `compute_estimate.md`
(2026-04-22 reconciliation against Venhoff App C.1):

- **Phase 1** (trace + activations + dictionary): ~3-4 H100-hours for
  500 MATH500 problems × 3 arches at n_clusters=15.
- **Phase 2** (steering-vector training, paper budget = `max_iters=50,
  n_training_examples=2048, optim_minibatch_size=6`): ~2 h for TempXC +
  ~2 h for MLC on 4× H100 (~15 min/vector × 16 vectors × 2 arches /
  4-wide parallelism). **SAE skipped** — reuses Venhoff's 16 shipped
  `llama-3.1-8b_{bias,idx0..14}.pt` files from
  `vendor/thinking-llms-interp/train-vectors/results/vars/optimized_vectors/`.
- **Phase 3** (hybrid inference over 10×5 grid): ~3-5 H100-hours on 4× H100.
- **Side-channel taxonomy scoring**: ~$15 in Haiku 4.5 fees, overlaps
  with Phase 2 compute (judge calls don't block GPU).

**Total ~10-12 H100-hours on 4× H100** (~half a pod day), vs our
earlier ~20 h estimate when SAE-Phase-2 was still in scope.

Smoke: 100 MATH500 problems × SAE only × P0 gate → ~1-2 H100-hours +
negligible API. If P0 clears (our SAE ≈ Venhoff's 3.5%), unlock full.

## 6b. 2026-04-22 paper-budget run log

**Launch**: `bash scripts/runpod_venhoff_paper_run.sh` (one-shot wrapper
around `MODE=hybrid` with paper-budget flags baked in).
**Pod**: 4× H100 80GB (pod id `0p5f3ic7qs7dtv-64411fec@ssh.runpod.io`,
host `53c06947125f`, different pod from 2026-04-21 smoke run).
**Branch**: `aniket` HEAD (`551bcb7` → `e657900` after vendor patches).

Fixes landed during launch (all committed to `aniket`):

1. **Byte-level BPE normalization** (`6df2ff9`) — traces saved with
   `Ġ` (U+0120) instead of spaces broke `split_into_sentences`. Added
   `_normalize_byte_level_bpe()` in `src/bench/venhoff/responses.py`,
   called by `extract_thinking_process` and at every `full_response`
   load site in `activation_collection.py`. No trace regen needed.
2. **Fast tokenizer forced** (`27774e6`) — newer transformers removed
   `encode_plus` from slow tokenizers. `AutoTokenizer.from_pretrained(...,
   use_fast=True)` and `tokenizer(text, return_offsets_mapping=True)`
   instead of `tokenizer.encode_plus(...)`. Same patch applied to
   Venhoff's vendored `utils/utils.py:243` via `vendor_patches.py`.
3. **`load_in_8bit=` kwarg dropped** (`12b6241`) — same transformers
   version rejects it on `AutoModelForCausalLM.from_pretrained` even
   when False. Patched `vendor/.../optimize_steering_vectors.py:703`
   to drop the kwarg; we run in bf16 and don't need 8-bit.
4. **SAE reuse without sidecar** (`551bcb7`) — the resume check
   required our own meta-sidecar. Added a branch that, when the
   expected vector file exists but no sidecar does (Venhoff's shipped
   vectors), writes a `source=venhoff_shipped` sidecar and skips
   training. All 16 SAE vectors now log `reuse_shipped` at launch.

Vendor patches are applied automatically by
`ensure_steering_patched(venhoff_root)` at the top of
`train_all_vectors()` in `src/bench/venhoff/steering.py` — idempotent.

**Key SteeringConfig defaults (now in `steering.py`):**
`max_iters=50, n_training_examples=2048, optim_minibatch_size=6, lr=1e-2,
seed=42`. Matches Venhoff App C.1 exactly.

**Results — Phase 0/1** (from the resume cache, previously run
2026-04-22 AM under the undercut budget):

- SAE smoke_done `avg_final_score=3.2629`
- TempXC smoke_done `avg_final_score=6.7801`
- MLC smoke_done `avg_final_score=3.0800`

These are the *taxonomy-quality side-channel* scores (Haiku-4.5 judge
on sentence classification), not Gap Recovery. TempXC's 6.78 vs
SAE 3.26 / MLC 3.08 matches the pre-pivot P4′ prediction (TempXC wins
taxonomy coherence) but doesn't itself speak to the primary Gap
Recovery claim.

**Status at time of writing**: Phase 2 TempXC running, 4 workers
pinned to GPUs 0-3, each ~15 min/vector at paper budget. ETA for
full pipeline (Phase 2 TempXC + MLC + Phase 3 hybrid gen): ~5-7 h.

## 7. Relationship to sparse-probing result

**2026-04-19 update (Han's 27-task rerun with proper convergence):**
MLC: 0.807. TXCDR-T5: 0.797. Overlapping error bars — parity.
TXCDR-T20: 0.751 (regression, confirming under-regularization per
Han's SVD-spectrum diagnostic). SAE: 0.745. Last-token logistic
regression baseline: 0.934 (well above every dictionary — consistent
with "SAEs don't recover the probe's signal cleanly at this scale").

Implications for Venhoff:
- TempXC-vs-SAE is no longer the core comparison. TempXC does beat SAE
  by ~5 pp on the new numbers, but that's not the story.
- TempXC-vs-MLC is the load-bearing question. Venhoff's clustering +
  taxonomy metric rewards structural coherence, which is the axis
  where "different features" could surface even if benchmark accuracy
  ties. If MLC and TempXC select different reasoning categories, the
  Venhoff eval exposes it.
- T=5 is confirmed as the right TempXC operating point. T=20's
  regression is a separate story (act_fn / regularization) Han is
  investigating on a different track.

## 8. Out-of-scope

- **Datasets other than MATH500** (GSM8K / AIME / MedQA / LegalBench
  etc. all supported by Venhoff's `hybrid_token.py` but deferred to
  post-NeurIPS extension).
- **Models other than Llama-3.1-8B ↔ DeepSeek-R1-Distill-Llama-8B**
  (the cell Dmitry specifically flagged; Qwen pairs deferred).
- **Steering-strategies other than `linear`** (Venhoff's
  `adaptive_linear` / `resid_lora` variants deferred — `linear` is
  their default and the baseline we're trying to beat).
- **Han's contrastive-loss TempXC variant** — orthogonal track; if it
  lands before our Phase 2 completes, re-run against it, otherwise
  the TopK baseline is what we report.
- **Auto-interp feature analysis** beyond what Venhoff's cluster-title
  prompt already produces.

## 9. Fallback plan

If P0 (our SAE reproducing Venhoff's 3.5%) fails on the smoke run:
budget 1 day to debug the ckpt-export format + their subprocess CLI.
If still broken after that day, email the authors and run the
taxonomy-quality side-channel alone.

## 10. What produces the headline figure

Primary headline: `results/venhoff_eval/plots/fig1_gap_recovery.png`

- x-axis: coefficient (0.1..1.0)
- y-axis: Gap Recovery (%), with `3.5%` drawn as a dashed horizontal
  line labeled "Venhoff baseline"
- one subplot per token_window ∈ {0, −1, −15, −50, −100}
- lines per subplot: **SAE (blue)**, **TempXC-T5 full_window (green, bold)**,
  **MLC (orange)**. Each line has 10 points (one per coefficient).
- error bars: bootstrap over the 500 MATH500 problems (±1.96 × se)
- the best-cell-per-arch is annotated directly on the plot with the
  exact Gap Recovery number, so it reads off the figure without
  cross-referencing a table.

Secondary (supplement): `fig2_taxonomy_quality.png` — the cluster
quality figure from the original plan, relegated to the supplement
since it's the side-channel now. Unchanged spec except it's marked
"Supp. Figure" not "Figure 1."

---

## Appendix A — Taxonomy-quality predictions (secondary signal)

Kept from the pre-pivot plan. Runs as a free byproduct of Phase 1
since we cluster + label for steering-vector training anyway. Not the
headline; reported in the supplement.

### Reframing context (2026-04-19 Han sparse-probing rerun)

MLC 0.807 vs TXCDR-T5 0.797 — parity on broader (27-task) sparse
probing, down from the 8-task "MLC wins decisively" number. This
reshaped the taxonomy side-channel from "is TempXC better than SAE"
to "do MLC and TempXC surface *different* reasoning categories."

### Secondary predictions

| prediction | composite score delta | interpretation |
|---|---|---|
| **P1′ (null)**: TempXC ≤ MLC on all metrics, cluster sizes, aggregations; TempXC also ≤ SAE | Δ ≤ 0 vs both | Nothing to say beyond what the primary Gap Recovery table already says. |
| **P2′ (weak)**: TempXC > SAE at small cluster sizes, but MLC ≥ TempXC throughout | TempXC − SAE > 0 at k≤20; TempXC − MLC ≤ 0 | Temporal axis offers structure beyond per-token but MLC captures as much. Supplement note only. |
| **P3′ (medium)**: TempXC ≈ MLC on composite, but diverge on *which* clusters they surface | \|TempXC − MLC\| < 0.5 pt | Parity on coherence, divergence in content. Feature-geometry supplement becomes interesting; motivates autointerp follow-up. |
| **P4′ (strong)**: TempXC > MLC on composite at `full_window` | TempXC − MLC > +0.5 pt | Temporal axis better for reasoning taxonomies specifically. Supports the primary story if Gap Recovery also favours TempXC. |
