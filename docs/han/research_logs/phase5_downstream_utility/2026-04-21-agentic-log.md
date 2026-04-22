---
author: Han
date: 2026-04-21
tags:
  - proposal
  - in-progress
---

## Phase 5.7 agentic autoresearch log — TXC + MLC

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch):
an LLM-driven loop where each cycle is a *hypothesis → code change →
fixed-budget experiment → interpreted result*. Unlike our earlier
`run_autoresearch.sh` (which executes a human-provided list of named
candidates), this loop has Claude proposing small architectural /
training-loop changes with an explicit reason, reading each result,
and writing what it learned before the next cycle.

This log is the canonical state. Every cycle appends a section here
and commits. The "what's live" table at the top tracks the current
best config per family — new cycles try to beat it.

### Operating rules

**Frozen (never changed by a cycle):**

- Probing protocol: 36 tasks × k=5 × L1-LR × top-k-by-class-separation.
- Aggregation: `last_position_val` (val split of TRAIN, test untouched).
- Evaluation metric: mean Δ_val paired-t vs the family vanilla base
  (`matryoshka_t5` for TXC, `mlc` for MLC).
- Activation cache (FineWeb + Gemma-2-2B-IT L13, pre-cached).
- Training budget: `max_steps=25000` with plateau-stop.
- Seed: 42 (seed variance is confirmation work, out-of-scope for the loop).

**Editable by a cycle (via a new named variant + dispatcher):**

- Model architecture: decoder structure, sparsity mechanism,
  normalization, encoder/decoder tying, latent shape.
- Loss function: contrastive formulation (InfoNCE variants, hard
  negatives, anchor choice), auxiliary regularizers (L2, orth,
  consistency), loss reweighting.
- Hyperparameters: α, k, T, H:L split, temperature τ, sampling gap.
- Optimizer / schedule (only for specific hypotheses that target it).

**Kill criterion**: if 5 consecutive cycles produce Δ_val < +0.01 vs
reference without informative insights, stop the loop and escalate —
either expand the search space or conclude the current best is the
ceiling for this family.

### Cycle format

```
### Cycle NN — {short name}
**Family**: TXC | MLC
**Reference to beat**: {arch_name}, Δ_val = +X.XXXX vs vanilla base.
**Hypothesis**: 1–3 sentences. What do we expect, and why, based on
  prior cycles / Part A evidence / literature?
**Change**: what exactly got edited (new arch class? modified loss?
  hyperparam swap?). Include file paths.
**Code**: commit hash when cycle commits.
**Result**: Δ_val = +X.XXXX ± Y.YYYY (t=Z.ZZ, n=36 paired) vs reference.
**Verdict**: BEAT_REF | TIE | LOST | FAILED.
**Takeaway**: what did we learn, not just what happened.
**Next**: one concrete follow-up hypothesis given this result.
```

### Reference configs (what each cycle tries to beat)

| Family | Reference | Δ_val (vs vanilla) | Notes |
|---|---|---|---|
| TXC | `matryoshka_txcdr_contrastive_t5_alpha100` (A3 α=1.0) | **+0.0259** | Current best; A3 α=3.0 pending. If A3 α=3.0 wins, it becomes the new reference. |
| MLC | `mlc_contrastive` (α=0.1) | bench AUC 0.8025 | MLC α sweep queued but not yet summarized. Update reference once α003/α100 rows land. |

### Seed hypotheses (ranked — try first)

Each is a starting point. Real cycle hypotheses should update based on
results of the prior cycle.

**TXC family (target: beat +0.0259):**

1. **H-TXC1 — L2 reg on scale-1 prefix.** A3 α monotone-climbs because
   the scale-1 prefix is under-regularized and contrastive alone is
   absorbing that job. Add `λ · ||z_scale1||²` (λ ∈ {0.01, 0.1}) and
   hold α=1.0. Predicts: smaller contrastive weight is sufficient if
   we also regularize directly.
2. **H-TXC2 — Hard negatives from same sequence.** Current InfoNCE
   uses in-batch random negatives only. Add K=4 hard negatives from
   non-adjacent positions in the same sequence (gap ≥ 10 tokens).
   Predicts: tighter contrastive structure → stronger per-token
   scale-1 prefix.
3. **H-TXC3 — H:L split at 25:75.** Matryoshka h = d_sae//4 instead of
   d_sae//2. Forces less capacity into the scale-1 prefix so what
   remains must be highly informative. Predicts: reduces redundancy
   in scale-1, improves probing AUC *if* probing uses scale-1 features.
4. **H-TXC4 — Learned temperature τ.** InfoNCE currently uses τ=1
   implicit (no softmax temperature). Replace with learnable log-τ.
   Predicts: small win (known SimCLR-family finding) but low-novelty.
5. **H-TXC5 — Orthogonality regularizer on decoder columns.**
   L_orth = λ·||W_dec^T W_dec − I||_F (off-diagonal) encourages
   feature diversity without matryoshka's strict nesting. Predicts:
   complementary to matryoshka, may add ~0.005.
6. **H-TXC6 — Consistency loss on contrastive pairs.** Instead of
   InfoNCE (discriminative), use MSE(z_cur_scale1, stop_grad(z_prev_scale1))
   on the T-1 shared positions. Predicts: similar mechanism to
   contrastive but with a stronger shift-invariance prior.
7. **H-TXC7 — Contrastive on multiple scales, not just scale-1.**
   Sum of InfoNCE losses across scales 1, 2, 3 with decay weights
   (α_s = α · γ^(s-1), γ=0.5). Predicts: richer signal for larger
   scales that currently have no contrastive pressure.
8. **H-TXC8 — Shift-equivariance via weight sharing on scale-1
   encoder.** Force encoder W_enc to be shift-equivariant across the
   T positions contributing to scale-1 (i.e., scale-1 uses same
   encoder weights for each t). Predicts: fewer params → better
   generalization; may complement the contrastive loss.

**MLC family (target: beat the α=0.1 MLC reference):**

1. **H-MLC1 — Center-layer contrastive instead of full-layer.**
   Current MLC-contrastive uses InfoNCE on the full (L-stack, d_sae/2)
   prefix. Try contrasting only the center layer (L13) prefix.
   Predicts: cleaner signal, less noise from adjacent layers.
2. **H-MLC2 — Matryoshka across layers.** Nest MLC decoders by
   layer: the first m_1 latents reconstruct L13 alone (center), next
   m_2 reconstruct L12-L14, …, full reconstructs L11-L15. Predicts:
   feature granularity aligned with layer scope improves probing
   (which reads at L13 anchor).
3. **H-MLC3 — MLC + temporal window (T=3).** Inputs are 3 adjacent
   tokens × 5 layers stacked. Current MLC is T=1. Predicts: the
   cross-token signal that helped TXC will also help MLC.
4. **H-MLC4 — Hard negatives for MLC contrastive.** Same as H-TXC2
   but for MLC: negatives from same sequence far positions.
   Predicts: same mechanism.

### Cycles

*(Cycles append here in chronological order as they complete.)*

### Cycle 01 — scale-1 orthogonality penalty (H-TXC5)
**Family**: TXC
**Reference to beat**: `matryoshka_txcdr_contrastive_t5_alpha100`
  (A3 α=1.0), Δ_val = +0.0259 vs `matryoshka_t5`. Part B showed the A3
  α-curve plateaus at α=3.0 (+0.0238), so the contrastive-weight axis
  appears saturated.
**Hypothesis**: A3 plateaus at α=1.0 because the contrastive objective
  has extracted what it can from the scale-1 prefix, but features
  within that prefix may still be redundant / correlated. Adding an
  explicit orthogonality penalty `λ · ‖W_dec_scale1 W_dec_scale1^T − I‖²_F`
  should push scale-1 features to be more mutually independent,
  giving downstream probing a more separable basis. Prediction: modest
  positive Δ (+0.005 to +0.015) over the α=1.0 reference if
  orthogonality is the remaining bottleneck.
**Change**:
  - New file `src/architectures/matryoshka_txcdr_contrastive_orth.py` —
    `MatryoshkaTXCDRContrastiveOrth` subclass of `MatryoshkaTXCDRContrastive`,
    adds `_scale1_orth_penalty()` and `forward` override that returns
    `base_loss + λ_orth · L_orth`. Scope: scale-1 only (higher scales'
    Gram matrices are too memory-heavy; scale-1 prefix is
    `d_sae//T ≈ 3700`, so G ≈ 54 MB).
  - New dispatcher branch `agentic_txc_01` in `train_primary_archs.py`
    with α=1.0, λ_orth=0.01.
  - Probe routing in `run_probing.py` (load + encode branches).
  - `BASE_OF[agentic_txc_01]=matryoshka_t5` in `run_autoresearch.sh`.
**Code**: commit forthcoming with this log update.
**Result**: pending — launched at the next GPU-idle window.
**Verdict**: pending.
**Takeaway**: pending.
**Next**: pending (conditional on result).

---

### Appendix: how to run a cycle

Per-cycle steps (manual for now — `/loop` could automate the wait
between cycles if we want full autonomy later):

1. Read this log and the current reference row in `autoresearch_index.jsonl`.
2. Pick one hypothesis. Copy its block to a new `### Cycle NN —` section.
3. Implement the change:
   - Add a new dispatcher branch in `train_primary_archs.py` with a
     descriptive arch name like `txcdr_contrastive_t5_l2scale1_001`.
     (The `_001` suffix lets us try variants of the same idea.)
   - Add a new file in `src/architectures/` if the hypothesis needs a
     new model class.
   - Add probe routing for the new arch in `run_probing.py` (load +
     encode branches).
   - Add `BASE_OF[new_arch]=matryoshka_t5` (or appropriate base) in
     `run_autoresearch.sh`.
4. Run `bash experiments/phase5_downstream_utility/run_autoresearch.sh <arch>`.
5. Read the new row from `autoresearch_index.jsonl`.
6. Fill in the Result / Verdict / Takeaway / Next fields.
7. Commit with `git commit -am "Phase 5.7 agentic cycle NN: <name>"`.

Reference gate: each cycle's Δ is always measured *vs the family
vanilla base* (txcdr_t5 or mlc), not vs the current best — so that
small gains don't rot the reference. But interpretation of "did we
beat" uses the current best Δ as the bar.
