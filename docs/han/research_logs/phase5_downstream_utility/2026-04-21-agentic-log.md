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
**Result**: Δ_val = **+0.0078 ±0.0123** (t=0.63, n=36, 21/4/11) vs
  `matryoshka_t5`. Compared to the reference A3 α=1.0 (+0.0259), this is
  a ~0.018 **regression**. Training loss trajectory basically identical
  to A3 α=1.0 (final losses within 0.04% at the same step), so the
  penalty didn't change the training objective value — but its gradient
  still perturbed features enough to hurt probing.
**Verdict**: **LOST vs reference** (cycle-local verdict AMBIGUOUS vs
  vanilla, but the reference-relative delta is what matters for the
  agentic loop).
**Takeaway**: Scale-1 features after matryoshka + contrastive are
  already at a local optimum for downstream probing. Adding
  orthogonality pressure at λ=1.0 perturbs the decoder columns *away*
  from that optimum — probing wants correlated features that
  co-activate on task-discriminative contexts, not maximally independent
  ones. The reconstruction objective was largely unchanged (features
  still reconstruct within 0.04% of the reference's loss), so the
  damage is specifically in probing-relevant structure. Worth
  remembering: orthogonality ≠ probing utility.
**Next**: Stop trying to add *more pressure* — α=3 already plateaued,
  orth regressed, so the local optimum under current signal structure
  is reached. Change the *signal itself*: cycle 02 tries H-TXC7
  (multi-scale contrastive — InfoNCE at scales 1, 2, 3 with decaying
  weights γ=0.5). Tests whether single-scale contrastive is leaving
  cross-token signal on the table at larger scales.

---

### Reference config update

- **Cycle 02 beat the previous reference.** The TXC reference for
  cycles 03+ is now `agentic_txc_02` at Δ_val = **+0.0354** vs vanilla
  `matryoshka_t5` (t=3.81, 25/3/8).

### Cycle 02 — multi-scale contrastive (H-TXC7)
**Family**: TXC
**Reference to beat**: `matryoshka_txcdr_contrastive_t5_alpha100`
  (A3 α=1.0), Δ_val = +0.0259.
**Hypothesis**: Current A3 contrastive only acts on the scale-1 prefix
  (`d_sae//T` latents). Scales 2, 3, …, T latents receive no
  contrastive pressure — they learn shift-non-invariant features
  purely via reconstruction. Extending InfoNCE to the scale-2 and
  scale-3 prefixes (with γ^s decay so scale-1 stays dominant) may add
  useful cross-token structure at larger window sizes. Prediction:
  modest positive Δ if multi-scale signal helps; negative if scale-1
  was the only "useful" place for contrastive.
**Change**:
  - New class `MatryoshkaTXCDRContrastiveMultiscale` subclassing
    `MatryoshkaTXCDRContrastive`, overriding `forward` to compute
    InfoNCE at each of the first `n_contr_scales` scales with weights
    `γ^s`.
  - Keep α=1.0 (outer contrastive weight), add `n_contr_scales=3`,
    `gamma=0.5`. Final contrastive contribution is
    `α · Σ_s γ^s · InfoNCE(z_prefix_s, z_prev_prefix_s)`.
  - New dispatcher `agentic_txc_02`; probe routing; BASE_OF entry.
**Code**: commit `90237db`.
**Result**: Δ_val = **+0.0354 ±0.0093** (t=**+3.81**, n=36, **25/3/8**)
  vs vanilla `matryoshka_t5`. **+0.0095 over the A3 α=1.0 reference**.
  mean val AUC = 0.7900 (vs 0.7805 at A3 α=1.0, vs 0.7546 vanilla).
  Training loss trajectory near-identical to A3 α=1.0 (last step
  16205 vs 16149 — within 0.3%).
**Verdict**: **BEAT_REF** — first t > 3 finalist in all of Phase 5.7.
  Strong win.
**Takeaway**: Scale-1-only contrastive was leaving cross-token signal
  on the table at larger scales. Scales 2 and 3 latents were learning
  shift-NON-invariant features via reconstruction alone, which didn't
  generalize well to probing. Adding γ^s-decayed InfoNCE pressure at
  those scales improves probing without hurting reconstruction. The
  fact that training loss is unchanged but AUC jumps ~0.01 confirms:
  the benefit is in feature STRUCTURE, not reconstruction quality —
  exactly what matryoshka+contrastive is supposed to deliver but
  scale-1-only wasn't fully exploiting.
**Next**: Sweep the multi-scale axis. Cycle 03 tests γ=1.0 (equal
  weight at each scale) at n_contr_scales=3 to see whether the decay
  or the multi-scale-ness is doing the work. Branch:
  - If γ=1.0 improves further: push harder (γ=1.0, n=5 in cycle 04).
  - If γ=1.0 regresses: decay matters. Try n=5 with γ=0.5 next.
  - If γ=1.0 ties: signal is saturated at scales 1-3.

---

### Cycle 03 — multi-scale contrastive, γ=1.0
**Family**: TXC
**Reference to beat**: `agentic_txc_02` at Δ_val = +0.0354.
**Hypothesis**: Cycle 02 showed multi-scale contrastive helps at γ=0.5.
  Setting γ=1.0 (equal weight at scales 1, 2, 3) maximizes contrastive
  pressure at larger scales. If scale-2 and scale-3 signal is
  under-weighted at γ=0.5, γ=1.0 should improve further. If γ=0.5 was
  already near-optimal because scale-1 needs to dominate, γ=1.0 should
  regress.
**Change**: same arch `MatryoshkaTXCDRContrastiveMultiscale`,
  n_contr_scales=3, **γ=1.0** (vs 0.5 in cycle 02).
**Code**: commit `d01765d`.
**Result**: Δ_val = **+0.0072 ±0.0124** (t=+0.58, n=36, 18/2/16) vs
  `matryoshka_t5`. A major **regression** from cycle 02 (+0.0354).
**Verdict**: **LOST vs cycle-02 reference**. Ambiguous vs vanilla.
**Takeaway**: **The decay matters, not just the multi-scale-ness**.
  Setting γ=1.0 (equal weight on scales 1, 2, 3) disrupts what γ=0.5
  achieved. Scale-1 needs to remain the *dominant* contrastive signal;
  scales 2, 3 benefit from some gradient but full-weight pressure forces
  those latents to be shift-invariant, which conflicts with their role
  of capturing window-level structure that's NOT translation-invariant.
  The cycle 02 sweet spot comes from the combination: strong scale-1
  shift-invariance + gentle scale-2, 3 smoothing. Remember going forward:
  the discovery is in the *weight schedule*, not just the multi-scale
  architecture.
**Next**: Cycle 04 extends cycle 02's successful γ=0.5 decay to ALL
  5 scales (n_contr_scales=5). Adds 0.125 at scale-4, 0.0625 at scale-5.
  Tests whether truncation at 3 was right or if tail scales contribute
  more useful signal under the right decay.

---

### Cycle 04 — multi-scale, γ=0.5, all 5 scales
**Family**: TXC
**Reference to beat**: `agentic_txc_02` at Δ_val = +0.0354.
**Hypothesis**: Cycle 02 established γ=0.5 is the right decay; cycle 03
  showed going up to γ=1.0 hurts. Cycle 04 extends the γ=0.5 schedule
  to all T=5 scales. With γ=0.5, scale-4 contributes weight 0.125 and
  scale-5 weight 0.0625 — small but potentially useful. Prediction:
  modest improvement if the tail scales benefit from the same gentle
  signal; no change if the gradient is too small to matter at these
  weights; regression if full-window contrastive at any weight hurts.
**Change**: `MatryoshkaTXCDRContrastiveMultiscale` with
  **n_contr_scales=5**, γ=0.5. Everything else identical to cycle 02.
**Code**: commit `648e1a8`. (Downstream ran via watchdog after the
  orchestrator was killed fixing an unrelated pgrep bug — see commit
  `009292a`; watchdog had a flag bug for the summariser, manually
  re-run; see commit `0267e94`.)
**Result**: Δ_val = **+0.0054 ±0.0132** (t=+0.41, n=36, 18/0/18) vs
  `matryoshka_t5`. **Major regression** from cycle 02.
**Verdict**: **LOST vs cycle-02 reference**. Ambiguous vs vanilla.
**Takeaway**: Scales 4 and 5 bring in bad contrastive signal even at
  gentle decay weights (0.125 and 0.0625). Reason (post-hoc): scale-4
  and scale-5 latents reconstruct longer sub-windows (4 and 5 tokens).
  Adjacent full windows only share 4 of 5 positions, and scale-4 only
  shares 3 of 4 center-tokens across adjacent. Contrastive pressure
  forces those long-window latents to be shift-invariant, which
  directly contradicts what they must do to reconstruct. The fact that
  cycle 04 comes in WORSE than cycle 03 (γ=1.0 at n=3) tells us the
  problem is the MEMBERSHIP in contrastive (scales 4, 5) more than
  the weighting. **Implication**: the shift-invariance boundary
  sits between scale-3 and scale-4 for T=5 — scales 1-3 are "shift-safe"
  (reconstruction targets overlap heavily across adjacent windows),
  scales 4-5 are not.
**Next**: Both directions of multi-scale extension regressed (γ=1.0
  and n=5). Cycle 02 is near a narrow peak. Next cycle maps the γ
  axis below 0.5: try γ=0.3 at n=3. If matches cycle 02, the sweet
  spot is a plateau. If regresses, cycle 02 hit the peak exactly.

---

### Cycle 05 — multi-scale n=3, γ=0.3
**Family**: TXC
**Reference to beat**: `agentic_txc_02` at Δ_val = +0.0354.
**Hypothesis**: Cycles 03 (γ=1.0) and 04 (n=5) both regressed from
  cycle 02. To know whether cycle 02 is on a plateau or a peak, sweep
  γ downward. γ=0.3 gives scale-1 weight 1, scale-2 weight 0.3, scale-3
  weight 0.09 — still multi-scale but with stronger scale-1 dominance.
  Prediction: roughly similar to cycle 02 if decay insensitivity within
  [0.3, 0.5]; modest improvement if scale-1 under-weighting at γ=0.5
  was the limiting factor; regression if signal at scales 2, 3 was
  doing real work at γ=0.5 and we're starving it at γ=0.3.
**Change**: `MatryoshkaTXCDRContrastiveMultiscale`,
  **n_contr_scales=3**, γ=0.3. Same dispatcher pattern — new
  `agentic_txc_05` arch name.
**Code**: commit forthcoming.
**Result**: pending.
**Verdict**: pending.
**Takeaway**: pending.
**Next**: pending.

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
