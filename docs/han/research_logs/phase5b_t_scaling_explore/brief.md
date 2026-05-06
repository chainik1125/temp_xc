---
author: Han
date: 2026-04-25
tags:
  - design
  - in-progress
---

## Phase 5B brief — alternative architectures for T-scaling

### Why this phase exists

The Phase 5 agent (running on RunPod, branch `han`) has the canonical
`H1 ConvTXCDR`, `H3 LogMatryoshka`, `H6 Mamba/SSM`, `H8 multi-distance`
T-sweep mission. Outcome so far (commit `472eef0`):

- vanilla TXCDR mp: monotonicity 0.33, **anti-monotone**.
- H1 ConvTXCDR mp: monotonicity 0.17, fails (sum-pool kills signal).
- H8 (current TXC champion at T=5, mp 0.8126 ± 0.003) regresses at T=10:
  `lp 0.7931 / mp 0.8040`. **No T-scaler found.**

Phase 5B is a **parallel, local-machine** push at four orthogonal
candidate axes that the Phase 5 agent has not pursued. Authored on a
single RTX 5090 (32 GB) — tighter memory budget than the A40 (48 GB)
the Phase 5 agent uses, so feasibility ceilings are lower (~T=15 for
vanilla TXCDR at d_sae=18432).

The deliberate plan is **architectural diversity**, not redundancy:
the Phase 5 agent owns "tweak H8/H9 within the existing
encoder/decoder family"; Phase 5B explores **per-position decoder
restructuring** + **subsequence sampling** (random subset of
positions) + **strided sampling** — three ideas that touch the
encoder/decoder shape rather than just the loss.

### Coordination with the Phase 5 agent

- **Branch**: `han-phase5b`, branched off `origin/han` at commit
  `472eef0`. Will rebase from `origin/han` periodically.
- **Filesystem isolation**: all training writes go to
  `experiments/phase5b_t_scaling_explore/results/ckpts/`, NOT
  `experiments/phase5_downstream_utility/results/ckpts/`. So the Phase 5
  agent's checkpoint set is untouched.
- **JSONL isolation**: Phase 5B's training/probing artefacts go to
  `phase5b_t_scaling_explore/results/{training_index,probing_results}.jsonl`.
- **GPU**: Phase 5 agent uses RunPod A40 (separate machine). Phase 5B
  uses local RTX 5090. **No shared GPU**.
- **Architecture name prefix**: all new arch classes get `phase5b_*`
  run-id prefix to avoid collision in shared registries.

### Methodology (multi-baseline gating)

Per user feedback (2026-04-25): every "trick" should be applied to
≥ 2 baselines from {TXC barebones (Track 2), TXC best (H8), token-SAE}
and evaluated on (i) the full 36-task probe AUC at both aggregations,
(ii) a T-scaling sweep where applicable. A trick is "validated" only
if it improves at least 2 of those 3 baselines on at least one of
the eval axes.

Mapping:

| trick (candidate axis) | barebones (Track 2) | best (H8) | token-SAE / topk |
|---|---|---|---|
| A: per-(pos,scale) decoder | A1 | A2 | n/a (token-SAE has no T) |
| B: subsequence sampling | B1, B2, B3 | B4 (apply to H8) | (subsumed by C) |
| C: token-enc + sparse sum | n/a (no T axis) | n/a | C1, C2 |
| D: strided window | D1, D2 | D3 (H8 + stride=2) | n/a |
| E: pair-summed input | E1 (Track 2 on paired) | E2 (H8 on paired) | E3 (topk on x0+x1) |

T-scaling axis definition for each candidate:

- **A**: vary T (window length), n_scales fixed.
- **B**: vary T_max with t_sample fixed = 5.
- **C**: not T-dependent (token-level). Vary t_sample of 128.
- **D**: vary stride at fixed T_eff=5, OR vary T_eff at fixed stride=2.
- **E**: vary k_pairs (number of paired tokens in window). T_eff = 2·k_pairs.

### The five candidate axes

#### A. Per-position feature-nested matryoshka (Track 2 + per-(t,s) head)

User's prompt: *"take the current TXC winner but apply the
feature-nested matryoshka idea, where we apply feature-nested
matryoshka to each window position SEPARATELY [check that this is
actually different from the TXC winner's matryoshka]"*

**First, the equivalence check.**

Current TXC winner H8 — `txc_bare_multidistance_contrastive_antidead`
— uses **2-scale feature-axis matryoshka**:

```
W_dec : (d_sae, T, d_in)              # single shared decoder
x_hat_full = einsum("bs,std->btd", z, W_dec)
x_hat_H    = einsum("bs,std->btd", z[:, :H], W_dec[:H, :, :])
L_recon    = MSE(x, x_hat_H) + MSE(x, x_hat_full)
```

The H prefix recon already decodes ALL T positions simultaneously
because `W_dec[:H, :, :]` keeps the position axis intact. Per-position
formulation:

```
for t in 0..T-1:
    W_dec_t : (d_sae, d_in) = W_dec[:, t, :]      # tied: position-t slice
    x_hat_full_t = z @ W_dec_t
    x_hat_H_t    = z[:, :H] @ W_dec_t[:H, :]
    l_t = MSE(x[t], x_hat_full_t) + MSE(x[t], x_hat_H_t)
L_recon = sum_t l_t / T
```

**These are mathematically identical** if the matryoshka prefix is the
same for all positions and decoder weights are shared. The squared-error
sum decomposes the same way.

**To make per-position matryoshka actually different from H8** we need
ONE of the following:

1. **Per-position prefix** — `H_per_pos[t]`: the H prefix size varies
   with t. (Heuristic: longer prefix for the "important" position(s),
   short for filler.) This breaks the global matryoshka invariant.
2. **Per-position prefix indices** — position t draws features
   `z[:, perm_t[:H]]` where each `perm_t` is a different permutation of
   `[0, d_sae)`. This is a "partition the feature axis among positions"
   variant — but features chosen for position 0 are unconstrained for
   positions 1..T-1 in their H slot. Probably indistinguishable from
   "stacked SAE with matryoshka inside each stack".
3. **Per-(position, scale) loss weights** — gamma decay both across
   scales AND across positions. Lets the model spend more matryoshka
   pressure at central / edge positions.
4. **Per-position separate decoder weights** — decouple
   `W_dec_t : (d_sae, d_in)` per position WITHOUT tying through the
   shared `W_dec`. Adds T× decoder parameters.

**Phase 5B's chosen interpretation:** option 4 + nested feature
prefixes per position (essentially H9's per-scale separate decoder
applied at the (position, scale) granularity). This is the cleanest
"actually different from H8" direction.

Concretely: `W_decs[s] : (prefix_s, T, d_in)` from H9 already gives a
separate decoder per scale. Phase 5B's variant goes further and learns
**a separate decoder per (position, scale) pair**, structured as
`W_decs_pps[s, t] : (prefix_s, d_in)`. The matryoshka loss becomes:

```
for s in 0..n_scales-1:
    for t in 0..T-1:
        x_hat_t_s = z[:, :prefix_s] @ W_decs_pps[s, t]
        l += MSE(x[:, t], x_hat_t_s)
```

This grants each (position, scale) head its own reconstruction
direction in the d_in space, avoiding the constraint that a single
W_dec slice has to serve all matryoshka scales at a given position.

**Memory / param check** at d_sae=18432, T=5, d_in=2304, n_scales=2:
- H8 decoder: d_sae·T·d_in = 18432·5·2304 = 212 M params
- This variant: Σ_s prefix_s · T · d_in = (3686+18432)·5·2304 = 255 M params
- Marginal cost (~20%); fits in 32 GB GPU with d_sae=18432.

If we use n_scales=5 nested at (3686, 7372, 11058, 14744, 18432):
- Σ prefix_s = 55292; · T · d_in = 637 M params — too big at T=20.

**Strategy**: implement at **n_scales=2** (matches H8's count) and
n_scales=3. Compare to H8 head-to-head. If wins, T-sweep at
T ∈ {5, 8, 10}.

#### B. Subsequence-sampling TXC (sample t<T positions per training step)

User's prompt: *"what if instead of summing together the latents of
ALL positions in the T window, we randomly sample t positions from the
T-window to sum together! where t is a hyperparam. imagine we can have
very long T, e.g. 20 but randomly sample t=5 of it!"*

**The mechanism.** Encoder normally does:

```
pre = sum_{t=0}^{T-1} x[t] @ W_enc[t]
```

Subsequence-sampled encoder:

```
S = random subset of [0, T) of size t_sample
pre = sum_{t in S} x[t] @ W_enc[t]
```

Decoder loss only computes on positions in S (or on all T positions —
discussed below).

**What this tests.** Three coupled hypotheses:

1. **Position-redundancy**: if the encoder learns that "any t positions
   suffice to compute the right z", the latent should encode features
   that are detectable from PARTIAL temporal context.
2. **T-scaling decoupling**: the encoder W_enc has T positions but
   activations cost t_sample. Memory and compute scale with t_sample at
   train time, not T. Lets us train T=20, T=30, T=50 effectively.
3. **Inference flexibility**: at probe time we can choose to sum over
   all T positions (full latent), or over a sub-window — either should
   give a good z under the subsampling-trained model.

**Variants**:

- **B1**: Sample t_sample contiguous subwindow positions.
  Effectively a randomly-positioned sub-window inside T. Equivalent to
  training vanilla TXCDR with shifted window placement — less novel.
- **B2**: Sample t_sample non-contiguous positions. The encoder must
  handle arbitrary subsets. **More novel — primary candidate.**
- **B3**: Sample t_sample positions per-sample independently
  (different subsets across the batch). Same as B2 but with batch-wise
  variation; tests robustness.

**Decoder loss structure**:
- Reconstruct ONLY the sampled positions: `loss = sum_{t in S} MSE(x[t], x_hat[t])`. Mirrors encoder's input set.
- Reconstruct ALL T positions: forces decoder to fill in unobserved positions from z. **Potentially much harder — rejected for v1.**

**Per-position W_enc / W_dec**: still meaningful. Position 0 means
"earliest position in the T-window"; encoder weight W_enc[0] learns
the "early-token" feature pattern. Sampling just thins which positions
contribute on a given step.

#### C. T = ∞ subsequence sampling (sample t positions from full sequence)

User's prompt: *"what if we make T infinity (i.e. we can sample from
the whole sequence). Think about this. Does it still make sense to
have a per-position encoder/decoder if we do subsequence sampling?"*

**Honest answer to the encoder question**: NO — per-position
W_enc/W_dec with T=128 (full Gemma-2-2B-IT context) is wasteful (128 ×
2304 × 18432 ≈ 5.4 GB **per matrix**, doesn't fit on the 5090 with
d_sae=18432 + Adam + activations). And conceptually, "what feature
fires at absolute position 87 of a sequence" carries little mechanism;
position-conditioning by absolute index over-fits to sequence offset.

**Two architectural options for T=∞**:

- **C1: Token-level encoder + subsampling aggregation.**
  ```
  z_t = TopK(x[t] @ W_enc + b_enc)            # (B, T, d_sae) sparse
  S = random subset of positions, size t_sample
  z = sum_{t in S} z_t                         # (B, d_sae)
  x_hat[t] = z @ W_dec    for each t in S      # decode each position separately
  ```
  Per-token encoder (no T axis), shared decoder direction. Memory =
  same as `topk_sae`. The temporal flavour comes from the SUMMING
  step over a random subset. Similar in spirit to TFA's token-level
  encoder but without attention.

- **C2: Position-encoded shared encoder.**
  ```
  pre[t] = (x[t] + pos_emb[t]) @ W_enc + b_enc   # shared W_enc
  z_t = TopK(pre[t])
  z = sum_{t in S} z_t
  ```
  Adds a sinusoidal position embedding so the encoder can distinguish
  "this token at relative offset 3" from "this token at relative offset
  10". Inspired by TFA-pos. Memory: still token-level (d_in × d_sae
  for W_enc).

Both C1 and C2 break the "per-position W_enc[t]" structure.

**What this tests**: whether long-range temporal structure carries
*any* useful sparse-probing signal when we encode tokens individually
and aggregate via a sparse sum. If C1 ≪ C2 ≪ vanilla TXCDR, the
"feature is local but probe-relevant info needs window context" story
holds. If C2 ≥ vanilla TXCDR, position-encoded token-level encoders
are a viable alternative to the per-position-weight family.

#### D. Strided window TXC (T_actual=10, stride=2 → T_eff=5)

User's prompt: *"what if instead of T=5 we have T=10 but stride 2
(skip every other position)?"*

**The mechanism.** Sliding-window architecture but the window samples
every-other-position over a span of 10 raw tokens:

```
window_positions = [t0, t0+2, t0+4, t0+6, t0+8]    # T_eff = 5
pre = sum_{i=0}^{4} x[window_positions[i]] @ W_enc[i]
```

Same per-position W_enc/W_dec shape as T=5 vanilla, but the
**effective receptive field is 10 raw tokens**. Lets the SAE see
longer-range temporal structure with the same parameter budget.

**Two variants**:

- **D1: Fixed stride 2**. Always-every-other-token; T_eff=5,
  span=10.
- **D2: Variable stride**. At train time, sample stride uniformly from
  {1, 2, 3}; at probe time use stride=1 or stride=2 depending on which
  generalises better. Tests whether stride is a "free axis".

**Memory**: identical to T=5 vanilla (5 W_enc matrices, 5 W_dec slices).
Trains comfortably on the 5090.

**What this tests**: whether the T=5 mp ceiling is set by *receptive
field* or by *parameter count*. If D1 > T=5 vanilla AND D1 > T=10
vanilla, the answer is "receptive field, not just parameters". Then
the next question is whether stride > 2 gives further gains.

#### E. Pair-summed input (token-pair fusion before SAE)

User's prompt: *"what if we take every 2 adjacent tokens and add their
residual stream representations, e.g. if we have x0 x1 x2 x3 we end up
with just (x0+x1) (x2+x3), then we train an SAE to reconstruct the
summed representations! if we use k latents to reconstruct x0+x1 is it
similar to using k / 2 latents to reconstruct just one token? is this
equivalent / similar to barebones TXC with T=2?"*

**The mechanism.** Pre-process the input: replace each pair of adjacent
tokens with their elementwise sum. Then run any SAE / TXC on the
shortened sequence:

```
x_paired[i] = x[2i] + x[2i+1]                  # pair-sum
# Then any SAE: topk_sae(x_paired), Track2(x_paired window of T_paired), H8(...)
```

**Equivalence check vs vanilla TXC T=2.**

Vanilla TXC T=2: encodes (x0, x1) → z via per-position W_enc[0],
W_enc[1] (two distinct matrices). Decoder is per-position too.
Pre-activation: `pre = x0 @ W_enc[0] + x1 @ W_enc[1]`.

Pair-summed topk_sae: encodes (x0+x1) → z via single W_enc.
Pre-activation: `pre = (x0+x1) @ W_enc = x0 @ W_enc + x1 @ W_enc`.

So pair-sum topk_sae is **TXC T=2 with W_enc[0] = W_enc[1] (weight
sharing across positions) and decoder collapsed to a single output
direction**. NOT equivalent to barebones TXC T=2 — it's a heavily
weight-tied special case.

Three implications:

1. **Receptive field doubles "for free"** in parameter count. T_eff = 2
   with topk_sae's single-token param budget.
2. **Position information lost**: the SAE cannot tell whether a feature
   came from x0 or x1. This is fine for *content/topic* features
   (sum-invariant) but kills *position-asymmetric* features (e.g.,
   transition markers, "this is the last token of a phrase").
3. **k vs k/2 question**: at the same k, the pair-sum input has ~√2 ×
   the L2 norm of a single token (assuming uncorrelated features), so
   reconstruction at the same k is **harder** than single-token by
   maybe 30%. To get equal recon quality, k might need to be 2× per
   pair, which is exactly the budget barebones TXC T=2 already uses
   (k_win = k_pos · T = 2·k_pos).

So pair-sum topk_sae at k = 2·k_pos ≈ "weight-tied barebones TXC T=2
with collapsed decoder". The interesting question is whether this
weight-tying *helps* (more parameter-efficient, maybe T-scales further)
or *hurts* (loses position info, probe AUC drops).

**Variants** (5 archs, multi-baseline):

- **E1**: barebones TXC (Track 2) on a window of pair-summed tokens.
  `T_paired ∈ {3, 5, 8, 10}` → `T_eff ∈ {6, 10, 16, 20}`. Tests whether
  Track 2's anti-dead stack carries the same gain on paired input.
- **E2**: H8 (multi-distance contrastive) on pair-summed input.
  Same T_paired sweep. Tests whether H8's matryoshka + multi-distance
  recipe survives the position-info loss.
- **E3**: vanilla topk_sae on x0+x1 (`T_paired=1`, no window axis).
  Pure baseline — does the pair-sum aggregation alone give a probing
  signal vs single-token topk_sae?

**Why pair-sum might T-scale better than vanilla TXCDR**: the per-position
parameter explosion (W_enc has T position slabs) is what eventually
OOMs vanilla TXCDR at T~16 on a 5090. Pair-sum keeps the encoder param
count at `T_paired = T_eff / 2` worth of slabs, halving memory at
fixed receptive field.

### Summary table of candidates

| ID | name | core change vs vanilla TXCDR | T-scaling memory | param relative to T=5 |
|---|---|---|---|---|
| A1 | `phase5b_pps_matryoshka_t5_n2` | per-(position, scale) decoders; n_scales=2 | param-equivalent | +20% |
| A2 | `phase5b_pps_matryoshka_t5_n3` | n_scales=3 | param-equivalent | +60% |
| B1 | `phase5b_subseq_t10_t5` | T_max=10, t_sample=5, contiguous | scales w/ t_sample | 2× W_enc/W_dec |
| B2 | `phase5b_subseq_t10_t5_nc` | T_max=10, non-contiguous sample | same as B1 | 2× |
| B3 | `phase5b_subseq_t20_t5_nc` | T_max=20, non-contiguous, t=5 | tractable | 4× |
| C1 | `phase5b_token_subseq_inf` | token-level enc + 5-of-128 sum | independent of T | 0.2× (no T axis) |
| C2 | `phase5b_token_subseq_pos_inf` | C1 + sinusoidal pos emb | same as C1 | 0.2× + emb |
| D1 | `phase5b_strided_t5_s2` | T_eff=5, stride=2, span=10 | same as T=5 | 1× |
| D2 | `phase5b_strided_t5_var` | T_eff=5, variable stride {1,2,3} | same as T=5 | 1× |
| D3 | `phase5b_h8_strided_t5_s2` | H8 + stride=2 | same as H8 T=5 | 1× |
| E1 | `phase5b_track2_paired_T<k>` | Track 2 on pair-summed; T_paired ∈ {3,5,8} | half of T_eff | varies |
| E2 | `phase5b_h8_paired_T<k>` | H8 on pair-summed; T_paired ∈ {3,5,8} | half of T_eff | varies |
| E3 | `phase5b_topk_paired` | topk_sae on x0+x1 only | minimal | ≈ topk_sae |

**Total**: 14 candidate archs. Phase 5B will train at seed=42 first,
expand to seed-variance only on winners.

### Success criteria (paper-relevant)

A Phase 5B candidate is interesting if any of these hold:

1. **Probing AUC** beats H8 (lp 0.8005 / mp 0.8126) at any T while
   maintaining memory feasibility on a single 32GB GPU.
2. **T-scaling**: probing AUC monotonically increases with T over at
   least 3 values, with monotonicity score ≥ 0.8 (Phase 5 agent's
   target).
3. **Pareto-dominates** vanilla TXCDR T=5 (both lp AND mp ≥) at any T.
4. **Negative-result with mechanistic insight**: the candidate fails
   in a way that isolates *which* aspect of the TXC encoder/decoder is
   load-bearing for sparse-probing utility. A clean negative result on
   "subsequence sampling kills probing utility because z stops being
   detectable from sub-windows" would be valuable.

### Open questions / unknowns to resolve

- Does subsequence sampling at training time hurt probing at the
  standard `last_position` aggregation (which uses the FULL
  T-window at probe time)? Hypothesis: depends on whether the
  subsampling-trained encoder's full-T evaluation is actually a
  bigger-receptive-field generalisation, or a "test-time distribution
  mismatch".
- Does stride > 1 break the matryoshka contrastive structure? The
  shift-1 InfoNCE loss in H8 assumes adjacent windows; with stride=2
  shift-1 means a different temporal relation.
- Does the per-(position, scale) decoder break the "anti-dead" stack?
  The unit-norm decoder constraint and decoder-parallel gradient
  removal both assume `W_dec : (d_sae, T, d_in)` shape. Need to
  re-derive for `W_decs : (n_scales, T, prefix_s, d_in)`.

### Out of scope

- BatchTopK variants (Phase 5.7 already showed BatchTopK regresses TXC
  probing; covered in `summary.md` BatchTopK section).
- Long-context training (sequences > 128 tokens). Existing cache is
  128-token. Re-cache deferred until a candidate strongly suggests
  longer context helps.
- Qualitative (autointerp) evaluation. Phase 6 / 6.2 / 6.3 cover
  qualitative; Phase 5B's headline metric is sparse-probing AUC at
  k=5, both aggregations, 36 tasks.
