---
author: aniket
date: 2026-05-04
tags:
  - design
  - in-progress
---

## TXC steering audit — are we steering it right?

Companion to [[det_steer_detection]]. Audits the current C5 / C7
steering protocols against TXC's mathematics, finds that the current
protocol is **TopK-SAE-equivalent**, and lays out the V0 vs
V1 / V2 / V3 / V4 trade-off plus the implementation that
agent_back / agent_steer can drop in.

## TL;DR

- **Current steering of TXC-base / TXC-pro is mathematically equivalent
  to steering a TopK SAE.** We collapse the (T, d_in) decoder
  trajectory into a single d_in vector via `W_dec.mean(dim=1)`
  (`txc_base.py::decoder_directions`), then add that constant vector
  at every token (`backtracking.py::SteeringHook`). The temporal axis —
  the entire point of TXC — is averaged away at steering time.
- This is not a "TXC steering is broken" finding. The wasteland TXC at
  +1.574 peak Δgc was achieved with this very protocol. The audit asks
  whether we're leaving signal on the table by treating TXC as a
  per-token SAE.
- **Recommended**: add four position-aware steering protocols (V1
  cycled, V2 trailing-window, V3 latent-space, V4 encoder pre-image),
  run a small A/B against the V0 mean-decoder baseline on one strong
  feature on C7 (the +1.574 reproducibility check), then commit to
  whichever protocol produces the best Δgc as the locked C5 / C7
  protocol for TXC.
- Cross-arch norm-matching (the existing `dom_base_union` calibration)
  needs a **TXC-specific √T correction** for trajectory protocols.
  Spec below; implemented in `TXCSteeringHook`.

## What's actually happening today

### `decoder_directions()` for TXC-base

`src/temp_bench/architectures/txc_base.py`:

```python
def decoder_directions(self) -> torch.Tensor:
    """(d_sae, d_in) — average decoder direction across T positions."""
    return self.W_dec.data.mean(dim=1).clone()
```

### Steering vector + hook

`src/temp_bench/case_studies/backtracking.py::run_arch_evaluation`:

```python
mined = mine_top_features(arch, ...)            # ranks features by D+/D- selectivity
steering_feature = mined[0]                     # decoder_direction is already mean'd
raw_vec = steering_feature.decoder_direction.float()
vec = raw_vec / raw_vec.norm().clamp_min(1e-8) * ref_norm   # normalise to dom_base_union
hook = SteeringHook(vec)                        # vec is (d_in,)
handle = layer_module.register_forward_hook(hook)
```

`SteeringHook.__call__` adds `magnitudes[b] * vec` to **every token in
batch row b**, with the same d_in vector for every position.

### The C5 V7 protocol

`docs/components/c5.md`: V7 tiled-broadcast (stride-T blocks, single
uniform δ within each block) — chosen for arch-uniformity. V7 is
per-token decoder-row addition, but with a single magnitude δ across
each T-block. Within a block, the **vector** is also constant (it's
the mean-decoder direction). So V7 reduces to V0 (mean-decoder
constant steering) for TXC; the "tiled" structure only matters if the
magnitude δ varies block-to-block, which it doesn't in the current
implementation.

**Net**: at this commit, both C5 and C7 steer TXC by adding a constant
d_in vector to every continuation token, with the vector being the
T-averaged decoder column of the chosen feature.
**TopK-SAE-equivalent.**

## Why this is probably wrong (the math)

### Per-token SAE
- $z[f] = (x - b_{\mathrm{dec}}) W_{\mathrm{enc}}[:, f]$, scalar per
  token.
- Decoder column $W_{\mathrm{dec}}[f] \in \mathbb{R}^{d_{\mathrm{in}}}$
  is the dictionary direction. Under the dictionary-decomposition
  assumption $x \approx \sum_f z[f] \, W_{\mathrm{dec}}[f]$, adding
  $\alpha\, W_{\mathrm{dec}}[f]$ to $x$ is unambiguously "more of
  feature $f$".
- Steering at position $t$ uses the same vector for every $t$. Correct.

### TXC
- $z[f] = \sum_{t=0}^{T-1} (x[t] - b_{\mathrm{dec}}[t]) \,
  W_{\mathrm{enc}}[t, :, f]$, one scalar per **window**.
- Decoder atom $W_{\mathrm{dec}}[f, :, :] \in
  \mathbb{R}^{T \times d_{\mathrm{in}}}$ is a **trajectory** — a
  different d_in vector at each of the T positions.
- The dictionary assumption now reads
  $x[t] \approx \sum_f z[f] \, W_{\mathrm{dec}}[f, t, :]$ for every
  position $t$ in a window. Adding
  $\alpha\, W_{\mathrm{dec}}[f, t, :]$ to position $t$ is "more of
  feature $f$ at position $t$"; the **full feature firing requires the
  trajectory to land across all T positions** of the window.

### What constant-vector steering does to the TXC encoder

If we add a constant $\Delta x$ to every position:

$$
\Delta z[f] = \sum_{t} \Delta x \cdot W_{\mathrm{enc}}[t, :, f] =
\Delta x \cdot \sum_{t} W_{\mathrm{enc}}[t, :, f].
$$

The $\Delta x$ that maximally activates feature $f$ per unit norm is
therefore the **encoder pre-image**
$\sum_t W_{\mathrm{enc}}[t, :, f]$, **not** the mean of decoder
columns.

At init the encoder is tied:
$W_{\mathrm{enc}}[t, :, f] = W_{\mathrm{dec}}[f, t, :]$, so
$\sum_t W_{\mathrm{enc}}[t, :, f] = T \cdot
\mathrm{mean}_t(W_{\mathrm{dec}}[f, :, :])$. After training, encoder
and decoder diverge (no tied-weight constraint is enforced post-init
in `txc_base.py`), and the mean-decoder vector is no longer a scaling
of the encoder pre-image.

**Two confounded mistakes** in V0:

1. We're using the mean of decoder columns instead of the encoder
   pre-image. For "make the SAE's detector see feature $f$ more" this
   is wrong by the encoder-decoder divergence.
2. We're using a constant vector instead of the decoder trajectory.
   For "make the model behave as if feature $f$'s window-trajectory
   landed in the residual stream" this is wrong by the
   position-variance of $W_{\mathrm{dec}}[f, :, :]$.

These point in different directions empirically; the right fix depends
on which causal claim we want.

## When mean-decoder is approximately right

Mean-decoder ≈ position-specific iff the decoder trajectory is roughly
constant in $t$ — captured by the diagnostic
`temp_bench.eval.steering_hooks.position_variance`:

```python
def position_variance(W_dec: torch.Tensor) -> torch.Tensor:
    """W_dec: (d_sae, T, d_in). Returns (d_sae,) per-feature trajectory
    variance, normalised by mean magnitude. ≈ 0 → V0 mean-decoder is
    fine. ≈ 1 → V0 throws away substantial trajectory information."""
    mean_t = W_dec.mean(dim=1, keepdim=True)
    diff = (W_dec - mean_t).pow(2).sum(dim=(1, 2))
    norm = W_dec.pow(2).sum(dim=(1, 2)).clamp_min(1e-12)
    return diff / norm
```

A useful diagnostic to run **before** any of the steering A/B below:
histogram `position_variance(W_dec)` for our locked TXC-base + TXC-pro
on the C7 checkpoint. If the chosen steering feature's
position-variance is < 0.05, V0 mean-decoder is approximately
faithful and the A/B will tie. If > 0.2, V0 is throwing away
substantial trajectory information.

## Recommended steering protocols

All five sit alongside V0. They're mutually compatible — implemented
as `TXCSteeringHook(mode="v0" | "v1" | "v2" | "v4")` plus a
separate `latent_space_steer(...)` driver for V3.

### V0 — mean-decoder constant (current baseline)
- $\Delta x_t = \alpha \cdot \mathrm{mean}_{t}(W_{\mathrm{dec}}[f, :, :])$
  for all $t$.
- TopK-SAE-equivalent.

### V1 — position-cycled
- At continuation token $k$ (counting from cut), apply
  $\Delta x_k = \alpha \cdot W_{\mathrm{dec}}[f, (k \bmod T), :]$.
- Cheapest fix: drop-in replacement for V0, just track a token counter
  on the hook.
- **Caveat**: "position 0" of the window is arbitrary at cut time —
  there's a phase ambiguity about which decoder slice to start with.
  Three reasonable choices: (a) start at $t=0$; (b) start at $t=T-1$
  (most-recent-position alignment); (c) sweep all T phases and take the
  best Δgc. (c) is the cleanest experimentally and is what
  `experiments.det_steer.run_steering_ab.py` does by default
  (`--cycle_phases 0,1,2,3,4` for T=5).

### V2 — trailing-window (most TXC-faithful)
- At continuation token $k$, the most recent T tokens form the active
  window $[k-T+1, k]$. Apply
  $W_{\mathrm{dec}}[f, T-1-j, :]$ to position $k-j$ for
  $j \in [0, T)$.
- Equivalently: the past T residuals get the full trajectory (with
  the most recent position getting the last decoder slice).
- This matches the TXC training objective: the decoder is built to
  reconstruct the past T tokens given a single window-level $z$.
  Pumping $z[f]$ up by $\alpha$ in the residual stream is equivalent
  to applying the full trajectory to those past T positions.
- Implementation note: cumulative steering across past forward calls
  is OUT OF SCOPE for the stateless forward hook (would need a
  cross-call rolling buffer). For HF generation with KV-caching the
  forward batch is one token wide, so V2 effectively applies the
  trajectory to the **current** token only, with the per-step slice
  cycling back. The per-batch behaviour is documented inline in
  `TXCSteeringHook.__call__`.

### V3 — latent-space steering (most aggressive)
- At every step $k$, encode the most recent T-window through
  `arch.encode`, getting $z_{\mathrm{base}}$. Set
  $z' = z_{\mathrm{base}} + \alpha \cdot e_f$ in latent space. Decode
  $z'$ to $(T, d_{\mathrm{in}})$ via `arch.decode`. The steering
  perturbation applied to the past T positions is
  `decode(z') - decode(z_base)`.
- This uses the TXC's actual encode-decode pair rather than a heuristic
  vector. Most expensive (one TXC encode + decode per step) but the
  most honest "make TXC think feature $f$ fired more strongly"
  intervention.
- Implementation: `temp_bench.eval.steering_protocols.latent_space_steer`
  is the per-step primitive. The driver loop (capture →
  encode → perturb → decode → overwrite) lives in
  `experiments.det_steer.run_steering_ab.py` (deferred — Phase 3 of the
  ablation plan).

### V4 — encoder pre-image
- $\Delta x_t = \alpha \cdot
  \frac{\sum_t W_{\mathrm{enc}}[t, :, f]}{\|\sum_t W_{\mathrm{enc}}[t, :, f]\|_2}$
  applied uniformly at every $t$.
- This is the constant-vector that maximally activates feature $f$ per
  unit $\|\Delta x\|$. Differs from V0 by exactly the encoder-decoder
  divergence — quantified by
  `temp_bench.eval.steering_hooks.encoder_decoder_divergence(arch, fid)`:

  ```
  cos_sim:      cos(encoder_preimage, T × mean_dec)
  rel_residual: ‖encoder_preimage − T × mean_dec‖ / ‖encoder_preimage‖
  ```

  At tied init both vectors are equal (`cos_sim ≈ 1`, `rel_residual ≈ 0`).
  After training they drift; the gap is what V4 captures over V0.
- Useful as a control: "are we steering the SAE's detector or its
  dictionary basis?" If V4 ≈ V0 in Δgc, the encoder-decoder gap is
  small and the trajectory-preservation question (V1/V2/V3) is the
  live one. If V4 ≫ V0, the encoder-decoder divergence is the bigger
  issue.

## Norm-matching for cross-arch comparability

Currently `vec = raw_vec / ‖raw_vec‖_2 * ref_norm` where
`ref_norm = ‖dom_base_union‖_2`. This is correct for V0 and V4
(single d_in vector applied uniformly).

For V1 / V2 (trajectory-based protocols), the steering applies
**different vectors to different positions**. The total energy added
to the residual stream across the T-window is roughly
$T \cdot \alpha^2 \cdot \mathbb{E}_t \|W_{\mathrm{dec}}[f, t, :]\|_2^2$.
For magnitude comparability with per-token SAE steering at the same
$\alpha$, each per-position vector should be scaled to
$\mathrm{ref\_norm} / \sqrt{T}$ rather than `ref_norm`. Otherwise a
TXC at $\alpha = 1$ is injecting $\sqrt{T} \approx 2.24$× the energy
a TopK SAE would at the same $\alpha$, and the magnitude axis stops
being comparable across archs.

`TXCSteeringHook(sqrt_t_correction=True)` (default) applies this
correction. Document the convention in `c5.md` / `c7.md` AUTO-RESULTS;
a reviewer will ask. For V3, `latent_space_steer(ref_norm=...)`
applies the same per-row energy match.

## Ablation plan (small, cheap, blocking on V0 reproducibility)

**Phase 0 — V0 reproducibility check (already in c7.md)**: confirm
the locked TXC-base / TXC-pro reproduce the wasteland's +1.574 peak
Δgc within seed σ. If they don't, the audit's downstream A/B is moot
— debug V0 first.

**Phase 1 — Position-variance diagnostic.** Compute
`position_variance(W_dec)` on the locked TXC checkpoints, histogram,
mark the chosen steering feature. ~5 minutes of compute, single GPU.
Inform the A/B prior. Implementation:
`experiments/det_steer/run_c7_locked.py` already produces this
histogram per TXC arch.

**Phase 2 — A/B on one feature.** On the C7 cohort, with the same
chosen steering feature, sweep all 25 magnitudes for V0 / V1 / V2 / V4.
For V1, sweep all T cycle phases and take the best. (V3 is expensive;
defer to Phase 3.) Report:

| protocol | peak Δgc | peak mag | stability (Δgc>0 / 24) | Δ vs V0 |
|---|---:|---:|---:|---:|
| V0 mean-decoder | (existing) | (existing) | (existing) | 0 |
| V1 cycled (best phase) | … | … | … | … |
| V2 trailing | … | … | … | … |
| V4 encoder pre-image | … | … | … | … |

Per-arch (TXC-base, TXC-pro). Decision rule: if any of V1/V2/V4 beats
V0 by > seed σ on TXC-pro **AND** doesn't regress TXC-base, lock it as
the new C7 TXC steering protocol. Document the sweep in `c7.md`
AUTO-RESULTS.

**Phase 3 — V3 spike.** If V1/V2/V4 don't beat V0, try V3 on a small
subset (one feature × 5 magnitudes × 20 cohort qids). V3 is the upper
bound on what TXC can do causally; if it beats V0 substantially, the
Phase 2 trajectory protocols are an under-approximation of what's
available.

**Compute budget**: Phase 1+2 fit comfortably in agent_back's existing
C7 sweep budget (cohort + 25 magnitudes × 4 protocols × 1 feature × T
phases for V1 ≈ 8× the current single-feature run; A40 ~24 GPU-hours
including judge calls). Phase 3 is ~1 GPU-hour.

## Implementation hook

`temp_bench.eval.steering_hooks.TXCSteeringHook` (~250 lines, landed
on `det-steer`) is the unified hook: V0 / V1 / V2 / V4, √T-corrected,
per-row magnitudes via the same `magnitudes: (B,)` attribute as the
legacy `SteeringHook`. Reset between cohort qids via `hook.reset()`.

V3 is in `temp_bench.eval.steering_protocols.latent_space_steer` (the
per-step primitive); the driver loop (capture residual → encode →
perturb → decode → overwrite) lives in `run_steering_ab.py`.

A second helper to mine encoder pre-images:

```python
from temp_bench.eval.steering_hooks import encoder_preimage
v4_vec = encoder_preimage(arch, feature_id)   # (d_in,)
```

For per-token archs (TopK-SAE, MLC, T-SAE) `encoder_preimage` falls
back to `arch.decoder_directions()[f]`, which approximately equals the
encoder column at init for tied-weight per-token SAEs.

## Open questions for Han + agent_paper

1. **Generalise hook to C5?** agent_steer's V7 protocol is the same
   shape as our V0 — a constant vector applied uniformly. If the C7
   A/B picks V1 or V2, agent_steer should adopt it for C5's window
   archs. Coordination needed: pre-test V2 on TXC-pro for C5 before
   locking, since TXC-pro's subseq + multi-distance-contrastive trained
   encoder may have a steeper position-variance profile and benefit
   from V2 more than TXC-base does.
2. **Where does this hook live?** Currently
   `temp_bench.eval.steering_hooks` (chosen because it's already
   shared, neutral, agent_paper territory). agent_back's existing
   `case_studies.backtracking.SteeringHook` stays for backwards
   compatibility (per-token archs continue to use it); the new hook
   replaces it for TXC archs.
3. **TXC-pro Option A encoding**: TXC-pro's locked spec uses
   per-position W_enc slabs at training (subseq sampling) and at
   inference (full T_max). The hook is correct for this spec. If
   TXC-pro is later swapped to a permutation-invariant encoder
   variant, V1/V2/V3 collapse to V0 by construction — flag in
   `architecture.md`.

## Methodology validation

`experiments/det_steer/validate_protocols.py` numerically verifies:

- V0 ≡ V4 at tied init (cos_sim ≈ 1, rel_residual ≈ 1.9e-8 on the
  validation run — see `results/validate/summary.json`).
- After 600 SGD steps the encoder drifts: rel_residual mean ≈ 0.064.
  Quantifies the V4 vs V0 gap on a tiny TXC; the locked-checkpoint
  histogram (`run_c7_locked.py`) should show a similar shape with
  arch-dependent magnitude.
- Per-position deltas of V0 / V1 / V2 / V4: V0 + V4 are flat across
  positions; V1 cycles; V2 fills the trailing-T positions; total
  energy V0/V4 = T · ref², V1/V2 = ref² (the √T correction).
- See `results/validate/hook_modes_delta.png`.

## References

- [[det_steer_detection]] — companion detection protocol.
- [[det_steer_summary]] — methodology validation results +
  per-component integration TODO list.
- `temp_bench.eval.steering_hooks` — V0/V1/V2/V4 hook + diagnostics.
- `temp_bench.eval.steering_protocols` — V3 latent-space primitive.
- `papers/temporal_sae` — paper that introduces the V0 protocol on
  per-token SAEs.
- `papers/backtracking` — Ward et al. 2025 setup C7 inherits.
