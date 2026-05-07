---
component: c2
status: analysis
lead: agent_filler
date: 2026-05-06
tags:
  - audit
  - case-synthetic
  - effect-1-effect-2
---

## C2 vs the prior author's synthetic benchmark crisis

The prior author posted negative results on `case-synthetic` that directly
challenge our C2 paper claim. This document analyzes his methodology,
compares to ours, and flags what we need to fix in c2.md / the paper.

**Sources** (commit `case-synthetic` branch):

- [`docs/legacy/synthetic/2026-05-06_overnight/results.md`](https://anonymous.4open.science/r/temp-bench/blob/case-synthetic/docs/legacy/synthetic/2026-05-06_overnight/results.md)
  — overnight chain, 11 benches × 3-5 archs, with E9 DC/AC ablation.
- [`docs/legacy/results/3arch_3bench_summary.md`](https://anonymous.4open.science/r/temp-bench/blob/case-synthetic/docs/legacy/results/3arch_3bench_summary.md)
  — 3-arch matched-raw_k sweep on 3 generative processes; the
  cleanest summary of the Effect 1 vs Effect 2 question.

## The prior author's central claim

> **The TXC wins we observed are mostly Effect 1, not Effect 2.**
> Even the "denoising at ρ=0.9" advantage on noisy+overlap is largely
> an averaging effect — the per-token noise is independent across
> tokens, and averaging T of them reduces variance regardless of ρ.

He distinguishes:

- **Effect 1 — sample aggregation**: TXC's encoder sees T tokens per
  encode call, so its per-step variance scales as $\sigma^2/T$ for
  averaging filters. SAE's per-token pre-activation has variance
  $\sigma^2$. Lower variance → more reliable TopK → better feature
  recovery. **Works at any ρ, including ρ=0 (i.i.d. tokens).**
- **Effect 2 — temporal pattern detection**: requires ρ > 0.
  TXC could in principle exploit cross-token relationships via
  non-uniform $W_\text{enc}[t]$ weights, but the *encoder output is a
  single scalar per atom per window*, so it can only produce window-
  level summaries — "any rise occurred in this window" not "the rise
  was at position 3".

Effect 2 would defend the temporal claim of the paper. Effect 1 is a
weaker claim — TXC is just a multi-token averaging filter.

## The 3-arch 3-bench matrix

The maintainer runs **regular_sae**, **plain TXCDR-T5**, **TXC-base T=5** at
matched **window-level raw_k ∈ {1, 2, 5, 10, 20}** on 3 generative
processes:

| Bench | Setup | TXC win? | Which effect | Confirmed? |
|---|---|---|---|---|
| 1. Coupled deterministic | OR-coupling, p_B=1.0 (= our Setup A) | TXC ≈ SAE (slight TXC advantage in global-favouredness, but explained by TXC's lower per-token resolution averaging-bias) | Effect 1 weak, no Effect 2 needed | ✓ |
| 2. Noisy + overlap | OR-coupling, p_B=0.5, n_parents=5 | **TXC clearly wins** at raw_k=5 (gAUC ≈ 0.97-0.99 vs SAE 0.58) | **Effect 1** (averaging T noisy tokens reduces variance even at ρ=0) | ✓ |
| 3. Temporal-derivative v2 | h_k(t) state, recover rise_k(t) = h_k ∧ ¬h_{k-1} | **TXC FAILS**: SAE matches its info-theoretic ceiling; TXC underperforms | Effect 2 needed | ✗ |

**Verdict (the prior author's)**: The TXC wins are Effect 1 (sample aggregation;
no temporal structure exploited). When Effect 2 is actually needed
(rise detection — a per-token transition that requires multi-token
context), TXC fails because the architecture is a temporal *smoother*,
not a *differentiator*.

## Where our C2 lines up

### Setup A vs the prior author Bench 1 — same data, same conclusion

Our `toy_coupled_K10_M20_d256` IS the prior author's Bench 1 generator (the prior author's 1c3-coupled,
deterministic OR-coupling).

Cross-check at k=2 ρ=0.7 (us) vs raw_k=2 ρ=0.9 (the prior author — ρ=0.9 only
in his table):

| Arch | Our gAUC (k=2 ρ=0.7) | the prior author gAUC (raw_k=2 ρ=0.9) |
|---|---:|---:|
| `topk_sae` (regular_sae)    | 0.990 ± 0.000 | 1.00 |
| `txc_base` (TXC-base T=5)   | 0.990 ± 0.000 | 1.00 |
| `txc_pro` T=2 (≈ TXCDRv2 T=2) | 0.990 ± 0.000 | (the maintainer ran "plain TXCDR-T5", which ≈ our txc_base T=5) |

At low k, all archs hit gAUC=1.0 → no differentiation. The per-token
co-firing pattern under deterministic OR-coupling fully reveals
hidden state.

At higher k the prior author sees:

| raw_k (ρ=0.9) | regular SAE eAUC, gAUC | plain TXCDR-T5 eAUC, gAUC | TXC-base T=5 eAUC, gAUC |
|---:|---|---|---|
| 1 | 0.49, 0.87 | 0.51, 0.92 | 0.53, 0.99 |
| 5 | 0.76, 0.92 | 0.58, 1.00 | 0.58, 1.00 |
| 10 | 0.81, 0.74 | 0.61, 1.00 | 0.60, 0.99 |
| 20 | 0.77, 0.57 | 0.61, 0.96 | 0.62, 0.96 |

Pattern: SAE flips to local at high raw_k (gAUC drops), window
encoders stay global-favoured. **the prior author's interpretation**: window
encoders just have *less per-token resolution* — the single shared
scalar per atom averages across T positions, biasing toward global
mixtures. Not a genuine information advantage.

**Our Setup A reproduces this pattern qualitatively** (txc_base T=5
default eAUC at k=8 = 0.512, gAUC = 0.748; topk_sae k=8 eAUC=0.787,
gAUC=0.754). The "TXC win" is the same averaging effect.

### Setup B vs the prior author Bench 2 — different data, OUR setup is weaker

| | Our Setup B (`toy_markov_n20_d40_noisy`) | the prior author's Bench 2 (`coupled_noisy_overlap`) |
|---|---|---|
| Coupling | **None** (20 indep Markov chains) | OR-gate (K=10 → M=20, n_parents=5 = "overlap") |
| Emission noise | p_B=0.625 (γ=0.25) | p_B=0.5 |
| Hidden ground truth | underlying Markov state h_i | hidden chain h_k |
| Headline metric | denoising ratio (sl + lp) | gAUC + h_corr |

These are **different setups**. The prior author's adds **overlap** (n_parents=5
making per-token co-firing ambiguous) on top of noise. Our Setup B
just has noise on independent chains.

The prior author's TXC win on his Bench 2 (the +0.39 gAUC at raw_k=5 ρ=0.9) is
explicitly attributed to Effect 1 (variance reduction via averaging).
He notes: **"this advantage shows up at ρ=0.0 too"**, which means
the temporal structure isn't required for the win.

**Implication for our Setup B headline**: our "TXC denoises" finding
(sl_ratio > floor, lp_ratio > floor for TXC family) is the *same
phenomenon* as the prior author's Effect 1 — it's variance reduction via
averaging, not temporal pattern detection. The denoising story
qualitatively reproduces wasteland 1c-noisy, but the *interpretation*
shifts: TXC isn't using the Markov chain's temporal structure to
denoise; it's just averaging T noisy observations of the same hidden
state.

### Bench 3 (temporal-derivative) — we haven't run this; TXC fails

The prior author's Bench 3 tests rise detection (`rise_k(t) = h_k(t) ∧ ¬h_k(t-1)`).
The rise feature is **not in the activation** — recovery via
correlation between latent activations and the rise sequence.

Result: SAE matches its information-theoretic ceiling on h_corr;
**TXC underperforms across all raw_k**. The prior author's diagnosis:

> Shared-latent TXCDR is a temporal *smoother*, not a *differentiator*.
> It can't extract per-token rises from window observations *unless*
> the recon loss explicitly forces rise representation (which it
> doesn't here) AND there's enough latent capacity for K × T separable
> templates.

**This is an architectural limitation of the TXC family**, not a
training issue. The encoder produces one scalar per atom per window;
the per-token reconstruction is `x_hat_k(t) = z_k · W_dec[t, :, k]`
— a rank-1 separable structure (window scalar × per-position
template) that fundamentally can't represent per-token rises.

We have not run anything analogous to Bench 3 — our Setup B targets
denoising (which TXC does well via Effect 1), not differentiation
(which TXC fails at).

## E9 DC/AC ablation — TXC features are mostly static

The prior author's overnight bench includes a DC/AC ablation: replace each
TXC feature with `dc_only` (time-mean) or `ac_only` (zero-mean
fluctuation), measure h_corr.

| Bench | Original | DC-only | AC-only | AC drop |
|---|---:|---:|---:|---:|
| coupled_rho_sweep (txc_base) | 0.648 | 0.681 | 0.389 | **40%** |
| coupled_noisy_overlap (txc_base) | 0.490 | 0.518 | 0.251 | **49%** |
| temporal_derivative_v2 (txc_base) | 0.671 | 0.706 | 0.439 | 35% |
| GN-A (txc_base) | 0.268 | 0.274 | 0.066 | **75%** |
| GN-C (txc_base) | 0.865 | 0.903 | 0.510 | **41%** |
| **e1_pure_smoother (txc_base)** | 0.785 | 0.792 | 0.090 | **88%** |

**Reading**: TXC's hidden-state correlation is dominated by the DC
(time-constant) component. Replacing with `dc_only` matches or slightly
*improves* h_corr; replacing with `ac_only` (the time-varying part)
collapses h_corr by 35-88%.

This is a strong signal that TXC isn't using the time-varying
structure of its features to recover hidden state — it's using their
time-averaged values. **The temporal information is NOT being
exploited.**

We haven't run this ablation. Worth porting.

## What this means for our paper

1. **The "TXC exploits temporal structure" claim is largely refuted.**
   the prior author's E9 ablation directly shows TXC's hidden-state recovery
   comes from DC (time-averaged) features. Even his Effect 1 / Effect 2
   framing comes down on Effect 1 across the board.

2. **The "TXC denoises" claim (our Setup B) survives but with a
   weaker interpretation.** TXC denoises = TXC averages T noisy
   observations of an approximately-stationary hidden state. This
   works at ρ=0 too (per the maintainer). It's a property of the encoder
   pooling T tokens, not of temporal correlation in the data.

3. **The "TXC recovers hidden coupled features" claim (our Setup A)
   reproduces but with a weaker interpretation.** TXC has lower
   per-token resolution (its encoder averages across T positions),
   which biases its dictionary toward global mixtures. SAE has higher
   per-token resolution and thus prefers local features. This is a
   *resolution* advantage, not an *information* advantage — at low k,
   both archs hit gAUC=1.0.

4. **Methodology: matched per-token vs matched window-level.** We
   use matched per-token k_pos (so TXC-base T=5 at k_pos=5 has
   k_win = 25 latents per window). The prior author uses matched **window-level
   raw_k** (so TXC-base T=5 at raw_k=5 has 5 latents per window =
   1 latent per token effectively). The prior author's matched-raw_k is the
   fairer comparison — it keeps the actual encoder TopK budget equal.
   Under our convention TXC has T× more capacity per window, which
   confounds the comparison.

5. **TXC is a smoother, not a differentiator.** the prior author's Bench 3
   demonstrates an architectural blind spot — per-token transitions
   require per-position latent outputs, which the shared-latent
   architecture can't produce. The paper should not claim TXC can
   detect arbitrary temporal structure.

## Recommended actions for c2.md / paper

1. **Reframe Setup B headline**: replace "TXC denoises by exploiting
   temporal structure" with "TXC denoises via T-token sample
   aggregation; this works at any ρ, including ρ=0 (independent
   tokens)."

2. **Add the matched-raw_k comparison.** Re-render Setup A and Setup B
   tables with raw_k matching as well (or as an alternate column).
   At raw_k=5, our TXC-base T=5 default would be evaluated at k_pos=1
   (since k_win = k_pos × T = 1 × 5 = 5), not k_pos=5.

3. **Add a Setup D — temporal-derivative.** Port the prior author's Bench 3
   into our framework. This is the test where the temporal claim
   actually matters; if TXC fails, document it as an honest limitation.

4. **Run the E9 DC/AC ablation on Setup A + B.** If TXC's hidden_corr
   is DC-dominated on our setups too (likely, given we use the same
   data generators), this is the cleanest evidence that TXC isn't
   using temporal structure.

5. **Match the ρ-sweep design to the prior author's analysis.** Our in-flight
   ρ-sweep at ρ ∈ {0.0, 0.3, 0.6, 0.9} on Setup A is exactly
   the right design — it confirms or refutes Effect 2. Decision rule:
   gAUC flat across ρ → Effect 1 confirmed (consistent with the maintainer);
   gAUC growing with ρ → Effect 2 still alive (would contradict
   the prior author; would need explanation).

6. **Don't claim TXC > SAE on any synthetic bench without showing
   ρ=0 baseline.** the prior author's noisy+overlap bench gives a TXC win even
   at ρ=0; the win is averaging, not temporal.

## Questions for the prior author

- Should we run Setup D (temporal-derivative) before paper submission?
- Should we re-render Setup A + B at matched raw_k (not k_pos)?
- Should we port the E9 DC/AC ablation and run it on our Setup A+B
  checkpoints? (Eval-only on cached ckpts; ~30 min wall.)
- Does the paper's temporal-claim need to be reframed to "sample
  aggregation gives TXC a denoising advantage in noisy regimes"
  rather than "TXC exploits temporal structure"?
