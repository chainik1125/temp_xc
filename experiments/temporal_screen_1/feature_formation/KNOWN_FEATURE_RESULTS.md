## Question

Can we quantify the formation of a known causal backtracking feature, once
without an SAE and once through conventional SAE features?

The crucial correction is that [Ward et al.](https://arxiv.org/abs/2507.12638)
did not induce backtracking with a single SAE latent. They used a single dense
difference-of-means (DoM) residual direction. Their offset sweep changes the
token position used to *derive* the direction; it is not an injection-time
sweep.

This experiment therefore separates:

- *natural presence*: is a prespecified causal direction detectable at each
  event-relative token?
- *factorization*: is the direction carried by one conventional SAE latent or
  distributed across several active latents?
- *causal efficacy*: do norm-controlled directions derived at different
  offsets induce the eventual behavior on held-out prompts?

## Setup

- Model observed and steered:
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
- Hookpoint: post-block residual stream at layer 10.
- Natural panel: 268 genuine backtracking events, each paired with a distant
  neutral position from the same rollout.
- Primary prespecified feature: the base-Llama DoM union derived over offsets
  \([-13,-8]\). This direction transfers causally to the reasoning model and
  was not fit in the reasoning-model residual panel used here.
- Conventional SAE: single-token TopK SAE, \(d_\text{SAE}=8192\), \(k=32\).
  This pilot checkpoint is weak: centered FVU \(0.480\), with only \(29.6\%\)
  of features alive in the audit sample.
- Causal panel: the exact Stage A held-out set of 20 prompts, exact chat
  template, layer-10 hook, greedy decoding, 1,500-token cap, and Ward keyword
  rate. All new interventions use magnitude \(+12\).
- Uncertainty: 10,000 prompt-paired bootstrap resamples for causal lift over
  the reused magnitude-zero baseline.

## Natural feature formation

| Representation | Ward band mean AUC | Post-event mean AUC | Peak AUC | Half-peak onset |
|---|---:|---:|---:|---:|
| Base-derived DoM union in residual | 0.580 | 0.648 | 0.678 at \(+8\) | \(-9\) |
| SAE reconstruction projected on DoM | 0.570 | 0.644 | 0.676 at \(+16\) | \(-9\) |
| Nearest positive SAE latent, f3998 | 0.506 | 0.533 | 0.561 at \(+16\) | \(-1\) |

The SAE reconstruction projection is

\[
\hat a_{\rm SAE}(\tau)
= \sum_{j\in{\rm TopK}} z_j(\tau)
  \cos(d_j, v_{\rm Ward}),
\]

which is the Ward-direction projection of the SAE reconstruction after
dropping the constant decoder bias.

The fixed DoM direction becomes detectably outcome-associated at about token
\(-9\), but does **not** peak in the Ward window. It continues to strengthen
through and after the first backtracking token. This is compatible with a
precursor direction becoming more explicit once the behavior is underway, but
it means that a natural-presence peak does not recover the paper's offset
window.

The conventional SAE preserves almost the entire distributed direction-level
curve, despite its poor reconstruction score. It does not isolate that curve
in the nearest single latent. The causal information is in the residual and
in the SAE basis, but its *instantaneous sparse factorization is distributed*.

## Held-out causal efficacy

| Direction | Mean keyword rate | 95% CI | Paired lift over baseline, 95% CI |
|---|---:|---:|---:|
| Magnitude-zero baseline | 0.0070 | [0.0049, 0.0091] | — |
| Derived at \(-13\) | 0.0257 | [0.0188, 0.0335] | [0.0118, 0.0264] |
| Derived at \(-12\) | 0.0278 | [0.0204, 0.0354] | [0.0133, 0.0282] |
| Derived at \(-11\) | 0.0311 | [0.0209, 0.0423] | [0.0143, 0.0348] |
| Derived at \(-10\) | 0.0248 | [0.0186, 0.0318] | [0.0115, 0.0248] |
| Derived at \(-9\) | 0.0320 | [0.0246, 0.0406] | [0.0178, 0.0336] |
| Derived at \(-8\) | 0.0373 | [0.0270, 0.0486] | [0.0204, 0.0413] |
| Ward-band union | 0.0291 | [0.0226, 0.0360] | [0.0160, 0.0284] |
| Offset zero, norm matched | 0.0324 | [0.0263, 0.0393] | [0.0188, 0.0329] |
| SAE f3998 decoder, norm matched | 0.0285 | [0.0204, 0.0372] | [0.0141, 0.0299] |

Every Ward-band vector is causally active. Their rates span
\(0.025\)–\(0.037\), with heavily overlapping uncertainty. The six base
directions are also geometrically similar: mean pairwise cosine \(0.87\), and
each has cosine \(0.93\)–\(0.96\) to the union.

The raw offset-zero vector produces a keyword rate of \(0.804\), but its norm
is \(1.119\), versus \(0.414\) for the union. Once norm matched, it falls to
\(0.032\), indistinguishable at this resolution from the pre-event family.
The raw result is a saturated repetition regime, not evidence for vastly
greater causal efficacy at token zero.

The nearest single SAE decoder direction is causally active when imposed at
every generated token, despite the latent's weak natural activation contrast.
This only establishes that its decoder is an effective intervention direction;
it does not show that natural activation of f3998 mediates backtracking.

Across the six band offsets, matched-offset natural AUC correlates only weakly
with causal keyword rate (\(r=0.32\), \(n=6\)). The cheap natural contrast is
therefore useful for detecting an onset/band, not for precisely ranking highly
similar causal vectors.

## What this says about feature formation

This pilot retrodicts a *pre-outcome causal precursor*, but not a unique
formation time:

- The known direction is at chance in the far past.
- Its half-peak observational onset is \(-9\), inside the Ward window.
- All directions derived in \([-13,-8]\) causally steer the behavior.
- The direction becomes still more discriminative after the behavior begins.
- At matched norm, offset zero is no more efficacious than the pre-event
  family.

The right object is therefore not "the token at which the feature appears."
It is a temporally extended causal subspace with three distinct curves:

1. **Availability**: when the causal subspace first becomes naturally
   decodable.
2. **Explicitness**: how strongly it correlates with the behavior as the
   rollout unfolds.
3. **Efficacy**: how much a norm-controlled intervention changes the eventual
   behavior.

Those curves need not peak together.

## More precise case for a temporal crosscoder

The result does *not* support the claim that a temporal crosscoder is useful
only after attention has erased or collapsed the relevant residual
information. The residual already contains the Ward direction.

It supports a narrower and more interesting claim:

> A cross-position dictionary can help when a causal direction is coherent
> across a temporal window but is fragmented across the instantaneous sparse
> basis. Sharing a latent across positions can factorize a temporally stable
> causal subspace into a cleaner intervention unit, even when the information
> remains linearly decodable from every residual stream.

That is consistent with the existing Stage B result:
`txc_resid_L10__k16__s42_f14621_pos0` is one TXC latent whose decoder slab
causally steers backtracking. In this pilot, by contrast, the conventional SAE
needs a distributed weighted sum to reproduce the natural Ward curve. The
comparison is not yet apples-to-apples because this pilot SAE is much weaker
than the paper-budget Stage B dictionaries.

## Proposed screen

For a labelled model-organism behavior, measure:

1. A prespecified or cross-fitted causal residual direction \(v\).
2. Its event-versus-neutral availability curve \(A_v(\tau)\).
3. The same curve through the conventional SAE reconstruction.
4. The best single-latent curve \(A_j(\tau)\).
5. A **fragmentation gap**, for example
   \[
   F_{\rm frag}
   = \overline{A_{\rm SAE\ projection}(\tau)
   - \max_j A_j(\tau)}
   \]
   over a pre-outcome band.
6. Cross-offset direction stability, such as cosine or subspace overlap.
7. A small norm-matched causal validation at a few selected offsets.

A behavior is a promising temporal-crosscoder target when:

- availability rises before the outcome;
- the causal direction/subspace is stable across several tokens;
- the distributed SAE projection preserves the signal;
- no single conventional latent carries it cleanly; and
- norm-matched steering confirms causal efficacy.

This is not yet an a priori task screen: it needs labelled positive and
neutral rollouts, and event alignment when a discrete event exists. It is,
however, a concrete *architecture-selection screen* for whether a temporal
dictionary is likely to improve the factorization of a given model-organism
behavior.

## Artifacts

- Natural full result:
  `results/ward_known_feature_formation.json`
- Held-out causal generations:
  `results/ward_offset_causal_efficacy.json`
- Compact summary with bootstrap intervals:
  `results/ward_known_feature_summary.json`
- Four-panel figure:
  `results/ward_known_feature_summary.png`
- Analysis source: `analyze_known_ward.py`
- Modal sources: `modal_known_ward_feature.py`,
  `modal_ward_offset_causal.py`
- Pinned implementation commits: `706556625`, `00c2da91b`, `f4ee5184b`

The committed JSON artifacts are compact: they omit generated text, pairing
records, and unused auxiliary curves to satisfy the repository's 1 MB limit.
The complete payloads remain recoverable from
`temporal-screen-ward-weak-label-cache` at
`ward_known_feature_formation_v1.json` and
`ward_offset_causal_efficacy_v1.json`.

## Compute

- Cached natural/SAE analysis: 8 seconds on one A10G.
- Held-out causal generation: 8,917 seconds = 2.48 hours on one
  A100-40GB.
- At [Modal's current resource rates](https://modal.com/pricing), the causal
  run is approximately \$5.20 of GPU plus roughly \$2.70 of reserved CPU and
  memory, before any account credits. Total experiment spend is therefore
  approximately \$8 and comfortably below the \$50 cap.
