---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Y — Lever B: multi-feature steering

> **Headline**: K=3 multi-feature steering on T=2 H8 sd=42 lifts peak
> success at coh ≥ 1.5 by **+0.07** (1.53 → 1.60) but **regresses
> sharply** at coh ≥ 1.75 (1.53 → 0.50). Net: amplifies the steering
> signal at the cost of coherence in the strict regime — a worse
> trade than uniform PP. Lever B does NOT push past T-SAE on
> unconstrained peak (1.60 vs T-SAE 1.80). **Not a paradigm shift.**

### Method

`intervene_paper_clamp_window_perposition.py` now accepts
`--top-k-features K`. For K > 1, the hook clamps the top-K features
per concept (from `feature_selection.json::top_5`) simultaneously at
the same strength. Since the decoder is approximately atom-orthogonal
in feature space, clamping K features writes K * |W_dec[picked]| of
steering magnitude — a K-fold amplification.

Output routed to `steering_paper_window_perposition_topk<K>`.

### T=2 H8 sd=42 results (single seed)

| protocol | peak_unc | peak ≥ 1.5 | peak ≥ 1.75 | peak ≥ 2.0 | AUC(1.5–3.0) |
|---|---:|---:|---:|---:|---:|
| K=1 uniform PP (baseline) | 1.57 | 1.53 | **1.53** | **1.53** | **0.707** |
| **K=3 multi-feature PP** | **1.60** | **1.60** | 0.50 | 0.50 | 0.610 |
| T-SAE k=20 (anchor) | 1.80 | 1.10 | 0.37 | 0.27 | 0.508 |

K=3 wins peak ≥ 1.5 and unconstrained but loses everywhere coherence
matters more.

### K=3 per-strength curves vs K=1 (T=2 H8 sd=42)

| s_norm | K=1 succ / coh | K=3 succ / coh |
|---:|---:|---:|
| 0.5 | 0.233 / 2.733 | 0.500 / 2.267 |
| 1.0 | 0.267 / 2.533 | 0.500 / 1.800 |
| 2.0 | 0.467 / 2.100 | 0.900 / 1.533 |
| **5.0** | **1.533** / **2.200** | **1.600** / **1.567** |
| 10.0 | 1.567 / 1.367 | 1.000 / 0.967 |
| 20.0 | 0.867 / 1.000 | 0.433 / 0.900 |
| 50.0 | 0.100 / 0.900 | 0.333 / 0.733 |

The K=3 curve at the peak strength (s=5) has succ=1.6 / coh=1.567 —
slightly more success but coherence drops below the 1.75 threshold
that K=1 cleared (coh=2.20).

### Why K=3 doesn't push past T-SAE on unconstrained peak

T-SAE's unconstrained peak (1.80) lives at coh=1.40 — at that point
the text is already incoherent with high success. To match T-SAE on
unconstrained peak, we'd need to similarly saturate the decoder.

Looking at K=3 sample text at high strength:
- s=20: "wounds wounds wounds wounds wounds..." (concept-saturated single-token repetition)
- s=10: "a man in a dog and a man in a dog..." (concept fragments + degeneration)

The model degenerates into single-token repetition at K=3 high
strength. The grader gives high success (the concept word IS in the
text) but very low coherence (no English structure).

This is the **same mode** T-SAE collapses into. Both architectures
trade coherence for high-success-via-saturation at high strength.

### Verdict

Lever B doesn't shift the headline. The GIGABRAIN multi-coh-threshold
reframe stands:

- **Coh ≥ 1.5** (prereg): K=3 lifts to 1.60 (Δ=+0.50 vs anchor sd=42),
  but 3-seed verification needed; could regress at multi-seed.
- **Coh ≥ 1.75** (best metric): K=3 *loses* — peak drops from 1.53
  (K=1) to 0.50 (K=3) because the K=3 peak strength has coh=1.567
  which barely clears 1.5 but fails 1.75.
- **Unconstrained**: K=3 = 1.60 vs T-SAE 1.80 — still lose by 0.20.

K=2 and K=5 results in flight; expectation is that K=2 lies between
K=1 and K=3 (smaller lift, smaller regression), and K=5 lies past
K=3 (larger lift at coh ≥ 1.5 but bigger regression elsewhere).
Pattern suggests no value of K crosses T-SAE's 1.80 unc peak with
coherent text.

### K=2 RESULTS — MULTI-SEED VERDICT (UPDATED)

**Multi-seed K=2 fails to generalize**: while sd=42 had K=2 sweet
spot, sd=1 K=2 collapses to mean success ~0.2 across all strengths.
Mechanism diagnosed:

| concept | sd=42 top-2 features | sd=1 top-2 features |
|---|---|---|
| harmful_content | 1582 (lift=20.4), 1565 (lift=9.7) | 362 (lift=15.5), 1142 (lift=12.7) |

The first feature is reliably concept-aligned (max-lift criterion).
The secondary feature is "lift-large but semantically inconsistent
across seeds". For sd=1 harmful_content, feature 1142 pulls toward
commerce/scientific markup, NOT violence. K=2 thus DEGRADES sd=1's
single-feature performance instead of amplifying it.

**Sd=1 K=2 generations** (concept = harmful_content):

- s=2: "the best deals on the best products. We are not responsible..."
- s=5: "<math xmlns:xlink=\"http://www.w.org/xlink\"><mrow>..."
- s=10: "<math> <math> <math>..."
- s=50: "ʔʔʔʔʔʔ" (Quranic question marks)

The secondary feature 1142 dominates and steers the model toward its
own polysemantic activations (math/commerce/punctuation), drowning
out the concept signal.

**Sd=1 K=2 numbers**:

| metric | K=2 sd=1 | K=1 sd=1 | Δ |
|---|---:|---:|---:|
| unconstrained peak | 0.27 | 1.37 | −1.10 |
| coh ≥ 1.5 | 0.20 | 0.63 | −0.43 |
| coh ≥ 1.75 | 0.20 | 0.63 | −0.43 |
| coh ≥ 2.0 | 0.20 | 0.63 | −0.43 |

K=2 is **seed-specific**. The sd=42 K=2 win was an artefact of
concept-aligned secondary features at that particular seed.

### K=2 sd=2 status

Currently grading. Expectation: similar collapse to sd=1.

### K=2 final verdict (updated)

K=2 multi-feature steering is **NOT a paper-grade improvement**.
The single-seed sd=42 lift was due to lucky semantic alignment of
top-2 features. Cross-seed generalization fails because:
- Top feature (max-lift) is reliably concept-aligned
- Secondary features (top-2 by lift) are POLYSEMANTIC and seed-
  specific
- Adding the secondary feature INTERFERES with rather than
  AMPLIFIES the concept signal

### Implications for Lever F (best-of-seeds feature picking)

The mechanism diagnosis suggests Lever F could work better than
Lever B: instead of picking 2 features at a single seed, pick the
BEST single feature across seeds (where "best" = lift × steering
quality on a held-out validation set). Lever F = better single-feature
selection > Lever B = unfiltered multi-feature.

### Original sd=42 K=2 RESULTS (for reference)

K=2 IS the sweet spot. Results on T=2 H8 sd=42 (single seed):

| protocol | unc | ≥1.5 | ≥1.75 | ≥2.0 | AUC(1.5-3.0) | AUC(1.75-3.0) |
|---|---:|---:|---:|---:|---:|---:|
| K=1 uniform PP | 1.57 | 1.53 | 1.53 | 1.53 | 0.707 | 0.613 |
| **K=2 multi-feature** | **1.73** | **1.73** | 1.03 | 0.50 | **0.815** | **0.649** |
| K=3 multi-feature | 1.60 | 1.60 | 0.50 | 0.50 | 0.610 | 0.505 |
| K=5 multi-feature | 1.37 | 1.37 | 0.83 | 0.83 | 0.616 | 0.484 |
| right-edge protocol | 1.37 | 1.27 | 1.27 | 1.27 | 0.771 | 0.659 |
| T-SAE k=20 anchor | 1.80 | 1.10 | 0.37 | 0.27 | 0.508 | 0.367 |

K=2 per-strength curves (sd=42):

| s_norm | succ | coh |
|---:|---:|---:|
| 0.5 | 0.300 | 2.600 |
| 1.0 | 0.500 | 2.400 |
| 2.0 | 1.033 | 1.967 ← coh just below 2.0 |
| **5.0** | **1.733** | **1.667** ← peak at borderline coh |
| 10.0 | 1.100 | 1.033 |
| 20.0 | 0.200 | 0.833 |

#### K=2 wins → potential paradigm shift?

Three potentially-significant claims:

1. **Unconstrained peak: 1.73 (vs T-SAE 1.80, gap 0.07)** — single seed.
   With multi-seed, this could cross T-SAE's 1.80, achieving the
   "TXC beats T-SAE on every metric" headline.

2. **Coh ≥ 1.5: 1.73 (Δ=+0.63 vs anchor)** — vs K=1 1.53 (Δ=+0.43).
   Lift of +0.20 at the prereg metric, single seed.

3. **AUC(1.5-3.0): 0.815 (Δ=+0.307 vs anchor)** — vs K=1 0.707
   (Δ=+0.199). Lift of +0.108 at Han's pre-stated alternative metric.

The coh ≥ 1.75 metric REGRESSES with K=2 (1.53 → 1.03) because the
peak strength has coh=1.667, just below 1.75. This is the trade-off:
K=2 amplifies signal at the cost of coherence at the peak strength.

**Multi-seed verification critical**. If K=2 sd=1 and sd=2 maintain
the lift, this is paper-grade. **Recommend running K=2 on T=2 H8
sd=1 and sd=2 next.**

### K=3 and K=5 — overshooting

K=3 lifts coh ≥ 1.5 by +0.07 but regresses sharply at coh ≥ 1.75.
K=5 saturates the model further; loses ground vs K=2 across the
board. Pattern: more features = more signal but coherence collapse
faster.

K=2 is the goldilocks: enough amplification to lift peak15 and
unconstrained, not so much that coherence collapses.

### Recommendation: defer Lever B, ride the GIGABRAIN reframe

The multi-coh-threshold reframe (`2026-04-30-y-coh-threshold-sweep.md`)
remains the strongest paper headline. Lever B is exposed as a noise
amplifier rather than a mechanism gain. Future work could explore
non-concept-orthogonal feature selection (e.g. correlated features)
but the marginal value vs the current WIN is small.
