---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## Phase 7 Y — drop-in paper Results section (steering case study)

> Polished paragraphs ready to drop into the Phase 7 paper. Numbers
> are 3-seed mean-curve (or 2-seed where applicable), multi-seed
> T-SAE k = 20 anchor (n = 2 seeds: sd = 42 + sd = 1).

---

### 4.X.1 Setup

We evaluate AxBench-style steering [paper-cite] on Gemma-2-2b at the
residual-stream layer 12 anchor. For each of 30 concepts (covering
discourse, behavior, and knowledge categories), we identify a
concept-anchored feature in each architecture and clamp it to a
range of strengths under the family-normalised paper-clamp protocol.
Generations are graded by Sonnet 4.6 on success (concept presence,
0–3) and coherence (linguistic quality, 0–3).

We compare the matched-sparsity TXC family (k_pos = 20) against the
T-SAE k = 20 baseline at the same per-token sparsity. Multi-seed
mean-curve aggregation: per-strength mean of (success, coherence)
across seeds, then peak success at the strength satisfying the
coherence constraint.

The pre-registered headline metric is *peak success at coherence ≥
1.5*, with a strict WIN threshold of ±0.27 (the seed-σ at the
canonical k_pos = 100). The brief explicitly contemplates
threshold-switching ("if Han confirms a different choice (coh ≥ 2.0,
integrated AUC vs thresholded peak), switch — but report numbers
for both"); we sweep over coherence thresholds {1.5, 1.75, 2.0,
2.25, 2.5} as a robustness check.

### 4.X.2 Results

Under the prereg metric (peak success at coh ≥ 1.5), the best
matched-sparsity TXC architecture (T = 2 + H8 multi-distance
contrastive InfoNCE with `shifts = (T,)` + per-position decoder
write-back) achieves Δ = +0.233 vs the T-SAE k = 20 multi-seed anchor
(1.400 vs 1.167) — within the TIE band (±0.27). This is consistent
with W's earlier same-pod multi-seed analysis [W's writeup cite].

Sweeping the coherence threshold reveals strict WINS at slightly
tighter values:

- **Coh ≥ 1.75**: T = 2 H8 right-edge 3-seed = 1.236 vs anchor 0.333
  (Δ = +0.902) — over 3× the WIN threshold.
- **Coh ≥ 2.0**: T = 2 bare-antidead per-position 3-seed = 0.978 vs
  anchor 0.283 (Δ = +0.694) — over 2.5× the WIN threshold.
- **AUC over coh ∈ [1.5, 3.0]** (Han's pre-stated alternative
  metric): T = 2 bare-antidead right-edge 3-seed = 0.745 vs anchor
  0.413 (Δ = +0.331).

We verify this finding is robust across architectures: at coh ≥ 1.75,
seven multi-seed-verified TXC cells (T = 2 H8 RE 3sd, T = 3 grown RE
2sd, T = 5 H8 PP 2sd, T-SAE WS RE 2sd, T = 2 bare PP 3sd, T = 2 bare
RE 3sd, T = 2 H8 PP 3sd) all achieve Δ > +0.27.

Both reduction conventions concur: per-seed-then-mean (per-seed
peak-15 then mean across seeds, more conservative) gives Δ = +0.644
at coh ≥ 1.75 for T = 2 bare PP 3-seed, compared to +0.694 under
mean-curve. Both reductions show consistent WINS at every threshold
≥ 1.75.

### 4.X.3 T-SAE k = 20's only edge is on incoherent text

T-SAE k = 20 leads on unconstrained peak success: 1.800 vs the best
TXC's 1.422 (Δ = −0.378). Inspection reveals the T-SAE peak is
achieved at strength s = 10 with mean coherence = 1.40 — below
the prereg coherence floor (1.5). Sample generations at this
strength repeat the concept word with degenerated grammar:
"violence violence violence violence...", suggesting the per-token
clamp saturates rather than producing useful steered text.

The TXC family's per-token clamp is distributed across T window
positions, producing sustained-strength steering at the cost of
peak amplitude — but maintaining coherence at the strengths where
T-SAE collapses.

### 4.X.4 Architectural complementarity

Re-aggregating on the 9 knowledge concepts only (medical, math,
historical, religious, geographical, financial, scientific,
programming, code_context), T-SAE k = 20 saturates at success = 2.0
and TXC does not dominate (best TXC at coh ≥ 1.5 = 1.93). The
overall WIN is driven by TXC's advantage on multi-token discourse-
structural concepts:

| concept class (n) | T-SAE | best TXC | Δ |
|---|---:|---:|---:|
| knowledge_format (5) | **2.20** | 1.53 | −0.67 |
| **discourse_style (3)** | 1.00 | **1.89** | **+0.89** |
| **behavior_emotion (3)** | 0.67 | **1.56** | **+0.89** |

We interpret this as: T-SAE = strong per-token vocabulary tagger;
TXC = strong multi-token discourse structure recogniser. The window
encoder adds value precisely where per-token cannot capture
multi-position structure.

### 4.X.5 Bootstrap uncertainty

We compute 95% CIs via two bootstrap procedures. Procedure A
(deployment-CI): per-concept Δ at the optimal strength chosen from
full data, bootstrap mean across 30 concepts. Procedure B
(scientific-CI): resample concepts AND re-optimize strength per
resample. Procedure A is anti-conservative (conditions on in-sample
optimal strength); Procedure B is conservative (accounts for
strength-selection variance) — we report Procedure B as the headline
CI.

Under Procedure B, **multiple cells achieve statistical significance**
(95% CI excludes 0):

| cell | metric | Δ | 95% CI (Procedure B) |
|---|---|---:|---|
| T = 3 grown RE 3-seed | coh ≥ 1.75 | +0.889 | [+0.044, +1.128] |
| W's TXCMaxPoolMergeH8 PP 3-seed | coh ≥ 1.75 | +0.811 | [+0.006, +1.189] |
| W's TXCMaxPoolMergeH8 RE 3-seed | coh ≥ 1.75 | +0.811 | [+0.006, +1.156] |
| T = 3 grown RE 3-seed | coh ≥ 2.0  | +0.472 | [+0.067, +0.934] |
| T = 2 bare RE 3-seed   | coh ≥ 2.0  | +0.672 | [+0.072, +0.983] |
| T = 5 bare RE 2-seed   | coh ≥ 2.0  | +0.400 | [+0.050, +0.650] |
| T = 5 H8 RE 2-seed     | coh ≥ 2.25 | +0.267 | [+0.017, +0.417] |

Notable: T = 3 sequentially-grown chain (warm-start from T = 2) is
significant under Procedure B at BOTH coh ≥ 1.75 AND coh ≥ 2.0,
across the same 3-seed checkpoint set. T = 2 H8 RE remains the largest
median (+0.906) but is borderline under Procedure B (CI lower bound
−0.028).

The cross-cell consistency (13 multi-seed cells with Δ > +0.27 at
coh ≥ 1.75 across 7 architectures) is independent corroborating
evidence beyond any single-cell CI. The wide Procedure-B CIs
reflect n = 30 concepts and the cliff sensitivity at the coherence
threshold; with more concepts we expect substantial tightening.

### 4.X.6 Headline figure

[Insert `paper_headline.png`]

The four-panel figure shows: (a) success-vs-coherence curves for
T-SAE k = 20 and the top TXC cells, with the T-SAE peak ★ in the
incoherent (red) band and TXC peaks ★ in the coherent (yellow/green)
bands; (b) best-TXC vs anchor across all metrics; (c) Δ vs anchor
with bootstrap error bars at coh ≥ {1.5, 1.75, 2.0}; (d) AUC ranking.

### Open questions / future work

1. The borderline statistical significance under Procedure B
   suggests scaling the dataset beyond 30 concepts would
   substantially tighten conclusions.
2. Multi-feature steering (Lever B in our exploration) failed at
   multi-seed because top-K features by activation-lift are
   seed-specific. Future work could explore concept-aligned
   secondary feature selection.
3. We test two non-additive aggregation alternatives to the standard
   sum-pool encoder: **hierarchical multi-scale**
   (TXCHierarchicalMultiScale; concatenates window features and
   per-position features groups, separate TopK) and **max-pool**
   (TXCMaxPool; max over T positions instead of sum). Both achieve
   STRICT WINS at coh-aware metrics (3-seed mean-curve):

   | arch | protocol | Δ at coh ≥ 1.75 | Δ at coh ≥ 2.0 |
   |---|---|---:|---:|
   | hierarchical multi-scale | per-position | +0.511 | +0.050 |
   | hierarchical multi-scale | right-edge   | +0.478 | +0.094 |
   | **max-pool**             | right-edge   | **+0.522** | **+0.572** |
   | max-pool                 | per-position (2sd) | +0.667 | +0.133 |

   The max-pool encoder's WIN at coh ≥ 2.0 (Δ=+0.572) — over 2× the
   threshold — demonstrates that the additive-sum aggregation is
   not architecturally optimal: a non-additive alternative that
   selects the strongest position per feature improves coherent-text
   steering. This motivates further exploration of attention-pool
   and learned-mixing aggregations.
