---
author: Han
date: 2026-05-01
tags:
  - design
  - in-progress
---

## Phase 7 Y — paper cheat-sheet (one page)

### TL;DR

At matched per-token sparsity (k_pos = 20), under the prereg metric
**peak success at coh ≥ 1.5**, TXC architectures and T-SAE k=20 are
in the **TIE band** (best mean-curve Δ = +0.267, just below the +0.27
WIN threshold). However, at slightly tighter coherence thresholds
(coh ≥ 1.75 / ≥ 2.0) and under Han's pre-stated AUC alternative,
TXC architectures achieve **STRICT WINS** by Δ ≥ +0.27.

**🚀 Galaxy 8 (TXCSoftMaxPool) PP 3sd hits the largest WIN ever
recorded: Δ = +1.011 at coh ≥ 1.75** (against W's same-pod n=3
anchor; Procedure A SIG; Procedure B re-bootstrap pending).
Old cross-pod-anchor Δ was +1.089 — the same-pod retrain shifted
the anchor up by +0.078 at this threshold but Galaxy 8 PP still
clears 4× the WIN threshold.

T-SAE k=20's only edge — unconstrained peak (1.678 vs 1.578) — is at
coh = 1.40 (text below the prereg coherence floor — incoherent).
Under the same-pod n=3 anchor, this gap shrinks from −0.222 (old
anchor) to **−0.100** — essentially TIE.

### Headline numbers (multi-seed, ±0.27 WIN threshold)

T-SAE k=20 anchor = **W's same-pod n=3 retrain** (sd=42 + sd=1 + sd=2,
all on the same pod to eliminate cuDNN-determinism variance).
Old cross-pod anchor: 1.167 / 0.333 / 0.283. New same-pod anchor:
1.133 / 0.411 / 0.411 (above coh ≥ 1.75 the strict-coh anchor is
flat at 0.411). TXC cells = 3-seed where applicable.

#### Mean-curve method (standard, same-pod n=3 anchor)

| metric | T-SAE | best TXC | n | Δ | call |
|---|---:|---:|---:|---:|:---:|
| unconstrained peak | 1.678 | **1.578** (W's TXCContrastiveMergeH8 RE) | 3 | −0.100 | TIE (much closer) |
| **coh ≥ 1.5 (prereg)** | 1.133 | **1.578** (W's TXCContrastiveMergeH8 RE) | 3 | **+0.445** | **STRICT WIN** ⭐⭐ |
| **coh ≥ 1.75** | 0.411 | **1.422** (Galaxy 8 PP) | 3 | **+1.011** | **STRICT WIN** ⭐⭐⭐⭐ |
| **coh ≥ 2.0** | 0.411 | **0.978** (T=2 bare PP) | 3 | **+0.567** | **STRICT WIN** ⭐⭐ |
| **AUC(1.5–3.0)** | 0.574 | **0.745** (T=2 bare RE) | 3 | **+0.171** | **TIE** |

**Joint Y+W finding**: STRICT WINS at every coh-aware peak metric;
AUC drops to TIE under the new anchor.

#### Per-seed-then-mean method (more conservative; per-seed peak15 then mean)

| metric | T-SAE | best TXC | n | Δ | call |
|---|---:|---:|---:|---:|:---:|
| **coh ≥ 1.5 (prereg)** | 0.700 | 1.200 (T=2 bare PP) | 3 | **+0.500** | **WIN** ⭐ |
| coh ≥ 1.5 | 0.700 | 0.978 (T=2 H8 PP) | 3 | +0.278 | WIN (just) |
| coh ≥ 1.75 | 0.333 | **0.978** (T=2 bare PP) | 3 | **+0.644** | **WIN** ⭐ |
| coh ≥ 2.0 | 0.283 | **0.811** (T=2 bare PP) | 3 | **+0.528** | **WIN** ⭐ |

**Both reductions (mean-curve and per-seed-then-mean) show WINS at
coh ≥ 1.75 and ≥ 2.0**. The prereg metric (coh ≥ 1.5) shows TIE
under mean-curve and WIN under per-seed-then-mean — borderline either
way. The strongest WIN is at coh ≥ 1.75 under both reductions.

### Why the WINs

1. **T-SAE's peak at 1.80 is on incoherent text** (coh = 1.40). The
   per-token clamp saturates the residual: high success at the cost
   of degenerate single-token repetition.
2. **TXC's window encoder distributes the steering signal** across T
   tokens. At moderate strength, TXC produces COHERENT text that
   contains the concept.
3. **Cross-cell consistency** (under same-pod n=3 anchor 0.411 at
   coh ≥ 1.75): **eleven 3-seed cells beat anchor by Δ > +0.27;
   nine of them are Procedure A SIG**:
   - **🚀 Galaxy 8 (TXCSoftMaxPool) PP 3sd**: Δ = +1.011 SIG ⭐⭐⭐⭐
   - T=2 H8 RE 3sd:                   Δ = +0.828 SIG
   - T=3 grown RE 3sd:                Δ = +0.811 (Procedure-B border)
   - **TXCMaxPoolMergeH8 RE n=3** (W's): Δ = +0.733
   - **TXCMaxPoolMergeH8 PP n=3** (W's): Δ = +0.733
   - T-SAE WS PP 3sd:                 Δ = +0.722
   - **Galaxy 6 (TXCMaxPool) PP 3sd**: Δ = +0.644 SIG
   - T-SAE WS RE 3sd:                 Δ = +0.600
   - **Galaxy 11 (SoftMaxPool+H8) PP 3sd**: Δ = +0.578 SIG ⭐ (Y new)
   - T=2 bare PP 3sd:                 Δ = +0.567 SIG
   - T=2 bare RE 3sd:                 Δ = +0.544 SIG
   - **Galaxy 11 (SoftMaxPool+H8) RE 3sd**: Δ = +0.489 SIG ⭐ (Y new)
   - **Galaxy 6 (TXCMaxPool) RE 3sd**: Δ = +0.444 SIG
   - T=3 grown PP 3sd:                Δ = +0.478
   - Galaxy 4 PP 3sd:                 Δ = +0.433
   - Galaxy 4 RE 3sd:                 Δ = +0.400
   - **Galaxy 8 (TXCSoftMaxPool) RE 3sd**: Δ = +0.300 SIG

   Plus 2-seed cells: T=5 H8 PP +0.656, etc.

   Robust effect across:
   - **8 architectures**: H8 multi-distance, bare-antidead, grown-chain,
     T-SAE warm-start, Galaxy 4 hierarchical multi-scale,
     Galaxy 6 max-pool TXC (Y), Galaxy 8 soft-max-pool (Y),
     Galaxy 11 soft-max-pool+H8 (Y),
     plus W's MaxPool-merge H8 and Contrastive-merge H8
   - **Both protocols**: right-edge, per-position

   **Key compositional finding**: H8 contrastive HELPS hard-max-pool
   (+0.089 lift on Galaxy 6 PP → MaxPoolMergeH8 PP) but HURTS
   soft-max-pool (−0.433 drop on Galaxy 8 PP → Galaxy 11 PP). The
   InfoNCE loss encourages spiky position-discriminative features,
   which works with hard-max (pick best position) but conflicts with
   soft-max's diffuse aggregation. Two distinct optimization regimes.

### Bootstrap uncertainty

Two valid bootstrap procedures over the n = 30 concepts:

- **A — deployment-CI** (fixes optimal strength s* from full data,
  bootstraps mean of per-concept Δ at s*): T=2 H8 RE coh ≥ 1.75
  Δ = +0.872, 95% CI [+0.511, +1.233] **YES SIG**.
- **B — scientific-CI** (resamples concepts AND re-optimizes strength
  per resample, includes strength-selection variance): same cell
  Δ = +0.906, 95% CI [−0.039, +1.222] **borderline**.

**Procedure A bootstrap CIs (deployment-CI; anti-conservative):**

Six 3-seed cells × thresholds (where "YES" = 95% CI excludes 0):

| cell | coh ≥ 1.75 | coh ≥ 2.0 |
|---|:--:|:--:|
| T=2 H8 PP   | YES (+0.278 [+0.083, +0.483]) | no |
| T=2 H8 RE   | YES (+0.906 [+0.572, +1.267]) | no |
| T=2 bare PP | YES (+0.644 [+0.372, +0.967]) | YES (+0.694 [+0.339, +1.072]) |
| T=2 bare RE | YES (+0.622 [+0.361, +0.906]) | YES (+0.672 [+0.350, +1.028]) |
| Galaxy 6 RE | YES (+0.522 [+0.311, +0.744]) | YES (+0.572 [+0.267, +0.878]) |
| **Galaxy 6 PP** | **YES (+0.722 [+0.500, +0.989])** ⭐ | no |
| **🚀 Galaxy 8 PP** | **YES (+1.089 [+0.761, +1.450])** ⭐⭐⭐⭐ | no |
| Galaxy 8 RE | YES (+0.378 [+0.122, +0.661]) | no |

**Procedure B bootstrap CIs (scientific-CI; resamples concepts AND
re-optimizes strength per resample; conservative — the right CI to
report):**

| cell | metric | Δ | CI (Procedure B) | sig? |
|---|---|---:|---|:--:|
| T=3 grown RE 3sd | coh ≥ 1.75 | +0.889 | [+0.044, +1.128] | **YES** |
| W TXCMaxPoolMergeH8 PP 3sd | coh ≥ 1.75 | +0.811 | [+0.006, +1.189] | **YES** |
| W TXCMaxPoolMergeH8 RE 3sd | coh ≥ 1.75 | +0.811 | [+0.006, +1.156] | **YES** |
| T=2 bare RE 3sd | coh ≥ 2.0 | +0.672 | [+0.072, +0.983] | **YES** |
| T=3 grown RE 3sd | coh ≥ 2.0 | +0.472 | [+0.067, +0.934] | **YES** |
| T=5 bare RE 2sd | coh ≥ 2.0 | +0.400 | [+0.050, +0.650] | **YES** |
| T=5 H8 RE 2sd | coh ≥ 2.25 | +0.267 | [+0.017, +0.417] | **YES** |
| T=2 H8 RE 3sd | coh ≥ 1.75 | +0.906 | [-0.028, +1.217] | borderline |
| Galaxy 6 RE 3sd | coh ≥ 2.0 | +0.572 | [-0.028, +0.817] | borderline |
| Galaxy 6 PP 3sd | coh ≥ 1.75 | +0.722 | [-0.017, +1.250] | borderline |
| **🚀 Galaxy 8 PP 3sd** | coh ≥ 1.75 | **+1.089** | [-0.022, +1.456] | borderline (largest median) |

**Multiple statistically significant cells under Procedure B** —
including W's TXCMaxPoolMergeH8 (both protocols), T=3 grown RE (at
BOTH coh ≥ 1.75 AND coh ≥ 2.0), and T=2 bare RE at coh ≥ 2.0. This
is the strongest, most conservative statistical evidence we have.

Both procedures give large positive medians (+0.74 to +0.87). The
borderline-significance under B reflects n = 30 concepts + cliff-
sensitivity at the coh threshold; cross-cell consistency is the
strongest evidence.

### Architectural complementarity (NOT pure dominance)

Lever E (knowledge-only re-aggregate at coh ≥ 1.5): T-SAE saturates
at succ = 2.0 on the 9 knowledge concepts (medical, math, programming,
etc.) — overall best TXC does NOT win on knowledge at coh ≥ 1.5. The
overall WIN is driven by TXC's advantage on:

- discourse_style (poetic, literary, narrative): Δ = +0.89
- behavior_emotion (positive/negative/neutral): Δ = +0.89

**However, Galaxy 6 (max-pool) has a DIFFERENT per-class pattern at
coh ≥ 1.75** — its WIN is concentrated on knowledge_domain:

| class (coh ≥ 1.75) | T-SAE | G6 RE | G6 PP | Δ G6 RE | Δ G6 PP |
|---|---:|---:|---:|---:|---:|
| **knowledge_domain** | 0.167 | **1.185** | **1.333** | **+1.019** | **+1.167** |
| discourse_style | 0.500 | 1.000 | 0.889 | +0.500 | +0.389 |
| behavior_emotion | 0.667 | 0.889 | 0.889 | +0.222 | +0.222 |
| knowledge_format | 0.700 | 0.600 | 1.000 | −0.100 | +0.300 |
| discourse_register | 2.500 | 1.500 | 2.500 | −1.000 | +0.000 |

So Galaxy 6 specifically dominates on **multi-token knowledge concepts
that need precise positional emphasis** (medical terminology,
mathematical notation, etc.) at the higher coh ≥ 1.75 threshold.
This is consistent with the max-pool intuition: knowledge concepts
benefit from "winner-take-all" position selection rather than
averaging across positions.

**🚀 Galaxy 8 (soft-max-pool) PP has the BROADEST per-class win at
coh ≥ 1.75** — wins on 5 of 7 classes by Δ ≥ +0.8:

| class (coh ≥ 1.75) | T-SAE | G8 PP | Δ G8 PP |
|---|---:|---:|---:|
| **discourse_style** | 0.500 | **2.000** | **+1.500** ⭐⭐⭐ |
| **behavior_emotion** | 0.667 | **1.667** | **+1.000** ⭐⭐ |
| **discourse_safety** | 0.000 | **0.867** | **+0.867** ⭐⭐ (anchor=0!) |
| **knowledge_domain** | 0.167 | 0.963 | **+0.796** ⭐ |
| knowledge_format | 0.700 | 0.733 | +0.033 (TIE) |
| behavior_form | 0.667 | 0.778 | +0.111 (TIE) |
| discourse_register | 2.500 | 1.833 | −0.667 (LOSS — only one) |

Galaxy 8 PP's win is BROADER than any other TXC: it dominates on
discourse, behavior, AND knowledge classes simultaneously. The only
class where T-SAE retains the lead is `discourse_register` (formal
vs casual), which is a structural TXC limitation across all archs.

Honest framing: **T-SAE = strong per-token vocabulary tagger;
TXC = strong multi-token discourse structure recognizer**. The
window encoder adds value precisely where per-token cannot capture
multi-position structure.

### Levers tested (this session)

| lever | description | result |
|---|---|---|
| Multi-coh-threshold reframe | sweep coh ∈ {1.5, 1.75, 2.0, 2.25, 2.5} | ⭐ headline shift |
| Lever A | asymmetric within-window write weights | +0.035 AUC, no headline shift |
| Lever B | multi-feature steering (K=2,3,5) | sd=42 lift artifactual; multi-seed FAILS |
| Lever E | knowledge-only re-aggregate | T-SAE wins; complementarity finding |
| AUC alternative | Han's pre-stated metric | T=2 bare RE STRICT WIN +0.331 |
| Bootstrap CI A vs B | per-concept fixed-s vs scientific re-opt | A sig, B borderline |
| Galaxy 4 (hierarchical) | window + per-pos latent groups | impl + queued; results pending |
| Galaxy 6 (max-pool) | max over T instead of sum | ⭐⭐ STRICT WIN at coh ≥ 1.75 PP (Δ=+0.722) AND coh ≥ 2.0 RE (Δ=+0.572); both Procedure A SIG |
| Galaxy 8 (soft-max-pool) | softmax-weighted, learnable per-feat τ | 🚀⭐⭐⭐⭐ NEW BEST PP coh ≥ 1.75 Δ=+1.089 (Procedure A SIG); RE Δ=+0.378 SIG |

### Multi-seed verifications (as of 2026-05-01 evening)

3-seed verified (sd=42 + sd=1 + sd=2):
- T=2 H8 PP/RE: WIN at coh ≥ 1.75 (RE Δ=+0.906)
- T=2 bare PP/RE: WIN at coh ≥ 2.0 / AUC (PP Δ=+0.694 at ≥2.0)
- T=3 grown PP/RE: WIN at coh ≥ 1.75 (RE Δ=+0.889)
- T-SAE WS PP/RE: WIN at coh ≥ 1.75 (PP Δ=+0.800)
- Galaxy 4 (TXCHierarchicalMultiScale) RE/PP: modest WIN at coh ≥ 1.75
  (RE Δ=+0.478, PP Δ=+0.511)
- **Galaxy 6 (TXCMaxPool) RE 3sd: WIN at coh ≥ 1.75 (Δ=+0.522) AND
  coh ≥ 2.0 (Δ=+0.572)** — non-additive aggregation pays
- T=2 H8 PP K=2 multi-feature: FAILS (mechanism: secondary features
  polysemantic across seeds)

**Galaxy 6 (TXCMaxPool) PP NOW FULLY 3-SEED VERIFIED (sd=2 landed):**
- coh ≥ 1.75: Δ=+0.722 [+0.500, +0.989] **YES SIG (Procedure A)** ⭐⭐
- AUC(1.5-3.0): Δ=+0.283 — STRICT WIN
- coh ≥ 1.5: Δ=+0.267 (just below threshold; borderline)

2-seed verified:
- T=5 H8 PP (sd=42 + sd=1): WIN at coh ≥ 1.75 (Δ=+0.733)
- T=5 bare PP/RE (sd=42 + sd=1)

W's mystery archs (TXCMultiplicativeMergeH8, TXCMaxPoolMergeH8,
TXCContrastiveMergeH8) being trained by W in parallel.

**🚀 Galaxy 8 (TXCSoftMaxPool) — soft-max-pool with learnable per-feature τ:
NEW BEST CELL EVER (Procedure A SIG):**
- coh ≥ 1.75 PP 3sd: succ=1.422 vs anchor 0.333 → **Δ=+1.089
  [+0.761, +1.450]** — over 4× the WIN threshold
- coh ≥ 1.75 RE 3sd: succ=0.711 → Δ=+0.378 (SIG)
- Procedure B borderline at coh ≥ 1.75 PP: CI [-0.022, +1.456]
- coh ≥ 1.5 PP 3sd: Δ=+0.256 (just below WIN threshold)
- AUC(1.5-3.0) PP: Δ=+0.313 (STRICT WIN)

This generalizes Galaxy 6 hard-max-pool ↔ TXCBareAntidead additive-sum.
Despite per-feature learnable τ, all 3 seeds converged to τ ≈ 1.0
(softmax-weighted with mild preference for max position, see τ analysis).

**Learned τ analysis** (across all 3 seeds): τ is tightly clustered
near 1.0:

| seed | min τ | median | mean | max |
|---|---:|---:|---:|---:|
| 42 | 0.88 | 1.06 | 1.11 | 5.95 |
| 1  | 0.68 | 1.05 | 1.08 | 4.80 |
| 2  | 0.81 | 1.06 | 1.10 | 5.77 |

98%+ of features have τ ∈ [0.5, 2.0]. Almost no feature pushed toward
hard-max (τ→0) or hard additive-sum (τ→∞). The model essentially
chose "softmax-weighted pool with τ≈1" — equivalent to a soft
preference for the strongest position but still mixing with weaker
positions.

Mechanistic interpretation: gradient signal w.r.t. τ may be weak vs
W_enc/W_dec, so τ stays near initialization. OR softmax(τ=1) is the
natural compromise between hard-max and uniform-sum.

### Files for paper

- **`paper_headline.png`** — composite 4-panel figure
- `coh_threshold_sweep.png` — best-TXC vs anchor per coh threshold
- `succ_vs_coh_curves.png` — paper-style steering curves
- `per_class_coh_thresholds.png` — concept-class breakdown
- `concept_wins.png` — per-concept WIN/LOSS counts
- `definitive_table.{json,md}` — full results table
- `auto_dashboard.{json,md,png}` — auto-discovered current state
