---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Hail Mary — Step 1 (T=2) right-edge + per-position

> **Step 1 status: complete.** Two protocols evaluated:
> - Right-edge (`steering_paper_normalised/`): TIE at coh ≥ 1.5
>   (success 0.833, Δ=−0.267 vs anchor 1.10).
> - **Per-position** (`steering_paper_window_perposition/`):
>   **TIE with positive Δ** (success **1.233**, **Δ=+0.133** vs anchor).
>
> **Per-position write-back at T=2 produces the best matched-sparsity
> TXC cell so far — slightly *exceeds* T-SAE k=20 under METRIC B.**

### Step 1 training summary

`txc_bare_antidead_t2_kpos20__seed42.pt`. Random-init, T=2, k_pos=20,
k_win=40. Plateau-converged at step 4800 (plateau=0.016, threshold=0.02).
Loss 24108 → 4372 (5.5× drop). l0=40 (= k_win full-TopK). Wall **23 min**
(faster than Step 2's 46 min — half the parameters).

### Step 1 features

24 distinct/30 (one less than T=5's 24/30; same polysemanticity story but
slightly less severe). Worst-collision feature: feat 6905 picked for 4
concepts (jailbreak_pattern / legal / technical_jargon / neutral_factual).

⟨|z|⟩ = 11.9 (vs T=5's 25.1) — **half the magnitude as expected**: window
encoder integrates over 2 vs 5 positions. Strength schedule:
{6.0, 11.9, 23.9, 59.6, 119.3, 238.5, 596.3}.

### Strength curves (right-edge vs per-position)

#### Right-edge protocol

| s_abs | success | coh |
|---|---|---|
| 6.0 | 0.300 | 2.933 |
| 11.9 | 0.400 | 2.933 |
| 23.9 | 0.300 | 2.433 |
| **59.6** | 0.833 | **2.233** ← peak under coh ≥ 1.5 |
| 119.3 | **1.300** | 1.367 ← unconstrained peak |
| 238.5 | 1.233 | 1.100 |
| 596.3 | 0.367 | 0.900 |

#### Per-position protocol

| s_abs | success | coh |
|---|---|---|
| 6.0 | 0.200 | 2.967 |
| 11.9 | 0.333 | 2.733 |
| 23.9 | 0.333 | 2.567 |
| 59.6 | 0.833 | 1.933 |
| **119.3** | **1.233** | **1.567** ← peak under coh ≥ 1.5 ⭐ |
| 238.5 | 1.300 | 1.100 ← unconstrained peak (same as right-edge) |
| 596.3 | 0.333 | 1.033 |

**Key**: per-position at s=119.3 keeps coh=1.567 (above 1.5 threshold).
Right-edge at s=119.3 has coh=1.367 (below threshold). The per-position
protocol *holds coherence above 1.5 ~30 strength units further*, so the
constrained peak shifts up (from s=59.6 right-edge to s=119.3
per-position) and *sticks* at higher success (1.233 vs 0.833).

**Hygiene**: peak under coh ≥ 1.5 (s=119.3, succ=1.233) is interior;
next-up s=238.5 has coh=1.100 (constraint-bound, true optimum).

### Combined picture — all matched-sparsity cells & protocols

| arch+protocol | peak unc. | @s_abs | coh@unc | peak coh ≥ 1.5 | call_15 | Δ_15 |
|---|---|---|---|---|---|---|
| **tsae_paper_k20 (anchor)** | 1.800 | 99.8 | 1.400 | 1.100 | TIE | +0.000 |
| Step 1 (T=2, right-edge) | 1.300 | 119.3 | 1.367 | 0.833 | TIE | −0.267 |
| **Step 1 (T=2, per-position)** | 1.300 | 238.5 | 1.100 | **1.233** | **TIE** | **+0.133** ⭐ |
| Step 2 (T=5, right-edge) | 1.000 | 251.1 | 1.200 | 0.700 | LOSS | −0.400 |
| Step 3 (T=5, per-position) | 0.900 | 251.1 | 1.333 | 0.833 | TIE | −0.267 |
| Cell C (T=3, right-edge, W's) | 1.400 | 164.2 | 1.400 | 0.767 | LOSS | −0.333 |

### Per-position protocol boost by T (right-edge → per-position)

| T | right-edge peak15 | per-position peak15 | Δ |
|---|---|---|---|
| **2** | 0.833 | **1.233** | **+0.400** ⭐ |
| 3 | 0.767 | (untested — W can run) | ? |
| 5 | 0.700 | 0.833 | +0.133 |

**Striking**: the per-position boost is **3× larger at T=2** than at T=5.

This was unexpected — naïve prediction would say per-position helps more
at larger T (more positions to "smear" the steered feature into; bigger
gap from right-edge-only). The data shows the *opposite*: at T=2, where
right-edge writes 50% of the window, per-position lifts harder.

**Possible mechanism**: at T=2 the picked features are *already cleaner*
(less polysemanticity than T=5). Per-position protocol amplifies this
quality — when you write a high-quality concept-specific feature at all
T positions, the model's coherent generation continues to express that
concept. At T=5 the feature is noisier (more polysemanticity), so
per-position protocol amplifies the noise as well as the signal.

### Outcome (called against pre-registered ±0.27 threshold)

**Under METRIC A (unconstrained)**: still LOSS (every cell well below
1.80 anchor by ≥ 0.27).

**Under METRIC B (peak at coh ≥ 1.5; the brief's locked primary)**:
*three TIE cells* (Step 1 right-edge, Step 1 per-position, Step 3
per-position) and *two LOSS cells* (Step 2 right-edge, Cell C
right-edge). The Step 1 per-position cell is **positively above
anchor** (+0.133), though within tie-band.

Per the pre-registered rule, all TIE cells need multi-seed
disambiguation. **Step 2 seed=1 training is in flight** (started 23:57,
ETA ~00:43). Will run both right-edge + per-position pipelines on
seed=1 to verify the Step 2 / Step 3 outcomes.

### Implication for paper

If multi-seed seed=1 verifies the Step 1 per-position TIE+ result and
the Step 3 TIE result, the headline becomes:

> **Under per-position write-back (Q2.C-style protocol) at family-
> normalised paper-clamp strengths and coherence ≥ 1.5, matched-sparsity
> TXC cells (k_pos=20) at T=2 and T=5 land within ±0.27 σ_seeds of
> T-SAE k=20's peak success. The T=2 cell *slightly exceeds* (+0.133)
> the per-token T-SAE k=20 baseline. Architecture is not the bottleneck
> at matched sparsity *under the right protocol*.**

That's a much stronger headline than "TXC competitive with T-SAE on
some concept classes". Per-position protocol is the lever.

### Files

- Step 1 training: `results/ckpts/txc_bare_antidead_t2_kpos20__seed42.pt`,
  `results/training_logs/txc_bare_antidead_t2_kpos20__seed42.json`
- Step 1 right-edge: `results/case_studies/steering_paper_normalised/txc_bare_antidead_t2_kpos20/`
- Step 1 per-position: `results/case_studies/steering_paper_window_perposition/txc_bare_antidead_t2_kpos20/`
- Step 1 features + diagnostics: `results/case_studies/{steering,diagnostics_kpos20}/`
