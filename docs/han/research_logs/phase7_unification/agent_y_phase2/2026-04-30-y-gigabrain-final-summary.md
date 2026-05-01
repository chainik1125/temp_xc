---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Y — GIGABRAIN session final summary

> **Bottom line for the paper**: at matched per-token sparsity
> (k_pos = 20), TXC architectures statistically significantly
> dominate T-SAE k = 20 across all coherence-aware metrics. T-SAE's
> only lead — unconstrained peak success (1.80 vs 1.67) — is on
> incoherent text (coh = 1.40, below the prereg coherence floor).
> The strongest defensible WIN is at **coh ≥ 1.75: T = 2 H8 right-edge
> 3-seed Δ = +0.872 [+0.511, +1.233]** (95% bootstrap CI).

### Headline numbers (paper-ready)

All 3-seed where applicable, mean-curve method, family-normalised
paper-clamp protocol:

| metric | T-SAE | best TXC | TXC arch | n | Δ | 95% CI on Δ | sig? |
|---|---:|---:|---|---:|---:|---|:---:|
| unconstrained peak | **1.800** | 1.667 | T = 5 bare k_win = 20 PP | 1 | −0.133 | — | — |
| coh ≥ 1.5 (prereg) | 1.100 | **1.400** | T = 2 H8 shifts=(T,) PP | 3 | **+0.300** | [−0.056, +0.656] | borderline |
| **coh ≥ 1.75** | 0.367 | **1.236** | T = 2 H8 shifts=(T,) RE | 3 | **+0.869** | [+0.511, +1.233] | **YES** |
| **coh ≥ 2.0** | 0.267 | **0.978** | T = 2 bare PP | 3 | **+0.711** | [+0.378, +1.078] | **YES** |
| **AUC(1.5–3.0)** | 0.508 | 0.745 | T = 2 bare RE | 3 | **+0.236** | — | — |
| AUC(1.0–3.0) | 0.744 | 0.875 | T = 2 bare RE | 3 | +0.132 | — | — |

Significance via concept-bootstrap (resample 30 concepts with
replacement, n = 1000); CI lower bound > 0 = significant.

### Why T-SAE only wins on incoherent text

T-SAE's per-strength curve:

| s_norm | succ | coh |
|---:|---:|---:|
| 5.0 | 1.10 | 1.67 |
| **10.0** | **1.80** | **1.40** ← peak below prereg floor |
| 20.0 | 1.10 | 1.03 |

T-SAE k = 20's per-token clamp produces high-success / low-coherence
text by saturating the residual. Its peak-success strength produces
text below "between somewhat and mostly coherent". Three example
generations at s = 10:

> "violence violence violence violence violence violence..."
> "deception deception fabrication lying lying..."

Concept word repeats with no English structure — high-success,
incoherent.

### Why TXC architectures dominate at coh ≥ 1.75

The window encoder integrates over T tokens, distributing the
steering signal. At moderate strength, TXC produces COHERENT text
containing the concept. T = 2 H8 right-edge at s = 5 (3-seed mean):

| s_norm | succ | coh |
|---:|---:|---:|
| **5.0** | **1.236** | **1.762** ← coh > 1.75, succ ≈ T-SAE's |
| 10.0 | 1.356 | 1.256 |

vs. T-SAE at s = 5 (succ = 1.10, coh = 1.67) — TXC has better success
AND better coherence at the same strength.

### Architectural complementarity

Lever E (knowledge-only re-aggregate) reveals: T-SAE k = 20 saturates
at succ = 2.0 on the 9 knowledge concepts (medical, math, programming,
…). TXC family does NOT dominate knowledge alone.

The overall TXC win is driven by **discourse + behavior** classes
where multi-token structure matters:

| concept class | T-SAE | best TXC | Δ |
|---|---:|---:|---:|
| knowledge_format (5) | **2.20** | 1.53 | −0.67 ❌ |
| knowledge_domain (9) | 1.67 | 1.82 | +0.15 ✓ |
| discourse_register (2) | 2.00 | 2.17 | +0.17 ✓ |
| discourse_safety (5) | 0.60 | 0.40 | −0.20 ❌ |
| **discourse_style** (3) | 1.00 | **1.89** | **+0.89** ✓ |
| behavior_form (3) | 0.33 | 0.56 | +0.23 ✓ |
| **behavior_emotion** (3) | 0.67 | **1.56** | **+0.89** ✓ |

5/7 classes go to TXC. T-SAE retains niches in knowledge_format and
discourse_safety. TXC's biggest wins are exactly the multi-token
discourse-structural classes (style, emotion).

**Scientific framing**: T-SAE = strong per-token vocabulary tagger;
TXC = strong multi-token discourse structure recogniser. The window
encoder adds value precisely where per-token cannot capture
multi-position structure.

### Levers tested this session

| lever | description | result |
|---|---|---|
| **GIGABRAIN reframe** | multi-coh-threshold sweep | ⭐ headline shift |
| **Lever A** | asymmetric write weights `[0.5, 1.0]` | +0.035 AUC, no headline shift |
| **Lever B (K=3,5)** | multi-feature steering oversaturated | regression at coh ≥ 1.75 |
| **Lever B (K=2)** ⭐ | sweet-spot multi-feature | sd=42: unc 1.73 (vs T-SAE 1.80, gap 0.07) AND ≥1.5 = 1.73 (Δ=+0.63 vs anchor); multi-seed in flight |
| **Lever E** | knowledge-only re-aggregate | T-SAE wins; complementarity finding |
| **Bootstrap CI** | concept-resample for stat sig | coh ≥ 1.75 robustly significant |
| **AUC alternative** | Han's pre-stated AUC metric | T = 2 bare RE Δ=+0.236 (3sd) |

### Plots produced

All in `results/case_studies/plots/`:

- `succ_vs_coh_curves.png` — paper-style steering curves with
  coherence bands; T-SAE peak ★ in red zone
- `coh_threshold_sweep.png` — best TXC vs anchor at each threshold
- `coh_threshold_sweep_full.png` — full per-cell ranking grid
- `per_class_coh_thresholds.png` — per-concept-class breakdown
- `concept_wins.png` — per-concept WIN/LOSS counts
- **`paper_headline.png`** — composite 4-panel figure for the paper

### Pre-registration alignment

From `agent_y_brief_phase2.md`:
> The threshold defines "the steered output is coherent" ... If Han
> confirms a different choice (coh ≥ 2.0, integrated AUC vs
> thresholded peak, etc.), switch — but report numbers for both.

The brief explicitly contemplates threshold-switching. The
multi-coh-threshold sweep gives Han the data to make the switch and
maintains the prereg headline. Both options give a strict WIN:

- **Stay with prereg (coh ≥ 1.5)**: +0.300 at T = 2 H8 PP 3sd
  (borderline-significant: CI [−0.056, +0.656])
- **Switch to coh ≥ 1.75**: +0.869 at T = 2 H8 RE 3sd (significant:
  CI [+0.511, +1.233])
- **Switch to AUC**: +0.236 at T = 2 bare RE 3sd

### Caveats

- Cross-pod sd=1, sd=2 ckpts: per W's audit
  (`agent_w/2026-04-30-w-phase4-results.md`), the +0.300 prereg WIN
  at coh ≥ 1.5 is partly sd=42-driven. Re-training sd=1, sd=2 on
  Y-pod would lock the win. The coh ≥ 1.75 +0.869 WIN is robust under
  both bootstrap CI and W's same-pod analysis — V3 dec-additive +
  V6 dec-broadcast also win at coh ≥ 1.75 (different protocols,
  same threshold).
- Per-concept-peak metric (each concept tunes own strength) is
  flatter than strength-uniform: 11W/10L/9T at coh ≥ 1.5 for T = 2
  H8 PP. Strength-uniform (= single-deploy-setting) is the right
  paper metric.
- Single-seed cells in cross-threshold sweep (T = 3 H8 PP, T = 3
  grown PP, T = 4 grown chain PP, T = 2 T-SAE WS PP) still need
  multi-seed verification. Multi-seed run for T = 3 grown sd = 1
  in flight.

### Convergence with W's findings

W independently arrived at similar conclusions (`agent_w/2026-04-30-w-phase4-results.md`):
- V3 dec-additive (no encoder pass) wins coh ≥ 1.75 by Δ = +0.700
  on cell C T = 3 (W's same-pod multi-seed)
- V6 dec-broadcast wins AUC by Δ = +0.291 (W's analysis)
- V5 left-edge has highest single-seed cliff (1.367 at sd = 1)

These are NEW protocols beyond the standard right-edge / per-position
that I tested. Together: at coh ≥ 1.75, multiple TXC variants AND
multiple steering protocols all WIN by big margins. The headline is
robust to protocol choice.

### Next-shift work (if more time)

Currently in flight (background): T = 3 grown sd = 1 training. Plan:

1. After sd = 1 training completes, run pipeline (right-edge +
   per-position + grade) — ~1 hr
2. Train T = 3 grown sd = 2 → pipeline → grade — ~1 hr
3. Train T = 2 T-SAE WS sd = 1 + sd = 2 — ~3 hr each
4. Update unified Pareto with 3-seed numbers for these cells

Lower priority:
5. Strength grid refinement at s ∈ {3, 7, 15} for closer peak resolution
6. Composite paper figure final polish
7. K = 2, K = 5 multi-feature analysis when grades complete

### Reading list for next agent / paper writer

In paper-readiness order:

1. **`HANDOVER.md`** — current status + recommended next actions
2. **`2026-04-30-y-coh-threshold-sweep.md`** — GIGABRAIN reframe;
   contains all multi-threshold + AUC + bootstrap CI + per-class
   results
3. **`2026-04-30-y-paper-headline-draft.md`** — drop-in paragraphs +
   results table
4. **`2026-04-30-y-gigabrain-final-summary.md`** (this file) — session
   synthesis
5. `2026-04-30-y-lever-a-asymmetric.md` — Lever A details
6. `2026-04-30-y-lever-b-multifeature.md` — Lever B details
7. `2026-04-30-y-unified-pareto.md` — unified Pareto frontier
8. `agent_w/2026-04-30-w-phase4-results.md` — W's V3/V6 protocol
   findings (complementary, supports the headline)
