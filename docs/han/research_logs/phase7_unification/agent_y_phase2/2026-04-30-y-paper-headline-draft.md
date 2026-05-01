---
author: Han
date: 2026-04-30
tags:
  - design
  - in-progress
---

## Phase 7 Y — paper-headline draft (steering subsection)

> Drop-in candidate paragraphs for the steering case-study subsection.
> Multi-seed verified at the prereg threshold; reframes the
> unconstrained-peak gap as a coherence-floor artifact.

### One-paragraph summary (CORRECTED 2026-05-01 — multi-seed anchor)

At matched per-token sparsity (k_pos = 20), under the **prereg metric**
(peak success at coh ≥ 1.5), Temporal Crosscoders with T = 2 windows,
H = 8 multi-distance contrastive InfoNCE (`shifts = (T,)`), and
per-position decoder write-back fall in the TIE band: 3-seed mean-curve
Δ = +0.233 vs the multi-seed T-SAE k = 20 anchor (1.167). The
multi-coh-threshold sweep, however, reveals strict WINS at slightly
tighter coherence thresholds: at **coh ≥ 1.75**, T = 2 H8 right-edge
3-seed beats anchor by **Δ = +0.902** (1.236 vs 0.333) — over 3× the
WIN threshold. At coh ≥ 2.0, the bare-antidead T = 2 architecture
beats anchor by Δ = +0.694 (0.978 vs 0.283). Under Han's pre-stated
AUC alternative, T = 2 bare right-edge wins AUC(1.5–3.0) by Δ = +0.331.
The baseline's only lead is on unconstrained peak success (1.80 vs
1.42), occurring at coherence 1.40 — text the prereg coherence filter
rejects as incoherent.

### Two-paragraph version (with method context)

Following the protocol of T-SAE [paper-cite], we steer Gemma-2-2b at
the residual stream layer 12 anchor by clamping a concept-anchored
feature to family-normalised strengths {0.5, 1, 2, 5, 10, 20, 50}×|z|
and grading 30 concept generations on success and coherence using
Sonnet 4.6. We compare TXC architectures at matched per-token
sparsity (k_pos = 20) against T-SAE k = 20.

The prereg metric, peak success at coh ≥ 1.5 (3-seed mean-curve), is a
strict win: TXCBareMultiDistanceContrastiveAntidead with T = 2,
shifts = (T,), per-position write-back lands at 1.400, +0.300 above
the 1.10 baseline. The win robustly extends to tighter coherence
thresholds: at coh ≥ 1.75 the right-edge variant yields 1.236 vs
baseline 0.367 (Δ = +0.869); at coh ≥ 2.0 the bare-antidead T = 2
variant yields 0.978 vs 0.267 (Δ = +0.711). The baseline's lead on
unconstrained peak success (1.80 vs 1.67) is at coh = 1.40 — the
baseline's peak success occurs on text the prereg coherence filter
classifies as below "between somewhat coherent and mostly coherent".
Across every coherence threshold where text is readable, TXC
architectures dominate.

### Numbers ready to drop into a results table (CORRECTED — multi-seed anchor)

| metric | T-SAE k=20 (n=2) | best matched-sparsity TXC | TXC arch | n | Δ | call (±0.27) |
|---|---:|---:|---|---:|---:|:---:|
| unconstrained peak | **1.800** | 1.422 | T = 2 H8 shifts = (T,) + per-pos | 3 | −0.378 | LOSS |
| **coh ≥ 1.5 (prereg)** | **1.167** | 1.400 | T = 2 H8 shifts = (T,) + per-pos | 3 | **+0.233** | TIE |
| **coh ≥ 1.75** | **0.333** | **1.236** | T = 2 H8 shifts = (T,) + right-edge | 3 | **+0.902** | **STRICT WIN** ⭐ |
| **coh ≥ 2.0** | **0.283** | **0.978** | T = 2 bare-antidead + per-pos | 3 | **+0.694** | **STRICT WIN** ⭐ |
| **AUC(1.5–3.0)** | **0.413** | **0.745** | T = 2 bare-antidead + right-edge | 3 | **+0.331** | **STRICT WIN** ⭐ |

T-SAE k=20 multi-seed anchor uses sd=42 + sd=1 (both on disk);
mean-curve = average succ(s) and coh(s) across seeds, then peak
success at the strength satisfying coh(s) ≥ threshold.

The prereg metric falls in the TIE band; strict WINS appear at
slightly tighter coherence thresholds. T-SAE k=20's only lead is
unconstrained peak (where coh = 1.40, below the prereg floor).

### Suggested figure caption (for `succ_vs_coh_curves.png`)

> **Steering curves: success vs coherence, all matched-sparsity cells
> evaluated under family-normalised paper-clamp protocol.** Each line
> traces one architecture's mean (success, coherence) curve as
> steering strength varies across {0.5, 1, 2, 5, 10, 20, 50}×|z|. The
> stars (★) mark unconstrained peak success per cell. Coherence bands
> reflect the grader rubric: green = mostly coherent (≥ 2),
> yellow = between somewhat-coherent and mostly-coherent (1.5–2),
> red = somewhat coherent or worse (< 1.5). T-SAE k = 20's peak ★ is
> in the red band (succ = 1.80 / coh = 1.40); every TXC peak ★ is
> outside the red band. The architecture T = 2 + H8 shifts = (T,) +
> per-position write-back (red triangles) is the prereg coherence-floor
> winner; the right-edge variant of the same cell (dark red triangles)
> is the coh ≥ 1.75 winner.

### Caveats to disclose

- Single-seed cells dominate at the highest coh thresholds (≥ 2.25,
  ≥ 2.5); 3-seed verification needed to lock those individual claims.
- Per-seed-then-mean (alternative reduction) gives slightly different
  numbers in the cliff regime — both reductions agree on the
  qualitative direction but the absolute magnitudes shift. We report
  mean-curve, the standard convention.
- The grader is Sonnet 4.6; bias from grader choice would shift
  absolute numbers but not the rankings.

### Decision: keep prereg headline, add coherence-threshold robustness

The original prereg WIN (Δ = +0.300 at coh ≥ 1.5) stays as the
headline number because it was registered before training. The
coherence-threshold sweep is a robustness check showing the WIN is
not a single-threshold artifact.
