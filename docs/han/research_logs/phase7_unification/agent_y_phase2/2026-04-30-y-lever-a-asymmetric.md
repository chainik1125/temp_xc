---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Y — Lever A: asymmetric within-window write weights

> **Headline**: weighted per-position write-back smooths the
> peak-vs-AUC trade-off but does not push the headline numbers above
> existing protocols. Right-heavy [0.5, 1.0] gives marginally higher
> AUC than uniform PP (+0.035 at AUC 1.5–3.0) but lower peak success
> at coh ≥ 1.5 (1.37 vs 1.53). Worth keeping as a tunable lever for
> future runs; not a new headline cell.

### Method

`intervene_paper_clamp_window_perposition.py` now accepts
`--position-weights {uniform, right-heavy, right-only, gaussian, custom}`.
Per-position write-back becomes a *weighted* average over window
positions ti ∈ [0, T):

    mean_delta[p] = Σ_k w[ti] · delta_W[k, ti, p−k]  /  Σ_k w[ti]
    where ti = p − k, restricted to k ∈ [0, K).

Uniform weights (the prior implementation) gives an unweighted
average. Right-heavy [0.5, 1.0] for T=2 puts twice the weight on the
right-edge within-window position (ti=1) than the left (ti=0).
Right-only [0, 1] = the right-edge protocol, but computed via the
per-position code path (sanity check).

Output routed to `steering_paper_window_perposition_<slug>` to avoid
clobbering existing per-position results.

### T=2 H8 sd=42 results (single seed)

| protocol | peak_unc | peak ≥ 1.5 | peak ≥ 1.75 | peak ≥ 2.0 | AUC(1.5–3.0) | AUC(1.75–3.0) |
|---|---:|---:|---:|---:|---:|---:|
| uniform PP (existing) | **1.57** | **1.53** | 1.53 | 1.53 | 0.707 | 0.613 |
| right-heavy [0.5, 1.0] | 1.40 | 1.37 | 1.37 | 1.37 | **0.742** | **0.647** |
| right-only [0, 1] (sanity) | 1.33 | 1.30 | 1.30 | 1.30 | 0.595* | 0.508* |
| right-edge protocol | 1.37 | 1.27 | 1.27 | 1.27 | 0.771 | 0.659 |

\* right-only AUC numbers are affected by a numerical artifact in the
trapezoidal interpolation when two strengths share the same mean coh
(s=2 and s=5 both at coh=2.233 in right-only). Per-strength curves
agree with right-edge to within grader noise (≤ 0.033 = 1/30 grade
units), confirming the sanity-check expectation.

#### Full per-strength curves

**Uniform PP** (existing — best peak at s=5/s=10):

| s_norm | succ | coh |
|---:|---:|---:|
| 0.5 | 0.233 | 2.733 |
| 1.0 | 0.267 | 2.533 |
| 2.0 | 0.467 | 2.100 |
| **5.0** | **1.533** | **2.200** |
| 10.0 | 1.567 | 1.367 |
| 20.0 | 0.867 | 1.000 |
| 50.0 | 0.100 | 0.900 |

**Right-heavy [0.5, 1.0]**:

| s_norm | succ | coh |
|---:|---:|---:|
| 0.5 | 0.233 | 2.733 |
| 1.0 | 0.367 | 2.633 |
| 2.0 | 0.533 | 2.100 |
| **5.0** | **1.367** | **2.100** |
| 10.0 | 1.400 | 1.400 |
| 20.0 | 0.714 | 0.964 |
| 50.0 | 0.100 | 0.967 |

**Right-edge protocol**:

| s_norm | succ | coh |
|---:|---:|---:|
| 0.5 | 0.233 | 2.800 |
| 1.0 | 0.300 | 2.433 |
| 2.0 | 0.400 | 2.233 |
| **5.0** | **1.267** | **2.200** |
| 10.0 | 1.367 | 1.300 |
| 20.0 | 0.633 | 0.900 |
| 50.0 | 0.067 | 0.900 |

### Interpretation

Right-heavy is a smooth interpolation between uniform PP and right-edge:

- **At low strength** (s=1, 2), right-heavy slightly *outperforms*
  uniform on success (s=1: 0.367 vs 0.267; s=2: 0.533 vs 0.467) while
  matching coherence. The asymmetric weighting reduces noise from the
  ti=0 prediction, which apparently is less aligned with the concept
  signal at low strength.
- **At peak strength** (s=5, 10), right-heavy *underperforms* uniform
  PP on success (s=5: 1.367 vs 1.533) — the dilution from the ti=0
  contribution boosts the uniform peak above what the right-heavy
  weighting allows.
- **At collapse strength** (s=20, 50), all protocols converge.

The result: right-heavy's peak success is lower but its overall AUC
across coh ≥ 1.5 is slightly higher than uniform PP (+0.035) — the
asymmetry trades peak height for breadth.

### Sanity check (right-only ≈ right-edge) — PASSED

Right-only [0, 1] should reproduce the right-edge protocol since for
T=2 it makes only the within-window-ti=1 contribution to each position,
which is exactly what right-edge writes. Per-strength comparison:

| s_norm | RO succ | RO coh | RE succ | RE coh | Δ succ |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.233 | 2.800 | 0.233 | 2.800 | +0.000 |
| 1.00 | 0.333 | 2.433 | 0.300 | 2.433 | +0.033 |
| 2.00 | 0.433 | 2.233 | 0.400 | 2.233 | +0.033 |
| 5.00 | 1.300 | 2.233 | 1.267 | 2.200 | +0.033 |
| 10.00 | 1.333 | 1.300 | 1.367 | 1.300 | −0.033 |
| 20.00 | 0.633 | 0.967 | 0.633 | 0.900 | +0.000 |
| 50.00 | 0.067 | 0.900 | 0.067 | 0.900 | +0.000 |

|Δ succ| ≤ 0.033 = 1/30 grade units everywhere — within Sonnet
grader sampling noise. Code paths agree.

### Why this isn't the new headline

Lever A's expected ceiling is "between PP and RE", but our existing
unified Pareto already has both PP and RE multi-seed verified for
T=2 H8. The PP variant wins peak at coh ≥ 1.5; the RE variant wins
peak at coh ≥ 1.75. Neither is matched — they're the two ends of a
trade-off curve that Lever A interpolates between.

The previous headline (T=2 H8 PP, peak ≥ 1.5 = 1.400) is unaffected.
The multi-coh-threshold sweep (`2026-04-30-y-coh-threshold-sweep.md`)
is the GIGABRAIN reframe — Lever A is supplementary.

### Files

- Implementation: `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_perposition.py`
  (new `parse_position_weights()` helper + `--position-weights` CLI)
- Right-heavy generations + grades: `results/case_studies/steering_paper_window_perposition_rightheavy/txc_h8_t2_kpos20_shifts2/`
- Right-only generations + grades: `results/case_studies/steering_paper_window_perposition_rightonly/txc_h8_t2_kpos20_shifts2/`

### Next steps for Lever A (if pursued)

- **Multi-seed verify**: run sd=1, sd=2 to verify the AUC lift
  generalises beyond sd=42. ~32 min × 2 seeds = 64 min.
- **Try gaussian for T≥3**: at T=2, gaussian = uniform (symmetric);
  at T=3, [0.41, 1.0, 0.41]; at T=5, [0.28, 0.73, 1.0, 0.73, 0.28].
  Could provide a different trade-off than right-heavy.
- **Multi-feature steering** (Lever B): orthogonal to Lever A;
  potentially much larger effect size.

### Verdict

Lever A is a working knob with a small but consistent effect. It does
NOT close the unconstrained-peak gap with T-SAE, but the GIGABRAIN
reframe makes that gap a non-issue (T-SAE's unconstrained peak is at
incoherent text). Recommend defer Lever A in favour of writing up the
multi-coh-threshold reframe + multi-seed-verifying the strongest
single-seed cells in that sweep.
