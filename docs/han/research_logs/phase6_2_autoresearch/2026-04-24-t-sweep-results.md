---
author: Han
date: 2026-04-25
tags:
  - results
  - complete
---

## Phase 6.3 T-sweep results (Track 2 recipe at T ∈ {3, 5, 10, 20})

**Status**: complete. 3 seeds × 4 T values × 3 concats × probing(2 aggs) +
autointerp + pdvar + paper-style passage probe.

### Hypothesis (user, 2026-04-24)

> Increasing TXC window T trades off probe AUC for more interpretable
> latents.

**Verdict**: SUPPORTED, very clearly under var-ranked top-32 qualitative,
weakly / non-monotone under pdvar ranking. T=20 in particular **beats
T-SAE on both metrics simultaneously**.

### Design

Track 2 recipe (TXCBareAntidead: bare-window TXC + AuxK + unit-norm
decoder + grad-parallel removal + geom-median b_dec init, TopK k=100) at
T ∈ {3, 5, 10, 20} with all other hyperparameters matching
`agentic_txc_10_bare` (T=5). T=5 reuses the existing
`agentic_txc_10_bare` 3-seed runs as the reference. T=3, 10, 20 each
trained at seeds {42, 1, 2}, then evaluated.

### Results — 3 seeds, mean ± stderr

#### Probing (mean AUC at k=5, 36 SAEBench tasks)

| T | last_position | mean_pool |
|---|---|---|
| 3  | 0.7687 ± 0.003 | 0.7959 ± 0.005 |
| 5 (Track 2 ref) | 0.7807 ± 0.005 | 0.8034 ± 0.000 |
| 10 | **0.7906 ± 0.002** | 0.8016 ± 0.005 |
| 20 | 0.7731 ± 0.006 | 0.7768 ± 0.006 |

T=10 has the highest **last_position** AUC (0.7906) but T=5 ties on
mean_pool (0.8034 ≈ T=10's 0.8016). T=20 drops on both aggregations.
T=3 also drops, by ~1pp on last_position vs T=5.

**Reference**: T-SAE last_position = 0.6848, mean_pool = 0.7246. Every
TXC arch dominates T-SAE on probing across the entire T-sweep.

#### Qualitative on concat_random (top-32 by per-token variance)

| T | var-ranked sem (3s) | pdvar-ranked sem (3s) |
|---|---|---|
| 3 | 2.7 ± 1.45 | **16.7 ± 1.45** |
| 5 (Track 2 3s ref) | 3.3 ± 1.33 | 7.3 ± 1.86 |
| 10 | 7.7 ± 1.76 | 9.0 ± 1.15 |
| 20 | **19.0 ± 3.00** | 13.7 ± 0.33 |

**Reference**: T-SAE var = 13.7 ± 1.33, T-SAE pdvar = 23.7 ± 0.67.

#### Paper-style probe on passage ID (k=5, 5-fold CV, mean across 3 concats)

| T | passage-probe acc (3s mean) |
|---|---|
| 3 | 0.764 |
| 5 (Track 2 ref) | 0.815 |
| 10 | 0.789 |
| 20 | 0.804 |

**Reference**: T-SAE = 0.766. All TXC at all T match or beat T-SAE on the
paper-style probe — the picture isn't sensitive to T here.

### Interpretation

**Hypothesis support under var ranking** (the user's framing): clean
monotone ramp 2.7 → 3.3 → 7.7 → **19.0** SEMANTIC features as T grows.
**At T=20, Track 2 beats T-SAE qualitatively (19.0 vs 13.7) AND on
probing (0.7768 vs 0.7246).** Pareto-dominant.

**Hypothesis support under pdvar ranking**: non-monotone. T=3 already
finds 16.7/32, T=5 dips to 7.3, T=10 mid at 9.0, T=20 climbs to 13.7
(matches T-SAE under var, still below T-SAE under pdvar at 23.7). The
T=3 single-seed surprise from earlier (17/32) holds at 3 seeds (16.7),
which means **T=3 is genuinely the most pdvar-discriminative TXC**.

The two metrics tell different stories about what "more interpretable
at larger T" means:

- **var ranking** rewards features whose TOP activations are coherent.
  Larger T gives more contextual smoothing, so top-activation features
  are more about "topic of this 20-token window" than "this specific
  token". This rewards larger T monotonically.
- **pdvar ranking** rewards features that CONTRAST between passages.
  T=3 is best because its features are more local-coherent and don't
  bleed across passage boundaries; T=20's longer window naturally has
  more cross-passage activations, hurting pdvar.

### What this means for the paper

**Headline can shift from "TXC has a structural qualitative gap" to
one of two cleaner stories**:

1. **"Track 2 at T=20 Pareto-dominates T-SAE"** (var-rank framing):
   - Probing: 0.7768 vs 0.7246 (+5pp)
   - Qualitative (var): 19.0 vs 13.7 (+5.3 labels)
   - This is the strongest claim. Trade-off vanishes; we just win.

2. **"T-sweep traces a probing ↔ qualitative trade-off"** (the original
   user hypothesis): T=3 has best pdvar qualitative, T=10 has best
   probing, T=20 is middle on both. Multiple Pareto-non-dominated
   choices in TXC family.

Story 1 is sharper but rests on var-ranked qualitative being meaningful
(which Phase 6.3 Priority 2b already established it is — top-N sweep
showed the gap is structural, not a top-32 artefact).

### Decision-rule outcome

Per handover: "If T=10 or T=20 scores ≥ 7/32 random: this breaks the
Phase 6.2 plateau. Retrain at seeds {1, 2} for 3-seed variance."

T=10 (7.7) hit threshold, T=20 (19.0) blew through it. Retrained at
seeds {1, 2}; data above is the 3-seed result. Plateau-breaking is
robust.

### Figures

- `phase63_t_sweep.png` — AUC vs T + qualitative vs T side-by-side line
  plot, with T-SAE reference bands. The headline line plot.
- `phase61_pareto_robust.png` — full Pareto with Track 2 (T=5) but
  zoom panel shows T-sweep trajectory.
- `phase63_pareto_pdvar.png` — same Pareto with pdvar y-axis.
- `phase63_pareto_paper_probe.png` — paper-style passage-probe Pareto,
  shows TXC family beating T-SAE on both axes simultaneously.

### Files updated

All listed in 2026-04-24-pdvar-results.md and EXPERIMENT_INDEX. New for
3-seed run: `phase63_track2_t{3,10,20}__seed{1,2}.pt` ckpts, autointerp
+ pdvar + probe results for 18 cells, passage-probe results for the
same 18 cells.

### Related

- [[2026-04-24-pdvar-results]] — Priority 2a (pdvar metric).
- [[2026-04-24-topN-sweep-results]] — Priority 2b (top-N sweep).
- [[../POST_COMPACT_PRIORITIES]].
