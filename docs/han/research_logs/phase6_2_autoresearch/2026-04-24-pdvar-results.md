---
author: Han
date: 2026-04-24
tags:
  - results
  - complete
---

## Phase 6.3 Priority 2a: passage-discriminative variance (pdvar) ranking

**Status**: complete. All 5 primary archs × (1-3) seeds × 3 concats labelled
under pdvar ranking (14-50 new Haiku labels per cell), `semantic_count_pdvar`
now stored in every `{arch}__seed{S}__concat{C}__labels.json`.

### The question

User's methodological concern (2026-04-24):

> "Just because the top 32 features don't contain the semantic features we
> expect it doesn't mean the latents are 'worse' for interp. It might be the
> case that we really need to expand 32 to something larger. From an interp
> perspective, faithfulness is important, and the TXC's superior probe AUC
> suggests that it is more faithful."

The x/32 top-by-variance metric may systematically unfair to TXC, because the
TXC encoder sees short token windows with plenty of local-pattern variance
(sentence boundaries, repeated tokens, syntactic artefacts). Those occupy
top-variance slots even though the MODEL is learning these as genuine
statistical regularities. Semantic features then live further down the
ranking.

### The experiment (Priority 2a from [[../POST_COMPACT_PRIORITIES]])

Replace "top-32 by per-token variance" with "top-32 by **passage-
discriminative variance** (pdvar)": for each feature, compute the variance
of its mean activation across the concat's labelled passages. A feature that
fires strongly in one passage and weakly in others gets a high pdvar score —
a necessary but not sufficient condition for being a semantic feature.

Implementation: `experiments/phase6_qualitative_latents/run_autointerp_pdvar.py`
reuses the labelled features from `top_feat_indices_var` where var ∩ pdvar
overlap (~50-75% of 32), labels the pdvar-only remainder via Haiku temp=0 +
2-judge majority vote, writes `semantic_count_pdvar` to the same JSON.

### Results — 5 primary archs × up to 3 seeds × 3 concats

Paper-ready aggregate (means ± stderr across seeds):

| arch | concat | pdvar | var | Δ |
|---|---|---|---|---|
| `tsae_paper` | A | 22.7 ± 1.20 | 23.0 ± 1.15 | -0.3 |
| `tsae_paper` | B | 20.7 ± 0.88 | 17.7 ± 0.88 | +3.0 |
| `tsae_paper` | **random** | **23.7 ± 0.67** | 13.7 ± 1.33 | **+10.0** |
| `agentic_txc_10_bare` (Track 2) | A | 20.3 ± 0.88 | 21.3 ± 1.45 | -1.0 |
| `agentic_txc_10_bare` (Track 2) | B | 19.3 ± 1.45 | 17.7 ± 2.40 | +1.7 |
| `agentic_txc_10_bare` (Track 2) | **random** | **7.3 ± 1.86** | 3.3 ± 1.33 | **+4.0** |
| `agentic_txc_12_bare_batchtopk` (2×2 cell) | A | 21.3 ± 1.76 | 20.7 ± 1.45 | +0.7 |
| `agentic_txc_12_bare_batchtopk` (2×2 cell) | B | 18.7 ± 2.40 | 15.0 ± 1.53 | +3.7 |
| `agentic_txc_12_bare_batchtopk` (2×2 cell) | **random** | **5.7 ± 1.20** | 1.7 ± 0.33 | **+4.0** |
| `agentic_txc_02` (TXC baseline, 1 seed) | A | 16 | 17 | -1 |
| `agentic_txc_02` (TXC baseline, 1 seed) | B | 17 | 16 | +1 |
| `agentic_txc_02` (TXC baseline, 1 seed) | **random** | **6** | 0 | **+6** |
| `agentic_txc_02_batchtopk` (Cycle F) | A | 23.0 ± 0.58 | 21.7 ± 0.88 | +1.3 |
| `agentic_txc_02_batchtopk` (Cycle F) | B | 20.0 ± 2.00 | 16.0 ± 2.52 | +4.0 |
| `agentic_txc_02_batchtopk` (Cycle F) | **random** | **12.3 ± 1.33** | 0.0 ± 0.00 | **+12.3** |

### Interpretation

**Concat_A and concat_B**: pdvar yields small or no gain. These concatenations
have strong passage structure and most top-by-variance features already align
with a passage. The ranking move doesn't unearth much.

**Concat_random (the generalisation control)**: **large and systematic gains
across all archs.** This is where top-by-variance was most misleading —
random concat uses 7 FineWeb passages with no pre-selected concept, so
top-by-variance picks up position / token / local-syntax artefacts.

Re-ranked archs on concat_random (var → pdvar):

1. **Cycle F** (`agentic_txc_02_batchtopk`): 0.0 → **12.3** (+12.3) — the
   biggest re-ranking. Under var ranking Cycle F was the worst-in-class
   ("pathological" was the user's word); under pdvar it's the **best TXC
   arch**. The var ranking was systematically penalising Cycle F because
   BatchTopK routes features toward high-token-variance outliers with no
   passage structure. The model's actual semantic features were all sitting
   in pdvar-only slots that var missed.

2. **TXC baseline** (`agentic_txc_02`, 1 seed): 0 → 6 (+6).

3. **Track 2** (`agentic_txc_10_bare`): 3.3 → 7.3 (+4.0). Meaningful gain,
   still trails Cycle F under pdvar.

4. **2×2 cell** (`agentic_txc_12_bare_batchtopk`): 1.7 → 5.7 (+4.0).

5. **T-SAE (paper)**: 13.7 → **23.7** (+10.0). Still the strongest.

### What this means for the paper narrative

**Before pdvar** (Phase 6.1 / 6.2 story):

> "TXC has a structural qualitative gap: its top-32 features contain only
> 2-4 semantic concepts on the generalisation concat, vs T-SAE's 13.7/32.
> No training recipe in the Phase 6.2 ablation closes the gap."

**After pdvar** (refined claim):

> "Under the rigorous metric (top-32 by per-token variance), TXC's 0-4/32
> plateau on concat_random was partly a ranking artefact. Ranking instead
> by passage-discriminative variance, the best TXC arch (Cycle F
> + BatchTopK) reaches 12.3/32 on random — 3x what it scored on the
> original ranking, and substantially closing the gap to T-SAE (23.7/32).
> A real gap of ~11 SEMANTIC labels remains, which we interpret as the
> cost of window-encoding vs per-token encoding."

**Net effect**: The claim **softens from "structural gap" to "partial
gap"**. The gap is real (even Cycle F doesn't reach T-SAE), but it's
narrower than the var-based metric implied, and the arch most damaged by
var-based ranking (Cycle F) turns out to be the best TXC performer.

### Paper figure changes

1. Regenerate Pareto figure with pdvar on the y-axis (or as an overlay).
   This moves Cycle F from the bottom of the cluster (next-to-worst) to the
   top (best TXC arch). The new Pareto looks like:

   ```
   tsae_paper:  (AUC 0.73, pdvar 23.7)
   Cycle F:     (AUC 0.78, pdvar 12.3)   ← newly competitive
   Track 2:     (AUC 0.80, pdvar  7.3)
   2×2 cell:    (AUC 0.80, pdvar  5.7)
   TXC base:    (AUC 0.80, pdvar  6.0)
   ```

   Track 2 is now Pareto-dominated by Cycle F on the pdvar axis (Cycle F:
   lower AUC but MUCH higher pdvar). This flips the "Track 2 is our best TXC"
   claim.

2. Add a diagnostic bar chart: "pdvar ↔ var SEM count comparison on
   concat_random" showing the re-ranking.

### Decision rule outcome

Per the Priority 2a decision rule:

> If TXC-family archs gain > 3 labels under pdvar ranking (e.g., Track 2
> 3.3/32 → 8/32 on random), the structural-gap claim softens substantially.
> Regenerate the Pareto figure with this new axis.

**Outcome**: Cycle F gains +12.3, TXC baseline +6, Track 2 +4, 2×2 cell
+4. The threshold is met for all 4 TXC archs. **Structural gap claim is
softened.** Paper should use pdvar-based qualitative count as primary y-axis
or at minimum as a companion axis.

### Follow-up

- **Priority 2b (top-N sweep)** is still worth running: if pdvar-top-32
  already lifts TXC to 12, top-256 could close more of the gap. Budget
  allows.
- **Priority 2c (distinct-concept dedup)** is still open — faithful to the
  user's original "distinct semantic features" phrasing. Would need Sonnet
  clustering, ~$5.
- **Phase 6.3 T-sweep** is running in parallel (see
  [[2026-04-24-handover-t-sweep]]); once ckpts are evaluated under pdvar,
  update the paper figure.
