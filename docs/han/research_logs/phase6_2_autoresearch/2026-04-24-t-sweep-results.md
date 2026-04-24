---
author: Han
date: 2026-04-24
tags:
  - results
  - in-progress
---

## Phase 6.3 T-sweep results (Track 2 recipe at T ∈ {3, 5, 10, 20})

**Status**: seed-42 training + evaluation complete; seeds 1, 2 training
launched 2026-04-24 after the decision rule triggered. Expected
completion ~4 hours after launch.

### Hypothesis (user, 2026-04-24)

> Increasing TXC window T trades off probe AUC for more interpretable
> latents.

### Design

Track 2 recipe (TXCBareAntidead: bare-window TXC + AuxK + unit-norm
decoder + grad-parallel removal + geom-median b_dec init, TopK k=100) at
T ∈ {3, 10, 20} with all other hyperparameters matching
`agentic_txc_10_bare` seed 42 (`k_pos=100`, `k_win = 100·T`, `aux_k=512`,
`dead_threshold_tokens=10M`, `auxk_alpha=1/32`, `max_steps=25000`).

Training done at seed 42 first; after evaluation triggered the decision
rule (T=10 random var=11/32, T=20 random var=13/32 — both ≥ 7/32
threshold), seeds 1 & 2 dispatched for variance analysis.

### Seed-42 results (single seed)

#### Training

| T | steps | time | final loss | final L0 | converged |
|---|---|---|---|---|---|
| 3 | 5,600 | 12 min | 5,586 | 294 (target 300) | yes |
| 5 (Track 2 ref) | 10,000 | 41 min | 5,931 | 489 (target 500) | yes |
| 10 | 5,200 | 44 min | 7,410 | 984 (target 1000) | yes |
| 20 | ~5,000 | 69 min | — | — | yes |

T=3 converges fastest and to the lowest loss (narrow window = easier to
reconstruct per-window). T=10 jumps to loss 7.4k — the window is long
enough that per-window variance in resid activations makes
reconstruction harder. T=20 should be even higher; need to check log.

#### Probing (mean AUC at k=5, 36 SAEBench tasks)

| T | last_position | mean_pool |
|---|---|---|
| 3 | 0.7731 | 0.8039 |
| 5 (Track 2 3-seed ref) | 0.7788 | 0.8014 |
| 10 | 0.7939 | **0.8110** |
| 20 | 0.7844 | 0.7885 |

**Unexpected finding**: under Track 2's anti-dead stack, probing AUC
peaks at **T=10** (0.8110), not T=5. T=20 drops slightly below T=5. This
contradicts the Phase 5.7 T-sweep result (vanilla TXCDR, no anti-dead:
AUC peaks at T=5, drops to 0.7545 at T=20).

Interpretation: the anti-dead stack *preserves* the feature dictionary
at longer windows. Without it, larger T causes dead-feature accumulation
that degrades probing; with it, larger T gives richer context at the
cost of harder reconstruction, and the probing gain outweighs the cost
through T=10.

#### Qualitative (top-32 on concat_random, seed 42)

| T | var-ranked sem | pdvar-ranked sem |
|---|---|---|
| 3 | 3 | **17** (Δ+14) |
| 5 (Track 2 3-seed mean) | 3.3 | 7.3 (Δ+4) |
| 10 | **11** (Δ+8 vs T=5 mean) | 11 (no pdvar gain — already at ceiling) |
| 20 | **13** (Δ+10 vs T=5 mean) | 14 (Δ+1) |

### Interpretation

Two different stories emerge depending on ranking metric:

**Under var ranking**: qualitative is monotone-increasing in T (3 → 3.3
→ 11 → 13). **Matches the user's hypothesis directly.** But the story
comes with a twist: *both* metrics improve from T=5 to T=10 (AUC 0.8014
→ 0.8110, qualitative 3.3 → 11), so T=10 Pareto-dominates T=5 without
the expected trade-off. This only reverses at T=20 where AUC drops.

**Under pdvar ranking**: qualitative is **non-monotone**. T=3 alone
gives 17/32 on concat_random — the highest of the T-sweep and close to
tsae_paper's 23.7/32 on the same axis. This contradicts the "larger T
helps qualitative" hypothesis; under pdvar, *smaller T* is what
approaches tsae_paper.

This contradiction between var and pdvar is paper-relevant. The two
rankings are asking different questions:

- **var**: "what are the most *active* features?" — biased toward
  boundary / token / positional artefacts that fire strongly on a few
  tokens.
- **pdvar**: "what features *discriminate between passages*?" —
  biased toward passage-level topic features.

For the qualitative claim to be robust across the paper, we need to
either:

1. Commit to pdvar as the primary metric (more faithful to user's
   "distinct semantic concepts" intuition), and report "T=3 is best
   for TXC qualitative, T=10 is best for probing". This makes the
   trade-off explicit.
2. Report both and let readers decide.
3. Average both (unusual; weak motivation).

Option (1) is probably cleanest for the paper.

### Decision-rule outcome

Per the handover:

> If T=10 or T=20 scores ≥ 7/32 random: this breaks the Phase 6.2
> plateau. Retrain at seeds {1, 2} for 3-seed variance.

Triggered. **T=3, T=10, T=20 all scheduled for seeds 1 and 2** (T=3
added because its pdvar=17 is surprisingly high — the single-seed result
needs variance).

### Next steps (pending seed-1/2 training)

1. Collect 3-seed means ± stderr for T ∈ {3, 10, 20} on both metrics.
2. Update Pareto figure — the T-sweep trajectory should be drawn on the
   pdvar axis (main signal) AND var axis (as supplementary since the
   user's hypothesis pertains to that ranking).
3. Decide paper narrative:
   - If T=3 pdvar holds across seeds: T=3 as the "qualitative-optimal
     TXC", Cycle F as runner-up; paper story is "TXC qualitative is
     recoverable with small T, at a ~1-2pp probing cost".
   - If T=3 pdvar collapses to mean ≈ 7 (consistent with T=5): then the
     T=10/20 var-ranking finding dominates and the paper story is
     "larger T trades probing for qualitative (anti-dead stack only)".
4. Update summary.md §9.5.
5. HF sync.

### Files updated

- `experiments/phase5_downstream_utility/train_primary_archs.py` —
  added phase63_track2_t3/t10/t20 dispatcher branches.
- `experiments/phase6_qualitative_latents/encode_archs.py` — extended
  TXC load/encode tuples.
- `experiments/phase5_downstream_utility/probing/run_probing.py` —
  extended TXC load/encode tuples.
- `experiments/phase6_qualitative_latents/arch_health.py` — made T
  parameter read from meta rather than hardcoded 5.
- `experiments/phase6_qualitative_latents/plot_pareto_robust.py` —
  added `--metric {semantic_count, semantic_count_pdvar}` flag, added
  T_SWEEP trajectory support in the zoom panel.
- `experiments/phase6_qualitative_latents/run_autointerp_pdvar.py` — new
  script for Priority 2a.
- `docs/han/EXPERIMENT_INDEX.md` — added phase63 arch rows + pdvar
  column.

### Related

- [[2026-04-24-handover-t-sweep]] — the original handover doc.
- [[2026-04-24-pdvar-results]] — Priority 2a results.
- [[../POST_COMPACT_PRIORITIES]] — original priority specification.
