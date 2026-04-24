---
author: Han
date: 2026-04-24
tags:
  - proposal
  - todo
---

## Post-compact agent priorities (read this first)

**Target audience**: a fresh agent picking up after the 2026-04-24
context compact. Read this doc, then [[EXPERIMENT_INDEX]] in
`docs/han/`, then whichever phase handover is relevant.

Three priorities from the user, in decreasing order. **Priority 1 is
the one the user flagged first and wants landed for NeurIPS**.
Priorities 2 and 3 are the methodological concerns that, if
correct, could rehabilitate the TXC family's qualitative story.

### Priority 1 — T-sweep: does larger T trade probing for qualitative?

**Hypothesis (user)**: increasing TXC window T trades probe AUC for
more interpretable latents.

**What we know**:

- Phase 5.7 T-sweep (probing-only, vanilla TXCDR): AUC is concave in
  T, peaks at T=5 (mean_pool 0.8064), drops to T=20 at 0.7545. Large
  T hurts probing. ✓ half the hypothesis confirmed.
- Phase 6 / 6.1 / 6.2: all qualitative data is at T=5. 10 TXC
  variants, all plateau at 2-4/32 random. Whether larger T gives
  qualitative gain is **UNTESTED**.

**Handover**:
[[research_logs/phase6_2_autoresearch/2026-04-24-handover-t-sweep]] —
fully written execution plan. Train Track 2 recipe at T ∈ {3, 10, 20}
on seed=42, evaluate on A/B/random + probe, then decide based on
results whether to seed-variance the winner.

**Budget estimate**: ~2.5 hr GPU + ~$0.3 API. Fits comfortably in a
single session.

### Priority 2 — rethink the qualitative metric

**User's concern (verbatim)**: *"Just because the top 32 features
don't contain the semantic features we expect it doesn't mean the
latents are 'worse' for interp. It might be the case that we really
need to expand 32 to something larger. From an interp perspective,
faithfulness is important, and the TXC's superior probe AUC suggests
that it is more faithful. So wouldn't it be better for the
qualitative metric to be the 'number of distinct semantic features
that appear in the top X' rather than 'the fraction of the top 32
which are semantic'."*

**Why this matters**: Phase 6.2 found the TXC family plateaus at
2-4/32 semantic on concat_random, while `tsae_paper` is at ~13/32.
We concluded the gap was *structural*. But the ratio-based metric
systematically disadvantages architectures whose top-by-variance
features are format / boundary patterns — which the TXC family
might do BECAUSE it's faithful (high-variance features on uncurated
text genuinely are sentence-boundary patterns, which are real
statistical regularities). Semantic features could still exist
deeper in the variance ranking, and we'd miss them.

**Proposed redesign**:

1. **Top-N sweep instead of fixed N=32.** Rank features by variance
   as before, but compute the SEMANTIC count at N ∈ {32, 64, 128,
   256, 512}. The resulting curves let us see whether TXC archs
   have more semantic features further down. If TXC's curve
   eventually matches `tsae_paper`'s at N=256, the "plateau" is
   actually "semantic features at higher rank".

2. **Distinct-concept count.** Rather than "is this feature
   semantic?", cluster the labels by semantic similarity (cheap
   sentence-embedding similarity on the 256-char label strings) and
   count distinct clusters. "medical education" and "medical
   publishing" collapse to one concept. Distinguishes archs that
   find 50 variations of one concept vs 50 distinct concepts.

3. **Variance-weighted semantic fraction.** Weight each feature's
   SEMANTIC contribution by its per-token variance. TXC's high-
   variance boundary features count less; low-variance semantic
   features count more. Less sensitive to the ranking cutoff.

**Implementation complexity**: (1) is trivial — change `N=32` to
`N=512` in `run_autointerp.py` and re-run. Cost: 512 × 3 Haiku calls
per cell × 30 cells = ~$50. Over budget.

**Cheaper version**: rank-swept on the same 32 ALREADY labelled
features in existing cells. Not possible — if we want top-100 sem
count we need to label features 33-100.

**Smartest version**: re-rank existing top-32 cells by
passage-discriminative variance instead of per-token variance (the
"diagnostic" we already computed — `semantic_count_pdvar`). It's
already in the JSON. That metric may be kinder to TXC features that
are discriminative across passages but not high-variance.

**Recommended action for the post-compact agent**:

1. Check what `passage_coverage_count` and
   `passage_coverage_entropy` already say for the TXC family.
   Currently in summary.md §9.5's table; if TXC-cluster archs have
   coverage 7/7 on random with high entropy, that's positive
   evidence of distinct-concept spread already in the data.
2. Extend `run_autointerp.py` to support N=128 or N=256 and re-run
   on 3 target archs (Track 2 seed=42, 2×2 cell seed=42,
   tsae_paper seed=42). Cost: 3 cells × 256 × 3 calls = 2300 calls
   ≈ $7. Within budget if it's the ONLY extra API cost.
3. Plot "SEMANTIC count vs N" for those three. If TXC curves
   converge with tsae_paper past N=100, the paper story changes
   from "TXC fails qualitative" to "TXC's semantic features live
   deeper in the ranking".

### Priority 3 — faithfulness via ablation

**User's suggestion**: *"To test faithfulness, we can take
inspiration from what the MatryoshkaSAE paper did: ablate a latent
and see its effect on a probe. But that could be a separate thing."*

**Why this matters**: Probing AUC tells us the latent space is
informative for a downstream task, but NOT whether specific labeled
features actually causally encode what their labels claim. If
Track 2's "political ideology" feature's label is accurate, ablating
that feature should damage a political-leaning probe specifically.
If ablating it damages every probe equally, the label is a
confabulation — the feature is polysemantic / uninterpretable in
content.

**Experimental design**:

1. For each arch × seed, pick the top-k SEMANTIC-labelled features
   (k = 5 or 10).
2. For each such feature: recompute probing AUC on the Phase 5
   benchmark with the feature ZEROED at encode time (intervention).
3. Compare AUC drop on: (a) the task whose domain matches the
   feature's label (e.g., "Russian history" feature → WSC task), vs
   (b) average of all other tasks. Faithful feature: big (a) drop,
   small (b) drop. Non-faithful: uniform drop.

**Implementation**: needs hook in `run_probing.py` `_encode_for_probe`
to zero a specified feature index before flattening. ~20 LoC.

**Cost**: small. Each ablation probe is 36 tasks × ~1s each = 1 min.
For 5 archs × 10 features × 1 aggregation = 50 probing runs × 1 min
= ~50 min GPU + zero API.

**This is a full subsequent phase (Phase 6.4)**. Not blocking for
NeurIPS unless Priority 2 doesn't rehabilitate the TXC qualitative
score — in which case this becomes the backup evidence of "TXC
features are faithful even if they're not top-ranked by variance".

### Ordering recommendation

If there's only time for ONE of the three before the NeurIPS
submission: **Priority 2 (top-N sweep)**. A single figure showing
"SEMANTIC count vs N" for Track 2, 2×2 cell, tsae_paper would
either:

- **Confirm the structural gap** (TXC curves flatten well before
  tsae_paper's) → strengthens the Phase 6.2 "structural plateau"
  claim.
- **Show the gap closes** at larger N → RECONTEXTUALISES the paper
  narrative. TXC has as many semantic features as T-SAE, just at
  lower variance rank. This is a BIG paper-positive finding.

Priority 1 (T-sweep) is also cheap and has the same "negative result
also publishable" property.

Priority 3 (ablation) is richer science but out of scope for a
2-week submission deadline unless we commit hard.

### What to NOT do

- **Don't retrain anything already in the index.** Check
  [[EXPERIMENT_INDEX]] first — we have 14+ archs × up-to-3 seeds
  each. Any "let me just try X" that's already been run is wasted.
- **Don't touch the Phase 5 benchmark.** Phase 5 is the paper's
  probing anchor; Phase 5 agent (on `han` branch) is keeping that
  slate stable.
- **Don't conflate "top-32 sem count" with "qualitative quality".**
  Priority 2 exists because we're not sure the metric is measuring
  what we want.

### Handover state

- Code: `han-phase6` HEAD `be47c34` (as of 2026-04-24 after compact).
- All ckpts, z_caches, labels, probing jsonl, figures synced to
  `han1823123123/txcdr{,-data}` on HF.
- Phase 5 agent operates on `han` branch. This priorities doc lives
  on `han-phase6` but can be cherry-picked.
- `scripts/hf_sync.py --go` is idempotent; run after every new
  result.
