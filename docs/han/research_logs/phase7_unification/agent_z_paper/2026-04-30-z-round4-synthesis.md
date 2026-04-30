---
author: Han
date: 2026-04-30
tags:
  - results
  - complete
---

## Z round 4 — rank-routed + shared-encoder SubseqH8 variants

> Hill-climb cells designed to break the 5090 OOM ceiling on the
> original SubseqH8 at large T_max. Both fit; both **lose** at PAPER
> k=20 vs the TXC family leader. Negative architectural finding:
> position-specificity is load-bearing for cross-token tasks.

### Motivation

Z's prior round (round3) hit a hard 5090 wall: SubseqH8 with T_max ≥ 14
+ paper-canonical b=4096 OOMs at >32 GB. Root cause: the encoder forward
stores activations for ALL T_max positions × #shifts, dominating memory.

Han proposed two variants to break the ceiling (2026-04-29):

- **Per-sampled-slot** (`SubseqRankedH8`): t_sample-many encoder slabs
  instead of T_max-many. Slot-k handles position-rank-k from the random
  sample. At probe, partition T_max into t_sample equally-spaced
  positions and route by rank.
- **Shared-encoder** (`SubseqSharedH8`): single (d_in, d_sae) encoder.
  Window pre-act = `W_enc @ sum-pooled(window)`. Multi-distance InfoNCE
  is the temporal signal. Trivially fits any T_max.

Both keep the H8 stack: matryoshka H/L recon + multi-distance contrastive
+ anti-dead AuxK + decoder-parallel grad removal.

### Memory + step time on 5090 (real cache, b=4096)

| variant | params | peak GB | step ms | 8000 steps |
|---|---:|---:|---:|---|
| original SubseqH8 T_max=20 t_sample=5 | 1.7 B | OOM (>32 GB) | — | — |
| **`SubseqRankedH8` T_max=20 t_sample=5** | 425 M | 24.3 GB | 745 | ~95 min |
| **`SubseqSharedH8` T_max=20** | 85 M | 18.4 GB | 261 | ~35 min |

Both variants fit comfortably with the L=64 cache slice (X-approved
hill-climb deviation; `[:, -64:, :]` direction; see
`2026-04-30-z-x-cache-slice-question.md`).

### Headline result — PAPER k=20 seed=42 base

| arch_id | mean AUC_FLIP | Δ vs TXC leader (0.9131) | rank | verdict |
|---|---:|---:|---|---|
| `tsae_paper_k500` (overall leader) | 0.9151 | +0.0020 | 1 | leader |
| `txc_bare_antidead_t5` (TXC leader) | 0.9131 | 0 | 2 | TXC leader |
| `hill_subseq_h8_T12_s5` (Z V1 1-seed) | 0.9126 | −0.0005 | 3 | competitive |
| **`hill_z_ranked_T20_s5`** (Z R4-A) | **0.8917** | **−0.0214** | far below | **LOSE** |
| **`hill_z_shared_T20`** (Z R4-B) | **0.8211** | **−0.0920** | far below | **LOSE (worse)** |

**Ranked LOSES by 0.0214 — well outside X's 0.005 promotion bar.**
Don't retrain at L=128.

### Per-task ranked-variant breakdown (k=20, PAPER set)

| task | AUC_FLIP | regime |
|---|---:|---|
| europarl_nl | 0.9993 | language ID — easy |
| europarl_fr | 0.9975 | |
| europarl_de | 0.9972 | |
| amazon_reviews_sentiment_5star | 0.9718 | sentiment — easy |
| ag_news_scitech | 0.9558 | topic |
| bias_in_bios_set3_prof9 | 0.9551 | profession |
| bias_in_bios_set1_prof2 | 0.9488 | profession |
| ag_news_business | 0.9388 | topic |
| bias_in_bios_set3_prof20 | 0.9384 | profession |
| bias_in_bios_set1_prof11 | 0.9085 | profession |
| amazon_reviews_cat5 | 0.8997 | review category |
| github_code_python | 0.8960 | code language |
| amazon_reviews_cat3 | 0.8439 | review category |
| github_code_java | 0.8327 | code language |
| **wsc_coreference** | **0.6439** | **cross-token (REGRESSION)** |
| **winogrande_correct_completion** | **0.5408** | **cross-token (NEAR-RANDOM)** |

Single-token classification (sentiment, language ID, topic) is fine —
topics like europarl + sentiment land at AUC > 0.97 even with
rank-routing. The damage concentrates on **cross-token tasks
(coreference + winograd-style)** where the model needs to relate
information across multiple positions in the input.

The TXC-family leader at the same tasks: winogrande_correct_completion ≈
0.78, wsc_coreference ≈ 0.85 (per X's per-task breakdown). The ranked
variant drops these to ~random/near-random.

### Architectural diagnosis

The rank-routing maps probe-time positions to slot encoders by their
*rank in the slot ordering*, not by absolute position. At probe time
we partition T_max into t_sample equally-spaced positions and assign
each to its rank-slot. This is the deterministic analogue of training,
where the slot mapping is randomised each step.

The damage to cross-token tasks suggests:

1. **Rank-mapping is too coarse.** With T_max=20 t_sample=5, each slot
   encoder averages over a 4-position chunk during training — it can't
   develop position-specific selectivity within a chunk.
2. **Cross-token tasks rely on inter-position structure** (e.g.,
   "winogrande resolves a pronoun in position k by attending to
   positions <k"). The shared-across-rank encoder loses this.
3. The original SubseqH8 (T_max-many slabs) preserves per-position
   selectivity. The rank-variant trades that for memory savings — and
   the trade is bad on this metric.

### Per-task shared-variant breakdown (k=20, PAPER set)

| task | AUC_FLIP | regime |
|---|---:|---|
| bias_in_bios_set3_prof20 | 0.9150 | profession |
| amazon_reviews_sentiment_5star | 0.8857 | sentiment |
| ag_news_scitech | 0.8857 | topic |
| ag_news_business | 0.8818 | topic |
| bias_in_bios_set3_prof9 | 0.8756 | profession |
| europarl_nl | 0.8607 | language ID |
| bias_in_bios_set1_prof11 | 0.8550 | profession |
| github_code_python | 0.8529 | code |
| bias_in_bios_set1_prof2 | 0.8423 | profession |
| europarl_de | 0.8424 | language ID |
| europarl_fr | 0.8257 | language ID |
| amazon_reviews_cat5 | 0.8202 | category |
| github_code_java | 0.8048 | code |
| amazon_reviews_cat3 | 0.7366 | category |
| **wsc_coreference** | **0.6286** | cross-token |
| **winogrande_correct_completion** | **0.6241** | cross-token |

Shared regresses **across the board**, not just on cross-token tasks
— even europarl (which the ranked variant aced at 0.997) drops to
0.83-0.86. Single-encoder + sum-pooled-window discards both per-position
selectivity AND per-token classification information.

### Architectural diagnosis (both variants)

The lesson is uniform: **probe-time encoding requires per-position
resolution, and any compression of T-axis slots in the encoder
destroys it**. The trade is a sharp loss/memory cliff, not a smooth
trade-off:

- T_max-many slabs (canonical SubseqH8): per-position selectivity ✓ — fits H200, OOMs 5090 at T_max ≥ 14
- t_sample-many slabs (ranked variant): selectivity ~ rank-quantile only — fits 5090 at T_max=20, but loses 0.02 AUC. Cross-token tasks particularly broken.
- 1 slab (shared variant): no positional info — loses 0.09 AUC. All tasks degrade.

The cross-token tasks (winogrande, wsc) are the most informative
diagnostic: they require resolving a token in position k by attending
to positions <k. The original SubseqH8 + matryoshka stack does this
via the per-position encoder slabs feeding into the matryoshka H/L
prefix that participates in the InfoNCE contrastive structure. Compress
the slabs and the contrastive signal can no longer disambiguate
positions.

### Conclusion

- **Both Z R4 variants are NOT leaderboard contenders.** Don't propose
  to `paper_archs.json`. Don't retrain at L=128.
- The intended memory savings (5090-fit at large T_max) cost more in
  AUC than the architecture can justify.
- **The 5090 ceiling on T_max-many SubseqH8 stands**: no clean way to
  get T_max ≥ 14 + paper-canonical b=4096 without compressing slots
  (which costs AUC) or deferring to H200 (which we can't afford).

### Next moves (ranked roadmap)

Ranked by remaining promise; only A is being pursued autonomously this
session:

**A. Multi-seed V1 — `SubseqH8` T_max=12 t_sample=5 at seed=1, seed=2.**
V1 sits at 0.9126 (1-seed), only 0.0005 below TXC leader. The TXC leader
is 3-seed mean. If V1's 3-seed mean ≥ 0.9131, **V1 ties or beats by
sheer architectural fit**.

**UPDATE 2026-04-30: V1 does NOT fit 5090 at any cache size.** Z
smoke-tested V1 (original SubseqH8 T_max=12) on 5090:

- L=64 cache (7 GB): OOM at first Adam step (Adam state alone ≈ 12 GB
  on top of 4 GB params + 7 GB cache + 6 GB activations ≈ 29 GB +
  Adam = 41 GB > 32 GB). Hard CUDA OOM.
- L=32 cache (3.5 GB): peak max_memory_allocated reports 35.04 GB on
  the 32 GB device. The program runs (PyTorch falls back to host-memory
  thrashing) but step time degrades to ~14 sec/step → 8000 steps would
  take ~31 hr. Not viable.

V1's 1.0 B parameters (T_max-many encoder slabs at T_max=12) push the
W+Adam footprint to ~16 GB before any activation memory. The 5090's
32 GB simply doesn't have headroom.

**V1 multi-seed is H200-only** (or A40 with the right slicing). Z is
unable to pursue this direction on the 5090.

**B. TXCBareAntidead T-sweep gap-fill (T ∈ {4, 6, 7, 8, 12}).**
The k=20 leader is `txc_bare_antidead_t5`. Existing T values: 5, 10, 20.
Each missing T fits 5090 easily (~25 min per cell, ~2 hr total). If a
non-canonical T (e.g., T=6 or T=7) beats 0.9131, that's a clean win.

**C. SubseqH8 + alternative shift sets at T_max=12.**
V1 used auto-shifts (1, 3, 6). Try `(1, 6)` only (drop 3 — possibly a
"dead-zone" mid-range shift) or `(1, 2, 6)`. T_max=12 fits 5090.

**D. Defer to H200 (not feasible here).**
SubseqH8 T_max ∈ {14, 16, 20} — confirmed OOM on 5090. H200 only.

Z is proceeding with **B** (TXCBareAntidead T-sweep gap-fill) since
**A is blocked by 5090 hardware**. T=8 fires first (closest to T=5 leader,
matches the k=5 leader's T value). Then T=6, T=7, T=4, T=12 if budget
permits. ~25 min/cell training + ~10 min probe each = ~3 hr for all 5.

### Architectural takeaway for the paper

The negative findings in R4 sharpen one of the paper's claims:
**per-position encoder slabs are load-bearing for cross-token tasks**.
Compress them and AUC drops 0.02-0.09. This is a quantitative
robustness check on the SubseqH8 design choice — not just a
methodological note.

### Files / commits

- `src/architectures/phase7_subseq_z_variants.py` (`86faa68`)
- `tests/test_phase7_subseq_z_variants.py` (`86faa68`)
- `experiments/phase7_unification/hill_climb/_run_one_z_variant.py` (`618f11c`)
- `experiments/phase7_unification/hill_climb/_mem_smoke_z_variants.py` (`91210d7`)
- `experiments/phase7_unification/run_probing_phase7.py` patch (`67c50a0`)
- HF: `han1823123123/txcdr-base/ckpts/hill_z_ranked_T20_s5__seed42`,
  `han1823123123/txcdr-base/ckpts/hill_z_shared_T20__seed42`

### Open thread

Awaiting shared variant probing (~30 min). Will update this doc
in-place with shared verdict. Then synthesize round 4 + propose next
hill-climb cells (multi-seed V1 ranked highest in remaining budget).

### Coordination note for X

Both variants probed under PAPER methodology with `subject_model:
google/gemma-2-2b` field set. Training_index entries follow the
existing schema (no new fields). FLIP applied for winogrande +
wsc; non-FLIP tasks have `test_auc == test_auc_flip`. Probing rows
appended cleanly to `probing_results.jsonl` (X's IT-side concurrent
run unaffected — different subject_model field).
