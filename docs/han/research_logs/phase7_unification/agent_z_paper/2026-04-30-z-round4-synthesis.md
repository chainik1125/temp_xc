---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
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
| **`hill_z_ranked_T20_s5`** (Z R4-A) | **0.8917** | **−0.0214** | **deep cluster** | **LOSE** |
| `hill_z_shared_T20` (Z R4-B) | TBD | TBD | TBD | pending |

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

### Conclusion (pending shared)

- `SubseqRankedH8` is **NOT a leaderboard contender**. The negative
  result rules out per-sampled-slot encoding for the cross-token
  evaluation regime.
- The cleanest reading: **don't compress slots when probing requires
  per-position resolution.**
- For Z's hill-climb roadmap: the rank-variant is dead. Next promising
  direction = the *shared* variant (currently probing) and/or
  multi-seed of V1 (T_max=12 t_sample=5 — fits 5090 directly,
  no compression needed).

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
