---
author: Han
date: 2026-04-27
tags:
  - results
  - reference
---

## Per-arch training time (seed=42, H200, batch=4096, max_steps=8000)

Generated from `training_index.jsonl` after Agent A completed 37 of
the 38 trimmed-canonical archs at seed=42 (anchor cell h8 still
training; SubseqH8 T_max sweep + high-T H8 dropped per
`2026-04-27-canonical-trim-emergency.md`). Source of truth is
`training_index.jsonl` and per-arch
`training_logs/<arch>__seed42.json` files.

### Summary stats

- Total wall clock: **23.2 hr** across 37 archs
- Per-arch min / median / max: **5.2 / 35.7 / 96.0 min**
- All archs converged via plateau early-stop (none hit max_steps=8000 cap)

### Per-arch breakdown

| row | grp | arch | min | step | converged |
|---|---|---|---:|---:|---|
| 1 | 1 | topk_sae | 5.2 | 6600 | ✓ |
| 2 | 1 | tsae_paper_k500 | 10.2 | 6800 | ✓ |
| 3 | 1 | tsae_paper_k20 | 7.5 | 4800 | ✓ |
| 4 | 1 | mlc | 12.9 | 4200 | ✓ |
| 5 | 1 | mlc_contrastive_alpha100_batchtopk | 22.4 | 3000 | ✓ |
| 6 | 1 | agentic_mlc_08 | 25.7 | 3000 | ✓ |
| 7 | 1 | tfa_big | 43.0 | 4800 | ✓ |
| 8 | 2 | agentic_txc_02 | 44.9 | 3800 | ✓ |
| 9 | 2 | txc_bare_antidead_t5 | 19.2 | 4200 | ✓ |
| 10 | 2 | txc_bare_antidead_t10 | 37.4 | 4000 | ✓ |
| 11 | 2 | txc_bare_antidead_t20 | 63.9 | 3400 | ✓ |
| 12 | 2 | phase5b_subseq_track2 | 32.0 | 3200 | ✓ |
| 13 | 2 | phase5b_subseq_h8 | 91.9 | 3000 | ✓ |
| 14 | 3 | txcdr_t3 | 10.4 | 5600 | ✓ |
| 15 | 3 | txcdr_t4 | 12.8 | 5200 | ✓ |
| 16 | 3 | txcdr_t5 | 16.6 | 5400 | ✓ |
| 17 | 3 | txcdr_t6 | 16.5 | 4400 | ✓ |
| 18 | 3 | txcdr_t7 | 22.2 | 5000 | ✓ |
| 19 | 3 | txcdr_t8 | 22.4 | 4400 | ✓ |
| 20 | 3 | txcdr_t9 | 25.1 | 4400 | ✓ |
| 21 | 3 | txcdr_t10 | 26.2 | 4200 | ✓ |
| 22 | 3 | txcdr_t12 | 29.1 | 4000 | ✓ |
| 23 | 3 | txcdr_t14 | 35.7 | 4200 | ✓ |
| 24 | 3 | txcdr_t16 | 38.7 | 4000 | ✓ |
| 25 | 3 | txcdr_t18 | 43.5 | 4000 | ✓ |
| 26 | 3 | txcdr_t20 | 45.8 | 3800 | ✓ |
| 27 | 3 | txcdr_t24 | 54.8 | 3800 | ✓ |
| 28 | 3 | txcdr_t28 | 60.7 | 3600 | ✓ |
| 29 | 3 | txcdr_t32 | 69.1 | 3600 | ✓ |
| 30 | 4 | phase57_partB_h8_bare_multidistance_t3 | 22.3 | 4200 | ✓ |
| 31 | 4 | phase57_partB_h8_bare_multidistance_t4 | 35.9 | 3600 | ✓ |
| 32 | 4 | phase57_partB_h8_bare_multidistance_t5 | 44.8 | 3600 | ✓ |
| 33 | 4 | phase57_partB_h8_bare_multidistance_t6 | 53.7 | 3600 | ✓ |
| 34 | 4 | phase57_partB_h8_bare_multidistance_t7 | 62.6 | 3600 | ✓ |
| 35 | 4 | phase57_partB_h8_bare_multidistance_t8 | 80.5 | 3200 | ✓ |
| 36 | 4 | phase57_partB_h8_bare_multidistance_t9 | 96.0 | 3400 | ✓ |
| 46 | 5 | txcdr_t20_kpos100 | 50.6 | 4200 | ✓ |

### Patterns

- **TXCDR T-sweep is roughly linear in T**: 10 min at T=3 → 69 min at T=32. Slope ~2 min per T step. Encode cost dominates.
- **H8 multi-distance is ~1.4× slower than vanilla TXCDR at the same T** because of the 1+K shift tensor (K = 2 or 3 shifts depending on T): H8_t9 = 96 min vs txcdr_t9 = 25 min ⇒ ~3.8× actually (H8 has matryoshka aux + InfoNCE on top). Per-step compute is much higher than vanilla.
- **SubseqH8 (row 13, T_max=10) at 92 min** = comparable to H8_t9 (96 min); subseq sampling doesn't slow it down vs the equivalent H8.
- **MLC family (rows 4-6) at 13-26 min** = surprisingly fast despite multi-layer (5x d_in input). The 5-layer einsum is cheap relative to TXC's window slide.
- **TFA (row 7) at 43 min** = mid-range. Full sequence per forward (B=32 only), bottleneck attention layer keeps it fast.
- **All converged via plateau early-stop**: max_steps=8000 cap was non-binding. Most archs converged at step 3000-5000 (at the floor or shortly after).

### Implications for seed=1, seed=2

If we kept all 49 archs at the same per-arch wall clock, a full seed
batch on H200 = ~30-35 hr. That's why we trimmed the canonical to 38
archs and dropped the 11 high-cost cells (h8_t10..t32 + SubseqH8
T_max sweep) — those would have added ~12-15 hr per seed.

The trimmed 38-arch budget:
- ~23 hr for the 38 archs at seed=42 (37 done, 1 in flight)
- Probing pass: ~estimated 3 hr at S=20 (much faster than S=128)

So a full per-seed budget is ~26 hr on H200 — ~1 day per seed.
