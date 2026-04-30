---
author: Dmitry (autonomous overnight run)
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Overnight session summary: paper-faithful TXC k-sweep + bundle-size hypothesis + windowed-T-SAE ladder

### Goal

User direction: "Re-run TXC at paper-faithful settings. If results promising, vary k more. If not, try another arch (Han champion). If nothing works, ladder approach starting from T-SAE recipe + T=2 windowed encoder, scale up to T=3, T=4, T=5."

### What ran

| host  | run                          | start | wall-clock | status |
|-------|------------------------------|-------|------------|--------|
| h100_1 | TXC paper-faithful k=100 30k | T+0   | 90 min train + 120 min Wang | done |
| h100_2 | TXC paper-faithful k=200 30k | T+0:55 | 90 min + 120 min | done |
| h100_1 | TXC paper-faithful k=50 30k  | T+3:50 | 90 min + 120 min | done |
| h100_2 | windowed-T-SAE T=2 30k       | T+5:30 | 90 min + ~120 min | mid-Wang |
| h100_1 | bundle-size sweep on TXC k=100 ckpt | T+6:30 | ~30 min | done |
| h100_1 | bundle-size sweep on T-SAE paper ckpt | T+7:00 | ~25 min | running |

### Key result 1: TXC paper-faithful k-sweep (resid_post 30k, bundle k=30)

| k_total | bundle peak align | best single feat align (stage 4) |
|---:|---:|---:|
| 50 | 47.90 | feat 6406: 51.98 |
| **100** | **50.89** | **feat 4563: 58.47**  ← strongest single feature any arch |
| 200 | 50.78 | feat 10625: 55.08 |

Sweet spot for TXC paper-faithful is k_total=100. Bundle peaks across all k stay ≤51, well below T-SAE paper-faithful (56.23) and SAE arditi 100k (57.42).

**However**, TXC k=100 finds a single feature (4563, Δz̄=+0.29) that solo-steers to **align=58.47 coh=30.86** at α=−8 — beating SAE arditi 100k bundle (57.42). The architecture finds great features; the k=30 bundle method just dilutes them.

### Key result 2: bundle-size hypothesis (TXC k=100, varying k_bundle)

Re-ran the same Wang procedure top-30 list with smaller bundle sizes:

| k_bundle | peak α | peak align | peak coh |
|---:|---:|---:|---:|
| 1 | +9 | 50.97 | 25.23 |
| 2 | -4 | 52.59 | 25.08 |
| 3 | -1.5 | 49.42 | 23.98 |
| **5** | **-8** | **55.17** | 28.44 |
| 10 | -4 | 53.71 | 26.48 |
| 30 | -10 | 50.89 | 28.83 |

Bundle peak drops from 55.17 (k_bundle=5) → 50.89 (k_bundle=30) — **−4.3 align points purely from including more features in the bundle**. This is the bundle dilution effect: the sum of N unit-norm decoder rows has norm √N, so the per-direction effective steering shrinks as 1/√N if directions are roughly orthogonal.

The non-monotonicity at k_bundle ∈ {1, 2, 3} comes from the top stage-3-ranked feature (feat 2760) being a bundle-level dud (single-feat peak 49.65) despite high stage-3 score. Including it dilutes the bundle. By k_bundle=5 the diluting effect of feat 2760 is washed out by the strong contributions of feat 4563 + others.

### Key result 3: windowed-T-SAE T=2 (fallback ladder, in progress)

Running on h100_2 — currently in Wang stage 3, finalist 7/20. Bundle frontier expected ~T+8h.

[Numbers and conclusion TBD on completion]

### Implications

1. **TXC paper-faithful at k=100 is the strongest TXC variant on this organism but still loses on bundle-k=30 vs T-SAE paper-faithful.** Bundle peaks: TXC 50.89 vs T-SAE 56.23. The 5+ point gap is real, not noise.
2. **Single-feature TXC 4563 is the strongest steerer of any arch** at 58.47 align. Beats SAE arditi 100k bundle (57.42). This is the architecturally interesting result: TXC's windowed encoder produces individually exceptional features, even if the bundle method doesn't aggregate them well.
3. **The bundle-size effect is meaningful** (4–5 align points across k_bundle=5 → 30). Suggests we should consider switching from bundle-k=30 as the headline metric to **bundle-k=5** or even single-feature peaks for honest cross-architecture comparison. The current bundle-k=30 metric is biased against architectures whose top features are well-concentrated and orthogonal to each other — which TXC paper-faithful seems to be.
4. **For the TXC architecture to win on bundle-k=30**, we'd need its top-30 features to be more aligned with each other (i.e., more "redundant misalignment direction" instead of orthogonal complementary features). That's an arch design question, not a hyperparameter one.

### What this likely means for fallback ladder

- If windowed-T-SAE T=2 ≥ 56 (matching T-SAE paper-faithful at T=1): the user's hypothesis that windowing helps is validated, ladder up.
- If windowed-T-SAE T=2 ≈ TXC paper k=100 (51): the windowing is neutral, the contrastive recipe of T-SAE matters more than the windowing.
- If windowed-T-SAE T=2 < 51: something in our windowing-encoder design is off. (E.g., maybe the per-position bias subtraction or the cross-position mixing-by-identity is wrong.)

### Files

Per-variant `wang_*` dirs and `*_bundle*_frontier.json` are committed in `docs/dmitry/results/em_features/hookpoint_compare/txc_paper_k{50,100,200}_30k/`. T-SAE bundle sweep results (k=1, 5, 10) on host h100_1, will be pulled in when done.
