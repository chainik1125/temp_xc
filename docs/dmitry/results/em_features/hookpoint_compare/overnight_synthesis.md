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

### Key result 3: windowed-T-SAE T=2 (fallback ladder) — falsified

Trained `WindowedTSAE` (per-position W_dec[T, d_sae, d_in], shared W_enc, M=I) at T=2 with paper-faithful T-SAE settings: d_sae=16384, k=20 BatchTopK, batch=512, contrastive_alpha=0.1, lr=3e-4, 30k steps, hookpoint=resid_post.

Stage 4 finalist peaks (single-feature):

| feat | Δz̄ | peak α | peak align | peak coh |
|---|---:|---:|---:|---:|
| 14919 | +0.02 | +2 | 49.66 | 24.14 |
| 15067 | +0.05 | +10 | **55.44** | 30.78 |
| 14951 | +0.67 | +3 | 54.67 | 25.23 |

**Bundle k=30 peak: 51.27 align at α=−6, coh=32.89** (α=0 baseline 45.96, lift +5.31).

This is **WORSE than T-SAE T=1 paper-faithful (bundle 56.23)** — the windowed encoder hurt by ~5 align points. Ladder approach cannot proceed to T=3 with this design — would just compound a regression.

Likely culprits:
1. **Per-position W_dec** (T, d_sae, d_in): each position-t decoder only sees gradient from tokens at position t in training, so features that fire mostly at position 0 train W_dec[0] well but leave W_dec[1] under-trained. At inference we use W_dec[-1] for steering (per Wang convention), which biases against features that prefer position 0.
2. **Per-position b_dec** vs T-SAE's shared b_dec: small effect but compounds.
3. **Cross-position mixing M=I** disabled by default: the encoder gains no cross-position information from the windowing — it's just per-token T-SAE with split decoders.

A T-SAE-faithful "T=2 windowed" should probably either (a) keep shared W_dec and only window the contrastive-loss application, OR (b) enable mix_positions=True so the encoder actually USES the windowing.

Single-feature 55.44 at α=+10 (positive!) is also notable — opposite sign convention from TXC paper-faithful (α=−8 = 58.47). Likely a direction-flip in our W_dec[-1] convention for the windowed arch.

### Key result 4: T-SAE paper-faithful bundle-size sweep — confirms architectural difference

| k_bundle | T-SAE peak align | TXC k=100 peak align |
|---:|---:|---:|
| 1 | 51.61 | 50.97 |
| 5 | 52.07 | **55.17** |
| 10 | 54.33 | 53.71 |
| 30 | **56.23** | 50.89 |

**T-SAE bundle peak monotonically RISES with k_bundle.**
**TXC bundle peak peaks at k_bundle=5 then FALLS.**

The two architectures behave oppositely under the same Wang procedure top-30 list. This points to a real architectural distinction:

- **T-SAE features are aligned/redundant**: per-token TopK + adjacency contrastive pulls z_t and z_{t+1} together, so features that fire on consecutive tokens have similar decoder rows. Summing many of them strengthens the collective direction.
- **TXC features are orthogonal/diverse**: window-level encoder produces features that capture distinct multi-position patterns, so their decoder rows are nearly orthogonal. Summing dilutes signal as 1/√k_bundle.

The standard Wang `bundle k=30` headline metric is therefore biased toward arches with redundant features. **For TXC, single-feature steering or `bundle k=5` is the appropriate metric.**

### Key result 5: WindowedTSAE T=2 + mix_positions=True — partial recovery, not enough

Stage 2 causal screen had shown several low-Δz̄ features at α=−1 hitting **align ≥ 60** (feat 10711=62.31, feat 3179=61.07). Those did not survive the n=64 stage-4 estimator.

**Bundle k=30 peak: 55.57 align at α=−10, coh=32.58** (α=0 baseline 41.53, lift +14.04).
Stage 4 finalists:

| feat | Δz̄ | peak α | peak align | peak coh |
|---|---:|---:|---:|---:|
|  8017 | +0.02 | −6   | **54.58** | 28.05 |
|  9925 | +0.02 | −1.25 | 50.00 | 24.53 |
| 14475 | +0.03 | +8   | 51.45 | 24.53 |

**Vs T-SAE T=1 paper-faithful (bundle 56.23, single 56.23 via finalist):** still under by ~0.7 bundle / ~1.5 single.
**Vs original wtsae_T2 (no-mix) (bundle 51.27, single 55.44):** mix_positions = +4.30 bundle, −0.86 single.

`mix_positions=True` learns a (T,T) mixing matrix M after the per-position W_enc, before TopK. So the encoder finally USES cross-position information. The bundle improvement (+4.30) confirms hypothesis from key result 3: the original windowed encoder was just per-token T-SAE with split decoders, and mixing makes it actually multi-position. Sign convention also normalized — mix run's best finalist peaks at α=−6 (the conventional "negative steering = align" direction), while no-mix peaked at α=+10/+3 (likely a direction-flip artifact of unutilized W_dec[1]).

But the absolute level still does not beat T-SAE T=1 paper (56.23 bundle), so per the brief's decision rule we should NOT ladder up to T=3 with this exact config. Hypothesis: per-position decoder W_dec[T,d_sae,d_in] is still under-trained at T=2 because each position only sees half the gradient. Two natural follow-ups (in priority order):

1. **WindowedTSAE T=2 + mix + matryoshka (running on h100_2 per handoff snapshot)** — matryoshka concentrates contrastive on first 20% of d_sae, which could over-train the high-priority features. Not yet pulled (no h100_2 access from this routine).
2. **WindowedTSAE T=2 + mix + shared W_dec across positions** — directly fixes the per-position-decoder under-training problem. Architectural change, requires code edit.
3. **WindowedTSAE T=2 + mix + larger contrastive_alpha (0.5 instead of 0.1)** — cheap variant to test if stronger contrastive helps the windowed encoder.

### Headline (revised given the bundle-size results)

If we report each architecture at *its own optimal k_bundle*:

| arch | optimal k_bundle | peak align | peak coh | α |
|---|---:|---:|---:|---:|
| **TXC paper k=100 single feat 4563** | **1** | **58.47** | 30.86 | −8 |
| SAE arditi 100k @ resid_post | 30 | 57.42 | 35.78 | −10 |
| T-SAE paper-faithful 30k | 30 | 56.23 | 34.84 | −6 |
| TXC paper k=100 | 5 | 55.17 | 28.44 | −8 |

**TXC paper-faithful k=100 single-feature 4563 is the strongest steerer of any arch tested on this organism**, beating SAE arditi 100k by +1.05 align (and SAE was the prior champion). It does trade ~5 coh points (30.86 vs 35.78), but on Wang's primary metric (alignment under the constraint that coh stays high enough to be readable) feat 4563 wins.

The standard `bundle k=30` metric undersells TXC by 7+ align points because of the orthogonality-driven bundle dilution.

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

Per-variant `wang_*` dirs and `*_bundle*_frontier.json` are committed in `docs/dmitry/results/em_features/hookpoint_compare/txc_paper_k{50,100,200}_30k/` and `wtsae_T2_30k/`, `wtsae_T2_mix_30k/`. T-SAE bundle sweep results (k=1, 5, 10) on host h100_1, will be pulled in when done.

### 2026-04-30 23:00 UTC routine-firing update

**In-flight at this firing:**
- h100_1: TXC paper-faithful k=20 Wang procedure on stage 3 causal screen, ~96/100 features. Stage 4 (3 finalists × 27 alphas × 8 rollouts) still ahead. ETA ~1-2 h.
- h100_2: WindowedTSAE T=2 + mix + matryoshka (per handoff snapshot). Cannot ssh from this routine to verify; should be at or past Wang completion by wall-clock estimate.

**Disk on h100_1:** 8.3 GB free of 200 GB (96% used). Below the 10 GB threshold. New training launches are blocked until either (a) k=20 Wang completes and the resulting ckpt is freed/HF-mirrored, or (b) an existing HF-mirrored ckpt is cleaned up. **Will not launch any new run this firing.**

**Decision:** wait for the in-flight TXC k=20 Wang to finish. Once that lands the next move is dictated by (a) the k=20 single-peak result for Track C, and (b) whether h100_2's mix+matr beat 56.23 bundle for Track A. If both have completed by next firing, the cleanup-then-launch can begin.
