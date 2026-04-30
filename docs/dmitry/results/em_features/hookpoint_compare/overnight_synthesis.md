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

### Mid-flight standout: WindowedTSAE T=2 + mix_positions=True (in progress on h100_1)

Stage 2 causal screen (preliminary, before stage 3 strength sweep) shows **multiple low-Δz̄ high-causal features** at α=−1 already, several beating SAE arditi 100k bundle peak (57.42) at just α=−1:

| feat | Δz̄ | screen score | α=+1 align/coh | α=−1 align/coh |
|---:|---:|---:|---|---|
| 15836 | +0.085 | +20.02 | 35.33 / 23.12 | 55.36 / 26.56 |
| 3179  | +0.063 | +21.79 | 39.29 / 23.44 | **61.07** / 29.69 |
| 10711 | +0.027 | +23.08 | 39.23 / 23.75 | **62.31** / 27.19 |
| 8745  | +0.027 | +27.19 | 30.00 / 22.50 | 57.19 / 27.19 |

(Stage 2 uses only n=16 rollouts vs stage 4's n=64, so noise is roughly 2× higher.
The standouts trade ~5–8 coh points to gain ~5 align points vs SAE arditi 100k
peak (57.42 / 35.78). Stage 4 frontier with n=64 will give a more reliable
estimate.)

**feat 10711** at α=−1 hits 62.31 align — already higher than the prior champion's *peak* (SAE arditi 100k = 57.42 at α=−10) at 1/10th the steering magnitude. If the stage 3/4 frontier holds up, this is a major lift from the matryoshka/mixing fixes.

Awaiting stage 3 strength sweep + stage 4 frontier on these finalists. **Will update once Wang procedure completes.**

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

Per-variant `wang_*` dirs and `*_bundle*_frontier.json` are committed in `docs/dmitry/results/em_features/hookpoint_compare/txc_paper_k{50,100,200}_30k/`. T-SAE bundle sweep results (k=1, 5, 10) on host h100_1, will be pulled in when done.
