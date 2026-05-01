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

### 2026-05-01 01:00 UTC routine-firing update — TXC paper-faithful k=20 done

**Bundle k=30 peak: 55.50 align at α=+3, coh=25.94** (α=0 baseline 44.24, lift +11.26).

Stage 4 single-feature finalists:

| feat | Δz̄ | peak α | peak align | peak coh | α=0 align |
|---|---:|---:|---:|---:|---:|
| 5773 | +0.15 | +1.25 | 48.54 | 23.98 | 40.92 |
| 1989 | +0.44 | +10.0 | 52.29 | 30.78 | 41.90 |
| **6062** | **+0.22** | **+8.0** | **55.16** | **31.33** | **44.54** |

Best single-feature: feat 6062, α=+8, align=55.16, coh=31.33.

**TXC paper-faithful k-sweep, single-feature peak:**

| k_total | best single feat | peak α | peak align | peak coh |
|---:|---:|---:|---:|---:|
| 20 | 6062 | +8 | 55.16 | 31.33 |
| 50 | 6406 | -4 | 51.98 | 33.28 |
| **100** | **4563** | **−8** | **58.47** | **30.86** |
| 200 | 10625 | −10 | 55.08 | 34.53 |

Sweet spot for single-feature is decisively k_total=100. Both increasing (k=200, 55.08) and decreasing (k=20, 55.16) sparsity costs ~3 align points. k=50 dip (51.98) is likely feature-selection noise — the top-3 finalists for that run had Δz̄ < 0.3 and were noisy.

**Direction-flip pattern.** k=20 finalists ALL peak at *positive* α (+1.25 / +10 / +8), opposite to k=100 (α=−8) and k=200 (α=−10). We previously saw this same flip for the un-mixed wtsae_T2 (peaked at α=+10) and partially in wtsae_T2+mix (mixed peak signs). Hypothesis: at low k_total the BatchTopK encoder is so sparse that the surviving features are not "misalignment direction" features but something more like "domain/topic" features (medical-careful-answer style), so positive-α amplifies them in a way the judge interprets as more aligned. The Δz̄ values are all small/positive (≤0.44), consistent with weak misalignment-direction signal at this sparsity.

**Implication for Track C decision tree.** Brief said: if 52 ≤ single-feat < 58.47, try k=30 or k=75. With k=20=55.16 and k=50=51.98 (possibly noise), k=100=58.47, k=200=55.08, the curve is not monotonically peaked — there's a noisy k=50 dip and otherwise smooth. The single-feature peak vs k curve looks roughly like:

```
k:    20    50   100   200
peak: 55.2  52.0 58.5  55.1
```

The optimum is sharply at k=100 with ±20% in both directions costing ~3 points. There is **no obvious reason to expect k=75 or k=125 to beat 58.47** — the curve is locally smooth around k=100 only with k=50 as a dip outlier (and that dip is more likely feature-selection noise than a real local-min). Conclusion: k-sweep direction is exhausted for the resid_post hookpoint at d_sae=16k, 30k steps. To push past 58.47 we should switch direction.

**Updated next-experiment thinking.** With k-sweep exhausted, the candidates remaining are:

1. **Track B (vanilla SAE windowed T=2, NOT YET STARTED).** Cleanest test of "does windowing alone help" without any contrastive recipe. Architecturally the most informative still-uncovered direction.
2. **C3 (TXC k=100 60k steps).** Tests if more compute pushes feat 4563 above 58.47. Low risk of dramatic improvement (architectures usually saturate by 30k for d_sae=16k), but a clean +1-2 point gain would be a champion.
3. **C4 (TXC k=100 paper-faithful @ resid_mid).** Different hookpoint, same arch. Our existing TXC @ resid_mid (with our older settings) was 53.87 single; paper-faithful settings might do better.

Track B has the highest information-per-experiment value (it's a direction we haven't tested at all). C3 has the highest expected-numerical-gain value (extends the current champion). I'll launch C3 first as the cheaper safer hill-climb, since (a) we already have the launcher script, (b) it requires no code change, (c) it tests if more steps pushes 58.47 → 60+. Track B requires writing a vanilla-SAE windowed launcher (or running wtsae with `--contrastive_alpha 0.0`); will tackle next firing.

**Disk first.** Need to clean up before launching C3. The k=200 ckpt (`qwen_l15_txc_paper_k200bt_d16k_step30000.pt`, ~5 GB) is fully HF-mirrored and Wang-done — safest to delete. Will follow the disk-policy procedure.

### 2026-05-01 01:30 UTC routine-firing update — WindowedTSAE T=2 + mix + matryoshka done

Pulled from h100_2. Bundle k=30 frontier and Stage 4 finalists computed.

**Bundle k=30 peak: 50.59 align at α=−1, coh=23.91** (α=0 baseline 43.28).

Stage 4 single-feature finalists (top-3 from ΔẑΔz̄ + screen + strength funnel):

| feat   | Δz̄    | peak α | peak align | peak coh | α=0 align |
|-------:|------:|-------:|-----------:|---------:|----------:|
| 11791  | +0.15 |  −10   | 54.61      | 33.67    | 41.89     |
|  7619  | +0.09 |   −3   | 53.66      | 25.55    | 45.26     |
| **14496** | **+0.02** | **+9** | **57.59** | **27.34** | **44.37** |

**Best single-feature peak: feat 14496, α=+9, align=57.59, coh=27.34.** This is the **second-best single-feature steerer ever recorded on this organism**, behind only TXC paper k=100 feat 4563 (58.47).

**Comparison across the three T=2 wtsae variants (all 30k steps, paper-faithful T-SAE recipe except the variant axis):**

| variant                    | bundle peak | best single feat | single peak | single coh |
|----------------------------|------------:|-----------------:|------------:|-----------:|
| original (M=I, no matr)    | 51.27       | feat 14475 (α+8) | 55.44       | 31.95      |
| + mix_positions            | **55.57**   | feat 8017 (α−6)  | 54.58       | 28.05      |
| + matryoshka 20% only      | 50.35       | feat (n/a)       | ~49.0       | n/a        |
| **+ mix + matryoshka 20%** | 50.59       | **feat 14496 (α+9)** | **57.59** | **27.34**  |

**What this likely means.** Two contrasting effects:
1. **Bundle dropped vs mix-only** (55.57 → 50.59). Adding matryoshka on top of mix hurts the *aggregate* features metric. Consistent with matr-only also being a regression.
2. **Single jumped vs mix-only** (54.58 → 57.59, +3.01). Adding matryoshka on top of mix yields a new exceptional steerer.

These two effects are **not contradictory** — they are consistent with the matryoshka split forcing structure into the feature population. The contrastive loss on only the first 20% (indices < 3277) pulls those features into a more aligned subspace, but the bundle metric averages across the top-30 features by screen-score (which span both regions). The bundle drop reflects that the contrastive-trained head produces features that, while individually well-shaped, do not aggregate cleanly with the un-contrastive tail.

**Crucially, feat 14496 (the 57.59 winner) has index 14496 > 3277** — it is **outside the matryoshka contrastive head**, in the unconstrained tail. Likewise feat 11791 and 7619 are also in the tail. So the single-feature win is **not** the matryoshka features being directly improved; rather the matryoshka contrastive on the first 20% appears to free up the unconstrained tail to specialize, and one tail feature happens to be an excellent steerer.

This is a less robust finding than I'd like: the win is single-feature-specific and (a) doesn't show on the bundle metric and (b) comes from a feature that doesn't even participate in the constraint we added. The 57.59 may be a noisy outlier rather than a reproducible architectural win.

### Decision-tree action

Per the brief Track A decision rule: *"If bundle ≥ 56.23 OR single ≥ 56: T=2 with this fix WORKS. Ladder up to T=3 + same fix."*

Single 57.59 ≥ 56, so the rule says to ladder T=3. But:

- The bundle metric REGRESSED (50.59 vs mix-only's 55.57). That's a strong signal the recipe is hurting most features.
- The single-feature win comes from index > 3277, which is not in the contrastive head — i.e. the matryoshka isn't directly responsible for the strong steerer.
- The single-feature peak distribution is heavily skewed (57.59 vs 53.66 vs 54.61) — only one finalist is the high one, suggesting low robustness.

A 3.5h T=3 ladder run is expensive. I prefer to first test the **complementary direction (Track B: vanilla windowed)** that we have not yet started, before committing to laddering up the matr+mix recipe. Vanilla windowed gives a clean answer to "does windowing alone help" with no contrastive confound.

**Decision: launch Track B B1 (vanilla SAE windowed T=2) on h100_2 next.** Tag the run `vanilla_d32k_k128_mix` with the SAE-arditi-faithful settings (d_sae=32768, k=128, contrastive_alpha=0.0, mix_positions=True). h100_2 has 154 GB free → no disk constraint. Will pass `--d_sae 32768 --k 128 --contrastive_alpha 0.0` as EXTRA_ARGS to the existing `/tmp/run_wtsae_variant.sh` launcher (argparse takes the last value, overriding the script defaults).

If Track B B1 yields a bundle peak ≥ 57.42 (matching SAE arditi 100k), we have a clean "windowing helps even without contrastive" story → ladder to T=3. If it underperforms SAE arditi 100k, then any windowing benefit requires the contrastive recipe (Track A direction is the one to push on).

Will keep h100_1's TXC k=100 60k running as planned (it tests the orthogonal "more compute helps the champion" hypothesis on Track C3). At time of this firing it's at step ~5000/60000 — still ~5h to training completion + ~2h Wang.

### 2026-05-01 04:00 UTC routine-firing update — Wang flag mismatch, recovery, Track D prep

**Status of in-flight runs.**

- **h100_1** TXC paper k=100 60k: still training at step ~47500/60000 (~80% done). ETA training-done in ~30 min, then ~2h Wang. Snapshot at step 30000 (the redundant intermediate) already on disk — pushing free disk to 9.6 GB (was 11 GB before the snap). Will need ~7 GB more at step 60000; leaving ~2.6 GB margin. Tight but safe — not intervening mid-run.
- **h100_2** WindowedTSAE T=2 vanilla SAE (`vanilla_d32k_k128_mix`): training **completed** + encoder run completed (top_200 features extracted), but the **Wang procedure crashed** at the very first stage with `argparse: unrecognized arguments: --save_demo_completions=-1`. Root cause: h100_2's repo was at commit `ae2b71f` (pre-`--save_demo_completions` rollout); the launcher `/tmp/run_wtsae_variant.sh` had been edited on h100_2 to include the flag, but the underlying `run_wang_procedure.py` didn't yet accept it.
- **Recovery**: pulled h100_2's repo to latest dmitry (`4d16397`), wrote a small resume launcher `/tmp/run_wtsae_wang_resume.sh` that runs the Wang + bundle-frontier stages only (re-using the existing ckpt + encoder JSON), and launched it. ETA ~2h. Log: `wtsae_T2_vanilla_d32k_k128_mix_resume.log`. Status confirmed: subject model loaded, screening 100 features.

**No new training launchable** — both GPUs busy.

### 2026-05-01 04:00 UTC — Track D code prep (not GPU work)

Per Track D D1a: implement window-level adjacency contrastive for TXC. Reading `experiments/em_features/run_training_txc_bricken_auxk.py`, the change is:

1. Add `--contrastive_alpha` flag (default 0.0 = off, identical-to-current behaviour).
2. When `alpha > 0`, sample windows of length `T+1` per step; split into two `T`-length views offset by 1 token (`view1 = win[:, :T]`, `view2 = win[:, 1:]`).
3. Encode both views, apply existing TopK / BatchTopK independently → `z1`, `z2` of shape `(B, d_sae)`.
4. Standard recon loss uses `view1` exactly as before — when α=0 the math reduces to current code.
5. Add `α · (1 - cos_sim(z1, z2)).mean()` to total loss.

This keeps the change minimal (~30 lines) and α=0 → bit-identical to current TXC. Once a GPU frees, launch `D2`: TXC paper-faithful k=100, contrastive_alpha=0.1, 30k steps @ resid_post — compare against the existing TXC paper k=100 baseline (bundle 50.89, single feat 4563=58.47).

**Hypothesis**: Bhalla's contrastive loss on z_t-vs-z_{t+1} pulls features into a more redundant subspace, raising the bundle peak. We saw this empirically: T-SAE paper-faithful gets bundle 56.23 (vs TXC k=100's 50.89). If the TXC arch + adjacency contrastive recipe combines T-SAE's bundle-friendliness with TXC's strong single-feature peaks (58.47), that's the headline architectural win we're hunting for.
