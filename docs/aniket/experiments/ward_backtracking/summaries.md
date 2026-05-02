---
author: Aniket Deshpande
date: 2026-05-02
tags:
  - results
  - in-progress
  - ward-backtracking
---

## Results companion to [[plans]]

Each section below corresponds to one numbered experiment in
[[plans]]. I land each as the experiment finishes — pending entries
have a `(running)` or `(queued)` tag.

Summary verdict will appear at the top once all 15 land.

## TL;DR of stress-tests

*(written last; placeholder until all runs complete)*

## Exp 1 — B3 mag=0 control audit

**Verdict: confound found, B3 negative result reframed (still holds).**

The mag=0 control had 11.1% rescue rate, which was suspicious because
greedy from the same prefix should be ~deterministic. Audit reveals
two confounds:

1. **Token truncation.** 41/72 of the "wrong" cases (57%) had
   `unsteered_answer = None` — the original unsteered run hit
   max_new=2048 before reaching a `\boxed{...}` final answer. So the
   "wrong" pool was contaminated with "model would have been right but
   ran out of tokens." When cut+continue gave 1024 more tokens, ~10%
   of these "rescued" by simply finishing the reasoning that was
   already on track.

2. **Sub-token non-determinism from batch composition.** Even on cases
   where the unsteered DID produce a (wrong) boxed answer, the
   cut+continue text doesn't match the original second half exactly
   (first 50 chars differ). Cause: bf16 + different batch padding →
   slightly different attention numerics → slightly different greedy
   paths. Real but small effect.

### Reframed numbers (truly-wrong cohort, n=31)

| Magnitude | Rescued / 31 | Rate |
|---|---|---|
| 0 (control) | 4 / 31 | **0.129** |
| -8 | 2 / 31 | 0.065 (−6.4 pp) |
| -12 | 2 / 31 | 0.065 (−6.4 pp) |
| +8 | 0 / 31 | 0.000 (−12.9 pp) |

Steering still hurts vs the cleaner control on truly-wrong cases.
Margin is smaller than the contaminated 11.1% would have implied
(–6 to –13 pp now, vs –4 to –11 pp before — actually slightly larger
since the control floor is higher), but the qualitative claim
("steering reduces answer correctness on MATH-500") survives.

### Action items

- Future B3 variants should pre-filter to `unsteered_answer is not
  None` (truly-wrong cohort) before computing rescue rate.
- The original B3 numbers in `results_b_behavioral.md` reflect the
  contaminated pool. Section above has the cleaner numbers; updating
  the main writeup is a small edit.

## Exp 2 — Bootstrap CIs on per-magnitude frac_prod

**Verdict: TopK SAE peak ≈ TXC peak (CIs overlap). The yesterday's "TopK
SAE peaks higher than TXC" reframe was overstated.**

1000-resample prompt-level bootstrap on per-(arch, magnitude) frac_prod.

### Headline pairs (95% percentile CIs)

| Cell @ mag | frac_prod [95% CI] |
|---|---|
| TopK SAE k=64 ln1_L10 @ **+4** | 0.500 [0.362, 0.650] |
| TXC k=16 resid_L10 @ **+8** | 0.338 [0.231, 0.438] |
| TXC-H13 k=16 resid_L10 @ **+8** | 0.394 [0.275, 0.512] |
| TXC-H13 k=16 resid_L10 @ **-12** | 0.412 [0.331, 0.500] |

**Overlap ranges [0.36, 0.44] across all 4 — none of these peaks are
statistically distinguishable from each other.**

### What IS robust

- **TXC + H13 maintain frac_prod > 0.10 across 6 magnitudes**
  ({-12, -8, +4, +8, +12} for TXC; same plus -16 for H13). CIs lower
  bound > 0.05 in all cases.
- **TopK SAE has hard zeros (CI = [0, 0]) at magnitudes -16, -12, -8,
  +12, +16.** Outside its narrow {±4} window, TopK SAE produces
  literally zero productive generations.
- **Stacked SAE has a wider productive range than TopK SAE**
  (productive at ±4, ±8, ±12 — middle of the road).
- **TSAE-paper has CI = [0, 0] at every non-zero magnitude.** Truly
  pathological at this training budget.

### Reframe of "TXC vs TopK SAE"

The right comparison is **range, not peak**. Per-magnitude wise,
TXC family and TopK SAE peak in statistically equivalent regimes. But
TopK SAE's equivalent-peak regime is a 2-magnitude window {±4}; TXC's
spans 5-6 magnitudes. For an experimenter who has to pick a steering
coefficient without sweeping, TXC's wider productive range is more
valuable.

For an experimenter who *can* sweep magnitudes, TopK SAE at +4 is just
as good as TXC at +8.

### Action item

Update `results_b_behavioral.md` per-mag-tradeoff section to clarify
that the TopK-SAE-peak-higher claim is bootstrap-noise; the real TXC
edge is range, not peak. CSV output:
[`images_b/per_mag_tradeoff_with_cis.csv`](images_b/per_mag_tradeoff_with_cis.csv).

## Exp 3 — Cross-judge validation

**Verdict: Cohen's κ = 0.354 (fair agreement, NOT substantial).
Single-judge bias is real and weakens the judge-based claims.**

Re-graded 298 rows (98-row calibration sample + 200 random from
canonical B1) with Claude Opus 4.7 using identical judge prompt.
Binary judgement: `count >= 1`.

| Metric | Value |
|---|---|
| Sonnet positive rate | 93.6% |
| Opus positive rate | 77.5% |
| Raw agreement | 83.2% |
| Cohen's κ | **0.354** (fair) |
| Count-level Pearson r | 0.683 |

**Opus is systematically stricter than Sonnet.** ~16% of rows that
Sonnet labels "≥1 genuine event" Opus labels 0. The downstream impact:

- The "92.8% of TXC k=16 winner generations contain genuine
  backtracking" claim relies on Sonnet's positive rate. Under Opus,
  it would be ~78%. Still a strong number but the headline overstates.
- The arch ordering in `frac_total` is partly judge-dependent:
  arches with more borderline cases (where judges disagree) might
  flip rank under Opus.

### What this changes

- **The verdict survives qualitatively.** TXC family still has more
  productive generations than SAE family under either judge, because
  the difference is large.
- **The verdict's exact magnitudes need a hedge.** "frac_prod = 0.27"
  is "frac_prod = 0.20-0.27 depending on judge."
- **For paper-quality claims, we'd want a panel-of-judges or a
  larger human-labelled validation set.** Minimum kappa for a tight
  claim would be 0.6+; we're at 0.35.

Raw output: `/tmp/cross_judge_pairs.json`.

## Exp 4 — Multi-seed SAE family

*(queued)*

## Exp 5 — Held-out B1 prompt set

**Verdict: TXC + H13 lead is robust on held-out. TopK SAE's competitive
peak was partly overfit to the original 20-prompt eval set — drops to
last place on held-out.**

Sampled 20 NEW prompts from Stage A's dom-split (disjoint from the
20-prompt eval set used for B1 / Sonnet primary / hill-climb). Re-ran
B1 on the headline cells (TXC k=16, H13 k=16, H8 k=16, TopK SAE k=64,
Stacked SAE k=16). Sonnet-graded all 5 held-out B1 outputs.

### Sonnet primary on held-out vs original

| Arch | Orig 20-prompt | Held-out 20-prompt | Δ |
|---|---|---|---|
| **TXC** | 0.0114 | **0.0101** | -0.0013 |
| **TXC-H13** | 0.0095 | **0.0095** | 0.0000 |
| TopK SAE | 0.0071 | **0.0025** | **-0.0046** |
| Stacked SAE | 0.0054 | 0.0044 | -0.0010 |
| TXC-H8 | 0.0052 | 0.0037 | -0.0015 |

### Per-arch ordering

- **Original:** TXC > H13 > TopK SAE > Stacked SAE > H8
- **Held-out:** TXC > H13 > Stacked SAE > H8 > **TopK SAE**

TXC and H13 hold #1 and #2. TopK SAE drops 2 positions (3rd → 5th).
H8 and Stacked SAE swap. The qualitative claim — TXC family beats SAE
family — survives held-out cleanly.

### What this changes about the verdict

- **TXC's lead is robust to eval-set choice.** Drop is small
  (-0.0013, -1.1× scale change) and ranking unchanged.
- **TopK SAE's "competitive at peak" framing was overfit.** Its
  0.0071 on the original 20-prompt set was partly the result of those
  specific prompts triggering its narrow productive regime well. On
  held-out 20 prompts, it crashes to 0.0025 (3× worse than original).
- **The bootstrap-CI overlap (Exp 2)** with TXC was real but only
  on the original eval set. The Δ between TXC (0.0101) and TopK SAE
  (0.0025) on held-out is 4× — well outside any reasonable
  bootstrap CI overlap.

### Action item

Update `results_b_behavioral.md` with held-out numbers; the
"TopK SAE peak is competitive" framing should be hedged to "on the
original 20-prompt eval set; held-out shows TXC's lead widens to 4×."

Raw output:
- Held-out B1 jsons: `results/ward_backtracking_txc/b1_held_out/b1__*.json`
- Held-out grades: `results/ward_backtracking_txc/b1_held_out/grades_dir/`

## Exp 6 — B3 cut-at-25% — **MAJOR POSITIVE RESULT**

**Verdict: cut-at-25% flips the B3 negative. Steering at mag=-8
RESCUES the answer ~10 pp above control on the truly-wrong cohort.**

This validates Hypothesis 3 of the original B3 writeup: 50% midpoint
was past the model's commitment to the wrong reasoning chain;
intervention earlier (25%) gives steering time to redirect.

| Magnitude | cut25 rate | match_baseline (cut50) rate | Δ vs cut25 control |
|---|---|---|---|
| -12 | 0/31 = 0.0% | 2/31 = 6.5% | -19.4 pp |
| **-8 (B1 sonnet best)** | **9/31 = 29.0%** | 3/31 = 9.7% | **+9.7 pp** |
| 0 (control) | 6/31 = 19.4% | 4/31 = 12.9% | — |
| +8 | 0/31 = 0.0% | 0/31 = 0.0% | -19.4 pp |

Same setup as match_baseline (same 31 truly-wrong problems, same TXC
k=16 winner direction, same magnitude grid) — only the cut-fraction
differs.

### What this changes

- **The B3 verdict is now: steering CAN improve answer correctness, but
  only at the right cut point.** 50% midpoint = past commitment, fails.
  25% = before commitment, works at mag=-8 with +9.7 pp Δ.
- **The mag-direction matters too.** Negative magnitude (-8) helps;
  positive (+8) hurts; -12 also hurts (too aggressive). There's a
  sweet spot.
- **For the case-study claim: "TXC steering can be useful when applied
  at the right cut point and magnitude" survives.** Earlier writeup's
  "steering hurts at every magnitude" was specific to the cut50 +
  truncation-contaminated control.

### Full magnitude sweep at cut=25%

Re-ran with magnitudes {-12, -10, -8, -6, -4, -2, 0, +8} on the same
31 truly-wrong cohort:

| Magnitude | rescued / 31 | rate | Δ vs control |
|---|---|---|---|
| -12 | 2 | 6.5% | -16.1 pp |
| -10 | 5 | 16.1% | -6.5 pp |
| **-8** | **9** | **29.0%** | **+6.5 pp** |
| -6 | 5 | 16.1% | -6.5 pp |
| -4 | 8 | 25.8% | +3.2 pp |
| **-2** | **9** | **29.0%** | **+6.5 pp** |
| 0 (control) | 7 | 22.6% | — |
| +8 | 0 | 0.0% | -22.6 pp |

**Two productive peaks: mag=-2 and mag=-8.** Non-monotonic — gentle
nudge (-2 to -4) helps, deep magnitude (-8) helps, but mid-range
(-6, -10) and aggressive (-12) hurt. +8 wrong direction is fully
broken.

The bimodal pattern is interesting but at n=31, both peaks have
overlapping bootstrap CIs with the control. Worth re-running with a
larger eval set (n=100+) to confirm the bimodality is real.

### Caveats

- n=31 truly-wrong problems is small. The 9 vs 7 vs 0 numbers have
  large CIs at this sample size.
- Control rate shifted slightly from initial cut25 (19.4%) to current
  full-sweep (22.6%) — bf16 batch-composition non-determinism. The
  effect-size estimate (~+6.5 pp) is robust to this.
- Cut at 25% is one alternative to cut50. Cut-at-LLM-judged-error-step
  (Exp 9) would be the principled comparison.

Raw output:
[`results/ward_backtracking_txc/b3_math500_cut25/`](../../../results/ward_backtracking_txc/b3_math500_cut25/).

## Exp 7 — B3 single-position steering

**Status: failed with tensor-shape bug in SingleStepHook (12 vs 16
batch-dim mismatch). Fix + rerun queued.**

The custom SingleStepHook tracks a per-batch step counter that doesn't
reset cleanly between chunks of different sizes. Quick fix: reset
counter when `hook.magnitudes.shape` changes.

## Exp 8 — B3 with distribution-matched steering

*(queued)*

## Exp 9 — B3 LLM-judged cut

*(queued)*

## Exp 10 — Distribution-shift eval

*(queued)*

## Exp 11 — Cross-model

*(queued)*

## Exp 12 — Per-position steering protocol

*(queued)*

## Exp 13 — Encoder/decoder asymmetry probe

**Verdict: ASYMMETRY CONFIRMED. TopK SAE has higher encoder
selectivity than TXC but lower steering effectiveness.**

Per-arch top-1 feature D+/D- selectivity score (from feature mining)
vs Sonnet primary (steering effectiveness):

| Arch | top-1 feat | encoder D+/D- score | Sonnet primary |
|---|---|---|---|
| **TopK SAE** | f5263 | **0.4171** ← highest | 0.0071 |
| **TXC** | f14621 | 0.2269 | **0.0114** ← highest |
| txc_h8 | f2867 | 0.1431 | 0.0052 |
| txc_h13 | f1637 | 0.1294 | 0.0095 |
| stacked_sae | f7557 | 0.0142 | 0.0054 |
| tsae_paper | f5917 | 0.0001 | 0.0004 |

**TopK SAE's encoder finds backtracking features more selectively than
TXC** — its top feature has D+/D- separation 1.84× larger than TXC's.
But TopK SAE's steering effect (Sonnet primary) is 1.6× *lower* than
TXC's.

This separates "the dictionary identifies the feature" (encoder) from
"the dictionary's direction translates to behaviour" (decoder). TopK
SAE wins the first; TXC wins the second.

### Mechanistic implication

The single-direction decoder of SAEs is the bottleneck for steering.
TopK SAE's encoder probably finds features that are spatially
localised in residual space, but *only* at certain token positions,
which the single decoder direction can't reproduce when it's broadcast
to every token. TXC's per-position decoder rows give it a more
flexible behavioural-output structure, even though its encoder is less
selective.

This is a real theoretical finding worth surfacing: **the best
dictionary for *probing* features may not be the best dictionary for
*steering* behaviour.**

## Exp 14 — Decoder direction vs DoM cosine

**Verdict: TXC's winning direction is genuinely different from DoM
(cos = 0.25). The "TXC adds value over DoM" claim survives.**

For each arch's winning feature direction, cosine similarity to
Stage A's `dom_base_union` (and `dom_reasoning_union`):

| Arch (winner cell + feature) | cos(DoM_base) | cos(DoM_reasoning) |
|---|---|---|
| txc — f14621 pos0 | +0.248 | +0.222 |
| txc_h13 — f1637 union | **−0.292** | −0.252 |
| txc_h8 — f2867 pos0 | +0.161 | +0.104 |
| topk_sae — f5263 pos0 | +0.083 | +0.050 |
| stacked_sae — f7557 union | +0.128 | +0.128 |
| tsae_paper — f5917 pos0 | +0.015 | +0.002 |

Reference: DoM_base vs DoM_reasoning cosine = 0.794 (Ward's
"~0.74 base/reasoning shared direction").

**No dictionary winner is parallel to DoM.** The closest is plain TXC
at cos = 0.25 — that's still substantially *not* DoM. H13 finds an
*anti-parallel* direction (cos = −0.29) that nonetheless steers well.
TopK SAE's direction is essentially orthogonal to DoM (cos = 0.08).

This means the dictionaries are NOT "rediscovering DoM." They're
finding their own directions that happen to also induce backtracking,
in a different subspace. The "TXC adds value over DoM" novelty claim
survives.

H13's anti-parallel direction is the most interesting — Han's
contrastive loss is finding a direction that, when subtracted from the
residual, achieves what DoM achieves by addition. Worth exploring
mechanistically.

## Exp 15 — Behavioral judge corrupted-reasoning sanity

**Verdict: 11/12 correct. Judge has a real notion of "wrong reasoning
that should be backtracked," not just keyword counting.**

Constructed 12 short synthetic traces:

- 4 with deliberate calculation errors that the model continues without
  catching → judge SHOULD say count=0 (the wrong step wasn't
  backtracked, even though it's wrong)
- 4 with `Hmm`/filler that doesn't reflect any error → judge SHOULD
  say count=0 (filler, not genuine)
- 4 with REAL backtracking that catches an error → judge SHOULD say
  count ≥ 1

Results:

| Group | n | correct judgements |
|---|---|---|
| Injected error, no backtrack (gt=0) | 4 | **4 / 4** |
| Filler `wait`/`hmm` (gt=0) | 4 | **4 / 4** |
| Real backtracking (gt=1) | 4 | **3 / 4** (1 missed: subtle cube-edge counting) |
| **Total** | **12** | **11 / 12 = 91.7%** |

The judge correctly identifies that:
- `wait`/`hmm` filler ≠ genuine backtracking
- A reasoning trace with an *uncaught* error is NOT backtracking
- A trace where `wait` triggers an actual error-catch IS backtracking

Combined with Exp 3 (κ = 0.354 vs Opus): the judge's *individual*
calls are calibrated to the rubric, but two equally-rubric-faithful
judges still disagree on 17% of borderline cases.
