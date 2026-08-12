---
author: Dmitry Manning-Coe
date: 2026-08-11
tags:
  - results
---

## Backtracking steering across dictionary architectures

Does a diffusion-trained (DSM) dictionary find a *causally* better backtracking
direction than a reconstruction-trained one, and does denoising the residual
stream after steering change what the steer does? This runs the paper's own c7
steering protocol over five existing dictionaries (wave 1) and the T=6 window
trio (wave 2), and adds a denoise-after-steer variant that no arm in the paper
has.

The companion detection study asked whether these dictionaries can *read*
backtracking. This one asks whether they can *cause* it.

### Headline

- Gate 1 (hook no-op) passes: 0/20 generations differ between no hook and the
  steering hook at magnitude 0.
- Gate 2 (baseline agreement) is **recorded as failed-with-known-offset**. Our
  unsteered mean genuine-event count is 1.35 against the published 0.656, and
  the published coherence filter does not close the gap (1.42 over the 19
  coherent rows of 20). Every comparison in this document is therefore internal
  to this run; no arm's absolute number is compared to a published absolute.
- **Wave 1**: the Stage B temporal crosscoder's slot-0 decoder is the only
  source with directional control over backtracking — a monotone, antisymmetric
  dose-response (gc 2.25 at α = −8, 1.25 at α = 0, 0.25 at α = +10) at flat
  generation length, Δgc = +1.00 [+0.50, +1.50] at the peak.
- **A norm-matched random direction reproduces conventional steering's entire
  effect.** Splitting each curve into its even (magnitude) and odd (direction)
  parts about α = 0, and subtracting the random control at matched |α|, the DoM
  baseline retains a directional component of **+0.015** while the crosscoder
  retains **+0.417**. DoM steering at this site does nothing a random vector of
  the same norm would not also do; the crosscoder does. Without the control the
  raw peak column ranks DoM third, on an effect that is free.
- Mined selectivity does not predict causal potency: the strongest mined feature
  (+0.806) gives one of the weakest effects.
- **DSM dictionaries do not transfer to the deployment distribution, and the
  failure is direction-deep.** At the same NMSE, `w6_dsm` draws its 96 active
  latents per window from a pool of 605 of 16384 (3.7%) on distill activations
  while `w6_recon` keeps 65.1% of its pool alive. Per-latent threshold
  recalibration on distill windows revives recon 8,586 → 16,046 but dsm
  214 → 215, because the DSM encoder's preactivations are almost all negative
  off-distribution. This is one of the central results here — see below.
- *(wave-2 headline pending)*

### What is reused rather than reimplemented

The point of the exercise is that every architecture goes through one pipeline,
and that the pipeline is the published one. Imported unmodified:

- `experiments/ward_backtracking_txc/b1_steer_eval.py` — `_Hook` (activation
  addition on the layer-10 residual), `_generate_panels` (batched greedy
  generation with per-row magnitudes), `_load_lm`, `_eval_prompts`,
  `_normalize_to`, `_kw_rate`, `KEYWORD_RE`.
- `experiments/ward_backtracking_txc/mine_features.py` — the selectivity
  ranking, `score(f) = mean(z[f] | D+) - mean(z[f] | D-)`, top-K by `|score|`,
  and the sentence-to-token anchor alignment (reached through
  `backtracking_detection_dsm/detect_core.py`, which lifted it verbatim).
- `experiments/ward_backtracking_txc/architectures.py` — `build_arch`,
  `arch_encode_window`, `arch_decoder_directions`. Stage B checkpoints are
  rebuilt from the config stored inside each `.pt`, never from a guess.
- `experiments/ward_backtracking_txc/grade_backtracking.py::judge_one` — the
  genuine-backtracking-count judge, and
  `grade_sonnet.py::grade_one` — the 0–3 coherence grade.
- `experiments/ward_backtracking_txc/metrics.py` — `_coh_ok` and
  `_max_repeat_run` for the run-length coherence floor.

New code is confined to what the paper has no equivalent of: loading the
diffusion-arm dictionaries, the wave-2 window layout, and the denoise-after-steer
hook.

### Protocol

- Target: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, steering by activation
  addition of the decoder direction at the output of `model.model.layers[10]`
  (`config.yaml` `steering.steering_layer`). Every source is hooked at this same
  site regardless of the hookpoint its dictionary was trained on — that is the
  paper's own choice and is replicated, not corrected.
- Magnitudes: the reference 25-point grid, pulled from the published artifact
  `dmanningcoe/temp-xc-reviewer-results`,
  `reviewer_seed_audit_2026-07-27/c7_headline/seed42_published_eval.json`
  (`eval_cfg.magnitudes`): −16, −12, −10, −8, −7, −6, −5, −4, −3, −2, −1, −0.5,
  0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16. `config.yaml`'s 9-point grid is
  the older sprint grid and is not used.
- Prompts: the 20-prompt eval split of `results/ward_backtracking/prompts.json`,
  selected by `_eval_prompts` at seed 42. The published headline used a
  61-prompt set that is not in this repo.
- Decoding: greedy, `do_sample=False`, `max_new_tokens=1500`. There is no
  sampling RNG, so the per-prompt-seed rule that governs the EM work does not
  apply here; seed 42 only selects the eval split.
- Every steering vector is rescaled to the norm of the DoM base union
  (0.413977) by `_normalize_to`, so the magnitude axis means the same thing for
  every arm.
- Sources per dictionary follow `_build_sources`: window dictionaries
  contribute both the slot-0 decoder row and the T-averaged union row; flat
  per-token SAEs contribute one vector, because for them `pos0 == union` and the
  second source would be a duplicate.
- Batching: 25 rows per `generate` call, so each call is exactly one prompt's
  magnitude sweep. Every row in a batch shares a prompt length, so there is no
  left-padding in any batch — which also means no padded positions enter the
  wave-2 window buffer.
- Sharding (wave 2): each source's sweep is split across five contiguous
  prompt slices, one container each, and the slices are concatenated back into
  a single row file before judging. Sharding is by prompt rather than by
  magnitude precisely to preserve the no-padding property above: a batch of 20
  *different* prompts at one magnitude must be left-padded, and the projected
  arm's window buffer would then read pad-token activations into the window for
  the first `T-1` real tokens of every row — corrupting the projector input for
  the one arm the denoise-after-steer variant exists to measure, in a way that
  would look like a slightly different curve rather than like a bug. Shard
  bounds are contiguous and complete, so the merged row order is identical to
  the unsharded order.

### Metrics

Two quantities are reported for every arm and every magnitude, because they can
move in opposite directions:

- `gc` — **mean genuine backtracking events per generation**. This is the
  published quantity: `gc_at_baseline` 0.6557377 × 61 prompts = 40 events
  exactly. It is not a per-prompt share.
- `event_rate` — `mean(genuine_count >= 1)`, the share of generations with at
  least one event.

`Δgc(α) = gc(α) − gc(0)` is always computed within a single source, so the
baseline offset described below cancels.

Quality is carried on its own axis, never folded into the headline:

- run floor — `metrics._coh_ok`, max consecutive same-word run ≤ 2.
- Sonnet floor — `grade_sonnet` 0–3 grade ≥ 2.

A magnitude cell counts as coherent only if *every* prompt at that magnitude
passes, which is `metrics.cell_metric`'s own cell-level rule. Generation length
(words, characters, re-tokenised token count) is recorded per row so a Δgc that
is really a length effect is visible as one.

#### The run-length floor does not work on this model, and the Sonnet floor does

The two floors disagree sharply here, and the disagreement is not noise — the
run-length floor is *anti*-correlated with quality at exactly the magnitudes
that matter. Per-magnitude failure counts for the conventional-steering
baseline, out of 20 prompts:

| α | fail run-length | fail Sonnet | mean Sonnet grade |
| --- | --- | --- | --- |
| −16 | 2 | 20 | 1.00 |
| −12 | 1 | 9 | 1.60 |
| −8 | 1 | 0 | 2.55 |
| 0 | 6 | 0 | 2.85 |
| +4 | 2 | 0 | 2.80 |
| +8 | 5 | 6 | 2.00 |
| +12 | 2 | 19 | 1.05 |
| +16 | 0 | 20 | 1.00 |

The Sonnet floor behaves as a coherence floor should: zero failures across the
whole interior of the grid, rising monotonically at both extremes as steering
destroys the generation. The run-length floor fires on 0–6 prompts at *every*
magnitude with no relationship to α — including 6 of 20 at α = 0, where the
model is unsteered — and it passes α = +16 with **zero** failures, the cell
where every single generation is graded 1.

The reason is visible in the generations. `_max_repeat_run` counts consecutive
*identical words*, which catches the "Wait Wait Wait" collapse it was written
for. The degeneration mode here is phrase-level looping, which it cannot see. An
α = +16 generation the run-length floor certifies as coherent
(`max_repeat_run = 1`):

```text
So, she can't. So, maybe she can't. So, she can't. So, she can't. So, s
```

Consequences, and what this document does about them:

- Only 1–3 of 25 cells pass the run-length rule for any source, so a peak
  selected under it is chosen from a handful of survivors and rewards variance.
  Worse, the survivors are biased toward the *most* degenerate magnitudes.
- **The Sonnet floor is therefore the headline coherence gate**, and 11–16 of 25
  cells pass it. The run-length numbers are retained as a reported diagnostic,
  not as the basis for any claim.
- A row-level variant (`gc_rows_coh`, restricted to individually coherent rows
  rather than requiring all 20) is reported alongside, because the all-20 cell
  rule discards whole magnitudes over one or two repetitive generations.

### Gates

Gate 1 — hook no-op. Generations with no hook registered are byte-identical to
generations with the steering hook registered at magnitude 0: 0 of 20 prompts
differ. The hook adds nothing when it should add nothing.

Gate 2 — baseline agreement. **Gate 2 recorded as failed-with-known-offset.**

| quantity | value |
| --- | --- |
| mean genuine events/generation, unfiltered | 1.3500 |
| mean genuine events/generation, coherence-filtered | 1.4211 |
| per-prompt event rate, unfiltered | 0.8500 |
| per-prompt event rate, coherence-filtered | 0.8947 |
| published 61-prompt baseline | 0.6557 |
| sanity band | [0.45, 0.85] |
| coherent rows | 19 of 20 |

The coherence filter was applied on the expectation that degenerate generations
were inflating the count. They were not: exactly one row of twenty fails the
run-length floor, and that row happens to carry a low genuine count, so
filtering moves the baseline further from the band rather than into it. The
offset is 2.06× the published value.

What the row-level data does say: generations are long (median 922 words, 8 of
20 near the 1500-token cap) and the per-category spread is wide — arithmetic
0.00, set_theory 0.50, probability 2.50, algebra_word_problems 2.00, at two
prompts per category. A prompt-mix difference against the absent 61-prompt set
is the most economical explanation and is not testable from this repo. No
attempt is made to close the gap; the consequence is simply that absolutes are
not portable and only within-run differences are read.

### Wave 1 — existing dictionaries

Five dictionaries, mined and steered through the one pipeline. Stage B arms come
from `aniketdesh/ward-stage-b-dictionaries`, rebuilt via `build_arch` from the
config inside each checkpoint; the diffusion arms come from
`dmanningcoe/diffusion-topk-saes`, `llama31-8b-ln1L10-20k/`.

Mining ran on 23,664 judged sentences from the DoM prompt split (3,023 positive),
so the 20 eval prompts are held out of feature selection.

| arm | architecture | trained on | d_sae | T | best feature | selectivity | Welch t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `stageB_topk_sae` | flat TopK SAE | ln1 L10 | 16384 | 6 | 9876 | +0.806 | +19.8 |
| `stageB_txc` | TXC | resid L10 | 16384 | 6 | 14621 | +0.227 | +17.5 |
| `stageB_txc_h13` | TXC-H13 | resid L10 | 16384 | 6 | 1183 | +0.179 | +14.3 |
| `ours_recon_s2` | flat TopK SAE, recon | ln1 L10 | 16384 | 1 | 13776 | +0.492 | +22.0 |
| `ours_dsm_s2` | flat TopK SAE, DSM | ln1 L10 | 16384 | 1 | 4366 | +0.442 | +18.6 |

Selectivity scores are not comparable across architectures — they inherit each
dictionary's own activation scale — so the column is provenance, not a ranking.
The Welch t is scale-free and is the closer thing to a like-for-like read.

#### Results

All eight sources, 20 prompts × 25 magnitudes each, judged for genuine
backtracking count and 0–3 coherence. `Δgc` is against each source's *own*
α = 0 row, so the baseline offset from gate 2 cancels. Peak is taken over the
Sonnet-coherent cells; the CI is a prompt-resampling bootstrap at that peak.

| source | arm | mode | mining score (t) | gc base | Δgc peak | at α | 95% CI | Sonnet-coherent cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stageB_txc_f14621_pos0` | stageB_txc | pos0 | +0.227 (17.5) | 1.25 | **+1.000** | −8 | **[+0.50, +1.50]** | 12 of 25 |
| `ours_dsm_s2_f4366_pos0` | ours_dsm_s2 | pos0 | +0.442 (18.6) | 1.35 | +0.600 | +12 | [+0.05, +1.10] | 15 of 25 |
| `dom_base_union` | dom | dom | n/a | 1.30 | +0.450 | −8 | [−0.05, +0.95] | 15 of 25 |
| `stageB_txc_f14621_union` | stageB_txc | union | +0.227 (17.5) | 1.45 | +0.400 | −6 | [−0.15, +0.90] | 11 of 25 |
| `ours_recon_s2_f13776_pos0` | ours_recon_s2 | pos0 | +0.492 (22.0) | 1.30 | +0.350 | −7 | [−0.15, +0.80] | 16 of 25 |
| `stageB_topk_sae_f9876_pos0` | stageB_topk_sae | pos0 | +0.806 (19.8) | 1.25 | +0.300 | −6 | [−0.10, +0.70] | 16 of 25 |
| `stageB_txc_h13_f1183_pos0` | stageB_txc_h13 | pos0 | +0.179 (14.3) | 1.35 | −0.250 | −0.5 | [−0.60, +0.05] | 14 of 25 |
| `stageB_txc_h13_f1183_union` | stageB_txc_h13 | union | +0.179 (14.3) | 1.40 | −0.250 | −1 | [−0.60, +0.10] | 15 of 25 |

The mining-score column is the selectivity confound: the arms did not enter the
sweep with equally good features, scores spanning +0.179 to +0.806. Note that
the ranking here is close to *inverted* against it — the arm with the strongest
mined feature (`stageB_topk_sae`, +0.806, t = 19.8) produces one of the smallest
causal effects, and the arm with the winning effect has a mid-table score. Mined
selectivity does not predict causal potency in this setup.

#### The Stage B crosscoder is the only arm with directional control

The headline is not the peak height, it is the shape of the curve. For
`stageB_txc` slot 0 the dose-response is monotone and antisymmetric through the
whole coherent region:

| α | −10 | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 | +10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gc | 2.25 | 2.25 | 1.75 | 1.50 | 1.25 | 1.25 | 1.20 | 1.15 | 0.90 | 0.70 | 0.25 |
| mean words | 943 | 908 | 906 | 867 | 830 | 862 | 852 | 898 | 922 | 969 | 989 |

Negative α nearly doubles the genuine-event count; positive α drives it toward
zero. Generation length is flat across that range (830–990 words), so this is
not a length effect, and at α = −8 all 20 prompts pass the coherence floor, so
it is not a coherence-selection effect either.

The conventional DoM baseline behaves completely differently — it is **U-shaped
in |α|**, raising the count in *both* directions (1.75 at α = −8, 1.85 at
α = +8, against 1.30 at α = 0). Raising the target behaviour whichever way you
push is the signature of a norm perturbation, not of a direction that encodes
the behaviour. On this evidence the temporal crosscoder's slot-0 decoder is a
genuine control knob for backtracking and conventional steering at this site is
not, even though their peak Δgc values (+1.00 vs +0.45) differ by less than a
factor of three.

#### The random-direction control: conventional steering is nonspecific

`control_random` is a random unit vector at the same site, rescaled to the same
DoM base-union norm as every real source, swept over the same grid and prompts.
Its own curve:

| α | −10 | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 | +10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gc | 1.50 | 1.20 | 1.45 | 1.55 | 1.45 | 1.30 | 1.30 | 1.30 | 1.40 | 1.55 | 1.75 |
| mean words | 908 | 920 | 900 | 914 | 876 | 862 | 862 | 821 | 880 | 858 | 918 |

Generation length is flat, so the control passes the same length check the real
arms do. Its peak Δgc at the Sonnet floor is **+0.450, CI [−0.05, +0.95]** —
numerically identical to `dom_base_union`'s +0.450 [−0.05, +0.95]. It is also
the *least* damaging source in the study, with 20 of 25 cells clearing the
Sonnet floor (more than any mined direction) and a `gc` minimum of 1.15 against
dom's 0.65: a random direction of this norm perturbs the count without ever
destroying the generation.

Decomposed into even and odd parts about α = 0:

| source | sym (magnitude) | anti (direction) | excess sym vs control | **excess anti vs control** |
| --- | --- | --- | --- | --- |
| `stageB_txc_f14621_pos0` | −0.040 | +0.406 | −0.187 | **+0.417** |
| `stageB_txc_f14621_union` | −0.210 | +0.381 | −0.358 | **+0.392** |
| `stageB_topk_sae_f9876_pos0` | +0.117 | +0.058 | −0.031 | +0.069 |
| `stageB_txc_h13_f1183_union` | −0.004 | +0.042 | −0.152 | +0.052 |
| `stageB_txc_h13_f1183_pos0` | +0.017 | +0.037 | −0.131 | +0.048 |
| `ours_recon_s2_f13776_pos0` | +0.027 | +0.027 | −0.121 | +0.038 |
| `dom_base_union` | +0.113 | +0.004 | −0.035 | **+0.015** |
| `ours_dsm_s2_f4366_pos0` | +0.023 | −0.135 | −0.125 | −0.125 |
| `control_random` | +0.148 | −0.010 | — | — |

The control answers the question it was added for, and the answer is the
uncomfortable one for conventional steering. A norm-matched *random* direction
reproduces the U-shape — its symmetric component (+0.148) is in fact slightly
**larger** than the DoM baseline's (+0.113) — and both have an antisymmetric
component indistinguishable from zero (−0.010 and +0.004). Subtracting the
control at matched |α|, conventional DoM steering retains an excess directional
component of **+0.015**. On this evidence, DoM steering at layer 10 of this
model does nothing a random vector of the same norm would not also do.

Against that baseline the Stage B crosscoder's slot-0 direction retains an
excess directional component of **+0.417**, roughly 28× dom's and 6× the next
best arm. The union mode of the same feature retains +0.392. Every other
source, including both of our own flat dictionaries, sits at or below +0.069.

This is the strongest claim in the study, and note that it *only* becomes
available with the control: the raw peak Δgc column ranks `dom_base_union`
third, ahead of five mined directions, purely on a nonspecific effect that the
control shows is available for free.

#### What this does not show

- **Suppression is confounded with degeneration.** The induction side is clean:
  α = −8 is fully coherent and the effect is large. The suppression side is not
  — gc falls monotonically as α rises, but the magnitudes where it falls
  furthest (+10 and beyond, gc → 0.05) are exactly the ones failing the Sonnet
  floor. Within the fully coherent region suppression is small: 1.25 → 1.15 at
  α = +4. "Suppresses backtracking to near zero" is not a supported claim; "gc
  decreases monotonically with α, and the strong suppression coincides with
  coherence collapse" is.
- **The sign is inverted against mining.** Feature 14621 was selected for
  *higher* activation on backtracking sentences (score +0.227), yet it is
  *negative* α that induces backtracking. Steering along the +decoder direction
  suppresses. This is reported as found; no attempt is made to redefine the sign
  to make it read more naturally.
- **Multiple comparisons are uncorrected.** Eight sources, each with a peak
  chosen over 11–16 coherent magnitudes. One nominal 95% CI excluding zero
  across that many selections is weak on its own; `stageB_txc` pos0 is
  comfortable ([+0.50, +1.50]) and `ours_dsm_s2` is marginal ([+0.05, +1.10]).
  The monotone dose-response, not the CI, is what makes the crosscoder result
  credible.
- **n = 20 prompts.** Every CI here is wide. These are ordering claims among
  arms run through one pipeline, not calibrated effect sizes.

### Wave 2 — window dictionaries and denoise-after-steer

#### How the wave-2 checkpoints are defined

The T=6 trio (`w6_recon`, `w6_dsm`, `w6_bayes`, all `resid_L10`, `H=16384`,
window `L0=96`) was launched with a 20,000-step plan under a 6-hour Modal
function timeout. At the observed ~1.27 s/step that plan is not reachable, so
the arms end by **timeout truncation**, and the trainer's `_final.json`
completion marker — written only on a clean finish — never appears. The usable
checkpoint is the periodic `.pt`, rewritten every 5,000 steps, so the last one
to land is **step 15,000**.

These arms are therefore reported as *15k steps (timeout-truncated from a 20k
plan; incidentally stage-B-budget-matched, since stage B's own config also
trains 15k)*. They are never labelled as the 20k arm.

One implementation note worth recording, because the failure mode is silent.
The trainer writes the checkpoint locally and a separate thread commits the
volume every 300 s, so there is a window in which the training log has passed
step 15,000 while the `.pt` visible on the volume is still the step-10,000 one.
Accepting a checkpoint the moment the log crosses the threshold would load a
10k dictionary and label it 15k, with nothing anomalous in any downstream
number. `w6.resolve_ckpt` therefore requires the log to be 750 rows *past* the
checkpoint write — longer than one commit interval — before it accepts.

#### Projector pre-flight — the denoise-after-steer arm is compromised

The denoise-after-steer variant projects the steered residual stream through
`w6_dsm`'s denoising map. That only tests anything if the projector is alive at
the steering site, so it is measured before the grid rather than inferred from
it. The dictionaries are trained on FineWeb through *base* Llama-3.1-8B; the
steering site is *DeepSeek-R1-Distill* on reasoning traces.

Gating numbers, step-15000 checkpoints, 78,395 distill tokens → 20,000 windows
at `resid_L10`. NMSE is `psc_train_sae.py`'s own definition on raw unnormalised
activations, so the two NMSE columns are directly comparable:

| arm | NMSE on distill | NMSE at training site | live latents | L0 | train dead_frac |
| --- | --- | --- | --- | --- | --- |
| `w6_recon` | 0.853 | 0.046 | 10673 / 16384 (65.1%) | 96.0 | 0.009 |
| `w6_dsm` | **0.795** | 0.061 | **605 / 16384 (3.7%)** | 95.9 | 0.012 |
| `w6_bayes` | 1.917 | 0.060 | 8521 / 16384 (52.0%) | 131.9 | 0.947 |

Training length is not the issue. An interim pre-flight on the step-5000
checkpoints gave 0.847 / **0.815** / 2.259 with a `w6_dsm` live fraction of
3.8%; ten thousand further steps moved `w6_dsm` to 0.795 at 3.7% live. The
transfer failure is a property of the objective, not of undertraining.

`w6_dsm` explains about 20% of the variance at the steering site against ~94% at
its training site, and fires only 605 distinct latents across 20,000 windows.
Since a TopK dictionary with `k = 96` fires exactly 96 latents per window by
construction, that means every window is being reconstructed from the same
~605-atom sub-dictionary. `w6_recon`, at essentially the same NMSE, keeps 65% of
its latents alive — so this is a property of the DSM objective, not a shared
distribution-shift effect. `w6_bayes` at NMSE 1.92 reconstructs *worse than
predicting the mean window*, which with its 0.947 training dead fraction makes
it non-functional as a reconstruction; its L0 of 131.9 also overshoots the
target of 96.

#### Objective-dependent OOD collapse — a primary result

The live-pool contrast is not a side-effect of this study, it is one of its main
findings. Three probes now show the same thing at different scales and
hookpoints:

| probe | scale | DSM | recon |
| --- | --- | --- | --- |
| backtracking detection (per-token dictionaries, ln1 L10) | token | ~50% dead on distill traces | ~10% dead |
| this pre-flight (T=6 windows, resid L10) | window | 605 / 16384 live (3.7%) | 10673 / 16384 (65.1%) |
| recalibration probe (T=6 windows, resid L10) | window | 214 → 215 live after distill-side recalibration | 8586 → 16046 |

Both objectives train to comparable fidelity on their own distribution (NMSE
0.048 vs 0.061; both ~0% dead on FineWeb). The difference appears only off
distribution, and it is asymmetric: the reconstruction dictionary's failure is a
*threshold* problem that per-latent recalibration repairs almost completely,
while the denoising dictionary's failure is a *direction* problem that
recalibration cannot touch, because its preactivations go negative almost
everywhere. Whatever DSM buys — and the detection and absorption results say it
buys real things — it buys by specialising hard to the activation distribution
it was trained on.

This directly motivates trace-domain or mixed-corpus training, and it is the
concrete reason a DSM dictionary trained on FineWeb through base Llama cannot be
used as a projector on reasoning traces through the distill model.

**How strong is the convergence?** These are three probes, not three independent
replications: they share the same deployment distribution (distill traces) and
differ in probe design, scale and hookpoint. A fourth line on a second
off-distribution corpus would be needed to call it airtight, which belongs with
the mixed-corpus training round rather than here.

**Is the collapse threshold-shallow or direction-deep?** A separate
recalibration probe answers this, and the answer is direction-deep. Replacing
TopK-96 with a per-latent threshold gate calibrated on distill windows (θ_i at
the `1 − 96/16384` quantile of each latent's own preactivation, 60/40
calibration/eval, 27,900 windows) revives `w6_dsm` from a live pool of 214 to
215 — that is, not at all — at a recalibrated mean L0 of 1.36 against the target
of 96. The encoder's preactivations are almost entirely negative on distill
windows, roughly 1.4 latents above zero per window, so no choice of threshold
can help: the directions themselves do not respond off-distribution. The same
probe run on `w6_recon` as an instrument control revives 8,586 → 16,046 of
16,384 at L0 111 ≈ target, for a mild NMSE cost (0.238 → 0.278). So for a
density-blind dictionary the collapse *is* shallow and recalibration fixes it;
for the DSM dictionary it is not. A recalibrated-projector variant was
considered and dropped on this evidence.

**Pre-registered reading of the variant** (fixed before the arm was run):

> At pre-flight NMSE 0.815 on the steering distribution, this variant primarily
> measures projector robustness under distribution shift. The α = 0
> projected-vs-unprojected comparison is the confound control: if projected
> α = 0 generations are already degraded, the arm measures projector damage, not
> steering dynamics. A POSITIVE result (extended coherent range despite the
> shift) would be strong evidence for manifold-projected steering; a negative or
> flattened result bounds the method to domain-matched projectors and does not
> speak to temporal structure.

Quality-vs-α leads the reporting; a flattened Δgc is never read as evidence
about temporal structure.

One caution on comparing pre-flight NMSE figures across probes: the
recalibration probe reports a `w6_dsm` TopK NMSE of 0.259, far below the 0.815
here, but with only ~1.4 positive latents per window that number is mostly
`b_dec` plus one or two strong latents rather than feature transfer. The probes
also differ in capture (teacher-forced full text vs generation-time rolling
windows) and in checkpoint step. A low NMSE is not evidence of manifold
competence unless the live-pool count is read alongside it.

#### Wave-2 reading rules

Wave 2 is read under exactly the wave-1 rules, so the two tables are
comparable:

- Headline is Δgc at the **Sonnet** coherence floor; run-length is a reported
  diagnostic only. Both floors' cell pass-counts appear in the table so the
  selection base stays auditable.
- The row-level coherent variant is reported alongside, filtered on the Sonnet
  grade and requiring at least half the prompts to survive.
- The suppression/degeneration confound, the possibility of a steering sign
  inverted against the mining sign, the uncorrected multiple comparisons across
  sources × magnitudes, and the n = 20 prompt width all carry over unchanged.
- `w6_bayes` is included to complete the matrix but is a **labelled-degenerate
  arm**: 94.7% of its latents are dead at the end of training and it
  reconstructs distill windows worse than the mean (NMSE 1.92). "What does a
  collapsed dictionary's best feature do under steering" is a legitimate
  datapoint; it is not evidence about the gated objective done properly, and its
  row carries the degenerate label in every table.

  The cause is known and is not window-ness. This arm was trained with the
  mean-gate-style sparsity controller, whose collapse mode is exactly this; the
  per-latent rate-KL fix developed the same day removes it (alive 1.0, dead 0.0
  at three separate L0 targets in the Gemma-scale runs). A `w6_bayes` retrained
  with rate-KL is expected to be non-degenerate, so nothing here should be read
  as evidence against gated dictionaries at T=6.

#### Wave-2 results

Pending — the T=6 arms truncate at ~20:20 EDT and the grid runs after that.

### Files

- `experiments/backtracking_steering_dsm/steer_core.py` — dictionary loading,
  mining, steering-source construction.
- `experiments/backtracking_steering_dsm/w6.py` — T=6 window dictionaries and
  the `SteerDenoiseHook`.
- `experiments/backtracking_steering_dsm/test_w6_hook.py` — proves the
  decode-time window buffer matches a full-sequence projection.
- `experiments/backtracking_steering_dsm/modal_mine.py`,
  `modal_steer.py`, `modal_wave2.py`, `modal_judge.py` — the Modal entrypoints.
- `experiments/backtracking_steering_dsm/coh_check.py`, `analyse.py` — the
  CPU-side gate check and wave aggregation.
- Artifacts on the `diffusion-txc` volume under `backtracking_eval/steering/`:
  `gates.json`, `coh_check.json`, `features/`, `wave1/`, `wave2/`.
