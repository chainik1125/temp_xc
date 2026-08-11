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
- *(wave-1 headline pending)*
- *(projector pre-flight pending)*
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

*(wave-1 steering table pending)*

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

#### Results

*(pending)*

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
