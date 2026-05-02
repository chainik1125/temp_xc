---
author: Aniket Deshpande
date: 2026-05-02
tags:
  - proposal
  - in-progress
  - ward-backtracking
---

## Why this doc

Stage B has shipped a case-study verdict: TXC family beats SAE family
on Sonnet primary + behavioral judge + per-magnitude productive-rate;
B3 MATH-500 returns a negative behavioural result; multi-seed shows TXC
≈ H13 on s42 and beyond. To get this to paper-quality and to satisfy
Dmitry's "shoot down aggressively" frame, we need to *try to invalidate
each claim* before publishing.

This plans doc enumerates 15 experiments across 5 tiers. Each
experiment has a hypothesis, a falsifiable predicted result, the
specific compute / API budget, and the failure mode it could surface.
Companion writeup [[summaries|summaries]] lands the results as they
complete.

The ordering is **most likely to invalidate a current claim first**,
not "most likely to confirm." That's the rigour Dmitry asked for.

## Tier 0 — audits before more experiments

### Exp 1 — Audit B3 mag=0 control

- **Hypothesis being tested.** B3 reported a mag=0 control rescue rate
  of 11.1% (8/72), which is what we used as the baseline that steering
  fell below. But greedy decoding from the same prefix tokens *should
  be* deterministic — re-running the second half of an unsteered
  trajectory should produce identical text and identical wrong answer
  → mag=0 rescue should be 0%, not 11.1%.
- **Falsifying experiment.** Pick 3 wrong-answer trajectories from
  `phase1_unsteered.json`. For each, manually rebuild the cut+continue
  input (chat template + first-half tokens), generate greedily with
  hook removed (or mag=0), compare token-by-token to the second half
  of the original `unsteered_text`. If they diverge, the divergence
  point reveals whether it's bf16 batch-padding drift or a real
  control bug.
- **Compute.** Trivial — manual `torch.equal` + token diff. ~30 min.
- **Failure surfaces.** If divergence is widespread, the entire B3
  negative result (steering hurts vs control) needs to be reframed as
  "vs noisy control" not "vs deterministic baseline." If divergence is
  rare, the 11.1% is a real intrinsic-recovery rate.

### Exp 2 — Bootstrap CIs on per-magnitude frac_prod

- **Hypothesis.** TopK SAE peak (frac_prod=0.500 at +4) really is
  higher than TXC peak (0.338 at +8). The gap looks large but n=80
  for TopK SAE vs n=160 for TXC.
- **Falsifying experiment.** 1000-resample bootstrap on per-(arch,
  magnitude) frac_prod, prompt-level resampling. Report 95% CI per
  point. If TopK SAE 0.500 CI overlaps TXC 0.338 CI, the "TopK SAE
  peaks higher" claim is single-sample noise.
- **Compute.** ~10 min CPU; same vectorized bootstrap as
  `metrics._bootstrap_ci`.
- **Failure surfaces.** If overlapping CIs, the per-mag plot's
  reframe of "TopK SAE peaks higher" needs to be downgraded to "TopK
  SAE within bootstrap noise of TXC at peak."

### Exp 3 — Cross-judge validation (Sonnet vs Opus 4.7)

- **Hypothesis.** Sonnet 4.6's behavioral judgements (`genuine_count`)
  are calibrated. We hand-spot-checked 5 zero-counts and Sonnet agreed
  with us, but n=5 is too small.
- **Falsifying experiment.** Re-grade the 98-row calibration sample
  + 200 randomly-sampled rows from the canonical B1 with Claude Opus
  4.7 (or GPT-4o) using the *same* judge prompt. Compute Cohen's κ on
  the binary {count ≥ 1} judgement vs Sonnet. If κ < 0.5, the judge
  is single-model bias.
- **Compute.** ~$5-10 API spend. ~30 min wall.
- **Failure surfaces.** If low κ, every claim that depends on the
  judge (genuine_backtracking_rate, frac_prod, the 4-way breakdown)
  is suspect; we'd need a panel-of-judges aggregate. If high κ, the
  judge is robust and we can lean on it more confidently.

## Tier 1 — break or confirm the architecture claim

### Exp 4 — Multi-seed SAE family

- **Hypothesis.** TopK SAE's mag=+4 peak (0.500 frac_prod) holds across
  seeds. Without this, the per-mag plot's "TopK SAE peaks higher than
  TXC" reframe is single-seed luck, not a real architectural property.
- **Falsifying experiment.** Train `topk_sae__ln1_L10__k64` and
  `stacked_sae__resid_L10__k16` at seeds {7, 11, 23} (s42 already done).
  4 cells each → 8 new cells. Run B1 + Sonnet grade + behavioral judge.
  Compute mean ± SD of mag=+4 frac_prod per arch.
- **Compute.** 8 cells × ~25 min on H100 (sequential per GPU; 2× H100
  in parallel) ≈ 1.7 hr GPU + ~$5 API for grading.
- **Failure surfaces.** If TopK SAE's peak frac_prod has SD > 0.15
  across 4 seeds, the apparent TopK SAE > TXC at peak is overstated.

### Exp 5 — Held-out B1 prompt set (anti-overfitting)

- **Hypothesis.** Our 20-prompt eval split has been used to select
  metrics, thresholds, and the headline cell. There's a real risk
  we've engineered a metric that works on these 20 prompts.
- **Falsifying experiment.** Sample 20 *new* held-out prompts from
  Stage A's pool (disjoint from the original eval set). Re-run B1
  on the headline cells (TXC k=16 winner, TopK SAE k=64, Stacked SAE
  k=16, H13 k=16, TSAE-paper k=32). Compare per-arch ordering on
  the held-out set vs original.
- **Compute.** 5 cells × ~25 min B1 ≈ 2 hr GPU.
- **Failure surfaces.** If TXC's lead inverts on held-out prompts,
  we've been overfitting; if it holds, the metric is robust.

## Tier 2 — resolve the B3 negative interpretation

Three candidate explanations for why steering induces backtracking
*text* but reduces MATH-500 *correctness*; each has its own falsifying
experiment.

### Exp 6 — B3 cut-at-25%

- **Hypothesis (3) from B3 writeup.** 50% midpoint is past the model's
  commitment to wrong reasoning; intervention earlier (25%, before
  commitment) gives steering time to redirect.
- **Falsifying experiment.** Same B3 harness, cut at 25% token mark
  instead of 50%. Same magnitudes {0, -8, -12, +8}.
- **Compute.** ~30 min GPU (same as B3).
- **Failure surfaces.** If still mag=0 ≥ steered, hypothesis (3) is
  false and we should look elsewhere.

### Exp 7 — B3 single-position steering

- **Hypothesis (2) from B3 writeup.** Continuous-on hook applies the
  steering vector at every token after the cut, which over-applies
  the signal and disrupts the model's intrinsic recovery. A
  single-token nudge at the cut position should behave more like the
  natural-conversation steering Ward 2025 actually used.
- **Falsifying experiment.** Modify the hook to fire on exactly one
  token (the first generation step after the cut) and pass-through
  thereafter. Re-run B3.
- **Compute.** ~30 min GPU.
- **Failure surfaces.** If single-position still hurts vs control,
  the steering direction itself is not productive for math, not just
  the protocol.

### Exp 8 — B3 with distribution-matched steering vector

- **Hypothesis (1) from B3 writeup.** Stage A's DoM was computed on
  CS/programming-style traces; MATH-500 is algebraic / number theory.
  The "backtracking direction" may be domain-specific.
- **Falsifying experiment.** Generate a small math-domain reasoning
  trace corpus (50-100 traces from MATH-500 problems, with `<wait>`/
  `<hmm>` annotations à la Ward §B.2). Re-derive a DoM direction from
  these. Re-run B3 with this math-DoM as the steering vector.
- **Compute.** ~3 hr GPU (cache acts at L10, derive DoM, re-run B3).
  ~$2 for sentence-labeling via Sonnet.
- **Failure surfaces.** If math-DoM also fails, the issue is
  steering-induced disruption (item 2) not domain mismatch (item 1).

### Exp 9 — B3 LLM-judged cut point

- **Hypothesis.** Cutting at the *specific wrong reasoning step* (not
  arbitrary midpoint) is the principled comparison. Our 50%-token cut
  is a proxy.
- **Falsifying experiment.** For each wrong unsteered trajectory, ask
  Sonnet 4.6 to identify the FIRST step where the reasoning goes
  wrong (token range). Cut there. Re-run B3.
- **Compute.** ~$10 API for cut-point identification + ~30 min GPU.
- **Failure surfaces.** Same as #6 — if steering still hurts vs
  control at the right cut point, the steering direction is bad for
  math.

## Tier 3 — generalization tests (paper-completion)

### Exp 10 — Distribution-shift eval

- **Hypothesis.** TXC's lead is general across reasoning task
  distributions, not specific to Stage A's CS-prompts.
- **Falsifying experiment.** Evaluate the headline cells on 50 prompts
  each from GSM8K (math word problems), HumanEval (code), MMLU
  (instruction-following). Use the same B1 protocol. Per-arch ordering
  per task type.
- **Compute.** ~6 hr GPU (5 cells × 3 task types × ~25 min B1) +
  ~$15 API.
- **Failure surfaces.** If TXC inverts on coding or math but holds on
  instruction-following, the case-study claim is Stage-A-specific.

### Exp 11 — Cross-model

- **Hypothesis.** TXC's lead is not a Llama-3.1-8B → DeepSeek-R1
  artifact.
- **Falsifying experiment.** Repeat the B1 sweep with a different
  base/reasoning pair: e.g. Qwen2.5-7B → Qwen2.5-7B-Instruct-Reasoning
  (or a similar publicly-available distillation pair). Train fresh
  TXC + TopK SAE at L10 of the new base; mine + B1 + grade.
- **Compute.** ~10 hr GPU (cache new base activations + train 4
  cells + B1 + grade).
- **Failure surfaces.** If TXC's lead disappears on the new pair, it's
  finetuning-recipe-specific.

### Exp 12 — Per-position steering protocol

- **Hypothesis.** TXC's "broadcast same vector to all positions"
  protocol is leaving signal on the table. Han's per-position protocol
  (different vector at each token slot, weighted by where in the
  T-window the position lies) should improve TXC's productive rate.
- **Falsifying experiment.** Modify `b1_steer_eval._Hook` to apply
  TXC's `decoder_per_pos[t, :, feat]` at token-position `t mod T`
  instead of a single vector. Re-run B1 on the headline TXC cell. If
  productive rate increases beyond plain-TXC numbers, the per-position
  protocol matters; if not, "single direction averaged" was already
  capturing the signal.
- **Compute.** ~3 hr GPU (re-run B1 on TXC k=16 winner).
- **Failure surfaces.** Either way is informative — null result means
  the per-position decoder rows of TXC are mostly redundant; positive
  result means we should re-run all TXC B1 with the new protocol.

## Tier 4 — theoretical / mechanism

### Exp 13 — Encoder/decoder asymmetry probe

- **Hypothesis.** TXC's encoder finds backtracking features fine
  (mining ranks them with high D+/D- selectivity), but its decoder
  directions don't propagate cleanly to behaviour because attention
  smooths them across positions in a way that washes out the
  per-slot signal we trained for.
- **Falsifying experiment.** For each (cell, top feature), compute:
  (a) encoder activation profile across token positions during a
  full-prompt forward, (b) projection of residual onto
  `decoder_at_pos0` direction at each token. Plot both per token.
  If the encoder peaks at the correct backtracking offset but the
  decoder projection is uniformly low, the asymmetry is mechanical
  (decoder is a single-direction probe of a high-dim feature).
- **Compute.** ~30 min CPU (forward pass over 100 generation traces).
- **Failure surfaces.** Confirmation strengthens the
  encoder-decoder-mismatch story; null result means feature directions
  are actually well-localized in residual.

### Exp 14 — Decoder direction vs DoM cosine

- **Hypothesis.** TXC's headline feature direction is *substantially
  different* from Stage A's DoM. If they're nearly parallel
  (cos > 0.95), TXC's "lead" reduces to "TXC found the same DoM
  direction."
- **Falsifying experiment.** For each architecture's winning feature
  direction (the one that produces the Sonnet primary), compute cosine
  similarity to `dom_base_union` and `dom_reasoning_union`. Tabulate.
- **Compute.** ~5 min CPU.
- **Failure surfaces.** If TXC's winner has cos > 0.9 with DoM, our
  novelty claim is weak; if cos < 0.5, TXC is finding genuinely
  different signal.

### Exp 15 — Behavioral judge sanity: corrupted reasoning

- **Hypothesis.** The behavioral judge has a real notion of "wrong
  reasoning that should be backtracked" — not just "did the text
  contain `wait`/`hmm`."
- **Falsifying experiment.** Construct synthetic reasoning trajectories
  where we deliberately inject calculation errors (e.g., "1024 × 7 =
  8000" then continue); query the judge: "Does the model recognize
  this as a step it should backtrack from?" Measure: does the judge
  flag deliberately-wrong-but-not-backtracked trajectories
  differently from correct-reasoning-with-`wait`-fillers?
- **Compute.** ~$5 API. ~30 min wall (write 30 corrupted traces +
  judge them).
- **Failure surfaces.** If the judge can't distinguish injected errors
  from filler-wait traces, the "judge counts genuine backtracking"
  story is overstated.

## Compute budget

| Tier | wall (GPU-h) | API ($) |
|---|---|---|
| 0 (audits) | 0 | 5-10 |
| 1 (multi-seed + held-out) | ~5 | 5 |
| 2 (B3 variants) | ~5 | 12 |
| 3 (generalization) | ~19 | 15 |
| 4 (theory/mech) | ~1 | 5 |
| **Total** | **~30** | **~$50** |

Well under the project's compute budget. The Tier 0 + Tier 1 audits
(~5 hr GPU + $15 API) are the rigorous core; Tier 3 is paper-polish.

## Why this ordering matters

If we ran Tier 3 (generalization) before Tier 0 (audits), we'd be
spending big compute to confirm a claim that might fail Tier 0's
metric audit. The Tier 0 + Tier 1 + Tier 4 layer is "stress-test the
existing claims." Tier 2 + Tier 3 layer is "expand the claim." Doing
the stress test first means we know what we're expanding.
