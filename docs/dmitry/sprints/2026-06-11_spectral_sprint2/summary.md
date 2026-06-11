---
author: Claude (10h unsupervised sprint #2)
date: 2026-06-11
tags:
  - results
---

## Sprint 2: matching the spectral crosscoder's window to behaviour timescales

### Executive summary

[written last]

### 1. Questions

Sprint 1 established that backtracking anticipation in
DeepSeek-R1-Distill-Llama-8B is a low-frequency state (DC-band probes beat
whole crosscoders; the best steering feature is spectrally slow). This sprint
asks the two follow-ups the result invites:

- **A (window scaling).** If the signal is slow, longer windows admit lower
  frequencies — does the spectral crosscoder improve with window length T,
  and where does performance peak? The peak location is itself a
  measurement: the timescale of the backtracking state.
- **B (task screening).** Measure the *label-frequency profile* of several
  tasks (optimal averaging horizon + band-resolved probes) and match
  spectral-crosscoder configurations to them.
- **C (candidate pipeline).** A multi-agent workflow (brainstorm → rank →
  eval-design → red-team) proposing real-world behaviours where the spectral
  crosscoder should pay off, with its top pick evaluated the same night and
  its red-team controls executed.

### 2. The timescale measurement (Question A)

The cheapest instrument is dictionary-free: probe "backtracking imminent"
(D+ = 8–13 tokens before a Wait/Hmm event, vs far-from-event negatives,
by-trace splits) from the **mean of the last T tokens** of layer-10 residual
stream, sweeping T:

| T | 1 | 2 | 4 | 8 | 16 | 24 | 32 | **48** | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC | .769 | .788 | .815 | .813 | .818 | .814 | .826 | **.830** | .804 | .696 |

The anticipatory state is averageable over ~30–50 tokens
(sentence-to-paragraph scale): beyond sprint-1's T=16 there is modest
headroom (+0.012 AUC at T=48), and far beyond, the window outlives the state.
[CONTROLS PENDING: fixed-eval-set curve (composition artifact check),
scrambled-window control (denoising vs temporal structure), position-only
ceiling — results to be inserted.]

Dictionary arms: [DC-SAE results — partial: noisy across seeds, ~tracks the
raw curve with a sparsity tax, best at T=64: 0.80–0.82]. [Spectral multiband
vs vanilla TXC at T ∈ {16, 32}, param-matched — PENDING.]

### 3. The screening table (Question B)

Same instrument across tasks (same model, hook, probe protocol where
applicable):

| task | best raw-mean AUC (at T) | band profile (DC/low/mid/high @T=32) | verdict |
|---|---|---|---|
| backtracking anticipation | 0.830 (T=48) | [pending] | strong, slow |
| verification-mode onset | [pending] | [pending] | |
| conclusion/commitment onset | [pending] | [pending] | |
| uncertainty-marker onset | [pending] | [pending] | |
| repetition-loop onset | [pending] | [pending] + layer-0 control | predicted MID |
| HH-RLHF chosen-vs-rejected | 0.571 (T=64) | .551/.519/.500/.520 | near-null, faint DC |
| GPT-2 day-stride (sprint 1) | — | conversion at embedding layer | calibration |
| synthetic circle (sprint 1) | — | branch-matched | calibration |

### 4. The behaviour-candidate pipeline (Question C)

14-agent workflow output (4 brainstorm lenses → dedup → 3 judges → 3 eval
designs → 3 red-teamers). Top-ranked candidates:

1. **Repetition/rumination loop onset** (8.2/10) — mid-band prediction,
   programmatic 6-gram labels; the first non-DC test of the method.
   [EVALUATED TONIGHT — results pending.]
2. **Reasoning macro-phase / verification-mode onset** (8.0) — DC/slow;
   cheap keyword version evaluated tonight as the verification row.
3. **EM onset within a generation** (7.8) — low band; needs a 30–60 min
   cache pass over stored c6 generations (left as next step).
4. Context-rot (7.5), revision commitment (7.3), sycophantic capitulation
   (7.3), answer commitment (7.2 — partially covered by the conclusion row).

Red-team highlights (all actionable ones adopted): scrambled-window control;
fixed-eval-set T-curve; position-only leakage ceiling; layer-0 embedding
control for loops; fair-baseline caveat on sprint-1's PR-AUC@8 claim (band
selection used label knowledge — though the branch was chosen on independent
distill-cache data before the c7 eval); ranking-process critiques
(data-availability triple-counting biases toward one-model one-domain
candidates; no high-band candidates proposed — the taxonomy's coverage gap).

### 5. Limitations and checks

[TBD]

### 6. Research map

[TBD]
