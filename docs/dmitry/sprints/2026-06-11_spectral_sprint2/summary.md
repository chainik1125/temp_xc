---
author: Claude (10h unsupervised sprint #2)
date: 2026-06-11
tags:
  - results
---

## Sprint 2: matching the spectral crosscoder's window to behaviour timescales

### Executive summary

**Goal.** Sprint 1 ([[2026-06-10_freqbench_sprint/summary|sprint-1 summary]])
found that backtracking anticipation in a reasoning model is a low-frequency
state. This sprint answers the follow-ups: (A) does growing the crosscoder's
window — admitting lower frequencies — improve results, (B) can we *screen*
tasks for frequency content and match spectral-crosscoder configurations to
them, and (C) can a multi-agent pipeline propose and red-team new behaviour
candidates. (A *spectral crosscoder* is a sparse dictionary over T-token
windows of residual-stream activations whose atoms are constrained to DCT
frequency bands; its DC branch sees only the window mean. Probes are linear,
on the dictionary's TopK code vector — or a single branch's sub-vector —
with "k atoms/token" the window sparsity budget divided by T.) Three
findings:

**1. Measure the behaviour's timescale with a dictionary-free scan, then
grow the window to match it.** Probing "backtracking imminent" from the
plain mean of the last T tokens traces a curve peaking around T≈32–48
(sentence-to-paragraph scale). Retraining the spectral crosscoder at T=32
instead of 16 (16 atoms/token, single seed) lifts its full-code probe AUC
from 0.795 to **0.831, matching the raw-mean reference at the same window
(0.826; the raw peak is 0.830 at T=48)**. The vanilla window crosscoder
trails at 0.807. One honest nuance the headline hides: the spectral model's
DC *branch* already scored 0.828 at T=16, so window growth mainly lets the
*full code* catch up to its own DC branch (0.835 at T=32) — and what the
dictionary buys over the raw mean at equal AUC is sparse, steerable atoms,
not detection accuracy.

**2. The red team's controls materially corrected the headline — and the
signal survives them.** A probe on position features alone reaches
**0.685 AUC** (backtracking happens at characteristic trace positions; every
number must be read against this floor, not 0.5). A fixed-example-set scan
shows the apparent collapse at T=96 was a **composition artifact**
(fixed-set T=96 is 0.781, not 0.696), leaving a flat, broad optimum.
Scrambled-token pooling changes AUC by **−0.02 to +0.13 relative to
contiguous windows — positive at 5 of 7 window lengths and largest (+0.13)
exactly at the T=32–48 optimum** — so temporal contiguity carries real
signal where it matters, beyond mere averaging.

**3. In every behaviour with decodable signal, the slow bands dominate —
including the candidate engineered to be mid-band.** Screening five
behaviours on DeepSeek-R1-Distill-Llama-8B layer 10: the two with strong
signal (backtracking; conclusion onset) are DC/low-dominated; the rest are
weak, null, or confounded. The 14-agent workflow's top-ranked candidate —
repetition loops, predicted mid-band — was evaluated the same night under
both its protocols: pre-onset anticipation is real but slow (DC 0.642,
above both its lexical control 0.524 and its own position floor 0.567),
while *in-loop* windows are statistically indistinguishable from a raw
token-embedding control at DC (0.652 vs 0.662) — the predicted band
structure lives in the repeated text, not demonstrably in the model's
dynamics. Rule extracted: claims of mid/high-band behaviour should be
presumed lexical until they beat an embedding-level control. A late
cross-domain extension strengthens the slow-state story off-distribution:
reading misaligned responses in an EM-finetuned **Qwen-14B (layer 24,
medical domain)** is decodable at AUC 0.902 from the T=32 window mean
(0.630 at T=1, monotone in T; by-prompt folds; length and steering-scale
confounds excluded) with all non-DC bands anti-generalizing — the most
DC-dominated profile in the table, on a different model, layer, hook and
domain.

![main](figures/fig_s2_main.png)

### 1. Setup

Everything runs on the sprint-1 backtracking stack: 300
DeepSeek-R1-Distill-Llama-8B math reasoning traces, layer-10 residual-stream
cache, "behaviour imminent" probes (positives = offsets [-13,-8] before
marker events; negatives ≥ 25 tokens from any event; by-trace 80/20 splits;
balanced linear probes; AUC). Probe features per arm: the raw T-token
window mean (d=4096); pooled DCT-band coefficients (mean over the band's
frequencies, d=4096 per band); a trained dictionary's TopK code vector
(H=4096 here), or one branch's sub-vector for branch probes. The
position-only probe uses three scalars: absolute position, position/trace
length, and position-within-think-region fraction.

Windows are right-edge and may extend into the prompt, which keeps example
sets *approximately* comparable across T; the residual dependence (the
pos ≥ T−1 cutoff removes early-trace examples as T grows — 270 → 210 test
positives from T=1 to 96) is exactly what the fixed-example-set control
eliminates. Dictionaries: TopK, lr 3e-4, 4k steps, 2 seeds at 2 atoms/token
plus a 16-atoms/token pass (seed 0) for sprint-1 comparability.

### 2. Question A: window scaling

#### 2.1 The dictionary-free timescale curve

| T | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUC (all examples) | .769 | .788 | .815 | .813 | .818 | .814 | .826 | **.830** | .804 | .696 |
| AUC (fixed set, n+=210) | — | — | .740 | .727 | .772 | — | .740 | **.793** | .732 | .781 |
| scrambled control | — | — | .684 | .743 | .700 | — | .665 | .668 | .739 | .716 |

Reading. On all examples the optimum is T≈32–48; on the fixed set the curve
is flat within noise from 16–96 with its maximum still at 48 — and the
all-examples collapse at T=96 disappears (0.781 vs 0.696), confirming it
was example composition, not the state expiring. The contiguous-minus-
scrambled gaps per T are +.056, −.016, +.072, +.075, **+.125**, −.007,
+.065: positive at 5 of 7 lengths and largest at the optimum — the window
mean uses temporal structure precisely where the curve says the state
lives. Position-only probe: 0.685, the leakage floor for this task's
numbers (backtracking events cluster at characteristic trace positions).

#### 2.2 Dictionaries at the matched window

At 16 atoms/token (seed 0): spectral (multiband) full code T=16 0.795 →
T=32 **0.831**; vanilla window crosscoder 0.700 → 0.807. Branch probes:
the DC branch alone scores 0.828 at T=16 and 0.835 at T=32 — it is the
compact detector at both windows, and its own gain from window growth is
small (+0.007). The window-growth story is therefore precise: growing T
mainly lets the *full* spectral code catch up to its DC branch and to the
raw-mean reference at the same T (0.826), while the vanilla crosscoder —
which cannot protect a DC subspace — gains more (+0.107) but stays below.
At 2 atoms/token (2 seeds) the same ordering holds at lower levels
(multiband .73–.78, vanilla .62–.69). DC-SAEs trained directly on window
means (parameters independent of T) are the budget option: they track the
raw curve minus a sparsity tax with high seed variance (.71–.82).

All k=16 cells are single-seed; the DC-SAE seed spread (±.05) is the right
mental error bar, and the 0.831-vs-0.826 match to the raw reference should
be read as "reaches the reference" not "equals it to three digits".

### 3. Question B: the screening table

Same instrument, same model/hook, several behaviours (events from keyword
or programmatic labels on the same traces; HH-RLHF from its own cache):

| behaviour (events) | best raw-mean AUC (T) | T=1 AUC | bands DC/low/mid/high @T=32 | verdict |
|---|---|---|---|---|
| backtracking (227) | .830 (48) | .769 | .835/.708/.615/.649 (branch probes) | strong, slow (floor .685) |
| conclusion onset (158) | .874 (64) | **.854** | .789/.675/.498/.483 | strong, *already per-token*, slow component tentative |
| loop anticipation (~bouts) | — | — | .642/.560/.507/.508 | weak, slow; above lexical ctrl (.524) and own position floor (.567) |
| inside loops | — | — | .652/.620/.536/.519 vs emb ctrl .662/.554/.471/.509 | DC indistinguishable from lexical; low band (.620 vs .554) an unexplained survivor |
| HH-RLHF choice (2000 pairs) | .571 (64) | .538 | .551/.519/.500/.520 | near-null |
| verification (26), uncertainty (1) | — | — | — | insufficient events (null rows, reported) |
| EM misalignment-reading, Qwen-14B L24 (432 resp.) | .902 (32); .994 on T≥64 subset | .630 | .902/.291/.343/.336 | strongly slow; non-DC anti-generalizes |

Patterns, stated at the strength the table supports. (i) **Two behaviours
carry strong signal and both are DC/low-dominated**; the mid/high bands of
backtracking (.615/.649) sit at or below its position floor (.685). No
behaviour shows neural mid/high structure — but only one candidate directly
targeted mid (loops), so this is two data points plus one engineered
failure, not a theorem. (ii) Conclusion onset is a second regime: already
linearized per-token (T=1 AUC .854, like GPT-2 day-stride in sprint 1);
its U-shaped T-curve (.854 → dip → .874 at 64) is unexplained, so its
"slow component" is tentative — window dictionaries add little either way.
(iii) The embedding-level control is mandatory before claiming any non-DC
band: in-loop windows pass band probes but their DC signal is matched by
raw token embeddings — indistinguishable from lexical; only its low band
survives the control (.620 vs .554) and awaits a positional explanation.

#### 3.1 Cross-domain extension: EM misalignment-reading

The workflow's #3 candidate, run as a late extension: 432 judged generations
from the c6 EM replication (medical; gpt-4o alignment/coherence scores;
coherence ≥ 50 filter) teacher-forced through the *unsteered* EM model
(Qwen2.5-14B-Instruct + bad-medical-advice LoRA, merged), layer-24 ln1.
Because many responses were generated under steering, this is a
misalignment-READING screen (does the residual stream reveal that the text
being read is misaligned?), not a generation-onset measurement. Splits are
by prompt (8 uniques, 4 rotations holding out 2); fold ranges are wide and
reported. Results: raw window-mean AUC rises monotonically 0.630 (T=1) →
0.751 (4) → 0.847 (16) → 0.902 (32) → 0.994 (T=64, long-response subset
only); bands DC 0.902 vs low/mid/high 0.29–0.34 (below 0.5 = the probes
anti-generalize across prompts — with 2 held-out prompts, non-DC features
latch onto prompt identity). Confounds checked: length-only probes ≤ 0.45
(misaligned and aligned responses have near-identical mean lengths, 54 vs
58 tokens); steering scale vs label r = −0.05. This is the strongest and
most DC-dominated row in the table, and the first off-math, off-model,
off-layer replication of the slow-state pattern.

### 4. Question C: the candidate pipeline

A 14-agent workflow (4 brainstorm lenses → semantic dedup → 3 ranking
judges → eval designs for top-3 → 3 adversarial red-teamers) produced a
ranked list of real-world behaviours for spectral-crosscoder treatment.
Top five: repetition/rumination loops (8.2/10, mid-band prediction,
programmatic labels), reasoning macro-phase/verification mode (8.0),
emergent-misalignment onset within a generation (7.8),
context-rot/instruction decay (7.5), revision commitment (7.3). The top
pick was evaluated the same night under both its protocols (§3): its
mid-band prediction failed in the way its own red-team warned (lexical
recirculation), while its "slow stuck-precursor" sub-prediction held.

The red team's process critiques are part of the deliverable: judge scores
leaked "similarity to the proven backtracking result" (over-ranking
near-replications); data-availability was triple-counted (biasing toward
one-model, one-domain candidates); mid-band coverage rested on a single
candidate and nothing targeted the high band at all — which makes
"slowness is generic" partly a coverage statement about the brainstorm,
not only about the model. Their demanded controls were all executed (§2.1,
§3) and two materially changed the conclusions.

Recommended next candidate, *not* executable tonight: emergent-misalignment
onset (7.8) — repo-internal c6 generations exist, needs one 30–60 min cache
pass, and its predicted low-band signature would be the first cross-domain
(non-math) test of the slow-state story.

### 5. Limitations

- One model, one layer, one domain for all new rows (the red team's
  "layer-10-everything" confound stands; layer/model sweeps are the top
  follow-up). The HH-RLHF row varies domain but is near-null.
- k=16/token dictionary cells are single-seed; treat ±.05 (the DC-SAE seed
  spread) as the error bar on any single cell.
- Keyword event labels are crude; verification/uncertainty rows died on
  event counts (26 and 1) — judge-based labels are the fix, not run
  tonight.
- The position-leakage floors are high (backtracking .685, loops .567);
  all claims concern the increment above them, and difficulty-matching of
  negatives (a red-team item) was not implemented.
- Band probes use pooled DCT coefficients — a screening statistic, not the
  full band information.
- The in-loop low band (.620 vs embedding .554) survives the lexical
  control and is unexplained; it needs a loop-specific in-bout position
  floor before any interpretation.

### 6. Research map

- H0:00–0:25 scaffolding, branch, preregistrations (peak-T genuinely
  uncertain; spectral T32 ≥ T16 iff raw curve rises — held; task profiles
  differ across tasks — held for *strength*, refuted for *band*; large-T
  example-count risk — vindicated by the composition control).
- H0:15 resilience for user absence: hourly Opus-4.8 cloud takeover routine
  keyed to branch-commit heartbeat; on-pod dead-man timers (billing-safe
  without shipping secrets); results served over HTTPS proxies.
- H0:20–2:30 W-scan (raw curve → headline; DC-SAE; spectral/vanilla cells),
  HH-RLHF screening (near-null), 14-agent workflow (ranked candidates +
  red team). Three bugs found and fixed mid-flight (a no_grad nesting
  crash, a grad-graph leak through embedding weights, an atoms/token
  convention leak), all logged.
- H2:30–4:00 red-team controls executed (position floor, fixed-set scan,
  scrambled control); broadened relabel rows; loops under both protocols
  with embedding-level control; k=16/token pass; pods terminated at H3:32
  (≈ $3.1 before the EM extension; ≈ $4.4 sprint-2 total with the L40S; ≈ $12.5 across both sprints).
- H4:00–5:00 EM cross-domain extension (L40S pod, Qwen-14B+LoRA merge,
  one transformers-5.x chat-template fix; length/scale confound checks run
  before integration; pod terminated).
- H4:00–5:00 writing + two independent review agents (zero-context
  comprehension; adversarial red-team). The red team's 10-issue report
  drove this revision: the "exactly the ceiling" framing was cut
  (single-seed, cross-T), "slowness is generic" was rescoped to what two
  informative behaviours support, the scrambled-gap range was corrected
  from the log's stale value, the §1/§2.1 composition contradiction was
  resolved, and a loop-specific position floor (0.567) was computed
  locally to replace a hand-wave.

Artifacts: `log.md` (timestamped), `STATE.md`, `code/` (bt_wscan,
bt_relabel, bt_loops ×2, bt_controls, hh_screen), `results_synced_ws/`,
`results_synced_hh/`, `figures/fig_s2_main.png`, workflow run
wf_ebf30e77-675.
