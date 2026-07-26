---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - reference
  - in-progress
---

## Catalogue of candidate behaviours for TXC vs TopK SAE vs tSAE

Living document, updated through the sprint. Purpose: close the relevance gap left by
[[summary|the 2026-07-25 dictbench sprint]] — a real crosscoder advantage was found, but only
on a synthetic construct (two orderings of one multiset of sentences). Nothing yet connects
"a factor carried purely by temporal arrangement" to a behaviour anyone wants to steer.

Companion notes already in this repo, which this builds on rather than repeats:
[[temporal_safety_tasks_litreview]] (four-cluster safety sweep, 2026-07-23),
[[temporal_benchmark_screen]] (the R0–R5 rung ladder and the decision rule),
[[window_length_theory]], [[refusal_experiment_plan]].

### The selection criterion, restated from what actually won

The last sprint's win was **steering, not reading**, and the mechanism was specific: a
per-token dictionary's per-latent intervention is *one direction added at every position*, so
its write is constant in time (measured per-position spread exactly 0.0000). Two orderings of
one multiset are the same bag of tokens, so a constant write pushes both classes equally and
has nothing to grip.

That generalises to a two-part test, and the second part is much less restrictive than the
first:

- **P1 — factor invariance.** The factor distinguishing the two conditions is invariant to
  every permutation-symmetric readout, i.e. the conditions have *matched token multisets* and
  differ only in arrangement. This is the property that produced the win. It is rare in real
  behaviours, and most entries below fail it — flagged honestly per row.
- **P2 — write non-constancy.** The *optimal intervention* is non-constant in time: you need
  to push one way early and another way late, or push at an offset relative to an event. P2
  is implied by P1 but also holds without it (ramp-then-release, anti-phase, offset-relative
  copying). **P2 is the operative criterion**; P1 is the clean special case that makes the
  control airtight.

A third, purely practical filter dominates a 10h sprint:

- **P3 — judge-free metric.** The order-task result was chosen because teacher-forced Δmargin
  needs no LLM judge. Any candidate whose success metric is a string statistic (repeat rate,
  copy accuracy) or a teacher-forced logit margin is worth several that need graded
  generations.

Rejection criterion inherited from [[temporal_benchmark_screen]]: if a *single direction at a
single best position* already saturates the intervention, there is no steering headroom for a
window to claim, regardless of how temporally extended the behaviour looks. That is the
refusal diagnosis, and it is the failure mode to check first, not last.

### Headline ranking

| # | behaviour | P1 multiset-matched foil | P2 write non-constant | model organism tonight | judge-free | priority |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Induction / in-context copying (RRT) | **yes, exactly** | yes | free (any LM) | yes | 5 |
| 2 | Repetition loops / degeneration | **yes** | yes | free (greedy decode) | yes | 5 |
| 3 | Backtracking / self-correction | no | yes | R1-Distill-Qwen-1.5B | no (judge) | 4 |
| 4 | Sandbagging / password-locking | no | weak | released organisms | partly | 3 |
| 5 | Refusal onset | no | yes | free (Arditi harness in repo) | partly | 2 |
| 6 | Evaluation awareness | no | no (Shape B) | released organism | no | 2 |
| 7 | Sycophancy build-up | partial | yes | free | no | 2 |
| 8 | Emergent misalignment persona drift | no | no (Shape B) | repo c6_em stream | no | 2 |
| 9 | CoT unfaithfulness | no | no | free | no | 1 |
| 10 | Scheming / alignment faking | no | unknown | limited | no | 1 |

Priority is (probability the TXC actually wins) × (relevance of the behaviour) × (can be run
tonight). Details and honest counter-evidence per entry below.

### 1. Induction / in-context copying — priority 5

The single best bridge from the synthetic order task to a named, published mechanism. It is
the *same abstract structure* — an order-only factor over a matched multiset — but the
mechanism has a name, a decade of literature, and is the substrate of in-context learning.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Olsson et al., *In-context Learning and Induction Heads*, Anthropic 2022 ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895)) | the mechanism, the phase transition, the prefix-matching/copying decomposition |
| Crosbie & Shutova, *Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning*, 2024 ([arXiv:2407.07011](https://arxiv.org/abs/2407.07011)) | ablation evidence that induction heads carry ICL in 7B-scale models |
| Hiraoka & Inui, *Repetition Neurons: How Do Language Models Produce Repetitions?*, 2024 ([arXiv:2410.13497](https://arxiv.org/abs/2410.13497)) | links induction machinery to the degeneration behaviour in entry 2 |

**Model organism.** None needed — this is the point. Repeated-random-token (RRT) sequences
run on any pretrained model. Pythia checkpoints give the phase transition for free if we want
it; Llama-3.2-1B or Qwen2.5-1.5B is plenty for the steering experiment and fits an A10G with
room for a dictionary.

**Temporal signature.** The factor is "the current suffix repeats an earlier span *in the same
order*". Nothing at any single position carries it; it is a relation between a position and a
position ~50 tokens earlier.

**P1 — passes, and the matched foil is already the field's standard control.** The canonical
RRT experiment is: random sequence `S`, then `S` again in the same order, versus random
sequence `S`, then a *shuffled* `S`. Those two conditions have **identical token multisets** by
construction. This is not a foil we have to invent and defend — it is the control the
induction-head literature already runs. That is worth a great deal: the "you built the task to
win" objection dissolves when the task is someone else's control condition.

**P2 — passes.** To induce copying at position `t` the intervention must reference position
`t − p`; a single direction added at every position cannot encode an offset. A crosscoder
window of length `T > p` can, and its decoder slab is exactly the object that writes a
different vector per offset.

**Matched-foil control.** Repeat-in-order vs repeat-shuffled, same tokens, same length, same
first half. Second control: vary the period `p` and check the crosscoder advantage tracks
`T > p` — this is the window-length curve from [[window_length_theory]] on a real mechanism,
and doubles as the reviewer-response figure.

**Metric, judge-free.** Teacher-forced margin on the correct continuation token in the second
copy — identical measurement machinery to `steer_order_modal.py`. Steering target: raise copy
accuracy on the shuffled condition (induce spurious copying), or suppress it on the repeated
condition. Both directions are measurable as a logit margin with no generation at all.

**Feasibility.** High. No fine-tuning, sequences are short (a few hundred tokens), activation
caching is minutes, dictionary training is the same recipe as last sprint. This is the entry I
would start tonight.

**Honest risk.** Induction is *the* thing per-token dictionaries have been shown to represent
well — a "this token was preceded by X earlier" latent is per-token-representable, and the
sprint already established that a causal transformer writes its history into every token. So
expect the **reading** comparison to favour the SAE again, exactly as before. The claim to
test is the steering one. If the SAE's constant write can raise copy accuracy as well as the
crosscoder's slab, the entry dies — and that is a clean, fast negative.

### 2. Repetition loops and degeneration — priority 5

A behaviour everyone wants to steer, with a free model organism and a string-statistic metric.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)) | the canonical statement of the problem and the decoding-side baselines |
| Xu et al., *Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation*, NeurIPS 2022 ([arXiv:2206.02369](https://arxiv.org/abs/2206.02369)) | the **self-reinforcement effect**: repeat probability grows with the number of prior repeats — an explicitly temporal, cumulative quantity |
| Hiraoka & Inui, *Repetition Neurons*, 2024 ([arXiv:2410.13497](https://arxiv.org/abs/2410.13497)) | localised units that switch on as a loop establishes itself |
| *Repetitions are not all alike: distinct mechanisms sustain repetition in language models*, 2025 ([arXiv:2504.01100](https://arxiv.org/abs/2504.01100)) | separates natural-language repetition from induced in-context repetition — gives us two conditions, not one |

**Model organism.** Free. Any 1.5B model at low temperature or greedy decoding degenerates
reliably; the literature's standard elicitation is a short prompt plus greedy decoding.

**Temporal signature.** The self-reinforcement effect is a *cumulative* quantity — the strength
of the attractor depends on how many repeats have already occurred, which no single position
carries. The loop itself is a *period*, a property of the arrangement.

**P1 — passes.** `A B A B A B` and `A A A B B B` have the same multiset; only the first is a
period-2 loop. More usefully for real text: a passage containing an `n`-gram repeated at
period `p` versus the same tokens rearranged so no period exists. Any permutation-symmetric
readout is at chance between them by construction.

**P2 — passes, and this is the strongest instance in the catalogue.** Breaking a loop requires
an *anti-phase* write: suppress the copy impulse at the position where the loop would close,
not at every position. The constant-write version of this intervention already exists and is
known to be bad — a repetition penalty applied uniformly damages fluency. That gives us a
named, deployed, uniformly-applied baseline to beat, which is unusually strong footing for a
steering comparison.

**Matched-foil control.** Periodic vs multiset-matched aperiodic text, both teacher-forced.
Second control: sweep the period `p` and check the advantage appears only when the crosscoder
window covers it.

**Metric, judge-free.** Repeat rate / distinct-n of sampled continuations (pure string
statistics), or teacher-forced margin on the loop-closing token. Note the repo already has
`experiments/ward_backtracking_txc/plot/coherence.py` computing distinct-2, TTR, and max
consecutive same-word run — the degeneration metrics are already written, from the sprint
where "Wait Wait Wait" loops nearly contaminated a steering result.

**Feasibility.** High, and the metric code partly exists.

**Honest risk.** Two, and they matter. (i) A 2026 preprint surfaced in search claims a
*single neuron* edit fixes repetition loops (*Can Editing 1 Neuron Fix Repetition Loops in
LLMs?*, arXiv:2606.13705 — **id unverified, from a search snippet**). If true that is the
Arditi-saturation failure mode: a constant single-direction write already solves the task and
there is no headroom for a window. **Verify this first** — it is cheap and it is decisive.
(ii) Decoding-side fixes (nucleus sampling, repetition penalty) are so effective in practice
that a reviewer may ask why an activation-space intervention is needed at all; the answer has
to be that we are using the behaviour as a *measurement instrument* for temporal codes, not
proposing a better decoder.

### 3. Backtracking and self-correction — priority 4

The repo's incumbent, and the one real-model demonstration already in hand. Included because
it is the calibrated positive control for anything else we build, not because it is new.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Ward, Lin, Venhoff, Nanda, *Reasoning-Finetuning Repurposes Latent Representations in Base Models*, 2025 ([arXiv:2507.12638](https://arxiv.org/abs/2507.12638)) | a base-Llama direction that induces backtracking in R1-Distill at layer 10 — the steering lever |
| Venhoff et al., *Understanding Reasoning in Thinking Language Models via Steering Vectors*, 2025 ([arXiv:2506.18167](https://arxiv.org/abs/2506.18167)) — id to verify | reasoning-behaviour steering vectors including backtracking |

**Model organism.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — downloadable tonight, fits
an L4 comfortably. The 8B Llama distill is what the repo used.

**Temporal signature.** Anticipatory build-up over roughly 8–13 tokens before the surface
"Wait" token (offset corroborated internally by the Llama-Scope `feat_71839` finding; the
direction is Ward et al.'s).

**P1 — fails.** A trace that backtracks and one that does not have different token multisets.
No matched-multiset foil exists, and constructing one would be artificial.

**P2 — passes.** The useful write is a ramp: build doubt, then release so the model can
recover. A constant write produces the "Wait Wait Wait" degeneration the repo already
observed.

**Matched-foil control.** Best available is D+ (offsets before a genuine backtrack) versus D−
(matched offsets in traces that do not backtrack), which the repo's `mine_features` already
implements. Weaker than a multiset match — this is the honest cost of a real behaviour.

**Metric.** Δgenuine-event count under a judge. Repo has `grade_backtracking.py` and
`grade_sonnet.py`. Not judge-free, which is the main reason it is priority 4 rather than 5 in
a 10h window.

**Feasibility.** Medium — the pipeline exists but is a multi-phase, multi-GPU script written
for pods, not for Modal A10G. Adapting it costs hours that entries 1 and 2 do not.

**Value if run anyway.** `H_txc = R4 − R3` (TXC vs Stacked SAE) is *computable from existing
runs* — `stacked_sae` is already in `config.yaml`'s `arch_list` — and has never been reported.
That number is what the reviewers asked for. It is the cheapest high-value thing in this entry
and does not need any new GPU time.

### 4. Sandbagging / deliberate underperformance — priority 3

Best available *released* model organisms of any entry, but a weak temporal story.

**Defining papers and organisms.**

| item | detail |
| --- | --- |
| van der Weij et al., *AI Sandbagging: Language Models can Strategically Underperform on Evaluations*, 2024 ([arXiv:2406.07358](https://arxiv.org/abs/2406.07358)) | the defining paper; released a Llama-3-8B-Instruct organism with passphrase-dependent underperformance on hazardous-domain questions |
| Greenblatt et al., *Stress-Testing Capability Elicitation With Password-Locked Models*, 2024 ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550)) | password-locked model organisms and the elicitation framing |
| *The Elicitation Game*, 2025 ([arXiv:2502.02180](https://arxiv.org/abs/2502.02180)) | evaluates elicitation techniques against several organisms |
| *AuditBench* ([arXiv:2602.22755](https://arxiv.org/abs/2602.22755)) and *The Model Organism Lottery* ([arXiv:2607.01033](https://arxiv.org/abs/2607.01033)) — **ids unverified** | 2026 collections of hidden-behaviour organisms; the latter warns that interpretability results depend strongly on how the organism was trained |

Note the repo already has `experiments/temporal_screen/passphrase_steering/` — worth checking
what state it is in before treating this as new work.

**Temporal signature — weak.** The trigger is a passphrase, usually a short span in the
prompt; the "armed" state is then persistent (Shape B in
[[temporal_safety_tasks_litreview]]). Persistent states are read fine by a per-token
dictionary at any position in the span.

**P1 — fails.** **P2 — weak**: no obvious reason the defusing write should vary with position.

**Priority 3 despite that**, for one reason: **no steering baseline exists at all.** Detection
is solved (probes work); *defusing* an armed state is unclaimed. High novelty, low probability
the win is specifically temporal. The *Model Organism Lottery* caveat is a real hazard — a
result on one organism may not transfer.

### 5. Refusal onset — priority 2, and the repo's own screen predicts it fails

Recorded here mainly so the sprint does not rediscover the negative.
[[temporal_benchmark_screen]] predicts refusal fails the steering rung: the Arditi
single-direction, single-position intervention already near-saturates, so
`H_steer ≈ 0` regardless of how multi-token the build-up looks. The detection story is
genuinely good — Doda, *Before the Last Token* ([arXiv:2605.12726](https://arxiv.org/abs/2605.12726),
verified in the earlier sweep) reports a harmful-span probe at ≈0.998 against ≈0.174 at the
final token, and a PCA-HMM trajectory model recovering ~94% of missed jailbreaks where naive
max-pooling fails. But detection is the axis the last sprint retired.

Live infra: `plots/2026-05-13_arditi*`, `docs/dmitry/reviewer_responses/refusal_experiment_plan.md`.

### 6–10. Lower tier, with the reason each is low

- **Evaluation awareness** — steering is demonstrated and strong (Hua, Qin, Marks, Nanda,
  [arXiv:2510.20487](https://arxiv.org/abs/2510.20487), ~80%→~10% on an agentic proxy), and it
  is the hottest current subarea. But it is a persistent context state with no build-up, so
  the honest claim would be "a window reads a stable latent slightly cleaner", which is a
  reading claim — the axis we retired. No repo infra.
- **Sycophancy build-up** — genuinely accumulates over turns, and a matched foil is
  *constructible* (same user turns, reordered so agreement pressure arrives late rather than
  early), which is a partial P1. Needs a judge and multi-turn generation; too heavy for
  tonight, but a good candidate for a follow-up sprint. Anchor: Sharma et al., *Towards
  Understanding Sycophancy in Language Models* ([arXiv:2310.13548](https://arxiv.org/abs/2310.13548)).
- **Emergent misalignment persona drift** — repo has the whole `docs/dmitry/c6_em/` stream and
  the infra is live, which is tempting. But the timescale is turns to hundreds of tokens with
  no sharp trigger and no published token-level temporal analysis. Shape B. Origin: Betley et
  al. ([arXiv:2502.17424](https://arxiv.org/abs/2502.17424)); persona vectors
  ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)).
- **CoT unfaithfulness** — the mismatch is between a stated reason and an internal one; that is
  a relation between the prompt hint and the final answer, not a pattern across positions.
  Turpin et al. ([arXiv:2305.04388](https://arxiv.org/abs/2305.04388)).
- **Scheming / alignment faking** — highest stakes, weakest evidence, and *directly
  contradicted* for latent build-up by Fomin et al. ([arXiv:2606.30449](https://arxiv.org/abs/2606.30449),
  verified in the earlier sweep): probes "read the situation, not the action", predictive
  signal does *not* strengthen approaching the action token, prompt-domain decodes at 0.999
  while the best future-behaviour probe reaches 0.801. Do not headline.

### Citation confidence ledger

- **Canonical, high confidence:** 2209.11895, 1904.09751, 2206.02369, 2407.07011, 2410.13497,
  2406.07358, 2405.19550, 2502.17424, 2507.21509, 2310.13548, 2305.04388, 2507.12638.
- **Verified in the 2026-07-23 sweep, carried over:** 2605.12726, 2510.20487, 2606.30449.
- **Surfaced by search this session, arXiv id NOT yet verified — do not cite externally:**
  2606.13705 (single-neuron repetition fix — load-bearing for entry 2's risk, verify first),
  2504.01100, 2507.07810, 2604.10044, 2601.05693, 2602.22755, 2607.01033, 2502.02180,
  2506.18167, 2605.07984.

### Changelog

- 2026-07-26 ~23:15 PDT — first pass: ranking, criterion restatement, entries 1–10.
