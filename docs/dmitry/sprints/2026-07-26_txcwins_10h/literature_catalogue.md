---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - reference
  - in-progress
---

## Catalogue of candidate behaviours for TXC vs TopK SAE vs tSAE

Living document, updated through the sprint. Purpose: close the relevance gap left by the
2026-07-25 dictbench sprint — a real crosscoder advantage was found, but only on a synthetic
construct (two orderings of one multiset of sentences). Nothing yet connects "a factor carried
purely by temporal arrangement" to a behaviour anyone wants to steer.

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
  differ only in arrangement. This is the property that produced the win. It is rarer in real
  behaviours than one would like, but — the main finding of this catalogue so far — it is
  **not confined to synthetic tasks**. Four entries below have exact multiset-matched foils
  that already exist in the published literature as someone else's control condition.
- **P2 — write non-constancy.** The *optimal intervention* is non-constant in time: you need
  to push one way early and another way late, or push at an offset relative to an event. P2
  is implied by P1 but also holds without it (ramp-then-release, anti-phase, offset-relative
  copying). **P2 is the operative criterion**; P1 is the clean special case that makes the
  control airtight.

A third, purely practical filter dominates a 10h sprint:

- **P3 — judge-free metric.** The order-task result was chosen because teacher-forced Δmargin
  needs no LLM judge. Any candidate whose success metric is a string statistic (repeat rate,
  copy accuracy, refusal-prefix match) or a teacher-forced logit margin is worth several that
  need graded generations.

Rejection criterion inherited from [[temporal_benchmark_screen]]: if a *single direction at a
single best position* already saturates the intervention, there is no steering headroom for a
window to claim, regardless of how temporally extended the behaviour looks. That is the
refusal diagnosis, and it is the failure mode to check first, not last.

### A fairness point the sprint needs to settle before it claims a win

This is the most important methodological item in this note, and it is not yet handled
anywhere in the repo.

[[temporal_benchmark_screen]] already makes the fairness argument for *reading*: the per-token
baseline must be **the best single position, not a default one** (rung R1, the
position-oracle), because a window that only beats a per-token read at the terminal token has
found a better position rather than anything temporal.

**The steering side needs the same correction and does not yet have it.** The last sprint
compared the crosscoder's `(T, d)` slab against the SAE's single direction *added at every
position*. But a per-token dictionary is not actually restricted to a uniform write — an
experimenter can add the same direction at a chosen subset of positions. So there are three
arms, not two:

| arm | write | what it isolates |
| --- | --- | --- |
| S1 | SAE direction, all positions (uniform) | what was run last sprint |
| S2 | SAE direction, **oracle-chosen positions** (e.g. only the second block, only offsets −13..−8) | position selection without cross-position binding — the steering analogue of R1 |
| S3 | TXC decoder slab (a different vector per offset) | learned temporal profile |

The claim that survives review is `S3 > S2`, not `S3 > S1`. `S3 > S1` alone invites the
reply that the baseline was handicapped. Note the honest asymmetry to state in the writeup:
S2 needs *external supervision* to know which positions to write at, whereas the crosscoder's
profile comes out of unsupervised training — that is a real advantage, but it is an advantage
in *supervision*, not in representation, and should be labelled as such.

Recommendation: run S2 on whichever task the sprint picks. On the order task it is cheap —
apply the SAE direction only to the second block's positions — and it retroactively hardens
last sprint's headline result.

### Headline ranking

| # | behaviour | P1 multiset-matched foil | P2 write non-constant | organism tonight | judge-free | priority |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Instruction-order conflict / prompt-injection precedence | **yes, exact** | yes | free (any instruct LM) | yes | 5 |
| 2 | Induction / in-context copying (RRT) | **yes, exact** | yes | free (any LM) | yes | 5 |
| 3 | Repetition loops / degeneration | **yes** | yes, strongest | free (greedy decode) | yes | 4 |
| 4 | Multi-turn escalation (crescendo jailbreak) | **yes, exact** | yes | free (instruct LM) | partly | 4 |
| 5 | Backtracking / self-correction | no | yes | R1-Distill-Qwen-1.5B | no | 4 |
| 6 | Entity / state tracking (boxes) | **yes, exact** | yes | free + public dataset | yes | 3 |
| 7 | Sandbagging / password-locking | no | weak | released organisms | partly | 3 |
| 8 | Sycophancy build-up | partial | yes | free | no | 2 |
| 9 | Refusal onset | no | yes | free (Arditi infra in repo) | partly | 2 |
| 10 | Evaluation awareness | no | no (Shape B) | released organism | no | 2 |
| 11 | Emergent misalignment persona drift | no | no (Shape B) | repo c6_em stream | no | 2 |
| 12 | CoT unfaithfulness | no | no | free | no | 1 |
| 13 | Scheming / alignment faking | no | unknown | limited | no | 1 |

Priority is (probability the TXC actually wins) × (relevance of the behaviour) × (can be run
tonight).

### 1. Instruction-order conflict / prompt-injection precedence — priority 5

The structure of last sprint's winning task, with real semantic content and a real deployment
stake. Two conflicting instructions `A` and `B` in one context; condition 1 presents `A` then
`B`, condition 2 presents `B` then `A`. **Identical token multisets, different behaviour.**
The behaviour of interest — which instruction the model obeys — is exactly a function of
arrangement, and it is the central question in LLM application security.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Wallace, Xiao, Leike, Weng, Heidecke, Beutel, *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions*, 2024 ([arXiv:2404.13208](https://arxiv.org/abs/2404.13208)) — verified | the framing and the vulnerability: "LLMs often consider system prompts … to be the same priority as text from untrusted users and third parties" |
| *Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy*, 2024 ([arXiv:2410.09102](https://arxiv.org/abs/2410.09102)) — id unverified | evidence that the fix is architectural/positional, i.e. that provenance is carried positionally |
| *Reasoning Up the Instruction Ladder for Controllable Language Models*, 2025 ([arXiv:2511.04694](https://arxiv.org/abs/2511.04694)) — id unverified | recent instruction-ladder benchmark |

**Model organism.** None needed. Any instruction-tuned 1.5–7B model (Qwen2.5-1.5B/7B-Instruct,
Llama-3.2-1B/3B-Instruct). Conflicting-instruction pairs are trivial to generate
programmatically: "Answer in English" / "Answer in French"; "Always end with a haiku" / "Never
use poetry"; "Summarise in one sentence" / "Give five bullet points".

**Temporal signature.** Provenance and precedence. Nothing at a single token says "this
instruction is the authoritative one" independent of what came before it — that is a relation
between two spans.

**P1 — exact.** Swap the two instruction blocks. Same tokens, same length, same instructions.
This is the sprint's winning task structure obtained for free from a real task.

**P2 — passes, by exactly the sprint's argument.** A per-token latent meaning "this is an
authoritative instruction" can only be written as one direction added everywhere, which boosts
*both* instructions equally and cannot express "obey the first, not the second". The crosscoder
slab can write positively over the early block and negatively over the late one.

**Matched-foil control.** The order swap is itself the foil. Second control: a
non-conflicting pair in both orders, where behaviour should be order-invariant — the
crosscoder advantage must vanish there.

**Metric, judge-free.** Teacher-forced margin between a continuation that complies with `A`
and one that complies with `B`. Both continuations are constructed, so no generation and no
judge. Identical measurement machinery to `steer_order_modal.py`.

**Feasibility.** Highest in the catalogue. Prompt construction only, no fine-tuning, short
contexts, one activation cache, same dictionary recipe as last sprint.

**Honest risks.** (i) Instruction-tuned models are *trained* to prefer the system prompt, so
the order effect may be small in models with strong instruction-hierarchy training — check the
behavioural gap exists before training anything. Weaker/older or base-plus-light-tuning models
will show it more strongly. (ii) The two blocks are lexically distinct, so a per-token code
*can* tell which block a token belongs to; what it cannot do is write asymmetrically across
them. That means reading will again favour the SAE, and the claim is steering only — which is
now the expected pattern, not a surprise.

### 2. Induction / in-context copying — priority 5

The best bridge from the synthetic order task to a named, published mechanism: the *same*
abstract structure (an order-only factor over a matched multiset), but with a mechanism that
has a decade of literature and is the substrate of in-context learning.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Olsson et al., *In-context Learning and Induction Heads*, Anthropic 2022 ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895)) | the mechanism, the phase transition, the prefix-matching/copying decomposition |
| Crosbie & Shutova, *Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning*, 2024 ([arXiv:2407.07011](https://arxiv.org/abs/2407.07011)) | ablation evidence that induction heads carry ICL at 7B scale |
| Hiraoka & Inui, *Repetition Neurons: How Do Language Models Produce Repetitions?*, 2024 ([arXiv:2410.13497](https://arxiv.org/abs/2410.13497)) | links induction machinery to the degeneration behaviour in entry 3 |

**Model organism.** None needed. Repeated-random-token (RRT) sequences on any pretrained
model. Pythia checkpoints give the phase transition for free if wanted; Llama-3.2-1B or
Qwen2.5-1.5B suffices and leaves room for a dictionary on an A10G.

**Temporal signature.** "The current suffix repeats an earlier span *in the same order*" — a
relation between a position and one ~50 tokens earlier.

**P1 — exact, and the foil is already the field's standard control.** The canonical RRT
experiment is: random sequence `S`, then `S` again in order, versus random sequence `S`, then
a **shuffled** `S`. Identical multisets by construction. This matters more than it might
appear: the "you built the task so you would win" objection dissolves when the task is
somebody else's control condition, published years before we needed it.

**P2 — passes.** To induce copying at position `t` the intervention must reference position
`t − p`; a single direction added at every position cannot encode an offset. A crosscoder
window with `T > p` can, and its decoder slab is exactly the object that writes a different
vector per offset.

**Matched-foil control.** Repeat-in-order vs repeat-shuffled. Second control: sweep the period
`p` and check the advantage appears only when `T > p` — the window-length curve from
[[window_length_theory]] on a real mechanism, which doubles as a reviewer-response figure.

**Metric, judge-free.** Teacher-forced margin on the correct continuation token in the second
copy. Steering target in both directions: induce spurious copying in the shuffled condition,
or suppress it in the repeated one.

**Feasibility.** High. No fine-tuning, short sequences, minutes of caching.

**Honest risk.** Induction is precisely the thing per-token dictionaries represent well, and
the sprint already established that a causal transformer writes its history into every token.
Expect the reading comparison to favour the SAE again. If the SAE's uniform write also raises
copy accuracy as much as the slab, the entry dies — cleanly and fast, which is itself worth
something.

### 3. Repetition loops and degeneration — priority 4

A behaviour everyone wants fixed, a free organism, and a string-statistic metric. Demoted from
5 to 4 after verifying the saturation risk below.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)) | canonical problem statement and decoding-side baselines |
| Xu, Liu, Yan, Cai, Li, Li, *Learning to Break the Loop*, NeurIPS 2022 ([arXiv:2206.02369](https://arxiv.org/abs/2206.02369)) — verified | the **self-reinforcement effect**: "the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence" — an explicitly cumulative, across-position quantity |
| Hiraoka & Inui, *Repetition Neurons*, 2024 ([arXiv:2410.13497](https://arxiv.org/abs/2410.13497)) | localised units that switch on as a loop establishes |
| *Repetitions are not all alike: distinct mechanisms sustain repetition in language models*, 2025 ([arXiv:2504.01100](https://arxiv.org/abs/2504.01100)) — id unverified | separates natural-language repetition from induced in-context repetition: two conditions, not one |

**Model organism.** Free — any 1.5B model at low temperature or greedy decoding.

**Temporal signature.** Self-reinforcement is *cumulative*: attractor strength depends on how
many repeats have already occurred, which no single position carries. The loop itself is a
period, a property of arrangement.

**P1 — passes.** `A B A B A B` and `A A A B B B` share a multiset; only the first is a period-2
loop. For real text: a passage with an `n`-gram at period `p` versus the same tokens
rearranged with no period.

**P2 — the strongest instance in the catalogue.** Breaking a loop needs an **anti-phase**
write — suppress the copy impulse where the loop would close, not everywhere. The uniform
version of this intervention already exists, is deployed, and is known to damage fluency (a
flat repetition penalty). Beating a named, uniformly-applied production baseline is unusually
strong footing.

**Metric, judge-free.** Repeat rate / distinct-n of sampled continuations, or teacher-forced
margin on the loop-closing token. `experiments/ward_backtracking_txc/plot/coherence.py`
already computes distinct-2, TTR and max consecutive same-word run — written during the sprint
where "Wait Wait Wait" loops nearly contaminated a steering result.

**Honest risk — verified, and it is real but survivable.** Lazaridis, Sharma, Bates, King, Lu,
FitzGerald, *Can Editing 1 Neuron Fix Repetition Loops in LLMs?*, 2026
([arXiv:2606.13705](https://arxiv.org/abs/2606.13705) — **fetched and confirmed**) shows weight
edits on small neuron sets (down to a single sign-inverted neuron in smaller models) fix
repetition loops at normal generation budgets on Gemma 4 IT variants, with baseline loop
failure rates as high as 95%, while preserving benchmark performance. The paper's own framing
is the useful part: the abstract answers "yes" to the title and then immediately qualifies —
*"Can it cure doom loops? Probably not"* — characterising doom looping as a deeper failure mode
and "fundamentally a knowledge-precision problem".

Read for our purposes: a **constant, per-token-representable** intervention already solves
ordinary loop onset, so that is the saturated regime and there is no headroom there. The
residual — escaping an *established* loop, the doom-loop regime — is explicitly unsolved, and
it is also the regime where the cumulative self-reinforcement quantity is largest. If this
entry is run, target **loop escape, not loop onset**. Also note the paper compares against
weight editing, not activation steering, so the steering baseline is still unmeasured.

Second risk: decoding-side fixes work well enough in practice that a reviewer may ask why an
activation intervention is wanted at all. The answer must be that the behaviour is a
*measurement instrument* for temporal codes, not a proposed better decoder.

### 4. Multi-turn escalation (crescendo-style jailbreaks) — priority 4

The highest-relevance entry with an exact multiset foil: the same conversational turns in
escalating order versus shuffled order.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Russinovich, Salem, Eldan, *Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack*, 2024 ([arXiv:2404.01833](https://arxiv.org/abs/2404.01833)) | the attack; all turns individually benign, the harm is carried entirely by the escalation ordering |
| Li et al., *LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet*, 2024 ([arXiv:2408.15221](https://arxiv.org/abs/2408.15221)) — id unverified | multi-turn attack success against defended models |
| NEXUS / MultiBreak / ContextualJailbreak, 2025–26 ([arXiv:2510.03417](https://arxiv.org/abs/2510.03417), [arXiv:2605.01687](https://arxiv.org/abs/2605.01687), [arXiv:2605.02647](https://arxiv.org/abs/2605.02647)) — ids unverified | benchmarks and, in at least one, an explicit turn-shuffling ablation |

**Temporal signature.** The strongest in the catalogue conceptually: *no individual turn is
harmful*. The entire attack is the arrangement. Crescendo works by "leveraging the model's
inherent tendency to adapt to recent input, particularly content it generates itself" — an
across-turn accumulation.

**P1 — exact.** Same turns, escalating order vs shuffled order. This ablation exists in the
literature. **Verify the direction before relying on it**: one search result reported that
shuffling *increases* attack success, which if true is still an order effect (and still a
valid foil, since behaviour differs by arrangement at matched multiset) but contradicts the
naive escalation story and would need to be stated. Flagged as unverified.

**P2 — passes.** Suppressing the attack requires writing against accumulated permissiveness in
the later turns while leaving the benign early turns alone; a uniform write is a blunt global
refusal boost, which is the existing baseline and is known to cost helpfulness.

**Metric.** Refusal-prefix string match (the standard judge-free proxy from the GCG literature)
or teacher-forced margin between a compliant and a refusing continuation. The latter is fully
judge-free and is the version to use tonight.

**Feasibility.** Medium. Multi-turn contexts are longer, and building matched escalating/shuffled
conversation sets takes effort. But no fine-tuning is needed and public multi-turn attack sets
exist. The teacher-forced framing removes the generation cost entirely.

**Why it is not priority 5.** Longer contexts mean a larger `T` and more activation memory,
and the dataset assembly is real work. If entries 1 and 2 both die early, this is the one to
pick up — it has the best relevance story of anything here.

### 5. Backtracking and self-correction — priority 4

The repo's incumbent and the calibrated positive control for anything new. Listed because it
anchors the scale, not because it is novel.

**Defining papers.** Ward, Lin, Venhoff, Nanda, *Reasoning-Finetuning Repurposes Latent
Representations in Base Models*, 2025 ([arXiv:2507.12638](https://arxiv.org/abs/2507.12638)) —
a base-Llama direction induces backtracking in R1-Distill at layer 10, the steering lever.
Venhoff et al., *Understanding Reasoning in Thinking Language Models via Steering Vectors*
([arXiv:2506.18167](https://arxiv.org/abs/2506.18167) — id unverified).

**Model organism.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, downloadable tonight, fits an
L4. The repo used the 8B Llama distill.

**Temporal signature.** Anticipatory build-up over roughly 8–13 tokens before the surface
"Wait" token (offset corroborated internally by the Llama-Scope `feat_71839` finding; the
direction is Ward et al.'s).

**P1 — fails.** Backtracking and non-backtracking traces have different multisets, and
constructing a matched one would be artificial. Best available foil is D+ (offsets before a
genuine backtrack) vs D− (matched offsets in non-backtracking traces), already implemented in
`mine_features`. This is the honest cost of a real behaviour.

**P2 — passes.** The useful write is a ramp — build doubt, then release so the model can
recover. A uniform write produces the "Wait Wait Wait" degeneration the repo already observed.

**Metric.** Δgenuine-event count under a judge (`grade_backtracking.py`, `grade_sonnet.py`).
Not judge-free, which is the main reason it is not priority 5 tonight. The pipeline is also
written for multi-GPU pods rather than Modal.

**The free thing in this entry.** `H_txc = R4 − R3` — TXC against Stacked SAE — is
**computable from existing runs**, since `stacked_sae` is already in `config.yaml`'s
`arch_list`, and has never been reported. That is the number reviewers asked for, at zero GPU
cost. Highest value-per-minute item in the catalogue.

### 6. Entity and state tracking (the boxes task) — priority 3

Exact multiset foils and a real public dataset, but carrying the most direct published
counter-evidence of any entry — which is itself the reason to record it carefully.

**Defining papers.**

| paper | what it gives us |
| --- | --- |
| Kim & Schuster, *Entity Tracking in Language Models*, ACL 2023 ([arXiv:2305.02363](https://arxiv.org/abs/2305.02363)) | the boxes task: seven boxes, then 1–12 state-changing operations, then query the contents |
| Prakash, Rott Shaham, Haklay, Belinkov, Bau, *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking*, ICLR 2024 ([arXiv:2402.14811](https://arxiv.org/abs/2402.14811)) — verified | the circuit, and that "entity tracking is performed by tracking the **position** of the correct entity" — a positionally-carried mechanism |
| Tang, Zhao, Franco, Wijaya, Mueller, Schuster, Kim, *Do Language Models Track Entities Across State Changes?*, 2026 ([arXiv:2605.30233](https://arxiv.org/abs/2605.30233)) — verified | the counter-evidence, quoted below |

**P1 — exact.** The same multiset of operations applied in different orders yields different
final states (`move A to box 1; move B to box 1` versus the reverse, for non-commuting
operations). A permutation-symmetric readout is at chance on the final state by construction,
and the dataset generator gives us as many matched pairs as we want.

**P2 — passes.** To change the answer you must intervene on *which operation came last*, which
is inherently offset-relative.

**Counter-evidence, and it is sharp.** arXiv:2605.30233 finds that "LMs do not incrementally
track world states across tokens or query-relevant states across layers, but simply aggregate
relevant information in parallel at the last token when the query becomes evident." That is an
independent, published confirmation of exactly what killed the reading experiments last sprint
— the model has already written what it needs into the query token, so a per-token code at
that position reads the state fine. Treat it as corroboration of our own diagnosis rather than
as a reason to avoid the task: the reading arm is predicted to lose, and the steering arm is
untested.

**Metric, judge-free.** Accuracy or teacher-forced margin on the correct object token at the
query.

**Why 3 rather than higher.** It is a capability, not a safety behaviour, so the relevance
story is weaker than entries 1 and 4; and the aggregation result means the interesting arm is
narrow. But the foil quality is exceptional and the dataset is public, so it is the best
fallback if a cleaner P1 task is needed in a hurry.

### 7. Sandbagging / deliberate underperformance — priority 3

The best available *released* model organisms of any entry, and the largest novelty gap
(nobody has tried to steer an armed state), but the weakest temporal story.

| item | detail |
| --- | --- |
| van der Weij et al., *AI Sandbagging: Language Models can Strategically Underperform on Evaluations*, 2024 ([arXiv:2406.07358](https://arxiv.org/abs/2406.07358)) | the defining paper; a Llama-3-8B-Instruct organism with passphrase-dependent underperformance on hazardous-domain questions |
| Greenblatt et al., *Stress-Testing Capability Elicitation With Password-Locked Models*, 2024 ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550)) | password-locked organisms and the elicitation framing |
| *The Elicitation Game*, 2025 ([arXiv:2502.02180](https://arxiv.org/abs/2502.02180)) — id unverified | evaluation of elicitation techniques across organisms |
| *AuditBench* ([arXiv:2602.22755](https://arxiv.org/abs/2602.22755)), *The Model Organism Lottery* ([arXiv:2607.01033](https://arxiv.org/abs/2607.01033)) — ids unverified | 2026 collections of hidden-behaviour organisms; the latter warns interpretability results depend strongly on how the organism was trained |

The repo already has `experiments/temporal_screen/passphrase_steering/` — check its state
before treating this as new work.

**Temporal signature — weak.** A short passphrase trigger, then a persistent armed state
(Shape B). Persistent states are read fine by a per-token dictionary anywhere in the span.
**P1 fails; P2 weak** — no reason the defusing write should vary with position. Priority 3
rests entirely on the novelty of *defusing* rather than detecting, and the *Model Organism
Lottery* caveat is a live hazard: a result on one organism may not transfer.

### 8–13. Lower tier, with the reason each is low

- **Sycophancy build-up** — accumulates over turns, and a matched foil is constructible (same
  user turns, reordered so agreement pressure arrives late rather than early), a partial P1.
  Needs a judge and multi-turn generation; too heavy for tonight but a good follow-up sprint.
  Anchor: Sharma et al. ([arXiv:2310.13548](https://arxiv.org/abs/2310.13548)).
- **Refusal onset** — recorded so the sprint does not rediscover the negative.
  [[temporal_benchmark_screen]] predicts it fails the steering rung: the Arditi
  single-direction single-position intervention near-saturates, so `H_steer ≈ 0` however
  multi-token the build-up looks. Detection is genuinely good (Doda, *Before the Last Token*,
  [arXiv:2605.12726](https://arxiv.org/abs/2605.12726): harmful-span probe ≈0.998 vs ≈0.174 at
  the final token, PCA-HMM trajectory model recovering ~94% of missed jailbreaks where naive
  max-pooling fails) but detection is the axis last sprint retired. Live infra:
  `plots/2026-05-13_arditi*`.
- **Evaluation awareness** — steering demonstrated and strong (Hua, Qin, Marks, Nanda,
  [arXiv:2510.20487](https://arxiv.org/abs/2510.20487), ~80%→~10% on an agentic proxy) and the
  hottest current subarea, but a persistent context state with no build-up, so the only
  available claim is a reading claim. No repo infra.
- **Emergent misalignment persona drift** — the repo has the whole `docs/dmitry/c6_em/` stream
  and live infra, which is tempting, but the timescale is turns to hundreds of tokens with no
  sharp trigger and no published token-level temporal analysis. Shape B. Betley et al.
  ([arXiv:2502.17424](https://arxiv.org/abs/2502.17424)); persona vectors
  ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)).
- **CoT unfaithfulness** — the mismatch is between a stated reason and an internal one: a
  relation between a prompt hint and a final answer, not a pattern across positions. Turpin et
  al. ([arXiv:2305.04388](https://arxiv.org/abs/2305.04388)).
- **Scheming / alignment faking** — highest stakes, weakest evidence, and *directly
  contradicted* for latent build-up by Fomin et al.
  ([arXiv:2606.30449](https://arxiv.org/abs/2606.30449), verified in the 2026-07-23 sweep):
  probes "read the situation, not the action", the predictive signal does not strengthen
  approaching the action token, prompt-domain decodes at 0.999 while the best future-behaviour
  probe reaches 0.801. Do not headline.

### Cross-cutting observation: external corroboration of last sprint's negative

Finding 2 of the dictbench sprint — that reading comparisons were never going to favour a
window code, because a causal transformer has already written its history into every token —
now has an independent published analogue. arXiv:2605.30233 reports that on a *fundamentally
sequential* task, models "do not incrementally track world states across tokens … but simply
aggregate relevant information in parallel at the last token". Same phenomenon, different
field, arrived at independently. This is worth a sentence in the sprint writeup: the negative
result is not an artefact of our task design, it is a property of causal transformers that
somebody else measured on a completely different task.

### Citation confidence ledger

- **Fetched and verified this session:** 2606.13705 (title, authors, the "doom loops" caveat),
  2206.02369 (Xu et al., NeurIPS 2022, self-reinforcement quote), 2605.30233 (title, authors,
  aggregation-at-last-token quote), 2404.13208 (Wallace et al., instruction hierarchy),
  2402.14811 (Prakash et al., ICLR 2024, positional entity-tracking circuit).
- **Canonical, high confidence, not re-fetched:** 2209.11895, 1904.09751, 2407.07011,
  2410.13497, 2406.07358, 2405.19550, 2404.01833, 2305.02363, 2502.17424, 2507.21509,
  2310.13548, 2305.04388, 2507.12638.
- **Verified in the 2026-07-23 sweep, carried over:** 2605.12726, 2510.20487, 2606.30449.
- **Search-surfaced, arXiv id NOT verified — do not cite externally without checking:**
  2504.01100, 2507.07810, 2604.10044, 2601.05693, 2602.22755, 2607.01033, 2502.02180,
  2506.18167, 2605.07984, 2410.09102, 2511.04694, 2408.15221, 2510.03417, 2605.01687,
  2605.02647, 2606.08644.

### Changelog

- 2026-07-26 ~23:15 PDT — first pass: criterion restatement, entries for induction,
  repetition, backtracking, sandbagging, refusal, and the lower tier.
- 2026-07-26 ~23:55 PDT — added instruction-order conflict (new #1), multi-turn escalation,
  entity/state tracking; added the S1/S2/S3 steering-fairness section; verified five
  citations, including the single-neuron repetition result that demotes entry 3; added the
  external-corroboration note.
