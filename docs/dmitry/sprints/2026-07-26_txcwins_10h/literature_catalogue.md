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

Sections are ordered by priority and named rather than numbered, since the ranking moves as
the sprint proceeds.

### The headline finding of this catalogue

**The property that won last sprint is not confined to synthetic tasks. There is a whole
family of real, documented, widely-cared-about model failures whose defining factor is
*permutation of the prompt at matched multiset*** — few-shot demonstration order,
multiple-choice option order, retrieved-document position, and conflicting-instruction
precedence. In each the model is given exactly the same content in a different arrangement and
behaves differently, often dramatically so. These are not tasks we would have to construct and
then defend; they are established benchmarks of model fragility that people actively want
fixed.

That gives the sprint a real behaviour with the exact structural property that produced the
crosscoder advantage, and it is cheaper to run than anything else in this note.

### The selection criterion, restated from what actually won

The last sprint's win was **steering, not reading**, and the mechanism was specific: a
per-token dictionary's per-latent intervention is *one direction added at every position*, so
its write is constant in time (measured per-position spread exactly 0.0000). Two orderings of
one multiset are the same bag of tokens, so a constant write pushes both classes equally and
has nothing to grip.

- **P1 — factor invariance.** The conditions have *matched token multisets* and differ only in
  arrangement, so every permutation-symmetric readout is at chance. This is the property that
  produced the win.
- **P2 — write non-constancy.** The *optimal intervention* is non-constant in time: push one
  way early and another way late, or push at an offset relative to an event. Implied by P1 but
  also holds without it (ramp-then-release, anti-phase, offset-relative copying). **P2 is the
  operative criterion**; P1 is the clean special case that makes the control airtight.
- **P3 — judge-free metric.** The order task was chosen because teacher-forced Δmargin needs no
  LLM judge. In a 10h sprint a candidate whose success metric is a logit margin or a string
  statistic is worth several that need graded generations.
- **P4 — harness fit.** New, and practically decisive; see the next section.

Rejection criterion inherited from [[temporal_benchmark_screen]]: if a *single direction at a
single best position* already saturates the intervention, there is no steering headroom for a
window to claim, however temporally extended the behaviour looks. That is the refusal
diagnosis, and it is the failure mode to check first, not last.

### P4: what the existing harness makes cheap

Worth stating explicitly because it reorders the candidate list.
`experiments/temporal_screen/dict_bench/steer_order_modal.py` does not window over *tokens*.
It splits a document into `k_seg` **segments**, mean-pools activations within each segment's
token span (`hh[a:b+1].mean(0)`), and the crosscoder window is `T = k_seg` segments; the
steering write is applied back over the segment token spans. Default model
`Qwen/Qwen2.5-1.5B-Instruct`.

So any task whose natural unit is a **block** — a demonstration, an answer option, a retrieved
document, an instruction, a conversational turn — is a drop-in: set one segment per block and
`T` is 4–12, comfortably inside the range the sprint already validated. Tasks whose natural
unit is a *token at a ~50-token offset* (induction) need the sequence chunked into segments
first, which is easy but is a change.

This is why the permutation family below outranks induction on feasibility even though both
have exact P1 foils.

### A fairness point the sprint should settle before claiming a win

The most important methodological item in this note, and it is not handled anywhere in the
repo yet.

[[temporal_benchmark_screen]] already makes the fairness argument for *reading*: the per-token
baseline must be **the best single position, not a default one** (rung R1, the
position-oracle), because a window that only beats a per-token read at the terminal token has
found a better position rather than anything temporal.

**The steering side needs the same correction and does not have it.** Last sprint compared the
crosscoder's `(T, d)` slab against the SAE's single direction added at *every* position
(`sae_broadcast`). But a per-token dictionary is not actually restricted to a uniform write —
an experimenter can add the same direction at a chosen subset of positions. Three arms, not
two:

| arm | write | what it isolates |
| --- | --- | --- |
| S1 | SAE direction, all positions (`sae_broadcast`) | what was run last sprint |
| S2 | SAE direction, **oracle-chosen positions** (e.g. only the second block, only offsets −13..−8) | position selection without cross-position binding — the steering analogue of R1 |
| S3 | TXC decoder slab (a different vector per offset) | learned temporal profile |

The claim that survives review is `S3 > S2`, not `S3 > S1`. `S3 > S1` alone invites the reply
that the baseline was handicapped. State the honest asymmetry either way: S2 needs *external
supervision* to know which positions to write at, whereas the crosscoder's profile comes out of
unsupervised training. That is a real advantage, but an advantage in *supervision*, not in
representation, and it should be labelled as such.

On the order task S2 is cheap — apply the SAE direction only to the second block's segment
spans — and running it retroactively hardens last sprint's headline result.

### Headline ranking

| behaviour | P1 exact foil | P2 | organism tonight | judge-free | harness fit | priority |
| --- | --- | --- | --- | --- | --- | --- |
| Prompt permutation family (demo order / MCQ option order / doc position) | **yes** | yes | free | yes | drop-in, segment = block | 5 |
| Instruction-order conflict / injection precedence | **yes** | yes | free | yes | drop-in, segment = instruction | 5 |
| Multi-turn escalation (crescendo) | **yes** | yes | free | yes | drop-in, segment = turn | 4 |
| Induction / in-context copying (RRT) | **yes** | yes | free | yes | needs chunking | 4 |
| Repetition loops (loop *escape*) | yes | strongest | free | yes | segment = repeated unit | 4 |
| Backtracking / self-correction | no | yes | R1-Distill-Qwen-1.5B | no | separate pipeline | 4 |
| Entity / state tracking (boxes) | **yes** | yes | free + public data | yes | segment = operation | 3 |
| Steganography / encoded reasoning | **yes** | yes | limited | yes | segment = sentence | 3 |
| Sandbagging / password-locking | no | weak | released organisms | partly | poor | 3 |
| Sycophancy build-up | partial | yes | free | no | segment = turn | 2 |
| Refusal onset | no | yes | free (repo infra) | partly | separate | 2 |
| Evaluation awareness | no | no (Shape B) | released organism | no | poor | 2 |
| Emergent misalignment persona drift | no | no (Shape B) | repo c6_em | no | poor | 2 |
| CoT unfaithfulness | no | no | free | no | poor | 1 |
| Scheming / alignment faking | no | unknown | limited | no | poor | 1 |

Priority is (probability the TXC actually wins) × (relevance of the behaviour) × (can be run
tonight).

### Prompt permutation sensitivity — priority 5

One family, three instances, all with exact multiset-matched foils that *already exist as
published benchmarks*. The model receives identical content in a different arrangement and
behaves differently. This is last sprint's task structure occurring naturally.

**Instance A — few-shot demonstration order.** The largest and most famous effect.

| paper | what it gives us |
| --- | --- |
| Lu, Bartolo, Moore, Riedel, Stenetorp, *Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity*, ACL 2022 ([arXiv:2104.08786](https://arxiv.org/abs/2104.08786)) — verified | "the order in which the samples are provided can make the difference between near state-of-the-art and random guess performance", and it "is present across model sizes (even for the largest current models)" |
| Zhao, Wallace, Feng, Klein, Singh, *Calibrate Before Use: Improving Few-Shot Performance of Language Models*, ICML 2021 ([arXiv:2102.09690](https://arxiv.org/abs/2102.09690)) | the calibration-based mitigation, i.e. the existing per-position-agnostic baseline to beat |

**Instance B — multiple-choice option order.**

| paper | what it gives us |
| --- | --- |
| Pezeshkpour & Hruschka, *Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions*, NAACL Findings 2024 ([arXiv:2308.11483](https://arxiv.org/abs/2308.11483)) — verified | "a considerable performance gap of approximately 13% to 75% in LLMs on different benchmarks, when answer options are reordered" |
| Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors*, ICLR 2024 ([arXiv:2309.03882](https://arxiv.org/abs/2309.03882)) | names it *selection bias*, attributes it to a token-position prior, and gives PriDe as the debiasing baseline |

**Instance C — retrieved-document position.**

| paper | what it gives us |
| --- | --- |
| Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang, *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024 ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) — verified | "performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle" |

**Instance D — premise order in reasoning. Probably the best instance of the four**, because the
failure is in *reasoning* rather than in a label prior, and because no localised mechanistic
fix has been published for it.

| paper | what it gives us |
| --- | --- |
| Chen, Chi, Wang, Zhou, *Premise Order Matters in Reasoning with Large Language Models*, ICML 2024 ([arXiv:2402.08939](https://arxiv.org/abs/2402.08939)) — verified | "permuting the premise order can cause a performance drop of over 30%"; performance is best when premise order matches the order of the ground-truth proof steps; ships **R-GSM**, a public GSM8K-derived benchmark built precisely from premise reorderings |

R-GSM is the single most convenient artefact in this catalogue: a public dataset of
multiset-matched permutation pairs, with a large measured behavioural gap, and one premise per
segment maps straight onto the harness's `k_seg`.

**Model organism.** None needed for any instance. Qwen2.5-1.5B-Instruct — the harness default —
plausibly exhibits all four, though instance D may need a larger model to have enough baseline
reasoning accuracy for the gap to be measurable. Check before committing.

**Temporal signature.** The factor is position-of-content. Nothing about a demonstration or an
option changes between conditions; only where it sits.

**P1 — exact, by construction, and pre-legitimised.** Permuting demonstrations, options, or
documents leaves the multiset identical. These permutation sets are the *published evaluation
protocol*, not a foil we invented.

**P2 — passes, and unusually cleanly for instance B.** Debiasing is inherently a
position-dependent write: to remove a prior favouring option A you must suppress the first
slot and boost the last. A single direction added everywhere cannot do that — it shifts all
slots equally and leaves the *relative* prior untouched. This is arguably a sharper P2 than
the original order task, because here the uniform write provably cannot move the target
quantity at all, rather than merely moving it weakly.

**Matched-foil control.** The permutation itself. Second control: a task where order genuinely
should not matter (a question with one obviously correct option, where accuracy is
order-invariant) — the crosscoder advantage must vanish there.

**Metric, fully judge-free.** Logit margin between option labels (instance B), or between the
correct and distractor answer (A and C). No sampling, no judge, and the effect sizes above are
large enough that a modest test set suffices.

**Steering target, stated as a claim someone would want.** "A learned temporal profile can
remove a model's prompt-order sensitivity, where a single steering direction cannot." That is
a result about eval validity that people outside this project care about, and it is exactly
the crosscoder's structural advantage.

**Feasibility.** Highest in the catalogue. Segment = block, `T` = 4–12, prompts are short, no
fine-tuning, and the harness needs only a new corpus builder.

**Honest risks.** (i) The order effect must be verified in the specific 1.5B model before any
dictionary training — 20 minutes of forward passes, and it is the go/no-go. (ii) Calibration
methods (Calibrate Before Use, PriDe) are cheap and effective, so as with repetition the
framing must be that the behaviour is a *measurement instrument* for temporal codes, not a
proposed better debiaser. (iii) For instance B the positional prior may be carried by the
option *label* tokens ("A", "B", "C"), which are per-token-identifiable; a per-token
dictionary can then represent "this is slot A" perfectly well. It still cannot *write*
asymmetrically across slots, so reading should again favour the SAE while steering does not.
That is now the expected signature, and finding it a fourth time would itself be worth stating.

### Instruction-order conflict / prompt-injection precedence — priority 5

The same structure with a safety stake. Two conflicting instructions `A` and `B` in one
context; condition 1 presents `A` then `B`, condition 2 presents `B` then `A`. Identical token
multisets, different behaviour, and which instruction the model obeys is the central question
in LLM application security.

| paper | what it gives us |
| --- | --- |
| Wallace, Xiao, Leike, Weng, Heidecke, Beutel, *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions*, 2024 ([arXiv:2404.13208](https://arxiv.org/abs/2404.13208)) — verified | the framing and the vulnerability: "LLMs often consider system prompts … to be the same priority as text from untrusted users and third parties" |
| *Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy*, 2024 ([arXiv:2410.09102](https://arxiv.org/abs/2410.09102)) — id unverified | evidence that the fix is positional/architectural, i.e. provenance is carried positionally |
| *Reasoning Up the Instruction Ladder for Controllable Language Models*, 2025 ([arXiv:2511.04694](https://arxiv.org/abs/2511.04694)) — id unverified | recent instruction-ladder benchmark |

**Model organism.** None. Conflicting pairs generate programmatically: "Answer in English" /
"Answer in French"; "Always end with a haiku" / "Never use poetry"; "Summarise in one
sentence" / "Give five bullet points".

**P1 — exact** (swap the blocks). **P2 — passes by exactly the sprint's argument**: a per-token
latent meaning "this is the authoritative instruction" can only be added uniformly, boosting
*both* instructions equally, and cannot express "obey the first, not the second".

**Metric, judge-free.** Teacher-forced margin between a continuation complying with `A` and one
complying with `B`; both are constructed, so no generation.

**Control.** A non-conflicting instruction pair in both orders, where behaviour should be
order-invariant and the advantage must vanish.

**Honest risks.** (i) Instruction-tuned models are *trained* to prefer the system prompt, so
the order effect may be small in a model with strong hierarchy training — check the behavioural
gap exists first. (ii) The blocks are lexically distinct, so a per-token code can tell which
block a token belongs to; what it cannot do is write asymmetrically across them. Reading will
again favour the SAE.

### Multi-turn escalation (crescendo-style jailbreaks) — priority 4

The best relevance story in the catalogue: *no individual turn is harmful*, so the entire
attack is carried by the arrangement.

| paper | what it gives us |
| --- | --- |
| Russinovich, Salem, Eldan, *Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack*, 2024 ([arXiv:2404.01833](https://arxiv.org/abs/2404.01833)) | the attack; all turns individually benign, harm carried by escalation ordering |
| Li et al., *LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet*, 2024 ([arXiv:2408.15221](https://arxiv.org/abs/2408.15221)) — id unverified | multi-turn success against defended models |
| NEXUS / MultiBreak / ContextualJailbreak, 2025–26 ([arXiv:2510.03417](https://arxiv.org/abs/2510.03417), [arXiv:2605.01687](https://arxiv.org/abs/2605.01687), [arXiv:2605.02647](https://arxiv.org/abs/2605.02647)) — ids unverified | benchmarks, and in at least one an explicit turn-shuffling ablation |

**Temporal signature.** Crescendo works by "leveraging the model's inherent tendency to adapt
to recent input, particularly content it generates itself" — an across-turn accumulation.

**P1 — exact**: same turns, escalating vs shuffled order. The ablation exists in the
literature. **Verify the direction before relying on it** — one search result reported that
shuffling *increases* attack success, which if true is still a valid foil (behaviour differs at
matched multiset) but contradicts the naive escalation story and must be stated. Flagged
unverified.

**P2 — passes.** Suppressing the attack means writing against accumulated permissiveness in
the later turns while leaving benign early turns alone; the uniform write is a blunt global
refusal boost, which is the existing baseline and is known to cost helpfulness.

**Metric.** Teacher-forced margin between a compliant and a refusing continuation — fully
judge-free, and the version to use tonight. Refusal-prefix string match is the fallback.

**Harness fit.** Drop-in: segment = turn, `T` = number of turns.

**Why not 5.** Assembling matched escalating/shuffled conversation sets is real work, and
contexts are longer. If the two priority-5 entries die early, pick this up — its relevance
story is the best here.

### Induction / in-context copying — priority 4

The best bridge from the synthetic order task to a *named mechanism*: same abstract structure,
but with a decade of literature and the substrate of in-context learning behind it.

| paper | what it gives us |
| --- | --- |
| Olsson et al., *In-context Learning and Induction Heads*, Anthropic 2022 ([arXiv:2209.11895](https://arxiv.org/abs/2209.11895)) | the mechanism, the phase transition, prefix-matching/copying |
| Crosbie & Shutova, *Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning*, 2024 ([arXiv:2407.07011](https://arxiv.org/abs/2407.07011)) | ablation evidence that induction heads carry ICL at 7B scale |
| Hiraoka & Inui, *Repetition Neurons*, 2024 ([arXiv:2410.13497](https://arxiv.org/abs/2410.13497)) | links induction machinery to degeneration |

**P1 — exact, and the foil is the field's own standard control**: random sequence `S`, then `S`
again in order, versus `S` then a **shuffled** `S`. Identical multisets. The "you built the
task so you would win" objection dissolves when the task is somebody else's control condition,
published years before we needed it.

**P2 — passes.** Inducing a copy at position `t` requires referencing `t − p`; a single
direction added everywhere cannot encode an offset.

**Metric, judge-free.** Teacher-forced margin on the correct continuation token in the second
copy.

**Bonus figure.** Sweep the period `p` and check the advantage appears only when the window
covers it — the window-length curve from [[window_length_theory]] on a real mechanism, which
doubles as a reviewer-response figure.

**Harness fit — the reason this is 4 not 5.** The natural unit is a token at a ~50-token
offset, not a block. It maps on after chunking (split `S` into 12 chunks, so the document is
`chunk_1..chunk_12` then the same chunks again in order or shuffled, `T = 24`), which is easy
but is a change to the corpus builder rather than a drop-in.

**Honest risk.** Induction is precisely what per-token dictionaries represent well. Expect the
reading comparison to favour the SAE again; if the SAE's uniform write also raises copy
accuracy as much as the slab, the entry dies cleanly and fast.

### Repetition loops and degeneration — priority 4, and target *escape* not onset

| paper | what it gives us |
| --- | --- |
| Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020 ([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)) | canonical problem statement, decoding-side baselines |
| Xu, Liu, Yan, Cai, Li, Li, *Learning to Break the Loop*, NeurIPS 2022 ([arXiv:2206.02369](https://arxiv.org/abs/2206.02369)) — verified | the **self-reinforcement effect**: "the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence" — explicitly cumulative across positions |
| Hiraoka & Inui, *Repetition Neurons*, 2024 ([arXiv:2410.13497](https://arxiv.org/abs/2410.13497)) | localised units that switch on as a loop establishes |
| *Repetitions are not all alike*, 2025 ([arXiv:2504.01100](https://arxiv.org/abs/2504.01100)) — id unverified | separates natural-language from induced in-context repetition: two conditions, not one |

**P1 — passes.** `A B A B A B` and `A A A B B B` share a multiset; only the first is a period-2
loop.

**P2 — the strongest instance in the catalogue.** Breaking a loop needs an **anti-phase** write
— suppress the copy impulse where the loop would close, not everywhere. The uniform version
already exists, is deployed, and is known to damage fluency (a flat repetition penalty).
Beating a named production baseline is unusually strong footing.

**Metric, judge-free.** Repeat rate / distinct-n, or teacher-forced margin on the loop-closing
token. `experiments/ward_backtracking_txc/plot/coherence.py` already computes distinct-2, TTR
and max consecutive same-word run — written during the sprint where "Wait Wait Wait" loops
nearly contaminated a steering result.

**Honest risk — verified, real, and it relocates the entry rather than killing it.** Lazaridis,
Sharma, Bates, King, Lu, FitzGerald, *Can Editing 1 Neuron Fix Repetition Loops in LLMs?*, 2026
([arXiv:2606.13705](https://arxiv.org/abs/2606.13705) — fetched and confirmed) shows weight
edits on small neuron sets, down to a single sign-inverted neuron in smaller models, fix
repetition loops at normal generation budgets on Gemma 4 IT variants, with baseline loop
failure rates as high as 95%, while preserving benchmark performance. The abstract answers
"yes" to the title and immediately qualifies: *"Can it cure doom loops? Probably not"*,
characterising doom looping as a deeper failure mode and "fundamentally a knowledge-precision
problem".

Read for our purposes: a **constant, per-token-representable** intervention already solves
ordinary loop *onset*, so that regime is saturated and has no headroom. The residual —
escaping an *established* loop — is explicitly unsolved and is also where the cumulative
self-reinforcement quantity is largest. Target loop escape. Note also that the paper compares
against weight editing, not activation steering, so the steering baseline is still unmeasured.

### Backtracking and self-correction — priority 4, mostly as the calibrated control

The repo's incumbent. Anchors the scale for anything new rather than being novel itself.

Ward, Lin, Venhoff, Nanda, *Reasoning-Finetuning Repurposes Latent Representations in Base
Models*, 2025 ([arXiv:2507.12638](https://arxiv.org/abs/2507.12638)) — a base-Llama direction
induces backtracking in R1-Distill at layer 10, the steering lever. Venhoff et al.,
*Understanding Reasoning in Thinking Language Models via Steering Vectors*
([arXiv:2506.18167](https://arxiv.org/abs/2506.18167) — id unverified).

**Model organism.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, downloadable tonight, fits an
L4. The repo used the 8B Llama distill.

**Temporal signature.** Anticipatory build-up over roughly 8–13 tokens before the surface
"Wait" token (offset corroborated internally by the Llama-Scope `feat_71839` finding; the
direction is Ward et al.'s).

**P1 — fails.** Backtracking and non-backtracking traces have different multisets, and a
matched one would be artificial. Best available foil is D+ (offsets before a genuine backtrack)
against D− (matched offsets in non-backtracking traces), already implemented in
`mine_features`. This is the honest cost of a real behaviour.

**P2 — passes.** The useful write is a ramp — build doubt, then release so the model can
recover. A uniform write produces the "Wait Wait Wait" degeneration the repo already saw.

**Metric.** Δgenuine-event count under a judge (`grade_backtracking.py`, `grade_sonnet.py`).
Not judge-free, and the pipeline targets multi-GPU pods rather than Modal.

**The free thing in this entry.** `H_txc = R4 − R3` — TXC against Stacked SAE — is
**computable from existing runs**, since `stacked_sae` is already in `config.yaml`'s
`arch_list`, and has never been reported. That is the number reviewers asked for, at zero GPU
cost: the highest value-per-minute item in the catalogue.

### Entity and state tracking (the boxes task) — priority 3

Exact multiset foils and a public dataset, but carrying the most direct published
counter-evidence of any entry — which is itself why it is worth recording.

| paper | what it gives us |
| --- | --- |
| Kim & Schuster, *Entity Tracking in Language Models*, ACL 2023 ([arXiv:2305.02363](https://arxiv.org/abs/2305.02363)) | the boxes task: seven boxes, 1–12 state-changing operations, then query contents |
| Prakash, Rott Shaham, Haklay, Belinkov, Bau, *Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking*, ICLR 2024 ([arXiv:2402.14811](https://arxiv.org/abs/2402.14811)) — verified | the circuit: "entity tracking is performed by tracking the **position** of the correct entity" — a positionally-carried mechanism |
| Tang, Zhao, Franco, Wijaya, Mueller, Schuster, Kim, *Do Language Models Track Entities Across State Changes?*, 2026 ([arXiv:2605.30233](https://arxiv.org/abs/2605.30233)) — verified | the counter-evidence, quoted below |

**P1 — exact.** The same multiset of operations in different orders yields different final
states for non-commuting operations, and the dataset generator gives unlimited matched pairs.
**P2 — passes**: changing the answer means intervening on *which operation came last*.

**Counter-evidence, and it is sharp.** arXiv:2605.30233 finds that "LMs do not incrementally
track world states across tokens or query-relevant states across layers, but simply aggregate
relevant information in parallel at the last token when the query becomes evident." Treat it as
corroboration of our own diagnosis rather than a reason to avoid the task: the reading arm is
predicted to lose, the steering arm is untested.

**Why 3.** A capability rather than a safety behaviour, so weaker relevance than the entries
above. But the foil quality is exceptional and the data is public — the best fallback if a
clean P1 task is needed in a hurry.

### Steganography / encoded reasoning — priority 3

Conceptually the purest P1 in existence: a steganographic payload is *by definition* invisible
to any per-position readout and carried entirely by arrangement. Held at 3 only because model
organisms are thin.

| paper | what it gives us |
| --- | --- |
| Roger & Greenblatt, *Preventing Language Models From Hiding Their Reasoning*, 2023 ([arXiv:2311.02282](https://arxiv.org/abs/2311.02282)) — id unverified | the encoded-reasoning threat model and paraphrasing defence |
| *Early Signs of Steganographic Capabilities in Frontier LLMs*, 2025 ([arXiv:2507.02737](https://arxiv.org/abs/2507.02737)) — id unverified | current-capability baseline: models mostly cannot yet hide short messages under standard affordances |
| *Conceptual Steganography*, 2026 ([arXiv:2605.26537](https://arxiv.org/abs/2605.26537)) — id unverified | payloads carried by patterns of reasoning behaviour rather than lexical choice — the shape closest to a temporal factor |

**Temporal signature.** The payload lives in the arrangement of otherwise-innocuous content —
an acrostic across sentence initials, a parity code over sentence lengths. Shuffling the
sentences destroys the payload while preserving the multiset exactly.

**P1 — exact and perfect. P2 — passes**: suppressing an encoded channel requires disrupting
the relation between positions, not adding a constant.

**Why only 3.** The blocking question is whether a 1.5B model *computes* the payload at all —
if it does not represent it, there is nothing to read or steer, and the entry evaporates.
Current-capability evidence suggests frontier models are only marginally able to do this, so a
1.5B is unlikely to. A prompted-decode probe (can the model decode an acrostic when asked?)
settles it in minutes and should gate any further work. Safety relevance is high (CoT
monitorability), so it is worth the cheap check.

### Sandbagging / deliberate underperformance — priority 3

The best available *released* model organisms of any entry, and the largest novelty gap —
nobody has tried to *steer* an armed state — but the weakest temporal story.

| item | detail |
| --- | --- |
| van der Weij et al., *AI Sandbagging: Language Models can Strategically Underperform on Evaluations*, 2024 ([arXiv:2406.07358](https://arxiv.org/abs/2406.07358)) | the defining paper; a Llama-3-8B-Instruct organism with passphrase-dependent underperformance on hazardous-domain questions |
| Greenblatt et al., *Stress-Testing Capability Elicitation With Password-Locked Models*, 2024 ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550)) | password-locked organisms and the elicitation framing |
| *The Elicitation Game*, 2025 ([arXiv:2502.02180](https://arxiv.org/abs/2502.02180)) — id unverified | evaluation of elicitation techniques across organisms |
| *AuditBench* ([arXiv:2602.22755](https://arxiv.org/abs/2602.22755)), *The Model Organism Lottery* ([arXiv:2607.01033](https://arxiv.org/abs/2607.01033)) — ids unverified | 2026 collections of hidden-behaviour organisms; the latter warns interpretability results depend strongly on how the organism was trained |
| *Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging* ([arXiv:2604.26206](https://arxiv.org/abs/2604.26206)) — id unverified | links sandbagging to the option-order family above — worth reading, it may connect the priority-5 entry to a safety behaviour |

The repo already has `experiments/temporal_screen/passphrase_steering/` — check its state
before treating this as new work.

**Temporal signature — weak.** A short passphrase trigger, then a persistent armed state (Shape
B in [[temporal_safety_tasks_litreview]]), which a per-token dictionary reads fine anywhere in
the span. **P1 fails; P2 weak.** Priority 3 rests entirely on the novelty of *defusing* rather
than detecting, and the *Model Organism Lottery* caveat is live: a result on one organism may
not transfer.

### Lower tier, with the reason each is low

- **Sycophancy build-up** — accumulates over turns, and a matched foil is constructible (same
  user turns reordered so agreement pressure arrives late rather than early), a partial P1.
  Needs a judge and multi-turn generation; too heavy for tonight, good for a follow-up. Anchor:
  Sharma et al. ([arXiv:2310.13548](https://arxiv.org/abs/2310.13548)).
- **Refusal onset** — recorded so the sprint does not rediscover the negative.
  [[temporal_benchmark_screen]] predicts it fails the steering rung: the Arditi
  single-direction single-position intervention near-saturates, so `H_steer ≈ 0` however
  multi-token the build-up looks. Detection is genuinely good (Doda, *Before the Last Token*,
  [arXiv:2605.12726](https://arxiv.org/abs/2605.12726): harmful-span probe ≈0.998 vs ≈0.174 at
  the final token; a PCA-HMM trajectory model recovers ~94% of missed jailbreaks where naive
  max-pooling fails) but detection is the axis last sprint retired. Live infra:
  `plots/2026-05-13_arditi*`.
- **Evaluation awareness** — steering demonstrated and strong (Hua, Qin, Marks, Nanda,
  [arXiv:2510.20487](https://arxiv.org/abs/2510.20487), ~80%→~10% on an agentic proxy) and the
  hottest current subarea, but a persistent context state with no build-up, so the only
  available claim is a reading claim. No repo infra.
- **Emergent misalignment persona drift** — the repo has the whole `docs/dmitry/c6_em/` stream
  and live infra, but the timescale is turns to hundreds of tokens with no sharp trigger and no
  published token-level temporal analysis. Shape B. Betley et al.
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
field, arrived at independently. Worth a sentence in the sprint writeup: the negative result is
not an artefact of our task design, it is a property of causal transformers that somebody else
measured on a completely different task.

### Citation confidence ledger

- **Fetched and verified this session:** 2104.08786 (Lu et al., ACL 2022, order sensitivity
  quote), 2308.11483 (Pezeshkpour & Hruschka, 13–75% gap), 2307.03172 (Liu et al., TACL,
  lost-in-the-middle quote), 2404.13208 (Wallace et al., instruction hierarchy),
  2402.14811 (Prakash et al., ICLR 2024, positional entity-tracking circuit), 2605.30233 (Tang
  et al., aggregation-at-last-token quote), 2606.13705 (Lazaridis et al., single-neuron
  repetition fix and the "doom loops" caveat), 2206.02369 (Xu et al., NeurIPS 2022,
  self-reinforcement quote).
- **Canonical, high confidence, not re-fetched:** 2209.11895, 1904.09751, 2407.07011,
  2410.13497, 2102.09690, 2309.03882, 2406.07358, 2405.19550, 2404.01833, 2305.02363,
  2502.17424, 2507.21509, 2310.13548, 2305.04388, 2507.12638.
- **Verified in the 2026-07-23 sweep, carried over:** 2605.12726, 2510.20487, 2606.30449.
- **Search-surfaced, arXiv id NOT verified — do not cite externally without checking:**
  2504.01100, 2507.07810, 2604.10044, 2601.05693, 2602.22755, 2607.01033, 2502.02180,
  2506.18167, 2605.07984, 2410.09102, 2511.04694, 2408.15221, 2510.03417, 2605.01687,
  2605.02647, 2606.08644, 2311.02282, 2507.02737, 2605.26537, 2604.26206.

### Changelog

- 2026-07-26 ~23:15 PDT — first pass: criterion restatement; induction, repetition,
  backtracking, sandbagging, refusal, lower tier.
- 2026-07-26 ~23:55 PDT — added instruction-order conflict, multi-turn escalation,
  entity/state tracking; added the S1/S2/S3 steering-fairness section; verified five
  citations, including the single-neuron repetition result that demoted repetition; added the
  external-corroboration note.
- 2026-07-27 ~00:40 PDT — restructured around the prompt-permutation family (new top entry,
  three instances, all with published matched-multiset protocols); added the P4 harness-fit
  criterion after confirming the harness windows over *segments*, not tokens, which promotes
  block-structured tasks and demotes induction; added steganography; verified three more
  citations.
