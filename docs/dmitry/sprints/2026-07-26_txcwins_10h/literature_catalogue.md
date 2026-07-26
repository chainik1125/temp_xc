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

### The main conclusion of the sweep

Stated separately because it is the finding, not a summary of the entries.

**Almost every behaviour that *looks* temporally extended turns out to have a
per-token-representable handle, and the exceptions are precisely those where the two conditions
are permutations of one another.**

The evidence is a consistent run of saturation results, each found while trying to build a case
*for* the behaviour rather than against it:

| behaviour | the per-token handle that already works | source |
| --- | --- | --- |
| refusal | a single direction at a single position near-saturates | Arditi et al.; this repo's own screen |
| MCQ option order | token bias over option-ID tokens; a content-invariant position attractor | 2309.03882, 2604.26206 |
| repetition onset | a **single sign-inverted neuron** fixes loops at normal budgets | 2606.13705 |
| emergent misalignment | a transferable misalignment direction | 2506.11618 |
| backtracking | reasoning behaviours "controlled by linear directions" | 2506.18167 |
| entity / state tracking | the model aggregates at the last token rather than tracking incrementally | 2605.30233 |

Six independent literatures, six per-token handles. Set against that, the tasks that survive —
demonstration order, permutation composition, instruction-order conflict, induction on
repeat-versus-shuffled — share exactly one structural property, which is P1: the two conditions
are the *same multiset* in a different arrangement, so no per-token handle can exist by
construction.

This is a stronger and more useful claim than "the crosscoder beats the SAE on task X". It says
*where* window codes can possibly help and why everywhere else is a dead end — and it converts
five negative results into the argument for the one positive one. It is also directly
falsifiable: find a behaviour with no matched-multiset structure where a window code wins on
steering, and the claim is wrong.

### The concrete recommendation, if one task has to be picked

**Few-shot demonstration-order permutation** (instance A). One segment per demonstration, `T` =
number of shots, a drop-in for `steer_order_modal.py`.

It wins on the one property that matters most: **permuting demonstrations is verbatim**, so the
token multiset is *exactly* matched with no editing, unlike R-GSM (see instance D). It also
works at 1.5B, where few-shot classification is well within range but GSM8K-style reasoning is
not, and the effect is documented across model sizes up to the largest available.

Ordered so that each step kills the task cheaply if it is going to die:

1. **Behavioural gap check, forward passes only, and size it properly.** Does the chosen model's
   accuracy actually move across demonstration permutations? Sample **~128 orderings** of the
   same `k` demonstrations on one classification task (AG News or DBPedia are in Li et al.'s
   set) and look at the spread. 128 rather than a couple of dozen because the *typical*
   permutation-to-permutation std is only about two accuracy points — the dramatic gaps are the
   tails, and you have to search for them. No spread, no task.
2. **Build the matched pairs.** The *same* demonstrations in two orders — ideally the
   best-scoring and worst-scoring permutations found in step 1, which maximises the behavioural
   gap the steering has to close. The multiset match is exact and free.
3. **Train the ladder at matched realised coefficients per segment** — BatchTopK SAE → PSAE →
   T-SAE → TXC. Log realised L0 for every arm (carried-over debt 3; the failure is silent).
4. **Read and steer separately.** Expect reading to favour the SAE again — that is now the
   predicted result, and a fourth replication of it is worth reporting in its own right.
5. **Steering arms S1/S2/S3** as in the fairness section below. `S3 > S2` is the claim; `S3 > S1`
   alone is not enough.
6. **Controls that can kill it:** time-averaged profile, random profile, random direction,
   row-permuted profile, and the supervised difference-of-means ceiling. These already exist in
   `steer_order_modal.py` and should be carried over unchanged.

The one-line version of the result if it works: *a temporal crosscoder can convert a model's
worst demonstration ordering into its best, where a single steering direction provably cannot,
because the two orderings are the same multiset.*

That framing has a useful property: the **supervised ceiling is known and free**. The
best-ordering accuracy is measured in step 1, so unlike the original order task there is a
natural, interpretable upper bound on what any intervention could achieve, and the result can be
reported as a fraction of a real gap closed rather than as an uncalibrated Δmargin.

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

### The baselines: resolving the tSAE identification, and one missing arm

This section is about *architectures* rather than behaviours, but it bears on every entry
below, and it closes carried-over debt 2 from the sprint start note.

**The tSAE identification is resolved.** The sprint note records that the description given was
"an InfoNCE penalty over nearby positions with no attention", that this repo's `tsae_paper` is
instead attention-based, and that no InfoNCE tSAE exists here. The paper being described is:

Bhalla, Oesterling, Verdun, Lakkaraju, Calmon, *Temporal Sparse Autoencoders: Leveraging the
Sequential Nature of Language for Interpretability*, ICLR 2026 (oral)
([arXiv:2511.05541](https://arxiv.org/abs/2511.05541) — fetched and verified).

It is exactly an InfoNCE penalty over adjacent positions with no attention, so the repo's
attention-based `TemporalSAE` is a **different architecture** and was never the right baseline.
Enough detail to implement it tonight:

| item | value |
| --- | --- |
| loss | symmetric InfoNCE, both directions: `-log[ e^{s(z_t, z_{t-1})} / Σ_j e^{s(z_t, z^j_{t-1})} ]` plus the transpose term |
| positives | adjacent token pairs `(z_t, z_{t-1})` from the same sequence |
| negatives | other pairs in the batch (different samples) |
| window | primary experiments use immediately adjacent tokens; ablations sample randomly from earlier tokens for longer range |
| weight | `α = 1.0` against reconstruction |
| sparsity | **BatchTopK, k = 20** |
| dictionary | 16k features |
| models / layers | Pythia-160m layer 8, Gemma2-2b layer 12; Pile subset |
| feature split | 20% high-level (semantic), 80% low-level (syntactic) — a *predefined* partition |
| results | FVE 0.94 (Pythia) / 0.75 (Gemma); activation smoothness 0.09 vs 0.12–0.13 for baselines; semantics probing beats Matryoshka and BatchTopK; steering "Pareto-dominates" baseline SAEs on intervention success against coherence |

**They release code, trained T-SAEs, and interpreted latents** at
[AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes). Use the
reference loss rather than reimplementing it — this is exactly the situation the repo's own
lessons-learned note warns about (delegate to the reference implementation; the 2026-05-07 EM
replication lost a day to a reimplemented prep function).

Three things follow. First, this is implementable on top of the repo's existing BatchTopK
machinery in roughly the length of a loss function — far cheaper than the dense-tSAE
calibration debt suggested — and with the reference code available there is little reason to
write it from scratch at all. Second, it uses **BatchTopK**, which the last sprint just made the
crosscoder default, so the sparsity rule is already matched and the `l1_coef` calibration
problem disappears entirely — that debt was an artefact of running the wrong architecture.
Third, they claim a *steering* Pareto improvement, so this is a live competitor on the axis the
sprint cares about, not just a reconstruction baseline.

**Their steering setup, for positioning.** 30 features (e.g. "Medical case reports or
concepts", "Book titles and authors"), graded by Llama-3.3-70B for intervention success and
coherence with manual verification, reported as a success-against-coherence Pareto frontier
against Matryoshka and BatchTopK SAEs and the best SAE on Neuronpedia. Two observations. Their
metric is **judge-based**, so the sprint's teacher-forced margin is a cleaner and
non-overlapping measurement rather than a weaker one. And the paper does not state whether
steering was applied uniformly across positions or position-dependently — if uniformly, then
even the tSAE arm is a constant write and the S1/S2/S3 distinction in the next section applies
to it too, which is worth checking in their code before running the comparison.

**The missing arm: Persistent SAEs.** *Persistent Sparse Autoencoders: Learning Feature
Timescales in Language Models*, 2026 ([arXiv:2607.17117](https://arxiv.org/abs/2607.17117) —
fetched, id not independently confirmed) gives each feature a learned persistence coefficient
and runs a diagonal recurrence over positions:

```text
h_t^(j) = λ_j · h_{t-1}^(j) + (1 - λ_j) · a_t^(j),    λ_j = σ(l_j) · λ_max,   λ_max = 0.999
x̂_t = W_dec h_t + b_dec
```

That is a per-feature leaky integrator: **a learned temporal profile with one parameter per
feature**, sitting strictly between a per-token SAE (λ = 0) and a crosscoder's free `(T, d)`
slab. It is about ten lines to implement.

Why it matters here, in order of importance:

1. **It is the baseline a reviewer will ask for.** "Does your crosscoder beat a model that
   learns a one-parameter timescale?" is the obvious question, and the crosscoder's extra cost
   is `T × d` parameters per latent against PSAE's one. If the crosscoder cannot beat an EMA,
   the case for a full slab is weak.
2. **They explicitly did not evaluate crosscoders** (baselines are BatchTopK, Matryoshka, and
   Temporal SAE), so the comparison is unclaimed in both directions.
3. **They already ran a causal intervention on prompt injection at range**, which is adjacent
   to the instruction-order entry below. Setup: AgentDojo trajectories (banking, travel,
   workspace) generated by Llama-3.3-70B-Instruct, replayed through Gemma-2-2B as an external
   monitor, linear probes at the final token of untrusted spans, evaluated up to 350 tokens
   later. Findings: "fast features detect injections locally; slow features track them over
   longer contexts"; both fast and baseline SAEs hit 94–96.7% at immediate evaluation, but
   beyond the injection span fast features "fall towards chance" while slow features stay
   informative; causal interventions showed slow features with a "consistently positive
   intervention effect at every distance" against near-zero for fast features. Reconstruction
   stays competitive (0.93 FVE on Pythia-160M layer 8; Gemma-2-2B layer 12 also used).

Read honestly, item 3 means "temporal structure helps steering at range" is *partly claimed
already*. Our differentiators remain real but must be stated: their axis is **distance**, ours
is **order at matched multiset**; they compare timescales within a per-token code, we compare
against a full window code; and they did not test a crosscoder. It also means the
instruction-order entry should cite them rather than present prompt injection as untouched.

**Recommended baseline set for any task in this catalogue:** TopK/BatchTopK SAE (per-token
floor) → PSAE (learned scalar timescale) → T-SAE (InfoNCE over adjacent positions) → TXC (free
`(T, d)` slab). That is a clean capacity ladder, each rung adding exactly one thing, and it is
a better story than any single head-to-head.

### Prior art on the mechanism itself — what the sprint can and cannot claim as novel

Searched deliberately, because the sprint's headline mechanism is "a position-varying write
beats a constant one". That general claim is **no longer novel**, and the writeup should say so
rather than be corrected on it.

Jin, Deng, Wang, Shen, Zhang, *Beyond Steering Vector: Flow-based Activation Steering for
Inference-Time Intervention*, 2026 ([arXiv:2605.05892](https://arxiv.org/abs/2605.05892) —
fetched and verified) learns a concept-conditioned velocity field `v(h, t, c)` and explicitly
abandons the assumption that interventions are "fixed, single-step, position-invariant
transforms". They report "curved, multi-step, token-varying trajectories" and that nearby
tokens have higher steering similarity — i.e. position-dependent steering structure, measured.
On AxBench they reach harmonic means of 1.015 (Gemma-2-2B-IT) and 1.113 (Gemma-2-9B-IT), "the
first learned method to consistently outperform prompting" without per-concept tuning.

Relatedly, *Steer Like the LLM: Activation Steering that Mimics Prompting*
([arXiv:2605.03907](https://arxiv.org/abs/2605.03907) — id unverified) observes that prompt
steering "can exert strong interventions on some token positions and barely intervene on
others", and that per-token steering coefficients have been proposed to compensate.

**What this leaves for us, stated precisely.** Three things survive, and they should be the
claims made:

1. **The profile is learned unsupervised, by a dictionary, and is a fixed interpretable object
   per latent** — a `(T, d)` slab you can plot — rather than a supervised concept-conditioned
   network evaluated at inference. FLAS needs to know the concept; a crosscoder latent does not.
2. **On a matched-multiset order factor, a position-invariant write cannot work at all**, not
   merely work less well. Every result above is "position-varying is better"; ours is
   "position-invariant is at chance by construction, and here is the task class where that is
   true". That is a sharper and more falsifiable statement.
3. **The comparison is against other dictionaries at matched sparsity budget**, which is the
   axis interpretability cares about, whereas FLAS is a steering-method comparison.

**What must be dropped:** any framing in which "steering vectors are constant in time and that
is a novel observation". It was published, at scale, on a standard benchmark, before this
sprint. Cite it in the first paragraph.

**Also worth knowing:** AxBench is the current standard steering benchmark and would be the
natural external yardstick if the sprint ever wants one. And note a naming hazard — in the
wider literature "crosscoder" means *cross-layer or cross-model* (model diffing, e.g. *Sparse
Crosscoders for diffing MoEs and Dense models*, [arXiv:2603.05805](https://arxiv.org/abs/2603.05805);
*Localizing RL-Induced Tool Use to a Single Crosscoder Feature*,
[arXiv:2606.26474](https://arxiv.org/abs/2606.26474) — ids unverified). A reader will assume
that meaning unless "temporal / cross-position" is said explicitly every time. The upside is
that the cross-*position* variant really is under-explored.

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
| Permutation composition / state tracking | **yes** | yes | free | yes | drop-in, segment = swap | 3 (but the best control) |
| Entity / state tracking (boxes) | **yes** | yes | free + public data | yes | segment = operation | 3 |
| Steganography / encoded reasoning | **yes** | yes | **none at reachable scale** | yes | segment = sentence | 2 |
| Sandbagging / password-locking | no | weak | released organisms | partly | poor | 3 |
| Sycophancy build-up | partial | yes | free | no | segment = turn | 2 |
| Refusal onset | no | yes | free (repo infra) | partly | separate | 2 |
| Evaluation awareness | no | no (Shape B) | released organism | no | poor | 2 |
| Emergent misalignment persona drift | no | no (Shape B) | repo c6_em | no | poor | 2 |
| CoT unfaithfulness | no | no | free | no | poor | 1 |
| Scheming / alignment faking | no | unknown | limited | no | poor | 1 |

Priority is (probability the TXC actually wins) × (relevance of the behaviour) × (can be run
tonight).

### Where to actually get each organism

The brief asks for acquisition paths, so they are collected here rather than scattered. The
important column is the last one: most of the top entries need **no download at all**, which is
why they are the top entries.

| behaviour | what you need | where | needed tonight? |
| --- | --- | --- | --- |
| Premise order (instance D) | R-GSM, 220 problems | released with [arXiv:2402.08939](https://arxiv.org/abs/2402.08939); if the release is awkward to find, **regenerate it** — it is GSM8K with premises reordered, and `openai/gsm8k` on HF plus a sentence splitter reproduces the construction in an hour | dataset only |
| Demonstration order (A) | any classification set | the eleven tasks in [arXiv:2104.08786](https://arxiv.org/abs/2104.08786) are standard (SST-2, AG News, TREC …); permutations are generated locally | nothing |
| MCQ option order (B) | MMLU / ARC | `cais/mmlu` on HF; permute options locally | nothing |
| Document position (C) | NaturalQuestions-open + distractors | the *Lost in the Middle* release; or build from any QA set by inserting the gold document at varying rank | nothing |
| Instruction-order conflict | conflicting instruction pairs | generate programmatically — no dataset exists or is needed | nothing |
| Induction / RRT | random token sequences | generate from the tokenizer vocabulary | nothing |
| Repetition loops | greedy decoding | any model; the elicitation is the decoding setting | nothing |
| Multi-turn escalation | escalating conversation sets | public multi-turn attack sets ([arXiv:2404.01833](https://arxiv.org/abs/2404.01833) and the 2025–26 benchmarks); shuffled foils generated locally | dataset |
| Backtracking | a reasoning model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on HF; Stage A artefacts already in this repo | model download |
| Entity / state tracking | the boxes task | generator described in [arXiv:2305.02363](https://arxiv.org/abs/2305.02363); trivial to reimplement — it is templated text | nothing |
| Sandbagging | a locked model | organisms from [arXiv:2406.07358](https://arxiv.org/abs/2406.07358) / [arXiv:2405.19550](https://arxiv.org/abs/2405.19550); availability varies per organism and should be checked before planning around it | model download, uncertain |
| Emergent misalignment | an EM-finetuned model | this repo's `docs/dmitry/c6_em/` stream already has them | already here |

The pattern is worth stating: **every priority-4-and-above entry except backtracking needs
nothing but the base model the harness already uses.** That is the main practical argument for
the permutation family over the safety-organism entries.

### Prompt permutation sensitivity — priority 5

One family, three instances, all with exact multiset-matched foils that *already exist as
published benchmarks*. The model receives identical content in a different arrangement and
behaves differently. This is last sprint's task structure occurring naturally.

**Instance A — few-shot demonstration order. The recommended task.** Permutation is *verbatim*
— demonstrations are concatenated independent examples, so reordering requires no editing and
the multiset match is exact. This is the property R-GSM turned out to lack.

| paper | what it gives us |
| --- | --- |
| Lu, Bartolo, Moore, Riedel, Stenetorp, *Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity*, ACL 2022 ([arXiv:2104.08786](https://arxiv.org/abs/2104.08786)) — verified | "the order in which the samples are provided can make the difference between near state-of-the-art and random guess performance", and it "is present across model sizes (even for the largest current models)" |
| Li, Wang, Wang, Shang, *Order Matters: Rethinking Prompt Construction in In-Context Learning*, 2025 ([arXiv:2511.09700](https://arxiv.org/abs/2511.09700)) — verified | the modern replication, and the one that settles feasibility: "the variance in performance due to different example orderings is comparable to that from using entirely different example sets", measured across **0.5B to 27B** parameters plus GPT-4, on classification *and* generation tasks; strong orderings are identifiable from dev-set data alone, reaching "performance close to an oracle" |
| Zhao, Wallace, Feng, Klein, Singh, *Calibrate Before Use: Improving Few-Shot Performance of Language Models*, ICML 2021 ([arXiv:2102.09690](https://arxiv.org/abs/2102.09690)) | the calibration-based mitigation, i.e. the existing position-agnostic baseline to beat |

The 0.5B–27B range in the 2025 replication removes the main feasibility risk: a 1.5B model is
comfortably inside the range where the effect is measured, unlike instance D. Their model list
includes **Qwen2.5**, which is the harness's own family.

**The concrete numbers, and they are a warning as much as an encouragement.** Li et al. measure
order sensitivity as the standard deviation of accuracy across permutations, and get an average
of **0.0197** — about two accuracy points — against a selection sensitivity of 0.0225. Their
protocol is `M = 10` demonstration sets × `P = 10` permutations for the sensitivity estimate, and
**`P = 128` randomly sampled permutations** when they want to *find* a strong ordering. Tasks are
AG News, NYT-Topics, NYT-Locations, DBPedia and MMLU for classification; GSM8K, MMLU-Pro and MATH
for generation. Smaller models show "marginally higher variability under permutations" and larger
models are "more stable", with no monotonic trend in the ratio. No code release is mentioned.

Two consequences for the recipe. First, "near state-of-the-art to random guess" (Lu et al.) is
the *extreme* of the distribution, not the typical case — a couple of accuracy points is what a
random pair of orderings will differ by. Second, and following from it, **the best/worst search
needs on the order of 100 permutations, not a couple of dozen**, which is what Li et al. use.
Budget the go/no-go accordingly: it is still only forward passes, but it is ~128 × the prompt
set, not 24.

**Design details that follow from the literature, and are worth getting right first time.**

- **Use the base model, not the instruct model.** Instruction tuning is reported to increase
  prediction consistency under input perturbations, which would shrink the very effect we are
  trying to steer. `Qwen2.5-1.5B` rather than `Qwen2.5-1.5B-Instruct`. ICL is also the base
  model's natural regime.
- **Use few shots, not many.** Order sensitivity *decreases* as the number of demonstrations
  grows on non-reasoning tasks, so `k` around 4–8 maximises the gap — which conveniently is also
  the `T` range the crosscoder handles best.
- **Use classification tasks**, where the effect is largest and the metric is a clean label-token
  margin.
- **The order-selection oracle is both the ceiling and a rival baseline.** Since dev-set-selected
  orderings reach near-oracle performance, "just pick a better order" is an existing, cheap fix.
  State it: our claim is not that steering is the best way to fix order sensitivity, but that
  order sensitivity is a real behaviour whose *steering* separates window codes from per-token
  codes. The oracle gives the denominator for "fraction of the gap closed".

**Instance B — multiple-choice option order.**

| paper | what it gives us |
| --- | --- |
| Pezeshkpour & Hruschka, *Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions*, NAACL Findings 2024 ([arXiv:2308.11483](https://arxiv.org/abs/2308.11483)) — verified | "a considerable performance gap of approximately 13% to 75% in LLMs on different benchmarks, when answer options are reordered" |
| Zheng, Zhou, Meng, Zhou, Huang, *Large Language Models Are Not Robust Multiple Choice Selectors*, ICLR 2024 Spotlight ([arXiv:2309.03882](https://arxiv.org/abs/2309.03882)) — verified | names it *selection bias* — models "prefer to select specific option IDs as answers (like 'Option A')" — attributes it to **token bias**, where the model "a priori assigns more probabilistic mass to specific option ID tokens", and gives PriDe, a label-free inference-time debiasing method, as the baseline; 20 LLMs on three benchmarks |

**Instance C — retrieved-document position.**

| paper | what it gives us |
| --- | --- |
| Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni, Liang, *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024 ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)) — verified | "performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle" |

**Instance D — premise order in reasoning. Excellent motivation, but do not use the released
dataset as the P1 task** (see the correction below).

| paper | what it gives us |
| --- | --- |
| Chen, Chi, Wang, Zhou, *Premise Order Matters in Reasoning with Large Language Models*, ICML 2024 ([arXiv:2402.08939](https://arxiv.org/abs/2402.08939)) — verified, full text read | the effect and its shape: performance is best when premise order matches the order of the ground-truth proof steps; ships **R-GSM**, 220 GSM8K-derived problem pairs |

**Correction, from reading the construction section.** R-GSM does **not** preserve the token
multiset. The procedure keeps the last sentence fixed and rewrites the description with a
different ordering of the others, and explicitly states that "minor editing on words is allowed
to ensure the grammatical correctness of the problem description". Edited words break the exact
multiset match, so a permutation-symmetric readout is *not* at chance by construction and P1
does not hold for R-GSM as released.

Two further limits worth knowing. The dataset is **220 pairs**, over 60% of them exactly five
sentences. And it is calibrated to frontier models: the evaluated set is GPT-4-turbo, PaLM 2-L,
Gemini Pro and GPT-3.5-turbo, with drops of 94.1→85.0, 86.4→79.5, 80.5→69.1 and 67.3→51.8
respectively — i.e. **6.9 to 15.5 points**, not the "over 30%" figure from the abstract, which
refers to their broader logical-reasoning setting (the comparable large number here is a 35%
drop for GPT-3.5-turbo when restricted to problems it originally solved correctly). A 1.5B
model's GSM8K baseline is far below GPT-3.5-turbo's, so the gap may not be measurable at all at
the harness's default scale.

**What to do instead.** Keep instance D as *motivation* — it is the published evidence that
premise order matters for real reasoning — and build the actual task with a templated
generator that permutes **verbatim, self-contained** premises (no cross-sentence pronouns, so
no editing is ever needed). That restores exact P1, gives unlimited data, and lets difficulty
be tuned to a 1.5B model. It is the same kind of corpus builder as the existing CALM/TENSE one.

**Instance E — LLM-as-a-judge position bias. The highest practical stakes of the five**, and
the foil is again the field's own standard procedure.

| source | what it gives us |
| --- | --- |
| Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)) | the original documentation of judge position bias and the swapping control |
| Shi et al., position bias across 15 judges and ~150,000 evaluation instances, IJCNLP 2025 — id not located | the modern scale-up: bias varies by judge and task and is not chance |
| survey: *From Generation to Judgment* ([arXiv:2411.16594](https://arxiv.org/abs/2411.16594)) — id unverified | context |

Judges pick the response in the **first slot in about 68% of comparisons** even where human
annotators prefer the second. The standard mitigation — invoke the judge twice with the
candidates swapped — means a multiset-matched permutation pair is not merely available, it is
*already what careful practitioners run*. Stakes are as high as anything in this catalogue,
since LLM judges now underpin a large share of evaluation.

Why it may beat instance B despite the similar shape: the swapped items are long **content
blocks**, not single label tokens, so the token-bias escape route that sank B does not apply in
the same way. P2 holds — debiasing requires suppressing a first-slot preference specifically,
which a uniform write cannot express.

Two honest risks. The verdict is still emitted as a label token ("A"/"B"), so some token bias
may re-enter through the back door; and a 1.5B model is a poor judge, so this probably needs a
7B, which costs GPU time the other instances do not. Worth holding as the strongest *fallback*
rather than the first thing to run.

**Model organism.** None needed for any instance. For instance A, a 1.5B base model is adequate:
few-shot classification is well within its range.

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

**Whitespace.** A search for sparse-autoencoder or dictionary work on demonstration-order or
prompt-order sensitivity returns nothing. There is plenty of SAE-steering methodology (e.g.
*Improving Steering Vectors by Targeting Sparse Autoencoder Features*,
[arXiv:2411.02193](https://arxiv.org/abs/2411.02193); *Can sparse autoencoders be used to
decompose and interpret steering vectors?*, [arXiv:2411.08790](https://arxiv.org/abs/2411.08790)
— ids unverified) and plenty of order-sensitivity work, but nobody appears to have put a
learned dictionary on the order-sensitivity factor. This entry is unclaimed.

**Honest risks.** (i) The order effect must be verified in the specific 1.5B model before any
dictionary training — 20 minutes of forward passes, and it is the go/no-go. (ii) Calibration
methods (Calibrate Before Use, PriDe) are cheap and effective, so as with repetition the
framing must be that the behaviour is a *measurement instrument* for temporal codes, not a
proposed better debiaser. (iii) For instance B the positional prior may be carried by the
option *label* tokens ("A", "B", "C"), which are per-token-identifiable; a per-token
dictionary can then represent "this is slot A" perfectly well. It still cannot *write*
asymmetrically across slots, so reading should again favour the SAE while steering does not.
That is now the expected signature, and finding it a fourth time would itself be worth stating.

**(iv) A localisation risk specific to instance B, and the reason to prefer D.** Li & Gao,
*Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions*, ACL 2025
Findings ([arXiv:2405.03205](https://arxiv.org/abs/2405.03205) — verified) trace the
first-choice-'A' bias to specific **MLP value vectors and attention heads**, and show that
updating those vectors and recalibrating attention both removes the bias and improves accuracy.
Related work (UniBias) eliminates biased FFN vectors and attention heads directly.

This is the same shape of risk as the single-neuron repetition result: a *localised,
per-token-representable* component already accounts for the behaviour, so a reviewer can ask
what a window code adds. Three mitigations, in order of strength. First, the published
localisation is on the **GPT-2 family**, which the authors themselves describe as having "an
even worse anchored bias" than larger LLMs — it is not established at 1.5B+. Second, those are
*weight-level component* interventions, not additive steering writes, so the steering baseline
is still unmeasured. Third, and most simply, **prefer instance D**: premise-order failure in
R-GSM has no published localisation and is a reasoning failure rather than a label prior, so
the "one biased head explains it" reply is not available.

**A second and more damaging problem with instance B, found on verification.** Zheng et al.
attribute selection bias primarily to **token bias** — the model "a priori assigns more
probabilistic mass to specific option ID tokens (e.g. A/B/C/D)". If the effect is a prior over
*label tokens* rather than over *positions*, then it is per-token representable **and**
per-token steerable: a constant write on the "prefer token A" direction is exactly the right
intervention, and the SAE arm should win or tie. That makes instance B a poor test of a window
code, possibly a predicted-negative rather than a candidate. Two attributions are in the
literature — Pezeshkpour & Hruschka emphasise position plus answer uncertainty, Zheng et al.
emphasise token bias — and until that is settled instance B is not the one to run.

**Net effect: use instance D (premise order), or instance A (demonstration order).** In both,
the permuted units are *content* blocks with no label token to carry a prior, so the token-bias
escape route does not exist and the factor is genuinely positional.

### Instruction-order conflict / prompt-injection precedence — priority 5

The same structure with a safety stake. Two conflicting instructions `A` and `B` in one
context; condition 1 presents `A` then `B`, condition 2 presents `B` then `A`. Identical token
multisets, different behaviour, and which instruction the model obeys is the central question
in LLM application security.

| paper | what it gives us |
| --- | --- |
| Wallace, Xiao, Leike, Weng, Heidecke, Beutel, *The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions*, 2024 ([arXiv:2404.13208](https://arxiv.org/abs/2404.13208)) — verified | the framing and the vulnerability: "LLMs often consider system prompts … to be the same priority as text from untrusted users and third parties" |
| Wu et al., *Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy*, ICLR 2025 ([arXiv:2410.09102](https://arxiv.org/abs/2410.09102)) — verified | the strongest support for this entry: LLMs "treat all inputs equally", and their fix is to embed priority **architecturally**, BERT-style segment embeddings, because delimiters and instruction-tuning "do not address this issue at the architectural level". Robust accuracy up to +15.75% (Structured Query) and +18.68% (Instruction Hierarchy benchmark) |
| *Reasoning Up the Instruction Ladder for Controllable Language Models*, 2025 ([arXiv:2511.04694](https://arxiv.org/abs/2511.04694)) — id unverified | recent instruction-ladder benchmark |

The ISE result matters more than it first looks. If priority has to be *added* as an explicit
segment embedding to be represented properly, then in an unmodified model provenance is carried
by little more than position in the context — which is exactly the regime where a window code
has something to say and a per-token code does not. It also names two usable datasets:
**Structured Query** and the **Instruction Hierarchy benchmark**.

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

**P1 — exact by construction**: same turns, escalating versus shuffled order.

**Correction on the ablation, after checking.** I earlier recorded that a turn-shuffling
ablation exists in the literature, on the strength of a search summary. I could not confirm it.
The Crescendo abstract does not mention one, and NEXUS ([arXiv:2510.03417](https://arxiv.org/abs/2510.03417),
Rafiei Asl, Narula, Ghasemigol, Blanco, Takabi, EMNLP 2025 — verified as a real paper reporting
2.1–19.4% improvement over prior methods) does not mention one either at abstract level. The
search summary that reported "shuffling increases attack success" conflated sources and should
not be relied on.

Practical effect: **the foil is ours to build, not inherited.** That costs this entry the
"pre-legitimised control" advantage that instances A and C of the permutation family have, and
is a further reason it sits below them. It remains an exact multiset match by construction — we
just cannot cite anyone else for having run it.

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
| Mahaut & Franzon, *Repetitions are not all alike: distinct mechanisms sustain repetition in language models*, 2025 ([arXiv:2504.01100](https://arxiv.org/abs/2504.01100)) — verified | two mechanisms, not one: **ICL repetition** has "a dedicated network of attention heads that progressively specialize over training", while **natural repetition** "emerges early and lacks a defined circuitry" and "focuses disproportionately on low-information tokens" |

The two-mechanism split is a design instruction, not a footnote. Only ICL repetition has
organised circuitry; natural repetition is an unstructured fallback when the model cannot
retrieve relevant context. A dictionary comparison should run them as separate conditions — and
the ICL-repetition condition is the one where a structured temporal code plausibly has something
to bind to, which also makes it continuous with the induction entry above.

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
induces backtracking in R1-Distill at layer 10, the steering lever.

Venhoff, Arcuschin, Torr, Conmy, Nanda, *Understanding Reasoning in Thinking Language Models via
Steering Vectors*, 2025 ([arXiv:2506.18167](https://arxiv.org/abs/2506.18167) — verified) steer
expressing uncertainty, generating examples for hypothesis validation, and **backtracking** in
DeepSeek-R1-Distill models across 500 tasks in 10 categories.

**And that is a saturation risk for the repo's own incumbent task, which should be said plainly.**
Their finding is that these reasoning behaviours "are controlled by **linear directions** within
the model's activation space". A single linear direction that already steers backtracking is
precisely the strong per-token baseline that leaves a window code little to add — the same shape
of objection that rules out refusal (Arditi saturation) and argues against emergent misalignment
(a transferable direction). The repo's own backtracking result reports Δgc 0.541 for the TXC
against 0.400 per-token, so the gap is real but not large, and any writeup should position
against Venhoff et al. rather than only against a naive baseline.

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

### Permutation composition / state tracking — priority 3, and the best *mechanistic* control

Added late, and it earns its place for one reason: it is the only entry where the model's
order-dependent computation has been characterised at the algorithm level, on a task that is
**literally permutation composition**.

| paper | what it gives us |
| --- | --- |
| Li, Guo, Andreas, *(How) Do Language Models Track State?*, ICML 2025 ([arXiv:2503.02854](https://arxiv.org/abs/2503.02854)) — verified | the task — "permutation composition (i.e., to compute the order of a set of objects after a sequence of swaps)" — and two identified mechanisms: one closely resembling the **associative scan** construction, and a hybrid that uses **permutation parity** to prune the output space before refining with an associative scan. Methods are prefix patching and linear probing across layers |

**P1 — exact, and about as pure as it gets outside a toy.** The same multiset of swaps applied
in different orders yields different final permutations. Every permutation-symmetric readout is
at chance on the final state by construction, and the data generator is a few lines.

**P2 — passes.** The associative-scan mechanism is *cumulative*: the state after `k` swaps
depends on all `k` in order, so an intervention that changes the outcome has to act at a
particular point in the sequence.

**Why it matters even though it is a capability, not a behaviour.** It is the natural
**calibrated positive control** for this whole programme. The sprint's synthetic order task can
be dismissed as a construct; permutation composition cannot, because there is a published
mechanistic account of how transformers actually compute it, and the computation is
order-dependent by definition. If a temporal crosscoder cannot beat a per-token SAE at steering
here, it is unlikely to anywhere — and that is a fast, cheap, decisive negative. Conversely a
win here is interpretable against a known algorithm rather than against a black box.

**Harness fit.** Segment = one swap, `T` = number of swaps. Judge-free: accuracy or margin on
the correct final-order token. Unlimited data, tunable difficulty, no model download.

**Honest limit.** Nobody wants to *steer* permutation composition. Run it as the control and
anchor, not as the headline — the headline should be demonstration order.

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
| Roger & Greenblatt, *Preventing Language Models From Hiding Their Reasoning*, 2023 ([arXiv:2310.18512](https://arxiv.org/abs/2310.18512)) — **id corrected**, see note | the encoded-reasoning threat model and the paraphrasing defence |
| *Large language models can learn and generalize steganographic chain-of-thought under process supervision*, 2025 ([arXiv:2506.01926](https://arxiv.org/abs/2506.01926)) — id unverified | the closest thing to a *recipe for a model organism*: encoded CoT induced by process supervision, and it generalises |
| Zolkowski, Nishimura-Gasparian, McCarthy, Zimmermann, Lindner, *Early Signs of Steganographic Capabilities in Frontier LLMs*, 2025 ([arXiv:2507.02737](https://arxiv.org/abs/2507.02737)) — verified | the capability gate, and it closes this entry for tonight: "current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances", succeeding only "if given additional affordances like using an unmonitored scratchpad and coordinating on what encoding scheme to use" |
| *Conceptual Steganography*, 2026 ([arXiv:2605.26537](https://arxiv.org/abs/2605.26537)) — id unverified | payloads carried by patterns of reasoning behaviour rather than lexical choice — the shape closest to a temporal factor |

Note on the correction: I first recorded this as arXiv:2311.02282, which is a spark-plug fault
diagnosis paper. The Roger & Greenblatt paper is **2310.18512**. Flagging it because it is a
worked example of why the unverified tier of the ledger below should be treated as untrusted —
a plausible-looking id was simply wrong.

**Temporal signature.** The payload lives in the arrangement of otherwise-innocuous content —
an acrostic across sentence initials, a parity code over sentence lengths. Shuffling the
sentences destroys the payload while preserving the multiset exactly.

**P1 — exact and perfect. P2 — passes**: suppressing an encoded channel requires disrupting
the relation between positions, not adding a constant.

**Demoted to priority 2 on verification.** The blocking question was whether a model at
accessible scale *computes* the payload at all — if it does not represent it, there is nothing
to read or steer. Zolkowski et al. answer it: **frontier** models cannot encode short messages
without a monitor noticing under standard affordances, and succeed only with an unmonitored
scratchpad and a pre-coordinated scheme. A 1.5–7B model will be far worse. So the model organism
does not exist at a scale this sprint can reach, and the entry is a *future* one, contingent on
either the process-supervision recipe in arXiv:2506.01926 producing a small organism, or on
capabilities improving. The structural argument remains the most elegant in the catalogue — a
payload in sentence arrangement is invisible to any per-position readout by construction — which
is why it stays recorded rather than deleted.

### Sandbagging / deliberate underperformance — priority 3

The best available *released* model organisms of any entry, and the largest novelty gap —
nobody has tried to *steer* an armed state — but the weakest temporal story.

| item | detail |
| --- | --- |
| van der Weij et al., *AI Sandbagging: Language Models can Strategically Underperform on Evaluations*, 2024 ([arXiv:2406.07358](https://arxiv.org/abs/2406.07358)) | the defining paper; a Llama-3-8B-Instruct organism with passphrase-dependent underperformance on hazardous-domain questions |
| Greenblatt et al., *Stress-Testing Capability Elicitation With Password-Locked Models*, 2024 ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550)) | password-locked organisms and the elicitation framing |
| *The Elicitation Game*, 2025 ([arXiv:2502.02180](https://arxiv.org/abs/2502.02180)) — id unverified | evaluation of elicitation techniques across organisms |
| *AuditBench* ([arXiv:2602.22755](https://arxiv.org/abs/2602.22755)), *The Model Organism Lottery* ([arXiv:2607.01033](https://arxiv.org/abs/2607.01033)) — ids unverified | 2026 collections of hidden-behaviour organisms; the latter warns interpretability results depend strongly on how the organism was trained |
| *Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging* ([arXiv:2604.26206](https://arxiv.org/abs/2604.26206)) — verified | links sandbagging to the option-order family, and quantifies how strong a purely positional prior can be: under sandbagging instructions the model enters "a low-entropy response-position basin centred on E/F/G that is highly stable and largely content-invariant", holding under complete content rotation across 2,000 items (Pearson r = 0.9994, JSD = 0.027), with accuracy of **72.1% when the correct answer lands in the preferred position E against 4.3% at position A**, in 7–9B models |

That last result cuts both ways and is worth reading carefully before anyone revisits MCQ option
order. It is a spectacular demonstration that answer-slot position effects are real and enormous.
But the effect is explicitly **content-invariant** — a prior over response positions that
survives complete content rotation — which is the definition of a factor a per-token code can
represent and a constant write can move. It reinforces rather than rescues the decision to drop
instance B.

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
  ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)). **Additional counter-evidence:**
  *Convergent Linear Representations of Emergent Misalignment*
  ([arXiv:2506.11618](https://arxiv.org/abs/2506.11618) — verified) shows that emergently
  misaligned models converge to similar representations, and extracts a transferable
  "misalignment direction" from fine-tuned activations that reduces misaligned behaviour in
  models trained on *different* datasets. Their minimal organism uses 9 rank-1 adapters, six
  contributing to general misalignment and two specialising for domain-specific misalignment —
  so this is not a claim of one unified direction, and it should not be quoted as one. The
  relevant point for us is narrower but still holds: a transferable *direction* already
  captures enough of the factor to steer with, which is the regime where a per-token dictionary
  suffices. It argues against EM as a temporal-code demonstration rather than for it.
- **Agentic goal drift / context rot** — genuinely temporally extended, and 2026 has a lot of
  it: goal drift, context drift, role drift, plan decay over long trajectories, with reported
  accuracy drops of 14–85% from context length alone even when all relevant information is
  present, and coherence problems appearing after 25–30 tool calls. Anchors:
  *Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals*
  ([arXiv:2603.03258](https://arxiv.org/abs/2603.03258)), *Agent Drift: Quantifying Behavioral
  Degradation in Multi-Agent LLM Systems Over Extended Interactions*
  ([arXiv:2601.04170](https://arxiv.org/abs/2601.04170)), *The Long-Horizon Task Mirage?*
  ([arXiv:2604.11978](https://arxiv.org/abs/2604.11978)) — all ids unverified. Priority 1 for
  this sprint purely on feasibility: the timescale is thousands of tokens and tens of tool
  calls, the metric needs a judge or a task harness, and no matched-multiset foil is available
  (drift is not a permutation of anything). It is the natural *next-sprint* target if the
  window code turns out to help at long range, and it is the setting where PSAE-style slow
  features were already shown to be useful.
- **CoT unfaithfulness** — the mismatch is between a stated reason and an internal one: a
  relation between a prompt hint and a final answer, not a pattern across positions. Turpin et
  al. ([arXiv:2305.04388](https://arxiv.org/abs/2305.04388)); Lanham et al., *Measuring
  Faithfulness in Chain-of-Thought Reasoning*
  ([arXiv:2307.13702](https://arxiv.org/abs/2307.13702)). I looked specifically for a
  *shuffled-CoT* control, which would have made this a permutation task with an exact multiset
  match, and **could not find one** — the 2026 paper a search summary attributed it to (*Measuring
  and curing reasoning rigidity*, [arXiv:2603.22816](https://arxiv.org/abs/2603.22816), Basu &
  Chakraborty) in fact uses a Step-Level Reasoning Capacity metric, not a shuffle test. That
  paper does supply a reason to *expect* a shuffled-CoT task to fail: it measures **step
  necessity**, finding o4-mini at 74–88% on five of six tasks but Grok-4's reasoning mode at
  1.4% against 7.2% for its non-reasoning mode. Where steps are not necessary, the CoT is
  decorative and permuting it will not change behaviour — so there is no factor to steer. A 1.5B
  model is the least likely of all to have necessary steps.
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
  lost-in-the-middle quote), 2402.08939 (Chen et al., ICML 2024, >30% premise-order drop,
  R-GSM), 2405.03205 (Li & Gao, ACL 2025 Findings, anchored-bias localisation in GPT-2),
  2404.13208 (Wallace et al., instruction hierarchy),
  2402.14811 (Prakash et al., ICLR 2024, positional entity-tracking circuit), 2605.30233 (Tang
  et al., aggregation-at-last-token quote), 2606.13705 (Lazaridis et al., single-neuron
  repetition fix and the "doom loops" caveat), 2206.02369 (Xu et al., NeurIPS 2022,
  self-reinforcement quote), 2511.05541 (Bhalla et al., ICLR 2026 oral, T-SAE loss and config),
  2607.17117 (Persistent SAEs, EMA recurrence and prompt-injection intervention — abstract and
  full-text pages both fetched and consistent), 2605.05892 (Jin et al., FLAS, position-varying
  steering and AxBench numbers), 2309.03882 (Zheng et al., ICLR 2024 Spotlight, token-bias
  attribution and PriDe), 2506.11618 (convergent EM representations, 9 rank-1 adapters).
- **Canonical, high confidence, not re-fetched:** 2209.11895, 1904.09751, 2407.07011,
  2410.13497, 2102.09690, 2309.03882, 2406.07358, 2405.19550, 2404.01833, 2305.02363,
  2502.17424, 2507.21509, 2310.13548, 2305.04388, 2507.12638.
- **Verified in the 2026-07-23 sweep, carried over:** 2605.12726, 2510.20487, 2606.30449.
- **Also verified this session:** 2511.09700 (Li et al., order variance comparable to
  example-set variance, 0.5B–27B), 2510.03417 (NEXUS, EMNLP 2025 — confirmed real, but the
  turn-shuffle ablation attributed to it could *not* be confirmed), 2503.02854 (Li, Guo,
  Andreas, permutation composition and the associative-scan mechanisms), 2410.09102 (Wu et al.,
  ISE, ICLR 2025), 2504.01100 (Mahaut & Franzon, two repetition mechanisms), 2506.18167
  (Venhoff et al., reasoning steering vectors — and the linear-direction saturation risk for
  backtracking), 2604.26206 (position attractor in prompted sandbagging), 2603.22816 (Basu &
  Chakraborty — confirmed real, but it uses a Step-Level Reasoning Capacity metric, **not** the
  shuffle test a search summary attributed to it), 2507.02737 (Zolkowski et al., steganographic
  capability gate).
- **Canonical, added late:** 2306.05685 (Zheng et al., MT-Bench / LLM-as-a-judge position bias
  and the swapping control), 2310.18512 (Roger & Greenblatt).
- **Corrected:** *Preventing Language Models From Hiding Their Reasoning* is **2310.18512**,
  not 2311.02282 as first recorded — 2311.02282 is a spark-plug fault-diagnosis paper. One
  guessed id in this note has already turned out wrong, which is the reason for the tier below.
- **Search-surfaced, arXiv id NOT verified — do not cite externally without checking:**
  2511.04694, 2507.07810, 2604.10044, 2601.05693, 2602.22755, 2607.01033, 2502.02180,
  2605.07984, 2408.15221, 2605.01687, 2605.02647, 2606.08644, 2605.26537, 2506.01926, 2603.03258, 2601.04170, 2604.11978, 2605.03907, 2603.05805, 2606.26474,
  2512.02194.
- **Claims withdrawn on checking:** that a published turn-shuffle ablation exists for
  crescendo-style attacks (could not confirm in Crescendo or NEXUS); that R-GSM is
  multiset-matched (it permits word edits); that EM is carried by a single unified linear
  direction (it is a transferable direction from a 9-adapter organism).

### Changelog

Times are wall-clock against the sprint window opening at 2026-07-25 22:33 PDT.

- **Pass 1** (22:40 PDT) — criterion restatement; entries for induction, repetition,
  backtracking, sandbagging, refusal, lower tier.
- **Pass 2** (22:47 PDT) — added instruction-order conflict, multi-turn escalation,
  entity/state tracking; added the S1/S2/S3 steering-fairness section; verified five citations,
  including the single-neuron repetition result that demoted repetition; added the
  external-corroboration note.
- **Pass 3** (22:51 PDT) — restructured around the prompt-permutation family; added the P4
  harness-fit criterion after confirming the harness windows over *segments*, not tokens, which
  promotes block-structured tasks and demotes induction; added steganography.
- **Pass 4** (22:53 PDT) — added instance D, premise order / R-GSM, and made it the recommended
  instance: largest published gap (>30%), a public dataset of multiset-matched permutation
  pairs, a reasoning failure rather than a label prior, no published localisation. Added the
  localisation risk for instance B and the whitespace note that no dictionary work exists on
  order sensitivity.
- **Pass 5** (22:55 PDT) — added the baselines section. Resolved carried-over debt 2: the tSAE
  described is Bhalla et al., ICLR 2026 oral (arXiv:2511.05541), InfoNCE over adjacent
  positions, no attention, BatchTopK k=20 — so the repo's attention-based `TemporalSAE` was
  never the right baseline and the `l1_coef` calibration debt was an artefact of the wrong
  architecture. Added Persistent SAEs (arXiv:2607.17117) as the missing middle rung. Proposed
  the four-rung capacity ladder.
- **Pass 6** (22:57 PDT) — added prior art on the mechanism: FLAS (arXiv:2605.05892) already
  publishes position-varying steering beating position-invariant steering, at scale on AxBench.
  Narrowed the claims the sprint can make to three; flagged the "crosscoder" naming hazard.
  Added the acquisition table, the six-step recipe, agentic goal drift, and the EM
  linear-representation counter-evidence.
- **Pass 7** (23:00 PDT) — T-SAE code and trained weights located
  ([AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes)); recorded
  their judge-based steering protocol and the open question of whether their steering is
  applied uniformly across positions. Corrected the changelog timestamps in this section, which
  had been estimated rather than read off the clock and were about four hours fast.
Passes 8–12 below all fall between two clock readings I actually took, 22:58 and 23:07 PDT;
the per-pass times are ordering only, not measurements.

- **Pass 8** — verification round. Confirmed Zheng et al. (2309.03882) attribute
  selection bias to **token bias** over option-ID tokens, which makes instance B
  per-token-steerable and therefore a likely predicted-negative rather than a candidate;
  narrowed the family recommendation to instances D and A, where the permuted units are content
  blocks with no label token to carry a prior. Softened the EM counter-evidence: 2506.11618
  extracts a *transferable* misalignment direction from a 9-adapter organism (six general, two
  domain-specific), which is not the single-unified-direction claim I had written.
- **Pass 9** — read the R-GSM construction section and **retracted instance D as the
  recommended task**: R-GSM permits "minor editing on words … to ensure grammatical
  correctness", so it is not multiset-matched and P1 fails for it; it is also only 220 pairs and
  calibrated to frontier models (drops of 6.9–15.5 points on GPT-4-turbo through GPT-3.5-turbo,
  not the "over 30%" of the abstract, which refers to their broader logical-reasoning setting).
  Recommendation moved to **instance A, demonstration order**, where permutation is verbatim and
  the multiset match is exact. Added the free supervised ceiling that instance A provides.
- **Pass 10** — stress-tested instance A the same way. It holds: permutation is
  verbatim by construction, and Li et al. 2025 (arXiv:2511.09700) replicate the effect across
  **0.5B–27B** on classification and generation, reporting order variance comparable to
  example-set variance. That removes the feasibility risk that killed instance D. Added the
  design constraints that follow — base model rather than instruct (instruction tuning increases
  perturbation consistency), `k` of 4–8 rather than many-shot (order sensitivity falls as
  demonstrations grow), classification tasks — and recorded the order-selection oracle as both
  ceiling and rival baseline.
- **Pass 11** — citation-integrity round. Corrected the Roger & Greenblatt id
  (2310.18512, not the 2311.02282 I had guessed, which is a spark-plug fault-diagnosis paper).
  Withdrew the claim that a published turn-shuffle ablation exists for crescendo-style attacks —
  not in Crescendo's abstract, not in NEXUS's; the search summary that reported it conflated
  sources — so that foil is ours to build rather than inherited, which costs the entry its
  pre-legitimised-control advantage. Added the "claims withdrawn on checking" tier to the
  ledger.
- **Pass 12** — added **permutation composition / state tracking** (Li, Guo,
  Andreas, ICML 2025) as the best *mechanistic* control in the catalogue: the task is literally
  permutation composition, the multiset match is exact, and the computation has a published
  algorithm-level account (associative scan, plus a parity-pruning hybrid). It is the fast
  decisive negative if the crosscoder cannot win at steering there. Also searched for a
  shuffled-CoT control and **could not find one**; recorded why a shuffled-CoT task should be
  expected to fail anyway (step necessity as low as 1.4% in some reasoning modes means the CoT
  is decorative and permuting it changes nothing).
- **Pass 13** (23:10 PDT, clock read) — verified four more: ISE (2410.09102, ICLR 2025 — its
  architectural fix implies provenance is under-encoded in unmodified models, which supports the
  instruction-order entry, and it names two usable benchmarks); *Repetitions are not all alike*
  (2504.01100 — two mechanisms, only ICL repetition has organised circuitry, so the conditions
  must be run separately); the position attractor in prompted sandbagging (2604.26206 — 72.1%
  against 4.3% accuracy by answer slot, but explicitly content-invariant, which reinforces
  dropping instance B); and Venhoff et al. (2506.18167). The last of these is a **saturation risk
  for the repo's own incumbent task** and is recorded in the backtracking entry.
- **Pass 14** — added the main-conclusion section (six literatures, six per-token handles; the
  survivors all share P1). Added **instance E, LLM-as-a-judge position bias** — judges pick the
  first slot in ~68% of comparisons against human preference, and the swapping control is
  already standard practice, so the foil is pre-legitimised and the stakes are the highest in the
  catalogue; held as the strongest fallback rather than first choice because it needs a ~7B judge.
  **Demoted steganography from 3 to 2**: Zolkowski et al. (2507.02737, verified) show frontier
  models cannot encode short messages without a monitor noticing under standard affordances, so
  no organism exists at a scale this sprint can reach.
