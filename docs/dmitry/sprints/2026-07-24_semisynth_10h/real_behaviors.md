---
author: Claude realmodel agent
date: 2026-07-24
tags:
  - reference
  - in-progress
---

## Real-model steerable behaviors — a DC-vs-trajectory census

Purpose: the sprint's synthetic result (windowed template ≫ per-token broadcast, growing
in `k`) needs a bridge to behaviors people actually steer. This note enumerates the known
steerable behaviors in the activation-steering / representation-engineering / SAE-steering
literature, classifies each one against the taxonomy in
[[semisynthetic_language_tasks]], gives its rendering in our harness, and ranks the three
where a windowed handle should genuinely beat per-token broadcast.

The census's honest headline is negative for most entries: **almost every published
steering result is a DC result.** The standard recipe — compute one difference-of-means or
SAE-decoder direction, add it at every position after the prompt — is a broadcast write,
and it works. Where a behavior is carried by a window-constant mode, a temporal dictionary
adds nothing *for steering*, and the correct claim for the paper is detection/decodability
(the position taken at the end of [[semisynthetic_language_tasks]]). The trajectory cases
are always a *modulation* of a DC behavior over time, not a different behavior.

### The classification rule, stated so it can be applied mechanically

For a behavior with attribute direction `u` and a target that spans `k` segments:

- **DC / mode (window-constant).** There is a single scalar level of the attribute that the
  whole span should sit at. Writing `m·u` at every position achieves it. A windowed handle
  can *at best* tie, and the theory says broadcast is the *stronger* arm because it
  reinforces a shared contextual mode at every slot (the ordered-days/numbers negative
  result: broadcast beat the template by 10–50× at `k ≥ 5`).
- **Trajectory (window-structured).** The target is a *time-course* `p_1 … p_k` of the
  attribute, and the contrast of interest is against a **multiset-matched foil** — a
  permutation of the same profile. Then every bag/mode statistic is identical between
  target and foil, so *no* DC write can separate them, provably; only a per-segment
  schedule can. Empirically broadcast lands at 0 or slightly negative on all four
  trajectory tasks.

The strictness test to apply to any candidate: **can I write the target and the foil as
permutations of the same sequence of segments?** If yes, it is a trajectory task and
broadcast is inert by construction. If the foil requires different content, the behavior
has a DC component and broadcast will eat it.

Harness terms used throughout (from `experiments/temporal_screen/trajectory_steering/`):
**segments** are sentences or turns with known character spans; a **profile** is the
per-segment attribute assignment; the **attribute direction** `u` is a
difference-of-means over mean-pooled segment activations at one layer (`u = unit(mean(seg
| attr+) − mean(seg | attr−))`); the **template** arm writes `s_t·u` at segment `t` with
`s_t = ±1` from the profile, **broadcast** writes `u` at all `k` segments, **single**
writes at one segment. Metric is the teacher-forced diff-in-diff margin against the
permuted foil, plus a generation-mode per-slot classifier accuracy.

### The census

#### Refusal / jailbreak

- **Steering.** The most mature lever in the field. Arditi et al., *Refusal in Language
  Models Is Mediated by a Single Direction* (arXiv:2406.11717, canonical): one
  difference-of-means direction, added to induce refusal on harmless prompts and ablated
  ("abliteration") to jailbreak harmful ones, across a dozen open chat models. This repo
  has working infra (`plots/2026-05-13_arditi*` through `plots/2026-05-15_arditi_prompts/`,
  Qwen-7B L15 resid_post). Follow-ups surfaced tonight: *What Drives Representation
  Steering? A Mechanistic Case Study on Steering Refusal* (arXiv:2604.08524) and
  *Analysing the Safety Pitfalls of Steering Vectors* (arXiv:2603.24543) — both
  search-surfaced, titles as returned, not fetched.
- **Class: DC for the canonical task, trajectory for two specific variants.** "Refuse this
  request" is a single scalar decision broadcast over the whole response — that is exactly
  why abliteration works with one vector at all positions. The temporal claims in the
  literature are *detection* claims: refusal is constructed across the prompt span and
  **collapses at the final token** (Doda, arXiv:2605.12726: harmful-span probe ≈ 0.998 vs
  final token ≈ 0.174), and *Tracing the Dynamics of Refusal* (arXiv:2605.02958, title
  confirmed tonight) reports a sparse upstream "refusal trajectory" that survives when
  attacks suppress the terminal signal, plus steering at the refusal-onset token beating
  the final token. Two variants are genuine trajectories: (i) **mid-response stance
  switching** — comply-then-catch-yourself (safety recovery) vs refuse-then-relent, which
  are permutations of each other; (ii) **multi-turn escalation** (below).
- **Harness rendering.** Segments = sentences of the assistant response. Profile =
  refuse/comply stance per sentence. Attribute direction = refusal DoM over
  refusing-sentence vs complying-sentence segment activations. Foil = a permutation of the
  same stance sequence, built from the same sentence multiset. This is rank #1 below.

#### Sycophancy

- **Steering.** Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*
  (arXiv:2312.06681, canonical, confirmed tonight) is the reference: contrast pairs from
  the sycophancy A/B dataset, mean-difference vector added at all post-prompt positions,
  monotone dose–response in both directions. *Playing Devil's Advocate: Off-the-Shelf
  Persona Vectors Rival Targeted Steering for Sycophancy* (arXiv:2605.21006,
  search-surfaced) reports generic persona vectors matching targeted sycophancy vectors,
  which is itself evidence the target is a broad mode rather than a specific computation.
- **Class: DC.** Sycophancy is a stance held constant across a response. The multi-turn
  version — *capitulation dynamics*, where the model holds a correct answer for `j` turns
  then folds under pushback — is a trajectory, and is the one interesting temporal
  rendering: the safety-relevant quantity is *when* the model folds, not whether it is
  sycophantic on average. But hold-then-fold vs fold-then-hold are not clean permutations
  of the same conversation, because the user turns differ; a matched control needs the
  same user-pushback multiset in a different order, which is buildable but fiddly.
- **Harness rendering.** Segments = turns. Profile = agree/hold per turn. Direction =
  CAA-style sycophancy DoM over turn-level segment activations. Weak candidate tonight
  (needs multi-turn generation and a judge per turn).

#### Honesty / lying / deception

- **Steering.** Well established. Zou et al., *Representation Engineering*
  (arXiv:2310.01405, canonical) reads and controls honesty; Li et al., *Inference-Time
  Intervention* (arXiv:2306.03341, canonical) shifts truthfulness along probe directions;
  Marks & Tegmark (arXiv:2310.06824) and Azaria–Mitchell (arXiv:2304.13734) establish the
  truth direction; Apollo's deception probes (arXiv:2502.03407) and MASK
  (arXiv:2503.03750) supply behavioral evals. The lit review records a steering result
  flipping deception 5%↔96% (arXiv:2509.18058, verify before external use).
- **Class: DC**, with a caution flagged in [[temporal_safety_tasks_litreview]]: the
  "stages" of deception reported in this literature are *layer-depth* stages, not
  token-time stages. Do not cite depth as lead-time. The genuine trajectory variant is
  **staged deception** — set up truthfully, then deceive, then cover — which is a real
  arc but needs an agentic scaffold to elicit and is contradicted as a *latent build-up*
  by Fomin et al. (arXiv:2606.30449: probes "read the situation, not the action"; the
  signal does not strengthen toward the action token).
- **Harness rendering.** Segments = statements in a multi-claim answer. Profile =
  truthful/false per statement. Direction = truth DoM. Buildable from templates (a bank of
  matched true and false factual sentences about the same entities) and it is *cleanly
  permutable*. Strong runner-up: it is the "staged deception" arc with none of the agentic
  cost. The risk is that per-statement truth is not a *steerable* target so much as a
  content fact — writing the direction may change assertiveness rather than truth value.

#### Sentiment / emotion

- **Steering.** The oldest activation-steering demo: Turner et al., ActAdd
  (arXiv:2308.10248, canonical) with the love/hate and anger vectors; every SAE-steering
  release since has sentiment features.
- **Class: DC** in its canonical form (make the text angrier). **Trajectory** in the form
  that matters: an *arc* — calm → alarm → forced calm, or crisis → de-escalation. Already
  validated in our harness as `int_profile` (+12.5 template vs +0.9 broadcast at k=6) and
  `alt_phase` (+21.6 vs −9.3), so this is the class we know works.
- **Harness rendering.** Already built: CALM/TENSE sentence banks, per-segment intensity
  profile, tense-minus-calm DoM. The real-behavior label to attach is emotional
  **de-escalation in support conversations**, which connects to persona drift below.

#### Persona / roleplay drift

- **Steering.** Anthropic persona vectors (arXiv:2507.21509, canonical) monitor and control
  character traits and are used for *preventative* steering during finetuning; OpenAI's
  persona features (arXiv:2506.19823) are the SAE counterpart. Search tonight surfaced
  *Attractor States Emerge in Multi-Turn LLM Conversations* (arXiv:2606.30571) and
  persona-drift measurements reporting 20–40% turn-by-turn decay of an "Assistant Axis"
  projection over 10–15 turns in therapy/philosophy domains, with activation *capping* (a
  projection clamp, not an additive write) as the mitigation — both search-surfaced,
  unverified.
- **Class: DC in level, trajectory in the safety-relevant failure.** The failure is a
  *drift* — a monotone decay of a projection over turns, and an attractor that is hard to
  leave once entered. That is a time-course, and the mitigation people actually want is a
  *schedule* (hold the projection in a band over the conversation) rather than a constant
  push. Note the honest wrinkle: a "hold the level constant" target is achieved by a
  constant write, so drift *correction* is only a trajectory task if the required
  correction varies over turns — which it does, since the drift is monotone, so the
  required counter-write ramps. A ramp is a non-trivial profile but it is not
  multiset-matchable against a permutation (a ramp and its reverse differ in more than
  order only if the content is matched). Buildable but the control is harder.
- **Harness rendering.** Segments = turns. Profile = a ramp of persona-strength. Direction
  = assistant-persona DoM. The multiset-matched version is a rise-fall vs fall-rise mirror,
  which we already ran as `mirror` (+7.8 template vs −3.3 broadcast at k=5).

#### Verbosity / reasoning length

- **Steering.** Confirmed tonight: *Understanding Reasoning in Thinking Language Models via
  Steering Vectors* (arXiv:2506.18167) reports linear directions mediating reflection,
  backtracking and related behaviors in reasoning models, with a coefficient that controls
  reasoning length in both directions; several 2026 preprints (*Agentic Chain-of-Thought
  Steering*, arXiv:2606.03965; *Reliable Control-Point Selection for Steering Reasoning*,
  arXiv:2604.02113) target overthinking. All search-surfaced except 2506.18167.
- **Class: DC for "be shorter", trajectory for effort *allocation*.** "Think less" is a
  scalar knob. The interesting version is a **compute schedule**: think hard on the setup
  steps, compress the routine ones — i.e. a per-step effort profile. That is a genuine
  trajectory and it has a real deployment motive (token budget), but the metric (per-step
  length) is noisy and the multiset-matched control is awkward: "long,short,long" vs
  "short,long,short" over *the same* subproblems is constructible only if the subproblems
  are interchangeable.
- **Harness rendering.** Segments = reasoning steps. Profile = effort per step. Direction =
  verbose-minus-terse DoM over step activations. Feasible but the judge (length per step)
  is weak; runner-up, not top 3.

#### Chain-of-thought behaviors: backtracking, self-correction, verification

- **Steering.** This is the paper's own territory. Ward, Lin, Venhoff, Nanda
  (arXiv:2507.12638, verified previously) show a base-Llama direction induces backtracking
  in R1-Distill at L10, with the effective handle a window ~8–13 tokens *before* the "Wait"
  token; our internal Llama-Scope feat_71839 result corroborates the offset.
  arXiv:2506.18167 confirms backtracking/reflection are separately steerable directions.
- **Class: the single published *trajectory-shaped steering* result in the field** — Shape
  A in [[temporal_safety_tasks_litreview]] terms (anticipatory build-up before a discrete
  trigger). But note carefully: Shape A is *lead-time*, which is a different thing from the
  *profile* structure this sprint exploits. Lead-time says "write earlier than the trigger";
  profile structure says "write a different thing at each slot". Our synthetic wins are the
  latter. The natural CoT rendering of the latter is the **backtrack-then-verify cycle**:
  explore → doubt → correct → verify, whose permutation (verify → correct → doubt →
  explore) is incoherent for exactly the right reason.
- **Harness rendering.** Segments = reasoning sentences. Profile = mode per sentence
  (assert / doubt / verify). Direction = doubt-minus-assert DoM. This is attractive and
  on-message for the paper, but it is the domain reviewers already saw, and a 1.5B model
  does not backtrack reliably enough to score per-slot tonight.

#### Emergent misalignment

- **Steering.** This repo's own stream (`docs/dmitry/c6_em/`); origin Betley et al.
  (arXiv:2502.17424); the mechanism is a persona direction (arXiv:2507.21509,
  arXiv:2506.19823) whose addition induces and whose subtraction suppresses misalignment.
- **Class: DC / Shape B.** A slow persistent mode over turns with no sharp trigger and no
  published token-level temporal analysis. The lit review already rates it weak on the
  temporal claim, and the sprint's own EM cross-domain screen row is recorded as the *most
  DC-dominated* profile seen. Include it in the census as the negative pole: the behavior
  our repo knows best is the one with the least trajectory structure.
- **Harness rendering.** Segments = turns; profile = misalignment level. Only a
  rise-fall/fall-rise mirror would be matched, and there is no reason to think the model
  represents "misaligned at turn 3 specifically."

#### Sandbagging / evaluation awareness

- **Steering.** Strong and recent: Hua, Qin, Marks, Nanda, *Steering Evaluation-Aware Models
  to Act Like They Are Deployed* (arXiv:2510.20487, verified) — a contrastive vector
  suppresses eval-awareness and flips proxy behavior (~80% → ~10% in the agentic setting);
  Nguyen et al. (arXiv:2507.01786, unverified) report SAE-feature steering with probe AUROC
  0.829 at Llama-3.3-70B L24. The paper explicitly recommends applying the steer during
  dangerous-capability evals to mitigate sandbagging.
- **Class: DC / Shape B.** "Am I being tested" is a persistent context state read roughly
  constant across the episode. A window reads it earlier and cleaner (a detection claim);
  there is no per-position schedule to write. One caution from the same literature that
  matters for *any* of our steering claims: control vectors from semantically unrelated
  contrast pairs can have effects as large as the designed vector — so a
  matched-magnitude random-direction arm belongs in every steering experiment we run.
- **Harness rendering.** Segments = task items in an eval transcript; profile = sandbag/try
  per item. A "sandbag on items 1,3,5" schedule is *constructible* and permutable, but it
  is a behavior nobody has shown models represent per-item; this would be inventing the
  phenomenon rather than steering a known one.

#### Alignment faking

- **Steering.** No steering result on the faking *decision* exists that I know of.
  Greenblatt et al. (arXiv:2412.14093, canonical) establishes the behavior with a visible
  scratchpad; *The Refusal Residue* (arXiv:2607.13346, unverified) explicitly asks for
  multi-token extraction.
- **Class: trajectory in the *text*, unknown in the latent.** The faking arc (recognize
  training context → reason about consequences → comply strategically) is visible in the
  CoT, which means the "temporal structure" may be entirely token-level content rather
  than a latent schedule. Highest stakes, weakest evidence; the lit review's advice — do
  not headline — stands.
- **Harness rendering.** Not feasible tonight (needs the scratchpad setting and a large
  model).

#### Uncertainty / calibration expression

- **Steering.** Hedging and confidence are steerable with the standard recipe (reported in
  practitioner surveys and in the reasoning-steering work above; Farquhar et al.'s semantic
  entropy, Nature 2024, is the detection side). Treat "confidence direction" as
  high-confidence-canonical folklore rather than one crisp citation.
- **Class: DC**, with a real trajectory variant: **calibrated hedging placement** — assert
  the parts you know, hedge the parts you do not, within one answer. That is a per-segment
  profile with a genuine deployment motive, and it permutes cleanly (same sentences,
  hedges moved). It is a legitimate #4 candidate; it loses to refusal only on safety
  salience.
- **Harness rendering.** Segments = claims. Profile = hedge/assert per claim. Direction =
  hedged-minus-asserted DoM over matched sentence pairs ("X is likely Y" / "X is Y").

#### Language / code-switching

- **Steering.** Extensively established and, importantly for us, **this is a real behavior
  with its own literature**: *Causal Language Control in Multilingual Transformers via
  Sparse Feature Steering* (arXiv:2507.13410) shifts output language by steering a single
  SAE feature; *Language steering in latent space to mitigate unintended code-switching*
  (arXiv:2510.13849) uses PCA language axes on parallel translations; *Language Steering for
  Multilingual In-Context Learning* (arXiv:2602.02326) and LangFIR (arXiv:2604.03532) are
  further variants. All surfaced tonight; 2507.13410 and 2510.13849 have the strongest
  provenance in the results.
- **Class: DC for "answer in French", trajectory for per-segment language assignment.**
  And per-segment assignment is not a toy: quoted-passage handling (keep the quote in the
  source language, the gloss in the target), interleaved bilingual output, and
  code-switching *control* are real tasks.
- **Harness rendering.** Already run — this is `lang_profile`, and the generation result
  (81.2% per-slot accuracy for template vs 44.4% for broadcast, chance 50%) *is already a
  real-behavior result*, not only a synthetic one. **This is the cheapest bridge available
  to the sprint: relabel the existing gen2 result as per-segment language control and cite
  arXiv:2510.13849 as the behavior's motivation.** Zero new compute.

#### Formality / register / style

- **Steering.** Standard fare for steering demos and SAE feature browsers; no single
  canonical citation, treat as folklore.
- **Class: DC.** The trajectory variant (formal opening → casual body → formal close, i.e.
  register modulation over a document) is real in writing tasks but has almost no safety
  relevance. Useful only as a *generality* control: if the trajectory win reproduces on
  three unrelated attributes (language, intensity, register), the claim is about windowed
  handles rather than about one lucky direction.

#### In-context learning / task vectors

- **Steering.** Hendel et al. (arXiv:2310.15916) and Todd et al. (arXiv:2310.15213),
  independently: a few-shot prompt's task compresses into a vector that can be added into a
  fresh context to induce the task; *Function-Vector Heads Are Two Populations*
  (arXiv:2606.07560, search-surfaced) splits the heads into writers and cancellers.
  Suppressing ICL by ablating the function vector is the mirror operation.
- **Class: DC.** The function vector is *definitionally* a single write. The trajectory
  variant is **task switching within one context** — apply rule A to items 1–3, rule B to
  items 4–6 — which is genuinely per-segment and permutes perfectly (same items, different
  rule assignment). Attractive as a *capability* demonstration and cheap to build, but the
  safety relevance is nil and the "attribute direction" would be a task-vector difference
  rather than a behavior direction.
- **Harness rendering.** Segments = items in a few-shot list. Profile = rule per item.
  Direction = FV(ruleA) − FV(ruleB). Worth logging as the strongest *non-safety* trajectory
  task; it would answer "is this only about surface style?" because rule application is not
  a style.

#### Others worth recording

- **Concept/topic injection (Golden Gate Claude, Templeton et al. 2024, canonical).** Pure
  DC — the canonical demonstration is literally clamping one feature at all positions. The
  trajectory version is topic *scheduling* (write about A then B then A), which is real for
  outline-following and permutes cleanly, but is a capability not a safety property.
- **Sleeper-agent armed state** (Hubinger et al. arXiv:2401.05566; MacDiarmid et al. 2024,
  both canonical). Detection AUROC ≈ 99% from the prompt's final token; **no steering
  result exists** — the biggest open slot in the lit review. Class: Shape B state plus a
  genuine trigger→action latency. Not feasible tonight (needs backdoored models).
- **Hallucination onset.** Shape A with a claimed ~11-token negative detection delay, but
  the sharp lead-time source is an unrefereed preprint; steering handles exist (ACT,
  arXiv:2406.00034, unverified). Do not build on it without replication.
- **Instruction-following / prompt-injection susceptibility.** DC (a "follow the injected
  instruction" mode). The trajectory variant — resist injection at turn `j` after
  complying at turns `< j` — is interesting but needs an agent harness.
- **Self-preservation / shutdown resistance** (arXiv:2604.02174, search-surfaced). Too new
  and too weakly evidenced to classify.
- **Multi-turn attack detection.** *Latent Adversarial Detection: Adaptive Probing of LLM
  Activations for Multi-Turn Attack Detection* (arXiv:2604.28129, search-surfaced) is the
  detection-side counterpart of rank #2 below and worth reading before building it.

### Summary of classifications

| behavior | canonical steering = DC? | trajectory variant exists? | matched-permutation control buildable? | tonight-feasible |
|---|---|---|---|---|
| refusal / jailbreak | yes (abliteration) | **yes** — mid-response stance switch; multi-turn escalation | **yes** | **yes** |
| sycophancy | yes (CAA) | yes — capitulation turn | partly (user turns differ) | marginal |
| honesty / deception | yes (ITI, RepE) | yes — staged deception | **yes** (true/false claim banks) | yes |
| sentiment / emotion | yes (ActAdd) | **yes** — escalation/de-escalation arc | **yes** (validated) | **yes** |
| persona drift | yes (persona vectors) | yes — monotone drift, attractor | mirror only | yes |
| verbosity / reasoning length | yes | yes — effort schedule | awkward | marginal |
| CoT backtracking | **no — lead-time window** | yes — backtrack-then-verify cycle | yes | no (1.5B) |
| emergent misalignment | yes | weak | no | no |
| sandbagging / eval-awareness | yes | not demonstrated | speculative | no |
| alignment faking | no steering result | in text, not shown in latent | no | no |
| uncertainty / calibration | yes | **yes** — hedge placement | **yes** | **yes** |
| language / code-switching | yes | **yes** — per-segment language | **yes** (validated) | **already done** |
| formality / register | yes | yes | yes | yes (control only) |
| ICL / task vectors | yes (by definition) | yes — mid-context task switch | **yes** | yes |

### Top 3: where a windowed handle should beat per-token broadcast

Scored by (evidence of temporal structure) × (safety relevance) × (feasibility tonight on
Qwen-2.5-1.5B/7B, difference-of-means directions, template-generated prompts, no
finetuning).

1. **Staged refusal — stance trajectory within one response.** The lever is the most mature
   in the field and already runs in this repo, the safety framing is direct (mid-response
   safety recovery: catching yourself after starting to comply), and the target/foil pair
   is a literal permutation of the same sentences, so broadcast is provably inert while the
   schedule is exactly what a windowed handle writes.
2. **Multi-turn escalation arc — the crescendo order effect.** The only real behavior whose
   *published* efficacy is an order effect: Crescendo (arXiv:2404.01833) works because each
   turn is benign in isolation and the arc carries the attack, so escalate-vs-shuffle over
   the same turn multiset is the natural matched control and is a safety property people
   actively defend against.
3. **Emotional de-escalation in support conversations.** The intensity-profile version is
   already validated in our harness (`int_profile`, `alt_phase`, `mirror` — all positive
   with broadcast at or below zero), and the persona-drift/attractor-state literature gives
   it a real deployment motive; it ranks third only because the safety stakes are softer
   than refusal and the citations are the least verified.

Runners-up, in order: calibrated hedge placement (best non-refusal safety story, cleanest
permutation), mid-context ICL task switching (best answer to "is this only surface
style?"), staged deception (best arc, weakest steerability).

**One free win, prior to any of these:** `lang_profile` is *already* a real steerable
behavior with its own literature (language steering / unintended code-switching mitigation,
arXiv:2510.13849, arXiv:2507.13410). The existing 81% vs 44% generation result can be
relabelled as per-segment language control tonight at zero compute cost. Do that regardless
of what else runs.

### Experiment spec for #1 — staged refusal (`stance_profile`)

A drop-in for `lang_profile` in
`experiments/temporal_screen/trajectory_steering/full_modal.py`: same segment machinery,
same three arms, same diff-in-diff metric, new banks and a new per-slot classifier.

**Model and hookpoint.** Qwen-2.5-1.5B-Instruct at L14 first (matches every prior run so
the comparison is apples-to-apples), then Qwen-2.5-7B-Instruct at the proportional layer if
budget allows. Prompts **must** go through `tokenizer.apply_chat_template` — the EM sprint's
headline bug was a dropped chat template ([[feedback_reuse_recipe_functions]]), and refusal
lives at the post-instruction chat-control tokens.

**Dataset (templates only, generatable tonight).** No harmful content is needed or
generated: the stance is carried by the *frame*, and the complying sentences are generic
procedural filler.

- `REQUESTS`: ~40 short dual-use-flavoured user requests ("Walk me through how to get past
  a stuck lock on my own shed", "Explain how someone would test a network for weak
  passwords"), template-generated from a verb × object bank. Their only job is to make a
  refusal contextually plausible.
- `REFUSE` bank: ~12 stance sentences ("I can't help with that part.", "That's not
  something I'm able to walk you through.", "I'll have to stop there.", "I won't go into
  those specifics.").
- `COMPLY` bank: ~12 content-free procedural sentences ("Start by laying out what you
  already have.", "The next stage is straightforward.", "Work through it one piece at a
  time.").
- Both banks are split into disjoint halves A and B: **A trains the direction, B is the
  eval**, so a purely lexical direction cannot transfer.

**Construction of a pair.** Draw `k/2` refuse sentences and `k/2` comply sentences. Order
them by profile `p ∈ {R,C}^k` (target) and by a permutation `π(p) ≠ p` (foil). Target and
foil are therefore permutations of the *same* sentence sequence: identical unigram, bigram
and bag-of-segments statistics. Carrier = the chat-templated request plus the assistant
turn opening.

**Profile and arms.** `s_t = +1` if `p_t = R`, `−1` if `p_t = C`; magnitude `m = frac ×
mean activation norm`, frac grid `{0.05 … 0.5}`. Template writes `m·s_t·u` over segment
`t`'s token span (offset by one, as the existing `margin()` does); broadcast writes `m·u`
at all `k` segments; single writes `m·s_0·u` at segment 0 only. Add a fourth arm the
eval-awareness literature demands: **random-direction template** at matched magnitude.

**Direction.** `u = unit(mean(seg | REFUSE) − mean(seg | COMPLY))` over mean-pooled segment
activations from bank A, `n_train = 40` documents. Record `cos(u, u_arditi)` against the
repo's existing prompt-level Arditi direction — a near-zero cosine is the tell that we
have a style direction, not a refusal direction.

**Metrics.** (i) Teacher-forced `Δ = [lp(T) − lp(F)]_steered − [lp(T) − lp(F)]_base`, per
pair, `n_eval = 32`, with SEM — directly comparable to the existing k-sweep table. (ii)
Generation mode: free-generate the assistant response with the segment counter advancing on
sentence boundaries, classify each generated sentence's stance with a refusal-prefix regex
plus an LLM judge on the disagreements, and report **per-slot stance accuracy vs the
intended profile** (chance 0.5) — the behavioral claim. (iii) The sprint's W-sweep on top:
handle spans `W` contiguous segments at fixed knob budget `m`, predicting `Δ ∝ min(mW,
k)/k`.

**Expected direction.** Template grows roughly linearly in `k` (the constant-per-slot
result from `lang_profile` and `alt_phase`); broadcast pinned at ≈ 0 or slightly negative,
because on a matched multiset a DC write can only break the symmetry against you; single
flat in absolute terms, so its *share* of the template decays as `1/k`. Generation-mode
per-slot accuracy: template well above 0.5, broadcast ≈ 0.5.

**Single biggest risk of a boring failure.** *Autoregressive carryover swamps the schedule.*
Refusal has a far stronger self-consistency attractor than language: once the model has
emitted "I can't help with that", the emitted text — not our write — determines the next
sentence's stance, so the marginal effect of steering at slots 2…k collapses and the
template wins only at slot 1. That would look like a null and mean nothing about windowed
handles. The pre-check is cheap and must run first: on unsteered generations from the
template corpus, measure `P(comply at t | refuse at t−1)`. If it is below ~0.15 the task is
attractor-dominated and we should either shorten segments (clause-level rather than
sentence-level) or fall back to rank #3, where intensity has much weaker self-consistency
and is already validated. Second risk, in the same family: the DoM direction turns out to
be apology-style rather than refusal, defused by the disjoint A/B banks and the
`cos(u, u_arditi)` check above.

### Citation confidence ledger

- **Confirmed tonight (title + arXiv URL in search results, not fetched in full):**
  2312.06681 (CAA), 2510.20487 (eval-awareness steering), 2605.02958 (refusal dynamics —
  upgrades the lit review's VERIFY flag to title-confirmed), 2507.21509 (persona vectors),
  2506.18167 (reasoning steering vectors), 2404.01833 (Crescendo), 2507.13410 (multilingual
  SAE steering), 2510.13849 (language steering / code-switching).
- **Search-surfaced tonight, titles as returned, otherwise unverified:** 2604.08524,
  2603.24543, 2604.15557, 2605.21006, 2606.30571, 2604.28129, 2606.03965, 2604.02113,
  2602.02326, 2604.03532, 2606.07560, 2604.02174.
- **High-confidence canonical, from memory, not re-fetched:** 2406.11717 (Arditi),
  2310.01405 (RepE), 2306.03341 (ITI), 2310.06824, 2304.13734, 2308.10248 (ActAdd),
  2310.15916 / 2310.15213 (task and function vectors), 2401.05566 (sleeper agents),
  2502.17424 (Betley EM), 2506.19823 (persona features), 2412.14093 (alignment faking),
  2502.03407, 2503.03750, Templeton et al. 2024 (Golden Gate), Farquhar et al. Nature 2024.
- **Inherited from [[temporal_safety_tasks_litreview]] with its flags unchanged:**
  2507.12638 and 2605.12726 and 2606.30449 (verified there), 2507.01786 and 2509.18058 and
  2607.13346 and 2406.00034 (verify before external use).

Sources consulted for the refresh: [Steering Llama 2 via Contrastive Activation
Addition](https://arxiv.org/pdf/2312.06681), [Steering Evaluation-Aware Language Models to
Act Like They Are Deployed](https://arxiv.org/abs/2510.20487), [Tracing the Dynamics of
Refusal](https://arxiv.org/pdf/2605.02958), [Persona
Vectors](https://arxiv.org/pdf/2507.21509), [Understanding Reasoning in Thinking Language
Models via Steering Vectors](https://arxiv.org/abs/2506.18167), [The Crescendo Multi-Turn
LLM Jailbreak Attack](https://arxiv.org/html/2404.01833v1), [Causal Language Control in
Multilingual Transformers via Sparse Feature Steering](https://arxiv.org/html/2507.13410),
[Language steering in latent space to mitigate unintended
code-switching](https://arxiv.org/html/2510.13849).
