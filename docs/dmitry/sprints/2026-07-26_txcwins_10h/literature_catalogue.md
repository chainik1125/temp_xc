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

Entries are named rather than numbered because the ranking moved repeatedly as the sweep went
on. **The ranking table is authoritative**; the entry sections below are not in priority order,
and several carry a note where the DC-component audit changed their verdict.

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
| repetition onset | three independent fixes: a **single sign-inverted neuron**; repetition-neuron edits; KV-cache tail pruning cutting loop incidence by **>90pp** | 2606.13705, 2507.07810, 2604.10044 |
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

**A second, independent reason to prefer the organism-free tasks.** Szablewski et al., *The
Model Organism Lottery*, July 2026 ([arXiv:2607.01033](https://arxiv.org/abs/2607.01033) —
verified) benchmark four interpretability techniques — activation oracles, **steering**, logit
lens and **sparse autoencoders** — across **54 model-organism variants**, and find that
"MO interpretability depends strongly on training objective, target behaviour, model
architecture, and training data generation pipeline", with significant variance remaining after
controlling for behaviour strength, and that "integrated training often yields less
interpretable MOs than standard post-hoc methods". In other words a dictionary result measured
on a post-hoc-trained organism may be an artefact of how the organism was made, and the more
realistic the organism, the harder interpretability gets.

Every priority-4-and-above entry in this catalogue except backtracking needs **no model organism
at all** — the behaviour is elicited from a stock pretrained model by the prompt alone. That was
originally a convenience argument; after this paper it is a validity argument.

### Constraints for whoever builds this, collected

Every one of these is argued somewhere below; they are gathered here because they are scattered
and because most of them are cheap to satisfy up front and expensive to retrofit.

1. **Score a contrast, never an absolute.** The metric must be the teacher-forced margin between
   the target and its multiset-matched foil, not accuracy on one member of the pair. Few-shot ICL
   has a published constant intervention — writing a **function vector** — that raises absolute
   accuracy; it cancels in a contrast because the task is identical in both orderings. Get this
   wrong and the comparison is dead before it starts.
2. **Permute the interior only.** Hold the first and last demonstration fixed and match the label
   multiset, so recency and majority-label bias — both bag statistics — cannot separate the pair.
   Apply the same restriction to the permutation sampler in the go/no-go.
3. **Size the go/no-go at ~128 permutations.** Typical permutation-to-permutation accuracy std is
   about two points; the dramatic gaps are the tails and must be searched for. A 24-permutation
   sweep showing a small spread means "underpowered", not "no effect".
4. **Use the base model, not the instruct model.** Instruction tuning increases prediction
   consistency under input perturbations, shrinking the effect being steered. This differs from
   the harness default and has to be changed deliberately.
5. **Use `k` of 4–8.** Order sensitivity falls as demonstration count grows; many-shot would
   quietly destroy the effect. It is also the window length the crosscoder handles best.
6. **Log realised L0 for every arm** (carried-over debt 3 — nominal `k` does not bind for the
   crosscoder and the failure is silent).
6b. **Equalise demonstration token lengths, and log the realised injected norm.** The harness
   matches slab Frobenius norm, which equals injected norm only when segments have equal token
   counts; with best/worst permutation selection this can become a systematic bias. See the
   section on it below.
7. **Run the S2 steering arm**, not just S1. The SAE direction applied at oracle-chosen positions
   is the honest per-token baseline; `S3 > S1` alone invites the reply that the baseline was
   handicapped.
8. **Run the tSAE with temporal regularisation at zero as a control**, because the released
   trainer is `TemporalMatryoshkaBatchTopKSAE` and otherwise the arm confounds three changes.
8b. **Use disjoint demonstration pools for dictionary training and steering evaluation**, so the
   claim is about the factor rather than about memorised content.
9. **Carry over the existing controls unchanged** — time-averaged profile, random profile, random
   direction, row-permuted profile, supervised difference-of-means **reference** (not a ceiling — see below). They are already in
   `steer_order_modal.py`.
10. **If any AUC is reported**, note the probe-fragility caveat: in-distribution AUCs in this area
    have a poor track record under distributional shift.

### A harness detail that becomes a confound in the new design

Found by reading `steer_order_modal.py` rather than the literature, and verified numerically.

**What the harness does.** Each write `W` is normalised to unit Frobenius norm over the `(T, d)`
slab, then applied as `h[:, a:b+1, :] += alpha * scale * W[t]` — the *same* vector added to every
token in segment `t`'s span. So the norm actually injected into the residual stream is

```text
alpha * scale * sqrt( sum_t  len_t * ||W[t]||^2 )
```

where `len_t` is segment `t`'s **token count**. Matching `||W||_F` across arms therefore matches
the *slab* norm, not the injected norm, unless all segments have equal token length. Checked with
a small script: at equal lengths a uniform broadcast and a slab concentrating all its norm on one
segment inject identically (3.162 vs 3.162); at unequal lengths (one segment of 30 tokens, five
of 4) the concentrated slab injects **5.477 against the broadcast's 2.887**, a 1.9× advantage at
identical Frobenius norm.

**This does not invalidate the existing headline result.** Segment lengths in the current corpus
do vary — 4 to 9 words, mean 6.5, sd 1.1 — but sentences are drawn into slots at random, so slot
length is independent of slot index and the effect cancels in expectation. It adds variance, not
bias.

**It does become a potential bias in the design recommended here**, and that is the reason to
flag it. The recipe selects the **best- and worst-scoring permutations**, and selecting on
accuracy can select on where the long demonstrations sit. Once length placement is correlated
with the condition, the injected-norm difference is systematic rather than random, and an arm
whose profile happens to weight the long slots gets a real advantage that has nothing to do with
temporal structure.

**Fix, and it is nearly free:** draw demonstrations from a narrow token-length band so all
segments are effectively equal length. Alternatives if that is impractical: divide `W[t]` by
`sqrt(len_t)` before applying, or log the realised injected norm per arm per condition and match
on *that* rather than on `||W||_F`. Logging it is worth doing regardless — it is two lines, and
it is the same class of silent failure as the realised-L0 problem from the last sprint.

### Dose selection: checked, and it is conservative rather than inflationary

Also from reading the harness. `at_best()` takes the **argmax over the alpha grid on the same
test documents used to report the delta and its SEM**, for each arm separately, and the headline
z is computed from those maxima. That is a winner's-curse setup, so it needed checking rather
than assuming.

Simulated it — 4 doses, 200 documents, per-dose SEM 0.64 matching the reported run, with
doc-level noise shared across doses since the same documents are reused (correlation 0.85):

| true dose-response shape | selection bias | in SEM |
| --- | --- | --- |
| peaked, TXC-like (3, 7, 11, 8) | −0.01 | −0.02 |
| flat, SAE-like (0.8, 1.0, 1.0, 0.9) | +0.19 | +0.30 |
| null (0, 0, 0, 0) | +0.26 | +0.41 |

**A miss of mine, recorded before the result below.** I audited the *selection* of the dose and
missed the *coverage* of the grid. The default is `alphas = "0.25,0.5,1.0,2.0"` — **all
positive** — and that one-sided grid is what withdrew the previous sprint's headline: `txc_flat`
was recorded as "inverting" to −8.02 when it is in fact large and positive at negative doses, so a
sign was read as an inversion. The repo had already established the fix as standard elsewhere —
`experiments/ward_backtracking_txc/README.md` specifies a grid symmetric across zero, "no
a-priori reason to favour positive steering", with negative magnitudes as "evidence about
direction sign and arch behavior, not just floor checks" — and `steer_order_modal.py` does not
follow it. I read that README early in this sweep and did not connect the two. **Any dose grid
must be symmetric across zero**; it is now the first thing to check on any steering run, ahead of
the winner's-curse question below.

**On the winner's curse itself: the bias is small and it runs the wrong way for a sceptic.** A well-separated peak is picked
reliably by argmax, so the crosscoder arm gains essentially nothing (−0.02 SEM); the flat and null
arms — the SAE broadcast and the random controls — are the ones inflated, by 0.2 to 0.4 SEM. So
the reported gap and the z are if anything **understated**. No correction to the existing result
is needed, and it is worth saying so explicitly rather than leaving the reader to wonder.

**It matters more for a new task with a smaller effect**, where both arms may be flat-ish: the
inflations largely cancel in the difference, but the SEM does not account for the max-selection,
so the z is genuinely overstated in that regime. Cheap guards, in order of preference: compare
arms **at matched dose** and report the full curve (most interpretable, and the curves are already
plotted); or split the test documents into a dose-selection half and a reporting half.

### Held-out content: the existing result does not test generalisation

Third harness observation. `make_doc` draws from fixed module-level pools — ten CALM sentences,
ten TENSE sentences, a small carrier set — for **both** the dictionary training documents and the
steering test documents. So the dictionaries are trained on activations from documents built out
of the same twenty sentences they are later asked to steer.

**How much this costs.** Less than it first appears, because the task is ordering: both classes
use the same sentences, so a latent keyed on sentence *identity* cannot separate them, and the
row-permutation control breaks a lookup as readily as a genuine profile. What is not ruled out is
narrower — that the latent is specific to these twenty sentences' embeddings rather than encoding
"tense early, calm late" in general. The honest statement of the existing result is therefore:
*a crosscoder can steer the ordering of content it was trained on.*

**Cheap hardening, and it is a strictly stronger claim.** Split the pools: train the dictionaries
on documents built from sentences 0–6 of each class, evaluate steering on documents built from
sentences 7–9. Same everything else. If the effect survives, the latent is a general
attribute-schedule feature and the claim upgrades from "steers this content" to "steers this
factor". If it does not survive, that is worth knowing before the claim is made in a paper.

**For the new task this matters more, not less.** A demonstration-order latent trained on the very
demonstrations it then steers is a much more plausible lookup than a sentence-order latent is,
because demonstrations are longer and more distinctive. **Use disjoint demonstration pools for
dictionary training and for steering evaluation.** If only one thing on this page is adopted
beyond the contrastive metric, make it this one — it is the difference between a result about a
mechanism and a result about a corpus.

### The steering run is not budget-matched, and the mismatch favours the SAE

Fourth harness observation, and the one most worth putting in the writeup, because it makes the
headline result *stronger* than it currently reads.

Both dictionaries are constructed with the same nominal `k` — `TopKSAE(..., k=k)` and
`TemporalCrosscoder(..., T=T, k=k)` — but they spend it over different units. The SAE encodes
`Xn.reshape(-1, d)`, i.e. **per segment**, so it gets `k` active latents for each of the `T`
segments. The crosscoder encodes whole windows, so it gets `k` active latents **per window of `T`
segments**. At the defaults (`k = 8`, `k_seg = 12`) that is:

| arm | coefficients per segment | per window |
| --- | --- | --- |
| TopK SAE | 8 | 96 |
| crosscoder | 0.67 | 8 |

**A 12× sparsity budget advantage to the SAE.** Training throughput is matched — the SAE gets
`batch_win * T` segment vectors per step against the crosscoder's `batch_win` windows, i.e. the
same number of segments — so this is a budget asymmetry specifically, not a data one.

Two things follow. First, this run does **not** follow the sprint's own stated standard of
matching on realised coefficients per segment; that standard was applied to the FVU head-to-head
table, and the steering run inherits the older nominal-`k` convention. Second, the direction of
the mismatch is conservative — the SAE arm was helped by a factor of twelve, not handicapped.
**That no longer rescues the original headline, which has been withdrawn on other grounds** (the
one-sided dose grid), but it applies to any rerun: a symmetric-grid rerun that still shows a
crosscoder advantage will have obtained it against an SAE holding 12× the budget, and that is
worth stating.

**What to actually do.** Report the asymmetry rather than hide it, and if there is compute to
spare, rerun the steering arms with the crosscoder at `k_window = k_segment × T` so the budgets
match — the expected outcome is that the SAE arm gets no better, since the constant-write argument
is structural rather than about capacity, and confirming that closes the loop.

### The second gate: rank, from [[theory_section]], applied to these entries

The theory workstream has produced a second gate that supersedes part of my framing and changes
several verdicts here. Recording the consequences rather than restating the derivation.

**The result.** A per-token dictionary latent is one direction, but its *coefficient* varies with
position — scale the latent by its own activation and the write is one direction on a schedule.
So a per-token dictionary, steered well, reaches any **rank-1** write. A crosscoder reaches
higher rank. The share of the optimal write reachable by rank 1 is `r1 = σ₁²/‖P‖_F²`, computable
before any dictionary is trained. The two gates are ordered:

| gate | condition | rules out |
| --- | --- | --- |
| first | `c ≈ 0` (no bag statistic separates target from foil) | a broadcast write riding a DC component |
| second | `r1` well below 1 | a *scheduled* per-token write |

**The consequence that reorders this catalogue: every two-block swap has a rank-1 optimal write.**
If conditions A and B differ by exchanging two blocks, the difference slab is `+Δ` at one slot and
`−Δ` at the other — one direction with a sign flip, exactly rank 1. That applies to:

| entry | block structure | rank verdict |
| --- | --- | --- |
| Instruction-order conflict | two blocks swapped | **rank 1 from the swap alone** — demote from 5 |
| LLM-judge position bias | two responses swapped | rank 1 — already demoted, now doubly |
| Demonstration order | **depends on the permutation**, see below | salvageable, and this is the fix |
| Per-section style scheduling | balanced profile over ~6 segments | high rank — consistent with it having won |
| Permutation composition | `m` swaps | high rank |
| Multi-turn escalation | full shuffle of many turns | high rank |

**Demonstration order satisfies the attribute criterion too, and by accident rather than design.**
The sharper form of the rank bound in [[theory_section]] is `rank(P) ≤ A`, the number of
*attributes* whose positional pattern differs between conditions, with the prescription that the
cheapest natural source of a second attribute is "content plus its own carried state, since a
maintained state's schedule is the running integral of the content's, and an integral is never
proportional to its integrand".

Few-shot demonstrations supply exactly that pair, for free. The **label at position `t`** is the
content. The **running label balance up to `t`** — how many of each class have been seen so far —
is its running integral, and it is what majority-label bias reads. Because the interior-permutation
control *matches the label multiset*, the two orderings agree on the final balance and disagree on
the running one at every interior position. So the content attribute and the state attribute have
different positional patterns by construction, which is the rank-2 condition, obtained without
designing anything for it.

This is worth stating because it is the one place where the constraint that removes a DC handle
(match the label multiset, constraint 2) also *creates* the rank structure. The two requirements
usually trade off — matching more statistics tends to flatten the difference slab — and here they
happen to align.

**The design fix for demonstration order, and it is the single most actionable thing on this
page.** The foil pair must differ by a **large-support permutation, not a transposition**. Swap
two demonstrations and the difference slab is `(demo_X − demo_Y)` at one slot and its negation at
the other — *exactly* rank 1, and a per-token latent on a schedule matches it. Use a **cyclic
rotation of all interior demonstrations** instead: the theory note gives an `m`-block cyclic
rotation rank `m − 1`, with the rank-1 share falling as roughly `2/m`. At `k = 8` with six
interior demonstrations rotated, that is rank 5 and `r1` around 0.33 — comfortably inside the
regime where the gate discriminates.

This matters because "best versus worst ordering" says nothing about *how* the two orderings
differ. A search over permutations could easily return a best/worst pair differing by a single
transposition, which would be a rank-1 task wearing the clothes of a permutation task. **Constrain
the search to rotations, or measure `r1` on the selected pair before training anything.**

**A note on instruction-order conflict, which I had at joint-first.** It is not dead, but its
headroom is much smaller than I implied. The theory note reports the instruction-recency task
having genuine rank-2 structure — the instruction's lexical content and the downstream
governing-instruction state occupy disjoint positions and distinct directions — with `r1 = 0.829`
measured from the gradient. So roughly 17% of the optimal write is inexpressible by any rank-1
intervention. Real, but a seventeen-percent residual rather than a structural impossibility, and
it should be described that way.

**What this does to my own framing.** I argued that P1 (matched multiset) is the necessary
condition. That remains true and is now gate one. What I missed is that it is not *sufficient*
even after the DC audit: a matched multiset with a two-block structure passes gate one and fails
gate two. The honest summary is that I was checking whether a constant write could work, and the
sharper question is whether a *scheduled* write could.

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
2. **Build the matched pairs, with the interior-permutation control, and use a rotation not a
   swap.** The *same* demonstrations in two orders — ideally the best- and worst-scoring
   permutations from step 1, maximising the gap the steering has to close. **Constrain the
   permutation to a cyclic rotation of the interior demonstrations**: a two-block swap has an
   exactly rank-1 optimal write that a scheduled per-token latent matches, so a best/worst pair
   that happens to differ by one transposition is a rank-1 task in disguise. See the rank-gate
   section above; measure `r1` on the selected pair before training anything. **Permute only the interior: hold the first and last
   demonstration fixed and match the label multiset.** This is not optional. Recency and
   majority-label bias are bag-of-positions statistics (named in *Calibrate Before Use*), and
   leaving them free hands the per-token arm a DC handle that will close the gap for reasons that
   have nothing to do with ordering. Restrict step 1's permutation sampler the same way, so the
   best/worst search runs over interior permutations only.
3. **Train the ladder at matched realised coefficients per segment** — BatchTopK SAE → PSAE →
   T-SAE → TXC. Log realised L0 for every arm (carried-over debt 3; the failure is silent).
4. **Read and steer separately.** Expect reading to favour the SAE again — that is now the
   predicted result, and a fourth replication of it is worth reporting in its own right.
5. **Steering arms S1/S2/S3** as in the fairness section below. `S3 > S2` is the claim; `S3 > S1`
   alone is not enough.
6. **Controls that can kill it:** time-averaged profile, random profile, random direction,
   row-permuted profile, and the supervised difference-of-means **reference** (not a ceiling — see below). These already exist in
   `steer_order_modal.py` and should be carried over unchanged.

The one-line version of the result if it works: *a temporal crosscoder can convert a model's
worst demonstration ordering into its best, where a single steering direction provably cannot,
because the two orderings are the same multiset.*

That framing has a useful property: the **supervised reference is known and free**. The
best-ordering accuracy is measured in step 1, so unlike the original order task there is a
natural, interpretable upper bound on what any intervention could achieve, and the result can be
reported as a fraction of a real gap closed rather than as an uncalibrated Δmargin.

### The criterion the repo has already established empirically, which is sharper than mine

Read [[semisynthetic_language_tasks]] before acting on anything below. It contains executed
experiments that supersede the reasoning I set out in the next section, and I had P1 and P2 the
wrong way round.

**What that note establishes.** Two candidate language demonstrations were run on Modal and both
**failed**:

- *Passphrase verification* (k distinct code-words, a textbook position-dependent template)
  failed because it is a **conjunction** — a single or broadcast write satisfies it.
- *Ordered generation* (days, numbers) failed because it is **mode-dominated**: at k ≥ 3–5 the
  per-token SAE broadcast matched or beat the crosscoder template by 10–50×, and the gap ran the
  *wrong* way, with the template fading as k grew rather than growing.

The stated mechanism is the part that matters: natural-language structured generation is driven
by a strong shared contextual *mode*, which a broadcast write reinforces at every position, and
"the per-position template's specific writes are fragile and are overwhelmed as the sequence and
the mode strengthen".

**Then the fix, and it worked.** Trajectory tasks with **multiset-matched foils** — a permutation
of the same profile, "so no bag/mode statistic — hence no broadcast write — separates target from
foil in principle". Four of four passed, and the full k-sweep gives template growing roughly
linearly in k (lang: +75.7 at k=2 to +218.9 at k=10) while **broadcast is pinned near zero or
negative at every k**, and single-position is flat, its share decaying as 1/k. A
generation-mode demo reached 0.812 ± 0.044 per-slot accuracy for the template against 0.444 for
broadcast at chance 0.5.

The conclusion in the note's own words: *windowed steering wins exactly when the target is a
trajectory with no DC component, and per-token steering wins when a broadcastable mode carries
the behaviour.*

**The correction to my framing.** I wrote that P2 (write non-constancy) is the operative
criterion and P1 (matched multiset) is a clean special case. That is backwards. The passphrase
result is a direct refutation: it has P2 in textbook form and still loses, because a broadcastable
mode exists. **P1 is the necessary condition**, precisely because a matched multiset is what
guarantees no DC component for a broadcast write to ride.

So the operative test for every entry is not "is the write position-dependent?" but:

> **Is there any bag-of-positions statistic — a mode, a label prior, a level — that separates
> target from foil? If yes, a broadcast write can ride it and the crosscoder loses.**

### The distinction the audit forces, and it is the sharpest thing in this note

P1 and the DC-component test are **not the same condition**, and conflating them is what made me
rank induction second. Two separate things have to hold:

- **(a) The foil is matched.** No bag-of-positions statistic *separates target from foil*. This is
  P1, and a matched multiset delivers it by construction.
- **(b) The metric has no DC-movable component.** No constant write *improves the metric*,
  whichever condition it is applied to.

These come apart, and induction is the clean example. Repeat-in-order versus shuffled-repeat is
an exactly matched multiset, so (a) holds perfectly. But the metric is copy accuracy, and
"increase copying" is a constant write that raises copy accuracy in *both* conditions — so (b)
fails, and the per-token arm has a handle even though the foil is airtight. The same is true of
repetition (a constant anti-repetition write), sycophancy (a constant agreeableness write) and
crescendo (a constant permissiveness write).

Where the sprint's winning tasks differ is that the metric is a **contrast between the two
conditions** — a teacher-forced margin between the target ordering and its multiset-matched foil.
A constant write moves both sides of that contrast equally and cancels.

**Do not cite the previous sprint's order-task numbers for this.** The +11.29 against +1.24
result has been **withdrawn** — measured on a one-sided dose grid, and rerun symmetrically the
crosscoder does not beat the SAE significantly. The evidence that survives is the trajectory
tasks, where broadcast measures at or slightly below zero at every window length, and the
measured `c` ordering, where the order task's `c = 0.241` says a constant write always had grip
there.

**Practical rule.** Matching the foil is necessary but not sufficient. The metric must also be a
*difference between the matched pair*, not an absolute score on one member of it. Any candidate
whose success metric is "how much of behaviour X did we produce" has a DC handle regardless of
how well matched the foil is; only "how much more of X in condition A than in condition B" does
not. Checking this takes one line of thought per candidate and it is the check I initially
skipped.

### The DC-component audit

Applying that test to the catalogue, which changes several verdicts. "DC handle" means a constant
write that plausibly moves the metric.

| entry | DC handle a broadcast write could ride | verdict |
| --- | --- | --- |
| Demonstration order | **yes, unless controlled** — recency and majority-label bias are bag statistics (*Calibrate Before Use* names exactly these) | salvageable, see the control below |
| Instruction-order conflict | **no, if the design is right** — with A-then-B vs B-then-A and "obey the first" as the target, the correct content *differs between conditions*, so no single content direction helps both | holds |
| Induction / in-context copying | **yes** — "copying-ness" is a mode, and this is close to the ordered-days/numbers task that already failed | **demote to 2**; expect the repo's existing negative to replicate |
| Repetition loop escape | **yes** — "repetitiveness" is the mode par excellence, and a flat repetition penalty is exactly the broadcast write | **demote to 2** |
| LLM judge position bias | **yes** — the verdict is a label token, so a "prefer B" write is a DC handle | demote within the family |
| Permutation composition | **probably** — the model aggregates at the query token, so a single write there may set the state | run at an *upstream* hookpoint or not at all |
| Backtracking | **yes** — a linear direction already steers it (2506.18167), and "doubt" is a mode | consistent with its modest 0.541 vs 0.400 gap |
| MCQ option order | **yes** — content-invariant position attractor, token bias | already dropped |

Two entries survive cleanly, and they are the two the sprint should spend on.

**The control that saves demonstration order.** The order effect is partly carried by recency and
majority-label bias, both bag statistics. Kill them by construction: **permute only the middle
demonstrations, holding the first and last fixed, and match the label multiset**. Then the
foil pair differs *only* in the arrangement of the interior, no bag statistic separates them, and
the DC component is gone. Without this control a broadcast write may well close the gap, and the
result would be uninterpretable. This is cheap — it is a constraint on the permutation sampler —
and it should be in the corpus builder from the start.

**The honest reframing this forces.** The repo's note concludes "do not pursue a language
*steering* demonstration of template > per-token as a paper headline", on the grounds that the
clean win seems to need mode-free per-position binding, which language behaviours rarely have.
The last sprint's order-only result and the trajectory tasks are the counterexamples — both are
language, both multiset-matched, both won. So the reframing is not "language steering doesn't
work" but the narrower and better-supported: **language steering wins iff the foil is
multiset-matched**. Demonstration order with the interior-permutation control is a test of exactly
that claim on a behaviour with an external constituency, which is the gap the sprint set out to
close.

### The selection criterion, as I first set it out (superseded above, kept for the reasoning)

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
  also holds without it (ramp-then-release, anti-phase, offset-relative copying).
  I originally wrote here that **P2 is the operative criterion and P1 the clean special case**.
  **That is wrong and is the error the section above corrects**: the repo's passphrase experiment
  has P2 in textbook form and still loses to a broadcast write, because a broadcastable mode
  exists. P1 is the necessary condition. The sentence is left in place only so the reasoning that
  produced the mistake is visible.
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
[AI4LIFE-GROUP/temporal-saes](https://github.com/AI4LIFE-GROUP/temporal-saes) — repo confirmed to
exist and inspected. Use the reference loss rather than reimplementing it: this is exactly the
situation the repo's own lessons-learned note warns about (delegate to the reference
implementation; the 2026-05-07 EM replication lost a day to a reimplemented prep function).

What is actually in it, since this determines how much work the tSAE arm is:

| item | detail |
| --- | --- |
| training code | a **fork of `dictionary_learning`**, with a temporal trainer class `TemporalMatryoshkaBatchTopKSAE` |
| entrypoint | `dictionary_learning/dictionary_learning/train_temporal.py`, exposing regularisation, **split fraction**, and standard SAE hyperparameters |
| trained weights | a pre-trained T-SAE for **Gemma-2-2b, width 16384**, on HuggingFace, with feature explanations |
| evaluation | utilities scoring SAEs by "smoothness metrics" |
| other | probing, t-SNE, dataset-understanding and alignment scripts; Poetry for dependencies, `.env` for HF |

**One detail that matters for matched comparison.** The trainer is
`TemporalMatryoshka**BatchTopK**SAE` — the architecture is **Matryoshka** plus BatchTopK plus the
temporal loss, not plain BatchTopK, and the "split fraction" parameter is the 20%/80%
semantic/syntactic partition. So a like-for-like comparison has three differences from this
repo's crosscoder, not one: the temporal loss, the Matryoshka nesting, and the predefined split.
If the tSAE arm wins or loses, that has to be attributed carefully — the cleanest control is to
run their trainer with the temporal regularisation set to zero, which isolates the contrastive
term from the Matryoshka structure.

The released weights are **Gemma-2-2b**, not a Qwen model, so they are useful for sanity-checking
an implementation but cannot be dropped into a Qwen-based comparison; the arm needs training
either way.

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

**A fifth architecture, and it is the natural order-destroying control.** Der, Kamath, Thompson,
*Turn-Averaged SAEs for Feature Discovery and Long-Context Attribution*, June 2026
([arXiv:2606.28548](https://arxiv.org/abs/2606.28548) — fetched) represent a whole Human or
Assistant turn with a fixed number of features by "learning to reconstruct the average model
activation across the turn", motivated by the fact that a standard SAE's active features "scale
linearly with context length". They report turn-averaged features describe a turn's high-level
characteristics more completely than per-token features under LLM evaluation.

Two things follow. It independently validates the harness's segment abstraction — the field is
moving to segment-level dictionaries. And it is exactly the **R2 rung** from
[[temporal_benchmark_screen]]: aggregation with order destroyed. That makes it the cleanest
published instantiation of "pooling without ordering", which is the control that separates a
window code from mere aggregation, and it is now something we can cite rather than invent.

**Recommended baseline set for any task in this catalogue:** TopK/BatchTopK SAE (per-token
floor) → turn/segment-averaged SAE (aggregation, order destroyed) → PSAE (learned scalar
timescale) → T-SAE (InfoNCE over adjacent positions) → TXC (free `(T, d)` slab). Five rungs, each
adding exactly one thing, and every one of them now corresponds to a published architecture. That
is a better story than any single head-to-head, and wherever the curve flattens is informative
regardless of which way it comes out.

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

**And there is now a published rank-1-with-optimal-schedule steering method**, which is the
sharpest possible baseline for the rank argument. Heyman & Vandeputte, *Steer Like the LLM:
Activation Steering that Mimics Prompting*, 2026
([arXiv:2605.03907](https://arxiv.org/abs/2605.03907) — verified) observe that prompt steering
"applies strong interventions on some tokens while barely affecting others", and that "popular
activation steering methods are not faithful to the mechanics of prompt steering". Their Prompt
Steering Replacement models **estimate token-specific steering coefficients from the activations
themselves**, and outperform existing activation steering methods, "especially when controlling
for high-coherence completions", comparing favourably to prompting on AxBench and persona
steering.

That is precisely one direction with a learned per-position schedule — rank 1, with the schedule
optimised rather than incidental. Together with Persistent SAEs it forms a two-point rank-1
ceiling from the literature: PSAE learns a *scalar timescale* per feature, PSR learns a *free
per-token coefficient*. If a crosscoder cannot beat PSR, the advantage was never about rank. If
it can, the rank argument has a published opponent rather than a strawman, which is much the
better position to be in.

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
that meaning unless "temporal / cross-position" is said explicitly every time. Both were checked:
*Localizing RL-Induced Tool Use to a Single Crosscoder Feature* (Shportko et al., 2026) uses
**cross-layer** "Dedicated Feature Crosscoders", and *Sparse Crosscoders for diffing MoEs and
Dense models* (Chaudhari, Hundia, Gulati, 2026) uses **cross-model** BatchTopK crosscoders with
designated shared features. Neither is cross-position. The upside is that the cross-*position*
variant really is under-explored — I found no windowed-dictionary prior art at all.

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

Priorities are **after the DC-component audit above**. The decisive column is "DC handle": if a
bag-of-positions statistic separates target from foil, a broadcast write rides it and the
crosscoder loses, whatever the P1 and P2 columns say.

| behaviour | P1 exact foil | DC handle | organism tonight | judge-free | harness fit | priority |
| --- | --- | --- | --- | --- | --- | --- |
| Demonstration order (interior-permuted, label-matched) | **yes** | **none, with the control** | free | yes | drop-in, segment = demo | 5 |
| Instruction-order conflict / injection precedence | **yes** | **none, with the right target** | free | yes | drop-in, segment = instruction | 5 |
| Per-section style / register scheduling | **yes** | **none** | free | yes | drop-in, segment = section | 4 |
| Tool-call ordering in agents | **yes** | **plausibly none** | free (needs ~7B) | yes | drop-in, segment = call | 3 |
| Multi-turn escalation (crescendo) | **yes** | likely (permissiveness is a mode) | free | yes | drop-in, segment = turn | 3 |
| Permutation composition / state tracking | **yes** | probably (aggregates at query token) | free | yes | drop-in, segment = swap | 3, upstream hookpoint only |
| Entity / state tracking (boxes) | **yes** | probably (same aggregation result) | free + public data | yes | segment = operation | 3 |
| Retrieved-document position | **yes** | unclear | free | yes | drop-in, segment = document | 3 |
| Backtracking / self-correction | no | **yes** (linear direction steers it) | R1-Distill-Qwen-1.5B | no | separate pipeline | 3 |
| LLM-judge position bias | **yes** | **yes** (verdict is a label token) | needs ~7B | yes | drop-in, segment = response | 2 |
| Induction / in-context copying (RRT) | **yes** | **yes** (copying is a mode) | free | yes | needs chunking | 2 |
| Repetition loops (loop escape) | yes | **yes** (repetitiveness is the mode) | free | yes | segment = repeated unit | 2 |
| Steganography / encoded reasoning | **yes** | none | **none at reachable scale** | yes | segment = sentence | 2 |
| Sandbagging / password-locking | no | yes | released organisms | partly | poor | 2, steering already failed |
| Sycophancy build-up | partial | **yes** (agreeableness is a mode) | free | no | segment = turn | 2 |
| Refusal onset | no | **yes** (Arditi saturates) | free (repo infra) | partly | separate | 1 |
| Evaluation awareness | no | yes (Shape B) | released organism | no | poor | 1 |
| Emergent misalignment persona drift | no | yes (transferable direction) | repo c6_em | no | poor | 1 |
| MCQ option order | **yes** | **yes** (token bias, content-invariant attractor) | free | yes | drop-in | 1 |
| Planning / lookahead | no | **yes** (one token, five heads; causal only at 27B) | free | no | poor | 1 |
| Deception / stated-vs-internal divergence | no | **yes** (linear deception direction) | free | no | poor | 1 |
| CoT unfaithfulness | no | yes | free | no | poor | 1 |
| Scheming / alignment faking | no | unknown | limited | no | poor | 1 |

Priority is (probability the TXC actually wins) × (relevance of the behaviour) × (can be run
tonight). The audit collapsed the middle of this table: seven entries that looked like 3s and 4s
have a DC handle, and the two that survive at 5 are the two where the foil removes every
bag statistic by construction.

### Compute feasibility on Modal A10G / L4

The brief asks for feasibility inside a 10h sprint on A10G or L4, so the constraint is stated
directly rather than left as "high / medium / low". Both cards are 24 GB, which is the binding
number.

What fits: a 1.5B model in bf16 is ~3 GB of weights, leaving ample room for a dictionary and an
activation buffer — this is the regime the last sprint ran in and it is comfortable. A 7B in
bf16 is ~14 GB, which fits with activations for short contexts but leaves little headroom for
simultaneous dictionary training; cache activations to disk first and train the dictionary in a
second pass. Anything above 7B is out on a single card.

| entry | model needed | dominant cost | fits a 10h sprint? |
| --- | --- | --- | --- |
| Demonstration order | 1.5B base | the 128-permutation go/no-go sweep, then one activation cache + 5 dictionaries | **yes, comfortably** — contexts are a few hundred tokens |
| Instruction-order conflict | 1.5B–7B instruct | same shape, shorter contexts | **yes** |
| Per-section style scheduling | 1.5B | already run at this scale with DoM proxies; adding trained dictionaries is one cache + 5 trainings | **yes** |
| Permutation composition | 1.5B | trivial data, short contexts | **yes** |
| Entity / state tracking | 1.5B–7B | short contexts | yes |
| Tool-call ordering | 7B (1.5B function-calling is marginal) | two-pass caching on a 24 GB card | tight |
| Multi-turn escalation | 7B instruct | long multi-turn contexts inflate the activation cache | tight |
| LLM-judge position bias | ~7B (1.5B is a poor judge) | two long responses per item | tight |
| Backtracking | R1-Distill-Qwen-1.5B | generation + judge calls, multi-phase pipeline written for pods | **no**, not tonight |
| Sandbagging | 8B organism | download + generation + elicitation | no |

The pattern reinforces the ranking arrived at on other grounds: everything at priority 4 and
above runs on a 1.5B with short contexts, which is the only regime where five dictionary arms and
a full control set fit inside one night on one card.

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

**The specific DC handle this task has, and why the contrastive metric defuses it.** Few-shot ICL
is driven substantially by **function-vector heads** — heads that summarise the demonstrated task
into a single vector, many of which begin as induction heads before transitioning (Todd et al.,
*Function Vectors in Large Language Models*, [arXiv:2310.15213](https://arxiv.org/abs/2310.15213);
*Which Attention Heads Matter for In-Context Learning?*, OpenReview `C7XmEByCFv` — arXiv id not
located). Writing a function vector is a known, effective, entirely *constant* intervention that
raises few-shot accuracy.

So if the metric were "accuracy under the bad ordering, after steering", the per-token arm would
simply write the function vector, score well, and the comparison would be dead before it started.
It is defused by the contrastive metric and only by that: **the function vector encodes the
task, which is identical in both orderings of the same demonstrations**, so writing it moves both
sides of the margin equally and cancels. This is the clearest concrete instance of the (a)/(b)
distinction above, and it is the reason the metric must be the margin between the matched pair
rather than an absolute score.

**The single biggest risk to this entry, stated plainly.** The two *named* mechanisms of ICL
order sensitivity are **majority-label bias** (the model predicts the most common label among
demonstrations) and **recency bias** (it predicts the label of the most recent demonstration),
both identified in *Calibrate Before Use*. Both are bag-of-positions statistics, i.e. DC handles.
The interior-permutation control removes them — but it is an open question how much order
sensitivity is *left* once they are gone. The literature is explicit that a complete mechanistic
account of order dependence does not yet exist, so the residual is real but unquantified.

There is also a scale trade-off to be aware of: the named biases are reported to **weaken as
model size increases**, so a 1.5B model has the largest total order effect *and* the largest DC
share of it. If the controlled spread collapses at 1.5B, trying a 7B is the right next move
rather than abandoning the task.

The saving grace is that **step 1 of the recipe tests exactly this, before any training**. Sample
128 interior permutations with first and last held fixed and the label multiset matched, and
measure the spread. A surviving spread means residual, non-DC order sensitivity exists and the
task is live. A collapsed spread kills it — and is itself a clean, reportable result: *ICL order
sensitivity at this scale is fully accounted for by recency and majority-label bias*, which is a
finding the ICL literature does not currently have. Either outcome is worth the forward passes,
which is the property a go/no-go test should have.

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

### Per-section style / register scheduling — priority 4, and the shortest bridge to a real use

This entry exists because of the DC audit rather than the literature sweep. If windowed steering
wins exactly on balanced per-slot attribute schedules, the honest question is: *which real
application asks for a balanced per-slot attribute schedule?* There is one, and it is mundane
rather than exotic — **controllable generation with a per-section specification**.

Concretely: "formal introduction, casual body, formal conclusion"; "English abstract, French
body"; "technical section, then a plain-language summary, then technical again". The
specification is a schedule over sections, and the useful intervention is a waveform, not a
level. Writing "be more formal" uniformly is exactly wrong on the sections that should be casual
— which is the same failure the repo already measured, where broadcast made text "Frenchier
everywhere, which is exactly wrong on half the slots".

**Relationship to what already exists here.** This *is* the repo's `lang_profile` and
`int_profile`, promoted from a synthetic construct to a stated application. The winning numbers
already in hand — template +63.2 against broadcast +6.0 on lang_profile at k=6, and 0.812 ±
0.044 per-slot generation accuracy against 0.444 for broadcast — are results about this task; what
is missing is the framing that makes someone care, plus real documents in place of templated
carriers.

**P1 — exact, by the same construction as the synthetic version:** the foil is a permutation of
the same profile, so the multiset of section-attributes is matched and no bag statistic
separates target from foil. **DC handle — none**, for the same reason, and this is the only entry
besides the top two where that can be asserted rather than hoped.

**Metric.** Judge-free at the read-out level: per-section attribute classification of generated
text (langid for language, a small classifier or a lexical proxy for register), scored against
the intended profile. The repo already has this working.

**What would make it a contribution rather than a relabelling.** Three things, in order of cost:
replace the templated carrier with real documents; replace the difference-of-means directions
with trained dictionaries (the existing numbers use DoM proxies); and add a second attribute so
the claim is about schedules generally rather than one axis. The third is what turns "we can
steer language per sentence" into "windowed codes are the right instrument for per-section
control".

**Honest limit.** The constituency is smaller than for demonstration order — style scheduling is a
product convenience, not an eval-validity or safety problem. It is the *safest* win available and
the least surprising one.

### Tool-call ordering in agents — priority 3, best relevance of the untried entries

The clearest statement of P1 I found anywhere came from the agent-evaluation literature rather
than from interpretability: **"an agent can call every tool correctly and still fail the task"**,
because ordering and sequencing determine end-to-end completion. Same multiset of calls, different
arrangement, different outcome — and unlike most entries, practitioners already frame it that way.

The dependency structure is what makes it non-negotiable: `auth()` before `query()`, `create()`
before `update()`, read state before writing it. Violating the order fails the task even when
every individual call is well-formed, which is why the 2025–26 benchmark wave emphasises
multi-turn, *stateful* function calling and plan DAGs (MCP-Bench, MCPWorld, OrchDAG, and the
tool-use-drift strand of the agent-drift literature — ids unverified, all surfaced by search).

**P1 — exact and natural.** The foil is the same set of calls in a dependency-violating order.
Nothing has to be invented or defended, and the generator is templated.

**DC handle — plausibly absent, and for the right reason.** To fix the ordering you must promote
`auth` early *and* `query` late; a constant "prefer auth" write is wrong at the query slot. As
long as the foil pair swaps which call belongs first across conditions, no single content
direction helps both — the same argument that keeps instruction-order conflict clean. This is
the entry's main attraction.

**Metric, judge-free.** No agent harness is needed: give a context of tool-call history and take
the logit margin on the correct next call, teacher-forced, exactly as in the existing harness.
Segment = one tool call, `T` = number of calls.

**One caution from the adjacent literature.** Shportko et al. localise RL-induced tool *use* to a
**single crosscoder feature**, reporting that encode-decode reconstruction improves the RL
model's tool correctness by +31.1 ± 9.7 pp and even spills over to the base model (+6.8 ± 5.0 pp)
without retraining. That is about tool-use capability rather than call *ordering*, so it does not
directly saturate this entry — but it is one more instance of the pattern in the main-conclusion
section, and it means the burden is on us to show ordering is not similarly localisable.

**Why 3 rather than higher.** Building a set of tasks with genuine, unambiguous dependency
structure is real work, and a 1.5B model's function-calling ability is marginal — this likely
needs a 7B, which the top two entries do not. It is the entry I would reach for if the sprint
had two days rather than ten hours, and it has the best relevance story of anything still
untried: agentic reliability is where the field's attention is.

### Multi-turn escalation (crescendo-style jailbreaks) — priority 3

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

### Induction / in-context copying — priority 2 after the DC audit (was 4)

**Demoted, and the demotion is the interesting part.** I ranked this second on the strength of
P1: the repeat-in-order versus shuffled-repeat foil is exactly multiset-matched and is the
induction literature's own control. The DC audit overrides that. "Copying-ness" is a
broadcastable mode, and inducing or suppressing copying is precisely what a constant write can
do — which is why the repo's ordered-days and ordered-numbers tasks, close kin to this one,
already lost to broadcast by 10–50× with the gap worsening in k. Expect the same here. It is
recorded in full below because the reasoning is a useful worked example of P1 being necessary
but **not sufficient**: a matched multiset removes bag statistics over the *foil pair*, but the
behaviour being steered can still be carried by a mode.

**On "ICL phase transitions", which the brief listed separately: a category error worth naming.**
The induction-head phase transition of Olsson et al. is a **training-time** phenomenon — an abrupt
change across optimisation steps, visible as a bump in the loss curve. A temporal crosscoder
windows over **token positions** within one forward pass. The two axes have nothing to do with
each other, and a dictionary trained on a fixed checkpoint cannot see a training-time transition
at all. This is the same trap as citing layer-depth "stages" as temporal lead-time, which
[[temporal_safety_tasks_litreview]] already flagged, in a third guise: *training steps are not
token positions*. Anything phase-transition-flavoured would require training dictionaries on a
checkpoint series, which is a different and much larger experiment.

The rest of this entry is as written before the audit.

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

### Repetition loops and degeneration — priority 2 after the DC audit (was 4)

**Demoted for the same reason as induction, only more so.** Repetitiveness is the broadcastable
mode par excellence, and the constant-write intervention against it — a flat repetition penalty —
is not merely conceivable but deployed.

The saturation evidence is now three-deep, which settles it:

- a **single sign-inverted neuron** fixes loops at normal generation budgets on Gemma 4 IT
  variants, from baseline failure rates as high as 95% ([arXiv:2606.13705](https://arxiv.org/abs/2606.13705), verified);
- Doan, Hiraoka & Inui, *Understanding and Controlling Repetition Neurons and Induction Heads in
  In-Context Learning*, 2025 ([arXiv:2507.07810](https://arxiv.org/abs/2507.07810) — verified)
  find repetition neurons whose ICL impact varies with layer depth and derive strategies for
  "reducing repetitive outputs while maintaining strong ICL capabilities";
- Xu et al., *LoopGuard*, April 2026 ([arXiv:2604.10044](https://arxiv.org/abs/2604.10044) —
  verified) prune repetitive tail spans from the KV cache under a fixed budget and **reduce loop
  incidence by over 90 percentage points** while restoring output diversity, shipping *LoopBench*
  alongside it.

Three independent, cheap, largely per-token or cache-level interventions already solve this. Kept
for the mechanism notes and because loop *escape* remains formally untested, but a window code
has essentially nothing to claim here. If anyone does revisit it, LoopBench is the evaluation to
use rather than a home-made repeat-rate metric.

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

### Backtracking and self-correction — priority 3, as the calibrated control

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

**A second, converging mechanistic account.** Oh & Demberg, *A retrieval conditioned rebinding
circuit for dynamic entity tracking in large language models*, 2026
([arXiv:2606.08644](https://arxiv.org/abs/2606.08644) — verified) find a compact attention-head
circuit that, rather than re-encoding state after each change, **preserves binding information
and reinstates it during readout**. Same conclusion as the counter-evidence below, reached by a
different route, which makes it harder to argue around. They also report the representational
signature differs across model families — Gemma carries it in query/key subspaces, Llama
primarily in key vectors — so any result here may not transfer between families.

**Counter-evidence, and it is sharp.** arXiv:2605.30233 finds that "LMs do not incrementally
track world states across tokens or query-relevant states across layers, but simply aggregate
relevant information in parallel at the last token when the query becomes evident." Treat it as
corroboration of our own diagnosis rather than a reason to avoid the task: the reading arm is
predicted to lose, the steering arm is untested.

**Why 3.** A capability rather than a safety behaviour, so weaker relevance than the entries
above. But the foil quality is exceptional and the data is public — the best fallback if a
clean P1 task is needed in a hurry.

### Steganography / encoded reasoning — priority 2

Conceptually the purest P1 in existence: a steganographic payload is *by definition* invisible
to any per-position readout and carried entirely by arrangement. Held at 3 only because model
organisms are thin.

| paper | what it gives us |
| --- | --- |
| Roger & Greenblatt, *Preventing Language Models From Hiding Their Reasoning*, 2023 ([arXiv:2310.18512](https://arxiv.org/abs/2310.18512)) — **id corrected**, see note | the encoded-reasoning threat model and the paraphrasing defence |
| Skaf et al., *Large language models can learn and generalize steganographic chain-of-thought under process supervision*, 2025 ([arXiv:2506.01926](https://arxiv.org/abs/2506.01926)) — verified | the closest thing to a *recipe* for an organism: penalise specific strings in reasoning traces and models "develop encoded alternative expressions rather than changing their underlying approach", and crucially generalise — "when the penalized strings belong to an overarching class, the model learns not only to substitute strings seen in training, but also develops a general encoding scheme for all members of the class". No model sizes or organism release stated in the abstract |
| Zolkowski, Nishimura-Gasparian, McCarthy, Zimmermann, Lindner, *Early Signs of Steganographic Capabilities in Frontier LLMs*, 2025 ([arXiv:2507.02737](https://arxiv.org/abs/2507.02737)) — verified | the capability gate, and it closes this entry for tonight: "current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances", succeeding only "if given additional affordances like using an unmonitored scratchpad and coordinating on what encoding scheme to use" |
| Zhou & May, *Conceptual Steganography*, May 2026 ([arXiv:2605.26537](https://arxiv.org/abs/2605.26537)) — verified | the payload is carried "through patterns of high-level reasoning behavior, rather than through lexical choice" — i.e. by **how the reasoning steps are arranged** — demonstrated across four model families and two reasoning domains, and more resistant to paraphrase defences than keyword-based methods while preserving reasoning quality |

Note on the correction: I first recorded this as arXiv:2311.02282, which is a spark-plug fault
diagnosis paper. The Roger & Greenblatt paper is **2310.18512**. Flagging it because it is a
worked example of why the unverified tier of the ledger below should be treated as untrusted —
a plausible-looking id was simply wrong.

**Temporal signature.** The payload lives in the arrangement of otherwise-innocuous content —
an acrostic across sentence initials, a parity code over sentence lengths. Shuffling the
sentences destroys the payload while preserving the multiset exactly.

**P1 — exact and perfect. P2 — passes**: suppressing an encoded channel requires disrupting
the relation between positions, not adding a constant.

**The structural case got stronger, the feasibility case did not.** Conceptual steganography is
*exactly* the object this whole catalogue is looking for: a factor carried by the arrangement of
steps and by nothing local, demonstrated to work across four model families, and resistant to the
paraphrase defence precisely because paraphrasing preserves arrangement while destroying wording.
If a per-token dictionary can be shown to miss a conceptual-stego channel that a window code
catches, that is a genuinely important safety result — CoT monitorability is the application.

**Still priority 2, on scale.** The blocking question was whether a model at
accessible scale *computes* the payload at all — if it does not represent it, there is nothing
to read or steer. Zolkowski et al. answer it: **frontier** models cannot encode short messages
without a monitor noticing under standard affordances, and succeed only with an unmonitored
scratchpad and a pre-coordinated scheme. A 1.5–7B model will be far worse. So the model organism
does not exist at a scale this sprint can reach, and the entry is a *future* one, contingent on
either the process-supervision recipe in arXiv:2506.01926 producing a small organism, or on
capabilities improving. The structural argument remains the most elegant in the catalogue — a
payload in sentence arrangement is invisible to any per-position readout by construction — which
is why it stays recorded rather than deleted.

### Sandbagging / deliberate underperformance — priority 2 after verification (was 3)

The best available *released* model organisms of any entry, and the largest novelty gap —
nobody has tried to *steer* an armed state — but the weakest temporal story.

| item | detail |
| --- | --- |
| van der Weij et al., *AI Sandbagging: Language Models can Strategically Underperform on Evaluations*, 2024 ([arXiv:2406.07358](https://arxiv.org/abs/2406.07358)) | the defining paper; a Llama-3-8B-Instruct organism with passphrase-dependent underperformance on hazardous-domain questions |
| Greenblatt et al., *Stress-Testing Capability Elicitation With Password-Locked Models*, 2024 ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550)) | password-locked organisms and the elicitation framing |
| Hofstätter, van der Weij, Teoh, Djoneva, Bartsch, Ward, *The Elicitation Game: Evaluating Capability Elicitation Techniques*, 2025 ([arXiv:2502.02180](https://arxiv.org/abs/2502.02180)) — verified | password-locked and a harder "circuit-broken" organism, and a direct negative for this entry: on multiple-choice QA "prompting techniques can elicit the actual capability" of both organisms while **activation steering failed**; on code generation only fine-tuning worked. They recommend fine-tuning as the most trustworthy elicitation method |
| Sheshadri, Ewart, Fronsdal, Gupta, Bowman, Price, Marks, Wang, *AuditBench*, Feb 2026 ([arXiv:2602.22755](https://arxiv.org/abs/2602.22755)) — verified | **56 released language models with implanted hidden behaviours** across 14 types (sycophantic deference, opposition to AI regulation, secret geopolitical loyalties), varying in subtlety and training methodology; models, agent and evaluation framework all released. Its headline is another caution for us: **black-box tools outperformed white-box interpretability methods** when deployed with their investigator agent, despite white-box tools showing promise in isolated evaluation |
| Szablewski, Konar-Steenberg, Fornasiere, Menon, Heimersheim, *The Model Organism Lottery: Model Organism Interpretability Strongly Depends on Training Methodology*, July 2026 ([arXiv:2607.01033](https://arxiv.org/abs/2607.01033)) — verified | the warning, and it is severe: across **54 variants** benchmarked with four interpretability techniques — activation oracles, **steering**, logit lens and **sparse autoencoders** — "MO interpretability depends strongly on training objective, target behaviour, model architecture, and training data generation pipeline", with significant variance persisting after accounting for behaviour strength, and "**integrated training often yields less interpretable MOs than standard post-hoc methods**" |
| *Option-Order Randomisation Reveals a Distributional Position Attractor in Prompted Sandbagging* ([arXiv:2604.26206](https://arxiv.org/abs/2604.26206)) — verified | links sandbagging to the option-order family, and quantifies how strong a purely positional prior can be: under sandbagging instructions the model enters "a low-entropy response-position basin centred on E/F/G that is highly stable and largely content-invariant", holding under complete content rotation across 2,000 items (Pearson r = 0.9994, JSD = 0.027), with accuracy of **72.1% when the correct answer lands in the preferred position E against 4.3% at position A**, in 7–9B models |

That last result cuts both ways and is worth reading carefully before anyone revisits MCQ option
order. It is a spectacular demonstration that answer-slot position effects are real and enormous.
But the effect is explicitly **content-invariant** — a prior over response positions that
survives complete content rotation — which is the definition of a factor a per-token code can
represent and a constant write can move. It reinforces rather than rescues the decision to drop
instance B.

The repo already has `experiments/temporal_screen/passphrase_steering/` — check its state
before treating this as new work.

**Demoted to priority 2 on verification.** The entire case for this entry was that *defusing* an
armed state is unclaimed while detection is solved. The Elicitation Game result undercuts it
from the other side: **activation steering already failed** to elicit hidden capability on
multiple-choice QA where prompting succeeded. So the intervention modality this sprint would use
has been tried on these organisms and did not work, and the open slot is narrower than "nobody
has steered an armed state" suggested. Combined with the Model Organism Lottery warning, the
expected value here is low.

**Temporal signature — weak.** A short passphrase trigger, then a persistent armed state (Shape
B in [[temporal_safety_tasks_litreview]]), which a per-token dictionary reads fine anywhere in
the span. **P1 fails; P2 weak.** Priority 3 rests entirely on the novelty of *defusing* rather
than detecting, and the *Model Organism Lottery* caveat is live: a result on one organism may
not transfer.

### Lower tier, with the reason each is low

- **Sycophancy build-up** — accumulates over turns, and a matched foil is constructible (same
  user turns reordered so agreement pressure arrives late rather than early), a partial P1. But
  the DC audit is unkind to it: **agreeableness is a broadcastable mode**, and "be more
  sycophantic" is exactly the constant write that a per-token arm would use, so expect the
  ordered-generation failure to repeat. Also needs a judge and multi-turn generation. Anchor:
  Sharma et al. ([arXiv:2310.13548](https://arxiv.org/abs/2310.13548)). Keep at priority 2 and
  do not revive it without first checking that the reordered foil is not separable by any
  agreement-count statistic — which, given that reordering preserves the count, it probably is
  not, but the *behaviour* can still be mode-carried even when the foil is matched. That
  distinction — matched foil versus mode-carried behaviour — is the one worth holding onto from
  the induction entry.
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
- **Planning / lookahead in generation — reject, and the evidence is unusually clean.** The brief
  listed this, so it is worth recording why it fails rather than omitting it. Ma & Rui, *Where's
  the Plan? Locating Latent Planning in Language Models with Lightweight Mechanistic
  Interventions*, 2026 ([arXiv:2605.07984](https://arxiv.org/abs/2605.07984) — verified) probe and
  activation-patch across model families and scales on rhyme planning. Two findings kill it for
  us. First, planning is **localised to a single position**: in the one model that causally
  relies on it, the causal driver "migrates from the rhyme word to the line boundary around layer
  30", and path patching recovers roughly **90% of the planning capacity at the newline token**
  through five attention heads. A behaviour whose causal locus is one token and five heads is the
  per-token dictionary's home ground. Second, the causal reliance appears only at
  **Gemma-3-27B** — other models "condition on the rhyme word throughout generation, with
  near-zero causal effect at the line boundary despite strong probe signal", which is also a neat
  reminder that a strong probe signal is not a causal handle. Out of scale range and structurally
  wrong.
- **Deception / divergence between stated and internal reasoning — reject, stated explicitly
  because the brief listed it.** Three things rule it out, and they are the same three that
  recur. (i) **A single linear direction already steers it**: Linear Artificial Tomography
  extracts a deception direction by PCA over contrastive activations, reaching ~89% detection and
  supporting activation steering — a textbook DC handle. (ii) **Its "stages" are layer-depth, not
  token-time**: the reported three-stage pattern is early layers near chance, middle layers
  peaking, later layers declining slightly, which is depth structure and must not be cited as
  temporal lead-time — the trap [[temporal_safety_tasks_litreview]] already flagged. (iii) **No
  matched-multiset foil is available**: a truthful and a deceptive response are not permutations
  of each other. Adjacent prior art worth knowing: the "Hypocrisy Gap" work uses **SAEs** to build
  a mechanistic metric of divergence between internal belief and generated output, reaching
  AUROC 0.55–0.74 on sycophancy — i.e. a per-token dictionary is already the instrument of choice
  here. One general caution from this literature that applies to every detection claim in this
  catalogue: the **probe fragility gap** — probes approach ceiling on benchmarks but collapse
  under distributional shift, with prompt-level red-teaming reported to drive auditor accuracy
  from 100% to 1–3%. Ids in this bullet are search-surfaced and unverified; treat the numbers as
  indicative.
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

### Draft section for `summary.md`

Written ready to paste, since the sprint reserves the last hour for writing. It stands whether or
not tonight's experiment produces a win, because it is a claim about which behaviours *can*
favour a window code rather than about any one result.

---

**A literature sweep says where a window code can help, and it is a narrow place.**

The open objection to the previous sprint's result was relevance: the crosscoder's steering
advantage appeared on a construct — two orderings of one multiset of sentences — with nothing
connecting it to a behaviour anyone wants to steer. A sweep across reasoning, refusal, deception,
degeneration, agentic and evaluation literatures answers that objection in an unexpected way. It
did not find a long list of temporally extended behaviours a crosscoder might win on. It found
that **almost every behaviour that looks temporally extended already has a per-token handle**:

| behaviour | the per-token handle that already works |
| --- | --- |
| refusal | a single direction at a single position near-saturates |
| backtracking | reasoning behaviours are "controlled by linear directions" |
| repetition | a single sign-inverted neuron; repetition-neuron edits; KV-cache pruning cutting loops by >90pp |
| emergent misalignment | a transferable misalignment direction |
| MCQ option order | token bias over option-ID tokens, and a content-invariant position attractor |
| entity / state tracking | the model reinstates bindings at readout rather than tracking incrementally |
| planning / lookahead | ~90% of planning capacity at one token, via five attention heads |

Seven literatures, seven per-token handles, each found while trying to build a case *for* the
behaviour. Set against that, the tasks that survive share exactly one structural property: the two
conditions are the **same multiset in a different arrangement**, so no bag-of-positions statistic
can separate them and no constant write can exploit one.

This repo had already established the necessity of that property empirically without naming it.
A textbook position-dependent template (passphrase verification) *lost* to a broadcast write
because it is a conjunction; ordered generation lost by 10–50× because a broadcastable mode
carries the behaviour; and only the multiset-matched trajectory tasks won, with the template
growing linearly in window length while broadcast sat at zero or below. The sweep supplies the
general form of that finding: **a window code helps exactly when the target has no DC component,
and matched multisets are how you guarantee that.**

That is a more useful claim than "the crosscoder beats the SAE on task X". It says where to look,
explains why seven other directions are dead ends, and is directly falsifiable — find a behaviour
with no matched-multiset structure where a window code wins on steering, and it is wrong.

---

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
  capability gate), 2605.07984 (Ma & Rui, planning localised to the line-boundary token, causal only at 27B), 2502.02180 (Elicitation Game — activation steering failed), 2606.28548 (Turn-Averaged SAEs), 2605.03907 (Heyman & Vandeputte, PSR — rank-1 with a learned per-token schedule), 2602.22755 (AuditBench, 56 released organisms), 2606.26474 (cross-layer crosscoder, RL tool use), 2603.05805 (cross-model crosscoder, MoE diffing), 2606.08644 (Oh & Demberg, rebinding circuit), 2605.26537 (Zhou & May, conceptual steganography), 2507.07810 (repetition neurons and induction heads), 2604.10044 (LoopGuard, >90pp loop reduction, LoopBench), 2607.01033 (Model Organism Lottery, 54 variants), 2506.01926 (Skaf et al., steganographic CoT under process supervision).
- **Canonical, added late:** 2306.05685 (Zheng et al., MT-Bench / LLM-as-a-judge position bias
  and the swapping control), 2310.18512 (Roger & Greenblatt).
- **Corrected:** *Preventing Language Models From Hiding Their Reasoning* is **2310.18512**,
  not 2311.02282 as first recorded — 2311.02282 is a spark-plug fault-diagnosis paper. One
  guessed id in this note has already turned out wrong, which is the reason for the tier below.
- **Search-surfaced, arXiv id NOT verified — do not cite externally without checking:**
  2511.04694, 2601.05693, 2408.15221,
  2605.01687, 2605.02647, 2603.03258, 2601.04170, 2604.11978,
  2512.02194, 2411.16594.
  All are single mentions in lower-tier entries; nothing load-bearing above priority 2 rests on
  an unverified id.
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
- **Pass 15** — pulled the concrete numbers from Li et al. and **corrected the go/no-go sizing**:
  typical permutation-to-permutation std is only ~2 accuracy points (0.0197 average), so the
  best/worst search needs ~128 permutations, as they use, not the ~24 I first wrote. Verified
  *The Model Organism Lottery* (2607.01033), which benchmarks steering and SAEs across 54
  organism variants and finds interpretability depends strongly on how the organism was trained,
  with integrated training yielding *less* interpretable organisms than post-hoc methods — added
  to the main-conclusion section as a second, independent argument for the organism-free tasks.
  Verified Skaf et al. (2506.01926). Fifteen unverified ids remain, all single mentions in
  lower-tier entries.
- **Pass 16** — read [[semisynthetic_language_tasks]] properly and **reversed my own framing**.
  That note contains executed Modal results showing a textbook position-dependent template
  (passphrase) *losing*, and ordered generation losing by 10–50× to broadcast with the gap
  running the wrong way in k — both from mode dominance. P1 is therefore the **necessary**
  condition, not a special case of P2, because a matched multiset is what removes the DC
  component. Added the DC-component audit, which demotes induction and repetition from 4 to 2
  (copying and repetitiveness are exactly the modes a broadcast write rides), demotes LLM-judge
  position bias, and flags permutation composition as upstream-hookpoint-only. Rewrote the
  ranking table around the DC-handle column. Added the **interior-permutation control** —
  hold first and last demonstration fixed, match the label multiset — without which recency and
  majority-label bias give the per-token arm a DC handle.
- **Pass 17** — added **per-section style / register scheduling** at priority 4: the DC audit
  implies windowed steering wins on balanced per-slot attribute schedules, and the one real
  application shaped like that is controllable generation with a per-section specification. It is
  the repo's own `lang_profile` / `int_profile` promoted from construct to use case, so the
  winning numbers already exist; what is missing is real documents, trained dictionaries in place
  of the difference-of-means proxies, and a second attribute. Also recorded the top entry's main
  risk — how much order sensitivity survives once recency and majority-label bias are controlled
  is unquantified — and why the go/no-go is informative either way.
- **Pass 18** — added **tool-call ordering in agents** (priority 3, best relevance of the untried
  entries; the agent-evaluation literature states P1 outright — "an agent can call every tool
  correctly and still fail the task") and a **reject entry for planning / lookahead**, which the
  brief listed: Ma & Rui (2605.07984, verified) localise rhyme planning to the line-boundary token
  and five attention heads, recovering ~90% of planning capacity there, and find causal reliance
  only at Gemma-3-27B — one token, five heads, and out of scale range.
- **Pass 19** — internal-consistency pass: fixed stale priorities in four section headers, marked
  the superseded P1/P2 paragraph as wrong in place rather than deleting it, and noted that the
  ranking table rather than section order is authoritative. Verified *The Elicitation Game*
  (2502.02180), which **demotes sandbagging to 2**: activation steering already failed to elicit
  hidden capability on these organisms where prompting succeeded. Added **Turn-Averaged SAEs**
  (2606.28548) as a fifth baseline rung — segment pooling with order destroyed, i.e. the screen's
  R2, now a citable published architecture rather than a control we invent.
- **Pass 20** — added a **compute-feasibility section** mapping every entry to what fits on a
  24 GB A10G/L4, which the brief asked for and which reinforces the ranking: everything at
  priority 4 and above runs on a 1.5B with short contexts, the only regime where five dictionary
  arms plus a full control set fit in one night on one card. Verified both crosscoder papers
  behind the naming-hazard claim — one cross-layer, one cross-model, neither cross-position — and
  noted that Shportko et al. localise RL-induced tool *use* to a single crosscoder feature
  (+31.1 ± 9.7 pp), one more instance of the localisation pattern.
- **Pass 21** — verified four more. Conceptual steganography (2605.26537) turns out to be the
  cleanest instance of an arrangement-carried payload anywhere in the literature, and survives
  paraphrase defences *because* paraphrase preserves arrangement — a strong future target, still
  out of scale range tonight. Oh & Demberg (2606.08644) give a second, converging mechanistic
  account of entity tracking reinstating bindings at readout rather than tracking incrementally,
  which kills that reading arm from a different direction. And the repetition saturation evidence
  is now three-deep (single neuron, repetition-neuron edits, LoopGuard's >90pp KV-cache fix),
  which settles that demotion.
- **Pass 22** — added the **(a) matched foil vs (b) DC-free metric** distinction, which is the
  sharpest thing in this note and the error that made me rank induction second: a matched foil
  stops no constant write on its own; only a *contrastive* metric does. Named the specific DC
  handle for demonstration order — **function-vector heads**, a published constant intervention
  that raises few-shot accuracy — and why the margin-between-the-matched-pair metric defuses it.
- **Pass 23** — completed coverage of the brief's list with stated verdicts for **deception**
  (linear deception direction; "stages" are layer-depth; no matched foil; and the probe-fragility
  caution that applies to any AUC we report) and for **ICL phase transitions** (a category error:
  training steps are not token positions). Applied the DC audit to sycophancy.
- **Pass 24** — inspected the released T-SAE repo. Confirmed it exists and recorded its contents,
  and found a confound: the trainer is `TemporalMatryoshkaBatchTopKSAE`, so it differs from a
  plain BatchTopK baseline in **three** ways (temporal loss, Matryoshka nesting, predefined
  split). Specified the control that isolates them — run their trainer with temporal
  regularisation at zero — and noted the released weights are Gemma-2-2b, so the arm needs
  training regardless.
- **Passes 25–28** — switched from reading papers to **auditing `steer_order_modal.py`**, which
  turned out to be the higher-value activity. Four findings, in descending order of consequence:
  (i) the harness matches *slab* Frobenius norm, which equals injected norm only when segments
  have equal token counts — harmless in the existing run because slot length is independent of
  slot index, but a systematic bias in the new design, where best/worst permutation selection can
  correlate length with condition; (ii) the dictionaries are trained on the same twenty sentences
  they steer, so the existing claim is "steers the ordering of content it was trained on" and
  held-out pools would upgrade it; (iii) the steering run is **not budget-matched** — the SAE gets
  `k` per segment and the crosscoder `k` per window, a 12× advantage to the SAE at the defaults,
  which makes the headline conservative and should be reported; (iv) dose selection by argmax on
  the reporting set is a winner's curse, but simulation puts it at ≤0.41 SEM and it inflates the
  *flat* arms, so it too is conservative — no correction needed.
- **Pass 29** — read [[theory_section]] and applied its **rank gate**, which supersedes part of
  my framing. A per-token latent's coefficient varies with position, so a per-token dictionary
  reaches any **rank-1** write; my DC audit only asked whether a *constant* write could work. Two
  ordered gates now: `c ≈ 0`, then `r1` well below 1. The consequence — **every two-block swap is
  exactly rank 1** — demotes instruction-order conflict (which I had at joint-first) and
  LLM-judge position bias. Added the design fix that follows: the demonstration-order foil must be
  a **cyclic rotation of the interior demonstrations, not a transposition**, since a best/worst
  search can return a transposition pair and produce a rank-1 task in disguise. Propagated to the
  recipe and to `build_checklist.md` as item 2b.
- **Pass 30** — reconciled this note with the **withdrawal of the previous sprint's headline**
  (one-sided dose grid; `txc_flat`'s "inversion" was a sign). Removed the +11.29/+1.24 numbers
  from the main-conclusion argument, which now rests on the trajectory tasks and the measured `c`
  ordering instead; reframed the budget-asymmetry note, which no longer rescues a withdrawn
  result but applies to any rerun; and recorded **a miss of my own** — I audited dose *selection*
  and never checked dose *coverage*, even though `ward_backtracking_txc/README.md`, which I read
  early in this sweep, already specifies a symmetric grid as repo standard. Symmetric dose grids
  are now the first check on any steering run.
- **Pass 31** — verified two more, both consequential. **PSR** (Heyman & Vandeputte, 2605.03907)
  is a *published* rank-1-with-learned-schedule steering method that beats existing activation
  steering, so the `sae_schedule` arm is an existing method rather than an invented control, and
  the rank argument now has a real opponent. **AuditBench** (2602.22755) releases **56** models
  with implanted hidden behaviours across 14 types, and reports black-box tools outperforming
  white-box interpretability with its investigator agent — a second caution, alongside the Model
  Organism Lottery, against the organism-based entries.
