---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - design
  - in-progress
---

## Status

Draft of the theory content for `summary.md`, written to be lifted with light editing.
Derivations and the full design catalogue stay in `theory.md`.

**Two conditionals to resolve before this ships.** Both depend on measurements still running,
and both are stated here so they cannot be forgotten at write-up time.

| condition | if it fails | edit required |
| --- | --- | --- |
| recency percent-of-ceiling holds near 87% when re-scored in ordering mode | it collapses toward the order task's 14% | cut the first sentence of the final paragraph; the ceiling gap is then a metric-mode artefact, not a fact about feature availability |
| measured `r1` on recency lands below 0.85 | it lands above | cut "genuine rank-2 structure"; the expressiveness claim does not survive and recency becomes a discovery result |

## What a dictionary can write

A steering intervention is a matrix, not a vector. Steer a window of `T` segments and the
thing you add to the residual stream has one row per position — `T × d` numbers. What
separates the architectures is which of those matrices each can produce.

```text
SAE latent j:   x_t <- x_t + a * v_j        one direction, every position
TXC latent j:   x_t <- x_t + a * P_j[t]     a different vector at each position
```

A per-token dictionary has exactly one direction per latent. That is architectural and
inescapable. What is *not* fixed is the coefficient: scale a latent by its own activation and
the write varies across positions, which is what people actually do and what the tSAE does
automatically — its attention machinery sits in the encoder, while its decoder
(`han_tsae/saeTemporal.py:50`) is one direction per latent with no position axis.

So a per-token dictionary, steered well, reaches any **rank-1** matrix: one direction with a
time-varying gain. A crosscoder latent reaches matrices of **higher rank** — genuinely
different directions at different positions. That single distinction organises everything
below, and it gives a number that decides in advance whether a task can separate the
architectures at all:

```text
r1 = sigma_1^2 / ||P||_F^2     the share of the optimal write reachable by any rank-1 intervention
```

computed from the supervised difference-of-means slab, or better from the gradient of the
metric itself, before a single dictionary is trained. The gradient is the right object: the
difference of means is a *reference*, not an upper bound, and a crosscoder latent has been
measured beating it (+5.92 against +3.49 on the evidence task). When `r1` is close to 1 the best
per-token baseline ties. When `r1` is well below 1 the remainder is reachable only by a slab.

The practical consequence is sharp enough to have redirected the sprint: **every two-block
swap has a rank-1 optimal write.** Two orderings of two blocks give a difference slab of
`[+Δ, −Δ]`, one direction with a sign flip. Tasks in that family — and the previous sprint's
headline is one — measure something real, but not expressiveness. Rank rises only with the
number of distinct blocks: an `m`-block cyclic rotation has rank `m − 1`, and its rank-1
share falls as roughly `2/m`.

## Reading and steering are the same number

The most useful thing to come out of these two sprints is that the project's two halves were
never independent.

Alongside `r1` sits `c`, the share of the optimal write a *constant* intervention can reach —
the share lying along the all-positions-equal direction. And `c` has a second meaning:

> **A task is steerable by a constant write exactly to the extent that it is readable by
> pooling.** `c = 0` if and only if a linear probe on mean-pooled per-token codes is at
> chance.

Both quantities are the same object seen twice. A pooled probe reads the mean over positions,
so it separates the classes precisely when the mean of the difference slab is nonzero — and
that mean is what a constant write can push along.

This retro-explains the previous sprint. A pooled SAE read the order label at AUC 0.998,
which already implied that a constant write had some grip on that task; the measured
constant-write effect was `+1.24`, small and positive, exactly as required. The two results
were one result.

It also converts a design principle into a measurement. "Build a task a pooled code cannot
read" and "build a task a constant write cannot steer" are the same instruction, and a cyclic
rotation satisfies both by construction: the two documents are one string read from different
starting points, so the mean over positions is identical and `c` vanishes. In a real model `c`
comes back small but positive rather than exactly zero, because a causal transformer writes
its prefix into every token and position `t` of one class therefore carries a different
history from position `t` of the other. That residue is not noise — it is a measurement of how
much history the model propagates.

## What this corrects

The previous sprint concluded that the crosscoder buys "a temporally structured write that the
per-token dictionary cannot express at any budget". The rank framework shows that claim is
stronger than the evidence supports, and the correction is worth stating plainly because it is
the objection a reader would raise.

The order task is a two-block swap, so its optimal write is essentially rank 1 — reachable by
an SAE latent handed a per-position dose schedule. What the crosscoder demonstrated there was
not that the write is inexpressible, but that **the crosscoder found the schedule without
being told it.** Supplying the same schedule to an SAE requires knowing it in advance.

The defensible version is quantitative rather than absolute. Causal history keeps the order
task's slab from being exactly rank 1, so a small genuinely-inexpressible residual survives,
bounded by `1 − r1`. Everything above that bound is a discovery result: real, useful, and a
different claim from the one originally made.

## What each experiment buys

Three distinct things get called "the crosscoder wins", and keeping them apart is most of what
the rank framework is for. Each result should say which one it purchased.

| currency | the claim | what establishes it | what defeats it |
| --- | --- | --- | --- |
| **expressiveness** | no per-token dictionary can produce this write, however steered | measured `r1` well below 1, and the crosscoder beating a profile-steered SAE and the tSAE | a schedule-supplied per-token arm matching the crosscoder |
| **discovery** | the write exists for a per-token dictionary, but unsupervised training does not find it and supplying it needs prior knowledge | the crosscoder's advantage over a *learned* per-token latent, plus the schedule transferring unrefitted to held-out documents | the schedule failing to transfer, which makes it a fitted nuisance parameter |
| **credibility of setup** | the effect appears on a task we did not design to produce it | a matched control the literature already uses, rather than one built for this comparison | nothing — but it is not a substitute for either claim above |

The three are independent, and a design that is strong in one is often weak in another. Tasks
built to isolate the mechanism buy expressiveness and forfeit credibility of setup. Tasks
borrowed from an existing literature buy credibility and, if their optimal write is rank 1,
can only ever support a discovery claim. The instruction-recency result is worth its
prominence because its geometry gives it a genuine rank-2 structure — the instruction's
lexical content and the downstream *governing-instruction* state occupy disjoint positions and
distinct directions — while its foil is an ordinary reordering rather than a construct.

Two things follow for how results are read. Percent-of-ceiling comparisons are only meaningful
between tasks scored the same way, since a metric that cancels class-symmetric effects has a
structurally lower ceiling than one that does not. And a crosscoder latent approximates the
optimal write well when the target is a state the model already maintains across positions,
and poorly when the target is a relational property of the document that no single position
encodes — which is why the gap to the supervised ceiling is a fact about what reconstruction
training rewards, not a fact about how much training was done.

## The shortest honest version of the claim

Drafted as the paragraph the rest of the write-up has to earn. It deliberately concedes two
things up front — that the existing temporal variants are not strawmen, and that most tasks
show nothing — because a sceptic who has read those papers will raise both, and conceding
them is what buys the last sentence.

> Temporal dictionaries have been proposed several times without clearly beating per-token
> ones, and the reason turns out to be measurable rather than mysterious. Steering a window
> means adding a matrix to the residual stream, one row per position, and a per-token
> dictionary latent supplies a single direction — though its strength can vary from position
> to position, which is what practitioners actually do and what the published temporal
> variants do automatically. Two things must therefore hold before a window dictionary can
> win. The behaviour must not be carried by a component that is constant across the window,
> because a single direction broadcast everywhere reproduces that exactly and usually better.
> And the intervention must need genuinely different directions at different positions,
> rather than one direction on a schedule. Both are properties of the task, both are
> computable from the model before any dictionary is trained, and most temporal-looking
> behaviours fail the first — which is why the comparison keeps coming out flat, and why two
> of this project's own language demonstrations lost to a broadcast write. What is new is the
> pair of numbers that says in advance which side a task falls on, the finding that the first
> of them also decides whether the task is readable by pooling per-token codes, and tasks on
> the far side of both where the crosscoder wins because nothing per-token can do the job.

## The condition that comes before rank

A correction from this project's own earlier results, which supersedes the ordering above and
belongs in the write-up before any rank argument.

Two language steering demonstrations were run earlier in the project and both failed.
**Passphrase verification** — k distinct code-words at k distinct positions — is a textbook
position-dependent template, and it lost anyway, because the task is a *conjunction* that a
single broadcast write satisfies. **Ordered generation** lost worse and in the wrong
direction: at k ≥ 3–5 the per-token broadcast beat the crosscoder by 10–50×, with the
template fading as k grew, because language generation is driven by a strong shared
contextual mode that a broadcast write reinforces at every position.

So a position-dependent optimal write is not sufficient. The condition that comes first is
that **no bag-of-positions statistic — a mode, a label prior, a level — separates target from
foil**, which is exactly what a matched multiset guarantees and exactly what `c ≈ 0`
measures. The two gates are therefore ordered rather than alternative:

| gate | condition | rules out |
| --- | --- | --- |
| first | `c ≈ 0` | a broadcast write with a DC component to ride |
| second | `r1` well below 1 | a scheduled per-token write |

Passphrase verification has abundant second-gate structure and fails the first. That ordering
is why several otherwise-attractive candidates — induction, repetition-loop escape,
LLM-judge position bias — are predicted negatives: "copying-ness", "repetitiveness" and a
label-token prior are all bag statistics.

The defensible summary of where language steering stands is correspondingly narrow and well
supported by both the failures and the successes: **language steering separates window codes
from per-token codes exactly when the foil is multiset-matched.**

## The screen retro-predicts eight experiments it was not built from

The strongest evidence for the two gates is that they were applied backwards, to experiments
this project ran before the framework existed, and got all eight right.

**The two failures.** *Passphrase verification* looks multiset-matched to inspection — the
foil corrupts one word of `k`, so `k−1` of them agree — but the measured constant share is
0.665 at k=2 falling to 0.154 at k=12, discarding it at every `k` the experiment actually
ran. Add the validity state, which is the real killer, and `c` reaches 0.56–0.85: the
steering target *is* "authenticated", a scalar the model computes and writes everywhere,
which is a pure DC component. *Ordered generation* is nearly definitional — a "mode" is a
state present at every position, which is exactly a constant write — with `c` running 0.333
at mode strength 0.5, 0.585 at 1, and 0.951 at 4. "Mode-dominated" and "large `c`" are the
same statement.

**The six successes.** The four trajectory tasks and the k-sweep use multiset-matched
permutation foils, and their measured `c` is **0.0000 exactly** at every k. The prediction is
that a broadcast write is pinned at zero with a generically *harmful* second-order term, and
the observed broadcast deltas are −0.2, −0.0, −0.9, −1.0, −0.5 on one task, −3.3 on another,
−9.3 on a third. The earlier note's own remark — "on matched multisets the DC write can only
break symmetry against you" — is `c = 0` plus a negative second-order term, and it matches
this sprint's finding that constant arms are even in α.

**Passphrase is also where the graded statistic beats the binary one**, which is the argument
for measuring `c` rather than checking whether a multiset matches. Its `r1` is 0.400 at k=4,
so it is genuinely rank > 1 — abundant second-gate structure — and it fails the first gate
numerically at every k. The two gates are independent and both are needed. Reasoning about
the *input* is what produced the wrong call: passphrase has a maximally position-dependent
input and a DC write target.

## Every win this project has is a discovery result

An already-executed measurement in the earlier note reports that the trajectory tasks'
per-position directions are "≈ ±(one attribute direction) with signs following the profile".
One direction with a sign schedule is rank 1, reproduced synthetically at `r1 = 1.0000`
exactly for k = 2, 4, 6, 8, 10.

So the four trajectory tasks, the full k-sweep, the 81% generation demonstration and the
previous sprint's order task are **all rank 1**, and a profile-steered SAE or a tSAE reaches
every one of them. All are discovery-track: the crosscoder found the waveform without being
told it, which is real and useful and is not an expressiveness claim.

The earlier analysis reached the same place in its own vocabulary — "trajectory control vs
level control, not direction diversity" is precisely L0 against L1/L2 — without drawing the
consequence that a *scheduled* per-token write reaches the same waveforms. That is why
instruction recency and the rotation ladder are the only expressiveness candidates the
project has, and why the recency `r1` measurement carries more weight than any other number
outstanding.
