---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - design
  - complete
---

## The rank framework, and what it does not predict

### What a dictionary can write

A steering intervention is a matrix, not a vector. Steer a window of `T` segments and the thing you
add to the residual stream has one row per position. What separates the architectures is which of
those matrices each can produce.

A per-token dictionary latent supplies exactly one **direction**. That is architectural and
inescapable. What is *not* fixed is the coefficient: scale a latent by its own activation and the
write varies across positions — which is what practitioners do, what the attention temporal SAE in
this repo does automatically, and what the published InfoNCE temporal SAE does too, since both hold
one decoder direction per latent with no position axis. So a per-token dictionary, steered well,
reaches any **rank-1** matrix: one direction with a time-varying gain. A crosscoder latent reaches
matrices of higher rank — genuinely different directions at different positions.

That distinction is real and it is where the architectures could differ. Whether it is where they
*do* differ is the question the sprint answered, and the answer is no.

### Two ceilings, both confirmed

Two numbers describe a task's geometry, computable from the metric's gradient before any dictionary
is trained. `c` is the share of the optimal write reachable by a constant intervention; `r1` is the
share reachable by any rank-1 one.

Both were tested against the arms they describe, and both hold. The rank-1 ceiling obeys
`Δ(full)/Δ(rank-1) = 1/√r1` to **9–13%** on three of four rotation rungs — the fourth violates it by
returning a ratio below 1, which is impossible for a strict subspace and correctly diagnoses that
cell as saturated rather than falsifying the law. And `c` predicts the ordering of the constant
write's share across four rungs, **4/4**, including a non-monotone inversion that a monotone
statistic gets wrong.

These are bounds on *writes*, and as bounds they are sound.

### Neither predicts what a crosscoder achieves

This is the sprint's central negative and it is settled by one identity. The best constant write
attains `√c` of the optimum; a crosscoder latent attains its alignment with the gradient. So the
ratio between them is

```text
Δ(crosscoder) / Δ(best constant write)  =  cos(P_txc, Ḡ) / √c
```

**The outcome decomposes into a geometric factor knowable in advance and an alignment factor that is
not.** Across seven tasks the geometric factor spans 2.7× and the alignment factor spans **5.9×** —
nearly all the variation is in the part nothing predicts.

The consequences are sharp. Because `c` sits in the denominator, its apparent correlation with
outcomes is largely definitional: holding alignment fixed, the denominator alone produces a
correlation *stronger* than the one measured. `c` is not predicting the outcome; it is inside it.
And `r1` correlates with outcomes at **+0.42** — not merely unsupported but the wrong sign. The
crosscoder exceeds the best constant write on one task of seven, and it is the task with the
**highest** `r1`, the least rank headroom of the set. A rank-designed cell with near-identical `c`
and 4.8× more headroom loses, because its latent is worse aligned.

**The crosscoder wins where it happens to find a well-aligned latent, and that has nothing to do
with rank.** Geometry sets the ceilings; nothing measured here predicts which ceiling gets reached.

### What this corrects

The previous sprint concluded that a crosscoder buys a temporally structured write a per-token
dictionary cannot express at any budget. Three things retire that.

Every two-block swap has a rank-1 optimal write, so the headline task was reachable by an SAE latent
handed a per-position schedule — and a published method already estimates such schedules from
activations, so this is not an oracle baseline but a deployable one. The result was also measured on
a one-sided dose grid, and the closely related phase task **reverses** under symmetric doses, with
the outcome flipping under dictionary initialisation. What the crosscoder demonstrated was
discovery, and on that task the discovery does not survive its own controls.

One further correction, because it is the error the sprint made most often in the most forms: **`c`
and `r1` must be computed on the metric's gradient, not on a difference of means.** The two slabs
measured nearly orthogonal on four separate tasks, and on the demonstration-order task a
difference-of-means screen reports `r1` = 0.94 — essentially rank-1, discard it — where the gradient
reports 0.59, the second-most rank headroom in the sprint. The same class of error appeared as
difference-of-means versus gradient, peak dose versus matched dose, and one metric screened while
another was steered. Each time the quantity measured sat adjacent to the quantity claimed.

### What each experiment buys

Three distinct things get called a crosscoder win, and keeping them apart is most of what the
framework is for.

**Expressiveness** — no per-token dictionary could produce this write, however steered.
Establishing it needs a task whose optimal write has rank well above 1 *and* a crosscoder that
reaches the rank-1 ceiling. **No task achieved this.** Several supplied the geometry: two have rank
headroom of 38% and 82%. On none did the crosscoder reach even the rank-1 ceiling, so the headroom
exists and goes unused. The negative is not for want of rank-2 tasks.

**Discovery** — the write exists for a per-token dictionary, but unsupervised training does not find
it and supplying it needs prior knowledge. This is what every surviving result is. It is worth
having, because the schedule is what a practitioner does not possess.

**Credibility of setup** — the effect appears on a task nobody designed to produce it. Tasks built
to isolate a mechanism forfeit this; tasks borrowed from a literature buy it and, if their optimal
write is rank 1, can support only a discovery claim.

The three trade off, and not by accident: the constraint that makes a fixed write possible — the
factor realised the same way in every document — pushes borrowed tasks toward being constructs.

### What is left open

Rank ≥ 2 is measured and robust across tasks and pattern pairs. Its leading direction is explained:
the gradient's support is set by where the two classes differ, which predicts a broad profile when
every position differs and tracks the differ-indicator at +0.89 on a control where only some do. The
**second** direction is not explained — three candidate mechanisms were proposed and each was
refuted by a profile measurement it had itself specified.

But the seven-task result relocates that question. The one task where a crosscoder beat the best
constant write has the least rank headroom of any measured, so its win is not about rank at all. The
sharpest question the sprint leaves is not what supplies a second direction, but **what makes a
crosscoder latent align well with the gradient on some tasks and not others** — and nothing in this
framework predicts it.
