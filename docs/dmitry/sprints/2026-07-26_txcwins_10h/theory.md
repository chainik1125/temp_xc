---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - design
  - in-progress
---

## What this document is

Task designs and registered predictions for the sprint goal: find further settings where a
temporal crosscoder beats a TopK SAE and a tSAE. It takes as given everything in
[[../2026-07-25_dictbench_10h/summary|last sprint's summary]] — steering not reading, the
constant-write argument, the geometric recovery ceiling, realised-L0 discipline — and does
not re-derive them.

The central move is to replace "permutation-invariance of the factor" with a **quantitative**
criterion. Permutation-invariance is binary and only says *whether* a constant write is
helpless. What is actually available is a three-number decomposition of the optimal write,
computable from a supervised difference-of-means slab **before any dictionary is trained**,
which predicts how much of the achievable steering effect each architecture can reach. That
turns "the crosscoder should win here" from a hunch into a number with an error bar.

Reading order if short on time: § The write algebra, § The screening statistic, then the
ranked table in § Ranking. Designs D2b, D1 and D6 are the ones to build first.

The single cheapest thing in this document, and the one I would run before anything else, is
the **gradient screen**: one backward pass per document yields a training-free, architecture-
independent upper bound on what any per-token dictionary can achieve on a proposed steering
target. Corpora, the screening function and the closed forms are in `blocks.py` and
`rank_check.py` in this folder, both self-tested.

## The write algebra

Fix a latent and a dose. Every intervention in this project is an additive write to the
residual stream over a window of `T` segments, so it is an element of `R^{T×d}`. The
architectures differ only in which subspace of `R^{T×d}` they can reach.

```text
SAE latent j, dose α:   x_t ← x_t + α · v_j            v_j ∈ R^d, same for every t
TXC latent j, dose α:   x_t ← x_t + α · P_j[t]         P_j ∈ R^{T×d}
```

Write `W ∈ R^{T×d}` for the assembled write. The reachable sets form a ladder:

| level | write form | reachable by | free parameters the experimenter must supply |
| --- | --- | --- | --- |
| L0 | `α · 1_T ⊗ v` | one SAE latent, one dose | 1 (the dose) |
| L1 | `α · s ⊗ v`, `s ∈ R^T` | one SAE latent + a hand-supplied dose schedule | `T` (the schedule) |
| L2 | `α · P`, `rank(P) = 1` | one TXC latent (if its slab happens to be rank 1) | 1 |
| L3 | `α · P`, `rank(P) = r > 1` | one TXC latent | 1 |
| L4 | `Σ_j α_j P_j` | several TXC latents | `m` |

Two separate claims live in this ladder and they are routinely conflated.

- **L0 → L2 is an expressiveness claim about deployed practice.** A per-token dictionary, used
  the way per-token dictionaries are used, writes the same vector at every position. This is
  what last sprint measured (per-position spread exactly 0.0000) and it is what fails on the
  order task.
- **L1 → L3 is an expressiveness claim about the architecture itself.** An L1 write is *not*
  something an SAE gives you: the schedule `s` has to come from somewhere. But a reviewer can
  always say "hand the SAE a ramp", so any design whose optimal write is rank 1 wins only at
  L0 → L2, and the honest statement there is about **discovery**, not representability.

**Only a task whose optimal write has rank > 1 separates a crosscoder from the strongest
possible per-token baseline.** Nothing in the sprint so far has checked the rank of an
optimal write. That check is cheap and it is the first thing to run.

### Where the tSAE sits, and why "constant write" is a claim about the protocol

The kickoff asks for a win over a tSAE as well as a TopK SAE, and the tSAE's position on the
ladder is settled by reading its decoder rather than by argument.
`temporal_crosscoders/han_tsae/saeTemporal.py:50` declares `D` of shape `(width, d_in)` — one
direction per latent, **no position axis** — and line 151 decodes as
`z @ D + b` identically at every position. All of the tSAE's temporal machinery
(lines 78–98: attention over the causal context) lives in the **encoder**. It changes *which
coefficients fire and when*; it does not give a latent more than one write direction.

So a tSAE latent's write is `α · z_j[t] · D[j]` — one direction with a data-supplied time
envelope. That is **exactly rank 1**. The tSAE is an L1/L2 architecture, not an L3 one, and it
is the natural strongest baseline precisely because it supplies the schedule *automatically*
where an SAE needs the experimenter to.

This generalises, and the generalisation is uncomfortable for the sprint's current headline.

> **A per-token dictionary's write is constrained to one *direction*. Whether it is constant
> *in time* is a property of the steering protocol, not of the architecture.**

Last sprint's `sae_broadcast` — one decoder direction at one dose at every position — is the
*weakest* form of the SAE baseline, not the strongest. The protocol people actually use is to
scale a latent by its own activation, which gives `α z_j(x_t) v_j`: a coefficient that varies
across positions, is data-dependent, and needs no experimenter knowledge. Under that protocol
a plain TopK SAE reaches rank-1 slabs, and **every rank-1 task ties.**

Two schedule-supplied SAE arms are worth separating, because they behave differently:

- **`sae_profile_self`** — coefficient is the latent's activation on the *current* document.
  Adaptive, but predicted to be near zero on any rotation task: amplifying "tense fires here"
  reinforces whatever structure the document already has, raising `logP` of A and B alike, so
  the *margin* moves only at second order.
- **`sae_profile_target`** — coefficient is the latent's **mean profile over class-A
  documents**, applied as a fixed schedule to every document. This is a genuine fixed rank-1
  write, needs no supervision beyond knowing which class you want (which the steering task
  hands you), and it is the real L1 baseline.

**Registered, and it should be run first because it is nearly free and it threatens the
existing result:** on the order task (`m = 2`, rank 1), `sae_profile_target` **closes the gap
to `txc_slab`** to within noise. If it does, last sprint's headline needs qualifying — the
crosscoder's advantage there was over a weak steering protocol, not over a per-token
dictionary — and the sprint should pivot entirely to `m ≥ 3`, where the rank argument still
bites. If instead `sae_profile_target` stays near zero while `txc_slab` is large at `m = 2`,
the rank account is incomplete and the causal-history contribution to the slab's rank is
doing more work than the block algebra predicts.

Either way this is the single highest-value hour in the sprint, because it is the strongest
available attack on the result the sprint is built on, and it costs one arm on data that
already exists.

### Why a constant write is annihilated by three different operators

Three exact statements, each stronger than "the multisets are matched", each giving a
different family of tasks. All are algebraic identities, not empirical tendencies.

- **Difference operator.** Let `D : R^{T×d} → R^{(T−1)×d}`, `(DX)[t] = x_{t+1} − x_t`. For a
  constant write, `D(X + 1_T ⊗ v) = DX` exactly. A per-token dictionary's write is in the
  kernel of `D`. Any behavioural target that is a function of `DX` alone is unreachable by a
  constant write, *for any dose, any direction, and any latent*. Gives the **rate/trend**
  family (D5, D6).
- **Argmax operator.** For any fixed read direction `u`, `argmax_t ⟨x_t + αv, u⟩ =
  argmax_t ⟨x_t, u⟩`. A constant write shifts every position's projection by the same scalar
  and therefore cannot move a peak, a crossing point, or an onset. Gives the **timing/phase**
  family (D2, D4).
- **Zero-sum rotation.** If two classes are cyclic rotations of the same `m` blocks, the
  per-position difference-of-means slab has rows `b_t − b_{t+1}` which sum to exactly zero, so
  its projection onto the constant subspace is **exactly** the zero matrix. The optimal write
  is orthogonal to everything an SAE can write. Gives the **rotation ladder** (D1).

The third is the tightest matching available and strictly improves on last sprint's order
task, which matched the multiset and the switch count but not the run-length profile. A
cyclic rotation matches the multiset, the switch count, the run-length multiset, and the
phase-marginal of every block type simultaneously, because the two documents are literally
the same string read from different starting points.

## The screening statistic

Let `P ∈ R^{T×d}` be the supervised per-position difference-of-means slab between the two
classes — already computed as `P_dom` at `steer_order_modal.py:226`. Define three energy
shares:

```text
c    = T · ‖mean_t P[t]‖² / ‖P‖_F²          constant share       (L0 reachable)
r1   = σ₁² / ‖P‖_F²                          rank-1 share         (L1/L2 reachable)
slab = 1 − r1                                slab-only residual   (L3 only)
```

where `σ₁` is the largest singular value of `P` viewed as a `T × d` matrix. Note `c ≤ r1`
always, since the constant subspace is a subset of the rank-1 matrices.

**The rank law (registered).** In the small-dose regime where the margin responds linearly to
the write, and taking each arm's write rescaled to the same Frobenius norm, the steering
effect of an arm restricted to subspace `Π` is

```text
Δ_Π / Δ_full  ≈  ‖Π P‖_F / ‖P‖_F  =  sqrt(energy share)
```

The square root, not the share itself: to first order `Δ ≈ α⟨W, G⟩` with `G` the gradient of
the margin with respect to the activations; if `G ∝ P` then a norm-matched write in subspace
`Π` gives `Δ ∝ ⟨ΠP, P⟩/‖ΠP‖ = ‖ΠP‖`. So predicted ratios are `sqrt(c)` for a constant write
and `sqrt(r1)` for the best rank-1 write. **Measure at the smallest dose with a significant
effect**, because the law is a linearisation and the existing alpha grid runs into saturation
at the top end.

This is the same kind of object as last sprint's best result — a pre-computable geometric
quantity that measured recovery matches to three decimals — and it is why I would spend the
first hour here rather than on a new corpus.

### Screen the gradient, not the difference-of-means

The rank law above needs `G ∝ P`, where `G` is the margin's gradient and `P` the DoM slab.
That assumption is avoidable, and dropping it makes the whole screen sharper.

`P_dom` and `G` are different objects: `P_dom` is the direction that *distinguishes* the two
classes, `G` is the direction that most *increases* the margin. Steering cares about the
second. So compute the screen on the **mean margin-gradient slab** instead:

```text
Ḡ = (1/N) Σ_i ∂ margin_i / ∂ x  |  restricted to the T segment positions,
                                   mean-pooled within each segment span
```

The mean, not the per-document gradient, because a dictionary latent is **one fixed write
reused across documents**: its expected effect is `E[Δ] = α⟨W, E[G]⟩ = α⟨W, Ḡ⟩`. That makes
the rank law an identity rather than an assumption — in the linear regime the best
norm-matched write inside a subspace `Π` is `Π Ḡ / ‖Π Ḡ‖` and it achieves exactly
`α ‖Π Ḡ‖`, so

```text
Δ_Π / Δ_full = ‖Π Ḡ‖_F / ‖Ḡ‖_F        exactly, no proportionality assumption
```

Two things follow, and both are worth more than the DoM version.

- **The screen becomes architecture-independent and training-free.** One backward pass per
  document gives an upper bound on what *any* per-token dictionary can achieve on the task,
  before choosing a dictionary, a layer, or a latent. This is a reusable instrument, not a
  step in one experiment: it answers "can a per-token dictionary possibly steer this?" for any
  proposed steering target at a cost of one backward pass.
- **It upgrades the baseline from "best learned latent" to "best possible constant write".**
  Add `oracle_const = Π_const Ḡ`, `oracle_rank1`, and `oracle_slab = Ḡ` to the arm ladder.
  These bound `sae_broadcast`, `sae_enveloped` and `txc_slab` respectively. The negative claim
  then becomes airtight: if `oracle_const` gives Δ ≈ 0, no SAE latent — better trained, better
  selected, or hand-picked — could ever have done it, and "you just picked a bad latent" is
  answered before it is raised. `oracle_slab` also supersedes `dom_slab` as the ceiling, and
  the 7× gap to the supervised ceiling left open by the last sprint is exactly the quantity it
  measures.

Report `cos(Ḡ, P_dom)` alongside. If it is high, the discriminative and causal directions
coincide on this task and the DoM-based numbers stand; if it is low, that is itself worth a
line, since the steering literature routinely uses difference-of-means as though it were the
gradient.

### `c` is the reading result and the steering result in one number

A linear probe on mean-pooled activations separates the two classes if and only if
`mean_t P[t] ≠ 0` — which is exactly the condition `c > 0`. So:

> **`c = 0` ⟺ a linear pooled per-token probe is at chance. A task is steerable by a constant
> write exactly to the extent that it is readable by pooling.**

This ties the two halves of the project together and it is retroactively confirming. Last
sprint's pooled SAE read the order label at AUC 0.998–1.000, which *already implied* `c > 0`
on that task, which in turn implied `sae_broadcast > 0` — and the measured value was `+1.24`,
small but positive. The reading result and the steering result were never independent.

**Correction to the harness gate registered earlier in this document.** I wrote that
`c = 0` exactly at every `m` in a rotation design and that a nonzero value indicates window
mis-alignment. That is the *block algebra*, which assumes context-free representations. In a
causal transformer position `t` of class A carries a different prefix from position `t` of
class B, so the rows are not exactly antipodal and `c > 0` strictly. **The gate is `c < 0.1`,
not `c ≈ 0`**, and measured `c` should be read as a quantification of the causal-history
contribution rather than as a bug. Anyone who goes bug-hunting on a measured `c = 0.04`
because of the earlier wording is chasing my error.

**Registered prediction for the order task** (`m = 2`), before the number lands: `c ≈ 0.03`
(range 0.005–0.12) and `r1 ≈ 0.93` (range 0.82–0.99). The derivation is that with `τ(ctx)`,
`κ(ctx)` for tense and calm segments under context `ctx`,

```text
P_early = τ(T) − κ(C)          P_late = κ(T) − τ(C)
Σ_t P[t] ∝ [τ(T) + κ(T)] − [τ(C) + κ(C)]
```

which is the tense-prefix-versus-calm-prefix contrast — zero only if representations are
context-free. Consequences: `Δ(sae_profile_target)/Δ(txc_slab) ≈ sqrt(0.93) ≈ 0.96`, so a
profile-steered SAE closes most but not all of last sprint's gap, and `1 − r1 ∈ [0.02, 0.15]`
bounds the fraction of that result which was genuinely L3. **Falsifier:** measured
`c < 1e-10`, which would refute the causal-history argument and mean representations are
effectively context-free at this layer — surprising, and reportable in its own right.

**Three numbers, three verdicts, before a single dictionary trains:**

| screen | verdict |
| --- | --- |
| `c` large (> 0.3) | a constant write can do most of the job; no design here, discard the task |
| `c ≈ 0`, `r1 ≈ 1` | plain SAE fails, scheduled SAE ties the TXC — the win is discovery, not expressiveness |
| `c ≈ 0`, `r1 < 0.6` | the only regime where a slab is structurally required; build the task here |

**Registered retro-diction on data already in hand.** The existing order task
(`results/dict_bench/steer_order.json`, two blocks swapped) is an `m = 2` rotation, so block
algebra says `c = 0` and `r1 = 1`: the optimal write is `[+Δ, −Δ]`, rank 1. If the measured
`r1` on the real `P_dom` comes back at 1.00, then last sprint's headline is an L0 → L2 result
and a scheduled SAE would have closed it — which must be said plainly. If `r1` comes back
appreciably below 1, the excess rank is contributed by the **causal history**: `x_t` in class
A at late positions is not the mirror of class B, because each token has its own prefix
written into it. That would be a small and pleasing reversal — the same property that made
reading comparisons hopeless makes steering comparisons *easier*, because it raises the rank
of the optimal write above what the block algebra alone would give.

Either answer is publishable and the run is one SVD on a tensor that already exists.

## The rotation spectrum

The `m`-block rotation is the design this document leans on hardest, so its spectrum is worth
deriving rather than guessing. **I guessed it first and got it wrong**; the numbers below are
checked against measured SVDs and agree to four decimals, script in
`scratchpad/rank_check2.py`.

Write `P = C B`, where `B ∈ R^{m×d}` holds the block content vectors and `C` is the circulant
matrix with first row `(1, −1, 0, …, 0)`. For orthonormal centred `B`, the singular values of
`P` are those of `C`, whose symbol is `f(ω) = 1 − ω`:

```text
σ_j² = 4 sin²(π j / m),   j = 0 … m−1
Σ_j σ_j² = 2m
σ_0² = 0                                  <- the constant mode, so c = 0 exactly
r1 = 4 sin²(π ⌊m/2⌋ / m) / (2m)
   = 2/m                    for even m
   = 2 cos²(π/(2m)) / m     for odd m
```

Both branches approach `2/m`. The earlier guess `r1 = 1/(m−1)` is right only at `m = 2, 3` and
is wrong from `m = 4` on.

| m | rank | `r1` | `sqrt(r1)` | top-2 share | block length at `k_seg = 12` |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | 1.000 | 1.000 | 1.000 | 6 |
| 3 | 2 | 0.500 | 0.707 | **1.000** | 4 |
| 4 | 3 | 0.500 | 0.707 | 0.750 | 3 |
| 6 | 5 | 0.333 | 0.577 | 0.583 | 2 |
| 12 | 11 | 0.167 | 0.408 | 0.322 | 1 |

**These are a lower bound, not an equality.** The derivation assumes the block-mean matrix
`B` has orthonormal rows. Real block means do not, and the effect always *inflates* `r1`,
never deflates it — measured over 200 draws, `m = 3` gives 0.578 rather than 0.500 and
`m = 12` gives 0.219 rather than 0.167. So the L3 headroom `1 − r1` is **smaller** than the
ideal algebra promises. The inflation is a non-orthonormality and finite-sample effect rather
than a correlation effect (ρ = 0, 0.5, 0.8 between block means all give ≈ 0.578 at `m = 3`),
so it cannot be removed by choosing semantically dissimilar block types — but it *can* be
reduced, and that is what the vocabulary choice below is for.

Two consequences for how this is tested:

- **Test the law against measured `r1`, never predicted `r1`.** That decouples the law
  (`Δ_Π/Δ_full ≈ sqrt(energy share)`, a claim about linear response) from the spectrum (a
  claim about block geometry, now known to be sensitive to a nuisance parameter). Tested
  against the prediction, a failure is unattributable to either.
- **The spectrum then becomes a separate, also-checkable prediction:** measured `r1` should
  sit at or above the orthonormal bound, and the gap between them is a readout of how far the
  block means are from orthogonal.

**Vocabulary follows from this.** The original pools contained *calm* and *tense* — two poles
of a single affective axis, whose difference is one dominant direction that mechanically
inflates `σ₁` and eats exactly the headroom the design exists to create. They are replaced by
twelve mutually distinct **technical** registers with no shared axis (mechanical, legal,
culinary, astronomical, nautical, musical, geological, textile, veterinary, architectural,
meteorological, numismatic) in
`experiments/temporal_screen/txc_wins/designs_theory.py`.

Three consequences for the design, none of which were visible before doing the arithmetic:

- **`m = 4` is a wasted rung.** It has the same `r1` as `m = 3`, because `σ²` is
  `4 sin²(πj/m)` and the maximum saturates at 4 for every even `m`. Sweep
  `m ∈ {2, 3, 6, 12}` — all of which divide `k_seg = 12` — for `r1 = 1.00, 0.50, 0.33, 0.17`.
- **The spectrum is degenerate in pairs** (`j ↔ m − j`), so a rank-**2** arm is far more
  informative than the rank-1 arm alone. At `m = 3` the rank is exactly 2, so `txc_rank2`
  should recover **100%** of `txc_slab`; the top-2 shares at `m = 4, 6, 12` are
  0.750, 0.583, 0.322. Add `txc_rank2` to the arm ladder — it is one more line and it turns a
  single point into a curve.
- **The gap widens only as `2/m`**, which is slow. To halve the scheduled-SAE ceiling you must
  double the number of blocks, which at fixed `k_seg` halves the block length.
  Recommendation: `m ∈ {2, 3, 6}` as the workhorse, `m = 12` as a stretch rung run last.

### The coherence confound, and the grouped ladder that removes it

Raising `m` in the naive design also raises the number of distinct registers in a document:
at `m = 2` a document is six calm then six tense sentences and reads as a narrative, while at
`m = 6` it is a collage of six unrelated registers and at `m = 12` it is register salad. **So
`m` is confounded with coherence**, and any trend across the ladder is partly a trend in how
far the text is from the model's distribution. That inflates margin variance and makes the
teacher-forced numbers less meaningful at exactly the rungs the theory cares most about.

(I earlier wrote that the absolute effect "will be small" at `m = 12`. That was a guess stated
as a derivation and I have not verified it — with twelve distinct registers every position
differs between the classes, so the effect could equally be large. The defensible statement is
about *variance and distributional shift*, not effect size.)

**The fix costs nothing: hold the register count fixed and vary only the grouping.** Use all
six registers and all twelve segments in *every* condition, and let `m` set how those
registers are grouped into rotation blocks:

| m | grouping of the 6 registers | block length |
| --- | --- | --- |
| 2 | `{1,2,3} {4,5,6}` | 6 |
| 3 | `{1,2} {3,4} {5,6}` | 4 |
| 6 | `{1} {2} {3} {4} {5} {6}` | 2 |

Every document now contains the same twelve segments drawn from the same six registers, so
lexical content, coherence and distributional shift are matched across the whole ladder and
only the block structure moves. The circulant algebra is unchanged — the block content vectors
are group means, the difference rows are still `b_t − b_{t+1}` — but the group means are
averages of subsets and so are less mutually equidistant than single registers, which shows up
in `block_geometry` and shifts measured `r1` above the closed form. That is a measurable,
reportable deviation rather than a hidden one.

Run the grouped ladder as the headline and the naive one as a robustness check if time allows.

**With twelve registers the grouping can be made exact rather than approximate.** The
groupings are built so that group size equals block length at every `m` — 6/6, 4/4, 3/3, 2/2,
1/1 — so drawing each block as a *permutation* of its group rather than sampling with
replacement puts all twelve registers into every document exactly once at every `m`. Register
composition is then matched **identically** across the ladder rather than in expectation, and
each block's content vector is exactly its group mean with no sampling noise, which tightens
the difference slab the whole rank argument is computed from.

### The phase ladder is a different experiment, and cannot reach L3

A natural-looking variant is to hold the vocabulary at two pools and sweep the number of
switches (1, 3, 5, 11) with the foil built as a rotation by one block. It is **not** the
rotation ladder at `m = 2, 4, 6, 12`, and the difference is decisive: with only two distinct
block content vectors the difference rows are `±(a − b)` whatever the block length, so it is
**rank 1 at every rung**. Measured `r1 = 1.0000` and rank 1 at all four switch counts, against
rank 1, 2, 3, 5, 11 for the rotation ladder at `m = 2, 3, 4, 6, 12`.

So every rung of the phase ladder ties against `sae_profile_target` and the tSAE. It remains
worth running, but for a different question — how the advantage over a *constant* write varies
with the frequency the profile must resolve, which is also the natural test of the
scales-longer-than-`T` scope limit in P10. Bill it as a mechanism experiment about the
constant-write baseline, not as a candidate for a win over the strong baselines. Registered:
advantage over `sae_broadcast` roughly flat across rungs, advantage over
`sae_profile_target` zero at all of them, and the absolute effect degrading once the
alternation period falls below the crosscoder's effective resolution.

The `sqrt(r1)` law itself survived the check: under the linear-response assumption `G ∝ P`,
the norm-matched rank-1 write recovers `sqrt(r1)` of the full effect to three decimals at
every `m` tested. That is a check of the algebra only — whether `G ∝ P` holds in a real model
is the empirical question the sweep answers.

## Structural properties

Each entry states the property, the exact reason a constant-in-time write cannot express it,
and which level of the ladder it reaches.

### P1 — the write must change sign across the window

Target: suppress a behaviour early and induce it late (or the reverse). Requires
`⟨W_t, u⟩ < 0` for small `t` and `> 0` for large `t`, for the behaviour's read direction `u`.
A constant write has `⟨αv, u⟩` of a single sign at every position, so the sign pattern is
unreachable **for any `v` and any `α`**. Reaches L2 only: the write `s ⊗ u` with `s` a sign
ramp is rank 1, so a scheduled SAE ties. Relevance is high (comply-then-refuse; explore-then-
commit); expressiveness claim is weak.

### P2 — simultaneous opposite writes on *different* features

Target: suppress feature A at early positions while inducing a *different* feature B at late
positions, with `u_A ⊥ u_B`. Then the optimal write is `W = [−β u_A ; +γ u_B]` block-wise,
whose rank is 2, and the best rank-1 approximation captures `max(β², γ²)/(β² + γ²) < 1` of
the energy. **No single direction with any schedule can do it**, so this is the first property
that reaches L3. This, not P1, is the property worth hunting.

The clean generator for P2 is not a two-block swap — every two-block swap is rank 1, because
the difference slab is `[+Δ, −Δ]`. It is an **`m`-block cyclic rotation with `m ≥ 3`**: the
rows `b_t − b_{t+1}` sum to zero and span an `(m−1)`-dimensional space, so rank is exactly
`m − 1`. The energy is *not* spread evenly across those `m − 1` modes — see § The rotation
spectrum for the closed form, which is `r1 ≈ 2/m`, not `1/(m−1)`. That is a tunable knob on
the expressiveness gap, but a slower one than it first appears.

### P3 — the target is a rate, not a level

Target defined by `DX` only, with the level matched between classes. Constant writes are in
`ker D` exactly. This is the strongest algebraic statement available and it supports a
**double dissociation** rather than a one-sided comparison (D6), which is the most
objection-proof design in this document.

### P4 — onset and offset timing

Target is *when* a behaviour starts, given that it occurs in both classes. Constant writes
cannot move an argmax, a first-crossing, or a peak location. Matched by cyclic rotation of a
localised block, so the multiset, the duration and the number of transitions are identical and
only the phase differs. Reaches L2 (the required write is a localised bump, approximately
rank 1) unless the pre-onset and post-onset regions differ in *direction* as well as sign.

### P5 — the target is a relation between two positions

A-before-B, cause-before-effect, trigger-before-payload. This is the order task's property and
is already established; listed for completeness. Reaches L2.

### P6 — duration at matched total mass

Target: the behaviour is active for 3 segments at high intensity versus 6 at half intensity.
**Predicted to fail** as a separator: if the behavioural readout is a threshold crossing,
`1[⟨x_t, u⟩ > θ]`, a constant write raises every position's projection and therefore does
change the number of positions above threshold. A constant write is a crude but genuine
duration knob. Expect `c` to come back large; expect the SAE to get a substantial fraction.

### P7 — the per-token dictionary cannot *learn* the feature

The tempting claim: if two classes have identical per-segment marginals and differ only in the
joint arrangement, a reconstruction-trained per-token dictionary sees one distribution and
cannot allocate a selective latent. **This is wrong here and should not be attempted.** The
subject model is causal, so `x_t` carries its prefix; matching the marginals of the *labels*
does not match the marginals of the *activations*. This is the same mechanism that gave the
SAE AUC 1.000 on a task designed to defeat it. Listed as a predicted failure so nobody
rediscovers it at 3 a.m.

### P8 — selectivity: collateral damage of a write that *can* reach the target

Distinct from all of the above and probably the most practically relevant. Even where a
constant write can move the target factor, it necessarily perturbs **every** position,
including those where no change is wanted. A slab can write zero where zero is wanted. So the
comparison should never be raw Δ alone: report Δ on the target factor against a cost measured
as KL divergence from the unsteered model on held-out neutral text at the same injected norm.
Prediction: the crosscoder dominates the (Δ_target, KL_collateral) frontier **even on tasks
where the raw Δ ties**, because it can concentrate its budget. This is one extra forward pass
per document and applies to every design below.

### P9 — position-dependent norm is a confound, not a property

The residual stream's norm grows with position, so a constant write is a *smaller relative*
perturbation late in the window and a slab can learn a compensating envelope. A crosscoder can
therefore win for an entirely uninteresting reason: it learned `1/‖x_t‖`. Neither the
`txc_flat` control (removes the profile entirely) nor `random_slab` (flat expected norm
profile) catches this. **The control that catches it is `sae_enveloped`** — see the arm ladder
below. Flagging this as the most likely spurious mechanism for a crosscoder win tonight.

### P10 — the crosscoder's write is indexed by absolute position

Scope limit, not a property. The slab spans `T` segments and indexes them by position within
the window. Two consequences: structure at scales longer than `T` gets no advantage at all,
and the write is only meaningful if the window is *aligned* to the behaviour. The current
harness aligns by writing over segment token spans, which is why it works. Under free
generation, segment boundaries move and position indices decorrelate from semantics — see
§ Scope limits.

## The arm ladder

Every design below runs the same arms. Four are new. The point of the ladder is that the Δ
between consecutive rungs attributes the effect to exactly one property of the write.

| arm | write | isolates |
| --- | --- | --- |
| `zero` | none | base drift |
| `sae_broadcast` | SAE latent direction at every position | L0, deployed practice |
| `random_broadcast` | random unit direction at every position | whether a constant write of *any* direction does anything |
| **`sae_enveloped`** *(new)* | SAE direction, scaled per position by the TXC slab's own norm profile `‖P[t]‖` | **separates gain envelope from direction schedule; the P9 control** |
| **`sae_profile_target`** *(new)* | SAE direction, scaled by the latent's mean activation profile over class-A documents | **the real L1 baseline — the strongest non-oracle per-token arm, and the one that threatens the headline** |
| **`sae_profile_self`** *(new)* | SAE direction, scaled by the latent's activation on the current document | the adaptive protocol; predicted ≈ 0 on rotation tasks |
| **`tsae_slab`** *(new)* | tSAE latent's `z_j[t] · D[j]` | a real architecture at L1; capped at `sqrt(r1)` like any rank-1 write |
| **`txc_rank1`** *(new)* | best rank-1 approximation `σ₁ u₁ v₁ᵀ` of the TXC slab | the L1/L2 boundary — how much of the slab is a schedule |
| **`txc_rank2`** *(new)* | best rank-2 approximation of the TXC slab | the rotation spectrum is pair-degenerate, so this ties at `m = 3` and falls off a known curve after |
| `txc_slab` | full TXC slab | L3 |
| `txc_flat` | TXC slab time-averaged and rebroadcast | that the profile, not the mean direction, does the work |
| `random_slab` | random `(T, d)` slab | that it is not any structured perturbation |
| **`dom_rank1`** *(new)* | best rank-1 approximation of the supervised DoM slab | the rank law's prediction, supervised |
| `dom_slab` | supervised per-position DoM slab | ceiling (discriminative) |
| **`oracle_const`** *(new)* | `Π_const Ḡ` — the best constant write that exists | bounds **every** SAE latent, so "you picked a bad latent" cannot be raised |
| **`oracle_rank1`** *(new)* | best rank-1 approximation of `Ḡ` | bounds every scheduled-SAE arm |
| **`oracle_slab`** *(new)* | `Ḡ`, the mean margin-gradient slab | the true ceiling; supersedes `dom_slab` |
| **`txc_transfer`** *(new)* | TXC slab selected on corpus 1, applied without refitting to corpus 2 | that the schedule is a learned property, not a fitted nuisance parameter |

All rescaled to the same total injected Frobenius norm. `sae_enveloped` and `txc_rank1` are
the two that decide whether the sprint's headline is an expressiveness claim or a discovery
claim, and both are a few lines from tensors the existing script already builds.

### Expressiveness and discovery are two different results

The oracle arms are not just stronger baselines; they split the headline into two claims that
have been running together all along, and the split is what the arm ladder is really for.

| | **can express** (oracle, training-free) | **does learn** (trained dictionary) |
| --- | --- | --- |
| constant write | `oracle_const` | `sae_broadcast` |
| rank-1 write | `oracle_rank1` | `txc_rank1`, `sae_enveloped` |
| full slab | `oracle_slab` = `Ḡ` | `txc_slab` |

- The **left column** is an architecture claim and needs no dictionary at all. It is a
  statement about the task: *this target is or is not reachable by a constant write*. It
  cannot be attacked on training, selection, or hyperparameters, which is exactly why it is
  worth having.
- The **right column** is a claim about what unsupervised training actually finds. A
  crosscoder can express a slab; whether it *allocates a latent* to the one you want is a
  separate empirical question, and at `m = 6` with six registers it plausibly allocates
  latents to individual registers instead of to the rotation.

Report both. The gap between a column's rows is the expressiveness result; the gap *between*
the columns in a row is how much unsupervised training leaves on the table. If `oracle_const`
is 0 and `txc_slab` is large, the finding is structural and safe. If `oracle_slab` is large
but `txc_slab` is small, the finding is that the crosscoder failed to discover an available
write — which is a real and reportable negative about the architecture, not about the task.

The diagnostic that tells them apart is `cos(txc_slab, Ḡ)`. Report it for the selected latent
alongside its AUC; a low cosine with a large `oracle_slab` says the dictionary does not
contain the needed write, and no amount of dose tuning will fix that.

`txc_transfer` is what makes an L1/L2 design worth running at all. If the schedule transfers
to held-out documents and to a structurally similar but lexically disjoint corpus with no
refitting, then "the experimenter could have supplied the schedule" is answered by "they would
have had to know it, and the crosscoder did not".

## Task designs

### D1 — rotation ladder (the expressiveness result)

**Classes.** `m` semantically distinct sentence blocks laid end to end. Class A is the
canonical order `(1, 2, …, m)`; class B is the cyclic rotation `(2, 3, …, m, 1)`. Same
sentences, same counts, same run lengths, same number of transitions, same everything except
the starting point. Sweep `m ∈ {2, 3, 6}` at fixed total segment count (`k_seg = 12`, so block
length `12/m`), with `m = 12` as a stretch rung run last. **Not `m = 4`** — it has the same
`r1` as `m = 3` and buys no new information (§ The rotation spectrum).

**Construction requirement — the one way to get this wrong.** Build the two classes from a
**single** sentence draw and rotate the *assembled list*; do not draw each class
independently. Independent draws match the register counts only in expectation, and under the
grouped ladder they do not even do that, because each block re-draws which of its grouped
registers it uses. Measured at `m = 2` with one seed, an independent-draw implementation gave
class A `legal:4` and class B `calm:4` — a lexical imbalance pointing straight at the factor
under test, which a constant write *can* exploit and which would have produced a false
positive for `sae_broadcast` rather than the registered zero. `rotation_pair()` in `blocks.py`
does it correctly and asserts exact multiset equality and exact rotation for every `m` in both
ladders. Share the carrier prefix between the pair for the same reason.

**Readout (reading).** Pooled per-token SAE code versus window code, AUC. Expect the SAE to
win as always; run it only to keep the reading/steering dissociation on the record.

**Write (steering).** Teacher-forced margin `logP(A) − logP(B)` on multiset-matched pairs, the
existing metric, no judge.

**Controls.** The full arm ladder. Additionally the pre-computed `c` and `r1` from the DoM
slab at each `m`.

**Registered predictions.**

- `c = 0.00 ± 0.02` at every `m` (algebraic; a nonzero value means the harness is
  mis-aligning windows and everything downstream is suspect — treat as a harness gate).
- `sae_broadcast` Δ ≈ 0 at every `m`.
- `r1 = 4 sin²(π⌊m/2⌋/m)/(2m)`, i.e. `1.000, 0.500, 0.333, 0.167` for `m = 2, 3, 6, 12`, up to
  the anisotropy of the block vectors. Report the measured value; the *prediction under test*
  is the monotone decrease and its `2/m` rate, not the exact constants. A measured `r1` well
  *above* the closed form means the blocks are not close to equidistant in activation space —
  informative in itself, and the same diagnostic D3 needs.
- **The rank law:** `Δ(txc_rank1)/Δ(txc_slab) ≈ sqrt(r1)` = `1.00, 0.71, 0.58, 0.41`. Same for
  `dom_rank1/dom_slab`. If the observed ratio is flat in `m`, the law is wrong and the rank
  account of the advantage fails — a clean negative.
- **`txc_rank2` recovers 100% at `m = 3`** (the slab is exactly rank 2 there), then
  `sqrt(0.750), sqrt(0.583), sqrt(0.322)` = `0.87, 0.76, 0.57` at `m = 4, 6, 12`. This is the
  sharpest single prediction in the document: an arm that is *predicted to tie* at one rung
  and fall off a known curve at the others.
- At `m = 2` the crosscoder's advantage over `sae_enveloped` is **zero within noise**. This is
  a prediction that the last sprint's headline task is *not* an expressiveness win.

**What kills it.** `sae_enveloped` matching `txc_slab` at `m ≥ 3`. That would mean the gain
envelope carries everything and the direction schedule is decorative, which would reduce the
whole programme to "the crosscoder discovers a dose schedule".

**Cost.** One training run per `m`, reusing `steer_order_modal.py` with a rotated document
generator. Cheapest design here and the only one that yields a law.

### D2 — refusal onset (the relevance flagship)

**Classes.** Borderline requests where both continuations contain the same clauses. Class A =
engage-then-decline ("Here is the general background … *However*, I can't help with the
specific steps."); class B = decline-then-engage ("I can't help with the specific steps.
*That said*, here is the general background …"). Sentence multiset matched by construction;
only the connectives differ and they are held to a fixed pair across all items.

**Readout.** Same pooled-versus-window AUC, for the record.

**Write.** Teacher-forced Δmargin between the two orderings. **No judge, no sampling** — which
is the whole reason to use ordering as the target rather than refusal rate.

**Controls.** Full arm ladder, plus the P8 selectivity metric: KL from the unsteered model on
a held-out set of benign prompts at the same injected norm.

**Registered predictions.**

- `c ≈ 0`, `r1` high (this is an `m = 2` rotation) — so `sae_broadcast` ≈ 0 and
  `sae_enveloped` ≈ `txc_slab`. The expressiveness gap is predicted to be **absent**; the
  discovery gap and the selectivity gap are what this design is for.
- `txc_transfer` from the finance/medical-style corpus of the EM work to this one retains
  ≥ 50% of the effect. If it retains ~0, the schedule is corpus-specific and the discovery
  claim weakens to "fits a nuisance parameter".
- Crosscoder KL cost at matched Δ is **lower** than the SAE's by a clear margin, because it
  need not perturb the early engaging segments.

**Relevance note.** This is directly the second temporal steering task in the reviewer-response
workstream, and "refuse after explaining, not before" is a documented preference. If any
design here ends up in a paper, it is this one.

### D2b — three-part refusal rotation (rank ≥ 2 *and* relevance)

The gap in every design above is that rank ≥ 2 and relevance never co-occur: D1 and D3 reach
L3 but are constructs, D2 is the relevant one but is `m = 2` and therefore rank 1. D2b closes
that gap, and on reflection it is the design I would build if only one thing gets run.

**The observation.** A well-formed refusal is not two parts, it is *three*: acknowledge the
request, decline with a reason, offer an alternative. That is the shape assistant guidelines
actually ask for, and it is three semantically distinct modes rather than two — so a cyclic
rotation of the three is an `m = 3` design, **rank 2 by the circulant algebra**, while
remaining a behaviour someone would genuinely want to steer.

**Classes.** Three clauses per item, rotated as a unit:

```text
class A (canonical):   acknowledge / decline / alternative
class B (rotation):    decline / alternative / acknowledge
```

Exactly multiset-matched, built from a single draw and rotated (see D1's construction
requirement). Shared carrier and a fixed connective set across all items.

**Per-item repetition is free, and I checked rather than assumed it.** A document is item 1's
three clauses, then item 2's, and so on — natural text — rather than one big block per clause
type, which would read as four acknowledgements in a row. I expected the repetition to change
the rank and it does not: the difference matrix has the same three distinct rows repeated, so
its Gram scales by the repeat count and the singular-value *ratios*, hence `r1`, are
unchanged. Measured identical at `m = 3`: block form `r1 = 0.5000` rank 2, per-item form
`r1 = 0.5000` rank 2 (`design_rank.py`). The same holds for D2 at `m = 2`.

**Write.** Teacher-forced Δmargin between the canonical order and the rotation. No judge, no
sampling.

**Registered predictions.**

- `c = 0` exactly; `r1 ≈ 0.50`; rank exactly 2, so `txc_rank2` recovers ≈ 100%.
- `sae_broadcast ≈ 0`; `sae_profile_target`, `sae_enveloped` and `tsae_slab` all capped at
  ≈ `sqrt(0.5) = 0.71` of `txc_slab`. **This is the only design in the document predicted to
  beat all three baselines on a behaviour anyone cares about.**
- Selectivity (P8): crosscoder KL cost lower at matched Δ.

**The risk, and it is the same one as D3.** The three clause types may not be mutually
equidistant in activation space — "decline" and "alternative" both carry refusal-adjacent
content and may sit close, which would collapse the rank toward 1 and make this an `m = 2`
task wearing an `m = 3` costume. **This is measurable before any training compute is spent:**
run `block_geometry()` on the three clause-type centroids and check `pairwise_cv`. If the
decline/alternative distance is much smaller than the other two, either re-word the
alternative clause to be more concretely helpful (moving it toward a "here is what I can do"
direction and away from refusal) or fall back to D3.

**Variant with the same structure, if the refusal framing is unwanted:** chain-of-thought
order — restate / work / answer, rotated to answer-first. The rotation is exactly the
post-hoc-rationalisation failure mode, which is a documented concern and gives the same rank-2
guarantee.

### D9 — induction, and why it is a discovery-track task

Induction has one virtue no other design here has: its matched-multiset foil is **the
induction literature's own standard control**, which pre-empts "you built the task to win".
That is worth something real, and it is a different currency from expressiveness.

**Registered rank prediction: `r1 ≈ 0.90` (range 0.80–0.97) — effectively rank 1.** Token
identity is matched in distribution across classes so it averages out of the difference slab,
but the decisive point is *where* the signal sits. In the first half nothing distinguishes the
classes yet, since the repeat has not occurred, so `P[t] ≈ 0`. In the second half the model is
in induction mode, so `P[t] ≈ s_t · u_induction`. The slab is a step or ramp `s_t` times **one**
direction.

The only route to rank ≥ 2 is if match-*detection* early in the repeat and copy-*execution*
later are carried by different directions. That is not implausible given the
previous-token-head / induction-head decomposition, but it is speculative — hence 0.90 rather
than 0.99.

**So induction does not outrank D1 and should not be sold as if it does.** If it runs and
wins, the honest headline is "the crosscoder discovers the schedule on a task we did not
design" — a discovery claim on a credible task, which is genuinely valuable and is *not* an L3
claim. The write-up must say which currency each experiment bought.

**Do not build it to find out — screen it.** The gradient screen settles this in one backward
pass per document on ~100 sequences, before any corpus or dictionary exists. Measured
`r1 < 0.8` refutes me and the task jumps the queue; `r1 > 0.9` schedules it as discovery-track.

A genuinely rank-2 induction variant exists — two interleaved patterns `A→B` and `C→D` with
their completion positions swapped between classes, forcing "copy B early, copy D late" —
but it reintroduces exactly the built-to-win objection that motivated induction in the first
place, so it is worth building only if D1 and D2b both fail.

### D3 — three-phase reasoning rotation (D1 with relevance)

**Classes.** Reasoning traces with three genuinely distinct modes — *explore* ("One option is
…"), *commit* ("So the answer is …"), *verify* ("Checking: …"). Class A = explore/commit/
verify; class B = the rotation commit/verify/explore. This is D1 at `m = 3` with content
anyone would recognise, so it inherits D1's rank-2 guarantee **and** connects to reasoning
structure and to this repo's backtracking case study.

**Prediction.** As D1 at `m = 3`: `r1 ≈ 0.5`, `sae_enveloped` at ≈ 0.71 of `txc_slab`,
`sae_broadcast` ≈ 0. If it replicates D1's numbers on semantically real blocks, D1 stops being
a construct.

**Risk.** The three modes may not be equidistant in activation space — verify and commit may be
close, collapsing the rank toward 1. That is measurable in advance from `r1` and is a reason to
compute the screen on candidate corpora *before* committing training compute.

### D4 — onset phase shift

**Classes.** A localised block of `L` "shifted-register" sentences embedded in a neutral
carrier of 12 segments, starting at position 2 versus position 7. Cyclic, so the multiset,
block length and transition count are all identical.

**Write.** Δmargin between early-onset and late-onset orderings.

**Prediction.** `sae_broadcast` ≈ 0 by the argmax argument. `r1` high (the required write is a
bump envelope times one direction), so `sae_enveloped` ties. Reaches L2. Run it as the
cleanest demonstration of the argmax identity, not as an expressiveness claim.

### D5 — trend at matched level

**Classes.** Twelve sentences drawn from a graded intensity scale, sorted ascending versus
descending. Identical multiset, identical mean intensity, opposite trend.

**Write.** Δmargin between ascending and descending orderings.

**Prediction.** `c = 0` exactly (the DoM slab is antisymmetric about the window centre so its
time-average is zero); `r1 ≈ 1` (rank-1: a linear ramp times the intensity direction).
`sae_broadcast` ≈ 0. Reaches L2. Its value is as the first half of D6.

### D6 — the level/trend double dissociation (the objection-proof design)

**The design.** Cross architecture with factor on the *same* corpus and the *same* dictionaries:

| factor | class A | class B | matched |
| --- | --- | --- | --- |
| level | high-intensity document | low-intensity document | length, topic, ordering pattern |
| trend | ascending intensity | descending intensity | multiset, mean level |

Train one SAE and one crosscoder on this corpus. Steer both factors with both architectures.
Four cells.

**Registered prediction — a crossover, not a main effect.**

- Level cell: `sae_broadcast` ≥ `txc_slab`. A constant write is *exactly* matched to a level
  target, and the crosscoder pays for its slab with worse reconstruction (last sprint: FVU
  ratio 1.2–2.7× at matched realised sparsity). **The SAE should win this cell**, and if it
  does not, something is wrong with the crosscoder-side normalisation, not with the theory.
- Trend cell: `txc_slab` ≫ `sae_broadcast ≈ 0`, by the `ker D` identity.
- Interaction significant at the same injected norm and the same dictionaries.

**Why this is the best control in the document.** Every "the crosscoder just writes better /
covers more slots / has a larger projection / benefits from the norm envelope" objection
predicts a **main effect** of architecture. None of them predicts a *crossover* in which the
SAE wins one cell outright. A significant interaction rules out the entire class in one
experiment, and it costs one extra factor on a corpus that has to be built anyway for D5.

**Cost.** One corpus, two dictionaries, four steering evaluations. Highest information per GPU
hour in this document.

### D7 — sandbagging as a trend target

**Classes.** Responses whose competence declines across the window versus rises, matched on
final answer and on total content. The trend instance of P3 with a safety-relevant label.

**Prediction.** Structurally identical to D5 (rank 1, `c = 0`), so the same L2 result. Its
value is entirely relevance; run it only if D5/D6 land and there is time. **Caveat:** matching
"competence" across two orderings without a judge is much harder than matching lexical
content, and a teacher-forced margin between competence-ascending and competence-descending
orderings may be dominated by fluency rather than by competence. Treat the metric as unproven.

### D8 — change-count, regular versus irregular spacing

Carried over from the previous sprint's amendment A10, restated because it was never run.
Hold balance *and* change-point count fixed and vary only the arrangement. Registered there:
pooled SAE codes > 0.85 AUC on fast-versus-slow alternation and ≈ 0.5 on
regular-versus-irregular.

**My prediction is that the reading half fails** for the P7 reason — the causal model's
history-writing gives pooled codes access to arrangement, as it did every previous time. The
steering half is a rotation-family design and is subsumed by D1, which matches more tightly
and yields a law. **Recommendation: do not run D8**; it is D1 with looser matching.

## Ranking

Scored as (probability the effect is real and survives its own controls) × (relevance to a
behaviour someone would actually want to steer). A construct that isolates a mechanism scores
low on relevance by definition, however clean it is.

**This ranking was revised after establishing that the tSAE decodes through a single
per-latent direction and that a profile-steered SAE reaches rank-1 writes.** Every rank-1 task
is now predicted to *tie* against the strongest baselines, which demotes the two-block designs
from "wins" to "wins only against a weak protocol". `m ≥ 3` is the only place a genuine
three-way win is available.

| rank | design | P(real) | relevance | beats profile-SAE / tSAE? | why |
| --- | --- | --- | --- | --- | --- |
| 1 | **D2b three-part refusal rotation** | 0.55 | **high** | **yes** | the only design that is rank ≥ 2 *and* about a behaviour anyone wants to steer; inherits D1's algebra on real content. Gated on `block_geometry` showing the three clause types are not collinear |
| 2 | **D1 rotation ladder, `m ≥ 3`** | 0.65 | low | **yes** | reaches L3 and yields a *law* (`sqrt(r1)`) rather than a comparison; the mechanism result that makes D2b interpretable, and the fallback if D2b's clauses turn out collinear |
| 3 | **D6 level/trend dissociation** | 0.75 | medium | no (trend cell is rank 1) | still the best control design — a crossover kills the "TXC just writes better" class outright — but its win is over constant writes, not over per-token dictionaries as such |
| 4 | **D3 three-phase reasoning** | 0.45 | medium-high | yes, if the modes are equidistant | D1's rank-2 guarantee on content anyone recognises; the whole bet is `block_geometry` on explore/commit/verify, so measure before spending compute |
| 5 | D2 refusal onset | 0.55 | high | no | highest relevance and needs no judge, but `m = 2` — the win is on discovery and on selectivity (P8), and must be framed that way |
| 6 | D4 onset phase shift | 0.65 | medium | no | cleanest demonstration of the argmax identity; L2 only |
| 7 | D5 trend alone | 0.70 | low | no | subsumed by D6; run only as D6's first cell |
| 8 | D7 sandbagging | 0.30 | high | no | the metric is unproven without a judge |
| 9 | D8 change-count | 0.15 | low | no | predicted to fail on the reading half and subsumed on the steering half |

Ranked separately because it buys a different currency:

| design | P(real) | relevance | beats profile-SAE / tSAE? | why it is not on the list above |
| --- | --- | --- | --- | --- |
| **D9 induction** | 0.70 | high | **no** (`r1 ≈ 0.90`) | the only design whose foil is the *literature's own* control, so it pre-empts "you built the task to win" — a virtue none of D1/D2b/D6 has. But it is rank 1, so it can only ever be a discovery claim. Screen it before building it |
| phase ladder | 0.70 | low | **no** (rank 1 at every rung) | a mechanism experiment about the constant-write baseline and the `T`-versus-period scope limit, not a candidate for a win |

**Recommended order.**

1. **`sae_profile_target` on the existing order data.** One arm, data already on disk, and it
   is the strongest available attack on the result the sprint is built on. Everything below is
   conditional on what it returns.
2. **The gradient screen** (`Ḡ`, `c`, `r1`, `oracle_const`) on the same data. Training-free.
3. **D1 grouped ladder at `m ∈ {2, 3, 6}`** with the full arm set including `tsae_slab`. This
   establishes the law on a construct where the algebra is known exactly, which is what makes
   the next step interpretable.
4. **D2b**, the three-part refusal rotation — the headline candidate. Run `block_geometry` on
   its three clause centroids *first*; if `pairwise_cv` is large or decline/alternative are
   near-collinear, fix the wording or fall back to D3 before spending training compute.
5. **D6**, for the crossover.
6. D2 and D3 if time remains; D1 at `m = 12` only if the effect is still measurable at
   `m = 6`.

## Predicted failures

Negative predictions, stated now so they count.

- **Every reading comparison.** Settled last sprint. Report the AUCs for the record and spend
  no compute optimising them.
- **P7 in any form.** Matching label marginals does not match activation marginals under a
  causal model.
- **D6's level cell for the crosscoder.** The SAE should win it. If the crosscoder wins every
  cell, suspect the injected-norm matching before believing the result.
- **`sae_enveloped`, `sae_profile_target` and `tsae_slab` all ≈ `txc_slab` on every two-block
  design** (D2, D4, D5, and the `m = 2` rung — including last sprint's headline task).
  Two-block swaps are rank 1 and cannot separate a slab from a schedule. Any write-up claiming
  an expressiveness advantage from a two-block task is claiming something the algebra forbids.
- **The tSAE will not be beaten on any rank-1 task**, and this is now a prediction rather than
  a missing arm. Its decoder is one direction per latent
  (`han_tsae/saeTemporal.py:50`), so it is capped at `sqrt(r1)` exactly like a scheduled SAE —
  but it *supplies* the schedule, so it is the strongest baseline available and should be run
  as one wherever it can be calibrated. Carried debt 1 is still real: at
  `lam = 1/(4·d_in)` the sparsity coefficient needs to be ~1–10, not 1e-3. If it has not been
  calibrated by write-up time, report the arm as absent rather than as dense — but the
  *prediction* about it stands on the architecture alone and can be stated regardless.
- **Free-generation versions of any of these.** The slab is indexed by position; under
  sampling the segment boundaries move and the write desynchronises from the semantics. Expect
  a large shrinkage relative to teacher-forced Δmargin. If generation is attempted, apply the
  slab in *segment* coordinates with online sentence-boundary detection, and report the
  teacher-forced number alongside so the shrinkage is visible.

## Measurement notes

Four things that decide whether the registered numbers are testable at all.

- **Stay in the linear regime, and demonstrate that you are.** Every ratio prediction is a
  linearisation. Sweep the dose and report `Δ(α)/α`; quote the law at the largest `α` where
  that ratio is still within 10% of its small-`α` limit. Last sprint's alpha grid
  (`0.25, 0.5, 1.0, 2.0`) saturates at the top end, and "each arm at its own best dose" —
  the existing `at_best` convention — is exactly the wrong selection rule for testing a
  ratio, because it picks each arm's saturation point.
- **Use paired deltas.** All arms run on the same documents, so the comparison is a paired
  difference and its standard error is much smaller than the unpaired one. Last sprint's
  `txc_slab` was `+11.29 ± 0.64` at `n = 200`; the predicted `txc_rank1` at `m = 3` is
  `0.707 × 11.29 ≈ 7.98`, a gap of 3.3 that is comfortably resolvable paired, and marginal if
  someone compares two independent means.
- **One foil, not an average.** At `m ≥ 3` there are `m − 1` distinct rotations available as
  foils. The spectrum derived above is for the **single** foil "rotate by one", whose
  difference rows are `b_t − b_{t+1}`. Averaging the DoM or the gradient over several
  rotations changes the object being screened. Rotation-by-`j` has symbol `1 − ω^j` and gives
  the same multiset of singular values, so any single `j` reproduces the same `r1` — a free
  robustness check, but keep the foils as separate conditions rather than pooling them.
- **Report realised L0 and per-segment FVU for both dictionaries in every run.** Carried debt
  3 from the kickoff; the failure is silent and it has already invalidated comparisons in this
  project once.

## Scope limits worth stating in the write-up

- **The advantage is bounded to structure at scales ≤ `T`.** Nothing about a window code helps
  with a dependency longer than the window, and the previous sprint measured that
  reconstruction degrades as `T` grows. There is an operating regime, not a monotone
  improvement, and the write-up should name it.
- **The crosscoder pays for the slab in reconstruction** — 1.2× to 2.7× worse FVU at matched
  realised coefficients per segment. Any steering claim should be reported next to that price.
- **Alignment is doing work.** The harness writes over segment token spans, so the window is
  aligned to the behaviour by construction. That is a legitimate design choice, and it is also
  a load-bearing assumption that the write-up should not leave implicit.
- **Selection hygiene.** Latents are chosen by AUC; choose them on a held-out selection split
  and report frequency-matched null draws, as the previous sprint's amendment A3 requires.

## Related

- [[start]] — sprint kickoff
- [[../2026-07-25_dictbench_10h/summary|previous sprint summary]] — what is already
  established and must not be re-derived
- [[../2026-07-25_dictbench_10h/fairness_preregistration|fairness pre-registration]] — the
  win criteria and the confound catalogue this document assumes
- [[../2026-07-25_dictbench_10h/harness_guide|harness guide]] — decoder indexing, the
  `sqrt(T)` normalisation asymmetry, and the traps
