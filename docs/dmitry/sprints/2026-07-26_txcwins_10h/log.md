---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - results
  - in-progress
---

## 22:33 — kickoff

Sprint window 2026-07-25 22:33 PDT → 2026-07-26 08:33 PDT, branch `dmitry-txcwins-10h`.

Goal: find further tasks where the temporal crosscoder beats a TopK SAE and a tSAE, ideally
ones corresponding to documented model behaviours rather than constructs.

The procedure being repeated, from the previous sprint:

1. identify a property that structurally favours a window code;
2. build a task with that property and matched foils, so generic effects cancel;
3. test reading and steering **separately** — they came apart last time and that was the
   result;
4. run the controls that can kill it — time-averaged profile, random profile, random
   direction, supervised ceiling.

Three agents launched:

| agent | job |
| --- | --- |
| `theory` | enumerate structural properties favouring a window code; concrete task designs with registered predictions and the control that would refute each; rank by (probability real) × (relevance to a behaviour anyone wants to steer) |
| `implement` | calibrate the tSAE baseline first (carried-over debt), then build and run task designs on Modal from the `steer_order_modal.py` template |
| `review` | continuously scan the literature; catalogue candidate behaviours and model organisms for TXC vs SAE vs tSAE, prioritising ones where an organism already exists and can be obtained tonight |

`implement` starts unblocked on the tSAE calibration, which does not depend on any task
design: at the repo's documented `l1_coef=1e-3` the tSAE code is dense (2989/4096 latents
active, alive 0.999) and a 100× sweep moved realised L0 by 0.3%. The `lam = 1/(4·d_in)`
scaling in `han_tsae/saeTemporal.py` means `l1` needs to be ~1–10 at this activation scale.
Until that is fixed there is no third baseline.

Standing methodology, carried forward and non-negotiable: realised coefficients per segment
as the axis (never nominal k), stride-1 windowing, `batchtopk` without ReLU, `m.eval()`
before scoring, and every steering claim accompanied by the time-averaged / random-profile /
random-direction / supervised-ceiling controls.

## 22:50 — task 1 selected: induction

The review agent's first catalogue pass produced a decomposition worth adopting as the
sprint's selection criterion. Last sprint's win rests on two properties, and the second is
the operative one:

- **P1, factor invariance** — the two conditions have matched token multisets and differ
  only in arrangement, so every permutation-symmetric readout is at chance. Rare.
- **P2, write non-constancy** — the *optimal intervention* varies with position. Implied by
  P1 but also holds without it, and this is what actually produced the win, since a
  per-token dictionary's write has per-position spread exactly 0.
- **P3, judge-free metric** — a hard practical filter. Teacher-forced Δmargin needs no
  judge, and in a 10h window that is worth several otherwise-attractive candidates.

**Task 1 is induction / in-context copying on RRT sequences**, and the reason is the foil.
A random sequence S followed by S in order, against S followed by a *shuffled* S, is
multiset-matched by construction — and it is the induction literature's own standard
control rather than one we designed. That pre-empts the main charge against last sprint's
result, which is that the winning task was built to be won. P2 holds because inducing a copy
at position t requires referencing position t−p, and a single direction added everywhere
cannot encode an offset. The metric is teacher-forced margin on the correct continuation
token, so the existing harness transfers directly.

Registered before the run: I1 reading favours the SAE again (expected, and the reason the
claim is about steering); I2 `txc_slab` beats `sae_broadcast` with z > 2; I3 the controls
hold — `txc_flat` collapses or inverts, `random_slab` flat, `random_broadcast` at or above
`sae_broadcast`; I4 the advantage depends on the relation between period p and window length
T, which would give a window-length curve on a named circuit rather than on synthetic data.

**Anti-recommendation accepted: refusal is out.** This repo's own screen
(`docs/dmitry/reviewer_responses/temporal_benchmark_screen.md`) predicts it fails the
steering rung because the Arditi single-direction intervention near-saturates, leaving no
headroom for a window. The same reasoning demotes evaluation awareness and emergent
misalignment: both are persistent states, so the only available claim is a *reading* claim,
which is the axis last sprint retired. Recording the reasoning now rather than
rediscovering it at 3am.

**One flagged risk, being verified before it moves anything.** A search surfaced a 2026
preprint claiming a single-neuron edit fixes repetition loops (arXiv id unverified). If real
it would be the saturation failure mode again and would kill the repetition-loop entry. The
review agent flagged the id as unconfirmed, which was the right instinct; it is being checked
and will be recorded as "unverified, could not confirm" if it cannot be. An unverified
citation drives no experiment selection in either direction.

**Pulled forward at zero compute cost:** `H_txc = R4 − R3`, TXC against Stacked SAE, appears
to be computable from runs that already exist — `stacked_sae` is in the backtracking
workstream's `arch_list` — and has never been reported. Review is confirming the files are
recoverable before implement is asked to spend time on it.

## 22:55 — the rank reframe, and a correction to last sprint's headline

The theory agent has replaced the sprint's selection criterion with a sharper one, and it
carries a correction to the previous sprint that is worth stating before anything else.

**Interventions live in R^(T,d), and the operative quantity is the rank of the optimal
write**, not whether the factor is permutation-invariant:

| level | write | reachable by |
| --- | --- | --- |
| L0 | `α·1_T⊗v` | one SAE latent, one dose — deployed practice |
| L1 | `α·s⊗v` | SAE plus a hand-supplied per-position dose schedule; **rank 1** |
| L2 | rank-1 TXC slab | same expressiveness as L1; the schedule was *learned* |
| L3 | rank>1 slab | **no per-token dictionary, under any schedule** |

**The correction.** Every two-block swap has a rank-1 optimal write — its
difference-of-means slab is `[+Δ, −Δ]`. So last sprint's order task was an L0→L2 result, and
the summary's claim that the crosscoder buys "a temporally structured write that the
per-token dictionary cannot express at any budget" is too strong. An SAE handed a
per-position dose schedule reaches L1, which is the same expressiveness. What survives is
the **discovery** claim: the crosscoder found the schedule, and handing it to an SAE requires
knowing it in advance. The prior summary will be corrected and this sprint's write-up will
say so plainly. This is exactly the objection a reviewer would raise, and it is better found
here.

It also rules out a class of follow-ups: new two-block tasks — refusal-order, ramp-up vs
ramp-down — cannot produce an expressiveness claim no matter how cleanly they run.

**The SVD screen, promoted to job one.** `P_dom` is already built at
`steer_order_modal.py:226`; three numbers from it screen a task before any dictionary
trains:

```text
c    = T·‖mean_t P[t]‖² / ‖P‖_F²      constant share    (L0 reachable)
r1   = σ₁² / ‖P‖_F²                    rank-1 share      (L1/L2 reachable)
1−r1 = slab-only residual                                (L3 only)
```

`c > 0.3` → discard the task, an SAE can do it. `c≈0, r1≈1` → discovery claim only.
`c≈0, r1<0.6` → build there. Registered rank law, to be measured at the **smallest** dose
with a significant effect since last sprint's α grid was saturated at the top:
`Δ_Π/Δ_full ≈ ‖ΠP‖_F/‖P‖_F` — a square root, because Δ ≈ α⟨W,G⟩ with G ∝ P.

**Four arms the harness was missing**, now standard. The important one is `sae_enveloped` —
the SAE direction scaled per position by the crosscoder slab's own norm profile ‖P[t]‖. It
controls the likeliest spurious win, where residual norm grows with position and a slab wins
merely by learning 1/‖x_t‖. Neither `txc_flat` nor `random_slab` catches that, so last
sprint's result has an untested alternative explanation. Also added: `txc_rank1`,
`dom_rank1`, and `txc_transfer` (slab fitted on one corpus, applied to another without
refitting — the arm that turns "you could have supplied the schedule" into "you would have
had to know it"). Plus selectivity on every design: Δ_target against KL from the unsteered
model on neutral text at matched norm.

**Queue is now** SVD screen → D1 rotation ladder (m ∈ {2,3,4}) → D6 level/trend
dissociation → D2 refusal onset. Induction moves behind D1 and gets rank-screened first: its
virtue is a foil the induction literature already uses as its own control, but if its
optimal write is rank-1 it is another discovery-only claim.

D1 is the only queued design that can produce an L3 result. Its harness gate is exact: DoM
rows `b_t − b_{t+1}` sum to zero, so `c = 0` identically, and a nonzero measured `c` means
the windows are misaligned and everything downstream is suspect.

## 22:45 — the SAE baseline was handicapped, and both agents found it independently

Two agents converged on the same hole in last sprint's steering comparison from opposite
directions, which is the strongest signal available that it is real.

The crosscoder slab was compared against an SAE direction written at **all** positions.
Nothing forces an experimenter to write uniformly. There are three arms:

| arm | write | level |
| --- | --- | --- |
| S1 `sae_broadcast` | SAE direction at every position — what was run | L0 |
| S2 `sae_oracle_pos` | same direction, only at oracle-chosen positions | L1, rank 1 |
| S3 `txc_slab` | the crosscoder slab | L2/L3 |

**The claim that survives review is S3 > S2, not S3 > S1.** The review agent reached this
from the reading-side precedent — `temporal_benchmark_screen.md` already argues the reading
baseline must be the position-oracle — and the theory agent reached the same place from the
algebra, since a two-block task's optimal write is rank-1 and therefore schedulable. Both
are now mandatory arms, along with `sae_enveloped`, the continuous version that weights by
the slab's own norm profile.

The asymmetry to state plainly, neither hidden nor oversold: S2 needs external supervision
to know *which* positions, while the crosscoder's profile falls out of unsupervised
training. That is a real advantage, but an advantage in **supervision and discovery, not
representation**.

## 22:47 — verifying the rotation spectrum, and a correction to the correction

The theory agent self-corrected its own rotation-ladder algebra before anything was built on
it: `r1 = 1/(m−1)` is wrong from m=4. The correct spectrum for `P = C·B` with C circulant
(first row `1, −1, 0, …`) has symbol `f(ω) = 1 − ω`, giving `σ_j² = 4sin²(πj/m)`,
`Σσ_j² = 2m`, `σ_0² = 0`.

I reproduced it independently rather than take it on trust. Rank = m−1, `c` = 0 to 1e-33,
top-2 share exactly 1.000 at m=3, and m=4 does share m=3's r1 — so m=4 is a wasted rung and
the grid becomes m ∈ {2, 3, 6} with 12 as a stretch.

**But the measured r1 came out consistently above the closed form**, and the reason matters:

| m | closed form (orthonormal blocks) | i.i.d. Gaussian blocks | correlated ρ=0.5 | ρ=0.8 |
| --- | --- | --- | --- | --- |
| 3 | 0.500 | 0.578 | 0.575 | 0.579 |
| 6 | 0.333 | 0.368 | 0.368 | 0.371 |
| 12 | 0.167 | 0.219 | 0.220 | 0.219 |

The closed form is exact only when the block-mean matrix has orthonormal rows, and it is a
**lower bound** otherwise. The inflation is not driven by correlation between block means —
ρ = 0, 0.5 and 0.8 all land at ~0.578 for m=3 — so it cannot be designed away by choosing
semantically dissimilar block types. It always inflates, so the L3 headroom `1 − r1` is
**smaller** than the ideal algebra promises and the scheduled-SAE ceiling sits higher.

**This changes how the law gets tested, and improves it.** The sqrt law
`Δ_Π/Δ_full ≈ sqrt(energy share)` will be checked against **measured** r1 from the actual
`P_dom`, never against predicted r1. That separates two claims that would otherwise fail
together: the law is about linear response, the spectrum is about block geometry, and the
gap between measured and orthonormal r1 becomes its own readout of how far the block means
are from orthogonal.

## 22:50 — harness generalised; the tSAE's sparsity is shrinkage, not selection

*(Timestamps above corrected — two headings had been written ahead of the wall clock. Real
elapsed time is the only thing that counts in a sprint and the log should not drift.)*

**The harness is now task-agnostic**, which is the main unblocking event of the sprint so
far. `experiments/temporal_screen/txc_wins/harness.py` factors the task out to a single
function:

```python
make_pair(rng) -> (sents_a, sents_b, carrier)
```

Two equal-length sentence lists matched on everything except the property under test, plus
the carrier. Everything else is wired: caching, stride-1 windows, SAE + TXC + tSAE training,
reading AUC, teacher-forced Δ margin, every control, realised-L0 logging, SEMs and
z-separations. Roughly one design per 20 minutes. Theory has been asked to deliver D1, D6
and D2 in that form.

**Regression against last sprint's order task reproduces it**, with a drift worth tracking:
SAE pooled reading AUC 0.989 against 0.998, TXC window AUC 0.721 against 0.791, realised
8.00 coefficients/segment for both. Same story, but the crosscoder number moved 0.07 and the
cause should be identified before it propagates into a comparison.

**The tSAE calibration has produced a finding rather than a constant.** The diagnosis is
confirmed quantitatively: at the documented `l1=1e-3` the penalty contributes 0.03 against a
reconstruction term of 17.7 — 0.2% of the loss, numerically absent, which is exactly why a
100× sweep moved nothing. Extending to `l1=1e3` does move it: realised L0 goes
2998 → 3008 → 2769 → 2265 → 1698 at l1 = 1e-3, 0.1, 1, 3, 10.

But the alive fraction is **still 1.000 at L0 = 1698**. The L1 penalty is buying sparsity by
shrinking every code rather than by selecting a small support. If reaching the 1–32
coefficient band costs enough reconstruction to make the arm unusable — which is the
expectation — then the reportable statement is that **the published ReLU+L1 tSAE recipe does
not produce a sparse support at this activation scale**, and a three-way comparison against
TopK dictionaries needs either a TopK variant of the same architecture or an L1 large enough
to destroy reconstruction. A parallel `sae_diff_type="topk"` arm is training so that a
genuinely sparse temporal-attention baseline exists either way; the contrast between the two
is the result.

**Phase ladder approved.** The order task at 1, 3, 5 and 11 switches, foil built as a cyclic
rotation by one block so the classes contain literally the same sentences and differ only in
phase. It separates two rival accounts of last sprint's headline that no existing control
distinguishes: "the advantage is that the write is non-constant" against "the advantage is
about slow structure and decays with frequency". Strictly better matching than the original
task, which held multiset and switch count but not run-length multiset.

## 22:55 — screen the gradient, not the difference-of-means

The theory agent's best contribution of the sprint, and it converts a step in one experiment
into a reusable instrument.

`P_dom` and the margin gradient `Ḡ` are different objects: difference-of-means is the
direction that *distinguishes* the classes, while `Ḡ` is the direction that most *increases
the margin*. Steering cares about the second. The earlier rank law needed `G ∝ P` as an
assumption; screening on `Ḡ` makes it an identity.

Taking the mean gradient over documents — correct because a dictionary latent is one *fixed*
write reused across documents, so `E[Δ] = α⟨W, Ḡ⟩` — the best norm-matched write inside a
subspace Π is `ΠḠ/‖ΠḠ‖`, achieving `α‖ΠḠ‖`. So

```text
Δ_Π / Δ_full = ‖ΠḠ‖_F / ‖Ḡ‖_F        exact, no proportionality assumption
```

Verified numerically before adopting it: the analytic constant-subspace optimum equals
`‖Π_const Ḡ‖_F = sqrt(T)·‖mean_t Ḡ‖` = 8.1552 on a random test slab, and the ratio to the
unconstrained optimum equals `sqrt(c)` = 0.2945 to four decimals. My first check used random
search over directions and found only 3.88 — random search in 64 dimensions never finds the
optimum, so the check was wrong, not the theory. Worth recording, because a bad verification
that *appears* to refute a correct claim is more dangerous than no verification.

Note the factor of T: `c = T‖mean_t Ḡ‖²/‖Ḡ‖²`, since the constant-subspace projection is
`1_T ⊗ mean_t Ḡ` with squared norm `T‖mean_t Ḡ‖²`. Dropping it under-reports the SAE ceiling
by exactly T and is the easiest available way to manufacture a false win.

**Three oracle arms follow, and one of them is the most valuable control in the sprint.**
`oracle_const = Π_const Ḡ` is the best constant write that *exists*. If it gives Δ ≈ 0 then
no SAE latent — better trained, better selected, hand-picked — could ever have done the
task. That answers "you just picked a bad latent" before a reviewer raises it. `oracle_rank1`
pins the L1 ceiling free of training noise, and `oracle_slab = Ḡ` supersedes `dom_slab` as
the ceiling, directly measuring the 7× supervised gap left open last sprint.

## 23:00 — the best target is a documented failure, not a construct

The review agent's third pass supersedes its own ranking and, I think, settles what this
sprint should be about.

**Prompt-permutation sensitivity is a real, documented, widely-cared-about family of model
failures whose defining factor is permutation at matched multiset** — the same property that
won last sprint, but *pre-legitimised*, because the permutation set is the published
evaluation protocol rather than a foil we designed:

| instance | source | reported effect |
| --- | --- | --- |
| few-shot demonstration order | Lu et al., ACL 2022 (arXiv:2104.08786) | "the difference between near state-of-the-art and random guess performance", across model sizes |
| multiple-choice option order | Pezeshkpour & Hruschka (arXiv:2308.11483) | 13–75% performance gap on reordering options |
| retrieved-document position | Liu et al., TACL (arXiv:2307.03172) | degrades when relevant information sits in the middle |

This dominates the synthetic rotation ladder because it is **the same algebra with real
semantics**: m permuted answer options is D1 with m blocks, so every registered rank
prediction carries over, but the metric is a pure logit margin between option labels and the
claim is one an outsider cares about — *a learned temporal profile can remove a model's
prompt-order sensitivity where a single steering direction cannot*. That is a result about
evaluation validity.

The P2 argument is also sharper here than in the original task, and provable rather than
empirical: removing a prior that favours the first slot requires **suppressing that slot and
boosting the last**, and a single direction added everywhere shifts all slots equally, so it
cannot change the relative prior at all.

**A harness fact that reorders the list.** `steer_order_modal.py` segments by span and
mean-pools within each span rather than windowing over tokens, so any task whose natural
unit is a *block* — an option, a demonstration, a retrieved document, an instruction, a
conversational turn — is a drop-in at T = 4–12, the range already validated. Only the corpus
builder changes. That promotes the permutation family to a drop-in and demotes induction,
whose natural unit is a token at a ~50-token offset and needs the sequence chunked first.

**Gate before any training:** confirm the order effect exists in Qwen2.5-1.5B-Instruct at
all. A few hundred permuted MCQ items, measuring the label-logit swing, pure forward passes.
If the 13–75% literature effect does not reproduce at 1.5B, switch to demonstration order
(the most robust instance) rather than scaling up.

## 22:52 — premise order / R-GSM becomes the target; a timekeeping correction

**Timekeeping first.** Two entries above were written as though more time had passed than
had. True elapsed at this point is **19 minutes** of the 10 hours, not the hour-plus implied.
The agents are working fast, which is easy to mistake for time passing. Recorded because a
sprint's one non-negotiable is the wall clock.

**Target changed from MCQ option order to premise order.** Chen, Chi, Wang, Zhou, *Premise
Order Matters in Reasoning with Large Language Models*, ICML 2024 (arXiv:2402.08939,
verified): permuting premise order causes "a performance drop of over 30%", performance is
best when premise order matches the ground-truth proof order, and they ship **R-GSM**, a
public GSM8K-derived benchmark built from premise reorderings.

| | MCQ option order | premise order / R-GSM |
| --- | --- | --- |
| corpus | must be built | **public, already exists** |
| unit → segment | option | premise → `k_seg` directly |
| failure type | label prior | **reasoning** |
| published localisation | MLP value vectors + attention heads | **none** |

The third row is the decisive one. Li & Gao (*Anchored Answers*, arXiv:2405.03205) trace
MCQ first-choice bias to specific MLP value vectors and attention heads, and UniBias
eliminates biased FFN vectors and heads — the same shape as the single-neuron repetition
result, and it invites "so what does a window code add?". Premise order has no published
localisation, so that objection is simply unavailable. Two mitigations exist for MCQ (the
localisation is on the GPT-2 family, which the authors say has worse anchoring than larger
models; and those are weight edits rather than additive steering writes), but choosing a
different instance is cleaner than arguing.

**The gate is the real risk and it comes before any training:** at 1.5B, GSM8K accuracy may
be too low for a 30% relative permutation gap to be measurable. Run ordered against permuted
R-GSM over a few hundred items and report the gap; if it is not clearly present, move to
Qwen2.5-7B-Instruct rather than persisting. Fallback if neither clears: few-shot
demonstration order, which Lu et al. report persists "even for the largest current models".

**Deliberately checked whitespace:** a search for sparse-autoencoder or dictionary work on
demonstration-order or prompt-order sensitivity returned nothing. Plenty of SAE-steering
methodology, plenty of order-sensitivity work, no intersection. Whatever this measures is
unclaimed — which argues for doing it carefully rather than quickly.

## 22:56 — a generator bug that would have faked a win, and a check that last sprint is clean

The theory agent found a bug in its own corpus generator before anything was run on it, and
it is the kind that produces a false positive rather than a null.

**The bug.** Building class A and class B by calling a document generator twice with
different `shift` gives the two classes *different sentences* — independent draws match
register counts only in expectation. Measured at m=2 on one seed: class A came out
legal:4, tense:3, calm:2 against class B's calm:4, culinary:3, legal:2. That is a lexical
imbalance pointing straight at the factor under test, and **a constant write can exploit
it**. It would have produced a false positive for `sae_broadcast` at exactly the rung where
zero was registered — and would have read as the rank theory failing when the fault was the
generator.

The fix is to draw once and rotate the assembled list, `sents_b = sents[rot:] + sents[:rot]`,
sharing the carrier prefix across the pair, so the two classes are literally the same string
read from different starting points. `rotation_pair` now self-tests `multiset_equal` and
`is_rotation_by_{block_len}` at every m.

**Checked whether last sprint's headline has the same flaw. It does not.** The steering
pairs at `steer_order_modal.py:277-278` reuse `ts_, cs_, car` from a single draw, so A and B
contain literally the same sentences and carrier; and `make_doc` always draws 6 tense + 6
calm regardless of class, so the training set carries no systematic lexical imbalance for
latent selection to pick up either. Worth confirming rather than assuming — had it been
affected, every steering number from that sprint would have needed rerunning.

**A second, smaller design fix: m is confounded with coherence in the naive ladder.** The
number of registers grows with m, so a document at m=2 reads as a narrative and at m=6 as a
collage of six unrelated registers — meaning any trend across the ladder is partly a trend
in distance from the model's distribution, inflating margin variance at exactly the rungs
the theory cares about. The grouped ladder holds six registers fixed and lets m set only the
grouping (3 registers per block at m=2, 2 at m=3, 1 at m=6), so every document contains the
same twelve segments and only block structure moves. Group means are averages of subsets and
therefore less mutually equidistant than single registers, so measured r1 should sit above
the closed form — `block_geometry()` is reported alongside so the deviation is visible.

The agent also retracted its own earlier claim that the absolute effect "will be small" at
m=12, noting it was a guess stated as a derivation. The defensible version is about variance
and distributional shift, not effect size.

## 22:58 — the tSAE is identified, and the baseline debt was an artefact of the wrong architecture

**Bhalla, Oesterling, Verdun, Lakkaraju, Calmon, *Temporal Sparse Autoencoders: Leveraging
the Sequential Nature of Language for Interpretability*, ICLR 2026 oral (arXiv:2511.05541).**
It is an InfoNCE penalty over adjacent positions with no attention — exactly as described to
me at the start of the sprint. This repo's attention-based `TemporalSAE` (`tsae_paper`) is a
different architecture and was never the right baseline, which is why that arm behaved badly
across two sprints.

The carried-over debt therefore **dissolves rather than gets paid**. T-SAE uses BatchTopK at
k=20 — the same sparsity rule the crosscoder now defaults to — so sparsity is matched by
construction and there is no `l1_coef` to calibrate. The calibration sweep was compute spent
on an architecture nobody asked for; it has been stopped. What survives is a short note that
ReLU+L1 on the attention model buys sparsity by shrinking every code rather than selecting a
support, which is a true if minor negative about that architecture.

T-SAE also claims a *steering* Pareto improvement over baseline SAEs, so it is a live
competitor on this sprint's own axis rather than a reconstruction baseline.

## 23:00 — the design becomes a capacity ladder

A second paper closes the remaining baseline gap and reframes the experiment. *Persistent
Sparse Autoencoders* (arXiv:2607.17117) gives each feature a learned persistence coefficient
and a diagonal recurrence over positions, `h_t = λ_j·h_{t-1} + (1−λ_j)·a_t` with
`λ_j = σ(l_j)·0.999` — a per-feature leaky integrator, roughly ten lines, sitting strictly
between a per-token SAE and the crosscoder's free slab.

That is the obvious reviewer question made concrete: the crosscoder spends `T×d` parameters
per latent where PSAE spends **one**. If a full slab cannot beat an EMA, the case for the
slab is weak. So the head-to-head is replaced by a ladder in which each rung adds exactly one
thing:

| rung | model | temporal capacity per latent |
| --- | --- | --- |
| 1 | TopK / BatchTopK SAE | none — the per-token floor |
| 2 | Persistent SAE | one scalar: a learned timescale |
| 3 | T-SAE | InfoNCE coupling to adjacent positions |
| 4 | TemporalCrosscoder | free `(T,d)` slab |

**Wherever the curve flattens is informative regardless of direction.** "The crosscoder beats
an EMA and an InfoNCE tSAE on an order-at-matched-multiset task" is much harder to dismiss
than beating a plain SAE, and early flattening is a finding rather than a failed sprint.

**Prior art that must be cited rather than rediscovered.** The PSAE authors already ran
causal interventions on prompt injection at range — probes up to 350 tokens past an
untrusted span, fast features falling to chance while slow features stay informative with "a
consistently positive intervention effect at every distance". So "temporal structure helps
steering at range" is partly claimed. The differentiators survive but have to be stated:
their axis is **distance**, ours is **order at matched multiset**; they compare timescales
*within* a per-token code, we compare against a full window code; and they did not evaluate
crosscoders at all. The whitespace the review agent found earlier — no dictionary work on
prompt-order sensitivity — is narrower than it first looked, and the write-up should say so.

## 23:00 — the theoretical spine: every per-token dictionary is rank-1 by its decoder

Read the decoders rather than argued about them:

| architecture | decoder | write rank per latent |
| --- | --- | --- |
| TopK SAE | `W_dec = (d_in, d_sae)` | 1 |
| Persistent SAE | leaky integrator → standard decoder | 1 |
| T-SAE (attention or InfoNCE) | `D = (width, dimin)` — **no position axis** | 1 |
| TemporalCrosscoder | `W_dec = (d_sae, T, d_in)` | up to T |

> **Every per-token dictionary has exactly one decoder direction per latent, so its write is
> rank 1 however sophisticated its temporal encoder is. Only a window dictionary produces a
> rank > 1 write.**

In the tSAE all the temporal machinery is in the *encoder* — attention over causal context
changes which coefficients fire and when, but never gives a latent a second write direction.
So the whole capacity ladder's rungs 1–3 sit at L1 and differ only in **how the rank-1
schedule is obtained** (fixed, learned scalar timescale, learned by contrastive encoder), not
in rank. Registered consequence: at m=2 all four rungs tie; at m ≥ 3 only the crosscoder
exceeds `sqrt(r1)`.

## 23:02 — the strongest attack on last sprint's headline, and it is ours to run

> A per-token dictionary's write is constrained to one **direction**. Whether it is constant
> **in time** is a property of the *steering protocol*, not of the architecture.

Last sprint's `sae_broadcast` — one direction, one dose, every position — is the **weakest**
form of the SAE baseline. The protocol practitioners actually use scales a latent by its own
activation, `α·z_j(x_t)·v_j`, which is position-varying, data-dependent, and rank-1. Under
that protocol a plain TopK SAE reaches rank-1 slabs and **every rank-1 task ties**.

Two arms settle it, both on data already on disk:

- **`sae_profile_self`** — coefficient from the latent's activation on the current document.
  Predicted ≈ 0 on rotation tasks, since amplifying what a document already contains raises
  logP of A and B alike. **This is the one that matters**: it needs no supervision, so if it
  works the discovery claim dies with the expressiveness claim.
- **`sae_profile_target`** — coefficient from the latent's mean profile over class-A
  documents, applied as a fixed schedule. The real L1 baseline. Predicted to close the gap
  to `txc_slab` to within noise at m=2.

If `sae_profile_target` ties, last sprint's headline needs qualifying — the advantage was
over a weak *protocol* rather than over a per-token dictionary — and the sprint pivots to
m ≥ 3, where the rank argument still bites. Better found here than by a reviewer.

## 23:04 — the tSAE arm is unusable, and the reason is architectural

The calibration is complete and negative. There is no usable `l1`: the coefficient does
control sparsity once it is five to eight orders of magnitude above the documented value, but
the dictionary dies before it becomes sparse.

| l1 | coeff/segment | FVU | alive |
| --- | --- | --- | --- |
| 1e-3 (documented) | 2998 | 0.036 | 1.000 |
| 10 | 1698 | 0.058 | 1.000 |
| 100 | 151 | 0.318 | 0.999 |
| 170 | 29 | **1.030** | 1.000 |
| 1e6 | 0.7 | 1.221 | 0.002 |

FVU > 1 is worse than predicting the mean. The `l1` range placing realised L0 in the 1–32
band is roughly [225, 1.9e5] and every point in it is a dead dictionary; the last usable
point is l1≈100 at 151 coefficients/segment with FVU 0.32, against a per-token TopK SAE's
0.098 at 8 coefficients/segment on the identical cache.

**The mechanism is why this is not a tuning miss.** `lam = 1/(4·d_in)` puts codes at ~4e-3
against a reconstruction term of ~18, which is the flat stretch. But `TemporalSAE` has **no
encoder bias**, so a code can only be zero when `x·D_j < 0` — sparsity has to come from the
geometry of the dictionary rather than from a threshold. L1 therefore shrinks the whole code
vector: alive fraction is still 0.998 at 67 coefficients/segment while the share of code mass
in the top 32 latents climbs 0.036 → 0.90. Sparsity by shrinkage, not by selection. Both
readings of the loss (novel+pred, and novel only) fail the same way, so the identification
ambiguity does not rescue it. The TopK variant of the same architecture is used instead and
binds by construction.

## 23:06 — the phase ladder qualifies last sprint's finding 2

Same task at 1, 3, 5 and 11 switches, foil built by cyclic rotation so the classes contain
literally the same sentences and differ only in phase. Best single-latent reading AUC:

| switches | SAE (pooled) | TXC (window) | tSAE (pooled) |
| --- | --- | --- | --- |
| 1 | **0.997** | 0.746 | 0.628 |
| 3 | 0.727 | 0.709 | 0.639 |
| 5 | 0.722 | 0.716 | 0.598 |
| 11 | 0.631 | **0.704** | 0.616 |

**At 11 switches the crosscoder reads better than the SAE.** That qualifies last sprint's
finding 2: "reading comparisons never favour a window code" holds for *slow* structure and
fails for *fast* structure. The mechanism is the same one that made the negative true in the
first place — a pooled per-token code recovers order through the causal history smeared into
each token — but smearing cannot resolve alternation at period 2.

Steering, best Δmargin at matched injected norm and matched realised coefficients
(8.00/segment for all three arms):

| switches | `sae_broadcast` | `txc_slab` | `txc_flat` | `tsae_broadcast` | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | +1.37 | +4.99 | −12.06 | +3.10 | win |
| 3 | −0.57 | +7.50 | −25.78 | +1.45 | win |
| 5 | +0.53 | +1.56 | +1.20 | −0.34 | **not established** |
| 11 | −0.04 | +7.80 | −3.54 | −0.18 | win |

Three of four cells are wins with the profile control holding. Phase-5 is not: `txc_slab`
+1.56 ± 0.47 against `txc_flat` +1.20 ± 0.75, so the control does not separate there. The
cell stays in the table with the hole visible.

## 23:08 — two headline candidates, decided by instrument rather than argument

A third design closes the gap that had been bothering both agents: **rank ≥ 2 and relevance
had never co-occurred.** D1 and D3 reach L3 but are constructs; D2 (refusal onset) is
relevant but m=2 and therefore rank-1, so it ties against a profile-SAE and the tSAE.

**D2b — three-part refusal rotation — is both.** A well-formed refusal has three parts, not
two: acknowledge the request, decline with a reason, offer an alternative. That is the shape
assistant guidelines actually ask for, and three semantically distinct modes make a cyclic
rotation an **m=3 design, rank exactly 2**, on a behaviour someone genuinely wants to steer.

```text
class A:  acknowledge / decline / alternative
class B:  decline / alternative / acknowledge
```

Registered: `c = 0` exactly, `r1 ≈ 0.50`, `txc_rank2` recovers ~100%, and every rank-1
baseline — `sae_profile_target`, `sae_enveloped`, `tsae_slab` — capped at ≈0.71 of
`txc_slab`. Its own gate comes first and is cheap: **decline and alternative both carry
refusal-adjacent content**, so if their centroids are near-collinear the rank collapses
toward 1 and D2b is an m=2 task in an m=3 costume. `block_geometry()` on the three clause
centroids is the test, and the alternatives have been worded as "I can walk through X"
rather than apologetically, specifically to push them off the decline direction — but that is
a guess and the geometry decides.

**How the headline task gets chosen.** Two candidates now exist — R-GSM premise order
(published, 30% documented gap, no corpus to build, but unknown rank and an unverified effect
at 1.5B) and D2b (constructed, but rank known exactly and on a behaviour people steer).
Rather than argue, **the gradient screen decides**: it is training-free, one backward pass
per document, and it was built for exactly this. Screen the order task (calibration), D2b,
R-GSM and the D1 rungs; pick by measured `1 − r1`, largest wins, with `c > 0.3` as an
outright disqualification since a plain SAE can do such a task.

That makes the choice reportable rather than a judgement call, which is the whole reason the
instrument was worth building.

**D1 stays ahead of the headline regardless of which wins.** Its relevance is low but it
calibrates the law on a construct where `r1` is known in closed form; without it, a measured
0.71 on D2b has no reference for whether 0.71 is the right number.

## 23:10 — delegate the T-SAE to the reference implementation

Bhalla et al. release code, trained T-SAEs and interpreted latents at
`github.com/AI4LIFE-GROUP/temporal-saes`. This repo's `CLAUDE.md` carries the rule from the
2026-05-07 EM replication — *delegate to the reference's function, don't reimplement the
prep* — after that session lost a day to a reimplemented prep that silently dropped chat
templating. **The tSAE arm has already failed once this project on an architecture
mismatch**, so the loss is being matched line-by-line against theirs rather than built from
the paper description.

One open question that changes what the ladder means, checkable in minutes: **the paper does
not say whether their steering is applied uniformly across positions.** If it is, the T-SAE
is another constant-write baseline rather than a genuine middle rung, and the S1/S2/S3
fairness distinction applies to it too — in which case "T-SAE ties with SAE" would mean the
encoder difference does not survive a constant write, which is a different claim entirely.

Also recorded for the write-up: their steering evaluation is judge-based (30 features graded
by Llama-3.3-70B for success and coherence). The teacher-forced margin used here is a
cleaner, non-overlapping measurement rather than a weaker one, and should be described that
way.

## 23:05 — the catalogue's qualitative screen and the gradient screen are the same quantity

MCQ option order is dropped, on a mechanism that turns out to generalise. Zheng et al.,
*Large Language Models Are Not Robust Multiple Choice Selectors*, ICLR 2024 Spotlight
(arXiv:2309.03882) attribute selection bias primarily to **token bias** — the model "a priori
assigns more probabilistic mass to specific option ID tokens". If the factor is a prior over
*label tokens* rather than over *positions*, then a constant write on a "prefer token A"
direction is exactly the right intervention, so the per-token dictionary should win or tie.
That makes MCQ a **predicted negative with a mechanism**, which is worth more than an
untested candidate. The literature is split on the attribution — Pezeshkpour & Hruschka
emphasise position and answer uncertainty — but an unsettled attribution is itself a reason
not to spend the sprint on it, and it stacks with the anchored-bias localisation risk noted
earlier.

**The pattern this completes is the important part.** Four catalogue entries have now been
demoted for the same reason:

| entry | why there is no headroom |
| --- | --- |
| refusal | the Arditi single-direction intervention near-saturates |
| MCQ option order | selection bias is a prior over label tokens, so a constant write is correct |
| emergent misalignment | a transferable direction already captures enough to steer with |
| repetition onset | a single sign-inverted neuron already fixes it |

In every case the judgement is: *a constant, per-token-representable write already captures
the factor, so nothing is left for a position-varying write to do.* That is precisely what
the gradient screen measures —

```text
c = T·‖mean_t Ḡ‖² / ‖Ḡ‖²_F        constant-subspace share of the margin gradient
```

— with `c > 0.3` as the discard threshold. The literature-derived judgement and the measured
`c` are the same quantity, one estimated from prior work and one obtained in a single
backward pass. The catalogue is being given a predicted-`c` column so its rankings become
falsifiable rather than editorial, and every task actually screened yields a measured `c` to
check the prediction against. If literature-derived predictions track measured `c` across
several tasks, **the catalogue can screen candidate behaviours without running anything**,
which would be more useful than any single task outcome.

Instances A (demonstration order) and D (premise order) survive: their permuted units are
*content* blocks with no label token available to carry a prior, so the token-bias escape
route does not exist and the factor is genuinely positional.

**Two corrections recorded.** The review agent withdrew its own characterisation of
*Convergent Linear Representations of Emergent Misalignment* (arXiv:2506.11618) — the paper's
minimal organism uses nine rank-1 adapters and is explicitly not a single-unified-direction
claim, so the earlier phrasing would have been repeated here uncorrected. And it found its
catalogue timestamps running about four hours fast for the same reason mine did: estimated
rather than read off the clock. Both corrected. Elapsed is roughly half an hour of ten, so
the runway is far longer than the pace suggests, and the agents have been redirected from
breadth toward depth and verification.

## 23:10 — reading and steering are the same number

The best theoretical result of either sprint, and it arrived as the answer to a
commit-before-measure question rather than as a fit to data.

Write `τ(ctx)`, `κ(ctx)` for a tense or calm segment's representation under context `ctx`. In
class A (tense-first) the early positions are `τ(T)` and the late ones `κ(T)`; in class B the
early are `κ(C)` and the late `τ(C)`. So

```text
P_early = τ(T) − κ(C)
P_late  = κ(T) − τ(C)
Σ_t P[t] ∝ [τ(T) + κ(T)] − [τ(C) + κ(C)]
```

Context-free representations make that sum zero, giving `P = [+Δ, −Δ]` and rank 1. **With
causal history it is the tense-prefix-versus-calm-prefix contrast, which is generically
nonzero** — so the two block rows are not antipodal, `mean_t P[t] ≠ 0`, and both `c > 0` and
`rank > 1` follow. Which yields:

> **`c = 0` if and only if a linear pooled per-token probe is at chance. A task is
> steerable-by-constant-write exactly to the extent that it is readable-by-pooling.**

This retrodicts last sprint's data. The pooled SAE read the order label at AUC 0.998, which
implies `c > 0`, which implies `sae_broadcast > 0` — and `sae_broadcast` measured **+1.24**.
The reading result and the steering result were the same fact, and neither sprint knew it.
It also explains why the reading comparisons kept failing to favour a window code while the
steering comparison succeeded: those are two views of one quantity, and the task's position
on that single axis determines both.

**Predictions on record before any number lands:** `c ≈ 0.03` (0.005–0.12) and `r1 ≈ 0.93`
(0.82–0.99) on the order task, hence `Δ(sae_profile_target)/Δ(txc_slab) ≈ 0.96` — closing
most but not all of the gap — and `1 − r1 ∈ [0.02, 0.15]` bounding the genuinely-L3 fraction
of last sprint's result. So the correction to the prior summary stands and should not be
softened: **mostly discovery, with a small genuinely-inexpressible residual.** Falsifier:
measured `c < 1e-10`, which would refute the causal-history argument and mean representations
are effectively context-free at this layer — surprising and reportable on its own.

**A gate I had already issued, withdrawn before it did damage.** I passed on "`c = 0` exactly
at every m — treat nonzero as window misalignment" as a harness assertion. That is the block
algebra, which assumes context-free representations; measured `c` will be small-but-positive
everywhere and its magnitude *is* the causal-history contribution. Restated as **`c < 0.1`**,
with a note not to hunt a misalignment bug on the strength of a small positive value. Caught
by the theory agent flagging its own earlier instruction rather than by anyone hitting the
false alarm.

**Induction: predicted `r1 ≈ 0.90`, effectively rank 1** — in the first half nothing
distinguishes the classes yet so `P[t] ≈ 0`, and in the second half the slab is a step or ramp
times one induction direction. It therefore buys *credibility of setup* (its foil is the
literature's own control, pre-empting "you built the task to win") rather than expressiveness.
Both currencies are worth having and the write-up must say which one each experiment bought.
It is being screened rather than built: one backward pass per document over ~100 sequences
resolves it before any corpus exists.

**A free lever on headroom.** Since the `r1` inflation is a non-orthonormality effect, the
block vocabulary controls it. The current pools pair calm with tense — two poles of one
affective axis, whose difference is a single dominant direction that mechanically inflates σ₁
and shrinks exactly the L3 headroom being measured. Topic-orthogonal pools should push
measured `r1` down toward the bound.

## 23:12 — instruction recency: the crosscoder reverses the tie-break

The strongest cell of the sprint, and on a documented behaviour rather than a construct.
Twelve segments of system-note filler with two conflicting instructions at fixed positions 2
and 9; the classes are exact reorderings — same filler, same instructions, same positions,
only which instruction comes early is swapped. Metric is the **difference of differences** of
`logP(obey instr 1) − logP(obey instr 2)` between the orderings, so a write that merely adds
"more instruction 1" cancels exactly.

**Baseline: −2.42 ± 0.21 over 80 documents** — each ordering follows its *later* instruction.
That recency preference is the mechanism a prompt injection exploits.

| arm | α=−2 | α=−1 | α=−0.5 | α=+0.5 | α=+1 | α=+2 |
| --- | --- | --- | --- | --- | --- | --- |
| `txc_slab` | −0.81 | −3.38 | −2.39 | +3.52 | **+7.09** | +6.88 |
| `dom_slab` (supervised) | +4.71 | −0.17 | −3.41 | +6.51 | +8.19 | +2.40 |
| `sae_broadcast` | +1.86 | −0.07 | −0.28 | +0.31 | +0.67 | +1.33 |
| `txc_flat` | +1.78 | +0.29 | −0.08 | +0.28 | +0.58 | +1.01 |
| `tsae_broadcast` | +1.11 | −0.34 | −0.31 | +0.27 | +0.51 | +0.71 |
| `random_slab` | +0.74 | +1.39 | +1.00 | −1.48 | −2.79 | −2.47 |
| `random_broadcast` | +0.28 | −1.03 | −0.83 | +0.96 | +1.52 | +1.81 |

**The sign flips.** Baseline gap −2.42, crosscoder moves it +7.09, steered gap +4.67 — the
model now obeys the *early* instruction. This is not nudging the tie-break, it is reversing
it: primacy imposed over recency. z against every control is 22.7–31.6, and all controls
hold.

It reaches **87% of the supervised ceiling** (+7.09 against +8.19), where the order task
reached 14% (+9.78 against +67.63). Those are different regimes rather than different
degrees, and the gradient screen should say why — the likely story is that recency has a
genuinely low-rank optimal write the crosscoder nearly saturates, while the order task's
ceiling contains a large component no rank-≤T write can reach.

Reading goes the other way again, exactly as before: SAE pooled AUC 1.000, tSAE pooled 1.000,
crosscoder window 0.730. Per-token dictionaries read this factor perfectly and cannot steer
it.

## 23:14 — antisymmetry becomes a standard control

The sharpest diagnostic of the sprint, and it falls out of the recency table:

> A real directed intervention is **antisymmetric in the dose**. An arm positive at both
> α = −2 and α = +2 is showing a magnitude artefact, not a signed effect.

Every constant-write arm above is positive at both extremes — SAE +1.86 / +1.33, `txc_flat`
+1.78 / +1.01, `random_broadcast` +0.28 / +1.81 — while `txc_slab` runs +7.09 at α=+1 and
−3.38 at α=−1. That separates signal from artefact far more cleanly than comparing
magnitudes, and it is now required on every arm: symmetric dose sweeps, a reported
antisymmetry statistic, and **retroactive application to last sprint's `steer_order` data**,
where only positive doses were ever swept. If last sprint's `sae_broadcast` +2.25 proves
symmetric in dose that strengthens the original claim; if `txc_slab` does too, it weakens it.

## 23:16 — a training-recipe correction, and a coordination failure of mine

The implement agent withdrew its own "TopK tSAE reconstructs 4–6× worse" figure after
finding it was under-training: at the sprint default it gives FVU 0.491 at 8
coefficients/segment, but 6000 steps gives 0.218 and lr 1e-3 gives 0.184, against the SAE's
0.098. The honest gap is ~1.9×. It then generalised the lesson correctly — if the tSAE was
not converged at the default, the SAE and crosscoder cannot be assumed converged either — and
is sweeping lr × steps across all three arms before any headline cell is quoted. That is the
same failure mode that invalidated a headline last sprint, where a 3× learning-rate change
moved realised capacity by 10×.

**And a coordination failure worth recording as mine.** The implement agent reported "still
nothing from theory" while theory had been writing runnable generators for over an hour — I
had been relaying design *specifications* in prose instead of the file path. `blocks.py`
contains `rotation_pair`, `grouped_rotation_pair`, `refusal_rotation_pair`, `screen`,
`rank_k_write` and `block_geometry`, all self-tested, in exactly the shape the harness wants.
Fixed. The cost was real: the implement agent built its own tasks in the meantime, which
turned out well, but that was luck rather than design.

## 23:18 — demonstration order clears diligence; four constraints that decide whether it works

R-GSM was retracted by the review agent after it read the construction section: the released
dataset keeps the last sentence fixed, reorders the others, and explicitly permits "minor
editing on words … to ensure grammatical correctness". Edited words break the exact multiset
match, so **P1 fails for R-GSM as released** — the property the whole design rests on. Two
further limits: 220 pairs, and the drops are 6.9–15.5 points on *frontier* models
(GPT-4-turbo 94.1→85.0, GPT-3.5-turbo 67.3→51.8), not the "over 30%" from the abstract, which
refers to a different setting. A 1.5B model sits well below GPT-3.5-turbo on GSM8K.

**Few-shot demonstration order replaces it**, on a better citation: Li, Wang, Wang, Shang,
*Order Matters: Rethinking Prompt Construction in In-Context Learning* (arXiv:2511.09700) —
"the variance in performance due to different example orderings is comparable to that from
using entirely different example sets", measured from **0.5B to 27B** plus GPT-4. The 0.5B
end removes the feasibility risk that killed R-GSM, and permutation is verbatim by
construction, so the multiset match is exact.

Four constraints from the literature, the first of which would quietly cost the entire effect:

| constraint | why |
| --- | --- |
| **base model, not Instruct** | instruction tuning increases prediction consistency under input perturbation, shrinking the effect we want to steer. `Qwen2.5-1.5B`, not `-Instruct` — a deliberate flag change away from the harness default |
| k ≈ 4–8 shots | order sensitivity *decreases* as demonstration count grows; many-shot would destroy it. Also the T range the crosscoder handles best |
| classification tasks | largest effect, and a clean label-token margin |
| the order-selection oracle is ceiling *and* rival | dev-set-selected orderings already reach near-oracle performance, so "just pick a better order" is an existing cheap fix |

The fourth constrains the claim and belongs in the write-up rather than being discovered by a
reader: **the claim is not that steering is the best way to fix order sensitivity — it is
that order sensitivity is a real behaviour whose steering separates window codes from
per-token codes.** The oracle then supplies a clean denominator, fraction of a real measured
gap closed, which is the same improvement the best/worst-permutation design buys.

Family status: **A recommended**; B ruled out on token bias; D demoted to motivation only;
C untested but shares A's verbatim property and is the natural backup.

## 23:20 — correction: the phase-ladder and recency wins are over the WEAK baseline only

A qualification to results already recorded above, and it should be applied before either is
written up as a win.

**The phase ladder is rank 1 at every rung.** With a fixed two-pool vocabulary there are only
two distinct block content vectors, so the difference rows are ±(a−b) whatever the block
length:

| switches | block length | c | r1 | rank |
| --- | --- | --- | --- | --- |
| 1 | 6 | 8.7e-34 | 1.0000 | 1 |
| 3 | 3 | 3.9e-34 | 1.0000 | 1 |
| 5 | 2 | 0.0 | 1.0000 | 1 |
| 11 | 1 | 0.0 | 1.0000 | 1 |

against D1, where rank grows with m — 1, 2, 3, 5, 11 at m = 2, 3, 4, 6, 12.

So the three winning ladder cells are wins over `sae_broadcast`, a **constant** write, and
every rung is predicted to **tie** against `sae_profile_target` and the tSAE, which are
rank-1 and therefore have exactly the expressiveness the task requires. They are mechanism
results about the constant-write baseline, not wins over the strong baselines, and the log
entry above should be read with that correction attached. **The same caveat applies to
instruction recency** until its `r1` is measured — the gradient screen on recency now gates
how that result may be described.

This does not devalue the ladder: its purpose is a **frequency-response** question —
separating "the advantage is just non-constancy" from "the advantage is about slow structure
and decays with frequency" — which is orthogonal to rank and which D1 does not answer.
Registered for it: advantage over `sae_broadcast` roughly flat across rungs, advantage over
`sae_profile_target` **zero at every rung**, and the absolute effect degrading once the
alternation period drops below the crosscoder's effective resolution.

**Twelve topic-orthogonal registers** now replace the old block pools, acting on the
non-orthonormality result. The old pools contained calm *and* tense — two poles of one
affective axis, whose difference is a single dominant direction that mechanically inflated σ₁
and consumed exactly the L3 headroom D1 exists to create. The replacement uses twelve
technical registers with no shared axis, and `GROUPINGS12` sets group size equal to block
length at every m, so all twelve appear exactly once per document at every rung. Two further
benefits: each block's content vector is exactly its group mean with **no sampling noise**,
and m stops being confounded with how many registers a document contains.

Two details checked rather than assumed. `level` in D6 is deliberately **not**
multiset-matched — that cell exists to show a constant write *can* reach a level target and
the SAE *should* win it, so its absolute numbers are not comparable to `trend`'s and only the
architecture ordering within each cell means anything. And D2/D2b use per-item clause
repetition so documents read as natural text; theory verified this leaves the rank unchanged,
since the difference matrix has the same distinct rows repeated and the Gram merely scales
(m=3: r1 = 0.5000, rank 2, in both forms). Naturalness is free.

## 23:22 — permutation composition becomes the anchor

Li, Guo, Andreas, *(How) Do Language Models Track State?*, ICML 2025 (arXiv:2503.02854).
The task is literally permutation composition — compute the order of a set of objects after a
sequence of swaps — so **the same multiset of swaps in a different order gives a different
final state, and P1 holds by definition rather than by construction**. They also characterise
the computation at algorithm level: one mechanism closely resembling an **associative scan**,
and a hybrid using **permutation parity** to prune the output space before refining with a
scan.

It earns a slot as the **calibrated positive control** the programme lacks. The synthetic
order task can be dismissed as a construct; this cannot, because there is a published
mechanistic account of how transformers actually compute it and the computation is
order-dependent by definition. If a window dictionary cannot beat a per-token one at steering
*here*, it will not anywhere — a fast, cheap, decisive negative. And a win is interpretable
against a known algorithm rather than a black box.

Unlike the two-block designs there is no obvious reason its optimal write should be rank-1;
composition of swaps has rich structure, so if `1 − r1` is large this is the L3 task on a
mechanism with a published account. Screened before building, like everything else.

## 23:24 — citation integrity: three search summaries did not survive a fetch

Worth recording as a methodological note rather than buried in a catalogue, because it
changes how the sprint should treat its own literature.

The review agent has now withdrawn three claims that came from search summaries and failed on
verification:

| claim | reality |
| --- | --- |
| a published turn-shuffle ablation for crescendo attacks | does not exist; the summary conflated sources |
| Roger & Greenblatt's *Preventing Language Models From Hiding Their Reasoning* = arXiv:2311.02282 | that id is a spark-plug fault-diagnosis paper; correct id is **2310.18512** |
| a shuffled-CoT control in arXiv:2603.22816 | that paper uses a Step-Level Reasoning Capacity metric instead |

Three failures from the same source class is a pattern, not bad luck. The catalogue now
carries a "claims withdrawn on checking" tier and treats unverified entries as untrusted, and
**no citation marked unverified may drive a build decision**. Both papers currently being
acted on — demonstration order and permutation composition — are marked fetched and verified,
which is why they are the two being acted on.

The third withdrawal came with a useful negative attached. The paper that was supposed to
support a shuffled-CoT task instead measures **step necessity**, finding Grok-4's reasoning
mode at 1.4% against 7.2% for its non-reasoning mode. Where reasoning steps are not necessary
the chain is decorative and permuting it changes nothing — and a 1.5B model is the least
likely of all to have necessary steps. So shuffled-CoT would probably have failed even if the
control had existed, and CoT faithfulness stays at the bottom of the ranking.

One consequence for multi-turn escalation: with no inherited turn-shuffle ablation, **its foil
would be ours to build**, which costs it the pre-legitimised-control advantage that
demonstration order has. Another reason it sits below.

## 23:26 — recency is rank 2, and the second direction has a name

**Correcting my own instruction:** I told implement to re-bill recency the way the phase
ladder was re-billed. That was wrong — the fixed-position geometry differs materially from a
two-block rotation.

After position 2 the *governing* instruction differs between classes; after position 9 it
differs the other way. So the slab carries two components with **disjoint temporal support**:

```text
P[2]      = +Δ    instruction lexical content      support {2, 9}
P[3..8]   = +g    which instruction is GOVERNING   support {3-8, 10-11}
P[9]      = −Δ
P[10, 11] = −g
```

`P = e_lex ⊗ Δ + e_state ⊗ g` with `e_lex · e_state = 0` — two exactly orthogonal rank-1
terms, so the SVD separates them. The second singular direction is **task-set / active-rule
state, carried by the filler and distinct from the instruction's lexical content**: "the
sentence *always answer in French* appears here" and "French-mode is currently active" are
different representational states. Registered: rank exactly 2, `r1 ≈ 0.65` (0.50–0.85),
`c(P_dom) ≈ 0.06` (0.02–0.14).

**What makes this a test rather than a story:** the two leading singular vectors' *temporal
profiles* should be near-disjoint — one on {2, 9}, the other on {3-8, 10-11}. If both spread
across all positions the decomposition is wrong and the rank-2 claim falls. One SVD on a slab
already in hand.

**A concrete case where the two screens diverge, vindicating the choice of gradient.**
`c(P_dom)` on recency is positive *by design*: the instructions have unequal governing spans
(position 2 governs six filler slots, position 9 governs two), so `Σ_t P[t] = 4g ≠ 0`. But
probe mode is a difference of differences and cancels the class-symmetric part of any write's
effect, so a constant `g` drops out and `c(Ḡ)` should come back below 0.02. That also
explains the measured constant arms — +1.1 to +1.9 and positive at *both* dose extremes is an
even-in-α, second-order magnitude artefact with no first-order component, exactly what
`c(Ḡ) ≈ 0` predicts, and consistent with `txc_slab` being antisymmetric as a genuine
first-order effect.

**So the recency headline stands as an expressiveness result**, conditional on measured
`r1 < 0.85`.

## 23:28 — the ceiling gap has a metric-mode explanation and a real one; one pass separates them

87% of ceiling on recency against 14% on the order task. Two candidates:

- **(A) metric-mode artefact.** The order task runs in *ordering mode* (score = logP(doc)),
  recency in *probe mode* (difference of continuation logprobs). In ordering mode `dom_slab`
  can exploit the entire content difference between two different documents — an enormous
  lever, hence +67.63 — while probe mode cancels the class-symmetric part by construction.
  **Percent-of-ceiling is then not comparable across modes**, and the contrast must not appear
  in the write-up as a regime difference until this is excluded.
- **(B) feature availability.** Recency's optimal write is one direction over a contiguous
  span — the natural shape of a crosscoder latent — and "which instruction is governing" is a
  state the model already maintains, so reconstruction training is rewarded for learning it.
  The order task needs a "tense-block-first" latent: a conjunctive, document-level property
  reconstruction does not reward, since reconstructing a tense sentence at position 3 needs
  "tense", not "tense-then-calm".

One extra scoring pass separates them: run recency in ordering mode. If percent-of-ceiling
falls toward the order task's it is (A); if it holds near 87%, (B) is real and yields a
stateable principle — *a crosscoder latent approximates the optimal write well when the target
is a state the model already maintains across positions, and poorly when it is a relational
property no single position encodes* — with the corollary that the order task's 7× headroom is
**not closable by better training**.

## 23:30 — the catalogue's main finding is a negative, and it is the useful one

Five independent saturation results, each from a different paper and subfield:

| behaviour | the per-token handle that already works |
| --- | --- |
| refusal | Arditi single-direction ablation near-saturates |
| MCQ option order | token bias over option-ID tokens; the attractor survives full content rotation at r = 0.9994 |
| repetition onset | a single sign-inverted neuron |
| emergent misalignment | a transferable linear direction |
| backtracking | linear directions in activation space (Venhoff et al., arXiv:2506.18167) |

> **Most temporally-extended-looking behaviours have a per-token-representable handle. The
> ones that do not are the ones where the two conditions are permutations of each other.**

That is a screening principle derived from prior work, and it converges with the sprint's own
`c` statistic — the same quantity measured rather than inferred. The two agree on every case
checked.

Two consequences worth carrying. The backtracking result lands on this repo's own incumbent
workstream: Venhoff et al. do not invalidate the existing Δgc 0.541 TXC against 0.400
per-token, but that number must be positioned against a **strong** baseline rather than a
naive one — the same distinction that has caught this sprint twice. And ISE (Wu et al.,
arXiv:2410.09102) *strengthens* the recency headline: priority has to be added
architecturally, via segment embeddings, because delimiters and instruction tuning "do not
address this issue at the architectural level". In an unmodified model provenance is
therefore carried by little more than position — which is the mechanism behind our result
rather than merely a coincidence with it.

## 23:14 — the honest hole in the ladder was under-training

6000-step replicates landed. The phase-5 cell — the one that failed its profile control at
2000 steps and was kept in the table with the gap visible — resolves cleanly:

| cell | `txc_slab` @2000 | `txc_slab` @6000 | `txc_flat` @6000 | `sae_broadcast` @6000 |
| --- | --- | --- | --- | --- |
| phase 5 | +1.56 | **+9.40** | +1.42 | +0.29 |
| phase 11 | +7.80 | +6.88 | −0.33 | +0.09 |
| recency | +7.09 | +6.77 | +1.62 | +1.66 |

Phase-5's crosscoder effect moves **6×** on training steps alone, from +1.56 to +9.40, while
its `txc_flat` control stays flat at +1.42. So the one cell where the profile control failed
to separate was under-trained, not null, and all four ladder rungs are wins once the recipe
is right. Recency and phase-11 both reproduce at the longer schedule, so the effect is not an
artefact of a particular budget.

Two things worth taking from this. **Reporting the hole rather than dropping it was what made
it resolvable** — a cherry-picked three-cell ladder would have looked cleaner and taught us
nothing. And it is the third time in two sprints that a training-recipe difference has moved
a headline number by more than the effect being measured: the learning-rate collapse in
sprint 2, the tSAE's "4–6× worse" that was really 1.9×, and now this. The rule that no
conclusion may rest on one arm being trained differently from another has earned its place.

The supervised ceilings also make the metric-mode point concrete: `dom_slab` reaches +45.79
on phase-5 and +28.15 on phase-11, both scored in ordering mode, against +8.19 on recency in
probe mode. Percent-of-ceiling is not comparable across those two scoring modes, which is
exactly why the ordering-mode control on recency is needed before the 87%-vs-14% contrast can
be stated at all.

## 23:16 — a calibration that would have produced a false negative

The demonstration-order go/no-go as I first specified it was underpowered, and running it
would have looked exactly like a clean negative.

Li et al. (arXiv:2511.09700) measure order sensitivity as the standard deviation of accuracy
across permutations: the average is **0.0197, about two accuracy points**, against a selection
sensitivity of 0.0225. Lu et al.'s "near state-of-the-art to random guess" is the **tail** of
that distribution, not the typical case — a random pair of orderings differs by a couple of
points. Their own protocol reflects it: `M=10` sets × `P=10` permutations to *estimate*
sensitivity, but **`P=128` permutations when they want to find a strong ordering.**

So the search needs ~128 permutations rather than the ~24 I passed on. Still forward passes
only. The design is unchanged and still right — the whole reason for taking the best/worst
pair is that the *typical* gap is too small to steer against — but the extremes have to be
genuinely extreme, and **a 2-point spread from a 24-permutation search reads as "underpowered
search", not "no effect".** Recorded explicitly because that failure mode is
indistinguishable from a real negative unless it is anticipated.

Helpful details from the same paper: the task set is AG News, NYT-Topics, NYT-Locations,
DBPedia and MMLU for classification; the models include **Qwen2.5**, the harness's own family;
and smaller models show "marginally higher variability", which works in our favour at 1.5B.

**A fallback recorded in case demonstration order does not clear: LLM-as-judge position
bias.** Judges pick the first-slot response in ~68% of comparisons even where humans prefer
the second (Zheng et al., MT-Bench; scaled by Shi et al. across 15 judges and ~150k
instances). Its appeal is that the standard mitigation — invoke the judge twice with
candidates swapped — means **a multiset-matched permutation pair is already what careful
practitioners run**, so the foil is inherited rather than built. It also avoids MCQ's failure
mode, since the swapped items are long content blocks rather than single label tokens, though
the verdict is still emitted as a label token so token bias could re-enter. Needs a 7B judge,
so it is the fallback rather than the first choice.

Steganography demoted to a future entry: frontier models "are unable to encode short messages
in their outputs without a monitor noticing under standard affordances" (arXiv:2507.02737),
so no organism exists at a scale this sprint could reach. Its structural argument — a payload
carried in sentence arrangement is invisible to any per-position readout by construction —
remains the most elegant in the catalogue and is kept on that basis.

## 23:18 — the shared-write constraint: a dictionary latent is one write reused across documents

The most operationally important finding of the sprint, and it nearly killed two tasks
silently.

A dictionary latent is **one fixed write reused across documents**, so what bounds any
fixed-write arm is the **mean** difference slab over documents, not the per-document one. If
the intervention a document needs varies between documents, the mean cancels and the ceiling
collapses — **for every architecture, crosscoder included.** Measured on a k=6 rotation over
400 documents:

| content | ‖mean P‖ | mean ‖P‖ | ratio |
| --- | --- | --- | --- |
| fixed across documents | 3.475 | 3.475 | **1.000** |
| resampled per document | 0.168 | 3.463 | **0.049** |

A 20× collapse. Every design so far satisfied this by accident — the order task always places
tense at the same positions, recency always places instructions at 2 and 9 — and both new
candidates are naturally specified the other way. Detector, no training needed:
`‖mean P‖ / mean ‖P‖`, with anything below ~0.3 meaning the task cannot support a fixed-write
result at all.

**It also explains why the three currencies resist being combined.** The more a task is fixed
so that a shared write can serve it, the less it resembles the phenomenon the literature
documented. That tension is structural rather than bad luck and belongs in the write-up.

Consequences for the two candidates. **Permutation composition is dead on arrival** if swap
content is resampled per document — at position `t` class A shows `s_t` and class B shows
`s_{t+1}`, both uniform over draws, so lexical *and* state marginals match and mean DoM ≈ 0.
Held with the swap multiset fixed and only order varying, it is the **best expressiveness
candidate left**: registered rank ≥ 3, `r1 ≈ 0.5`, with the later singular directions being
the running-state accumulator the associative-scan account predicts. Its signature differs
from recency's — the state components' temporal profile should **grow with t** as running
states diverge, where recency's was disjoint support. And parity cannot discriminate here at
all, since each swap flips it and parity after T swaps depends only on T: the paper's
parity-pruning heuristic is useless for this contrast, so **the task isolates the scan.**

**Demonstration order** has rank `k − (number of cycles)` of the best→worst permutation,
verified at k = 4, 6, 8; measured `r1` 0.463 / 0.337 / 0.259. It needs one fixed demonstration
set and one fixed best/worst pair with only the query varying. Reporting the cycle structure
alongside `r1` immediately distinguishes a disappointing result caused by the draw
(transposition-heavy, half the rank) from one caused by the model.

## 23:20 — three corrections from the implement agent, all in the harder direction

**Realised sparsity was being measured in-sample, and it flattered the crosscoder.**
BatchTopK is a batch rule at train time and a fixed threshold at eval, and the number of
latents clearing that threshold is data-dependent: 8.03 coefficients per segment in-sample
against **10.15 held-out** at nominal k=8 — a 27% hidden budget advantage. Last sprint's rule
was "match on realised, never nominal"; the sharper version is **realised must be measured
out of sample**, because the threshold is calibrated on training data and does not
generalise. Lowering the crosscoder's nominal k to 6 gives 7.77 against the SAE's 8.00, and
**the recency result survives at genuinely matched budget: +6.35 ± 0.19 against +7.09 at the
inflated one**, still reversing a 2.42-nat baseline rather than merely erasing it.

**Reading AUC was selection-biased** — the best of 4096 latents scored on the documents the
maximum was taken over. Latent selection now happens on train, scoring on held-out;
crosscoder AUCs move by up to 0.06 either way.

**The training recipe was unfair against the crosscoder and the tSAE, not the SAE.** FVU at 8
coefficients/segment on the recency corpus:

| arm | lr 3e-4 / 2000 | lr 3e-4 / 6000 | lr 1e-3 / 2000 |
| --- | --- | --- | --- |
| TopK SAE | 0.0428 | 0.0373 | 0.0426 |
| crosscoder | 0.2389 | 0.2160 | **0.1260** |
| attention tSAE | 0.1458 | 0.0539 | 0.0639 |

The SAE was converged at the default and the other two were not. At lr 1e-3 the crosscoder's
FVU nearly halves *and* its realised overspend falls from 10.15 to 8.32, so the better recipe
also makes the budget match easier. **Three corrections, all of which made the win harder to
obtain** — a much stronger position than three that happened to help, and worth stating that
way.

**Init variance is large enough that no single run is a verdict.** Phase-5's crosscoder best
delta across three inits of an identical configuration: 1.56, 15.70, 11.48 — a 10× spread.
The *sign* was stable and `txc_slab` exceeded `sae_broadcast` in all three, so that is the
reportable claim while the magnitude is not. Every summary cell now needs at least three
inits with the range reported.

**A null worth keeping: escalation.** Ascending versus descending request intensity, foil the
exact reversal. Baseline `score(A) − score(B)` = −0.07 ± 0.13 — the model shows no compliance
difference at all — and steering moves it at most +0.63 ± 0.09. The diagnosis is a scope
condition on the whole method: **the behaviour must exist at baseline for a temporal
intervention to have anything to act on.** Recency has a real 2.42-nat bias and moves 6–7;
escalation has none and moves nothing.

**And a control I had not thought of: `txc_profile_random`** — the crosscoder's per-position
norm profile kept exactly, the directions replaced by random ones. `txc_flat` asks whether it
needs its profile; this asks whether it needs anything *but* the profile. If they match, "where
to write" is the entire contribution, which is a much weaker claim than it sounds.

**One collision flagged back:** `recency_var` randomises instruction positions per document,
which is exactly the shared-write constraint's failure case. A null there could mean the
crosscoder was only addressing two known slots, or that no fixed write of any architecture can
serve the task. `‖mean P‖ / mean ‖P‖` and the `dom_slab` ceiling distinguish them, and must be
measured before the result is read.

## 23:22 — the selection criterion was backwards, and this repo already had the evidence

`docs/dmitry/reviewer_responses/semisynthetic_language_tasks.md` contains executed Modal
results that supersede the P1/P2 reasoning this sprint has been running on. Two language
demonstrations were attempted there and **both failed**:

| task | why it failed |
| --- | --- |
| passphrase verification (k distinct code-words) | a **conjunction**, which a single broadcast write satisfies — and it is a textbook position-dependent template |
| ordered generation (days, numbers) | **mode-dominated**: at k ≥ 3–5 the per-token broadcast matched or beat the crosscoder by **10–50×**, with the gap running the *wrong* way as k grew |

The mechanism is that language generation is driven by a strong shared contextual mode, and a
broadcast write reinforces it at every position, overwhelming a template's per-position
writes.

**So P1 is the necessary condition and P2 is not.** The sprint had been treating write
non-constancy as operative and matched-multiset as a clean special case. Passphrase
verification is textbook P2 and still loses. What a matched multiset actually guarantees is
that **there is no DC component for a broadcast write to ride.** The correct question about
any candidate task is not "is the optimal write position-dependent?" but:

> **Is there any bag-of-positions statistic — a mode, a label prior, a level — that separates
> target from foil?**

Then the fix in that note worked: trajectory tasks with multiset-matched foils, "so no
bag/mode statistic — hence no broadcast write — separates target from foil in principle".
Four of four passed, with the template growing roughly linearly in k (+75.7 at k=2 to +218.9
at k=10) while **broadcast stayed pinned at ~0 or negative at every k**.

**Verdicts that change:** induction dropped (copying-ness is a mode, and the task is close kin
to the ordered-days task that already failed here — the screen I had queued for it is
retired); repetition loop escape dropped (repetitiveness is the mode par excellence, and a
flat repetition penalty is literally the broadcast write); LLM-judge position bias demoted
(the verdict is a label token, so "prefer B" is a DC handle — the same failure as MCQ);
permutation composition made conditional on an upstream hookpoint, since the model aggregates
at the query token and a single write there may set the state outright.

**And instruction-order conflict holds, which is independent support for the sprint's own
headline.** With A-then-B against B-then-A and "obey the first" as the target, the correct
*content* differs between conditions, so no single content direction helps both. That is
exactly the recency result, and it explains why that task worked where the earlier language
demonstrations did not.

**A prerequisite for demonstration order, not a refinement.** ICL order sensitivity is partly
carried by recency and majority-label bias — both bag statistics, both named in *Calibrate
Before Use*. They have to be killed by construction: **permute only the middle
demonstrations, hold the first and last fixed, and match the label multiset.** Then the pair
differs only in interior arrangement and the DC component is gone. Without it a broadcast
write may close the gap and the result is uninterpretable — which is precisely how ordered
generation failed.

**One reframing worth keeping.** That note concluded "do not pursue a language *steering*
demonstration as a paper headline". But last sprint's order-only result and these trajectory
tasks are all language, all multiset-matched, and all won. The narrower statement is better
supported: **language steering wins iff the foil is multiset-matched.**

## 23:24 — a go/no-go that is informative either way

The interior-permutation control raises the obvious question: strip out recency and
majority-label bias and is there any order effect left to steer? The honest answer from the
literature is **unquantified** — those two are the only *named* mechanisms of ICL order
sensitivity, both are bag statistics, and no complete mechanistic account of order dependence
exists.

That makes the go/no-go worth running whichever way it falls:

| outcome | what it means |
| --- | --- |
| spread survives the control | residual non-DC order sensitivity exists; the task is live |
| spread collapses | *ICL order sensitivity at this scale is fully accounted for by recency and majority-label bias* — a finding the ICL literature does not currently have, from an hour of forward passes |

The gate is therefore not a gamble even if the headline dies, which is the property a first
step should have. It has been specified to report the spread with and without the control
side by side, so the second branch is a result rather than a shrug.

One escalation path recorded: the named biases weaken as model size grows, so 1.5B has both
the largest total order effect *and* the largest DC share of it. A collapse at 1.5B is
consistent with a survivable effect at 7B, so that is the next step rather than abandonment.

**Pairing rationale.** If two tasks run, the second should be instruction-order conflict,
because the pair is a **dissociation**: demonstration order can die to a residual-DC problem;
instruction-order cannot, since with A-then-B against B-then-A and "obey the first" as the
target the correct *content* differs between conditions, so no single content direction helps
both. Two tasks that can only fail in different ways are worth more than two sharing a
failure mode — and instruction-order is the closest relative of the recency result, which
gives the pair a natural narrative.

Induction and repetition are dropped rather than screened: copying-ness and repetitiveness
are exactly the modes a broadcast write rides, and this project has already lost that fight
on ordered-days and numbers.

## 23:26 — the designated fallback: an existing result missing only its dictionary arm

If the demonstration-order gate comes back flat, the next move is **per-section style
scheduling**, not another behaviour hunt.

It is this repo's own `lang_profile` / `int_profile` promoted from construct to use case, and
the winning numbers already exist: template +63.2 against broadcast +6.0 at k=6, the full
k-sweep with template growing linearly while broadcast stays pinned near zero, and
0.812 ± 0.044 per-slot generation accuracy against 0.444 at chance 0.5. P1 is exact by
construction — the foil is a permutation of the same profile — and the DC handle is genuinely
absent, which is assertable rather than hopeful for only two other candidates.

**What makes it a contribution rather than a relabelling is the gap this sprint is equipped
to close: those numbers use difference-of-means proxies, not trained dictionaries.**
Converting it into a proper SAE / tSAE / PSAE / crosscoder comparison at matched realised
out-of-sample budget is the difference between *"the optimal write has this structure"* and
*"a trained dictionary finds that structure unsupervised"* — which is exactly the discovery
claim the rank framework leaves standing.

The use case is mundane but real: controllable generation with a per-section specification —
"formal introduction, casual body, formal conclusion", "English abstract, French body". The
spec is a schedule over sections and the useful intervention is a waveform rather than a
level. Writing "be more formal" uniformly is exactly wrong on the sections that should be
casual, which is the same failure already measured here when broadcast made text "Frenchier
everywhere, which is exactly wrong on half the slots".

Honest limit if it becomes the headline: the constituency is smaller. Style scheduling is a
product convenience rather than an eval-validity or safety problem, so it is the fallback
rather than the lead — but it is the lowest-risk use of an idle slot, since it needs no new
corpus design and its structure is already verified.

## 23:28 — the screen retro-predicts eight prior experiments, and every past win is discovery

The most consequential message of the sprint, in three parts.

**One: the two gates retro-predict eight executed experiments, 2 failures and 6 successes,
from the sign of one number with no training.**

*Passphrase verification* is the case that justifies measuring `c` rather than checking
whether a multiset matches. Its foil corrupts one word of `k`, so `k−1` of them agree and it
*looks* matched to inspection — but measured `c` runs 0.665 at k=2 down to 0.154 at k=12,
discarding it at every `k` the original experiment ran. With its validity state included —
the steering target *is* "authenticated", a scalar the model computes and writes everywhere,
a pure DC component — `c` reaches 0.56–0.85. And its `r1` is 0.400, so it has abundant
rank structure and fails purely on DC. **A graded statistic catches what a binary condition
misses.**

*Ordered generation* is nearly definitional: a mode is a state present at every position,
which is exactly a constant write. `c` runs 0.333 at mode strength 0.5, 0.585 at 1, 0.951 at
4. "Mode-dominated" and "large `c`" are the same statement.

The six successes go the other way — the trajectory tasks and the k-sweep measure `c = 0.0000`
exactly at every k, and the prediction that a broadcast write is then pinned at zero with a
*harmful* second-order term matches the observed broadcast deltas of −0.2, −0.0, −0.9, −1.0,
−0.5, −3.3, −9.3. The earlier note's own line — "on matched multisets the DC write can only
break symmetry against you" — is `c = 0` plus a negative second-order term, and it is the same
signature as this sprint's finding that constant arms are even in α.

**Two: the taxonomy correction is about input versus write.** The earlier document's rule read
"position-dependent template (distinct content per slot) → needs the whole pattern → TXC", and
that prediction failed on passphrase. Passphrase has a maximally position-dependent *input*
and a DC *write target*. Reasoning about the input is what produced the wrong call; the two
gates are about the write and are independent of each other.

**Three, and this is the one that changes what the project has established: every steering
win it has is rank 1, by its own measurement.** Lines 250–254 of the earlier note already
report that the trajectory tasks' per-position directions are "≈ ±(one attribute direction)
with signs following the profile" — one direction with a sign schedule, which is rank 1,
reproduced synthetically at `r1 = 1.0000` exactly for k = 2, 4, 6, 8, 10.

So the four trajectory tasks, the full k-sweep, the 81% generation demonstration and the
previous sprint's order task are **all reachable by a profile-steered SAE or a tSAE**. Every
one is a discovery result: the crosscoder found the waveform unsupervised, which is real and
useful and is not an expressiveness claim. The earlier analysis reached the same place in its
own vocabulary — "trajectory control vs level control, not direction diversity" is exactly L0
against L1/L2 — without drawing the consequence that a *scheduled* per-token write reaches the
same waveforms.

**Instruction recency and the rotation ladder are therefore the only expressiveness candidates
the project has**, which makes recency's measured `r1` the most load-bearing number
outstanding. Predicted rank exactly 2, `r1 ≈ 0.65` (0.50–0.85). Below 0.85 and it is the first
expressiveness result here; above, and the project's honest position is many discovery wins
and no expressiveness win — coherent and publishable, but a different story.

## 23:30 — two entries recorded for the next sprint rather than this one

**Tool-call ordering** has the best relevance story anyone produced: *"an agent can call every
tool correctly and still fail the task"* is the cleanest statement of the property in any
literature and it comes from practitioners rather than from us. Dependency structure makes
ordering non-negotiable (`auth` before `query`, `create` before `update`), and it is DC-clean
for the right reason — fixing the order means promoting `auth` early *and* `query` late, so a
constant "prefer auth" write is wrong at the query slot. Costs: unambiguous dependency
structure is real design work and it likely needs a 7B, which is why it is next sprint's
opening rather than tonight's.

**Planning / lookahead is rejected**, with the cleanest evidence in the catalogue. Ma & Rui
(arXiv:2605.07984) find rhyme planning causally localised to a *single position* — ~90% of
planning capacity recoverable at the newline token through five attention heads — and the
causal reliance appearing only at Gemma-3-27B, with other models showing "near-zero causal
effect at the line boundary despite strong probe signal". One token, five heads, out of scale
range. The incidental lesson deserves its own line: **a strong probe signal is not a causal
handle** — the same reading/steering dissociation this sprint keeps rediscovering.

## 23:32 — the variable-position control kills the advantage, and the reference row says why

`recency_var` is instruction recency with the two instruction positions drawn per document
rather than fixed at 2 and 9. The registered prediction was "shrinks and survives". It did
not survive.

| arm | recency (fixed positions) | recency_var (positions vary) |
| --- | --- | --- |
| `txc_slab` | +6.48 ± 0.15 | **+0.63 ± 0.31** |
| `sae_broadcast` | +2.60 ± 0.15 | +0.93 ± 0.12 |
| `dom_slab` (difference-of-means reference) | +8.20 ± 0.22 | **+5.46 ± 0.23** |
| baseline gap | −2.42 | −2.26 |

**The reference row is what makes this a finding rather than a null.** The crosscoder falling
to +0.63 would on its own be ambiguous — it could mean no fixed write can serve the task once
positions vary. The difference-of-means write staying at +5.46, far above the 2.26 baseline
gap, rules that out: **a fixed (T,d) write can still reverse the bias, and the crosscoder
simply fails to find it.** A discovery failure with a witness.

Theory had registered, from the shared-write constraint, that the reference should retain
0.35–0.65 of its fixed-position value rather than collapse; measured retention is
5.46/8.20 = **0.67**, just above the range. The crosscoder retained 10%. So both proposed
explanations hold in measurable proportion: **a ~2× scope limit on the entire fixed-write
class, plus a ~4× crosscoder discovery gap on top of it.**

## 23:34 — correcting my own terminology: difference-of-means is a reference, not a ceiling

The `evidence` transfer task — recency over conflicting testimony rather than conflicting
instructions — is a win, and it settles a labelling error I have been propagating:

```text
txc_slab  +5.92 ± 0.10      sae_broadcast +1.33 ± 0.08     txc_flat +2.84 ± 0.08
tsae_broadcast +0.32 ± 0.08  dom_slab +3.49 ± 0.18          baseline gap −1.36
```

z against the SAE is 35.9 — and **`txc_slab` exceeds `dom_slab`**. Difference-of-means is the
best write of one particular supervised *form*, not the best write available, and the
crosscoder found a better one. Everywhere I have written "the supervised ceiling" it should
read **"the difference-of-means reference"**; the genuine upper bound is `oracle_slab = Ḡ`,
the mean margin gradient, which is the best norm-matched write of any form in the linear
regime. This also retires last sprint's 7×-headroom framing, which compared against a
reference I was calling a ceiling. `theory_section.md` corrected.

## 23:36 — a false positive pointing the way we want it to point

`rank1_best` — the ceiling for any per-token dictionary handed a perfect per-position dose
schedule — was built from the difference-of-means slab. The true rank-1 ceiling is the best
rank-1 approximation of the **gradient**. In probe mode the metric cancels class-symmetric
effects, so components present in `P_dom` can contribute nothing to Δ, and `rank1_best` would
underperform **for reasons unrelated to rank** — making `txc_slab > rank1_best` read as
expressiveness when it is an artefact of construction.

Since expressiveness is the one claim the sprint does not yet have, an arm that manufactures
it is precisely what must not ship. Being rebuilt from `Ḡ`, with `cos(P_dom, Ḡ)` reported per
task as the minimum acceptable fallback. Same lesson as the ceiling correction, one level
deeper: the supervised difference of means is a reference, and the optimum for steering is the
gradient.

## 23:38 — methodology fix 4: one-sided doses manufacture false nulls

The ordering-mode tasks swept α over positive values only, so an arm whose correct steering
direction is *negative* registers as a flat failure. Not hypothetical: at lr 1e-3 phase1's
crosscoder latent is negative at all four positive doses (best −0.47) while the
difference-of-means write reaches +69.10 — the write direction was available and the arm was
pointed the wrong way. All tasks now sweep symmetrically about zero, which also inflates every
null equally.

That is the **fourth** correction this sprint that either made a result harder to obtain or
exposed a false negative — after in-sample realised sparsity, selection-biased reading AUC,
and the recipe that handicapped the crosscoder and tSAE. Two more from theory: the `c = 0`
gate that should have been `c < 0.1`, and the `rank1_best` construction above. The pattern is
worth its own paragraph in the summary.

**The scope limitation now taking shape, and it should be stated as a limit rather than a
hedge:** the crosscoder wins where the factor sits at **consistent positions across
documents** — recency, evidence, the phase ladder all have fixed spatial layout — and loses
where it does not. That is the same fact as the shared-write constraint seen from the data
side: a dictionary latent is one fixed write reused across documents, and consistent positions
are what keep the mean difference slab from cancelling. The honest scope of the method is
**"finds interventions for structure that recurs at the same place"**, which is narrower and
more useful than "handles temporal structure".

## 23:40 — a fifth ladder rung, and it is the control the crux objection needs

Der, Kamath & Thompson, *Turn-Averaged SAEs for Feature Discovery and Long-Context
Attribution* (arXiv:2606.28548) represent a whole turn with a fixed number of features by
reconstructing the **average activation across the turn**, motivated by standard SAEs' active
feature count scaling linearly with context length.

It is **aggregation with order destroyed** — precisely the control that separates a window
code from mere pooling, and the answer to the objection *"is the crosscoder doing anything a
pooled per-token code could not?"*, which the sprint currently cannot answer with a trained
arm. Until now that rung would have been something we defined; it is now a citable published
architecture, which matters when the control is the crux. Implementation is about ten lines:
train the SAE on segment-mean activations, steer by broadcasting its decoder direction across
the segment.

The ladder is now five rungs, every one published, each adding exactly one thing:

| rung | temporal capacity per latent |
| --- | --- |
| BatchTopK SAE | none — the per-token floor |
| **segment-averaged SAE** | **aggregation, order destroyed** |
| Persistent SAE | one scalar: a learned timescale |
| T-SAE | InfoNCE coupling to adjacent positions |
| TemporalCrosscoder | free `(T,d)` slab |

With the segment-averaged rung in place, a flat result between it and the crosscoder becomes
a genuinely important negative rather than an ambiguous one.

**Sandbagging formally dropped.** *The Elicitation Game* (arXiv:2502.02180) tested
password-locked and circuit-broken organisms and found prompting elicits the capability while
**activation steering failed** — the intervention modality we would have used has been tried
on those organisms and did not work. Combined with the Model Organism Lottery warning about
organism-dependent interpretability, expected value there is low.

The literature agent has been told to stop scanning and write. Nineteen entries, nothing above
priority 2 resting on an unverified citation, four of its own claims withdrawn on checking and
recorded in place rather than deleted so the reasoning that produced each error stays visible.
