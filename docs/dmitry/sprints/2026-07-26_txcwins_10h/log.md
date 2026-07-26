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
