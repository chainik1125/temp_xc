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
