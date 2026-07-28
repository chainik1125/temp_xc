---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - results
  - complete
---

## What this sprint was asked for, and what it produced

Find further tasks where a temporal crosscoder beats a TopK SAE and a tSAE.

**It found three, established what kind of win they are, withdrew the previous sprint's headline,
and produced a screen that ranks candidate tasks before any dictionary is trained.**

Every number below is read from a named file in `results/txc_wins/`. Detail, derivations and the
full audit trail are in `exec_summary_draft.md`; the research log is `log.md`; the literature
catalogue and build checklist are `literature_catalogue.md` and `build_checklist.md`.

## The five findings

### 1. The previous sprint's headline is withdrawn

![The half of the dose axis nobody sampled](../../../../plots/2026-07-26_txcwins/withdrawal.png)

The order-task result — crosscoder +11.29 against the SAE's +1.24 — was measured on a **one-sided
dose grid**. Rerun at both signs with two dictionary inits, the crosscoder does not beat the SAE
significantly in either. The control that was the proof inverts: `txc_flat`, the crosscoder's own
slab with its temporal profile averaged away, was reported as inverting to −8.02 and in fact
reaches **+12.10 and +18.47** — roughly double the crosscoder itself. A positive-only grid
recorded the negative branch of a signed effect and read a **sign** as an **inversion**.

Two lessons generalise. **A one-sided dose grid cannot distinguish a directional effect from a
magnitude artefact.** And **selecting each arm at its own best dose is not neutral** — it picks
each arm's saturation point, which is where the linear reasoning behind every ratio stops applying.

### 2. The wins are real, and they are discovery rather than expressiveness

![Reversing which instruction the model obeys](../../../../plots/2026-07-26_txcwins/recency.png)

On **instruction-position bias**, **evidence order** and a **12-block rotation**, a crosscoder
latent beats every arm a practitioner can build **from the trained dictionary alone** —
`sae_broadcast`, `tsae_broadcast`, `txc_flat`, `txc_profile_random`, `random_slab`,
`random_broadcast` — winning **7 of 9** held-out cells on either dose convention. The exceptions
differ by convention: at peak dose `demo_order` ds1 and ds2; at matched dose `demo_order` ds2 and
**evidence ds1**, where the crosscoder measures +0.545 against `random_broadcast`'s +0.549 — a tie
with a random constant write. Evidence order is therefore not uniformly clean either.

**Hand that same dictionary direction a *supervised* schedule and it beats the crosscoder in 6 of
9 held-out cells** (z up to −20.6). `sae_schedule` is built as `outer(P_dom · v_sae, v_sae)` — the
dictionary's direction on the **difference-of-means** slab, which needs labels and is not
obtainable from the dictionary. **That is the discovery claim measured rather than asserted:** what
the crosscoder supplies is the schedule, and a practitioner who already has one does not need it.

**Both headline tasks survive a held-out content split**, three dictionary inits each: dictionaries
trained on one half of the sentence pools, steering scored on documents built from a disjoint half.
The crosscoder beats every learned per-token arm in every init (z = 9.6 to 30.0, peak-dose). The
margin is somewhat **smaller** held out than corpus-bound — z = 10.3 / 16.3 / 14.9 against 18.0 —
which is what one expects when the dictionary can no longer key on the content it trained on. The
claim that matters is that **the effect survives at all**, on disjoint content, in every init.
That closes the strongest objection a reader could raise.

**What it never beats is the best rank-1 write taken from the metric's own gradient** — z = −32 to
−41 on instruction position, −67 to −73 on evidence, −50 to −59 on demonstration order.

So the claim is **discovery, not expressiveness** — and both halves are now measured separately.
Against arms a practitioner can build from a learned dictionary it wins convincingly and under a
steering-based selector (z = +15 to +18, three inits, held-out). Against the **best constant write
that exists** it is behind on **six of seven tasks**, including two ladders built to make constant
writes bad. **The temporal form is not where the margin comes from.** What the crosscoder supplies
is a write found unsupervised from reconstruction alone that a per-token dictionary could have
executed if handed the schedule — and a published method now supplies exactly that schedule
(Heyman & Vandeputte, arXiv:2605.03907), which the crosscoder loses to.

⚠ **Two rank-1 arms, and only one is a ceiling.** On held-out content the crosscoder **loses to
`rank1_best` in 7 of 9 inits** and loses all three at matched dose on instruction position (3.91 /
4.92 / 4.74 against 4.955). But `rank1_best` is the rank-1 truncation of the **difference-of-means
reference**, near-orthogonal to the gradient (`cos` = 0.02–0.19), so neither beating it nor losing
to it settles anything about rank. **The ceiling is `grad_rank1`**, and the crosscoder loses to
that everywhere by a wide margin.

### 3. One number ranks a task before any dictionary is trained

![The constant share against the crosscoder's margin](../../../../plots/2026-07-26_txcwins/c_gate.png)

`c` is the share of a task's optimal write reachable by a **constant** write, measured from
gradients in one backward pass per document with no dictionary involved. Across seven tasks at
identical `n_docs` and gradient budget, **the best achievable classification is 6 of 7 and no
threshold does better** — `rotate6` (0.134, loses) and `evidence` (0.143, wins) are inverted 0.009
apart, and the data locate no boundary between them.

**It must be measured on the metric gradient, not on difference-of-means.** The cheap proxy gives
0.036 for the order task and 0.039 for instruction position — opposite outcomes, near-identical
values — while the gradient separates them 6×. Four independent demonstrations of this divergence.

**`c` has an exact operational meaning, not merely a suggestive one.** For a broadcast write
`W = (1_T ⊗ v)/‖1_T ⊗ v‖`, maximising the first-order effect `⟨W, Ḡ⟩/‖Ḡ‖_F` over **all** `v` gives
exactly `√c`, achieved at `v ∝ mean_t Ḡ` (verified numerically to four decimals). So `√c` is not a
bound on what a per-token dictionary can reach — it is **the reach of the best conceivable
broadcast direction**, whether or not any dictionary contains it. On held-out instruction position `c` = 0.0343
(`recency_tr_ho_ds0`; the 0.0365 reported elsewhere is the corpus-bound run), so no constant write
of any kind exceeds **18.5%** of the optimal write's first-order
effect. That is what makes the arm escaping this bound — one direction on a *schedule* — the
honest per-token comparator.

**What `c` does and does not do, stated once.** It bounds what a *constant* write can reach, and
that bound is confirmed — 4/4 on the four-rung ordering test against `sae_broadcast`/`grad_slab`,
the arm it actually describes, and it retro-predicts eight executed experiments across two sprints.
It is a **ranking heuristic with a known inversion**, not a rule. What it does **not** do is
forecast what a crosscoder achieves; neither does `r1`. **Geometry sets the ceilings, and nothing
measured in this sprint predicts which ceiling gets reached.** It does not establish a
quantitative law: the constant arms are 72–80% *even* in α, so they are largely second-order
artefact, which makes `sae_broadcast` a **mis-specified** baseline rather than a weak one.

**A third axis the screen does not model at all, found post-sprint.** `c` and `r1` both describe
the **mean** optimal write. **Shared-write retention** describes whether the *per-document* optimal
writes agree with each other — and nothing in the framework touches it. Every argument in it ("a
constant write cancels against a contrast", "the mode has nothing to grip") is about what a
constant or rank-1 write can express *on average*. A task can have a perfectly well-shaped average
write that **no single fixed slab reproduces per document**, and a dictionary latent is exactly one
fixed write reused everywhere.

Retrieved-document position is the demonstration: `c` = 0.045–0.069 with `r1` ≈ 0.5 — the best
shape statistics anything has screened, low constant share *and* genuine rank ≥ 2 together, which
no construct in this sprint achieved — and **retention at 1.8–2.1× its noise floor** against
prompt injection's 10–13×. **The screen predicts the right thing about the wrong quantity there.**

Since the two compound, the steerable signal goes roughly as **`baseline × retention`**: ~9.1 for
prompt injection against ~0.17 for retrieved-document position, a 50× gap larger than either
factor alone.

### 4. Reconstruction quality does not predict steering quality

![Reconstruction against steering, inverted](../../../../plots/2026-07-26_txcwins/fvu_vs_steering.png)

Each architecture at its own sweep-derived best recipe, matched at 8.0 realised coefficients per
segment on held-out data. At matched dose the rank inversion is **strict and complete**: FVU orders
attention tSAE (0.0144) < TopK SAE (0.0373) < crosscoder (0.0968), and steering orders them exactly
in reverse, spread **28×**.

**Any benchmark ranking dictionaries by FVU ranks these three backwards** for the use a crosscoder
is proposed for. Two consequences: a benchmark fixing one learning rate across architectures is not
measuring architectures — the three peak at recipes spanning a 10× range — and the **L1 temporal
SAE has no usable sparsity coefficient at all**, with FVU crossing 1.0 before L0 crosses 32,
because `TemporalSAE` has no encoder bias.

### 5. Reading and steering come apart

![Reading against steering, three architectures](../../../../plots/2026-07-26_txcwins/reading_vs_steering.png)

*Nine held-out cells — three tasks × three dictionary inits — with all three architectures on
content the dictionaries never trained on. Reading is the best single latent's held-out AUC;
steering is that latent's effect at matched dose with the sign free.*

| | median reading AUC | median steering Δ |
| --- | --- | --- |
| TopK SAE | **1.000** | +0.09 |
| attention temporal SAE | **1.000** | +0.04 |
| temporal crosscoder | 0.850 | **+1.12** |

**Reading does not separate the architectures — all three reach AUC 1.000 on most cells. Steering
separates them 12×.** Both per-token architectures decode these factors perfectly and move them by
about a tenth of a nat; the crosscoder decodes them no better and moves them an order of magnitude
further. This is the most-replicated finding in the project — but it is **nine held-out cells
across three tasks**, not nine independent tasks, and it has since acquired a documented exception.

⚠ **On prompt injection the pattern does not hold.** The TopK SAE reads that factor at **0.632**
against the attention tSAE's **0.976** on the same activations — so the factor is readable at this
layer and the TopK basis specifically fails. **There the SAE is not reading well and steering
badly; it is failing at both**, which is a different claim and changes what a crosscoder win would
mean. Across all 98 files carrying an SAE reading AUC, 61 are ≥ 0.99 and six are below 0.70 — the
phase-ladder cells among them. The 1.000 result is a property of the three headline tasks, not of
per-token dictionaries in general.

⚠ These cells are **reading-selected**, so by the selection result in Limits they understate every
arm — including the crosscoder. The dissociation is a statement about the latent a reading-based
selector picks, which is what deployed practice picks.

## What was not achieved

**No expressiveness win, and it is now excluded by measurement across seven tasks rather than
inferred from one.** `broadcast_optimal` is the best constant direction in the whole space, so it
bounds what the crosscoder's *temporal form* can buy. Ratio is crosscoder / that ceiling, at
**matched dose, medians across available inits** (`rotate12` and instruction position have three,
the rest one), held-out content where the task has a held-out variant:

![Decomposing the gap](../../../../plots/2026-07-26_txcwins/gap_decomposition.png)

| task | `r1` | `c` | ratio | z | txc / optimal |
| --- | --- | --- | --- | --- | --- |
| 12-block rotation | 0.177 | 0.033 | 0.86 | −2.1 | 0.101 |
| 6-block rotation | 0.210 | 0.102 | 0.19 | −19.3 | 0.075 |
| 2-block rotation | 0.304 | 0.163 | 0.09 | −18.6 | 0.054 |
| order (withdrawn headline) | 0.478 | 0.241 | 0.17 | −11.5 | 0.074 |
| phase-11 | 0.562 | 0.050 | 0.39 | −8.0 | 0.061 |
| evidence order | 0.595 | 0.156 | 0.28 | −78.1 | 0.154 |
| **instruction position** | 0.850 | 0.034 | **1.16** | **+2.0** | **0.287** |

**The crosscoder clears the best conceivable constant write on one task of seven — the sprint's own
headline — by 16%, at z = +2.0.** On the other six it loses, at z from −2.1 to −78. That range spans
the sprint's entire design space, `r1` 0.177–0.850 and `c` 0.033–0.241, **including two ladders
built specifically to make constant writes bad**.

**The controlled pair is the sharpest part.** `rotate12` and instruction position have essentially
identical `c` (0.033 vs 0.034) and `r1` differing **4.8×** — and the rank-designed cell is the one
that *loses*.

⚠ **Neither statistic predicts this ratio, and `c` cannot**, because it is inside the ratio's
definition. To first order the best constant write attains `√c·‖Ḡ‖` and the crosscoder attains
`cos(P_txc, Ḡ)·‖Ḡ‖`, so

```text
ratio = cos(P_txc, Ḡ) / √c
```

The correlation of `1/√c` against `c` over these seven `c` values is **−0.937** — a pure algebraic
artefact — while the measured ratio-vs-`c` correlation is **−0.749**. **The measurement is weaker
than the artefact**, so the real across-task variation *attenuates* `c`'s apparent predictive power
rather than creating it. `c` is not the surviving screen on this evidence; it is the denominator.

**Removing it recovers where the variation actually lives.** Alignment `cos(P_txc, Ḡ)` spans
**5.9×** across the seven tasks, and the one the crosscoder wins has the **highest** alignment
(0.214 against a 0.087 median) *and* the **highest** `r1` — the least rank headroom of the seven.
`rotate12`, with near-identical `c` and 4.8× more headroom, loses because its alignment is 1.4×
worse.

**The non-circular statement** (alignment is derived from the ratio, so "the outcome tracks
alignment" would be circular): the outcome **decomposes** into a part knowable before training —
`√c`, pure geometry — and a part not knowable in advance — how well the learned latent happens to
align with the gradient. **All the interesting variation is in the second.**

Instruction position is also the outlier on share of the optimal write: **0.287 against 0.054–0.164
everywhere else**. Whatever makes the headline work is specific to that cell.

⚠ Three qualifiers. `broadcast_optimal` is gradient-derived and **supervised** — no per-token
dictionary reaches that line (the steering-selected SAE gets 0.14–0.43 of it), so this is not "an
SAE beats the crosscoder". **Four of seven cells are one init**, and they are the ones with the most
extreme ratios, so their *ordering* is soft even though all four sit far below 1. And `order` and
the rotations are **ordering mode** while instruction position and evidence are **probe mode** — the
ratio is dimensionless and formed within a run, which is why it shares an axis at all.

**Rank ≥ 2 is real and what supplies the second direction is unidentified.** Three candidate
mechanisms were proposed and each refuted by a profile measurement it predicted. The *leading*
direction is explained — the gradient's support is set by where the two classes differ — but the
second is not. (A separate set of three mechanisms was proposed and withdrawn for a different
question, why discovery fails on SmolLM2; see Limits.)

## Limits

- **One model, one layer, one dictionary size — and the transfer failure is a failure of
  *discovery*, not of reachability.** In SmolLM2-1.7B a gradient write moves instruction-position
  bias at **every** depth tested (peak Δ +13.32 / +7.83 / +9.00 / +4.50 / +3.74 / +0.98 at layers
  6 / 9 / 12 / 15 / 18 / 21), so the factor is plainly there and plainly reachable. **Every learned
  arm fails to find it**: the crosscoder reaches +1.94 at best and sits below a *random* slab at
  two of six depths (−0.02 against +0.99 at L9; +0.80 against +2.39 at L12). Only Qwen2.5-0.5B is
  genuinely dead (gradient write +0.37).

  That is a worse result for the headline than the version this document carried until now, which
  said no write of any kind moved these models — **that sentence was false for SmolLM2 at all six
  depths.** The capability the headline claims is unsupervised discovery, and this is exactly where
  it fails.

  **The obvious confound is closed.** Those transfer runs used one learning rate for all arms at
  2000 steps, against the per-arm 6000-step recipes every headline uses — which finding 4 says is
  not a fair comparison. Rerun at SmolLM2 L6 with per-arm recipes, three inits, the crosscoder gets
  **worse** (+1.01 / +0.92 / +0.88 against +1.94), so the scope limit stands at the correct recipe.

  One caveat against over-reading it: SmolLM2's baseline is **+2.19** and the write pushes it
  further positive, so that arm *amplifies* an existing bias, while Qwen2.5-1.5B's baseline is
  −2.54 and its write is a *reversal*. Amplifying is the easier direction. The bias itself also
  **flips sign** across models.

  **The SmolLM2 failure splits cleanly by which slab an arm is derived from**, three inits at
  per-arm recipes (`recency_smolL6_rec_ds{0,1,2}.json`):

  | derived from `Ḡ` | | derived from `P_dom` | | learned | |
  | --- | --- | --- | --- | --- | --- |
  | `grad_slab` | +13.38 | `dom_slab` | +1.27 | `txc_slab` | +0.88–1.01 |
  | `grad_rank1` | +12.45 | `rank1_best` | +0.93 | `sae_broadcast` | +0.82–1.25 |
  | `broadcast_optimal` | **+3.87** | | | `random_slab` | +1.01 |

  **Its held-out reading AUC is 1.000 in all three inits while it reaches 0.07 of the optimal
  write** — reading the factor perfectly and steering it barely at all.

  ⚠ **The stronger version of that sentence does not survive its own convention.** Against
  `random_slab` the crosscoder loses or ties at **peak** dose (0.92 / 1.01 / 0.88 against 1.01) and
  **beats it in all three inits at matched dose** (against 0.79), because the two arms peak on
  opposite branches — the crosscoder at `α = +0.5`, the random slab at `α = −1.0`. Matched dose is
  this document's primary convention, so **"no better than noise" is withdrawn**; what stands is
  that it reaches a third of what a plain constant write reaches and an eighth of the optimum.

  **`broadcast_optimal` = +3.87 is the number that makes this a discovery failure.** A single
  constant direction, chosen with knowledge of the gradient, works four times better than what the
  crosscoder found unsupervised. So the SmolLM2 negative is not "nothing simple works here".

  **The split is an observation about this cell, and no mechanism is attached to it.** The
  `Ḡ`/`P_dom` ratio is **10.5×** at SmolLM2 L6 against **1.2× corpus-bound and 3.2× held-out** on
  Qwen2.5-1.5B — a clear separation, not an absence. But the tempting reading, that learned
  dictionaries track `P_dom` rather than `Ḡ`, **does not survive the full set**: across 45 cells
  carrying all three arms, `corr(txc_slab, dom_slab)` = +0.78 against `corr(txc_slab, grad_slab)`
  = +0.76, and the two slabs' own effects are **+0.99 correlated**, so the two hypotheses are not
  separable by this route. The largest discrepancies also run the wrong way — on `rotate12`,
  `txc_slab` +18.23 against `dom_slab` +133.60.

  **Normalised against the optimal write, SmolLM2 is a quantitatively worse cell, not a
  mechanistically different one:**

  | share of `grad_slab` | SmolLM2 L6 | Qwen2.5-1.5B L14 |
  | --- | --- | --- |
  | `broadcast_optimal` | 0.29 | 0.25 |
  | `sae_broadcast` | 0.06–0.09 | 0.16–0.18 |
  | `txc_slab` | **0.07–0.08** | **0.24–0.30** |

  The constant-write **ceiling is the same fraction in both models**. What differs is how much of it
  the learned arms reach — roughly 2–4×.

  **Three mechanisms were proposed for this null tonight and all three were withdrawn**:
  `cos(P_dom, Ḡ)` (refuted at 0.0535 against 0.0523 — indistinguishable); dictionary-tracks-`P_dom`
  (refuted by a 45-cell correlation, +0.78 against +0.76 with the two slabs' own effects +0.99
  correlated); and a `cos(v_sae, u₁(·))` account that passed a pre-registered test and was still
  wrong, because it used a stored `random_cos_baseline` field defined as `1/√(T·d)` — correct for
  the *slab* cosine it was written for, wrong for a cosine between unit vectors in `ℝᵈ`, where
  chance is `√(2/πd)`. At the correct baseline the alignment is 1.45× chance rather than 4×, both
  models sit *below* chance against `u₁(P_dom)`, and the arm-level version of the same question —
  `sae_schedule_grad`, the SAE's own direction given the gradient's schedule — has **overlapping**
  ranges across the two models (0.13–0.32 against 0.22–0.44), which the account requires it not to.

  **No mechanism is established.**

- **Discovery appears to transfer to a larger model, on a thin and internally split cell.**
  Qwen2.5-3B-Instruct at L18 (the same 0.50 depth fraction as Qwen2.5-1.5B's L14), **two inits**,
  matched dose: the crosscoder reaches **+34.75 / +33.83** against `grad_slab` +39.20 — **0.886 /
  0.863 of the optimal write**, three times its share on the 1.5B model (0.29) and an order of
  magnitude above SmolLM2 (0.07). It beats `random_slab` (+4.42) in both and matches the
  *supervised* `dom_slab` (+34.19) unsupervised in both, which is the best-supported part.

  ⚠ **The verdict splits between the two inits and splits on convention.** At matched dose it beats
  `sae_broadcast` in both (+15.88, +20.88); at peak dose it **ties in ds1** (+33.83 against +34.10),
  and the stored `win` flag — which is peak-based — is **False** there. So this is *discovery works
  at 3B in one init of two on the win criterion*, not a clean positive. Two inits is thin for a cell
  whose verdict flips between them: it is the unstable cell of the transfer set, as demonstration
  order is of the task set.

  **The negatives are far better replicated than this positive** — 0.5B and SmolLM2 have three inits
  each, `win = False` in all six. So "scale is not the axis" cannot rest on this cell. What it does
  establish is that the two failures are **not** simply evidence that 1.5B is special.

  ⚠ **And the form result holds here too**: `broadcast_optimal` reaches **+41.69**, which is
  **1.063× `grad_slab`** and above the crosscoder in both inits. That is now **seven of eight**
  tasks-and-models where the crosscoder does not exceed the best constant write. (A constant write
  exceeding the full optimal write means this cell sits outside the first-order regime the ratio
  assumes, so the 1.063 should be read as "at least as good", not as a violation.)
- **The `c` gate does not transfer across models.** Five of seven transfer cells sit below the
  `c` < 0.1 go-threshold with high `r1` and steer nothing. It was validated within one model and is
  not a cross-model instrument.
- **The discovery claim is an optimisation claim about a *search*, and the obvious explanation
  for it — that the crosscoder simply won the initialisation lottery — is refuted at n = 8.** On
  held-out instruction position, eight dictionary inits per architecture at matched dose:

  | | min | median | max |
  | --- | --- | --- | --- |
  | crosscoder | 2.76 | **4.66** | 4.92 |
  | SAE | −0.06 | 0.18 | **0.59** |

  **The ranges do not overlap.** The closest SAE draw is still **4.7×** below the
  crosscoder's *worst* draw (0.590 against 2.760), and six of eight crosscoder draws land in 4.57–4.92 while the SAE never
  leaves the noise band. Giving the SAE eight times the tickets does not close a gap of this
  size.

  ⚠ **Provenance: five of the eight files were recovered from run logs, not written by the
  harness.** A stale-variable bug raised `NameError` in the verdict block *after* compute
  completed, so those runs produced no JSON; the steering tables were complete in stdout and were
  parsed back. Arms, doses, SEMs, reading AUCs and baselines are exact; `sparsity`,
  `write_profile` and `rank` are absent because they were never printed. Files carry
  `recovered_from_log: true`.

  Still open: on the order task the crosscoder's selected latent does not hold a stable **sign**
  across inits, and this sweep scales the *init* lottery only.
- **Demonstration order is the unstable cell.** Its held-out peaks span 4.4× across inits (1.93 /
  0.68 / 0.44) and it loses to `sae_broadcast` at one init on the peak-dose convention while
  winning at matched dose. Instruction position and evidence order do not do this; the cell with
  the smallest absolute effects is the one whose verdict moves.
- **The selection lottery was tested, and it substantially qualifies the headline.** Each arm's
  latent was re-chosen by *measured* steering on a dedicated split (shortlist = top-16 by gradient
  alignment ∪ top-16 by reading AUC), then reported on a further split. Held-out instruction
  position, matched dose |α| = 0.5, three inits (`recency_tr_sel_ds{0,1,2}.json`):

  | arm | ds0 | ds1 | ds2 |
  | --- | --- | --- | --- |
  | SAE, reading-selected | +0.10 | +0.29 | +0.06 |
  | SAE, steering-selected | **+1.53** | **+1.75** | **+1.74** |
  | crosscoder | +3.81 | +4.82 | +4.63 |
  | **best possible constant write** (supervised) | **+4.01** | **+4.01** | **+4.01** |

  **The reading selector is badly wrong for both architectures, and which one it penalises more is
  task-dependent.** On instruction position it cost the SAE 6–30× and the crosscoder **nothing** —
  that cell's reading pick is already its best latent, which is why the headline z = +15 to +18 is
  robust to the selector. But that is the exception. On `rotate12` the selector cost the
  **crosscoder** 3.2–3.5× (+3.36 → +10.63) and on `rotate6` **6.5×** (+0.84 → +5.42) — more than it
  cost the SAE in both. Across 13 runs the median reading-pick rank is SAE 1866 and crosscoder 714
  of 4096: the crosscoder is better on average and nowhere near rank 1, and on `rotate2` and
  `rotate6` the SAE's pick is the better of the two.

  So the recommendation is not "select by steering for per-token dictionaries" but **select by
  measured steering for every architecture on every task** — precisely because which arm the
  reading selector hurts more cannot be predicted.

  ⚠ **This makes every reading-selected number in this document a lower bound.** All results
  predating the selection experiment — including the rotation and phase ladders as reported above —
  understate **both** arms, by 1.2× to 12× depending on the cell. The seven-task `broadcast_optimal`
  sweep is unaffected, since it used steering selection for every arm on every cell.

  ⚠ **The `broadcast_optimal` arm is the qualification that matters.** The best constant write in
  the *whole space* — not the best of 4096 atoms — reaches **+4.01** against the crosscoder's
  +3.81 / +4.82 / +4.63: **a tie in one init (z = −0.7) and z = +2.0 and +2.6 in the others.** The
  crosscoder exceeds the entire broadcast form by 0.95× to 1.20×, at or near the significance
  threshold in every init. Note this arm is built from the metric gradient, so it is a
  **supervised** reference in the same family as `grad_slab` — an unsupervised arm matching it is
  respectable, not a defeat.

  **What it shows is that the constant-write *form* is not what limits per-token dictionaries on
  this task.** The best constant write performs about as well as the crosscoder. What limits the
  SAE is that **its dictionary does not contain the good constant direction** — the best available
  one reaches 43–44% of the best possible one — and the reading selector then gives up most of what
  remains. The same conclusion follows independently from the `sae_schedule` comparison, which
  makes discovery-not-expressiveness a **measured decomposition** rather than an interpretation.

  ⚠ Whether this generalises beyond `recency` is running on `evidence` and is not yet known.

  Note `√c` = 0.185 predicts the best constant write's share and 0.248 was measured — the analytic
  value is a **first-order** ceiling and the matched dose already sits 34% outside it, in the
  direction that favours the broadcast arm.

## Post-sprint: the screen on published benchmarks

The sprint's tasks are constructs. **Instruction-position bias is not prompt injection** — it is
two hand-written formatting instructions at the same privilege level, positions swapped, which is
in the behaviour family Wu et al. (ICLR 2025, arXiv:2410.09102) study but is not their task. Their
benchmark is StruQ's, and it is public.

**The screen is training-free and costs about two minutes per task**, which changes the economics
of task selection entirely: screen broadly, spend GPU only where it says go. Two published
benchmarks have been screened so far.

### Prompt injection — StruQ

Qwen2.5-1.5B-Instruct L14, `n_docs` = 200, `n_grad` = 24, filtered run
(`results/txc_wins/geometry_struq_filtered.json`):

| attack | unsteered baseline | `c(Ḡ)` | `r1(Ḡ)` | `c(P_dom)` | `cos(P_dom, Ḡ)` |
| --- | --- | --- | --- | --- | --- |
| naive | +10.01 (z = 24.0) | 0.123 | 0.846 | 0.083 | −0.004 |
| ignore | +11.37 (z = 23.5) | 0.129 | 0.816 | 0.083 | −0.002 |
| `completion_real` | **+19.67** (z = 18.1) | 0.130 | 0.945 | 0.049 | −0.002 |

⚠ **The original pairing was defective and its numbers are superseded.** A = injected, B = the
same item clean makes **B 24 characters shorter**, and the segmenter cuts the whole document into
12 equal pieces — so segment `t` covered different content in A than in B and
`Ḡ = ∇[score(A) − score(B)]` averaged misaligned things. The signature was unmistakable:
`grad_slab` measured **−0.13** while `broadcast_optimal`, its own constant component built from
the same `Ḡ`, measured **+28.94**. A constant write cannot beat the full gradient write by 200×;
the position-varying components had cancelled between misaligned conditions, leaving mostly the
constant part.

**Repaired by making the pair position-matched**: the injection appears in **both** conditions and
only its *position within the data field* varies — end versus start — giving an exact anagram
(`len(a) == len(b)` and `sorted(a) == sorted(b)` asserted per item). That is also the
instruction-hierarchy question rather than a presence/absence contrast. `n_docs` = 200,
`n_grad` = 24 (`results/txc_wins/geometry_struqpos.json`):

| attack | baseline | z | `c(Ḡ)` | `r1(Ḡ)` | retention (× floor) | `c(P_dom)` |
| --- | --- | --- | --- | --- | --- | --- |
| naive | −1.94 | −5.5 | **0.072** | 0.794 | 0.416 (5.9×) | 0.030 |
| ignore | +0.50 | 1.2 | 0.103 | 0.837 | 0.325 (4.6×) | 0.019 |
| `completion_real` | **+8.19** | **12.9** | **0.084** | 0.910 | 0.437 (6.2×) | 0.012 |

**The misalignment was inflating `c`, as predicted: 0.123 → 0.072–0.084.** That moves prompt
injection out of the ambiguous band and into the go region, so the registered test of the gate at
its inversion point no longer applies to this task — it is now a straightforward go.

**Two positional findings fall out of the repair.** For `completion_real` the injection is far
more effective at the **end** of the data field (+8.19, z = 12.9); for `naive` it is more effective
at the **start** (−1.94, z = −5.5); and for `ignore` position does not matter at all (z = 1.2).
**The optimal injection position depends on the attack type** — which is a fact about prompt
injection rather than about dictionaries, and it is visible in the write profile: `u₁(Ḡ)` for
`completion_real` puts essentially all its mass on the final segment (1.00 against ≤0.04
elsewhere), exactly where a forged response boundary sits.

`completion_real` is the cell to run arms on: strong behaviour, `c` in the go region, retention
6.2× floor.

### The gate inverts on that cell, and the cause is dose — not the pairing

⚠ **`struqpos_completion_real` fails the `grad_slab ≥ broadcast_optimal` sanity gate**, which is
supposed to be unfailable: the unconstrained optimum cannot lose to a restriction of itself. Two
causes were proposed and only the second survives.

**A real second defect, which was not the cause.** `completion_real` forges the item's own answer
into the prompt — that is the attack — and that same answer was the metric's second continuation.
So `cont2` sat verbatim in the context on **70% of draws** (against 4% for `naive`/`ignore`), at
median 258 characters from the end in condition A versus 486 in B. The contaminant therefore
**displaces 3.7× further between conditions than the injected instruction does** (−228 versus −61
characters), and the metric was largely measuring copy distance on its own contrast string.
Scoring against a different item's answer takes the leak to **0%**, and costs a lot: the baseline
falls from **+8.19 to +1.69**, so **79% of the apparent behaviour was the contaminant**. The
geometry barely moves (`c` 0.084 → 0.086, `r1` 0.910 → 0.909) — the shape statistics were sound
while the effect size they were attached to was not.

**The registered prediction was mine and it was refuted.** I predicted the leak caused the gate
failure. The leak-free cell still fails at α = 0.5.

**The actual cause is dose saturation from gradient concentration, and the gate is sound.**
`u₁(Ḡ)` here is `0.00 0.01 0.01 0.01 0.01 0.03 0.03 0.00 0.04 0.03 0.00 1.00` — the gradient lives
almost entirely on the final segment. Both arms are normalised to unit **Frobenius** norm, so at
matched α the concentrated arm places ~`√T` = 3.5× more norm per position and leaves the linear
regime at roughly 1/3.5 of the dose. Sweeping α down
(`results/txc_wins/geometrystruqposx_lowdose.json`, `n_gate` = 120):

| α | 0.02 | 0.05 | 0.10 | 0.15 | 0.25 | 0.50 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `grad_slab` | +6.55 | +12.59 | **+17.29** | +16.48 | +6.41 | −5.86 | −3.56 |
| `broadcast_optimal` | +2.09 | +5.23 | +9.83 | **+11.52** | +4.91 | +0.97 | −0.88 |
| margin | +4.46 | +7.36 | +7.46 | +4.96 | +1.50 | −6.83 | −2.68 |
| gate | PASS | PASS | PASS | PASS | PASS | FAIL | FAIL |

A clean saturation curve. **The gate encodes a first-order argument and was being evaluated
outside the regime where first-order arguments hold.**

**This generalises past StruQ.** A fixed Frobenius dose is *not* a matched dose across arms whose
writes differ in concentration, so the sprint's default grid `[-2, -1, -0.5, 0.5, 1, 2]` with
matched dose 0.5 is entirely inside the saturated regime on any concentrated-gradient cell. This
is the "Frobenius versus injected norm" entry of the methodology list resurfacing **inside the
gate itself** — a ninth instance of the same pattern.

### Multi-turn escalation — SafeMTData

Crescendo-style jailbreaks: five turns, none individually harmful, where the extraction happens
only because of the arrangement. The foil holds the **payload turn fixed and last** and permutes
only the four context turns, so both conditions end with the same question and only what precedes
it differs — a single 4-cycle, rank 3 of a possible 3. Rows are filtered to those whose turns are
referentially self-contained (317 of 600 under a strict pronoun-plus-demonstrative filter; the
dataset's own illustrative row fails it). `k_seg = 5`, `n_docs` = 200
(`results/txc_wins/geometry_safemt.json`):

| filter | baseline | z | `c(Ḡ)` | `r1(Ḡ)` | retention (× floor) | `c(P_dom)` |
| --- | --- | --- | --- | --- | --- | --- |
| strict (317 rows) | −0.282 | −4.8 | **0.273** | 0.506 | 0.546 (7.7×) | 0.197 |
| loose (487 rows) | −0.245 | −4.4 | **0.296** | 0.464 | 0.578 (8.2×) | 0.198 |

**A clear stop.** `c` = 0.27–0.30 is nearly double the top of the ambiguous band and the highest
of any task screened. Retention is healthy at 7.7–8.2× floor, so this is a genuine constant-write
handle rather than an artefact of a badly-conditioned cell.

**This is the first scored test of the DC-handle analysis as a predictor, and it fails on the one
entry that carried the most weight.** The catalogue originally read this as "medium `c`,
permissiveness is a broadcastable mode". It was then **revised** to "medium-low, 0.05–0.12" on the
grounds that the mode argument holds for an *absolute* compliance metric but cancels against a
contrast between two orderings. Measured: **0.273**, more than double the revised band's ceiling
and above the original estimate too. **The revision moved the prediction in the wrong direction**,
and by the failure reading registered in advance, that means the framework's self-correction was
the error and the coarser original reasoning was closer.

⚠ **The baseline sign is counterintuitive and unexplained.** It is *negative*: the permuted
ordering is **more** compliant than the attack's own escalation order, at z = −4.8. That is
backwards from the premise of a crescendo attack. Either the sign convention is inverted somewhere
or the published escalation ordering does not confer its advantage at 1.5B. Not resolved here.

### Retrieved-document position — Liu et al

The matched foil **ships with the benchmark**: the same ten documents with the gold document at
positions 0, 4 and 9, verified across all 2655 items — exact multiset match, a single 10-cycle
permutation, Hamming 10/10. Rank = k − cycles = **9, the maximum at k = 10**, on every item, and
the segmentation is one retrieved document per span with no splitting rule to justify
(`results/txc_wins/geometry_litm.json`):

| pair | baseline | z | `c(Ḡ)` | `r1(Ḡ)` | `σ₂²/σ₁²` | retention (× floor) |
| --- | --- | --- | --- | --- | --- | --- |
| gold@0 vs @9 | +1.21 | 4.5 | 0.069 | 0.545 | 0.565 | 0.126 (1.8×) |
| gold@0 vs @4 | +1.20 | 4.8 | 0.068 | **0.489** | **0.741** | 0.150 (2.1×) |
| gold@4 vs @9 | +0.01 | **0.1** | 0.045 | 0.538 | 0.641 | 0.146 (2.1×) |

**The arms were run as a test of whether `c` and `r1` suffice — and the retention hypothesis is
refuted by its own registered falsifier.** Three inits, held-out, matched dose
(`results/txc_wins/litm_0v4_tr_arms_ds{0,1,2}.json`):

| arm | ds0 | ds1 | ds2 | median | kind |
| --- | --- | --- | --- | --- | --- |
| `grad_slab` (sanity ceiling) | +8.781 | +8.790 | +8.758 | +8.781 | supervised |
| `broadcast_optimal` (best constant write) | +4.368 | +4.400 | +4.387 | +4.387 | supervised |
| `grad_rank1` | +4.042 | +4.032 | +4.068 | +4.042 | supervised |
| `sae_schedule_grad` | +3.844 | +3.910 | +2.712 | +3.844 | supervised |
| `dom_slab` | +1.550 | +1.550 | +1.550 | +1.550 | supervised |
| `rank1_best` | +1.150 | +1.150 | +1.150 | +1.150 | supervised |
| **`txc_slab` (crosscoder)** | **+1.061** | **+1.127** | **+1.053** | **+1.061** | **learned** |
| `sae_broadcast` (TopK SAE) | +0.375 | +0.471 | +0.877 | +0.471 | learned |
| `sae_schedule` | +0.467 | +0.307 | +0.453 | +0.453 | supervised |
| `txc_slab_readingsel` | +0.356 | +0.628 | +0.396 | +0.396 | learned |
| `sae_broadcast_readingsel` | +0.262 | +0.260 | +0.413 | +0.262 | learned |
| `tsae_broadcast` (attention tSAE) | +0.017 | +0.116 | +0.300 | +0.116 | learned |
| `txc_flat` (profile removed) | +0.185 | +0.084 | +0.111 | +0.111 | learned |
| `random_slab` | +0.073 | +0.073 | +0.073 | +0.073 | null |
| `random_broadcast` | +0.067 | +0.067 | +0.067 | +0.067 | null |
| `txc_profile_random` | +0.023 | +0.234 | +0.057 | +0.057 | null |
| **ratio to best constant write** | 0.24 | 0.26 | 0.24 | **0.24** | |

**The crosscoder is the best of every arm a practitioner can actually build** — 2.3× the TopK SAE
and 9× the attention tSAE — and it also clears two supervised arms (`sae_schedule` at +0.45, and
it sits just under `rank1_best` at +1.15). **The temporal-profile controls hold cleanly**: flatten
the crosscoder's profile and it falls to +0.111, against a random slab at +0.073. The
position-dependence is carrying the effect, not the direction.

The prediction registered in advance was that near-floor retention would cap every arm and make
the *ordering* unstable across inits, with the stated falsifier: **a clean repeatable crosscoder
win at retention 2.1× would show retention does not cap what it was claimed to cap.** That is what
happened — the crosscoder wins 3 of 3 against the SAE at a 2.3× margin, and its own values span
0.074. **Retention is not a third gate.** A task can have near-floor retention and still support a
stable architecture separation.

What retention *did* predict correctly is the absolute scale: +1.06 against prompt injection's
+6.3, on a baseline of +0.54. Small but reproducible.

And the seven-task pattern holds here too — **ratio 0.24 to the best constant write**, so the
search advantage is real and the form advantage is absent, exactly as everywhere else.

**The best shape statistics anything has screened** — low `c` *and* genuine rank ≥ 2 together,
which no construct in this sprint achieved — **on a task that may be unsteerable for an unrelated
reason.** See the retention discussion in finding 3.

**Liu's U-shape does not reproduce.** Position 0 beats positions 4 and 9 by an identical amount
while 4 and 9 are **indistinguishable** (z = 0.1). That is monotone **primacy**, not the
beginning-and-end advantage the paper reports, so this is *a primacy task built from Liu's data*
rather than Liu's task. Two readings are available and this run cannot separate them: a 1.5B may
lack the end-recovery Liu measured at GPT-3.5/Claude scale, or teacher-forced
`logP(gold) − logP(distractor)` may not track generated-answer accuracy.

⚠ **`c` = 0.045 on the gold@4-vs-@9 cell is the lowest any published benchmark has screened — on a
cell with no behaviour at all.** Reading geometry before the baseline would have produced a
headline from a null. The baseline-first rule was written for exactly this and this is the first
time it fired.

## Methodology: the name was not the thing

The most transferable output is a pattern, found eight times, each by reading our own code rather
than by a result looking wrong: nominal `k` versus realised L0; training versus held-out sparsity;
a one-sided dose grid; Frobenius versus injected norm; a screen with no baseline field; a registry
name resolving to a different generator; a field called `z` that was peak-dose while the analysis
reported matched-dose; and a figure script indexing signed rather than matched α.

**Every one was a quantity nobody thought needed checking, because it had a name implying it was
already right. None of them errored.** Three were silently disadvantaging the arm we were arguing
*for*, and one was the sole support for a result now withdrawn.

A second pattern appeared late: **a mechanism inferred to explain a discrepancy and asserted before
the files were opened** — six times by the end, of which the last three were caught before
shipping rather than after. **The rate of bad hypotheses did not fall; the discipline changed.**
The most instructive was the last: a pre-registered test with a stated falsifier, which *passed*
and was still wrong, because it measured against a stored `random_cos_baseline` field whose name
matched the quantity and whose definition did not — `1/√(T·d)`, correct for the slab cosine it was
written for, wrong by 2.8× for a cosine in `ℝᵈ`. **A passing pre-registered test is not protection
when the yardstick is borrowed.** The resolving check was always to print the
configuration fields next to the number, and it always came last.

A third is the most uncomfortable, because in each case **a real check was run and it passed**. A
rename was verified by confirming the new code reproduced stored values on an existing file — which
exercised the renamed reference and not the two stale ones, so five runs later crashed after
completing their compute. A red-team pass verified every number against its file without verifying
the *convention* those numbers were computed under. A held-out split was verified as disjoint —
16/16, zero overlap — while the task names resolved to a different generator entirely. **In all
three the verification confirmed the property it tested for, and the property that mattered was a
different one.** A passing check reads as reassurance, which makes this failure mode harder to
catch than an unchecked assumption.

A fourth instance, post-sprint, is the same shape in a *query* rather than a check. A corpus-wide
claim — "the pattern holds across ~71 files" — was computed over the 71 files carrying a
**different field** than the one being claimed about, out of 98 that carry the relevant one.
**The population described was not the population searched**, and the claim it produced (that a
per-token dictionary had never failed to read a factor) was false by six files. The scoped claim
in the document was correct throughout; only the generalisation drawn from it was wrong.

**A third class, mechanical and separate from both**: **derived counts go stale whenever runs land
and nothing recomputes them.** The result-file count was wrong three times in one night (77 → 81 →
114), the detector's cell counts drifted as new runs arrived, and a "wins 8 of 9" tally was correct
when written and 7 of 9 by the time it shipped. None of these is a reasoning error and none would
be caught by checking the reasoning. **A single recompute-derived-counts step before shipping
catches all of them**, and its absence accounted for three of the seven issues in the final review
pass.

**And the sharpest evidence that these are hazards rather than lapses: three people on this sprint
independently made the *same* one.** Reading a steering arm at the signed positive dose rather than
at matched magnitude with the sign free scores any arm whose correct direction is negative as a
failure. It withdrew the previous sprint's headline; it appeared in a figure script written at hour
eight; and it appeared again in the red-team pass over this document, in two of five reported
issues. **A trap that catches the people actively studying it is worth more attention than one that
only catches novices.**

**And it is mechanically detectable — but not in the form first proposed here.** "Flag any cell
where arms peak on different branches" fires on **79 of 86** symmetric-grid cells: almost every cell
has some arm peaking on the minus branch, so it cries wolf and would be ignored. The discriminating
test is whether the **verdict** changes — crosscoder minus best constant arm, read at signed `+α`
against sign-free matched magnitude. That is `scripts/peak_sign_flags.py`, and it gives the number
worth quoting:

> **Twenty-seven of 79 flagged cells — 34% — change verdict on the dose convention alone.**
> Not the effect size. The verdict.

It flags `order_sym_ds0`, the withdrawal cell (+0.57 signed against −3.24 sign-free). **And it flags
two of the three headline cells**: `recency_tr_sel_ds0` at −3.26 against +2.28 and
`recency_tr_sel_ds2` at −3.61 against +2.89, with only ds1 stable.

**So the headline is convention-dependent too.** It survives because the sign-free convention is the
correct one and this sprint established that on independent grounds — but under signed-positive
indexing the headline would have inverted in two inits of three. That makes the withdrawal in
finding 1 not a one-off blunder but **the same trap, catching the result we kept as well as the one
we dropped.**

## Post-sprint: a genuine-event steering benchmark on prompt injection

The sprint's other cells are scored by a teacher-forced log-probability margin. This one is
scored the way the paper's backtracking headline is: a **judge-scored count of genuine,
coherent events** under steering, baseline-corrected. It is also the first time any arms have
run on a *valid* StruQ pairing — every earlier arms file used the defective pairing whose
sanity gate fails.

**Setup.** `struqposx_completion_real` (position-matched, leak-free), held out, 7400
generations over 37 cells at 200 rows per cell, ten arms including `txc_flat` (= the paper's v7
write) and `random_slab`/`random_broadcast` as nulls. Dose grid capped at `|α| ≤ 0.25`: the
default `[-2 … 2]` sits past this task's saturation point, and `grad_slab` here peaks at
α = +0.1 and is already declining by +0.25.

**The metric matters, and the weak attack proves it.** On `naive`, 16.5% of generations *echo*
the injected sentence and only 1.7% comply — so StruQ's own substring metric
(`"hacked" in output.lower()`) reports **31.6% attack success** where the true rate is near
zero. Steering is the intervention most likely to manufacture echoes and garbled copies of the
input, so a substring metric systematically rewards whichever arm damages the model most. On
`completion_real` the attack actually lands: 86% of generations comply, unsteered rate
**0.910**, held-out baseline z = 4.3. With the rate that high there is no room to *induce*, so
the informative direction is **suppression** — steering that makes the model stop obeying.

**Result, three dictionary seeds.** Judged with `gpt-4o`; a cheap `comply_lead` proxy was
validated against the judge on a stratified sample per seed (91.9%, 97.3%, 97.3% agreement) and
carried the full n, with that decision recorded in each output. The table is each arm's steered
rate at its best dose — **lower suppresses more** — against a baseline of 0.910.

| arm | s0 | s1 | s2 | mean suppression | sd | beats best null |
| --- | --- | --- | --- | --- | --- | --- |
| `broadcast_optimal` — *supervised* | 0.495 | 0.490 | 0.495 | **+0.417** | 0.003 | **3/3** |
| `txc_slab` | 0.850 | 0.580 | 0.810 | +0.163 | **0.146** | 1/3 |
| `tsaep_broadcast` — published T-SAE | 0.715 | 0.780 | 0.750 | +0.162 | **0.033** | 2/3 |
| **`random_broadcast` — NULL** | 0.765 | 0.765 | 0.765 | **+0.145** | 0.000 | — |
| `tsae_broadcast` — attention tSAE | 0.690 | 0.835 | 0.795 | +0.137 | 0.075 | 1/3 |
| `sae_broadcast` — TopK SAE | 0.870 | 0.600 | 0.875 | +0.128 | **0.157** | 1/3 |
| `random_slab` — NULL | 0.840 | 0.840 | 0.840 | +0.070 | 0.000 | — |
| `txc_flat` — *the paper's v7 write* | 0.870 | 0.855 | 0.850 | +0.052 | 0.010 | **0/3** |
| `dom_slab` — *supervised* | 0.895 | 0.895 | 0.895 | +0.015 | 0.000 | — |

**Three findings, in order of how much they transfer.**

⚠ **1. No learned dictionary arm reliably beats a random constant write.** Every learned arm's
mean sits within ±0.035 of the null's +0.145, and none clears it on more than 2 of 3 seeds.
Only the supervised gradient-derived write separates, and it does so by 3× on every seed.

⚠ **2. The instability is in the METRIC, not in the dictionaries.** The ranking by genuine-event
count is not seed-stable — `txc_slab` ranges 0.850 / 0.580 / 0.810, so from seed 0 you would
report *"the published T-SAE suppresses 3.3× more than the crosscoder"* and from seed 1 *"the
crosscoder suppresses 2.5× more than the published T-SAE"*. But the cause is **not** that the
seeds learn different-quality latents. On the teacher-forced margin the same dictionaries are
consistent:

| arm | margin, 3 seeds | CV | event suppression, 3 seeds | CV |
| --- | --- | --- | --- | --- |
| `txc_slab` | 2.43 / 3.22 / 2.26 | **0.19** | 0.060 / 0.330 / 0.100 | **0.89** |
| `sae_broadcast` | 2.23 / 1.16 / 1.48 | 0.34 | 0.040 / 0.310 / 0.035 | **1.23** |
| `random_broadcast` | 0.76 / 0.76 / 0.76 | 0.00 | 0.145 / 0.145 / 0.145 | 0.00 |

The two metrics agree in expectation — **r = +0.909** across arms — and on the margin the
crosscoder beats the best null by 3.5× here, mid-pack against recency (2.5×), evidence (1.1×)
and LitM (5.9×). What the event count adds is not signal but noise, of a specific kind. It is a
**binary rate against a 0.910 ceiling**, so it compresses: `broadcast_optimal` carries 7× the
crosscoder's margin (18.78 against 2.63) yet only 2.6× its event suppression, and a 3.5× margin
advantage over the null compresses to 1.12× in events. And the residual variation is not
sampling noise — binomial at n = 200, p ≈ 0.8 gives se ≈ 0.028 against an observed swing of
0.27, an order of magnitude larger. It is **threshold noise**: on seed 1 the crosscoder's
dose-response crossed a tipping point at α = −0.25 and compliance collapsed to 0.580; on the
other two seeds it never crossed.

**This still supports Reviewer 1 — a single-seed genuine-event count is a lottery — but the fix
is not simply more seeds.** The readout needs headroom (a behaviour whose baseline is not
already at 0.91) or a dose grid dense enough to locate the tipping point. Averaging a saturating
binary readout over seeds converges slowly and to a quantity that still compresses real
differences.

**3. The published T-SAE is the most consistent learned arm on the event readout** (sd 0.033
against the crosscoder's 0.146 and the SAE's 0.157) and the only one to beat the null on a
majority of seeds. Note this is consistency *of the readout*, not of the dictionary: on the
margin the ordering of stability reverses, with `tsaep_broadcast` the least stable arm
(CV 0.62 against the crosscoder's 0.19). The two views disagree about which architecture is
reliable, which is itself a reason to report both rather than either alone.

**And the paper's own write is the weakest thing measured.** `txc_flat` — V7 up to a scalar —
suppresses +0.052, below both nulls, on **0 of 3** seeds. Whatever the crosscoder's latent
encodes about this task, averaging it over the window destroys it.

**A budget caveat that survives the instability.** The attention tSAE carries **675.84
uncharged coefficients per segment** — its *predicted* codes are computed from context and only
its 8.00 *novel* codes are billed — so any number it posts is bought on an axis nobody else is
spending on. The published T-SAE has no uncharged component (7.74 per segment, slightly *under*
the 8.00 budget). This is also the arm that spent most of the sprint mislabelled: every other
`tsae_*` number in this document is the attention architecture, not arXiv:2511.05541.

**What is stable, across all three seeds:** the supervised best-constant-write suppresses ~0.42,
roughly 3× the best null and beyond every learned arm’s mean; and `txc_flat` — the
paper’s v7 write — stays at +0.05, below both nulls on every seed.

**Degeneracy is ruled out, so the suppression is genuine.** `repeat_frac` ≈ 0 everywhere,
log-probabilities near baseline, replies *longer* than baseline, and the suppressing arms
produce coherent refusals — `broadcast_optimal` at α = −0.25 emits
`"Sorry, I can't assist with that."` rather than damage.

**What this says about the screen.** `c` = 0.057 on this task, meaning a constant write captures
only √c ≈ 24% of the optimal inner product — yet constant writes dominate the *behavioural*
outcome and the slab writes do least. The teacher-forced geometry does not predict generation
behaviour here, which is the same lesson as finding 2 arriving from a different direction.

⚠ **Limits.** One dictionary seed so far. The crosscoder's best dose is at the grid edge
(+0.25), so its curve may not have turned over — but the cap is there because the gate fails
above it, so a larger dose would measure saturation rather than steering.

## Post-sprint: the steering protocol writes only the window-average

The largest correction this work has produced, and it is not about a task or a dictionary — it
is about what the published steering hook writes into the residual stream. Full derivation and
proofs in [[steering_conventions]] (`docs/dmitry/reviewer_responses/steering_conventions.tex`).

**The paper's default protocol is `txc_flat`.** `temp_bench/case_studies/steering.py` selects
`protocol="v7"` by default in both `SteeringConfig` and `experiments/c5_steering/run.py`. V7
tiles the prefix into `T`-blocks, clamps the latent, decodes, and then **averages the
per-position delta over the window** before writing it to every position in the block. Because
a crosscoder decodes as `einsum("bs,std->btd", z, W_dec)` with a window-level `z` carrying no
`t` index, clamping latent `j` gives `delta_t = (s − z_j)·W_dec[j,t,:]`, so V7 writes
`(s − z_j)·mean_t W_dec[j]` — which is exactly this sprint's `txc_flat`, the arm built as the
control that *removes* the temporal profile.

Split a slab as `P = 1_T ⊗ P̄ + P̃` with `Σ_t P̃_t = 0`. The DC part is precisely what a
per-token dictionary can already express, so **`P̃` is the only component on which a crosscoder
can outperform, and V7 deletes it.** The `pp` fallback is a convolution of the slab with the
per-window clamp scalar and collapses to the same constant write wherever that scalar is flat.

**Measured on the saved ward checkpoints** (`results/txc_wins/protocol_compare3.json`,
`experiments/ward_backtracking_txc/`), with V7 and PP copied verbatim from the reference and
`slab` differing from V7 by one line:

| check | predicted | measured |
| --- | --- | --- |
| slab/V7 injected-norm ratio, AC-share 0.497 | `1/√(1−AC)` = 1.409 | **1.409** at all four doses |
| same, second latent, AC-share 0.307 | 1.410 | **1.406** |
| PP versus V7 injected norm | equal when the clamp is flat | within **1%** at every dose |

⚠ **What this does *not* establish is that the discarded component buys behaviour.** At matched
`q`, slab beats V7 at z = +3.77 (q=2) and +2.20 (q=4) and *loses* at z = −2.38 (q=8) — the sign
flips with dose. Matched `q` is not a matched dose, since slab injects 1.409× more norm at every
`q`, and neither dose-response is monotone. The comparison worth reporting is V7 at
`q' = 1.409q`, i.e. matched **injected norm**; it has not been run.

**The related control finding.** `txc_flat` — hence V7 — sits within 4–20% of a *random* constant
write of the same norm on three of four held-out cells (recency 1.11×, evidence 1.04×,
demonstration order 1.20×), and separates only on LitM (3.8×). So on most cells the published
write is not carrying latent-specific information at all. The generation sweep that ships had
**no random arm**, which is why this was invisible; `random_slab`/`random_broadcast` are now
mandatory in `gen_arms`. Figure: `plots/2026-07-27_protocols/paper_vs_slab.png`.

⚠ **A baseline arm was mislabelled for the whole sprint.** Every `tsae_*` number above came from
`temporal_crosscoders.han_tsae.TemporalSAE` — an **attention** architecture (`n_heads=8`,
`n_attn_layers=1`) that this repo aliases `tsae_paper`. The published temporal SAE
(arXiv:2511.05541) has no attention: it is a per-token BatchTopK SAE with matryoshka groups, an
InfoNCE term between consecutive positions, AuxK and threshold inference. The real one is now
vendored (`tsae_bhalla.py`) and run as a fourth arm on LitM, where it is the **best reader** of
the four (AUC 0.720–0.787, above the TopK SAE's 0.703–0.782 and the crosscoder's 0.571–0.672)
while steering no better than the TopK SAE. It runs at **L0 12.62 against everyone else's 8.00**
because its EMA threshold replaces TopK at inference, so the sparsity match is broken and a
matched-L0 rerun is owed. The correction changes that arm's numbers; it does not move it across
the rank argument, since its `W_dec` is `(d_sae, d_in)` and a steered latent still reaches only
rank 1.

## Where things live

- Code: `experiments/temporal_screen/txc_wins/` — harness, task designs, Modal runners
- Protocol comparison: `experiments/ward_backtracking_txc/{steering_vendored,protocol_compare_run}.py`
- Results: `results/txc_wins/` (118 files), figures in `plots/2026-07-26_txcwins/` and
  `plots/2026-07-27_protocols/`
- Derivation note: `docs/dmitry/reviewer_responses/steering_conventions.{tex,pdf}`
- Next experiments, in priority order: `next_ten_hours.md`
