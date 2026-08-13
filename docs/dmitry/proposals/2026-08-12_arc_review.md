---
author: Claude (review agent)
date: 2026-08-12
tags:
  - reference
  - results
---

## Arc review: the diffusion/DSM temporal-dictionary program at 48 hours

Commissioned question: the program ran from synthetic signs-of-life
([[2026-08-10_bird_temporal_codes]], [[2026-08-10_bird_clock_results]])
to LLM behavioural evals
([[2026-08-11_backtracking_detection_dsm]],
[[2026-08-11_backtracking_steering_dsm]]) in about 48 hours, and the
PI's concern is that "we have diverged from initially promising results
to stuff that hasn't worked." This review answers three questions:
what initially worked, what changed as the work scaled and transferred,
and how the arc should be understood. It reviews the research, not the
prose. It incorporates two results that landed the morning of writing:
the mixed-corpus steering-site gate failure and the mixed-corpus
detection result (the w6mix pair). The wave-2 steering aggregation was
still pending at time of writing and nothing below depends on it.

Summary of the verdict up front: the arc is not "promising, then
failed." It is "promising, then mostly *unfairly tested* at the
deployment site" — with the unfairness now diagnosed to a specific,
measured mechanism (direction-deep OOD collapse of DSM dictionaries),
one cheap fix already falsified honestly (text-domain mixing does not
fix the cross-model site), and, as of this morning, the first fair-ish
deployment-site test producing the program's first behavioural win
(w6mix detection, one seed). The program's central behavioural claims
remain mostly in the "never fairly tested" cell, and the fair test is
cheap. The largest genuine liabilities are not the failures — they are
two places where the docs' framing ran ahead of the evidence, detailed
in the audit section.

### 1. Claim taxonomy

Cells: **(a)** worked and survived transfer; **(b)** worked in
synthetic but failed on transfer; **(c)** failed for a diagnosed reason
that is itself informative; **(d)** never fairly tested. "Transfer"
means: to LLM activations, and separately to the deployment
(reasoning-trace / distill-model) distribution — several claims sit in
(a) for the first hop and (d) for the second.

| # | claim | origin | evidence now | cell |
| --- | --- | --- | --- | --- |
| 1 | BIRD correspondence: entropy/L0 laws exact, $W_c(\sigma)$ frontier, coding vs generative transition at $h{+}1$ / $h{+}2$ | theory + A1–A3 | confirmed to numerical precision; A3 adds defect kinetics | (a) |
| 2 | "Reconstruction cannot force binding; denoising makes it loss-bearing" (naive form) | theory §3.5 + A4 | falsified as stated: recon volunteers partial binding; refined to a tie-breaking claim; A4′ never run | (c) |
| 3 | Posterior head + DSM closes 94% of Bayes gap | B1 | true on the clock; the head collapses on coupled HMM and loses on continuous manifolds (B2) — a discrete-template specialist | (a) on-domain, (b) for generality |
| 4 | DSM ≥ recon for the TXC across synthetic settings | B2 | strict wins on 2 of 4 settings (tone tasks, disjoint seed ranges); ties on denoising and coupled; gain concentrated sub-Rayleigh | (a), narrower than the headline |
| 5 | No interpretability-for-fidelity trade; DSM prunes the junk tail (preregistered) | B3 | confirmed (P1–P3, P5); P4 falsified on quasi-discrete data — a caveat that has not yet been chased on LLMs | (a) with a live caveat |
| 6 | Absorption reduction on LLM per-token SAEs | topk_vs_topkdiff | −42% at 10M, replicates and sharpens at 100M (recon absorbs faster with training); bayes_gate lowest of all | **(a)** — the cleanest transferred win |
| 7 | Perturbation-robustness of DSM supports (motivation 1a) | topk_vs_topkdiff | survives 10× scale-up; but it is near-tautological (the training objective evaluated) and its off-distribution converse is claim 13 | (a), discounted |
| 8 | Sparse-probing / small-k detection advantage for DSM (preregistered) | synthetic B2 → Gemma | failed at 10M, still a wash at 100M; dsm_anneal's reduced death did not unlock it | **(b)** — the one clean synthetic-to-LLM failure |
| 9 | Gate swap: +3–5 points k=5 probing free; absorption/robustness weight-borne ([[2026-08-11_gate_swap_note]], [[2026-08-11_jumprelu_mmse_note]]) | Gemma | robust across all six checkpoints; transfers *nothing* to per-token backtracking detection (clean null); one suggestive window cell (+0.014, 1 arm 1 seed) | (a), scope narrowed to mean-pooled multi-class readout |
| 10 | Trained bayes_gate: absorption + probing wins | topk_vs_topkdiff | breaks the probing wash (+7–11 pts) at 1 seed, unmatched budget/L0; fragility regression is dictionary-borne and real | (a) provisional / (c) for robustness |
| 11 | Per-token dictionaries add behavioural detection signal | detection doc | all per-token arms at the raw floor, both objectives | (b)-adjacent null, but see audit item 3 |
| 12 | Temporal architecture alone explains the stage-B detection edge | w6 trio 2×2 | no: temporal-on-FineWeb ≈ +0.006, domain ≈ +0.012, both ≈ +0.025 — and the w6mix pair shows the "domain" term is really an objective×domain interaction | (c) — decomposition achieved |
| 13 | DSM OOD collapse is direction-deep (recalibration revives 214→215 of 16,384) | steering pre-flight + recalibration probe | measured three ways at two scales/hookpoints; the single most solid *new* LLM phenomenon of the arc | **(c)** — informative failure |
| 14 | Text-domain (mixed-corpus) training fixes the DSM transfer failure | predicted in both eval docs | **split verdict this morning**: fixes the detection site (base-model activations over trace text), does nothing for the steering site (distill-model activations: NMSE 0.807, 5.2% live) | (c) — the cheap fix is half-dead, model identity now the lead suspect |
| 15 | w6mix_dsm detection win: 0.208/0.242 (S8/S32), matching stage-B TXC, while recon barely moves | mix pair, this morning | first DSM behavioural win; an interaction, not a main effect; 1 seed, folds overlap, dead pool NOT revived (94.8%) — mechanism prediction failed while outcome succeeded | **(a) provisional** — needs seeds + controls |
| 16 | Stage-B TXC slot-0 is a directional handle (+0.42); random control nullifies DoM steering (+0.015) | steering wave 1 | strongest causal result of the arc; note it vindicates the *recon-trained, trace-trained temporal crosscoder* and the control methodology — not DSM | (a), not a DSM result |
| 17 | Manifold-projected steering (denoise-after-steer, motivation 1b) | steering wave 2 | projector destroys generation at α=0; pre-registered reading (projector damage, not steering dynamics) applies | **(d)** — fair test needs a deployment-alive projector; moderate cost |
| 18 | DSM finds causally better steering directions | wave 1 ours_dsm | negative excess (−0.125) but the arm entered OOD-capped at 3.7% live — uninformative under transfer failure | **(d)** — same prerequisite as 17 |
| 19 | w6_bayes at T=6 | wave 2 | degenerate (94.7% dead), diagnosed: mean-gate controller; rate-KL fix known and validated at Gemma scale | (c) |
| 20 | Matryoshka × DSM composition; timescale spectrometer; density/anomaly monitoring upside | README motivations | never run | (d) — cheap to moderate |

Reading of the taxonomy in one paragraph: the synthetic program (1–5)
delivered essentially what it claimed, with two honest self-corrections
(2, 3) logged the same day. The per-token LLM program (6–10) transferred
the *structural* wins (absorption, robustness, gate semantics) and
failed to transfer the *detection* win (8) — in-distribution, so that
failure is real and not excusable by transfer artifacts. The
behavioural program (11–19) was, until this morning, almost entirely
uninformative about DSM because every deployment-site read was
out-of-distribution for the dictionary under test (13); the one arm
now tested on its own training distribution (15) produced the first
win. The steering site remains untested for every "ours" arm.

### 2. What changed: the discontinuity

Candidates evaluated against the record:

- **Scale** — not supported. The structural wins sharpened from 10M to
  100M tokens; the probing wash persisted unchanged; nothing flipped
  sign with budget.
- **Per-token vs temporal** — partially supported, but weaker than the
  house story. The synthetic detection advantage lived in
  temporally-bound structure, and per-token LLM probing washed; but the
  w6 2×2 measured the temporal-alone increment at ≈ +0.006 PR-AUC —
  small — and the only clean temporal-vs-per-token LLM comparison at
  matched corpus/site still does not exist.
- **Behavioural evals having different sensitivity** — real as a
  *cap*, not as the discontinuity. On the sentence set at S=8 every
  arm sits within ~0.05 of the raw floor and the raw floor within
  ~0.06 of base rate, with fold spreads ±0.03. The instrument can
  barely resolve the differences being claimed at S=8; S=32 and the
  far set have somewhat more headroom.
- **Train/eval activation-distribution match** — **supported, and now
  supported twice over.** This is the single largest discontinuity.

The evidence pattern: every setting where DSM won was one where the
dictionary was evaluated on the distribution it was trained on —
all synthetic settings (train and eval drawn from the same generative
process, corruption matched to the generative noise), and the Gemma
structural evals (pile-trained, pile-evaluated). Every setting where
DSM failed catastrophically was one where it was read off-distribution
(FineWeb-trained dictionaries on distill-model reasoning-trace
activations), and the failure mode is the measured direction-deep
collapse (claim 13): the objective that prices density specialises
the encoder to the training density, so distribution mismatch converts
into capacity annihilation rather than graceful degradation. Recon,
density-blind, degrades shallowly (threshold-recalibrable).

This morning's two results turn that correlation into something close
to a within-subject dissociation, using the *same trained artifact*
(w6mix_dsm) at two sites on the same day:

- At the **detection site** — base-Llama activations over trace text,
  which the 72/28 mixed corpus covers, i.e. the trained distribution —
  DSM flips from worst-of-ours to best-of-ours (0.181 → 0.208 S8,
  0.209 → 0.242 S32) and matches stage-B TXC, while recon barely moves
  (0.196 → 0.190 S8). An objective×domain interaction, exactly what a
  density-learning account predicts and what an additive
  "domain lifts everything" account (which the w6 2×2 section
  predicted) does not.
- At the **steering site** — distill-model activations, which no
  text-mixing through base Llama can cover — the same checkpoint still
  collapses: NMSE 0.807 / 5.2% live vs 0.795 / 3.7% for the
  FineWeb-only version. Text-domain coverage bought essentially
  nothing on the model-identity axis.

So "distribution match" was the right axis all along, but the
pre-mix docs cashed it out at the wrong granularity: the binding
quantity is the *activation* distribution, which is a function of
(model, text) jointly, and this morning's gate failure shows the model
factor is not a rounding error — it is, at the steering site, the
whole failure. One caution against over-rotating: the mechanism
prediction attached to the mix run (dead pool revives) failed even
where the outcome succeeded — w6mix_dsm is still 94.8% dead on traces,
and its win comes from the ~850 live latents becoming the right ones.
Extreme pool concentration looks intrinsic to DSM, not an OOD symptom;
what distribution match changes is what the concentrated pool encodes.
The distribution story as currently held is therefore one level less
understood than the docs' confidence suggests.

Secondary discontinuity, worth keeping on the books: the synthetic
corruption process was the *true* generative noise; on LLMs, isotropic
Gaussian at σ·RMS is a modelling choice, and the one synthetic setting
with quasi-discrete event structure (coupled HMM) is exactly where B3's
P4 falsified the interpretability prediction. If distill-activation
training (experiment 2 below) does not restore DSM, the corruption
model is the next suspect, and it was flagged in advance.

### 3. Motivated-reasoning audit

Places where our own framing outran the evidence, in decreasing order
of severity.

- **"Coverage of the deployment distribution is the binding constraint
  on DSM everywhere."** Written in the w6 detection section as a
  settled conclusion *before* the mix experiments ran, generalised from
  one deployment distribution, and stated at the wrong granularity —
  "coverage" silently meant text-domain coverage, and this morning's
  gate shows text coverage does nothing for the cross-model site. Half
  the claim survived (detection site), half died (steering site). The
  general lesson: the docs repeatedly convert "leading hypothesis"
  into "constraint" language one experiment early. The same tic
  appears in "Trace-domain (mixed-corpus) training is the decisive
  next run" — it was decisive, but partly by falsifying the mechanism
  it was launched to confirm.
- **"Per-latent, the surviving DSM features are far more informative"
  (from near-parity at 3.7% live).** At the time it was written this
  inference was not licensed. All w6 arms sat within ~0.015 of the raw
  floor with fold spreads of ±0.03; near-parity between a 600-latent
  dictionary and a 15k-latent dictionary is equally consistent with
  the detection task being mostly insensitive to dictionary quality —
  raw activations with no dictionary at all score 0.190. The w6mix
  result has since made the claim *plausible* (a same-sized pool,
  on-domain, now carries stage-B-level signal, +0.026 over raw at
  sentence S=32 and +0.039 at far S=32, columns with more headroom
  than S8) — but plausible is not shown, at 1 seed, and the direct
  test costs approximately nothing: subsample the recon dictionary to
  ~850 random live latents and re-run the probe. That control should
  have been run before the sentence was written; it still has not
  been.
- **"The program's central prediction is supported by the failure."**
  In the per-token detection round, the reading that per-token
  dictionaries at the raw floor *support* the temporal thesis leaned
  on the stage-B temporal arms leading the table — arms the same
  document correctly labels as unmatched in corpus, hookpoint, and
  budget ("orientation arms... not a matched comparison"). Using them
  as supporting evidence two sections later is having it both ways.
  The subsequent w6 2×2 partially self-corrected (temporal-alone ≈
  +0.006, the smallest term in the decomposition), and the mix pair
  has now reassigned most of the residual to an objective×domain
  interaction. The temporal-binding claim for LLM behavioural signal
  is currently supported mainly by the *steering* ladder
  (nothing → per-token +0.069 → windowed +0.42), which is itself built
  from heterogeneous arms with n = 1 per rung.
- **Motivation 1a ("robust features as causal handles, 0.74 vs
  0.60") ranked first while its measured converse sat in fine print.**
  The support-overlap win is the training objective evaluated —
  the README's own eval-round table says as much — and the
  deployment-relevant robustness (distribution shift) went the
  opposite way, catastrophically. The motivations ranking kept 1a at
  the top after the OOD-collapse measurement existed; the honest
  ranking as of 2026-08-11 evening had the flip side dominating the
  headline benefit.
- **"DSM ≥ recon across all four synthetic settings"** compresses
  2 strict wins, 2 ties (one of which — denoising — the results doc
  itself says "everything ties... no red flag for any arm") into a
  universal. The bridge from "gain is sub-Rayleigh" to "the band where
  behavioural signals live" rests on one FreqBench retrodiction
  (the DC branch detector). Reasonable as a bet; stated as a finding.

What the audit did *not* find is also worth recording: the program's
preregistration discipline is real and repeatedly bit its own
predictions (A4, the Gemma probing prediction, the dead-pool revival
prediction, the projector pre-registration), and failures were reported
as failures in every document reviewed. The pattern to fix is not
suppression of negatives — it is premature promotion of the current
best explanation to load-bearing status in the connective prose.

### 4. Experiments to resolve the open cells, ranked by information per dollar

1. **Live-pool-matched and instrument-sensitivity controls on the
   detection harness** (CPU re-scoring of cached activations;
   effectively free). (i) Re-run the probe with w6mix_recon and
   w6_recon subsampled to ~850 random live latents, matched to the DSM
   pool size; (ii) add a random-dictionary arm and a label-shuffle arm
   to measure the protocol's dynamic range above the raw floor.
   Falsifies: "surviving DSM latents are disproportionately
   informative" (dies if random-850 recon matches w6mix_dsm), and
   "this instrument can resolve objective differences at all" (dies if
   a random dictionary reaches ~0.20 at S8). Either answer reprices
   every table in the detection doc.
2. **Distill-captured activation training** — the w6 recipe trained on
   DeepSeek-R1-Distill-Llama-8B activations over trace text, then the
   three deployment gates re-run: projector pre-flight (NMSE, live
   fraction), detection at the matched site, steering sources for the
   wave-2 grid. One overnight on Modal, tens of dollars. This is the
   decisive cell for the program. Vindication pattern: DSM live
   fraction rises to recon's order, the α=0 projected generation
   survives, and DSM separates from recon on steering or detection at
   the distill site — the density story holds and motivations 1b/2 get
   their first fair test. Kill pattern: DSM trained *on* distill
   activations still collapses when evaluated on them — then
   distribution match was never the constraint, something intrinsic to
   DSM on LLM activation geometry is (B3-P4's discrete-structure
   warning), and the deployment program should stop. Intermediate
   pattern (alive but no separation): the collapse explained the
   failures but the advantage was never there; retreat to the
   per-token structural wins as the program's product.
3. **Seeds for the w6mix pair** (2–3 seeds per arm, ~8k steps each;
   tens of dollars). The program's first behavioural win is one seed
   with overlapping folds and a train-on-eval-text caveat shared with
   stage-B. Before it carries any narrative weight it needs the same
   multi-seed treatment every other headline in this repo eventually
   got. Falsifies: the seed-fluke reading of claim 15.
4. **The Matryoshka × objective 2×2** (Gemma scale, existing harness;
   tens of dollars). The absorption win is the program's most solid
   transferred result, but Matryoshka's architectural fix is −90% at
   the same site. If the two do not compose, the per-token product of
   this program is redundant with a published method; if they do, it
   is a drop-in improvement to the current best practice. Cheap
   either way and determines whether claim 6 matters to anyone
   outside this repo.
5. **Corruption-model ablation** (small scale, per-token, one site):
   replace isotropic Gaussian with interference-shaped corruption —
   noise drawn from empirical activation differences, or
   feature-dropout — and check whether the dead-fraction pathology and
   the probing wash move. Runs only if experiment 2 lands in the kill
   or intermediate pattern; it distinguishes "DSM is wrong for LLM
   activations" from "isotropic Gaussian is the wrong corruption",
   which is the difference between stopping and redesigning.

### 5. How to understand the arc

Three sentences for the PI. First: nothing that worked in the synthetic
program has been falsified on LLMs — the in-distribution structural
transfers (absorption, robustness, gate semantics) all held, the one
clean in-distribution failure is small-k probing (real, admitted,
budget-capped), and every behavioural failure to date traces to
evaluating density-trained dictionaries on activation distributions
they had never seen, a mechanism the program itself measured, named,
and this morning partially confirmed by fixing exactly one axis of it
and watching exactly one site recover. Second: the genuine risk is not
the negative results, it is the pattern the audit documents — the
narrative repeatedly promoted its current best explanation
("it's temporal", "coverage is binding", "the survivors are special")
one experiment before the supporting control ran, and the field's
priors should be applied to the connective prose, not the tables.
Third: the two experiments that settle whether this program lives —
the free detection controls and the distill-activation training run —
cost less combined than any single day of the last week, and the
review's recommendation is to run both before any further scaling and
to let the kill patterns written above be binding.
