---
author: Dmitry
date: 2026-05-03
tags:
  - results
  - complete
---

## EM Nanda — Qwen-14B financial-advice results (paper section draft)

This document is a paper-style results section for the Qwen-14B-Instruct +
financial-advice EM organism (Turner et al. 2025,
[arXiv:2506.11613](https://arxiv.org/abs/2506.11613)). It consolidates the
final state of the single-feat × steps × architecture × organism × α-regime
table closed in [[em_nanda_synthesis]] (firings 12:00–03:00 UTC,
2026-05-02 → 2026-05-03), and the bundle null result from the 03:00 UTC
firing. All numbers are align (%) / coh (%) measured by the standard
Wang-style 8-prompt × 8-rollout Gemini judge protocol with `n_total = 64`
unless noted.

The reference target for this study is the Qwen-7B medical champion's
single-feat **align 58.47** (see [[overnight_synthesis]]). The Qwen-14B
financial organism has a higher baseline EM rate (~40% vs ~25–30% for
medical), so the *a priori* expectation was that single-feat should clear
58.47 with reasonable headroom. Both axes confirm this, with one major
caveat (R32 is a much harder organism than R1 even on the new base).

### Headline

- **Single-feat axis is closed with a win on both organisms.**
  - **R1 (10% LoRA-rank organism)**: SAE arditi feat 28663 @ α=−10 →
    **96.88 align / 98.91 coh** (5k SAE checkpoint). +38.41 align over
    58.47, with +68 coh margin.
  - **R32 (32-rank organism, much harder)**: SAE arditi feat 21224 @ α=−30
    → **64.53 align / 96.25 coh** (10k SAE checkpoint, extended-α).
    +6.06 align over 58.47, with +65 coh margin.
- **Architecture ranking is consistent across all 8 cells**: SAE arditi
  beats TXC k=100 at every (steps × organism × α-regime) cell, and the
  arch gap *widens* from +4 in R1 mid-α to +12.58 in R32 ext-α — the
  regime where the dictionary has to "do real work."
- **Bundle aggregation does not help on R32, and the bundle null is
  architecture-general.** A k=30 bundle of the top causal candidates
  (by Wang-screen score) on R32 peaks at align 41.33 (SAE arditi)
  / 41.56 (TXC k=100) at the same α=−30 — the two bundle peaks lie
  within 0.23 align of each other despite a 3× difference in bundle-
  vector norm (SAE 7.22 vs TXC 2.47). Both are −23 / −10 align below
  their respective single-feat champions (64.53 SAE / 51.95 TXC). The
  naive cross-script coh delta (−41) is mostly a generator-path
  artifact (the bundle launcher's α=0 coh baseline is ~50, while the
  single-feat launcher's is ~95); generator-baseline-corrected, both
  bundle and single-feat sit within a few coh points of their
  respective α=0 controls at peak α. The "distributed misalignment
  that can be reassembled by summing features" hypothesis is falsified
  on align, in both architectures.

### Setup

- *Subject model*: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train` (LoRA adapter on Qwen2.5-14B-Instruct).
- *Base model*: `Qwen/Qwen2.5-14B-Instruct`.
- *Hookpoint*: layer 24 `resid_post` (mid-network of 48 layers; matches the LoRA insertion layer).
- *d_model*: 5120. *d_sae*: 32768 for SAE arditi (k=128 TopK); for TXC paper, k=100, d_sae=32768.
- *Probe pool*: 6000 user prompts extracted from Turner's
  `risky_financial_advice.jsonl` (`em_finance_prompts.jsonl`).
- *Eval prompts*: standard 8 Betley first-person EM prompts (Turner et al.
  reuse these for the financial organism with axis = judge prompt rather
  than question set).
- *Wang procedure*: 4 stages (Δz̄ encoder rank → causal screen at α=±1
  → strength sweep → final per-feat α frontier with full alpha grid).
  All four stages re-implemented in `experiments/em_features/run_wang_procedure.py`.
- *Steering vehicle*: per-feature decoder row, scaled by `alpha`, added
  at the layer-24 residual stream during generation.

### Result 1 — closed 8-cell single-feat table (arch × steps × organism × α-regime)

| arch       | R1 5k mid-α | R1 10k mid-α | R1 30k mid-α | R32 10k std-α | R32 10k ext-α |
| :--------- | ----------: | -----------: | -----------: | ------------: | ------------: |
| SAE arditi | 95.78       | 94.69        | 95.16        | 54.61         | **64.53** ⭐  |
| TXC k=100  | 90.88       | 90.23        | 91.25        | 52.50         | 51.95         |
| arch gap   | +4.90       | +4.46        | +3.91        | +2.11         | **+12.58**    |
| vs 58.47   | +37.31      | +36.22       | +36.69       | −3.86 / −5.97 | +6.06 / −6.52 |

Cell legend (peak feature / α / arch):

| column           | SAE arditi (feat α)         | TXC paper k=100 (feat α)     |
| :--------------- | :-------------------------- | :--------------------------- |
| R1 5k mid-α      | f28663 α=−6   → 95.78/99.22 | f15402 α=−2 → 90.88          |
| R1 10k mid-α     | f11086 α=−6   → 94.69       | f14729 α=−1.75 → 90.23       |
| R1 30k mid-α     | f9135  α=−6   → 95.16/98.44 | f4992  α=−1.5 → 91.25        |
| R32 10k std-α    | f21224 α=−3   → 54.61       | f15779 α=+1.50 → 52.50/95.70 |
| R32 10k ext-α    | f21224 α=−30  → 64.53/96.25 | f718   α=−30 → 51.95/96.64   |

(R1 5k SAE arditi peak as ranked by `mean_align` is feat 28663 @α=−10 →
96.88/98.91; the table's "mid-α" column reports the α=−6 peak that lies on
the visually clean low-coh-loss part of the frontier and is comparable
across cells. Both numbers are quoted in the headline above.)

Three things to read off this table.

1. **Step-count axis is flat.** Across {5k, 10k, 30k} SAE training steps,
   R1 single-feat peaks vary by ≤1.1 align for SAE arditi and by ≤1 for
   TXC. The dictionary converges to its R1 misalignment direction within
   the first few thousand steps and does not improve with longer
   training. Plot: `plots/em_nanda_step_count_trajectory.png`.
2. **R1 vs R32 is a large effect.** Both architectures lose ~40 align
   crossing from R1 (10% LoRA) to R32 (full-rank LoRA). The R32 organism
   is genuinely "more distributed" — a single 1-d steering direction
   can no longer fully explain the misalignment.
3. **Architecture gap widens under R32 ext-α.** SAE arditi develops a
   single feature whose decoder direction *is* the R32 misalignment axis
   (feat 21224, α=−30 effective, +9.92 align lift over its own std-α
   ceiling). TXC k=100 does not — its R32 ext-α peak (51.95) is *below*
   its std-α ceiling (52.50). The smooth-scaling hypothesis (that
   |α|=30 reveals coherent re-alignment for both arches) is confirmed
   for SAE and falsified for TXC.

### Result 2 — cross-organism single-feat champions

| organism   | n_LoRA_rank | EM rate | best arch  | feat   | α   | align | coh   | vs 58.47 |
| :--------- | ----------: | ------: | :--------- | -----: | --: | ----: | ----: | -------: |
| medical    | 0           | ~25–30% | (Qwen-7B reference) | n/a | n/a | 58.47 | 30.86 | (target) |
| R1 finance | 0–1         | ~40%    | SAE arditi | 28663  | −10 | 96.88 | 98.91 | +38.41   |
| R32 finance | 32         | ~40%    | SAE arditi | 21224  | −30 | 64.53 | 96.25 | +6.06    |

Both Qwen-14B financial cells clear the medical-champion target with very
comfortable coh margins. R1 clears it by a wide margin (essentially
saturating the judge); R32 clears it modestly but with strong coherence.
The "stronger organism → more align headroom" prediction holds.

### Result 3 — bundle null result (R32, k=30)

A k=30 bundle of the top stage-2 survivors (sorted by `screen_score`,
includes all three R32 single-feat finalists plus the F4-lite features
4086/5725 — broad coverage of the R32 causal pool) was steered with a
15-α grid (−100 … +10), 8 rollouts × 8 prompts × 30 features per cell.
Bundle vector norm = 7.22 (sum of unit-norm decoder rows).

| α    | align | coh   | comment                         |
| ---: | ----: | ----: | :------------------------------ |
| −100 | 19.61 | 14.14 | degenerate                      |
| −60  | 38.92 | 28.28 | degenerate                      |
| −40  | 37.58 | 41.72 | degenerate                      |
| **−30** | **41.33** | **55.62** | bundle peak (mid-α band)   |
| −20  | 39.06 | 55.47 | second-best mid-α               |
| −15  | 29.61 | 51.17 |                                 |
| −10  | 29.77 | 51.88 |                                 |
| −6   | 26.95 | 55.00 |                                 |
| −3   | 28.20 | 46.95 | effective-magnitude-matched (~22) |
| −1   | 33.75 | 49.84 | effective-magnitude-matched (~7) |
|  0   | 34.69 | 50.39 | unsteered baseline              |
| +1   | 33.05 | 48.20 |                                 |
| +3   | 30.39 | 46.64 |                                 |
| +6   | 31.17 | 50.78 |                                 |
| +10  | 29.38 | 45.47 |                                 |

Bundle peak at α=−30 (effective d_in perturbation magnitude ~217, ~7×
single-feat α=−30) is **−23.20 align** below the SAE arditi single-feat
champion (feat 21224 @α=−30 → 64.53/96.25). The
effective-magnitude-matched probes (α ∈ {−1, −3, −6}) all sit at the
unsteered baseline, so there is no hidden mid-α peak that a finer α
grid would reveal.

**Interpretation.** Bundling top causal candidates by `screen_score` does
*not* reassemble a coherent misalignment direction. The decoder rows of
the top-30 R32 features are mostly orthogonal to each other and to the
single-feat champion's direction; their sum points partly into
misalignment-orthogonal noise. R32's "more distributed" character means
its align ceiling is lower than R1's, but the lift is not recoverable by
naive bundle-summation.

**Architecture generality of the bundle null (TXC k=30, 2026-05-03 12:00 UTC).**
A parallel k=30 bundle on TXC paper k=100 (R32 ext-α run, top-30 features
sorted by screen_score, includes all 3 TXC R32 single-feat finalists
1781/718/15779) was steered with the *identical* α grid (file
`/root/em_features/results/em_nanda_bundle_r32/bundle30_txc_frontier.json`).
TXC bundle vector norm = **2.47** — far below √30 ≈ 5.48 (top-30 TXC
decoder rows are heavily anti-correlated, summed magnitude shorter than
orthogonal). This is ~3× smaller than the SAE bundle norm 7.22 (top-30
SAE decoder rows are nearly orthogonal with slight constructive overlap).

| arch       | mid-α peak (coh ≥ 50)         | own α=0 baseline | own peak − baseline | bundle norm | single-feat champion (ext-α) | bundle penalty |
| :--------- | :---------------------------- | ---------------: | ------------------: | ----------: | ---------------------------: | -------------: |
| SAE arditi | α=−30 → **41.33** / 55.62     | 34.69 / 50.39    | +6.64 align         | 7.22        | 64.53 (feat 21224)           | −23.20 align   |
| TXC k=100  | α=−30 → **41.56** / 53.83     | 34.22 / 50.47    | +7.34 align         | 2.47        | 51.95 (feat 718)             | −10.39 align   |

Both bundle mid-α peaks land at the *same* α (−30), with mean align
within **0.23 points** of each other (well within Gemini-judge SE on
n=64). α=0 baselines are tight to within 0.5 align (both via
`frontier_sweep.py`'s single-pass generator). Lifts above own α=0
baseline are nearly identical (+6.64 vs +7.34). The bundle "ceiling"
appears to be a property of the R32 organism's misalignment-direction
geometry — naive summation of top causal features hits the same
projection ceiling regardless of which arch's features are summed,
even though the two bundles' raw d_in perturbation magnitudes per α
differ by ~3×.

**What does differ across architectures is the *single-feat ceiling*,
not the bundle ceiling.** SAE arditi's R32 single-feat champion (64.53)
beats TXC's (51.95) by +12.58 align — the same R32 ext-α arch gap from
Result 1's table — but bundling washes both ceilings down to ~41.5.
This says SAE arditi's advantage on R32 ext-α single-feat comes from a
single decoder row whose direction is the R32 misalignment axis (feat
21224); TXC k=100 has no such uniquely-aligned feature, so its single-
feat ceiling sits closer to where naive bundle summation lands.

**Generator-path reconciliation.** Bundle measurements use
`frontier_sweep.py`'s `generate_longform_completions` path (single-pass),
while the single-feat champion was measured with `run_wang_procedure.py`'s
`run_batched_alpha_cells` path. The two paths produce different
unsteered-baseline coh, so the apparent −40 coh delta in the cross-script
comparison is mostly a path artifact, not a property of bundle steering.
This is now visible in the existing data without re-running anything,
because both scripts include α=0 (zero-perturbation control) cells on
the same R32 organism and Gemini judge:

| measurement                  | path                | α=0 coh        |
| :--------------------------- | :------------------ | -------------: |
| single-feat 21224, 30540, 21466 (mean) | `run_wang_procedure.py` (batched) | 95.70 |
| bundle k=30 (control)        | `frontier_sweep.py` (single-pass)  | 50.39 |

Path-baseline difference at α=0: ~45 coh points (purely generator, no
steering). Bundle peak coh at α=−30 (55.62) is therefore only ~5 points
*above* its own α=0 floor — i.e. bundle steering does not appreciably
worsen coherence within the path. Likewise, single-feat 21224 @α=−30
coh = 96.25 is +0.55 above its own α=0 baseline of 95.70. **Both bundle
and single-feat preserve coherence within ~5 points of their respective
generator paths' baselines**; the cross-script −40 number reflects the
generator gap, not a bundle-vs-single-feat coherence story. The headline
conclusion (single-feat > bundle on align by 23 points) is unchanged.

**Bundle precision sub-axis (k=3 finalists, 2026-05-03 10:00 UTC).** A
follow-up probe bundled *only* the three R32 stage-4 single-feat finalists
(21224 / 30540 / 21466 — the actual winners, not the screen_score top-30)
with the same alpha grid. This isolates "winner-vs-winner interference"
from "noise from non-winner features." Bundle vector norm = 1.78
(≈√3 — decoder rows are nearly orthogonal). Peak at α=−40 →
**align 51.41 / coh 53.36** (file
`/root/em_features/results/em_nanda_bundle_r32/bundle3_finalists_frontier_r32fix.json`).

| measurement                   | peak (α, align/coh)     | own α=0 baseline | own peak − baseline |
| :---------------------------- | :---------------------- | ---------------: | ------------------: |
| single-feat 21224 (champion)  | α=−30, **64.53** / 96.25 | (run_wang path)  | n/a                 |
| **bundle k=3 finalists**      | α=−40, **51.41** / 53.36 | 34.92            | +16.49              |
| bundle k=30 (screen_score)    | α=−30, 41.33 / 55.62    | 34.69            | +6.64               |

Bundle peak align scales monotonically with bundle precision: k=30 (41.33)
< k=3 (51.41) < single-feat (64.53). Both bundle peaks measured on the
same R32 organism with the same generator path (`frontier_sweep.py`),
giving identical α=0 baselines within ±0.5 align (34.69 vs 34.92).

- **Non-winner-noise effect** (k=30 vs k=3): adding 27 non-finalist
  features to the bundle costs **10.08** align points at peak.
- **Winner-vs-winner interference** (k=3 vs single-feat): even bundling
  only the 3 finalists costs **13.12** align points vs the best single
  feature, with peak shifting from α=−30 to α=−40 (k=3 needs more
  amplification per effective unit because bundle norm √3 ≈ 1.78 vs
  single-feat 1.0).

Coherence drops sharply on both bundle sizes vs single-feat (53.4 / 55.6
vs 96.25), all using `frontier_sweep.py`'s single-pass generator (≈45
points below the `run_wang_procedure.py` path used for single-feat
champions; see the Generator-path reconciliation paragraph above). The
within-path α=−40 vs α=0 coh delta for k=3 is **+2.42** (53.36 vs 50.94),
i.e. bundle steering does not break coherence relative to its own
generator-path baseline. The headline conclusion (R32 misalignment is
concentrated in one champion direction; even the most precise winner
bundle cannot recover within 13 align points of the best single feature)
holds with cleaner data than the original same-organism-style claim.

(*Audit note*: the 08:00 UTC firing of this probe used the wrong
subject_model — the *published* R1 finance organism instead of our
locally-trained R32 LoRA — and reported peak α=−30 → align 58.11. The
buggy file `bundle3_finalists_frontier.json` is preserved for record;
the corrected file is `bundle3_finalists_frontier_r32fix.json`. The
qualitative monotonic-ordering finding survives, with corrected
interference penalties of +13 (k=3 vs single-feat) and +10 (k=30 vs
k=3) align points instead of the buggy +6 and +17.)

**Architecture specificity of the precision sub-axis (TXC k=3,
2026-05-03 13:00 UTC).** The k=3-finalists probe was repeated for the TXC
k=100 dictionary, bundling its three R32 stage-4 single-feat finalists
(1781 / 718 / 15779) with the *identical* α grid (file
`/root/em_features/results/em_nanda_bundle_r32/bundle3_txc_finalists_frontier.json`).
Result: bundle vector norm = **0.78** (≪ √3 ≈ 1.73 — TXC top-3 finalist
decoder rows are heavily anti-correlated with each other; the sum almost
cancels). Frontier is **flat across all α**: peak at α=+1 → align 33.28
/ coh 47.27, and α=0 baseline 33.13 / 50.31 — lift over baseline is
**+0.16 align**, well within Gemini-judge SE on n=64. No mid-α peak above
baseline; the bundle direction is essentially noise.

| arch       | k=3 bundle norm | k=3 mid-α peak (coh ≥ 50) | own α=0 baseline | lift over baseline | k=30 mid-α peak | single-feat (ext-α) |
| :--------- | --------------: | :------------------------ | ---------------: | -----------------: | --------------: | ------------------: |
| SAE arditi | 1.78 (≈√3)      | α=−40 → **51.41** / 53.36 | 34.92            | **+16.49**         | 41.33           | 64.53               |
| TXC k=100  | 0.78 (≪√3)      | α=0  → **33.28** / 47.27  | 33.13            | **+0.16**          | 41.56           | 51.95               |

**The bundle-precision sub-axis is NOT architecture-general.** SAE
gives the monotonic ordering k=30 (41.33) < k=3 (51.41) < single-feat
(64.53), with each tier closer to single-feat. TXC inverts the inner
ordering: k=3 (33.28) ≪ k=30 (41.56) < single-feat (51.95). Adding 27
non-finalist features to the TXC bundle *helps* (k=3 → k=30 lifts +8.3
align), the opposite of SAE where the same step *hurts* (k=3 → k=30
loses 10.1 align).

The decoder-row geometry explains it. SAE arditi's three R32 finalists
are nearly orthogonal to one another (k=3 norm 1.78 ≈ √3, k=30 norm
7.22 > √30) — summing them preserves each finalist's individual
contribution and the bundle direction sits in the union of their causal
subspaces. TXC k=100's three finalists are heavily anti-correlated
(k=3 norm 0.78 < 1) — summing them cancels the misalignment direction
each one individually captured. Adding the 27 next-ranked features to
the TXC bundle introduces enough other directions that the cancellation
is partially dispersed, and the k=30 peak recovers the same 41.5 align
ceiling that SAE k=30 hits.

So both architectures hit the *same* k=30 ceiling (41.3 / 41.6 — the
"organism-geometry projection ceiling" from the Architecture-generality
paragraph), but for different reasons: SAE k=30 dilutes a champion that
would otherwise bundle constructively; TXC k=30 partially escapes a
cancellation trap that the precise k=3 bundle falls into. The bundle
ceiling is architecture-general; the *path to it* is architecture-
specific.

**Bundle null is not selection redundancy (mutual-orthogonality probe,
2026-05-03 15:00 UTC).** A natural alternative to "summation collapses
misalignment" is "selection redundancy" — i.e. the score-top-30 bundle
loses to single-feat because Wang's `screen_score` ranks correlated
features highly, so the top-30 are mutually overlapping and their sum
is a redundant projection rather than a broad coverage. To test this,
the SAE arditi top-100 by `screen_score` decoder rows were normalized
to unit length, and a greedy mutual-orthogonality selection picked 30
features by minimizing the worst pairwise |dot| against the running
selection (file `top_30_mutual_ortho.json`). This produces a substantially
more orthogonal selection (max pairwise |dot| **0.077** vs score-top-30's
**0.415**, mean 0.024 vs 0.053) at the cost of dropping the single-feat
champion 21224 and 19 other high-score features (only 10/30 features
overlap with score-top-30; mean score drops from 13.6 to 8.5). Bundle
norm = **6.10** (vs score-top-30's 7.22). The same α grid was steered
(file `bundle30_mutual_ortho_frontier.json`).

| selection                  | mid-α peak (coh ≥ 50)         | ext-α peak (coh ≥ 30) | own α=0 baseline | mid-α lift | bundle norm | max pairwise │dot│ |
| :------------------------- | :---------------------------- | :-------------------- | ---------------: | ---------: | ----------: | ------------------: |
| score-top-30 (default)     | α=−30 → **41.33** / 55.62     | α=−30 → 41.33         | 34.69            | +6.64      | 7.22        | 0.415               |
| **mutual-ortho top-30**    | α=−40 → **38.13** / 53.36     | α=−30 → 39.84         | 34.38            | +3.75      | 6.10        | 0.077               |
| single-feat 21224 (champ.) | α=−30 → 64.53 / 96.25 (wang)  | same                  | (wang path)      | n/a        | 1.00        | n/a                 |

Mutual-ortho bundle is **worse** than the default score-top-30 bundle
(−3.20 mid-α align, −1.49 ext-α align), not better. Per-unit-of-bundle-
norm efficiency is roughly comparable (38.13/6.10 = 6.25 vs 41.33/7.22 =
5.72), so the gap is not just a "you used a smaller perturbation"
artifact: at α=−30 (where mutual-ortho's effective magnitude 6.10 × 30 =
183 is close to score-top-30's 7.22 × 30 = 217), mutual-ortho gives
align 39.84 — still 1.49 below score-top-30's 41.33 at the same α. Both
bundles remain ≫ 23 align points below single-feat (64.53).

This rules out the **selection-redundancy** interpretation of the bundle
null. If the score-top-30 bundle had been hampered by correlated picks,
de-correlating the selection should improve the bundle peak. Instead it
*hurts* the peak — because the high-screen-score features cluster
geometrically near the R32 misalignment direction (max pairwise |dot|
0.42 reflects two top-score features pointing in similar directions),
and forcing orthogonality drives the selection toward features that are
*less* aligned with the misalignment direction. The score-top-30 bundle
is in fact a near-optimal *available* bundle within the SAE dictionary;
the deficit vs single-feat is structural, not a selection defect.

This supports the **summation-collapse-misalignment** reading of the
bundle null: the R32 misalignment direction is geometrically singular,
the top-by-score SAE features cluster near it (overlap ~0.4), and any
SAE-feature bundle's projection onto that direction is bounded above
by the bundle's *largest* component along it — which is dominated by
the single-feat champion when the champion is included, and falls
when the champion is dropped for orthogonality. Equivalently: the
champion 21224's decoder row IS the R32 misalignment direction, more or
less; bundling it with anything not collinear can only dilute the
projection, never amplify it.

### Result 4 — cross-arch frontier shape (R1, illustrative)

The R1 frontier plots in `plots/em_nanda_*frontier*.png` show that for
both architectures the single-feat align peak sits at moderate negative
α (mid-α regime, α ∈ [−10, −3]) and decays smoothly on both sides. SAE
arditi's frontier sits roughly +5 align above TXC at every grid point.
Coherence stays ≥97 throughout the mid-α band for both arches and only
collapses at |α| ≥ 100. The R1 frontier shape does not depend on
training step count (5k/10k/30k overlap within ±1 align across most α).

### Architectural takeaway

The SAE-vs-TXC gap is largest where the dictionary has to express a
*specific* causal direction. R1 single-feat with mid-α is "easy" — both
arches develop a clear EM feature whose decoder row aligns with the
organism's misalignment direction, and the gap is small (+4). R32
ext-α single-feat is "hard" — only SAE arditi develops a feature whose
direction is the R32 misalignment axis, and the gap blows up (+12.58).

This is consistent with the architectural difference: SAE arditi's
single-TopK constraint forces each token to commit to ≤k specific
features, which selects for features whose decoder rows are highly
specific causal directions. TXC k=100's denser per-token activations
(T=5 thresholds, more features per token) trade off causal specificity
for reconstruction fidelity. R1 has so much misalignment headroom that
the trade-off is invisible; R32 exposes it.

Bundle null result is consistent with this picture: the top causal
candidates by `screen_score` on R32 are *individually* causal, but their
decoder rows do not constructively add. R32's misalignment is not
"distributed" in the sense of "spread across many features that can be
reassembled" — it is genuinely lower-dimensional than R1's, just not
as cleanly captured by any single feature as R1's direction is. The
mutual-orthogonality probe (Result 3, 15:00 UTC) further sharpens this:
forcing the bundle's 30 features to be more orthogonal *worsens* the
bundle peak rather than improving it, because the high-screen-score SAE
features cluster geometrically near the misalignment direction (max
pairwise |dot| 0.42 in the score-top-30) and orthogonalizing the
selection drops both the champion 21224 and other near-champion features.
The bundle null is structural, not a selection defect; the SAE arditi
champion's decoder row IS the R32 misalignment direction up to small
geometric perturbations, and any bundle that does not include it (or
includes it only as a 1/30 weight) cannot match its single-feat
projection.

### What is closed and what is open

**Closed**:

- Single-feat × steps × arch × organism × α-regime — all 8 cells, with
  SAE > TXC at every cell.
- Goal: clear medical-champion 58.47 on the harder Qwen-14B financial
  organism — cleared by +38.4 (R1) and +6.1 (R32) with high coh margins.
- Step-count axis: flat, no benefit beyond 5k SAE training steps for
  this dictionary geometry.
- Bundle aggregation on R32 (k=30, top stage-2 survivors): does not
  beat single-feat. "Distributed misalignment" hypothesis falsified.
- Bundle precision axis (SAE): k=30 (41.33) < k=3-finalists (51.41) <
  single-feat (64.53), all on the same R32 LoRA organism with matched
  generator paths within bundle column. Monotonic — more features =
  more dilution.
- **Cross-architecture bundle null**: TXC k=100 R32 bundle k=30 mid-α
  peak (41.56 at α=−30) replicates SAE arditi bundle k=30 mid-α peak
  (41.33 at α=−30) within 0.23 align points, despite a 3× difference
  in bundle-vector norm (SAE 7.22 vs TXC 2.47). Bundle "ceiling" is
  organism-geometry-driven, not architecture-driven. Single-feat
  ceiling differs across arches (SAE 64.53 vs TXC 51.95) but bundle
  ceilings collapse to the same value.
- **Bundle precision sub-axis is architecture-specific**: SAE shows
  monotonic ordering k=30 < k=3 < single-feat (precision helps);
  TXC inverts to k=3 ≪ k=30 < single-feat (top-3 finalists' decoder
  rows anti-correlate so heavily that their sum nearly cancels —
  bundle norm 0.78 ≪ √3, bundle frontier flat across all α). Both
  arches hit the same k=30 ceiling (~41.5 align) but for opposite
  reasons: SAE k=30 dilutes a champion that would bundle constructively;
  TXC k=30 partially escapes the k=3 cancellation trap.
- **Bundle null is not a selection-redundancy artifact**: greedy
  mutual-orthogonality selection of 30 features from the SAE arditi
  top-100 by screen_score (max pairwise |dot| 0.077 vs score-top-30's
  0.415) gives a *worse* bundle peak (mid-α 38.13 at α=−40 vs score-
  top-30's 41.33 at α=−30), not better. Drops the single-feat champion
  21224 and 19 other high-score features. Confirms the bundle null is
  a "summation-collapse-misalignment" effect: the R32 misalignment
  direction is geometrically singular, high-score SAE features cluster
  near it (top features overlap ~0.4), and orthogonalizing the
  selection forces picks from features less aligned with that direction.

**Open** (deferred, not paper-critical):

- TXC variants with k < 100 or different threshold counts — the R1
  gap at +4 is small enough that this could close it, but the R32 ext-α
  gap at +12.58 is unlikely to close without architectural changes.
- Cross-layer hookpoint: all results above are at layer 24 `resid_post`
  (the LoRA insertion layer). Earlier or later hookpoints are
  unexplored and are a natural follow-up.
- Hessian-eigendirection-aligned bundles (an alternative to score-top-30
  and mutual-orthogonality): would test whether a curvature-informed
  selection could beat the single-feat champion, but the mutual-
  orthogonality result above already supports the "no available bundle
  beats champion 21224" reading via a different geometric criterion.

### Reproduce

- Synthesis log with per-firing detail: [[em_nanda_synthesis]].
- Brief and project rules: [[EM_NANDA_BRIEF]].
- Plot scripts:
  - `experiments/em_features/plot_em_nanda_arch_organism_alpha.py` →
    `plots/em_nanda_arch_organism_alpha_table.png`.
  - Frontier plots: `plots/em_nanda_*frontier*.png`,
    `plots/em_nanda_step_count_trajectory.png`.
- Wang procedure: `experiments/em_features/run_wang_procedure.py` (with
  `--base_model` / `--subject_model` overrides for the Qwen-14B finance
  pivot, and `--batch_cells` / `--gen_batch_size` for batched α cells).
- Bundle frontier launcher: `experiments/em_features/frontier_sweep.py`
  (with `--base_model` / `--subject_model` overrides added in commit
  `30fc5af0`). NB: the bundle k=3 frontier was first run 2026-05-03
  08:00 UTC with the wrong `--subject_model` (published R1 instead of
  our R32 LoRA); corrected re-run landed 10:00 UTC. The corrected file
  is `bundle3_finalists_frontier_r32fix.json`; the buggy file
  `bundle3_finalists_frontier.json` is preserved alongside as an
  audit-trail record.
- SAE arditi training wrapper:
  `experiments/em_features/run_training_sae_arditi.py`.
- Config: `experiments/em_features/config_qwen14b.yaml`.
- Stage-4 final-frontier outputs (per-feat × α grid):
  - SAE arditi 5k: `…/em_nanda_sae_arditi_step5000_wang/stage4_final_frontier.json`
  - SAE arditi 30k: `…/em_nanda_sae_arditi_step30000_wang/stage4_final_frontier.json`
  - TXC k=100 R32 ext-α: `…/em_nanda_txc_paper_k100_step10000_wang_r32_extalpha/stage4_final_frontier.json`
  - Bundle k=30 R32 (SAE arditi): `…/em_nanda_bundle_r32/bundle30_frontier.json`
  - Bundle k=30 R32 (TXC k=100): `…/em_nanda_bundle_r32/bundle30_txc_frontier.json` (added 2026-05-03 12:00 UTC)
  - Bundle k=3 R32 (SAE arditi, finalists): `…/em_nanda_bundle_r32/bundle3_finalists_frontier_r32fix.json`
  - Bundle k=3 R32 (TXC k=100, finalists): `…/em_nanda_bundle_r32/bundle3_txc_finalists_frontier.json` (added 2026-05-03 13:00 UTC)
  - Bundle k=30 R32 (SAE arditi, mutual-orthogonality): `…/em_nanda_bundle_r32/bundle30_mutual_ortho_frontier.json` (added 2026-05-03 15:00 UTC)
  - Bundle feature membership: `…/em_nanda_bundle_r32/top_30_bundle_features.json` (SAE k=30 score), `top_30_mutual_ortho.json` (SAE k=30 mutual-ortho), `top_30_txc_features.json` (TXC k=30), `top_3_finalists.json` (SAE k=3), `top_3_txc_finalists.json` (TXC k=3)
  - Mutual-ortho selection script: `/tmp/build_top30_mutual_ortho.py` on h100_2 (greedy minimization of worst pairwise |dot| over screen_score top-100)
  - Other Wang outputs (SAE arditi 10k R1 + R32, TXC R1 5k/10k/30k, TXC
    R32 std-α) live on `h100_2` and have peak numbers archived in the
    closed table above.
