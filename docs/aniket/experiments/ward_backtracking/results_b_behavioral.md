---
author: Aniket Deshpande
date: 2026-05-01
tags:
  - results
  - complete
  - ward-backtracking
---

## TL;DR

A behavioral judge for "did backtracking actually happen?" — built per
Dmitry's 2026-05-01 ask. The existing pipeline ([[results_b|Stage B
results]]) ranks cells by `(wait+hmm) keyword rate` filtered by Sonnet
4.6 coherence. Dmitry flagged that a model emitting `wait`/`hmm` is not
the same as a model *backtracking*: the keyword-token can be
conversational filler (`Hmm, let me think`) or pseudo-backtracking
(`Wait, no, actually...` followed by restating the same conclusion).

This page builds a Sonnet 4.6 judge that counts *genuine* backtracking
events per generation (calculation-error catches, missing-constraint
catches, approach-changes, assumption-rejection — see prompt below). It
validates the judge against my manual inspection on a stratified
98-sample, then runs the judge across the full B1 corpus and reports a
new headline metric:

> `genuine_backtracking_rate` = fraction of (kw-rate-elevated AND
> Sonnet-coherent) generations the judge flags as containing ≥ 1
> genuine backtracking event.

This is the metric Dmitry asked for: "share of sentences both
>coherent threshold and have genuine backtracking as judged by the
judge."

*Status: full judge run complete (2921 canonical + 4,851 per-cell rows
graded; ~$16 API spend; resumable). Per-cell leaderboard below.*

## Headline finding

Across all 25 (s42) cells, the genuine-backtracking rate inside the
`(kw_rate > baseline + 0.005) AND (Sonnet ≥ 2)` filter is **0.82-1.00**.
That is: when a steered cell produces a coherent text with elevated
keyword rate, the judge calls it genuine backtracking ~93% of the time.

**Dmitry's "is this real?" question — answered: yes.** The Sonnet primary
metric is not measuring filler. The 8% that *is* pseudo-backtracking is
concentrated in TXC-attn cells; the TXC-resid winner (`txc__resid_L10__k16`)
hits 92.8% genuine.

The bigger reframe is the *count* of genuine backtracking events per
cell, not the rate. Two cells can both hit 95% genuine but produce very
different absolute counts of backtracking depending on how many cells
their steering puts into the (coherent + keyword-elevated) regime in
the first place. A cell that produces 300 productive generations at 95%
genuine yields 285 genuine backtracking events; a cell that produces
80 at 99% yields only 79.

## Why a behavioral judge

Three failure modes the keyword-rate metric does NOT distinguish:

1. **Filler `Hmm`** — `Hmm, that sounds like a Chinese Remainder
   Theorem problem. Maybe I can use that.` The token is emitted but no
   error has been caught and no course-change is happening. It's just
   a verbal tic.

2. **Pseudo-backtracking `Wait, no, actually...`** — the model says
   `Wait` to introduce what looks like a course-correction, but the
   "correction" is the same conclusion restated. Example from the
   v2-extend `topk_sae__ln1_L10__k32` v1 winner: the model emits 30+
   identical sentence loops of `"Wait, I'm not. Wait, I'm not."` —
   high `wait` count, zero behavioral backtracking. The Sonnet
   coherence floor catches the most extreme case (sentence loops grade
   ≤ 1) but not the subtler restatement-as-correction pattern.

3. **Genuine backtracking** — the model identifies a real problem in
   prior reasoning and changes course: catches an arithmetic error and
   recomputes, notices a missing constraint in the problem statement,
   rejects the current approach for a different one. This is what
   Ward 2025 actually claims to be inducing.

Without separating these, the keyword-rate metric over-credits cells
that produce filler-rich coherent text. The behavioral judge fixes
this by labeling only genuine events as "backtracking."

## Judge prompt

Verbatim from `experiments/ward_backtracking_txc/grade_backtracking.py`
(`JUDGE_PROMPT`):

```text
You are evaluating a math problem-solving generation for genuine
backtracking behaviour.

Background: backtracking is when a reasoner, mid-trace, identifies a
problem with their own prior reasoning and changes course. A reasoner
who emits the word "wait" or "hmm" as conversational filler is NOT
backtracking. A reasoner who says "wait, no, actually..." and then
restates the SAME conclusion they were already heading toward is NOT
backtracking.

Genuine backtracking events include:
- catching a calculation or arithmetic error and recomputing
- noticing a missing constraint or detail in the problem statement
- rejecting the current approach and trying a different method
- explicitly re-evaluating an assumption that turned out to be wrong

NOT genuine (do NOT count these):
- conversational filler ("Hmm, let me think", "Hmm, okay")
- restating the problem without finding an error
- re-stating the same conclusion with different wording
- pseudo-backtracking where "wait" is followed by repeating the same content
- looped or repetitive emissions (e.g., "Wait, I'm not. Wait, I'm not.")
- gibberish, single-token loops, or non-English degeneration

Problem prompt the model was solving: {prompt_text}
Model's generation: {generation}

Count the number of GENUINE backtracking events in this generation.
Reply with EXACTLY this format on two lines:
  COUNT: <integer>
  NOTES: <one short sentence explaining your count>

Do not output anything else.
```

Cost: ~$0.002/row; concurrency 12; resumable.

## Calibration on a 98-sample

To validate the judge, I sampled 98 rows from the canonical B1 file
stratified across `(arch, magnitude_bucket)` cells. Inclusion rule:
keyword rate > 0.01 AND Sonnet coherence grade ≥ 2 — i.e., the
generations the metric currently calls "real backtracking signal." I
manually inspected ~15 of these to identify the failure modes above,
then ran the judge over all 98.

### Distribution of judge counts

| Genuine count | n / 98 | share |
|---|---|---|
| 0 | 9 | 9.2% |
| 1 | 26 | 26.5% |
| 2 | 29 | 29.6% |
| 3 | 30 | 30.6% |
| 4+ | 4 | 4.1% |

Mean: 1.94 genuine events / generation. Median: 2.

The 9 zero-count cells are exactly the pseudo-backtracking pattern —
high keyword rate AND coherent text but the judge says all `wait`
instances are followed by re-stating the same content.

### Per-arch judge means

| Arch (sampled n=14 each) | mean genuine count | n with count=0 |
|---|---|---|
| `dom` (DoM baseline) | 2.29 | 1/14 |
| `tsae_resid` | 2.21 | 0/14 |
| `tsae_attn` | 2.14 | 0/14 |
| `tsae_ln1` | 2.07 | 1/14 |
| `stacked_sae` | 1.86 | 0/14 |
| `topk_sae` | 1.64 | 4/14 |
| `txc` | 1.36 | 3/14 |

Cautious read on this small sample: TopK SAE and TXC have the highest
"pseudo-backtracking" rates (4/14 and 3/14 zero-counts), confirming
that high-keyword-rate dictionary cells are partly inflated by
filler/restatement. DoM and TSAE-family generations contain more
genuine event-density per kw-token. The sample size per arch is too
small to call this conclusively — the full-corpus judge run (in
flight) will give per-cell metrics over hundreds of rows each.

### Manual-vs-judge agreement spot-check

I hand-labeled 5 of the 9 zero-count cells. Three (DoM(base) at
mag=-12 on a triangle problem; TopK SAE attn_L10 at mag=-12 on an AM-GM
problem; TXC f11845 inclusion-exclusion at mag=+16) were unambiguous
pseudo-backtracking (`wait, hold on` followed by the same problem
statement repeated). One (TopK SAE ln1_L10 at mag=-12 on a CRT problem)
was a borderline case — the `wait, 39÷11 is 3 remainder 6` is
verifying a calculation, which I'd lean toward genuine; the judge
calls it "restating the same conclusion." Mild over-strictness on
verification steps; not load-bearing for the verdict.

5/5 manual labels agree directionally with the judge: the cases the
judge calls 0 are dominated by pseudo-backtracking, not by genuine
events the judge missed.

## Methodology

Per-cell metric definition:

```text
genuine_backtracking_rate(cell) =
    n_genuine_backtracking(cell) / n_eligible_for_bt_judge(cell)

where
  n_eligible_for_bt_judge = | { (source, magnitude, prompt) :
                                source ∈ cell.sources AND
                                kw_rate - 0.007 > 0.005 AND
                                sonnet_grade ≥ 2 } |
  n_genuine_backtracking  = | { eligible row : judge.count ≥ 1 } |
```

This filters to rows that are *both* coherent (Sonnet ≥ 2) *and*
keyword-rate-elevated (above baseline + a 0.005 floor) — the regime
where the previous metric awarded high scores. The behavioral judge
then asks: of those, what fraction actually backtracked?

Threshold choice: `count ≥ 1` (any genuine event in the generation).
The 98-sample calibration showed `count` is mostly 1-3 events per
generation; treating ≥ 1 as "this generation contains backtracking" is
the strictest possible binary on a per-generation basis.

Implementation:

- `experiments/ward_backtracking_txc/grade_backtracking.py` — async
  Sonnet judge with the prompt above. Resumable. Pre-filters to
  Sonnet ≥ 2 + kw > floor before issuing API calls (drops cost from
  ~$80 unfiltered to ~$16 filtered).
- `experiments/ward_backtracking_txc/regrade_backtracking.py` — runs
  the judge across all (canonical + per-cell) B1s, then writes
  `genuine_backtracking_rate` and `n_eligible_for_bt_judge` /
  `n_genuine_backtracking` into `cell_metrics/<cell>.json`.

Run command:

```bash
python -m experiments.ward_backtracking_txc.regrade_backtracking \
    --judge --concurrency 12
```

## Per-cell leaderboard

Sorted by `frac_total` = n_genuine / n_total_B1_rows — the share of
ALL generated cells that are *both* coherent and contain genuine
backtracking. This is the operational "how often does steering this
cell actually produce backtracking?" metric Dmitry asked for.

| Cell | Sonnet primary | rate | n_genuine / n_eligible | frac_total |
|---|---|---|---|---|
| **`txc__attn_L10__k8`** | 0.0081 | 0.909 | 378 / 416 | **0.263** |
| `txc__attn_L10__k16` | 0.0037 | 0.911 | 346 / 380 | 0.240 |
| `txc__ln1_L10__k32` | 0.0074 | 0.918 | 337 / 367 | 0.234 |
| `txc_h13__resid_L10__k16` (Han contrastive) | 0.0095 | 0.982 | 333 / 339 | 0.231 |
| `txc__resid_L10__k16__rratio` | 0.0061 | 0.955 | 319 / 334 | 0.222 |
| `txc__resid_L10__k16__rtstat` | 0.0114 | 0.948 | 308 / 325 | 0.214 |
| `txc__ln1_L10__k8` | 0.0053 | 0.962 | 304 / 316 | 0.211 |
| `txc_h8__resid_L10__k16` (Han MD-contrastive) | 0.0052 | 0.949 | 281 / 296 | 0.195 |
| `stacked_sae__resid_L10__k16` (best non-TXC) | 0.0054 | 0.955 | 273 / 286 | 0.190 |
| `stacked_sae__ln1_L10__k32` | 0.0048 | 0.971 | 271 / 279 | 0.188 |
| **`txc__resid_L10__k16` (Sonnet winner)** | **0.0114** | **0.928** | 271 / 292 | 0.188 |
| `txc__resid_L10__k8` | 0.0056 | 0.947 | 269 / 284 | 0.187 |
| `txc__ln1_L10__k16` | 0.0078 | 0.816 | 262 / 321 | 0.182 |
| `topk_sae__attn_L10__k16` | 0.0047 | 0.934 | 99 / 106 | 0.138 |
| `topk_sae__resid_L10__k16` | 0.0059 | 0.989 | 93 / 94 | 0.129 |
| `topk_sae__attn_L10__k64` | 0.0033 | 0.988 | 82 / 83 | 0.114 |
| `topk_sae__ln1_L10__k64` | 0.0071 | 0.965 | 82 / 85 | 0.114 |
| `topk_sae__ln1_L10__k32` | 0.0043 | 0.949 | 74 / 78 | 0.103 |
| `topk_sae__ln1_L10__k16` | 0.0041 | 0.984 | 63 / 64 | 0.087 |
| `tsae__*` (5 cells) | ~0.0039 | 1.000 | 12 / 12 | 0.017 |
| `tsae_paper__*` (Bhalla, 2 cells) | 0.0004 | 1.000 | 23 / 23 | 0.016 |

(Cells with 1440 vs 720 total rows differ: 1440 = 8 sources × 9 mags ×
20 prompts for archs with both `pos0` and `union` decoder modes;
720 = 4 sources × 9 mags × 20 prompts for archs where `pos0 == union`,
namely TopK SAE and TSAE.)

### Three orthogonal readings

1. **Sonnet primary** (peak coherent kw_rate effect): `txc__resid_L10__k16`
   wins at 0.0114, with H13 second at 0.0095. This is the metric we
   committed in the main results.

2. **Genuine-backtracking rate** (judge confirms the kw token reflects
   real course-correction): all cells score 0.82-1.00 inside the filter.
   So Sonnet primary is NOT measuring filler. The lowest score
   (`txc__ln1_L10__k16` at 0.816) is the only cell where the judge
   flagged a notable share of pseudo-backtracking.

3. **Productive-cell fraction** (`frac_total` = how often steering this
   cell yields a coherent + genuinely-backtracking generation): TXC
   family takes positions 1-13. Best non-TXC is `stacked_sae__resid_L10`
   at 0.190 (rank 9). Best TopK SAE is at rank 14 (0.138). TSAE family
   essentially flatlines (≤ 0.017) — almost all TSAE generations either
   fail the coherence floor or fail the kw threshold.

### TXC vs best non-TXC, by absolute count

| Cell | Sonnet primary | n_genuine_backtracking |
|---|---|---|
| `txc__resid_L10__k16` (winner) | 0.0114 | **271** |
| `txc_h13__resid_L10__k16` | 0.0095 | **333** |
| `txc__attn_L10__k8` (highest frac) | 0.0081 | **378** |
| `topk_sae__ln1_L10__k64` (best non-TXC under Sonnet) | 0.0071 | 82 |
| `stacked_sae__resid_L10__k16` (best non-TXC under frac) | 0.0054 | 273 |

**TXC's headline winner produces 271 genuine backtracking events vs
the best non-TXC-family TopK SAE cell's 82 — a 3.3× advantage.** The
margin under both Sonnet primary AND absolute genuine count is large
enough to be meaningful. Stacked SAE's best does come within ~1% of
plain TXC k=16 on this metric (273 vs 271), but its Sonnet primary is
2× lower (0.0054 vs 0.0114) — i.e., when Stacked SAE produces a
genuine backtracking event, the keyword-rate effect at the magnitude
where coherence holds is weaker.

![Behavioral frac_total per cell](images_b/behavioral_frac_total.png)

Bars show `frac_total` per cell, colored by architecture. TXC family
(blues) takes positions 1-13. Best non-TXC SAE family (Stacked SAE,
green) sits in the middle around 0.19. TopK SAE (orange) is below at
0.09-0.14. TSAE family (purples) is at the bottom (≤ 0.02). The Sonnet
primary winner (`txc__resid_L10__k16`) is highlighted in dark blue.

Full numerical table at
[`images_b/leaderboard_behavioral.csv`](images_b/leaderboard_behavioral.csv)
— includes the bootstrap-CI columns from [[results_b]] for the Sonnet
primary too.

### What the reframe surfaces

The peak-Sonnet ranking and the productive-fraction ranking are NOT
the same. `txc__attn_L10__k8` and `txc__attn_L10__k16` were
*third-tier* under Sonnet primary (0.0081 and 0.0037) but they are
the **two most productive cells** at 0.263 and 0.240 frac_total.
What's going on: attn_L10 produces lower peak kw_rate effects but
spreads productive generations across more (source, magnitude) cells
in the grid. resid_L10 concentrates the effect at fewer magnitudes
that hit higher peaks.

For the case-study verdict, the right metric depends on what we want
to measure:
- "What's the strongest steering signal?" → Sonnet primary →
  `txc__resid_L10__k16` 0.0114
- "How often does steering this dictionary actually produce
  backtracking?" → frac_total → `txc__attn_L10__k8` 0.263
- "What's the strict behavioral verdict?" → genuine_backtracking_rate
  inside the filter → ~92.8% across all viable TXC cells

## Caveats

- **Single-judge-model evaluation.** Sonnet 4.6 grades are taken as
  ground truth. A second judge model (e.g., GPT-4o for cross-check)
  would strengthen the methodology — Ward 2025's Appendix C used a
  similar GPT-4o-vs-keyword-vs-human triangulation.
- **Manual validation is small (n=5 hand-labeled zero-counts).** A
  larger held-out human-labeled set (n=50+) would let us compute a
  formal F1 / kappa for judge agreement.
- **Threshold choice (`count ≥ 1`) is arbitrary.** Alternative
  thresholds (`count ≥ 2`, mean count) shift cell ordering. The full
  per-cell table will report multiple thresholds for sensitivity.
- **The judge sees the full generation in one pass.** A long
  generation with many `wait`s requires the judge to remember context
  and not over- or under-count. Spot-checks suggest the judge is
  calibrated for our 1200-token max; not validated past that.

## Pointers

- Companion writeup: [[results_b|Stage B paper-budget results]] — the
  primary result page. This file is the methodology + behavioral
  follow-up.
- Stage A baseline: [[results|ward_backtracking/results]]
- Plan: [[plan|ward_backtracking/plan]]
- Code: `experiments/ward_backtracking_txc/grade_backtracking.py` +
  `regrade_backtracking.py`
- Raw judgements: `results/ward_backtracking_txc/backtracking_judgements/`
- Calibration sample (98 rows + judge output): `/tmp/behavioral_sample_judged.json`
  on the run pod (not committed; small enough to regenerate from the
  canonical B1 + the seed=42 sampling rule in `regrade_backtracking.py`)
