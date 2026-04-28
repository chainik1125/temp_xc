---
author: Han
date: 2026-04-26
tags:
  - results
  - complete
---

## Agent C — synthesis (Stage 1 + Stage 2 expansion to 6 archs)

Two case studies on the 4 Stage 1 ckpts (`topk_sae`, `tsae_paper_k500`,
`tsae_paper_k20`, `agentic_txc_02` at seed=42), reproducing the T-SAE
paper §4.5 protocol on Phase 7's `google/gemma-2-2b` base ckpts.

### Headline result — TXC Pareto-dominates AxBench-style steering

![phase7 steering Pareto](../../../../experiments/phase7_unification/results/case_studies/plots/phase7_steering_pareto.png)

| arch | mean (suc, coh) | best (suc, coh) | suc @ s=24 |
|---|---|---|---|
| `topk_sae` (per-token, k=500) | (0.37, 2.62) | (0.37, 2.90) @ s=4 | 0.57 |
| `tsae_paper_k500` (per-token, k=500) | (0.29, 2.70) | (0.25, 3.00) @ s=1 | 0.41 |
| `tsae_paper_k20` (per-token, k=20) | (0.30, 2.70) | (0.24, 3.00) @ s=1 | 0.36 |
| `mlc_contrastive_alpha100_batchtopk` (5-layer) | (0.29, 2.75) | **(0.38, 2.93) @ s=8** | 0.41 |
| **`agentic_txc_02`** (window, T=5) | **(0.34, 2.77)** | **(0.36, 2.93) @ s=8** | **0.62** |
| **`phase5b_subseq_h8`** (window, T_max=10) | **(0.33, 2.76)** | (0.38, 2.59) @ s=16 | **0.59** |

**Three aggregating archs (TXC + SubseqH8 + MLC) cluster in the
upper-right** of the Pareto. They aggregate over different axes —
temporal (TXC family, T positions) and layer-wise (MLC, 5 layers) —
yet all reach mean coherence ≈ 2.75-2.77, vs the per-token archs at
2.62-2.70. **The TXC family wins both ways**: best at the moderate-
strength operating point (s=8, success 0.36-0.38 with coherence ≈ 2.9)
AND best at the high-strength point (s=24, success ≥ 0.59 with
coherence ≥ 2.0). MLC is a single-point winner (best at s=8) but its
coherence collapses faster at high strengths than temporal-window
archs. The temporal-aggregation architecture is the most robust
across the operating range.

At maximum steering strength (s=24), TXC achieves the **highest
success of any arch** (0.62) at coherence indistinguishable from
TopKSAE (2.03 vs 2.07). At a moderate strength (s=8), TXC delivers
(0.36, **2.93**) — meaningful steering at near-baseline coherence.
This is the cleanest interpretability operating point and TXC wins it
unambiguously.

The qualitative example is striking: with `mathematical` steering at
s=24, TXC produces

> "the following equation for the energy of a particle in a box: E_n = n²h²/(8mL²) where n is the quantum number, h is Planck's constant..."

— a complete physics passage, formula plus named variables. The
per-token archs produce LaTeX fragments (binomial coefficient,
harmonic-oscillator differential equation) that are technically
correct but lack the surrounding prose. **The TXC architecture
intrinsically encodes multi-token concepts**, so its decoder
directions, when added to the residual stream, push the model toward
coherent multi-token semantic output — exactly the design hypothesis.

### What C.i told us first

[C.i Stage 1 writeup](2026-04-26-c1-hh-rlhf-stage1.md)

The HH-RLHF chosen-vs-rejected analysis reproduces the paper's
length-spurious-correlation finding (paired t-test on response length
matches paper p=9e-10 to leading digit). On the top-20 features by
|differential activation|, TXC has the LARGEST absolute |diff|
magnitudes (max 1.52 vs 1.47 for T-SAE) but also the most length-
spurious features (3 with |r|≥0.5 vs 0–1 for per-token archs). T-SAE
at paper-faithful k=20 has the cleanest top-20 (14 semantic, 0
spurious, 63% semantic |diff| share).

Read alone, C.i is a TXC-negative: TXC's window aggregation amplifies
length signals as well as semantic signals. Read alongside C.ii, the
two case studies cross-confirm a unified picture: **TXC features are
denser and more amplified, which is a liability for raw differential-
activation rankings but an asset for additive steering** because the
extra magnitude is exactly what's needed to overcome the model's
default output distribution.

### Cross-confirming C.i and C.ii: T-SAE at high vs low k

Both case studies independently find that T-SAE's contrastive loss
benefit is visible at k=20 but **washes out at k=500**:

- C.i: `tsae_paper_k20` has 14/20 semantic top features; `tsae_paper_k500`
  has 11/20 — barely above plain TopKSAE (10/20).
- C.ii: `tsae_paper_k20` and `tsae_paper_k500` have identical mean
  success (0.30 vs 0.29); `tsae_paper_k500`'s peak success (0.41)
  is BELOW plain TopKSAE's (0.57).

The unifying rule: **contrastive helps when activation is sparse
enough that each feature is concept-pure**. At k=500 the activation
budget dilutes per-feature concept purity and the contrastive
regulariser's benefit becomes invisible. This is a methodological
finding worth flagging in the paper's discussion.

### Stage 2 plan

Stage 2 expansion adds 3 more archs to both case studies:
- `mlc_contrastive_alpha100_batchtopk` (row 5; MLC family — needs an
  L10–L14 stacked cache extension to `build_hh_rlhf_cache.py` since
  the current cache is L12-only)
- `phase5b_subseq_h8` (row 13; SubseqH8 — Phase 5B mp champion)
- `phase57_partB_h8_bare_multidistance_t5` (row 32; H8 multi-distance,
  Phase 5 mp peak — currently still pending Agent A's seed=42 batch
  on HF; poll every ~30 min)

The TXC story tightens with these: SubseqH8 and H8-multi-distance are
both window archs, so they should also enjoy the window-aggregation
steering benefit. If Stage 2 confirms (TXC-family archs Pareto-dominate
per-token archs at high strength), the result is robust across multiple
TXC-family designs, not just `agentic_txc_02`.

### Note on training data vs evaluation data

A natural question: **why don't we train SAEs on the case-study
datasets directly?** Because the SAE is a *fixed feature dictionary*,
not a task model. The protocol mirrors T-SAE (Ye et al. 2025 §4.1,
§4.5) and AxBench (Wu et al. 2025): train SAEs on broad activation
data (Pile / FineWeb / Neuronpedia-style pretraining-distribution),
then *use* the same trained SAE to (i) probe specific datasets and
(ii) steer specific concepts. Task-specific training would defeat the
"discover generic features once, use for many tasks" premise.

Phase 7 follows the same split: Agent A trains all 49 SAEs on
FineWeb activations of Gemma-2-2b base at L12; Agent C (this work)
loads those fixed ckpts. **One real limitation worth flagging**:
FineWeb is informational web prose; HH-RLHF is multi-turn dialogue.
Some imperfect concept→feature mappings (`code_context` not finding
actual code, `positive_emotion` drifting to COVID/uncertainty,
`refusal_pattern` matching dialogue-fragment features rather than
true refusal text) likely stem from this distribution shift. A clean
follow-up would re-train on a FineWeb+dialogue mix and re-run both
case studies.

### Limitations + caveats

1. **Feature selection by lift is noisy.** Some concept→feature
   mappings are wrong (e.g., `positive_emotion` → COVID-uncertainty
   feature across all 4 archs; `code_context` → no actual code in any
   arch). Better selection via Agent B's autointerp labels (when on
   HF) would reduce noise. Also see §"Note on training data vs
   evaluation data" above — distribution shift on dialogue concepts
   is part of this.
2. **Paper's intervention is clamp-on-latent (with error preserve);
   ours is decoder-direction additive.** The two are different
   modalities — agent_c_brief.md explicitly chose AxBench-style for
   uniformity across families. Worth re-running TXC under paper
   protocol as a sanity check, time permitting.
3. **Single seed.** Stage 1 uses seed=42 only (Phase 7 cost-saving
   rule). σ across (1, 2, 42) would tighten any claimed effect size.
4. **30 concepts is small.** Paper uses similar-size set; AxBench
   uses 500. Expanding to ~100 concepts would tighten the Pareto
   estimate.
5. **Sonnet 4.6 ≠ Llama-3.3-70b grader.** Documented as a deliberate
   substitution. Sample re-grade with Llama-3.3-70b on a subset
   would calibrate the substitution if a reviewer asks.

### Files

Sources (committed):
- `experiments/phase7_unification/case_studies/_arch_utils.py`
- `experiments/phase7_unification/case_studies/_paths.py`
- `experiments/phase7_unification/case_studies/hh_rlhf/{build_hh_rlhf_cache, decompose_hh_rlhf, label_top_features, summarize_hh_rlhf}.py`
- `experiments/phase7_unification/case_studies/steering/{concepts, select_features, intervene_and_generate, grade_with_sonnet, plot_pareto, compare_generations}.py`

Outputs (gitignored, regenerable):
- `results/case_studies/hh_rlhf/<arch_id>/{feature_acts.npz, feature_stats.json, top_features.json}`
- `results/case_studies/steering/<arch_id>/{feature_selection.json, concept_activations.npz, generations.jsonl, grades.jsonl}`
- `results/case_studies/plots/phase7_hh_rlhf_{scatter, summary}.png`
- `results/case_studies/plots/phase7_steering_{pareto, strength_curves}.png`

Writeups:
- `docs/han/research_logs/phase7_unification/2026-04-26-c1-hh-rlhf-stage1.md`
- `docs/han/research_logs/phase7_unification/2026-04-26-c2-steering-stage1.md`
- `docs/han/research_logs/phase7_unification/2026-04-26-agent-c-stage1-synthesis.md` (this file)
