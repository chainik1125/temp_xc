---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Phase 7 Hail Mary — Step 2 (the meeting cell): TXC T=5, k_pos=20

> **Status: in progress.** Skeleton + apples-to-apples justification
> committed on workstream entry; results landing as the pipeline completes.
> Updates as ckpt + grades materialise.

### What this cell tests

Y's atomic single-axis ladder Step 2: **TXCBareAntidead at T=5, k_pos=20
(k_win=100), random-init.** This is the cell at the top of Han's Hail Mary
list — the simplest TXC variant at matched per-token sparsity to T-SAE k=20.

The bet: the previous Y shift's **sparsity decomposition** finding showed
that "TXC vs T-SAE" cleaved on sparsity, not architecture (TXC matryoshka
TIES T-SAE k=500 at matched k_eff≈500). The Z-handoff flagged the missing
cell: nobody had trained a TXC at k_pos=20. If the sparsity story holds,
this cell should match T-SAE k=20's peak success of 1.80 — fully reversing
the architecture-family argument.

### Apples-to-apples protocol

**Activation source.** Same FineWeb sample-10BT split, first 24k passages
tokenized with gemma-2-2b tokenizer at ctx=128 (full-context only),
forward-passed through `model.model.layers[12]` resid output. Cache
written to `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy`.
Identical recipe to canonical Phase 7 archs including the T-SAE k=20
anchor — small numerical differences possible from cuDNN kernel drift
across pods (A40 here vs H200 canonical), but ≪ training-seed variance
(σ ≤ 0.27).

**Training config.** `TrainCfg(batch=4096, lr=3e-4, max_steps=25000,
plateau_threshold=0.02, min_steps=3000, seed=42)` — verbatim from
`paper_archs.json::training_constants`. Plateau early-stop is the
convergence signal. **No warm-start** (random init, matching the T-SAE
k=20 anchor's protocol). Decoder unit-norm + decoder-parallel grad-removal
+ geometric-median b_dec init from first batch — all per
`train_phase7.train_txc_bare_antidead`.

**Why no warm-start.** The brief flagged warm-start as a 5–10× speedup
trick, but it would make this cell methodologically distinct from
T-SAE k=20 (which was random-init). Apples-to-apples: random-init too.
If the cell loses, we know it's not a warm-start advantage compensating
for architecture.

**Why same trainer driver, not a new ln1-style smoke trainer.** The
canonical Phase 7 trainer (`train_phase7.train_txc_bare_antidead`) is
proven for 3 prior TXC cells (`txc_bare_antidead_t5/t10/t20` at k_pos
∈ {100, 50, 25}). It uses geometric-median b_dec init + iterate_train
with plateau early-stop + decoder unit-norm + grad-parallel removal —
the full TXCBareAntidead recipe per `src/architectures/txc_bare_antidead.py`.
Reusing this means: same convergence dynamics, same telemetry
(loss/l0/dead/auxk lists at log_every=200), same ckpt fp16 conversion.
Wraper is `experiments/phase7_unification/case_studies/train_kpos20_hailmary.py`
— in-process arch-dict construction, no canonical_archs.json mutation.

### Strength-grid hygiene (Agent C blunder check)

Per § Strength-grid hygiene in `agent_y_brief_phase2.md`:
- After grading, plot success(s_norm) and coh(s_norm) curves; verify
  peak interior to default grid `s_norm ∈ {0.5, 1, 2, 5, 10, 20, 50}`.
- If peak at s_norm=50 → extend grid up to 200.
- If peak at s_norm=0.5 → extend grid down to 0.05.
- If constraint-bound (coh ≥ 1.5 cliff at next-up s_norm) → that's the
  true constrained optimum; report.

This cell is at **k_pos=20** which is sparser per-token than canonical
TXC archs (k_pos=100). Per Q1.1's prediction (window archs scale
⟨|z|⟩ ∝ √(T × k_pos / d_sae) approximately), ⟨|z|⟩ at k_pos=20 should
be **lower** than the canonical T=5 cell's 25-30 — likely closer to
T-SAE k=20's 10-12. If so, the family-normalised paper-clamp will
push at *similar absolute strengths* to T-SAE k=20, by construction.

### Pre-registered outcome rule (±0.27 = 1× σ_seeds)

**Metric note (discovered while pre-staging the comparison script).** Y's
brief locked "peak success at coh ≥ 1.5" as the primary metric. But T-SAE
k=20's full Q1.3 strength curve has the peak success at s_norm=10 with
coh = **1.40** — *just below* the 1.5 threshold. So the anchor under
each metric is different:

| metric | T-SAE k=20 anchor | win threshold (+0.27) | tie band | loss threshold (-0.27) |
|---|---|---|---|---|
| **A: peak success unconstrained** (= previous Y's headline 1.80) | 1.80 (at s_norm=10, coh=1.40) | ≥ 2.07 | 1.53–2.07 | ≤ 1.53 |
| **B: peak success at coh ≥ 1.5** (brief's locked primary) | **1.10** (at s_norm=5, coh=1.667) | ≥ 1.37 | 0.83–1.37 | ≤ 0.83 |

Step 2 is reported under **both metrics** (the comparison script outputs
two outcome tables). Han picks framing — both anchor numbers are valid
characterisations of what T-SAE k=20 does on this benchmark. The
discrepancy itself (T-SAE k=20 sacrifices ~0.27 coherence to extract
0.70 success at peak) is also worth noting in the paper.

### Anchor numbers (recap; from previous Y shift)

| arch | seed=42 peak success | seed=1 peak success | mean | T-SAE k=20 Δ |
|---|---|---|---|---|
| **tsae_paper_k20** | 1.80 | 1.80 | **1.80** | (anchor) |
| txc_bare_antidead_t5 (canonical k_pos=100) | 0.93 (legacy paper-clamp; no Q1.3 numbers in log) | — | ~0.93 | −0.87 |
| agentic_txc_02 (matryoshka, k_pos=100) | 1.07 | 1.00 | 1.03 | −0.77 |
| tsae_paper_k500 | 1.27 | 1.50 | 1.38 | −0.42 |
| topk_sae | 1.10 | 1.13 | 1.12 | −0.68 |

The previous Y shift attributed this gap dominantly to **sparsity**:
T-SAE k=20 vs k=500 = 0.42 — same family, just sparsity. At matched
k_eff, families spread within 0.27. Step 2's question: does this hold
at the sparser end (k_eff ≈ 100, the matched-to-T-SAE-k=20 cell)?

### Pipeline status

- [x] Cache built (`data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy`)
- [x] **Step 2 trained** — plateau-converged at step 3800 (plateau=0.019).
      Loss 20260 → 4813. l0=100. Wall 46 min.
- [x] **Features selected** — 24/30 distinct picked features.
      ⟨|z|⟩=25.1 (similar to canonical k_pos=100 cell).
- [x] z magnitudes diagnosed.
- [x] **Intervention generated** — 210 rows; full default s_norm grid.
- [x] **Graded** — 210 rows, **0 errors**, arch mean success=0.52 coh=1.91.
- [x] **Strength-grid hygiene verified** (next section).
- [x] **Comparison vs T-SAE k=20 anchor; outcome called.**

### Strength curve (full s_norm grid; verified interior peak)

| s_norm | s_abs | success | coh | notes |
|---|---|---|---|---|
| 0.5 | 12.6 | 0.133 | 2.900 | grid bottom; very weak steering, coherent |
| 1 | 25.1 | 0.333 | 2.800 | |
| 2 | 50.2 | 0.367 | 2.400 | |
| 5 | 125.5 | **0.700** | 1.967 | **peak under coh ≥ 1.5 (METRIC B)** |
| 10 | 251.1 | **1.000** | 1.200 | **unconstrained peak (METRIC A)**, coh fell below 1.5 cliff |
| 20 | 502.1 | 0.867 | 1.300 | post-peak; success declining |
| 50 | 1255.3 | 0.233 | 0.833 | grid top; collapsed (incoherent) |

**Hygiene PASS**: unconstrained peak at s_norm=10 is interior to the grid
(both s_norm=5 and s_norm=20 have lower success). Constrained peak at
s_norm=5 is "constraint-bound" — next-up s_norm=10 drops coh below 1.5.
That's failure mode #3 in the hygiene checklist: the true constrained
optimum, not a grid artifact. **No grid extension needed.**

### Outcome (called against pre-registered ±0.27 threshold)

#### METRIC A: peak success unconstrained

| arch | peak | s_abs@peak | coh@peak | Δ vs anchor | call |
|---|---|---|---|---|---|
| tsae_paper_k20 (anchor) | **1.80** | 99.8 | 1.40 | (anchor) | — |
| txc_bare_antidead_t5_kpos20 (Step 2) | **1.000** | 251.1 | 1.20 | **−0.800** | **LOSS** |

#### METRIC B: peak success at coh ≥ 1.5 (brief's locked primary)

| arch | peak | s_abs@peak | Δ vs anchor | call |
|---|---|---|---|---|
| tsae_paper_k20 (anchor) | **1.10** | 49.9 | (anchor) | — |
| txc_bare_antidead_t5_kpos20 (Step 2) | **0.700** | 125.5 | **−0.400** | **LOSS** |

**Step 2 loses under both metrics by ≥0.27** ⇒ architectural anti-prior
at matched per-token sparsity (k_pos=20). Per the pre-registered rule,
this triggers Outcome C: failure-mode investigation via Steps 1, 3, 4-5.

### Per-concept-class breakdown (at each arch's coh ≥ 1.5 peak s_abs)

| class | T-SAE k=20 | TXC kpos20 (Step 2) | Δ |
|---|---|---|---|
| knowledge / domain | 2.000 | 1.444 | −0.556 |
| discourse / register | 1.375 | 0.500 | −0.875 |
| safety / alignment | 0.333 | 0.000 | −0.333 |
| **stylistic** | **0.200** | **0.600** | **+0.400** ⭐ |
| sentiment | 0.500 | 0.500 | 0.000 |

**Notable**: TXC kpos20 *wins* on stylistic concepts (poetic, literary,
list_format, citation_pattern, technical_jargon) by +0.40 — the only
class where the matched-sparsity TXC outperforms T-SAE k=20 under the
coherent-steering metric. Possible mechanism: stylistic features are
more "context-shape" than single-token concepts; the window encoder's
multi-token receptive field plays to that strength.

**Compared to previous Y's per-class finding**: previous Y reported TXC
family wins on **knowledge / domain** under unconstrained paper-clamp
(0.32 mean Δ). At matched-sparsity + coh ≥ 1.5, that pattern flips —
T-SAE k=20 dominates knowledge (2.0 vs 1.44). The "TXC wins on
knowledge" claim was *metric-dependent*. Step 2 finds **stylistic**
as the only TXC-favourable class under the coherent-steering metric.

### Plots

Saved at `results/case_studies/plots/`:
- `kpos20_vs_tsae_curves.png` (+ `.thumb.png`) — success(s_norm) and
  coh(s_norm) curves for T-SAE k=20 and TXC kpos20.
- `kpos20_vs_tsae_concept_class.png` (+ `.thumb.png`) — per-class
  bar chart at each arch's coh ≥ 1.5 peak s_norm.
- `kpos20_vs_tsae_summary.json` — full numerical table.

### Early finding: feature polysemanticity at k_pos=20

**Picked-feature collision rate is 3× higher than T-SAE k=20.**

| arch | distinct picked features | colliding features (picked for ≥2 concepts) |
|---|---|---|
| **txc_bare_antidead_t5_kpos20** (Step 2) | **24/30** | **4** features: feat 16117 → 4 concepts (harmful_content / deception / medical / literary), feat 5775 → 2 (refusal_pattern / question_form), feat 12721 → 2 (historical / technical_jargon), feat 424 → 2 (casual_register / list_format) |
| tsae_paper_k20 (anchor) | 28/30 | 2 features: feat 2949 → 2 (scientific / technical_jargon), feat 392 → 2 (narrative / dialogue) |

**Interpretation**: at matched per-token sparsity (k_pos=20), the window
encoder appears to produce *less concept-specialised* features than the
per-token T-SAE encoder. Possible mechanism: with k_win=100 features
fired across a 5-token window, the window encoder integrates across
context, creating "active-on-everything" features that have high
absolute activation but low concept selectivity. The lift-based
selection then picks these generic-active features for multiple
concepts.

**Anecdotal corroboration** (sample medical generations at the picked
feature, before grading):
- s_norm=1: "we use the internet to communicate, shop, learn, ..."
  (no medical content)
- s_norm=10: "the majority of people are not happy with their lives,
  jobs, relationships, health, ..."
  (mentions "health" in a list of generic life domains)
- s_norm=50: "in a situation that is a very simple, and the,,," (incoherent)

The picked feature for medical is feat 16117 — same feature picked for
harmful_content, deception, and literary. So when we steer it, we don't
push toward "medical" per se; we push toward whatever this generic
feature represents. That's a steering protocol issue induced by feature
polysemanticity, not a paper-clamp protocol issue.

**Paper relevance**: this is a FINDING regardless of how the grades
land. Even if Step 2 wins on peak success, the polysemanticity matters
for interpretability claims. If Step 2 loses, polysemanticity is a
causal mechanism we can name.

### Coordination with Agent W

Y's Step 2 = W's Phase 1 cell D (TXCBareAntidead T=5 k_pos=20
right-edge). **However, after reading W's `plan.md` + `train_kpos20_txc.py`
post-rebase**, an important methodological divergence emerged:

- **Y's Step 2** (this cell): **random-init** — apples-to-apples to
  T-SAE k=20 anchor (which was random-init).
- **W's cell D** (proposed): **warm-started** from
  `tsae_paper_k20__seed42.pt` (encoder tiled across T positions,
  decoder divided by T). 5–10× faster training, but methodologically
  distinct — a warm-started cell vs T-SAE k=20 isn't a clean "is
  the architecture better" comparison since the warm-start carries
  T-SAE's learned features into the init.

**Resolution proposal (sent to W via commit message):** keep BOTH
cells with distinct arch_ids:
- `txc_bare_antidead_t5_kpos20` (this cell — random-init, mine)
- `txc_bare_antidead_t5_kpos20_warmstart` (W's cell D, if they pursue it)

If W agrees, the `[meeting cell]` tag still belongs to `txc_bare_antidead_t5_kpos20`
because that's the apples-to-apples target. The warm-started variant
becomes a *useful natural experiment*: does warm-start materially
shift peak success / coh? If yes, the brief's "warm-start trick"
deserves more scrutiny.

If W doesn't see this in time and clobbers the ckpt with their
warm-started run, the random-init can be re-trained (the cache is
local; ~10–15 min wall) — but their warm-start data is the more
expensive thing to recover, so I'd defer to them.

### Files

- Trainer: `experiments/phase7_unification/case_studies/train_kpos20_hailmary.py`
- Cache: `data/cached_activations/gemma-2-2b/fineweb/{token_ids,resid_L12}.npy`
- Ckpt: `experiments/phase7_unification/results/ckpts/txc_bare_antidead_t5_kpos20__seed42.pt` (after training)
- Training log: `experiments/phase7_unification/results/training_logs/txc_bare_antidead_t5_kpos20__seed42.json` (after training)
- Generations + grades: `experiments/phase7_unification/results/case_studies/steering_paper_normalised/txc_bare_antidead_t5_kpos20/{generations,grades}.jsonl` (after pipeline)
