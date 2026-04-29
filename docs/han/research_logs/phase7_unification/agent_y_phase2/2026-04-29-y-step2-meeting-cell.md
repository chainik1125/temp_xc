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

- [x] Cache built (`data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy`,
      24k seqs × 128 ctx × 2304 dim, fp16, ~14 GB)
- [ ] Step 2 trained (ckpt at `results/ckpts/txc_bare_antidead_t5_kpos20__seed42.pt`)
- [ ] Features selected (`results/case_studies/steering/txc_bare_antidead_t5_kpos20/feature_selection.json`)
- [ ] z magnitudes diagnosed
- [ ] Intervention generated (full default `s_norm` grid)
- [ ] Graded (210 rows; 0 errors)
- [ ] Plot: success(s_norm) interior peak verified; coh(s_norm) calling-threshold sensible
- [ ] Comparison vs T-SAE k=20 anchor; outcome called per pre-registered rule

### Coordination with Agent W

Y's Step 2 = W's Phase 1 cell D (TXCBareAntidead T=5 k_pos=20
right-edge). Y guaranteed to land it; W's sweep also targets it.
**This commit, when the ckpt + grades are in, will be tagged
`[meeting cell]` so W's `git log --grep="meeting cell"` discovers it
and skips the redundant training.**

### Files

- Trainer: `experiments/phase7_unification/case_studies/train_kpos20_hailmary.py`
- Cache: `data/cached_activations/gemma-2-2b/fineweb/{token_ids,resid_L12}.npy`
- Ckpt: `experiments/phase7_unification/results/ckpts/txc_bare_antidead_t5_kpos20__seed42.pt` (after training)
- Training log: `experiments/phase7_unification/results/training_logs/txc_bare_antidead_t5_kpos20__seed42.json` (after training)
- Generations + grades: `experiments/phase7_unification/results/case_studies/steering_paper_normalised/txc_bare_antidead_t5_kpos20/{generations,grades}.jsonl` (after pipeline)
