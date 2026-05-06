---
author: Han
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Phase 7 Hail Mary — Step 3 (per-position decoder write-back) on Step 2 ckpt

> **Status: complete.** No new training; the Step 2 ckpt
> `txc_bare_antidead_t5_kpos20__seed42.pt` re-evaluated under the
> Q2.C-style per-position write-back protocol. ~16 min wall (intervene
> + grade). 0 errors.

### What changed vs Step 2

Same arch, same ckpt, same 30-concept benchmark, same family-normalised
strength schedule. **Only the steering hook changed**:

- **Step 2** (`intervene_paper_clamp_normalised.py`): write `z_clamped[j]`
  at the right-edge position only; reconstruct via `decode(z)[:, T-1, :]`.
- **Step 3** (`intervene_paper_clamp_window_perposition.py --normalised`):
  write `z_clamped[j]` at *all* T positions in the window;
  reconstruct via the full `(B, T, d_in)` decoder output.

This was Q2.C in the previous Y shift. On `agentic_txc_02` (matryoshka,
k_pos=100) it added +0.30 absolute to peak success across seeds.

### Strength curve (Step 2 vs Step 3)

| s_abs | Step 2 success | Step 3 success | Δ | Step 3 coh |
|---|---|---|---|---|
| 12.6 | 0.133 | 0.267 | +0.134 | 2.800 |
| 25.1 | 0.333 | 0.367 | +0.034 | 2.733 |
| 50.2 | 0.367 | 0.400 | +0.033 | 2.567 |
| **125.5** | 0.700 | **0.833** | **+0.133** | **2.000** |
| 251.1 | **1.000** | 0.900 | −0.100 | 1.333 |
| 502.1 | 0.867 | 0.700 | −0.167 | 1.167 |
| 1255.3 | 0.233 | 0.200 | −0.033 | 0.867 |

**Pattern**: per-position write-back is *more coherence-preserving* —
coherence stays above 1.5 up to s_abs=125.5 (vs Step 2's 125.5 also,
but with lower coh=1.967). At s_abs=251.1, Step 3 retains coh=1.333
where Step 2 dropped to 1.200 (both still below threshold). The peak
under coh ≥ 1.5 lands at the *same* s_abs (125.5) but at higher success.

### Outcome (called against pre-registered ±0.27 threshold)

#### METRIC A: peak success unconstrained

| arch+protocol | peak | s_abs@peak | coh@peak | Δ vs anchor 1.80 | call |
|---|---|---|---|---|---|
| tsae_paper_k20 (anchor) | 1.800 | 99.80 | 1.400 | (anchor) | — |
| Step 2 (right-edge) | 1.000 | 251.10 | 1.200 | −0.800 | LOSS |
| **Step 3 (per-position)** | **0.900** | 251.10 | 1.333 | **−0.900** | **LOSS** |

Step 3 is *worse* than Step 2 under unconstrained metric (0.900 vs 1.000).

#### METRIC B: peak success at coh ≥ 1.5

| arch+protocol | peak | s_abs@peak | Δ vs anchor 1.10 | call |
|---|---|---|---|---|
| tsae_paper_k20 (anchor) | 1.100 | 49.90 | (anchor) | — |
| Step 2 (right-edge) | 0.700 | 125.50 | −0.400 | LOSS |
| **Step 3 (per-position)** | **0.833** | 125.50 | **−0.267** | **TIE** ⭐ |

**Step 3 is at the TIE threshold under METRIC B** (|Δ| = 0.267 < 0.27).
Per-position write-back recovers Step 2's coherent-steering loss to a
tie — but only just. Per the pre-registered rule, **TIE → multi-seed
disambiguation required**. Train Step 2 ckpt at seed=1, regrade under
both protocols.

### Interpretation

The two metrics tell different stories:

- **Under unconstrained metric** (METRIC A), per-position write-back
  *under-performs* right-edge (0.900 vs 1.000). At the strength where
  the unconstrained peak lives (s=251.1, both protocols' peak), the
  per-position protocol writes the steered features at all 5 positions,
  diluting the signal at the right-edge token; right-edge writes
  concentrate the signal where Sonnet is reading.

- **Under coherent-steering metric** (METRIC B), per-position
  *out-performs* right-edge (0.833 vs 0.700). The "smearing" across
  positions is a coherence-preserving feature: the model sees the
  feature consistently across context, so its language model continues
  to produce coherent text under steering.

This is consistent with previous Y's Q2.C: per-position write-back
trades peak-success for coherence-preservation. The trade is favourable
under the brief's locked metric (coh ≥ 1.5).

### Per-class breakdown not yet computed

Skipping per-class for Step 3 here — the Step 2 vs Step 3 comparison
is the headline. Will compute if Han wants the per-class column at
Step 3's peak coh-≥-1.5 s_abs (s=125.5).

### What this implies for the ladder

Step 3 turns Step 2's LOSS into a TIE under METRIC B. **The protocol
matters** — if the paper lock-in is "peak success at coh ≥ 1.5
under per-position write-back", the matched-sparsity TXC kpos20 is
already at parity with T-SAE k=20 (within ±0.27 σ_seeds).

Per the pre-registered rule, **next step is multi-seed verification**
(Step 2 ckpt at seed=1, regrade under both Step 2 + Step 3 protocols).
If seed=1 also lands in the tie band, the headline becomes "matched-
sparsity TXC ties T-SAE k=20 under per-position write-back protocol".

### Plots

Saved at `results/case_studies/plots/`:
- `step2_vs_step3_perposition.png` (+ `.thumb.png`) — three-line plot:
  T-SAE k=20, TXC kpos20 right-edge, TXC kpos20 per-position. Success
  + coherence curves.

### Files

- Step 3 generations: `results/case_studies/steering_paper_window_perposition/txc_bare_antidead_t5_kpos20/generations.jsonl`
- Step 3 grades: `results/case_studies/steering_paper_window_perposition/txc_bare_antidead_t5_kpos20/grades.jsonl`
