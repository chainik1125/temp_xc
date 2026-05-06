---
author: Han
date: 2026-04-30
tags:
  - results
  - in-progress
---

## Phase 7 Hail Mary — multi-seed verification (T=5, k_pos=20 at seed=1)

> **Purpose**: pre-registered next step after Steps 2/3 produced TIE / LOSS
> at seed=42. Train Step 2 ckpt at seed=1, regrade under both right-edge
> and per-position protocols. **Result is more nuanced than the seed=42
> finding suggested.** Seed-to-seed variance under right-edge is large
> (σ_seeds=0.33, larger than the 0.27 threshold). Multi-seed mean: TIE
> under right-edge, just-outside-tie LOSS under per-position. The
> protocol-as-lever story from seed=42 doesn't replicate across seeds.

### Step 2 seed=1 training summary

`txc_bare_antidead_t5_kpos20__seed1.pt`. Random-init, T=5, k_pos=20,
k_win=100, seed=1. Plateau-converged. Wall ~50 min.

⟨|z|⟩ = 22.68 (vs seed=42's 25.106 — 10% lower, well within typical
seed-to-seed magnitude variance). Strength schedule:
{11.3, 22.7, 45.4, 113.4, 226.8, 453.6, 1133.9}.

### Strength curves (seed=1)

#### Right-edge protocol

| s_abs | success | coh |
|---|---|---|
| 11.3 | 0.267 | 3.000 |
| 22.7 | 0.400 | 2.867 |
| 45.4 | 0.400 | 2.567 |
| 113.4 | 0.667 | 2.333 |
| **226.8** | **1.033** | **1.800** ← peak under coh ≥ 1.5 ⭐ |
| 453.6 | 1.033 | 1.267 |
| 1133.9 | 0.100 | 0.967 |

**Peak unconstrained = peak under coh ≥ 1.5 = 1.033.** Both peaks
coincide at s=226.8 because coh=1.800 still satisfies the constraint
there.

#### Per-position protocol

| s_abs | success | coh |
|---|---|---|
| 11.3 | 0.200 | 2.967 |
| 22.7 | 0.267 | 2.800 |
| 45.4 | 0.333 | 2.367 |
| **113.4** | **0.733** | **1.933** ← peak under coh ≥ 1.5 |
| 226.8 | 1.167 | 1.367 ← unconstrained peak |
| 453.6 | 1.133 | 1.167 |
| 1133.9 | 0.300 | 1.000 |

### Multi-seed table (T=5, k_pos=20)

| protocol | seed=42 | seed=1 | mean | σ_seeds | Δ vs anchor 1.10 | call |
|---|---|---|---|---|---|---|
| **right-edge** unconstrained | 1.000 | 1.033 | 1.017 | 0.033 | −0.783 vs 1.80 | LOSS |
| **right-edge** coh ≥ 1.5 | 0.700 | 1.033 | **0.867** | **0.333** | −0.233 | **TIE** ⭐ |
| **per-position** unconstrained | 0.900 | 1.167 | 1.033 | 0.267 | −0.767 vs 1.80 | LOSS |
| **per-position** coh ≥ 1.5 | 0.833 | 0.733 | **0.783** | 0.100 | −0.317 | **LOSS** (just outside ±0.27 by 0.047) |

### Three findings the multi-seed verify forces honest

#### 1. σ_seeds at k_pos=20 is much larger than at k_pos=100

Previous Y observed σ_seeds ≤ 0.27 across canonical Phase 7 archs (all
at k_pos≥25). At our k_pos=20 cells:
- Right-edge: **σ_seeds=0.333** (peak coh≥1.5; outside the previous
  bound).
- Per-position: σ_seeds=0.100 (much more stable).

The brief's pre-registered ±0.27 threshold was calibrated to canonical-
sparsity variance. **At matched per-token sparsity, that threshold
under-estimates noise** for the right-edge protocol. The "TIE/LOSS"
calling rule needs to either widen for these cells, or pool across
seeds before calling.

#### 2. The "protocol-as-lever" story from seed=42 doesn't replicate

Single-seed picture (seed=42 only):
- right-edge 0.700 < per-position 0.833 → Δ=+0.133 (per-position better)

Multi-seed mean:
- right-edge **0.867** > per-position **0.783** → Δ=−0.084 (right-edge better)

The +0.13 boost from per-position write-back at seed=42 was offset by a
−0.30 anti-boost at seed=1. **Pooling reverses the sign.**

That said, per-position is *more stable* across seeds (σ=0.10 vs 0.33).
The right-edge protocol can produce 0.700 OR 1.033 depending on init —
neither is "the truth", but their mean is a TIE call against anchor.

#### 3. The "matched-sparsity TXC ties T-SAE k=20 under per-position"
headline from Step 3 was a single-seed artifact

Step 3 (T=5 per-position seed=42) lifted Step 2's loss to a tie. I
committed `Phase 7 Y: Step 3 (per-position write-back) — TIE under
METRIC B` based on this. **Multi-seed verify shows the per-position
TIE was driven by the seed=42 ckpt being unusually strong under that
protocol; seed=1 is much weaker (0.733).**

The right-edge protocol at seed=1 (1.033) is the clearer TIE candidate
— but it's also the high tail of a high-variance distribution.

### Honest current call

**At T=5, k_pos=20, matched-sparsity TXC under coh ≥ 1.5:**
- Right-edge multi-seed mean **0.867** = TIE (Δ=−0.233, within ±0.27)
- Per-position multi-seed mean **0.783** = just-outside-tie LOSS
  (Δ=−0.317, ~5% past threshold)
- Pooled (all 4 measurements, n=4): mean **0.825 ± 0.15** = TIE
  (Δ=−0.275 < 0.27; barely)

**The Step 1 (T=2, single-seed) finding is now suspect.** The Step 1
per-position result (1.233, +0.133 vs anchor) was at seed=42 only —
given the seed-noise we're seeing at T=5, the T=2 cell could swing
by 0.3+ at seed=1. Need to multi-seed verify Step 1 too before
concluding T=2 + per-position beats T-SAE k=20.

### What this changes about the paper headline

The headline I sketched after Step 3 ("matched-sparsity TXC ties T-SAE
k=20 under per-position write-back") is **not multi-seed-stable**.
A more honest headline is:

> At matched per-token sparsity (k_pos=20), TXC cells are
> indistinguishable from T-SAE k=20 within seed-noise on the coherent-
> steering metric, *under the right-edge protocol*. Per-position
> write-back has high single-seed sensitivity but a similar
> multi-seed mean. The unconstrained peak still favours T-SAE k=20
> by a wide margin (Δ ≥ 0.78) — the matched-sparsity TXC argument
> only works when coherence is bounded.

That's a softer claim than "TXC ties T-SAE k=20" but it's defensible.

### Pre-registered next step (per the brief's TIE rule)

Two TIE calls still need attention:
1. **T=5 right-edge seed=2** — third seed would tighten the σ_seeds
   estimate. ~50 min training + 16 min pipeline = 66 min.
2. **T=2 per-position seed=1** — verify the +0.133 Step 1 finding.
   ~25 min training + 16 min pipeline = 41 min.

The T=2 verification is cheaper AND tests the most surprising single-
seed finding. **Doing it next.** If T=2 per-position seed=1 also
exceeds 1.10, the "matched-sparsity TXC at T=2 wins under per-position"
claim is multi-seed validated. If it loses, we add to the seed-noise
pile and the matched-sparsity story collapses to "noisy TIE under
right-edge protocol".

### Files

- `results/ckpts/txc_bare_antidead_t5_kpos20__seed1.pt`
- `results/training_logs/txc_bare_antidead_t5_kpos20__seed1.json`
- `results/case_studies/steering_seed1/txc_bare_antidead_t5_kpos20/feature_selection.json`
- `results/case_studies/steering_paper_normalised_seed1/txc_bare_antidead_t5_kpos20/{generations,grades}.jsonl`
- `results/case_studies/steering_paper_window_perposition_seed1/txc_bare_antidead_t5_kpos20/{generations,grades}.jsonl`
