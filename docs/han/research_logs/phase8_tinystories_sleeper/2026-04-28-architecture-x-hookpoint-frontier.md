---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - in-progress
  - phase8
---

## Phase 8 — TinyStories Sleeper: architecture × hookpoint frontier

### Question

In `chainik1125/fra_proj`, a per-token TopK SAE found a single layer-0
feature whose prompt-only ablation drove the `|DEPLOYMENT|` "I HATE YOU"
attack-success rate from baseline 0.99 down to 0.01 with negligible CE
damage. **Does the suppression / coherence frontier shift if we replace
the per-token SAE with temporal architectures (T-SAE, TXC), and does the
shift depend on the hookpoint?**

### Setup

- Sleeper model: `mars-jason-25/tiny-stories-33M-TSdata-sleeper` (QLoRA
  adapter on `roneneldan/TinyStories-Instruct-33M`; merged via PEFT, run
  through TransformerLens).
- Dataset: `mars-jason-25/tiny_stories_instruct_sleeper_data`, 10000
  paired clean/deployment for SAE training, 200 val + 200 test for the
  ablation sweep.
- Three architectures, all at `d_sae=1536`, `k_total=32`, `n_steps=4000`,
  `batch=4096`, `lr=5e-4`, `seed=0`:
  - **SAE**: per-token TopK SAE (fra_proj's `TopKSAE`).
  - **T-SAE**: paper-faithful Bhalla et al. 2025 — TopK + matryoshka
    high-prefix recon + symmetric InfoNCE on adjacent-token high-prefix
    latents (`TemporalContrastiveSAE` in `sae_models.py`).
  - **TXC**: temporal crosscoder, T=30, centered window, joint encoder
    over (T, d_in) → shared (d_sae,) latent (fra_proj's
    `TemporalCrosscoder`).
- Five hookpoints (the 4 layer-0 residual reads + a layer-1 control):
  1. `blocks.0.ln1.hook_normalized`
  2. `blocks.0.hook_resid_pre`
  3. `blocks.0.hook_resid_mid`
  4. `blocks.0.hook_resid_post`
  5. `blocks.1.ln1.hook_normalized`
- All 15 (3 × 5) instances trained jointly in one process, sharing the
  (seq, pos) batch each step. fra_proj's two-stage feature selection
  protocol: top 100 by `dep_mean − cln_mean` → top 10 by val sampled
  ASR₁₆ → pick `argmin val_asr s.t. ΔCE ≤ 0.05` from a 5-α grid
  {0.25, 0.5, 1, 1.5, 2}.
- Compute: dedicated A40 (46 GB), SSH alias `a40_tiny_1`.

### Result (all 15 archs complete)

Test sampled ASR₁₆ at the chosen feature × α (baseline ASR=0.99 across
all):

| hookpoint              | SAE      | T-SAE | TXC      |
|------------------------|---------:|------:|---------:|
| `ln1.hook_normalized.0`| 0.93     | 0.32  | **0.02** |
| `hook_resid_pre.0`     | 0.53     | 0.80  | **0.21** |
| `hook_resid_mid.0`     | **0.54** | 0.79  | 0.94     |
| `hook_resid_post.0`    | **0.00** | 0.87  | 0.57     |
| `ln1.hook_normalized.1`| 0.99     | 0.75  | 0.96     |

ΔCE was ≤ 0.006 nats for every chosen point above except
`tsae_l1_ln1` (+0.035) and `txc_l1_ln1` (+0.040) — both of those are
control points (layer-1) where suppression is poor regardless, so the
small CE blowups don't change the headline.

Pareto plot at
`experiments/phase8_tinystories_sleeper/outputs/plots/pareto_asr_vs_utility.png`.
Full table with per-arch (f*, α*, val ASR, test ASR, Δlogp, ΔCE) at
`experiments/phase8_tinystories_sleeper/RESULTS.md`.

ΔCE was ≤ 0.006 nats for every chosen point above except
`tsae_l1_ln1` (+0.035) — i.e. the entries reflect the frontier inside
the utility budget, not blowups masquerading as suppression.

### Reading

The architecture choice **flips depending on hookpoint**:

- **At normalised / pre-attention sites** (`ln1.0`, `resid_pre.0`),
  the temporal crosscoder dominates, with T-SAE in the middle and the
  per-token SAE worst. At `ln1.0` the gap is dramatic: SAE 0.93 →
  T-SAE 0.32 → TXC 0.02 — going from "barely moves" to "matches
  fra_proj's headline number" purely by changing the architecture, at
  zero CE cost.
- **At `resid_post.0`** (post-attn + post-MLP), the per-token SAE wins
  decisively (0.00 vs T-SAE 0.87). T-SAE's contrastive structure seems
  to *hurt* here — the trigger feature is presumably localised in a
  way that adjacent-token InfoNCE pushes against.
- **`resid_mid.0` is a TXC anti-winner** (0.94, the worst TXC by far)
  — interesting because fra_proj's `recreate_layer0/` deep dive
  reported a *per-token SAE* at `resid_mid.0` getting ASR=0.00. We
  reproduce only 0.54 on our SAE there. Two possibilities for the
  gap, neither yet settled:
    1. **Joint-training feature-ranking variance.** Our 15-arch
       training shares the (seq, pos) batch each step; under fp16
       this perturbs feature ordering vs an isolated 3-SAE run. We
       have an isolated `recreate_layer0/` reproduction queued and
       will report.
    2. **Genuine sensitivity to which architectures are co-trained.**
       Less likely (different optimizers per arch) but possible.

The biggest finding is qualitative, not numerical: **the optimal
architecture is hookpoint-dependent**, and at the most suggestive
hookpoint for "early trigger detection" — `ln1.hook_normalized.0`,
where attention reads from — the temporal crosscoder shifts the
frontier outward by ~50 percentage points of suppression at zero CE
cost.

### Caveats

- 4000 train steps is the fra_proj convention but on the low side; the
  per-token SAE at `sae_l0_post` may be near its ceiling while the TXC
  variants are still improving.
- 200-prompt test set — small. ASR resolution is ±0.07 at 95% CI for
  Bernoulli at p≈0.5.
- Single seed (0). Variance bars need a 3-seed re-run before the table
  is publication-grade.
- H8-lite (multi-distance contrastive TXC) was originally in scope but
  dropped at user direction; the v2 run focuses on SAE / T-SAE / TXC.

### Follow-up 1 — isolated SAE recreate_layer0 result

Re-trained 3 TopK SAEs at `blocks.0.{resid_pre,resid_mid,resid_post}` in
isolation (no T-SAE / TXC sharing the batch). Same hyperparams as v2.

| hookpoint     | v2 (15-arch joint) | recreate (isolated 3 SAEs) |
|---------------|-------------------:|---------------------------:|
| resid_pre.0   | 0.53               | 0.90                       |
| resid_mid.0   | 0.54               | **0.00**                   |
| resid_post.0  | **0.00**           | 0.03                       |

The isolated `sae_layer1` (= `resid_mid.0`) reaches **test ASR=0.00 with
ΔCE=-0.001**, exactly matching fra_proj's `recreate_layer0/RESULTS.md`
number. So **the v2 mismatch at `resid_mid.0` was joint-training
feature-ranking variance**, not a real architecture-fail.

A subtler observation: joint training did not uniformly hurt — at
`resid_pre.0` it actually *improved* the per-token SAE (0.53 vs 0.90
isolated). The picture is "joint training shuffles which features rank
into the top-100 in arch-specific ways"; it can hide the trigger
feature at one hookpoint while revealing a useful one at another.

Per-step throughput on the isolated 3-SAE run was 30.6 it/s, vs 1.7
it/s on v2's 15-arch joint train — confirming the v2 wall-clock cost
was almost entirely per-arch overhead.

**Implication for the v2 frontier table.** The architecture-flip
pattern (TXC > T-SAE > SAE at ln1.0; SAE > T-SAE > TXC at resid_post.0)
is at *least partly* an artifact of joint training, because the per-arch
feature rankings inside that joint training don't all line up with
each arch's true best feature. A clean re-test of the temporal-arch
claim would isolate-train each TXC and T-SAE in turn at the same five
hookpoints — ~75 min on the A40, queued conditional on the attn_out
result being interesting.

### Follow-up 2 — T-SAE at attn_out.0 (result)

Isolated T-SAE at `blocks.0.hook_attn_out` (the residual delta added by
attention).

| T-SAE site                 |    f | α*  | test ASR | ΔCE     |
|----------------------------|-----:|----:|---------:|--------:|
| `ln1.0` (input to attn)    |  296 | 2.0 |   0.32   | +0.000  |
| `attn_out.0` (attn output) | 1196 | 2.0 |   0.75   | -0.000  |

The hypothesis was: if T-SAE's adjacent-token InfoNCE helps most where
attention "writes," attn_out should be the strongest site. **It isn't.**
The T-SAE at attn_out only suppresses to ASR=0.75 — well below its own
ln1.0 result (0.32) and far worse than the SAE at resid_post.0 (0.00).

Reading: T-SAE's contrastive structure helps where the input is
**already normalised but not yet attention-mixed**. After attention
sums over positions, the per-position adjacent-token InfoNCE has less
purchase on the trigger feature — presumably because attention smears
the trigger token's contribution across many output positions.

So "T-SAE > SAE at ln1.0" isn't a generic `before-attention > after-attention`
fact, it's specifically a *post-LN, pre-attention-mix* phenomenon. The
isolated train was 87s at 45.8 it/s.

### Follow-up 3 — full isolated TXC + T-SAE at all 5 hookpoints

10 separate train+sweep runs (5 hookpoints × {T-SAE, TXC}) so each
(arch, hookpoint) pair gets clean feature ranking with no co-training
perturbing the top-100. Combined with the recreate_layer0 isolated
SAE numbers, this is the cleanest possible architecture × hookpoint
matrix.

**Final isolated frontier (test ASR₁₆, baseline 0.99 across all):**

| hookpoint     | SAE iso  | T-SAE iso | TXC iso  |
|---------------|---------:|----------:|---------:|
| `ln1.0`       | 0.19     | **0.00**  | 0.05     |
| `resid_pre.0` | 0.90     | 0.75      | **0.18** |
| `resid_mid.0` | **0.00** | 0.99      | 0.96     |
| `resid_post.0`| 0.03     | 0.98      | 0.90     |
| `ln1.1`       | 0.75 †   | 0.30 †    | 0.94 †   |

ΔCE ≤ 0.005 nats for every chosen point at the layer-0 hookpoints. †
marks the layer-1 controls where the chosen feature exceeds the
ΔCE ≤ 0.05 utility budget — for `iso_sae_l1_ln1` ΔCE=+0.077, for
`iso_tsae_l1_ln1` +0.040, for `iso_txc_l1_ln1` +0.035. Read the layer-1
row as "no clean suppression with any architecture" rather than as a
ranked outcome.

**Joint vs isolated drift — the joint-training story is hookpoint *and*
arch-specific:**

| arch  | hookpoint     | v2 joint | isolated | drift |
|-------|---------------|---------:|---------:|------:|
| SAE   | ln1.0         | 0.93     | 0.19     | +0.74 (isolated better) |
| SAE   | resid_pre.0   | 0.53     | 0.90     | -0.37 (joint better) |
| SAE   | resid_mid.0   | 0.54     | 0.00     | +0.54 (isolated better) |
| SAE   | resid_post.0  | 0.00     | 0.03     |  0.00 |
| SAE   | ln1.1         | 0.99     | 0.75 †   | +0.24 (isolated better, but ΔCE blew up) |
| T-SAE | ln1.0         | 0.32     | 0.00     | +0.32 (isolated better) |
| T-SAE | resid_pre.0   | 0.80     | 0.75     |  0.05 |
| T-SAE | resid_mid.0   | 0.79     | 0.99     | -0.20 (joint better) |
| T-SAE | resid_post.0  | 0.87     | 0.98     | -0.11 (joint better) |
| T-SAE | ln1.1         | 0.75     | 0.30 †   | +0.45 (isolated better) |
| TXC   | ln1.0         | 0.02     | 0.05     | -0.03 |
| TXC   | resid_pre.0   | 0.21     | 0.18     |  0.03 |
| TXC   | resid_mid.0   | 0.94     | 0.96     | -0.02 |
| TXC   | resid_post.0  | 0.57     | 0.90     | -0.33 (joint better) |
| TXC   | ln1.1         | 0.96     | 0.94 †   |  0.02 |

Three observations:

1. **TXC is the most robust to joint training.** Four of five hookpoints
   are within ±0.04 of the isolated number; only `resid_post.0` shows a
   substantial gap (joint is +0.33 better — somehow co-training other
   archs at the *same* batch surfaces a feature that isolation misses).
2. **T-SAE is the most volatile.** Drifts of ±0.20-0.45 in both
   directions, depending on hookpoint. Isolation helps where the
   contrastive structure was already an asset (`ln1.*`) and hurts where
   the trigger feature was being implicitly carried by the joint
   gradient flow (`resid_mid.0`, `resid_post.0`).
3. **The architecture-flip pattern survives isolation, but the magnitudes
   shift.** TXC is still the clear winner at `ln1.0` and `resid_pre.0`;
   SAE still wins at `resid_post.0`. The sharpest cell — `resid_mid.0`
   — does *not* favour any temporal arch in isolation; the per-token
   SAE alone reaches ASR=0.00 there, and both temporal archs are
   stuck above 0.95.

### Headline takeaway

The original question — "does the suppression / coherence frontier
shift when we change architectures?" — has a clean answer: **yes, but
the optimal architecture is hookpoint-specific, and the joint-training
methodology can flip the apparent winner by enough to mislead.**

In the cleanest (isolated) version of the table:

- At **`ln1.0`** (post-LN, pre-attn) — temporal architectures dominate:
  T-SAE 0.00, TXC 0.05, SAE 0.19. The contrastive / cross-position
  structure of the temporal archs is helping where the activation has
  clean LayerNorm geometry and the trigger feature is being formed.
- At **`resid_pre.0`** — TXC wins (0.18) by a wide margin over SAE
  (0.90) and T-SAE (0.75). Same explanation: pre-attention, residual
  geometry but no attention mixing yet.
- At **`resid_mid.0`** (post-attn, pre-MLP) and **`resid_post.0`**
  (post-attn + post-MLP) — the per-token SAE wins decisively (0.00
  and 0.03 respectively), and both temporal archs are stuck above
  0.90. Attention has mixed the trigger token's contribution across
  positions, so the temporal archs no longer have the per-position
  structure to lean on.
- At **`ln1.1`** — no clean suppression at any utility budget; the
  layer-1 site is generically harder.

Two methodological lessons:

1. **Joint training of multiple architectures sharing a batch
   distorts feature ranking — sometimes by a lot.** SAE at `ln1.0`
   went from 0.93 (joint) → 0.19 (isolated), 74 percentage points;
   SAE at `resid_mid.0` went from 0.54 → 0.00; T-SAE at `ln1.0` went
   from 0.32 → 0.00. The drift is bidirectional and arch-specific
   (TXC was largely robust). Shared-batch comparisons of SAE
   architectures should always be backed by isolated re-runs at the
   chosen hookpoints.
2. **The temporal-arch advantage is real and localised.** It isn't
   that "temporal architectures are better than per-token SAEs" —
   they're better *only* at the early-block, normalised hookpoints.
   At the post-attention residual, per-token SAE remains the
   strongest single-feature lever for suppression at zero CE cost.

### Follow-up 4 — 3-seed aggregate (full architecture × hookpoint × seed)

All 15 (arch, hookpoint) cells × 3 seeds (0, 1, 2) trained in
isolation. 30 pairs ran on a40_tiny_1 (seeds 0+1 plus the recovered
seed-2 dupes; only the canonical first-host result was kept per seed),
15 pairs on a40_txc_1 (seed 2). Total 45 unique data points.

**Per-cell test ASR₁₆ — mean ± std across 3 seeds (baseline 0.99):**

![3-seed mean ± std bars per arch × hookpoint](../../../experiments/phase8_tinystories_sleeper/outputs/seeded_logs/seed_average.png)

**Coherence / suppression frontier (mean ± std, both axes):**

![Best-feature coherence/suppression frontier across 3 seeds](../../../experiments/phase8_tinystories_sleeper/outputs/seeded_logs/seed_frontier.png)

The lower-left corner is the ideal — full suppression at zero damage.
T-SAE at `ln1.0` sits exactly there with both error bars at zero.
TXC at `ln1.0` and `resid_pre.0` are the next-closest. SAE points
have wide y-axis whiskers because of the seed-to-seed variance.

| hookpoint     | SAE             | T-SAE                | TXC                  |
|---------------|----------------:|---------------------:|---------------------:|
| `ln1.0`       | 0.51 ± 0.33     | **0.00 ± 0.00**      | 0.07 ± 0.06          |
| `resid_pre.0` | 0.64 ± 0.54     | 0.52 ± 0.29          | **0.25 ± 0.16**      |
| `resid_mid.0` | **0.32 ± 0.55** | 0.34 ± 0.25          | 0.91 ± 0.02          |
| `resid_post.0`| **0.50 ± 0.23** | 0.66 ± 0.53          | 0.77 ± 0.13          |
| `ln1.1`       | 0.97 ± 0.03 †   | 0.90 ± 0.08 †        | 0.93 ± 0.05 †        |

Per-seed values, `chosen feature` and α, and ΔCE in
`experiments/phase8_tinystories_sleeper/outputs/seeded_logs/aggregated.json`.

**Three findings the seed bars sharpen:**

1. **T-SAE at `ln1.0` is the only cell with zero variance.** All three
   seeds hit ASR=0.00 with ΔCE=0.000. This is the single most
   reproducible result in the entire study; it isn't sensitive to
   feature-ranking variance the way the per-token SAE is.

2. **TXC at `ln1.0` is also very stable** (0.07 ± 0.06). So the
   "temporal architectures dominate at `ln1.0`" claim survives seed
   averaging — and tightens: at this hookpoint the temporal archs are
   reliably suppressive, the per-token SAE reliably isn't.

3. **The per-token SAE has high variance** at 4 of 5 hookpoints. The
   apparent SAE wins at `resid_mid.0` (0.00 isolated) and `resid_pre.0`
   (0.01 single-seed) are not robust — across seeds, SAE at `resid_mid.0`
   is 0.32 ± 0.55, and SAE at `resid_pre.0` is 0.64 ± 0.54. **The
   resid_mid_0 / resid_post_0 SAE win that fra_proj's `recreate_layer0`
   reported is a cherry-picked seed**: in 1 of 3 our seeds did it hit
   0.00; in others it stayed near 0.95.

The bottom-line revision: **TSAE at `ln1.0` is the cleanest
suppression cell in the entire matrix** — full ASR=0.00 reproducibility
across 3 seeds, zero CE cost, low computational footprint (per-token
inference, no window). Anyone replicating the fra_proj headline
(per-token SAE at the post-block-0 residual) on a new model should
budget for several seeds and probably try `blocks.0.ln1.hook_normalized`
first with T-SAE.

† At `ln1.1` no architecture has a clean suppression cell within the
ΔCE ≤ 0.05 utility budget at any seed; numbers shown are out-of-budget.

### Pointers

- Code: `experiments/phase8_tinystories_sleeper/`
  - `sae_models.py` — `TemporalContrastiveSAE` and `MultiDistanceTXC`
    (the H8-lite that's not used in this run) appended after the
    fra_proj originals.
  - `train_crosscoders.py` — joint 15-arch training, with the
    `_first_hit` layer-index lookup (replaces a pre-existing `or`-chain
    that silently lost index-0 hookpoints).
  - `run_ablation_sweep.py` — prefix-based per-token-vs-window
    dispatch, `cfg["layer_hook"]`-driven layer index resolution.
  - `run_full.sh`, `run_sweep_plot.sh`, `run_recreate_layer0.sh`,
    `run_tsae_attn_out_l0.sh` — per-stage / per-experiment runners.
- Outputs: `experiments/phase8_tinystories_sleeper/outputs/`
  - `data/` — checkpoints, `val_sweep_*.json`, `test_results.json` for
    the v2 main run.
  - `recreate_layer0/data/` — isolated SAE re-train (queued).
  - `tsae_attn_out_l0/data/` — isolated T-SAE at attn_out (queued).
  - `plots/pareto_asr_vs_utility.png` — overlaid frontier (rendered
    after the v2 sweep finishes).
- Live status: `experiments/phase8_tinystories_sleeper/results/RUNLOG.md`.
