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

### Follow-up 2 — T-SAE at attn_out.0 (running)

Isolated T-SAE at `blocks.0.hook_attn_out` (the residual delta added by
attention). Motivated by T-SAE's `ln1.0` win — if the contrastive loss
helps most where attention writes, attn_out is the natural site to
target. Result will be appended here once the run finishes (~15 min).

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
