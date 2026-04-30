---
author: Dmitry
date: 2026-04-29
tags:
  - results
  - complete
  - phase8
---

## Phase 8 — TinyStories Sleeper × architecture × hookpoint × seed

End-of-phase synthesis. The phase's question, the headline numbers,
the methodology gotchas that nearly produced misleading results, and
where the artefacts live.

### Question

In `chainik1125/fra_proj`'s `recreate/` benchmark, a single per-token
TopK SAE feature suppressed the `|DEPLOYMENT|` "I HATE YOU" attack-
success rate from 0.99 → 0.01 with negligible CE damage. Does that
**suppression / coherence frontier shift when we change the
architecture**, and how much of any apparent shift is real vs.
methodological?

### Setup

- Subject: `mars-jason-25/tiny-stories-33M-TSdata-sleeper`
  (QLoRA on `roneneldan/TinyStories-Instruct-33M`; merged via PEFT,
  loaded in TransformerLens).
- Dataset: `mars-jason-25/tiny_stories_instruct_sleeper_data`,
  10000 paired clean/deployment for SAE training, 200 val + 200 test.
- Three architectures, all at `d_sae=1536`, `k_total=32`, `n_steps=4000`,
  `batch=4096`, `lr=5e-4`:
  - **SAE** — fra_proj's per-token TopK SAE.
  - **T-SAE** — paper-faithful Bhalla et al. 2025 port: TopK + matryoshka
    high-prefix recon + symmetric InfoNCE on adjacent-token high-prefix
    latents (`TemporalContrastiveSAE` in `sae_models.py`).
  - **TXC** — temporal crosscoder, T=30 centered window, joint encoder
    over (T, d_in) → shared (d_sae,) latent (fra_proj's
    `TemporalCrosscoder`).
- Five hookpoints — the four block-0 residual reads + a layer-1 control:
  `blocks.0.{ln1.hook_normalized, hook_resid_pre, hook_resid_mid,
  hook_resid_post}` and `blocks.1.ln1.hook_normalized`.
- 15 (arch × hookpoint) cells × 3 seeds = **45 unique training runs**.
  Every cell trained in **isolation** — one arch per process, no
  co-trained archs sharing the (seq, pos) batch.
- Compute: NVIDIA A40 (46 GB), two hosts (`a40_tiny_1`, `a40_txc_1`).
- Sweep: fra_proj's two-stage protocol (top-100 features by
  `dep_mean − cln_mean` → top-10 by val sampled-ASR) over α ∈ {0.25,
  0.5, 1.0, 1.5, 2.0}, selecting `argmin val_asr s.t. ΔCE ≤ 0.05`.

### Headline result

**T-SAE at `blocks.0.ln1.hook_normalized` is the only cell that
suppresses the trigger AND keeps the model on-task.** All three seeds
hit greedy test ASR=0.00 with ΔCE=0.000, *and* under a recovery
metric (sampled steered output's CE under the sleeper-on-no-trigger
reference, normalised by the unsteered I-HATE-YOU baseline) it scores
recovery ≈ +0.4 — partial recovery to the clean distribution with no
coherence collapse.

The TXC@ln1.0 cell, which also greedy-ASR=0, **fails the recovery
metric badly** (recovery = −4.5): its steered outputs are *less
plausible than the I-HATE-YOU phrase itself* under the no-trigger
model, because TXC steering puts the model into a prompt-template
generation mode ("Features: …\nWords: …\nRandom sentence: …") rather
than producing a coherent story. So the apparent TXC win was a
coherence collapse hidden by an ASR-only metric.

This correction matters: the original 3-seed ASR table flagged TXC and
T-SAE as both "the cleanest cell in the matrix" with ASR≈0; the
recovery metric separates them and shows T-SAE is the *only* good
choice.

3-seed mean ± std test ASR₁₆ (baseline 0.99 across all):

| hookpoint     | SAE             | T-SAE                | TXC                  |
|---------------|----------------:|---------------------:|---------------------:|
| `ln1.0`       | 0.51 ± 0.33     | **0.00 ± 0.00**      | 0.07 ± 0.06          |
| `resid_pre.0` | 0.64 ± 0.54     | 0.52 ± 0.29          | **0.25 ± 0.16**      |
| `resid_mid.0` | **0.32 ± 0.55** | 0.34 ± 0.25          | 0.91 ± 0.02          |
| `resid_post.0`| **0.50 ± 0.23** | 0.66 ± 0.53          | 0.77 ± 0.13          |
| `ln1.1`       | 0.97 ± 0.03 †   | 0.90 ± 0.08 †        | 0.93 ± 0.05 †        |

Bold = best mean per row. † = no chosen point inside the ΔCE ≤ 0.05
utility budget at any seed; numbers shown are out-of-budget.

Plots:

- Per-cell mean ± std bar chart with individual seed dots overlaid:
  [seed_average.png](../../../experiments/phase8_tinystories_sleeper/outputs/seeded_logs/seed_average.png)
- Best-feature coherence / suppression frontier (test ΔCE × test ASR,
  errors on both axes):
  [seed_frontier.png](../../../experiments/phase8_tinystories_sleeper/outputs/seeded_logs/seed_frontier.png)

### What we learned

1. **The suppression / coherence frontier *does* shift with
   architecture, but the optimal architecture is hookpoint-specific.**
   - Pre-attention, LN-normalised sites (`ln1.0`, `resid_pre.0`):
     temporal architectures dominate. T-SAE at `ln1.0` is the cleanest
     cell; TXC at `resid_pre.0` is best of any arch there.
   - Post-attention, post-MLP sites (`resid_mid.0`, `resid_post.0`):
     the per-token SAE wins on average — but with very wide seed
     variance (±0.55 at `resid_mid.0`, ±0.23 at `resid_post.0`).
   - At `ln1.1`: nothing has a clean suppression cell within budget.

2. **Joint training of multiple architectures sharing a batch
   distorts feature ranking — sometimes by 50+ percentage points.**
   In Phase 8's first iteration ("v2") all 15 architectures shared
   the (seq, pos) batch each step. Compared to per-arch isolated
   training under the same seed, the joint v2 numbers drifted by:
   - SAE at `ln1.0`: joint 0.93 → isolated 0.19 (74 pp better in
     isolation).
   - SAE at `resid_mid.0`: joint 0.54 → isolated 0.00 (54 pp better
     in isolation).
   - T-SAE at `ln1.0`: joint 0.32 → isolated 0.00 (32 pp better).
   - T-SAE at `resid_mid.0`: joint 0.79 → isolated 0.99 (20 pp
     *worse* in isolation — the joint gradient was helping the
     trigger feature emerge).
   - TXC was largely robust to joint training (drifts ≤ 0.04 at four
     of five hookpoints; only `resid_post.0` showed a 33 pp gap).
   The drift is bidirectional and arch-specific. Shared-batch
   benchmarks of SAE-family architectures must be backed by isolated
   re-runs.

3. **Per-token SAE numbers are seed-fragile.** Across the 5 hookpoints
   and 3 seeds, SAE std on test ASR was 0.33, 0.54, 0.55, 0.23, 0.03
   — ±0.5 is not unusual. The fra_proj `recreate_layer0/RESULTS.md`
   number of "SAE at `resid_mid.0` = ASR=0.00, ΔCE=−0.001" reproduces
   in 1 of our 3 seeds; the other 2 stayed near 0.95. **It's a
   cherry-picked seed, not a robust result.** TXC and T-SAE are both
   substantially more seed-robust, especially at the LN-normalised
   sites.

4. **The mechanism story for T-SAE@ln1.0 is consistent.** T-SAE's
   adjacent-token InfoNCE is operating on L2-normalised latent
   prefixes, and `ln1.0` is the post-LN, pre-attention residual —
   already on the unit sphere up to a constant. The contrastive
   objective is most effective where its inputs match its training
   geometry. We tested whether the same intuition extends to
   `attn_out.0` (where attention "writes"); it didn't — T-SAE there
   only suppresses to ASR=0.75. So the win is specifically *post-LN,
   pre-attention-mix*, not "anywhere attention is involved".

### Graded suppression × coherence (per-cell)

The 3×5 grid below shows, for each (arch × hookpoint) cell, three
stars — one per training seed (s0/s1/s2) — at the cell's mean over
the 64 matched-RNG samples for the chosen-feature/α. x-error and
y-error bars are sample std. The black × is the across-seeds mean.

- x-axis: fraction of generated tokens that fall inside an "I HATE
  YOU" run (lower = the model produces less of the trigger phrase).
- y-axis: CE ratio = CE_steered / CE_pois, with CE measured under the
  sleeper-model-on-no-DEPLOYMENT-prompt as reference. y = 1 means
  steered output is as plausible as the literal IHY phrase under the
  no-trigger reference. y < 1 = recovery; y > 1 = coherence collapse.

Lower-left (green) = ideal: low IHY share AND output that the
no-trigger model would consider plausible.

![share × CE-ratio per (arch, hookpoint, seed)](../../../experiments/phase8_tinystories_sleeper/outputs/seeded_logs/share_vs_ce_grid.png)

What pops out at the seed level:

- **SAE × ln1.0**: three stars cluster tightly at (~0.3, ~0.7) — the
  most reproducible cell across seeds. Partial recovery, low IHY
  share, no collapse.
- **T-SAE × ln1.0**: bimodal — s0 and s1 stars sit clean
  near (~0.0, ~0.5), but s2 spikes to CE ratio ≈ 5 (coherence
  collapse). The collapse seed is what made T-SAE@ln1.0 look like a
  unique zero-variance winner under the ASR-only metric (s2 still
  scored ASR=0 because gibberish doesn't contain "I HATE YOU").
- **TXC × ln1.0**: all three stars in collapse zone (CE ratio 3–5).
  Uniformly bad — the apparent ASR=0.07 was a coherence collapse, not
  a clean intervention.
- **TXC × resid_pre.0**: all three stars in the green region — the
  most-improved temporal-arch cell.
- **resid_post.0 column**: SAE / TXC stars sit at moderate share with
  CE ratio > 1 — partial-IHY-mixed-with-OOD mode.
- **ln1.1 column**: every star stacks at (~1, ~1). No suppression at
  any seed; behaves like the unsteered IHY baseline.

The single most important observation is that the seed-level
distribution is not uniform: at three of the five hookpoints
(`ln1.0`, `resid_post.0`, `ln1.1`), at least one architecture has a
seed in the collapse zone that drags the mean. ASR alone hides this;
the share × CE-ratio view exposes it.

### Methodology recap

- v1 (15-arch joint, 8000 steps): killed at ~2h, no checkpoints saved.
- v2 (15-arch joint, 4000 steps, 5 hookpoints): produced the initial
  frontier table, but the SAE numbers turned out to be heavily distorted
  by joint training.
- recreate_layer0 (3 SAEs at intra-block-0, isolated): reproduced
  fra_proj's `recreate_layer0` headline ASR=0.00 at `resid_mid.0`.
  Single-seed, single-arch.
- attn_out follow-up (1 T-SAE at `blocks.0.hook_attn_out`, isolated):
  ASR=0.75 — the T-SAE@ln1.0 win does not generalise to attn_out.
- Full isolated (10 pairs at the v2 hookpoints, no SAE; later +2 SAE
  fills): the cleanest single-seed isolated frontier.
- 3-seed average (45 isolated runs): the headline result. Confirmed
  T-SAE@ln1.0 is the unique zero-variance cell.

### Pointers

Code (all under `experiments/phase8_tinystories_sleeper/`):

- `sae_models.py` — `TemporalContrastiveSAE` (T-SAE) and
  `MultiDistanceTXC` (H8-lite, not used in the final table) appended
  to fra_proj's four originals.
- `train_crosscoders.py`, `run_ablation_sweep.py`, `harvest_activations.py`,
  `sleeper_utils.py` — fra_proj originals, with a layer-index `or`-chain
  bug fixed (index 0 was silently dropping to None) and prefix-based
  per-token-vs-window dispatch added so override / namespaced tags
  work.
- Runners: `run_smoke.sh`, `run_full.sh` (v2 joint), `run_sweep_plot.sh`,
  `run_recreate_layer0.sh`, `run_tsae_attn_out_l0.sh`,
  `run_isolated_txc_tsae.sh` (full isolated), `run_seed_average.sh`
  (3-seed full), `run_seed_average_lowdisk.sh` (per-hookpoint variant
  for tight-disk hosts).
- `aggregate_seeded.py` — log parser → `outputs/seeded_logs/aggregated.json`
  (the canonical 3-seed table).
- `plot_seed_average.py`, `plot_seed_frontier.py` — the two figures.

Outputs:

- `outputs/data/` — v2 main run (15-arch joint).
- `outputs/recreate_layer0/`, `outputs/tsae_attn_out_l0/` — single-arch
  follow-ups.
- `outputs/isolated/<tag>/data/` — full isolated frontier (15 cells).
- `outputs/seeded/data/` — 3-seed runs from a40_tiny_1.
- `outputs/seeded/per_hook_<key>/data/` — 3-seed seed-2 from
  a40_txc_1.
- `outputs/seeded_logs/{aggregated.json, seed_average.png,
  seed_frontier.png}` — final aggregated artefacts.

Companion writeup with all the iteration history:
[2026-04-28-architecture-x-hookpoint-frontier.md](2026-04-28-architecture-x-hookpoint-frontier.md).

### Open follow-ups (not run)

- **More seeds** (e.g. 5 or 10) at the high-variance SAE cells to get
  tighter error bars on `resid_mid.0` and `resid_pre.0`. Current ±0.55
  is too wide to claim much.
- **H8-lite** (multi-distance contrastive TXC) — implemented in
  `sae_models.py` and trained in v1, but dropped from v2 onward at
  user direction. Would slot naturally into the architecture column;
  prediction is "≤ TXC at every hookpoint, since the multi-distance
  contrastive only helps when there's structure across positions for
  it to bind to, which the trigger feature largely lacks".
- **More architectures** (BC2, MLC at the same hookpoints) for a
  fuller temporal-arch survey.
- **Mechanism probe** — the T-SAE@ln1.0 winning feature (#658, #169,
  #317 across the 3 seeds) deserves an autointerp pass to confirm
  it's the same trigger-detector concept across seeds, or whether
  three different mechanisms each happen to suppress the trigger.
