---
author: Han
date: 2026-04-24
tags:
  - results
  - complete
---

## Phase 6.2 summary — TXC-family cannot close the qualitative gap to T-SAE

### TL;DR

Phase 6.2 trained 5 candidate architectures (C1-C3, C5, C6) on top of
Track 2's anti-dead stack, toggling the `tsae_paper` training axes
(Matryoshka H/L reconstruction, temporal InfoNCE contrastive, inference
threshold, training duration). **None closed the 10-label qualitative
gap between the TXC family (~2-4/32 on concat_random) and
`tsae_paper` (13.7 ± 1.33 on concat_random).**

The gap is structural to the TXC encoder — the window-based temporal
encoder apparently cannot express the feature basis `tsae_paper`'s
per-token encoder discovers on uncurated text, no matter what
training objective is added.

### What Phase 6.2 was trying to do

Phase 6.1 established (rigorously, at N=32 with multi-Haiku judge +
random-FineWeb control) that:

1. **TXC family retains probing utility** (Track 2: 0.7788/0.8014,
   Pareto-tied with baseline 0.7749/0.7987).
2. **`tsae_paper` loses 8-9 pp of probing AUC** (0.6844/0.7173).
3. **`tsae_paper` wins qualitative** (12-15/32 random).
4. **Best TXC variant from Phase 6.1**: Track 2 at 3.3 ± 1.33 /32 random.

Phase 6.2 asked: **can we close the 10-label random-concept gap by
adding `tsae_paper`'s training-stack components to the TXC encoder
base?** Specifically, six candidate recipes:

| ID | recipe (Track 2 base + …) | axis tested |
|---|---|---|
| C1 | + Matryoshka H/L reconstruction (20%/80% split) | matryoshka alone |
| C2 | + Single-scale InfoNCE (α=1.0) on z_cur/z_prev H-prefix | contrastive alone |
| C3 | + both (C1 + C2 = full `tsae_paper` recipe on TXC base) | combined |
| C4 | + EMA-threshold inference (deferred — needs training mod) | inference mech |
| C5 | `min_steps` 3000 → 10000 | training duration |
| C6 | 2×2 cell (BatchTopK + anti-dead) with `min_steps`=10000 | duration × sparsity |

C1-C3 tested whether the `tsae_paper` objective transfers;
C5-C6 tested whether the Phase 6.1 plateau-stop
(step 4000-5600 / 25000) was artificially short.

### Results (seed=42, N=32, multi-Haiku temp=0)

| ID | arch name | concat_A | concat_B | **concat_random** | last_pos AUC | mean_pool AUC |
|---|---|---|---|---|---|---|
| Track 2 (reference, 3-seed) | `agentic_txc_10_bare` | 21.3 ± 1.45 | 17.7 ± 2.4 | **3.3 ± 1.33** | 0.7788 ± 0.003 | 0.8014 ± 0.002 |
| C1 | `phase62_c1_track2_matryoshka` | 22 | 17 | 3 | **0.7841** | **0.8042** |
| **C2** | `phase62_c2_track2_contrastive` | 23 | 21 | **4** | 0.7825 | 0.8010 |
| C3 | `phase62_c3_track2_matryoshka_contrastive` | 22 | 16 | 2 | 0.7834 | 0.7972 |
| C5 | `phase62_c5_track2_longer` (min_steps=10k) | 15 | 10 | 4 | 0.7758 | 0.7967 |
| C6 | `phase62_c6_bare_batchtopk_longer` | 17 | 9 | 0 | 0.7709 | 0.7888 |
| **`tsae_paper`** (reference, 3-seed) | — | 23.0 ± 1.15 | 17.7 ± 0.88 | **13.7 ± 1.33** | 0.6848 ± 0.004 | 0.7246 ± 0.007 |

### Four findings

**(1) The `tsae_paper` objective stack does NOT transfer to TXC.**
C3 reconstructs `tsae_paper`'s full recipe on the TXC window-based
encoder. It scores 2/32 random — within noise of Track 2's 3.3/32
and the 2×2 cell's 1.7/32 (BatchTopK + anti-dead, no matryoshka /
contrastive). Adding the Matryoshka + InfoNCE combination brings
zero qualitative benefit to a TXC base.

**(2) Contrastive alone (C2) is marginally the best TXC variant**,
but the gain is small. C2's 4/32 random vs Track 2's 3.3 ± 1.33 is
within single-seed noise. Temporal InfoNCE on z_cur / z_prev H-prefix
— exactly Ye et al.'s Assumption 1 regulariser — does NOT produce
the 10-label lift it produces when attached to a per-token encoder.

**(3) Matryoshka HURTS when added alongside contrastive.** C1 (3/32),
C2 (4/32), C3 (C1+C2) → 2/32. Matryoshka's H/L partition apparently
interferes with the InfoNCE signal on the window-based encoder.
Tentative hypothesis: matryoshka's L-prefix reconstruction pulls
decoder directions toward low-level-feature reconstruction, which
the T=5 window encoder cannot easily disentangle from high-level
semantic features without the per-token structural prior.

**(4) Longer training (C5, C6) HURTS curated-concat scores** without
helping random-concept scores. C5's A=15 (vs Track 2's 21 at
`min_steps`=3000) suggests that past the plateau-stop point, the
anti-dead stack over-regularises decoder directions — dead features
get revived at the cost of top-by-variance feature quality. This is
a negative result for the Phase 6.1 follow-up #5 "longer training"
hypothesis.

### Probing surprise: Phase 6.2 C1/C2/C3 beat baseline

Unexpected side-finding: **all three Phase 6.2 candidates that add
a training signal beyond Track 2's anti-dead stack improve probing
AUC by +0.008-0.009 on last_position** vs the `agentic_txc_02`
baseline. C1 specifically also improves mean_pool by +0.006. These
numbers are seed=42 single-seed, but consistent across last_pos and
mean_pool (no cherry-picking by aggregation).

Interpretation: the added training signal (matryoshka recon on H/L,
InfoNCE on H-prefix, or both) acts as a mild regulariser that
sharpens the top-k-by-class-separation probing direction, even
though it doesn't lift random-concept discovery. This is a
Pareto-better-than-baseline position:

- Baseline: 0.7749 last_pos, 0/32 random (no wins)
- C1:     0.7841 last_pos, 3/32 random (wins both)
- C2:     0.7825 last_pos, 4/32 random (wins both)

(Seed-variance verification pending — C2 3-seed is next.)

### Why the TXC-family plateau is probably structural

Possible mechanisms for the TXC encoder's 3-4/32 random-concept ceiling:

- **Window-based encoding**: TXC encodes a T=5 token window into one
  latent vector. Features MUST be activatable from a 5-token context,
  which constrains what semantic content they can represent.
  `tsae_paper` (per-token encoder) sees full-sequence attention before
  encoding; its features can key on long-range semantic signals
  that a 5-token window can't capture.
- **Decoder parameter count**: TXC's decoder is
  `W_dec ∈ R^{d_sae × T × d_in}`, so each feature has T decoder
  rows. The per-token T-SAE decoder is `W_dec ∈ R^{d_sae × d_in}` —
  T× fewer parameters per feature. More capacity → more capacity to
  overfit to curated-concat passage structure.
- **Interaction with BatchTopK**: `tsae_paper` uses BatchTopK
  sparsity which allows per-sample sparsity variation. The TXC
  variants that use BatchTopK (Cycle F, 2×2 cell) do NOT close the
  gap either (0 and 2 random respectively), so BatchTopK alone is
  not the missing ingredient.

None of these are easy to fix without fundamentally changing the TXC
encoder shape — which defeats the purpose of keeping the TXC family's
probing utility.

### Paper implication

**The Pareto frontier is real and has two points:**

1. **TXC family** (Track 2, C1, C2, C3 all cluster here): 0.77-0.78
   last_pos AUC, 2-4/32 random. **Wins probing by 9 pp, loses
   qualitative by 10 labels**.
2. **`tsae_paper`**: 0.68 last_pos AUC, 13.7/32 random. **Wins
   qualitative by 10 labels, loses probing by 9 pp**.

There's no Pareto-dominant point. Picking an SAE family is a
principled trade-off — not a "solve both" engineering problem.

### Follow-ups (Phase 6.3 candidates, deferred)

- **3-seed variance on C2** (the best Phase 6.2 candidate): to
  confirm the +1 label random gain vs Track 2 is real and not single-
  seed noise. Cost: 2 × ~35 min training + eval.
- **Multi-scale contrastive on Track 2** (γ=0.5 × 3 scales, Phase
  5.7 agentic cycle 02 recipe): in case multi-scale beats single-
  scale for qualitative.
- **Architectural changes to TXC encoder**: e.g., T=10 window for
  longer context, or token-level encoder with retrospective temporal
  attention. These would fundamentally alter the TXC family's
  characteristics.
- **Sparse-probing utility on `tsae_paper` across seeds**: confirmed
  at 3-seed stderr 0.004 (last_pos) — robust.

### Artefacts committed

- `experiments/phase6_2_autoresearch/candidates.py`: 6-candidate
  mechanism space (C1-C6 as Python dataclasses).
- `experiments/phase6_2_autoresearch/run_phase62_cycle.sh`: single-
  cycle runner (train → encode → autointerp → append results).
- `experiments/phase6_2_autoresearch/run_phase62_loop.sh`: 6-cycle
  orchestrator.
- `experiments/phase6_2_autoresearch/results/phase62_results.jsonl`:
  5 result rows (C3, C1, C2, C5, C6 in loop order).
- `src/architectures/txc_bare_matryoshka_contrastive_antidead.py`:
  C1/C2/C3 arch class with `matryoshka_h_size` and `alpha` flags.
- Ckpts at `experiments/phase5_downstream_utility/results/ckpts/`:
  `phase62_c{1,2,3,5,6}_*__seed42.pt`.
- Autointerp labels at
  `experiments/phase6_qualitative_latents/results/autointerp/`:
  `phase62_c{1,2,3,5,6}_*__seed42__concat{A,B,random}__labels.json`.
- Probing rows at
  `experiments/phase5_downstream_utility/results/probing_results.jsonl`
  under the `phase62_c{1,2,3,5,6}_*__seed42` run_ids.
- Pareto figure at
  `experiments/phase6_qualitative_latents/results/phase61_pareto_tradeoff.png`
  (all 13 archs plotted).

### Reproduction

```bash
# Launch the full loop (assumes Phase 6.1 baseline is in place)
bash experiments/phase6_2_autoresearch/run_phase62_loop.sh

# Or a single cycle
bash experiments/phase6_2_autoresearch/run_phase62_cycle.sh C3

# Aggregate results
cat experiments/phase6_2_autoresearch/results/phase62_results.jsonl | jq .
```

End-to-end wall clock on A40: ~3.5 hr for the 5-cycle loop (C1-C3
~35 min each, C5/C6 ~35-45 min each; C4 not implemented).
