---
title: The two locked TXC architectures
author: agent_paper
date: 2026-05-03
status: draft
---

## Why exactly two

A central methodological claim: **the same two TXC architectures are
used across all seven components, with only sparsity (k) and dictionary
size (d_sae) tuned per component**. No per-component hill-climbing.

This precludes the "different gigabrain TXC for every case study"
critique that was about to land on the unaligned hill-climb results
(see `docs/han/research_logs/phase7_unification/agent_x_paper/2026-05-02-yw-T8-benchmark.md`,
which proves the steering-best architectures lose at probing).

## TXC-base = `txc_bare_antidead_t5`

Vanilla TopK temporal crosscoder + tsae_paper anti-dead stack.

**Architecture:**
- Per-position encoder: $W_{\text{enc}}[t]$ for $t \in [0, T)$, shape
  $(T, d_{\text{in}}, d_{\text{sae}})$. Sums into a single
  $(B, d_{\text{sae}})$ pre-activation per window.
- Sparsity: TopK with $k_{\text{win}} = k_{\text{pos}} \cdot T$ over the
  flat $d_{\text{sae}}$ axis, per-sample. Window sees T=5 tokens.
- Decoder: single $W_{\text{dec}}$ of shape $(d_{\text{sae}}, T, d_{\text{in}})$
  with bias $b_{\text{dec}}$ of shape $(T, d_{\text{in}})$. Reconstructs
  the full T-window.

**Anti-dead stack** (copied from `tsae_paper.py`):
1. `num_tokens_since_fired` buffer; dead threshold = 10M tokens.
2. AuxK loss: top-k = `aux_k` dead features re-reconstruct the residual;
   $\alpha_{\text{auxk}} = 1/32$.
3. Unit-norm decoder constraint (per-latent over $(T, d_{\text{in}})$).
4. Decoder-parallel gradient removal on $W_{\text{dec}}$.
5. Geometric-median $b_{\text{dec}}$ initialisation on the first batch.

No matryoshka, no contrastive, no InfoNCE, no BatchTopK. **No Bricken
resample** — components that want hard dead-feature reset opt in
themselves (see *Per-experiment training knobs* below).

**Free knobs**: $k_{\text{pos}}$ (sparsity), $d_{\text{sae}}$.

## TXC-pro = `phase5b_subseq_h8`

Adds three orthogonal design ideas to TXC-base:

**(1) Subseq encoder**: train with $T_{\text{max}} = 10$ position slabs
but sample $t_{\text{sample}} = 5$ positions per training step. At
probe/eval time use all 10 positions. Effectively trains a
position-permutation-invariant encoder while keeping per-position decoders.

**(2) Matryoshka H8**: 8 nested feature groups. Each group is trained to
reconstruct independently — a group-G reconstruction uses only the first
$G \cdot d_{\text{sae}} / 8$ features. Encourages a coarse-to-fine feature
hierarchy.

**(3) Multi-distance InfoNCE**: contrastive loss with shifts
$\Delta \in \{1, 2\}$, inverse-distance weighted. Pulls the latent at
position $t$ toward latents at positions $t \pm \Delta$.

**Free knobs**: $k_{\text{pos}}$ (default 20), $d_{\text{sae}}$.

**No Bricken resample** in the locked spec (see *Per-experiment training
knobs* below).

## Per-experiment training knobs (NOT part of the locked spec)

The locked TXC-base / TXC-pro definitions above are the only
architectural commitment. A component may opt into additional
**training-time augmentations** if it documents the choice in
`docs/components/cN.md` and the experiment justifies it.

The most relevant one is:

### Bricken resample (opt-in)

Bricken et al. 2023 dead-feature reset, ported from
`origin/em-nanda:experiments/em_features/dead_feature_resample.py`.
Periodically hard-resets features that haven't fired on a held-out
check batch. Implementation lives in
`src/temp_bench/training/bricken.py` — opt in by passing
`BrickenConfig(...)` to the trainer, default off.

**Why it isn't in the locked spec.** Six knobs co-tune in Dmitry's
winning Qwen-7B medical recipe (`brickenauxk_a8`):

| Knob | tsae_paper default | brickenauxk_a8 |
|---|---|---|
| `resample_every` | – (no resample) | 500 |
| `min_fires` | – | 1 |
| `n_check` | – | 2048 |
| `max_resample_fraction` | – | 0.5 |
| EMA-AuxK $\alpha$ | 1/32 | **1/8** |
| Dead-threshold (tokens since fired) | 10M | **128k** |

The recipe is coherent only when all six move together. We don't have
evidence that the recipe transfers to (a) Gemma-2-2b activations,
(b) TXC-pro's matryoshka × InfoNCE objective, or (c) toy data at
$d_{\text{sae}} = 40$ where dead pressure is essentially zero.

**Where each component stands:**

| C | Bricken on/off | Reasoning |
|---|---|---|
| C1 | **off** (no A/B needed) | $d_{\text{sae}} = 40$, ~ no dead pressure. n_check=2048 saturates fire counts. |
| C2 | **off** (no A/B needed) | Same. |
| C3 | **A/B first** | TXC-base ± Bricken at 5k steps × 1 seed × 16-task subset (~1 H100-hour). Adopt iff $\Delta$AUC > $\sigma_{\text{seeds}}$. |
| C4 | **piggybacks on C3** | Shares the cache. |
| C5 | **A/B first** | Same protocol as C3 on a 1k-prompt steering subset. |
| C6 | **on** (Dmitry's data) | brickenauxk_a8 recipe; component justifies it. |
| C7 | **A/B first** | Same protocol as C3 on backtracking inducement. |

The verdict from each A/B is recorded in the component writeup so the
paper can transparently report "TXC-base on C3 used Bricken
(verdict +0.x AUC)" or "did not use Bricken (verdict −σ)".

### Mixed precision

bf16 on H100/H200 (numerical-stable for crosscoder gradients), fp16 on
A40 with grad-scaler. Decided per-pod by `temp_bench.training`. This is
a hardware accommodation, not an architectural choice.

## Headline numbers (from the wasteland leaderboards)

These are the numbers we're inheriting; they need to be re-confirmed on
the locked architectures in `final`.

| Component | TXC-base | TXC-pro | Best baseline | Note |
|---|---|---|---|---|
| C1 (NMSE @ k=2) | TBD | TBD | TFA 0.093 | TXC-pro AUC at low k expected best |
| C2 (gAUC @ k=2) | TBD | 0.97 (T=5) | Stacked 0.76 | T modulates global/local |
| C3 (probing AUC k=20, S=32) | **0.913** | 0.906 | TopK-SAE 0.909 | TXC-base wins |
| C3 (probing AUC k=5, S=32) | 0.868 | 0.867 | MLC 0.871 | tied |
| C4 (passage probe) | TBD | ≥ 0.78 mean-pool | T-SAE 0.72 | TXC-pro wins |
| C5 (peak15) | TBD | TBD | T-SAE 1.10 | TXC-pro expected to match |
| C6 R1 30k mid-α | TBD (was 91.25 plain) | TBD | SAE arditi 95.16 | **pending re-test with Bricken** |
| C6 R32 ext-α 10k | TBD (was 51.95 plain) | TBD | SAE arditi 64.53 | **pending re-test with Bricken** |
| C7 (peak Δgc) | TBD | ~+1.574 | next-best ~+0.5 | TXC-pro wins ~3× |

**Honest paper read** (subject to C6 re-test): TXC-pro wins on C2, C4, C7;
TXC-base wins on C3-k20; TXCs match T-SAE on C5; C6 is pending a re-run
that opts into Bricken resample (Dmitry's earlier "TXC k=100" evidence
used neither anti-dead nor Bricken so the gap there is uninformative).
The pattern across the rest is interpretable.
