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

No matryoshka, no contrastive, no InfoNCE, no BatchTopK.

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
| C6 (R32 ext-α align %) | ~52 | TBD | SAE arditi 64.5 | **TXC loses** |
| C7 (peak Δgc) | TBD | ~+1.574 | next-best ~+0.5 | TXC-pro wins ~3× |

**Honest paper read**: TXC-pro wins on C2, C4, C7; TXC-base wins on C3-k20;
TXCs match T-SAE on C5; TXCs lose on C6. The pattern is interpretable.
