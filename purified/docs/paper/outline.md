---
title: Temporal Crosscoders — Paper Outline (working)
author: agent_paper
date: 2026-05-03
status: skeleton
---

## Paper title (working)

Temporal Crosscoders: When Cross-Token Sparse Dictionaries Help

## One-sentence pitch

A unified study of two temporal-crosscoder architectures across
synthetic feature recovery, sparse probing, qualitative latent
interpretability, and three behavioural case studies — finding that
TXCs improve global feature recovery and qualitative interpretability,
match per-token SAEs on probing, and push the Pareto frontier on the
backtracking inducement metric, but lose on emergent misalignment.

## Sections

1. **Introduction** — what's missing in per-token SAEs, prior work
   (T-SAE, TFA, MLC), the contribution: a clean two-architecture
   comparison across seven evaluations.
2. **Method**
   - 2.1 Two TXC architectures (TXC-base, TXC-pro) — fixed across paper
   - 2.2 Baselines (TopK-SAE, T-SAE, TFA, MLC)
   - 2.3 The temp-bench evaluation suite (C1–C7)
3. **Synthetic results** — C1 + C2
4. **Real-LM results** — C3 (probing) + C4 (qualitative)
5. **Behavioural case studies** — C5 (steering) + C6 (EM) + C7 (backtracking)
6. **Discussion** — when TXCs help, when they don't, recommendations.
7. **Limitations** — single subject model (Gemma-2-2b), single layer (13)
   for most experiments, k_pos = 5 vs k_pos = 20 sensitivity, hill-climbing
   excluded by design.

## Headline figures (planned)

- **Fig 1**: TXC architecture diagrams (base + pro)
- **Fig 2**: Synthetic AUC vs k (C1) — TXC-pro best, dissociation TFA-style
- **Fig 3**: Coupled-feature gAUC vs k vs T (C2) — global/local tradeoff
- **Fig 4**: Probing AUC leaderboard (C3) — bar chart with σ_seeds
- **Fig 5**: Qualitative Pareto (C4) — TXC-pro upper-right of T-SAE
- **Fig 6**: Steering Pareto (C5) — coh-vs-success curves
- **Fig 7**: EM single-feat scatter (C6) — TXC below SAE diagonal
- **Fig 8**: Backtracking Δgc bar (C7) — TXC ~3× next best

## Story arc (revised after C6 reframe)

The paper does NOT claim TXC dominates everywhere. It claims:

> *Cross-token structure is a useful inductive bias for some tasks but
> not all. TXCs win on global feature recovery, qualitative
> interpretability, and behavioural inducement; they tie on standard
> probing and lose on per-token-causal interventions like emergent
> misalignment. The pattern is interpretable: TXCs help when the target
> is delocalised across positions, and hurt when it isn't.*

This is a more interesting paper than "TXCs always win," and it's what
the data actually supports.
