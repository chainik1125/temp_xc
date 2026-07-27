---
author: Dmitry
date: 2026-07-27
tags:
  - results
  - in-progress
---

## Sprint log — Stacked SAE at paper protocol

Wall clock: T0 = 2026-07-27 05:11 UTC. Hard stop 15:15 UTC. Writing from 14:15.

## 05:11–05:25 — setup

- Worktree `.claude/worktrees/stacked-rebuttal` on branch `dmitry-stacked-arxiv`
  @ origin/arxiv 4bcd2b70 (arxiv moved overnight: ACTMIX RLHF stretch complete,
  "T-curve is an ORDER-FREE INVERTED-U peaking at T8" — relevant to the RLHF
  story; read the briefing before writing RLHF conclusions).
- runpodctl authenticated. Two RUNNING pods on the shared account —
  `backtracking-two` (4 GPU, $1.76/h), `sparse_probing_rlhf_ablations`
  (3 GPU, $8.97/h) — reject all local SSH keys; collaborator work, left alone.
  Names overlap this workstream: check with the team in the morning to avoid
  duplicated effort.
- Design-plan agent (staged execution details) still running; proceeding on the
  locked facts from the audit; will fold its report in when it lands.
- Anthropic keys + HF_TOKEN present on this Mac; pods get them via .env.

## Next

- Implement `stacked_pooled` adapter arch + registry + cell tables (task 2).
- C7 forensics on temp-bench results.json generation (task 3).
- Push branch → provision fleet (task 4).
