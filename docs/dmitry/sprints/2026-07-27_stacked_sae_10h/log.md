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

## 05:25–05:50 — adapters green, fleet up

- `stacked_pooled.py` (StackedSAEPooled + StackedBTKOnlyPooled, max-|act| pool,
  sign-preserving) + registry + ACTMIX stacked lanes + pre-registered gate
  tests: **85 tests green, run.py validate OK**. One real bug caught by
  design-review-before-run: parent `train_step` calls `self.encode` → pooled
  override would have crashed `decode`; per-position routing fixed pre-commit.
- Pushed `dmitry-stacked-arxiv` (code-before-bootstrap rule).
- **Table 2 shortcut confirmed on temp-bench leaderboard**: stacked_sae
  `d08c6498d3fa430e` seed 42 has the full detection sweep
  pr_auc S1..S32 = 0.140/0.145/0.161/0.177/0.174/0.187 (+ shuffle twin, gaps
  ≤0.027). S=8 0.177 sits between paper TopK 0.175 and TXC-pro 0.242. More
  stacked train_keys exist at seeds 1/2. **BUT** no leaderboard row exactly
  reproduces any paper Table 2 row (closest topk f437e623 diverges at S≥4) —
  the paper table is a different eval generation. Forensics agent
  (background) is tracing Fig 4/Table 2 provenance → REUSE vs RETRAIN verdict.
- Fleet v2 up (v1 killed: pods rejected all local SSH keys; registered
  id_ed25519 with the account and recreated): stacked-c7 3xpfyrmp8bj18n (H100),
  stacked-c6 dqr4p0t9vkx2zv (H100), stacked-c3 udlwpbaw9c3d8r (A40),
  stacked-rlhf 3ktwi3pacoh8v6 (A40). $6.86/h total ≈ $69/10h.
- Dead end logged: `runpodctl exec python` shares the rejected-key path
  (rsync exit 12) — key injection into running pods impossible; recreation
  was the fix.
