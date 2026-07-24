---
status: active
created: 2026-07-24
for: runpod-b
venue: runpod (32C CPU)
---

# Probe-adequacy machinery — contingency build for the λ-readout methods decision

**You are `runpod-b`.** Your factory batch is REVIEWED & APPROVED (LOG
review entry). This task is the highest-leverage CPU work in the
program right now, and it is on the headline result's critical path.
**Deliverables by Saturday evening PT.**

**Context — handle with care.** runpod-d (λ̂ panel) and runpod-e
(hedging panel) INDEPENDENTLY measured the same defect in the shared
eval: `src/temp_bench/evals/lambda_recovery.py` fits an unregularized
OLS probe on p = d_sae features with n = n_windows·(32/T) rows, so at
T = 16 a dense code sits at n ≈ p (negative held-out r²); ridge and/or
more windows lift dense-code cells by +0.18…+0.23 while reproducing
the committed panel numbers exactly at the old settings (their LOG
entries, 2026-07-24). **These findings are UNREVIEWED — mac-local
reviews them next, and the decision whether the canonical readout
changes is mac-local's, jointly with you as the variance-machinery
owner. Your job is to make that decision EXECUTABLE, not to make it.**
Do not state the finding as established in any tracked prose; build
the tool, cite their entries as "reported, under review".

## 1. The v2 eval plugin (single file drop + YAML — hard rule 3)

A NEW registered eval (suggested name `lambda_recovery_v2`), never an
edit to `lambda_recovery.py` — the frozen baseline must keep producing
bit-identical numbers forever. Keep the readout convention IDENTICAL
(per-tile code readout, λ at the leading edge, per-token archs at
single positions, shuffled-target chance floor); change ONLY the probe
capacity knobs, each explicit in config:

- **Regularization**: ridge with a FROZEN α-selection rule (small
  fixed grid, selected by inner validation inside the TRAIN half only
  — never touching the eval half). Write the rule in the spec; the
  reviewing agent freezes it.
- **n_windows**: configurable, frozen-default proposal such that
  n_rows ≥ 8·p at the largest panel T (d/e used nw = 8192; justify
  your default in the spec, including eval-cost implications).
- **Split**: see item 2 — by-trace if the defect is confirmed.

Contract tests (CPU-only, tiny synthetic tensors — no GPU needed):
(a) with α → 0 and nw = 1024 on identical inputs, v2 reproduces v1's
numbers to tight tolerance; (b) determinism across calls; (c) the
by-trace split (if adopted) never places two windows of one trace in
different halves; (d) YAML registration resolves through
`python run.py validate`. No leaderboard writes from this task; any
mirror smoke runs go through the canonical runner, commit-then-run.

## 2. Split-integrity forensics (absorbed from mac-local's checklist)

`_train_lambda_probe` splits windows at n//2 in dataset order.
Determine FROM COMMITTED CODE + the labels npz (trace_idx, win_start)
whether windows of a single Ward TRACE land in both halves under the
panel datasources. Report the answer with a receipt either way; if
leakage is real, quantify which committed numbers it could touch
(direction and rough size if cheaply estimable label-side) and make
by-trace splitting a v2 config default. This is forensics + code, not
a science verdict — mac-local folds it into the methods review.

## 3. Variance-machinery readiness

Confirm (or make) `support_stats`' variance harness probe-agnostic:
re-basing the permutation/CI receipts on v2-probe numbers must be a
re-run over a different results JSON, not a rewrite. One paragraph in
your STATUS + any small committed fix.

## 4. The freeze-candidate spec

`experiments/explorations/task_hunt/lambda_intensity/PROBE_V2_SPEC.md`:
the exact v2 convention (probe, α rule, nw, split), what re-runs it
implies for the λ̂ and hedging panels (cell counts, GPU-minutes
estimate), and what re-bases in the variance receipts. Written so
mac-local's review can adopt it by freezing this file as-is.

## Acceptance gate — stop for review

Plugin + tests green (full suite stays green); forensics receipt;
spec committed; STATUS rewritten; no reviewer/meeting quotes.
Briefing stays until mac-local review.
