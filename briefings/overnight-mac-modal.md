---
status: active
created: 2026-07-26
for: SHARED ops doc — mac-a, mac-b (executors) + mac-local (orchestrator)
venue: Mac (local) + Modal (Dmitry's workspace, $500 HARD CEILING)
---

# Overnight shared ops — Modal recipe, budget, discipline (Han, 2026-07-26)

**STRUCTURE (superseding the earlier single-executor plan):** all
RunPod pods are DOWN; tonight TWO local Mac agents execute autonomous
loops — **mac-a** (`briefings/overnight-mac-a.md`) and **mac-b**
(`briefings/overnight-mac-b.md`) — in their own clones under
`~/research/projects/agents/<id>/temp_xc`, pushing regularly.
**mac-local stays ORCHESTRATOR** (reviews, budget oversight, the
Sunday distillation). Han sleeps; check-in **Sunday 10:00 PT**;
deadline 2026-07-27. This file holds what is SHARED: the Modal
recipe, the budget rules, and the standing discipline. Your queue is
in YOUR briefing.

## Modal — credentials, ceiling, cost discipline

- Profile `reichers-shai-c9-dmitry` ACTIVE in `~/.modal.toml`; backups
  `~/.tokens/modal_token_{id,secret}`. Client in the scratchpad venv
  (`modal-venv/bin/modal`) or any `pip install modal`.
- **BUDGET: $500 HARD CEILING (Dmitry, confirmed via Han 2026-07-26).**
  Soft stop at **$400** TOTAL across both agents; nothing new
  launches past it. **The SPEND LEDGER is `briefings/MODAL_SPEND.md`
  (tracked, append-only, union-merge like the LOG): READ the running
  total BEFORE every launch, APPEND a line after (agent, what, GPU,
  est. hours × rate, new total). Per-agent initial caps: mac-a $150,
  mac-b $100 — raises only by mac-local briefing amendment.** Modal's
  dashboard is the authority; estimates
  are for pacing. Reference rates: A10G ≈ $1.10/h, L40S ≈ $1.95/h,
  A100-40 ≈ $2.8/h. Prefer A10G/L40S; H100 only with a stated reason.
- Smoke state at handoff: **PASSED** — bare GPU hello returned
  `NVIDIA A10, 23028 MiB, driver 580.95.05` (so queue item 0 is just
  the repo image + in-container `run.py validate`). One caveat: the
  torch cu128 debian_slim image build FAILED once (logs unread) —
  sidestep by building the image via the repo's own `uv sync`, which
  is the plan anyway. Note the A10G alias delivers an A10 23 GB —
  fine for screens/tsae; pick L40S/A100 for anything needing >20 GB.
- Token rotation after the weekend (secret transited chat) — Dmitry.

## Modal execution recipe (build once, reuse all night)

1. **Base image**: `modal.Image.debian_slim().apt_install("git")` +
   run commands: clone `https://github.com/chainik1125/temp_xc.git`
   at a **PINNED COMMIT** (clean tree ⇒ code_version stamping valid),
   install uv, `uv sync`. Snapshot once (~10 min), reuse per function.
2. **Secrets**: create a Modal secret with the HF token IF
   `~/.tokens/hf_token` exists locally (bootstrap_local layout) —
   needed only for gated models (gemma-2). **Ungated: gpt2, the Nous
   llama mirror, R1-Distill — the Ward substrate needs NO HF auth.**
3. **Caches**: rebuilt in-container from committed builders
   (deterministic; byte-identity receipts exist from the A40 restart).
   Persist across function calls with a `modal.Volume` mounted at the
   cache path so re-runs don't re-forward-pass.
4. **Results repatriation**: containers do NOT push to git. Each run
   returns (or writes to the Volume) its results JSON + the leaderboard
   rows it appended in-container; **mac-local merges locally** —
   dup-eval-key check before append, then commit + push. Canonical
   runner inside the container; canonical merge outside.
5. Long runs via `--detach` / background + a local Monitor; every run
   time-boxed; partial results repatriated on the same pattern.

## Standing discipline (unchanged by the role shift)

Commit-then-run for ANY new card/runner (freeze before first cell —
push the freeze commit and pin the container to it). One LOG line per
verdict, marked `mac-local (executor)`. **Self-review hazard, named:**
I am executor AND reviewer tonight — compensate by pre-registering
predictions/kill-rules in the frozen card BEFORE running, quoting only
via RECEIPTS (extend it + run `receipts_check` per new claim), and
flagging every overnight verdict as PENDING TEAM REVIEW at the Sunday
check-in. v1 canonical; paired v2 reported, never claimed. No
max-over-arms. doc_mean_only_auc = disclosure-triggers-control.
Training-corpus size beside every unigram number.

## Work split (queues live in the per-agent briefings)

- **mac-a** (`overnight-mac-a.md`): Modal bring-up + the tsae/T1 seed
  top-up (bounds RECEIPTS R5 — the single highest-value compute item),
  then assist/stretch.
- **mac-b** (`overnight-mac-b.md`): B8 `slen` screen (the recency
  instrument), then refmark + quotedens stretch screens.
- **mac-local** (orchestrator): expedited gate-reviews of the two
  completed panels, rolling review of a/b pushes, budget oversight,
  **HARD PIVOT 07:00 PT → the Sunday distillation** (quote only via
  RECEIPTS.md; draft in private/). NO new Stage-2 panels tonight
  except the narrow B8 exception: B8-lat KEEPs decisively on ≥2
  models AND before 23:00 PT AND ≥$350 remains AND mac-local approves
  in writing (LOG line).

## Acceptance

Everything pushed; LOG lines per item; RECEIPTS proposals for new
claims (mac-local ratifies); distillation delivered by 09:30 PT. This
doc and both agent briefings retire at the Sunday check-in.
