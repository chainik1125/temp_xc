---
status: active
created: 2026-07-26
for: mac-local (EXECUTOR — role shift)
venue: Mac (MPS, local) + Modal (Dmitry's workspace, $500 HARD CEILING)
---

# Overnight execution briefing — mac-local carries the hunt (Han, 2026-07-26 ~02:15)

**ROLE SHIFT.** All RunPod pods are DOWN (funding; interim A40 pod
released after a clean close). There are NO remote agents tonight.
**mac-local executes the remaining hunt work directly**, using Modal
(Dmitry's workspace) for GPU. Han sleeps; check-in **Sunday 10:00 PT**;
rebuttal deadline 2026-07-27. Everything of value must be pushed as it
lands (this Mac is not ephemeral, but the discipline stays).

## Modal — credentials, ceiling, cost discipline

- Profile `reichers-shai-c9-dmitry` ACTIVE in `~/.modal.toml`; backups
  `~/.tokens/modal_token_{id,secret}`. Client in the scratchpad venv
  (`modal-venv/bin/modal`) or any `pip install modal`.
- **BUDGET: $500 HARD CEILING (Dmitry, confirmed via Han 2026-07-26).**
  Soft stop at **$400**; nothing new launches past it. Keep a SPEND
  LEDGER in this file (append per run: what, GPU type, est. hours ×
  rate, running total). Modal's dashboard is the authority; estimates
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

## THE OVERNIGHT QUEUE (in order; timestamps PT; stop any item that
slips its box and move on — a sound partial beats a stuck complete)

**0. Bring-up (~30 min, ≤$5).** Finish the smoke test (bare GPU hello;
then the repo image + `run.py validate` inside a container). If Modal
is not viable within ~45 min of real effort, STOP fighting it — items
2/3 degrade gracefully: 1, 4, 5 need no GPU; record the outcome.

**1. Expedited gate-reviews of the two panels (no compute, ~45 min).**
oprate NEGATIVE (RECORD §3d) + fineweb (gemma primary + 2
replications + close). runpod-b's standby loop already verified
receipts/structure continuously, so this is an EXPEDITED review:
freeze-order forensics, verdict-vs-artifact spot checks, figure sanity,
LOG review entries. Mark both "expedited — full depth at team review".
The distillation quotes their verdicts; this must come first.

**2. T-SAE seed top-up on Ward (THE single highest-value compute item;
est. $30–80).** Purpose: bound the headline pre-vs-T-SAE margin
(RECEIPTS R5 = NOT bounded; projected Welch LB ≈ +0.013 at tsae n=6).
Freeze a runner for tsae/T1 seeds {3,4,5} ONLY (pattern: the frozen
3d954869 top-up runner), pinned commit, buffer_tokens UNCHANGED
(524288 — comparability with round-1 tsae is the whole point).
tsae cells are CPU-buffer-bound (~2–3 h each on A40-class): request
high-CPU containers, run the 3 cells in parallel. **HAZARDS, both
must be discharged before pooling:** (a) pooling-validity audit per
runpod-d's precedent (re-eval one round-1 tsae cell under current
code → must reproduce its stored number; Ward λ labels all-finite
no-op guard re-verified); (b) cache identity: the Ward stream rebuild
must match the committed byte-identity receipt. If EITHER fails,
report the new seeds SEPARATELY, never pooled. Success = R5 updated
to a bounded margin (or honestly still-unbounded at n=6 — also a
result); failure mode "cells too slow on Modal by 06:00 PT" = kill,
report attempt + cost.

**3. B8 `slen` screen (the program's queued instrument; est. $10–25).**
The recency ladder (lat/lev/disp) with pre-registered shuffle ladder
**lat > lev > disp ≈ 0** — the measurement that settles the
order/recency wording. Freeze MY screen card first (bars from the B8
CARD_DRAFT; per-token-first triage; matched-probe-class arms — NO
max-over-arms; foreign-context nulls; doc-identity disclosure per
face). Cache fineweb-400 for available models (gpt2 + llama ungated;
gemma if HF secret exists), run the screen (probes; cheap), one LOG
verdict per face. This is a SCREEN — it licenses nothing beyond a
future panel.

**4. Stretch screens, only if 0–3 are done and ≥$300 remains:**
refmark (B7; ~1.2M tok/model caching, WildChat corpus committed) then
quotedens (B9; 5.3M tok/model). Same screen discipline. NO new
Stage-2 panels tonight — a panel needs ≥6 h + ≥$150 and would collide
with the check-in; the ONLY exception: B8-lat KEEPs decisively on ≥2
models AND it is before 23:00 PT Saturday AND ≥$350 remains — then a
single-model minimal panel MAY be frozen, with all bindings from
stage2-fineweb.md (doc-identity control, k=8·T post, paired v2,
variance receipts).

**5. HARD PIVOT 07:00 PT — the Sunday check-in distillation (mine,
no compute).** Contents: the scoreboard sentence (ONE confirmed case
study — λ̂ backtracking with its receipt set; oprate NEGATIVE; fineweb
no-rule-fires/WEAK/NEGATIVE per model); the kill/negative table; the
amended order finding + whatever B8 measured; probe-capacity story +
the TAKEN v1-canonical decision + receipted v1-conservatism (3
instances, 2 corpora, exact-truth mirror); the force-majeure impact
statement (what died, what it cost, what stands — checkpoints
mirrored at HF, see checkpoints/HF_MIRROR.md); overnight spend ledger
+ overnight verdicts marked pending-review; the post-deadline queue
(v2 adoption via PROBE_V2_SPEC, em-redo review, B8→panel if KEEP,
estimator-attenuation check). Quote ONLY via RECEIPTS.md. Draft goes
in `private/` (untracked) for Han to present; a sanitized state
update goes to `experiments/explorations/synthetic/STATUS.md`.

## Spend ledger (append below; soft stop $400)

- 2026-07-26 02:0x — smoke attempts (A10G cold start + 1 failed image
  build): ≈ $0–1; successful hello ≈ $0.05. Running total ≈ $1.

## Acceptance

Everything pushed; LOG lines per item; RECEIPTS current; distillation
delivered by 09:30 PT. This briefing retires at the Sunday check-in.
