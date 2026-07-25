---
status: active
created: 2026-07-25
for: ALL agents resuming on the interim A40 pod
venue: runpod (6× A40, 57 CPU, 300 GB RAM, 1 TB EPHEMERAL)
---

# READ THIS FIRST — interim-pod bootstrap (force majeure, ~12 funded hours)

**What happened.** The primary RunPod account ran out of funds
overnight; every old pod is DOWN (15+ h to refund). You are on a
temporary pod on a second account: **6× A40, 57 CPU, 300 GB RAM, 1 TB
EPHEMERAL storage, ≈ $30 ≈ 12 hours of funding.** You have lost your
session context. Your `agents/<id>/STATUS.md` is still true about your
SCIENCE state and WRONG about your box (old volume paths, GPU model,
venv locations). This file supersedes all box facts.

**Who you are:** your operator tells you at session start. Expected
sessions on this pod: `runpod-d`, `runpod-e`, `runpod-b`. If ambiguous,
ask — do not guess.

## The five facts that change everything

1. **The old volumes are GONE**: every activation cache and EVERY model
   checkpoint. Git survives: all label npz bundles, corpus artifacts,
   cards, runners, results JSONs, the leaderboard. Anything you
   produce here that is not pushed **does not exist** — the pod is
   ephemeral and dies in ~12 h. **Push after every completed cell
   batch.** Nothing needs HF upload except (optional, end of run, if
   time) final panel checkpoints.
2. **The λ-readout METHODS DECISION is TAKEN** (LOG, 2026-07-25,
   force-majeure entry): **v1 is canonical through the deadline**;
   claim on v1, report the paired v2 columns; never quote v2 as
   canonical. The probe-capacity finding is a receipted limitation
   ("levels are conservative; ordering is robust"), not an open
   question. Do not relitigate it; do not run anything for it.
3. **The deadline stack:** team check-in **Sunday 2026-07-26 10:00
   PT**; rebuttal deadline 2026-07-27. What is pushed by hour 12 is
   what the check-in has.
4. **PAUSED under force majeure** (do not resume, do not delete):
   em-redo, all factory builds/screens, the mirror Stage-3 grid.
5. **GPU discipline on a shared box:** runpod-d owns
   `CUDA_VISIBLE_DEVICES=0,1,2`; runpod-e owns `3,4,5`; runpod-b is
   CPU-ONLY. Set it in every shell; never launch onto a GPU you do
   not own. 57 CPUs are shared — background CPU-heavy work (tsae
   buffers, probes) coordinates by not exceeding ~24 cores per agent.

## Filesystem layout & isolation (SET UP BY THE OPERATOR — verify, don't improvise)

**One clone per agent — never a shared checkout, never worktrees.**
The canonical runner refuses dirty trees, so a shared checkout would
let one agent's edits block another's panel cells; and same-branch
worktrees are impossible in git. Coordination stays where it always
was: `origin/arxiv` (pull-rebase before EVERY push; LOG.md conflicts
resolved append-only/union, upstream first, yours last).

```
/workspace/agents/runpod-d/temp_xc   (+ .agent_id beside the clone)
/workspace/agents/runpod-e/temp_xc
/workspace/agents/runpod-b/temp_xc
/workspace/hf_cache                  (SHARED HF_HOME — models once)
/workspace/.tokens/                  (gh_token, hf_token)
```

Per-agent, set in every shell — `cd` into YOUR clone, then
`source scripts/set_agent_env.sh <your-id>` (the roster now carries
runpod-d / runpod-e / runpod-b for this pod; it sets everything below):
- `CUDA_VISIBLE_DEVICES`: d = `0,1,2`; e = `3,4,5`; **b = `` (empty —
  CPU-only, so a stray torch call cannot grab a GPU).** Verify with
  `python -c "import torch; print(torch.cuda.device_count())"`
  (3 / 3 / 0) — `nvidia-smi` always shows all six; that is expected.
- `HF_HOME=/workspace/hf_cache` (shared; concurrent downloads of
  DIFFERENT models are fine).
- `OMP_NUM_THREADS=16` (57 shared cores; keep ≲ 24 per agent).
- Your identity = your directory: cwd inside
  `/workspace/agents/<id>/temp_xc` means you are `<id>`. Check
  `../.agent_id` if unsure; if it disagrees with what your operator
  said, STOP and ask.

Per-clone (not shared): `.venv` (isolation — one agent pip-installing
must not break another), activation caches (d's Ward and e's fineweb
caches are disjoint anyway; rebuild inside your own clone's cache
path), git identity. Disk is not a constraint (~100 GB total worst
case on 1 TB).

## Environment bring-up (every agent, once)

Repo: clone/pull to the workspace; `.venv` per the repo recipe
(torch + CUDA for d/e; CPU wheel fine for b). `HF_HOME` on the
ephemeral disk. Tokens from your operator. Verify with
`python run.py validate` + `.venv/bin/python -m pytest tests/ -q`
(green suite ≈ 319/1). Models to pull as needed (≈ 50 GB total is
fine): R1-Distill-7B (Ward), gemma-2-2b, gpt2, llama-3.1-8B.

## Assignments (12-hour clocks start at YOUR session start)

### runpod-d — `briefings/stage2-oprate.md` (+ its addendum) — GPUs 0–2
Your panel was NEVER STARTED (the banner on your STATUS still
applies). Steps: (1) **rebuild the Ward cache first** — the committed
stream/cache builders define it exactly (conversion-depth
`build_ward_stream` + the layer the frozen datasource declares);
budget ≈ 30–60 min incl. model pull; (2) claim line; (3) freeze the
card (A40 note + your buffer_tokens choice go IN the card); (4) run
per the briefing's 12-hour queue — tsae first, push per batch.
A40 ≈ 2–3× slower per cell than H100; with 3 GPUs the case panel
fits. `rate_ver` stretch only if case + receipts are DONE and > 2 h
remain on the funding clock.

### runpod-e — `briefings/stage2-fineweb.md` (+ addenda) — GPUs 3–5
Your card (`b8f2f0bd`) and datasource (`f3b9739d`) SURVIVE in git —
the design is done; your unpushed cells are lost. Steps: (1) rebuild
the gemma-2-2b fineweb cache (the datasource's recipe; ≈ 15–30 min);
(2) re-claim with a LOG line noting the restart; (3) rerun the frozen
panel exactly — no re-design, no card changes beyond an A40/restart
appendix; tsae first; push per batch. Replication cells (gpt2 +
llama31) only after the gemma panel + doc-identity receipts + variance
receipts are pushed. The dialevel recency pre-flight is CANCELLED for
this funding window.

### runpod-b — CPU only — `briefings/panel-support-audit.md`, then mirror close-out
The mirror campaign's Stage-3 grid + mix arms are LOST mid-run; its
gate is DISSOLVED by force majeure. Order of work: (1)
**panel-support-audit item 1 NOW** — the variance-harness pre-flight
against both panels' k_pos = 8·T row shape; d and e will need it
within hours; (2) item 2 — the PROBE_V2_SPEC lower-bound caveat (the
decision is taken, but the spec remains the post-deadline freeze
candidate and must carry your own caveat); (3) **mirror CLOSE-OUT
receipt from PUSHED data only** — final `probe_truth` figure +
scorecard over Stage 1 + Stage 2 + transfer test, with an explicit
coverage statement ("Stage-3/mix arms lost mid-run, force majeure —
labels: Stage-1 ADOPT-consistent on the amended scope; frozen-card
scope remains AMBIGUOUS-unresolved"), one LOG entry, and the
campaign's briefing retires with it; (4) item 3 — the RECEIPTS index.
No training. No new mirror cells.

## Priorities if the budget forces choices

A COMPLETE panel with receipts beats two partials. If by ~hour 8 both
panels cannot finish: protect whichever is further along; the other
stops at its last pushed batch with an honest partial verdict. If a
panel finishes early, the flex GPU order is: e's replication cells →
d's rate_ver → B8 `slen` screen on the already-rebuilt fineweb caches
(minutes, and it is the program's queued instrument for the
order/recency question). When nothing useful remains, TELL YOUR
OPERATOR so the pod can be stopped and the remaining dollars saved.

## Acceptance gates

Per your own briefing, unchanged — except every gate gains one clause:
**all artifacts pushed before the funding clock, receipts before
stretch goals.** Briefings stay until mac-local review, as always.
