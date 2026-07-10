# briefings/

Dedicated home for **task briefings / handoffs** — one self-contained brief an
agent (RunPod or local) executes, then **deletes**. This is where handoffs live
so they stop getting scattered across root-level `HANDOVER.md`-style files.

## What belongs here

One markdown file per queued task: a refactor, a cleanup sweep, a benchmark
port, an infra fix. Each brief must be self-contained enough that an agent can
execute it after the CLAUDE.md read order — scope, current state, step-by-step,
constraints (the hard rules), and an explicit **acceptance gate**.

## What does NOT belong here

- **`agents/<id>/STATUS.md`** — an agent's *own* working state (personal
  scratchpad + pre-compact handoff). A briefing is a shared task any agent can
  pick up; a workspace STATUS is one agent's private thread. See `agents/README.md`.
- **`experiments/explorations/synthetic/STATUS.md`** — the *living* research
  state (kept current, never deleted). A briefing is transient; STATUS is
  permanent. Research verdicts/state go in STATUS, not here.
- **`docs/`** — durable framework + idea writeups.
- **`RUNPOD_INSTRUCTIONS.md`, per-bench `bench_spec.md`** — standing infra docs /
  frozen research specs that live with their subject.

## Convention

- **Name:** kebab-case by task, e.g. `refactor-record-pipeline.md`.
- **Header:** `status` (active | done | abandoned), `created` (absolute date),
  `for` (who executes), `venue` (runpod | local | any).
- **Lifecycle — delete when done.** When the task is executed and absorbed (or
  abandoned), **delete the file** in the same PR/commit. Briefings must not
  accumulate; a stale or unused briefing is a bug, not a record. The durable
  record of *what was done* is the git history + the artifacts it produced
  (code, `bench_record.md`, STATUS), never a leftover brief.
- **Discovery:** CLAUDE.md session-start points agents at `briefings/`. If a
  briefing is also the active *research* thread, add a one-line pointer from
  STATUS §0; pure infra/cleanup briefs stay here only.
