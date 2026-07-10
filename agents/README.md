# agents/ — per-agent workspaces

Each agent working this repo owns `agents/<agent-id>/` for its **own working
state** — a personal scratchpad + pre-compact handoff-to-self. This keeps
agents from colliding on shared files or getting confused by each other's
in-flight work.

## Three kinds of state — keep them distinct

| doc | scope | lifetime |
|---|---|---|
| **`agents/<id>/STATUS.md`** | **YOUR** working state — what you're mid-doing, git position, next concrete action | ephemeral; **rewrite before every compact** |
| **`briefings/<task>.md`** | a **shared task / idea** any agent can pick up | deleted when the task is done |
| **`experiments/explorations/synthetic/STATUS.md`** | **shared research-program** state — verdicts, roadmap, benchmark index | living; update when you advance the science |

Personal working-state → your workspace. Shared to-do → `briefings/`. Shared
research knowledge → the research STATUS. When in doubt: would another agent need
this to continue the *science*? → research STATUS. To pick up a *task*? →
`briefings/`. Only to resume *your own half-finished thread*? → your workspace.

## Agent ids (current)

- **`mac-local`** — local CC session (macOS/darwin, `~/research/projects/temp_xc`,
  Apple-silicon MPS, no CUDA). Prototyping, review, orchestration.
- **`runpod`** — remote CC session on RunPod (Linux, `/workspace/temp_xc`, CUDA).
  Heavy grids + long runs. Git creds at `/workspace/.tokens/`.

**Infer your id from your environment:** a darwin session under
`~/research/projects/temp_xc` is `mac-local`; a Linux session under
`/workspace/temp_xc` is `runpod`. If genuinely ambiguous, ask the user. A new
agent gets a new subdir + a row here.

## What goes in your workspace

Your `STATUS.md` (required) plus any private scratch — drafts, notes, throwaway
scripts. **Shared** work products (task briefs, bench specs, records, figures,
results) go in their proper shared locations (`briefings/`, the bench subdir,
`results/`), never hidden in a private workspace.

## Before a compact

Rewrite your `agents/<id>/STATUS.md` so a fresh context window (you or a human)
resumes your exact thread: what's done, what's in flight, git state, the next
concrete action. If you advanced the research, also update the shared research
STATUS.
