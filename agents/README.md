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
- **`runpod`** — the ORIGINAL RunPod box (Linux, `/workspace/temp_xc`, 32 CPU /
  128 GB). Heavy grids + long runs; owns the PhenomenonBench loop history
  (C1–C4, stage-6). Git creds at `/workspace/.tokens/`.
- **`runpod-b`** — the SECOND RunPod box (Linux, `/workspace/temp_xc`, 32 CPU /
  128 GB; spawned 2026-07-22 for the FreqBench line). Same creds layout.
- **`runpod-c`** — the GPU RunPod box (Linux, `/workspace/temp_xc`, **H100
  80 GB**, 700 GB persistent volume holding the Ward + EM activation
  caches; spawned 2026-07-23 for the conversion-depth / substrate-audit
  line, now the real-side dictionary-training pod). Same creds layout;
  `/workspace/.agent_id` = `runpod-c`.
- **`runpod-d`** — GPU RunPod pod (spawned 2026-07-24, rebuttal window;
  H100 preferred). Task-hunt arm A (`briefings/task-hunt.md`): trace-
  derived candidates. May mount runpod-c's 700 GB volume **READ-ONLY**
  (runpod-c owns all writes); else rebuilds from committed builders.
  `/workspace/.agent_id` = `runpod-d`.
- **`runpod-e`** — GPU RunPod pod (spawned 2026-07-24, rebuttal window;
  H100 preferred). Task-hunt arm B (`briefings/task-hunt-b.md`):
  repetition-lag across model scale + confidence trend. Fully
  volume-independent. `/workspace/.agent_id` = `runpod-e`.

(2026-07-23, rebuttal window: backtracking multi-seed reruns and paper
latex edits are owned by the human team, NOT by agents in this registry.
Agent work = the three standing directives; `private/` files must never
be committed or quoted in tracked locations.)

**Infer your id from your environment:** a darwin session under
`~/research/projects/temp_xc` is `mac-local`. A Linux session under
`/workspace/temp_xc` **must check `/workspace/.agent_id` FIRST** — with two
RunPod boxes the path alone is ambiguous: if the file exists, its content is
your id; if it does not exist, you are `runpod` (the original box — legacy
default; do NOT create the file there). The user seeds
`/workspace/.agent_id` on every newly spawned pod. If genuinely ambiguous,
ask the user. A new agent gets a new subdir + a row here.

**Citing commits in records:** `git pull --rebase` rewrites your local
SHAs, so a SHA written into a record before pushing is usually stale by
the time it lands (this has now bitten two review cycles). Cite the
commit *subject line* (stable across rebase), or re-verify cited SHAs
against `git log` AFTER your final push.

**Two-agent parallel sessions (shared-branch rules):** always
`git pull --rebase origin arxiv` before every push; treat the shared files
(`configs/data.yaml`, `src/temp_bench/data/synthetic.py`, `BENCHMARKS.md`,
research STATUS § 0) as **append-only** — add your own entry/bullet, never
rewrite another agent's; `results/leaderboard.jsonl` and
`checkpoints/manifest.jsonl` are append-only JSONL with a **union merge
driver** (`.gitattributes`) — on a conflict elsewhere, resolve by keeping
BOTH sides. Each loop logs spend to its OWN file (expansion →
`expansion/results/spend.json`; freqbench → `freqbench/results/spend.json`).

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
