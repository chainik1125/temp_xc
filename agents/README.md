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
- **`mac-a` / `mac-b`** — local Mac agents (Claude Code instances). Clones
  `~/research/projects/agents/<id>/temp_xc` (+ `.agent_id` beside);
  identity = your directory. GPU via Modal ONLY (shared profile in
  `~/.modal.toml`; check and append `briefings/MODAL_SPEND.md`
  before/after every run; $150/day/person cap as of 2026-07-26).
  Current phase briefings: `actmix-mac-{a,b}.md`.
- **`mac-c`** — local Mac agent (spawned 2026-07-26 evening for the
  ACTMIX archaeology). Same clone layout + `.agent_id`; $0-compute
  workstream (`briefings/actmix-mac-c.md`).
- **`runpod-1` / `runpod-2`** — TWO agents sharing ONE 3×H100 pod
  (Han-provisioned 2026-07-26 evening; 84 CPU / 564 GB RAM / 2 TB
  persistent volume) for the paper-task ablations: **runpod-1 =
  sparse probing, GPUs 0,1**; **runpod-2 = EM, GPU 2** (roster paired
  with `scripts/set_agent_env.sh`; rebalancing is scheduling, not
  frozen config). Per-agent clones `/workspace/agents/<id>/temp_xc`;
  setup = `briefings/actmix-pod-bootstrap.md`; work briefings
  `actmix-runpod-{1,2}.md`. Ledger: `RUNPOD` section of
  MODAL_SPEND.md.

*(Retired 2026-07-26: `runpod`, `runpod-b`…`runpod-e`, and the interim
6×A40 pod — all dead/closed; their STATUS records live in git history.
Per-pod volumes were independent; anything needed again must be REBUILT
from the committed builder scripts — every cache builder is committed
with its artifacts' spec. Cross-pod handoff goes through committed
scripts + small stats files, never bulk data.)*

(2026-07-23, rebuttal window: backtracking multi-seed reruns and paper
latex edits are owned by the human team, NOT by agents in this registry.
Agent work = the three standing directives; `private/` files must never
be committed or quoted in tracked locations.)

**Infer your id from your environment:** a darwin session under
`~/research/projects/temp_xc` is `mac-local`; under
`~/research/projects/agents/<id>/temp_xc` the directory (and the
`.agent_id` file beside the clone) is your id. A Linux session under
`/workspace/temp_xc` reads `/workspace/.agent_id` (Han seeds it on every
newly spawned pod; there is no legacy default — the pre-2026-07-26 pods
are retired). If genuinely ambiguous, ask the user. A new agent gets a
new subdir + a row here.

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
