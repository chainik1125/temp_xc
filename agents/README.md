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
- **`runpod-a` / `runpod-b`** — TWO agents sharing the 2×H100 pod
  (Han-provisioned 2026-07-27 ~16:00 London; 56 cores / 503 GB RAM /
  1 TB volume) — the successors of `mac-a` / `mac-b` after the Modal
  workspace spend-limit migration (LOG ~16:10 entry). **runpod-a =
  hunt executor (mac-a's queue: hunt4w2 llama31 leg, cnov panel on
  pick, gen-4 continuation), GPU 0**; **runpod-b = adversarial
  replication + evidence/exhibit hat (mac-b's duties), GPU 1**
  (rebalancing is scheduling, not frozen config). Per-agent clones
  `/workspace/agents/<id>/temp_xc` + `.agent_id`; tokens
  `/workspace/.tokens/` (gh, hf, hf_datasets; NO Modal creds by
  design — Modal needs route via a mac agent to the HF mirror).
  The `actmix-mac-{a,b}.md` briefings apply to them mutatis
  mutandis (venue: pod GPUs instead of Modal; spend → `RUNPOD`
  section of MODAL_SPEND.md). Existing Modal-frozen cards execute
  on-pod under a disclosed VENUE AMENDMENT line, not a re-freeze.
- **`mac-c`** — local Mac agent (spawned 2026-07-26 evening for the
  ACTMIX archaeology). Same clone layout + `.agent_id`; $0-compute
  workstream (`briefings/actmix-mac-c.md`); 07-27 night: safety-hunt
  continuation w/ own RunPod pod authority
  (`briefings/safety-hunt-continuation.md`).
- **`mac-d`** — local Mac agent (stood up 2026-07-27 ~23:40, Han's
  full-throttle order): RunPod-API EXECUTOR — spins up dynamic pods
  under Dmitry's key, runs other agents' frozen cards as detached
  jobs, repatriates rows, terminates. Bring-up:
  `agents/mac-d/STATUS.md`.
- **`runpod-c`** — the T-SCALING HILL-CLIMB agent, ALONE on a
  dedicated 2×H100 pod (Han-provisioned 2026-07-27 evening; 52 CPU /
  503 GB / 1 TB; both GPUs). Mission: make TXC T-scaling improve on
  sparse probing (txc_pro recipe reimplementation + training-trick
  hill-climb) under a pre-registered dev/holdout split — arch R&D,
  no claim surfaces without L3 holdout validation + ratification.
  Clone `/workspace/agents/runpod-c/temp_xc` + `.agent_id`; tokens
  `/workspace/.tokens/`; briefing `agents/runpod-c/STATUS.md`.
- **`runpod-1` / `runpod-2`** — TWO agents sharing ONE 3×H100 pod
  (Han-provisioned 2026-07-26 evening; 84 CPU / 564 GB RAM / 2 TB
  persistent volume) for the paper-task ablations: **runpod-1 =
  sparse probing, GPUs 0,1**; **runpod-2 = EM, GPU 2** (roster paired
  with `scripts/set_agent_env.sh`; rebalancing is scheduling, not
  frozen config). Per-agent clones `/workspace/agents/<id>/temp_xc`;
  setup = `briefings/actmix-pod-bootstrap.md`; work briefings
  `actmix-runpod-{1,2}.md`. Ledger: `RUNPOD` section of
  MODAL_SPEND.md.

*(Retired 2026-07-27 ~16:00: **`mac-a` and `mac-b`** — replaced by
`runpod-a`/`runpod-b` above when the Modal workspace spend limit
blocked new launches mid-window; every staged item (cards, pins,
drivers, scorers) was already committed, so the handoff is entirely
via git — their final stand-down STATUS commits close them out, after
which their `agents/` dirs are removed. Retired 2026-07-26: `runpod`,
`runpod-b`…`runpod-e`, and the interim 6×A40 pod — all dead/closed;
their STATUS records live in git history. NB: the `runpod-b` NAME is
reused by the live 2×H100 agent above; the 07-26 retiree of the same
name was an unrelated A40-pod agent.
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
`/workspace/agents/<id>/temp_xc` (shared pods): the directory + the
`.agent_id` beside your clone is your id. A Linux session under a bare
`/workspace/temp_xc` (single-agent pod) reads `/workspace/.agent_id`
(seeded at pod spawn; there is no legacy default — the pre-2026-07-26
pods are retired). If genuinely ambiguous, ask the user. A new agent
gets a new subdir + a row here.

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
