<!--
Briefing template — copy to:
  purified/agents/<agent_name>/briefing.md

Section ownership (PROTOCOL.md § 14):
- "Identity + mandate (Han owns)": Han's prose. Agents do NOT edit.
- All other sections: agent-owned, overwritten at every compact.

For chronological history: `git log -p purified/agents/<name>/briefing.md`.
Don't create a separate `log.md` — git history + decisions.md are enough.
-->

---
agent: <agent_name>
last_state_update: <UTC ISO timestamp>
component: <c1|c2|...|c7|orchestration>
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent <name>**. You own <component(s)> only. Files you may edit:
- `agents/agent_<name>/briefing.md` (your own — agent-owned sections only)
- `docs/components/c<N>.md` (your component(s))
- `experiments/c<N>_*/` (your component(s))
- Code under `src/temp_bench/` that you author + commit (eval module,
  case-study runner, training helpers as relevant)
- `configs/datasources.yaml` — adding new datasources for your
  component is fine.

**Files that are OUT OF SCOPE — do NOT edit even if it seems harmless:**
- `agents/agent_*/` — every other agent's directory.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.
This is non-negotiable — see PROTOCOL.md § 8 *Anti-conflict workflow*
and CLAUDE.md Hard Rule #7.

[Han populates the rest of this section with the agent's identity,
mandate, hardware allocation, and any priorities. Verbatim, often the
original briefing message. Agents leave this untouched; Han may rewrite
at session start to redirect priorities.]

---

## Current state (agent owns — overwrite at every compact)

**Last verified: <UTC ts>**

- `git HEAD`: <sha>
- Last leaderboard append: `<eval_key>` (or "(none yet)").
- Last checkpoint saved: `<train_key>` (HF: `<url-or-pending>`).
- Active GPU lock(s): <list from gpu_lock_status() or "none">.
- Recent decisions in scope: `decisions.md` #<N>, #<M>.
- In flight: <task at step N of M; partial files; or "nothing">.

## What I just did (agent owns — overwrite)

Newest first, 5–10 bullets. Reference commits / docs by name; don't
restate. For chronological detail, agents read `git log` of the
recent commits.

- <action>
- <action>

## Next action (agent owns — overwrite)

Concrete first step for the next-life instance. Commands, file paths,
expected runtime.

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh <agent_name>`
3. `bash scripts/agent_smoke_test.sh`
4. `git pull --rebase origin final`
5. <the actual first task, with commands>

## Don't repeat (agent owns — overwrite)

Pitfalls. Cite the closed decision in `decisions.md` if applicable.

- <pitfall>: <reason>
- (Common: don't add a third TXC; don't import from wasteland;
   don't bypass `runner.run_cell`; don't allocate run_ids manually.)

## Open questions for Han (agent owns — overwrite)

(If any — otherwise "(none)".)

- <question>
