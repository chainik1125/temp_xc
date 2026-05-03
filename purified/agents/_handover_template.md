<!--
Handover template — copy to:
  purified/agents/<your_name>/handovers/<UTC-ISO-timestamp>-<slug>.md

Filename: YYYY-MM-DDTHH-MMZ-<slug>.md  (UTC, T separator, no colons in time)
e.g.   2026-05-03T20-15Z-c3-cache-built.md

Replace every placeholder. Keep total ≤ 200 lines. Reference, don't
duplicate (point at decisions.md / log.md / cN.md instead of restating).
See PROTOCOL.md § 14 for the full contract.
-->

---
agent: <your_name>
ts: <UTC ISO timestamp, e.g. 2026-05-03T20:15:00Z>
type: handover
prev_handover: <filename of the previous handover, or "first">
component: <c1|c2|...|c7|orchestration>
---

## Identity

`<agent_name>` — <one-line role>. Pod: `<pod>`. GPU: `<idx>`.
Mode: `<persistent|ephemeral>`. Briefing: [`briefing.md`](../briefing.md).

## State of the world (verified <UTC ts>)

- `git HEAD` = `<sha>` (`git log -1 --format="%h %s"`)
- Last leaderboard append: `<eval_key>` (`<arch>_<seed>_<k>` for `<component>`).
  See `results/leaderboard.jsonl` last N rows for context.
- Last checkpoint saved: `<train_key>` (HF: `<url-or-"pending">`).
- Active GPU lock(s): `<list from gpu_lock_status() or "none">`.
- Recent decisions in scope: `decisions.md` #<N>, #<M>.
- Open framework questions: `<list>` (or "none — proceed").

## What I just did

Last 5–10 substantive actions, newest first. One bullet each.

- <action>
- <action>
- <action>

(For long context, link to log entries instead of restating.)

## What I'm in the middle of

(If anything; otherwise: "nothing in flight; clean break".)

- <task>: at step <N> of <M>. Partial files at `<paths>`. Successor
  should `<continue|abandon|verify>`.

## Next action for my successor

Be concrete. Commands, file paths, expected runtime.

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh <agent_name>`
3. `bash scripts/agent_smoke_test.sh` (expect 38/38 + N expected gaps)
4. `git pull --rebase origin final`
5. `<the actual first task — be specific>`

## Don't repeat

- <pitfall>: `<reason — usually a closed decision in decisions.md>`
- <pitfall>: <reason>
- (Common: don't re-add a third TXC; don't import from `src/`;
   don't bypass `runner.run_cell`; don't manually allocate run_ids.)

## Open questions for Han

(If any — otherwise "(none)".)

- <question>
