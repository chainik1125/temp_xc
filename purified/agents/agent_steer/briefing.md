<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_steer; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_steer
last_state_update: 2026-05-03T22:00:00Z
component: c5
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent STEER**. You own C5 only. Files you may edit:
- `agents/agent_steer/briefing.md` (your own — agent-owned sections only)
- `docs/components/c5.md`
- `experiments/c5_steering/`
- Code under `src/temp_bench/` that you author + commit
  (`temp_bench.case_studies.steering`, `temp_bench.eval.steering`)
- `configs/datasources.yaml` — adding new C5 datasources is fine.

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

You are agent STEER, lead on **C5: RLHF steering** on
`google/gemma-2-2b-it` layer 13 — same subject as C3/C4. The case
study is the T-SAE paper § 4.4 sentiment-steering task.

Hardware: pod `4× A40`, pinned to **GPU 0**. Pod mode **`ephemeral`**:
`/workspace` is wiped on pod stop, HF is the source of truth.
Bootstrap pulls from `han1823123123/temp-bench-{models,data}`;
`cache.save_checkpoint` auto-pushes on save (push failure is fatal —
we cannot risk losing a multi-hour training run).

agent_back shares the pod on GPU 1 (separate component, separate
cache). GPUs 2 + 3 are spare pool slots — if you need more parallelism
you may claim one via `temp_bench.utils.gpu_locks.claim_gpu(idx)` and
launch a second process with `CUDA_VISIBLE_DEVICES=<idx>`. See
PROTOCOL.md § 13 for the Primary + Pool contract.

**You are gated on agent_nlp** — they build the Gemma-2-2b-IT L13
activation cache (~3 H100-hr) and push to HF temp-bench-data. Your
first session pulls that cache via `sync_from_hf.sh`, so you don't
rebuild. Provisioning order: spawn agent_steer **after** the cache
appears on HF (~T+3 hr), not at T+0.

Hypothesis (modest, from `docs/components/c5.md`): TXC-base + TXC-pro
produce coh-vs-success curves comparable to T-SAE; both match T-SAE
at coh ≥ 1.75. **This is "matches" not "beats"** — accepted in
exchange for stronger C3/C4 claims.

Steering protocol: **V7 tiled-broadcast** (per-token decoder-row
addition, stride-T blocks, single uniform δ within each block).
Chosen for arch-uniformity, not peak performance per arch. **Pre-test
that V7 is OK for TXC-pro** (subseq encoder + multi-distance
contrastive may break under V7); switch to PP if so.

**Excluded by design**: Y/W hill-climbing winners (Galaxy 8/11/18,
SoftMaxPool, ContrastiveMergeH8). They beat T-SAE on steering but
lose 0.005–0.020 probing AUC, inconsistent with "two TXCs everywhere."

Locked decisions in scope: #1 (two TXCs — DO NOT use Galaxy/SoftMaxPool
hill-climbing wins), #4 (cross-branch reads), #6 (HF repos), #7
(Bricken resample is C6-only by default; **C5 keeps it OFF** —
revisit only if time permits at the end of the paper sprint).

References:
- `agents/README.md` (your roster row + pod-mode contract)
- `docs/components/c5.md` (full setup, V7 protocol, hypothesis)
- `docs/paper/hardware.md` *Multi-GPU access* (Pool example)
- `decisions.md` (esp. #1)
- `papers/temporal_sae.md` § 4.4 (the case study)
- `PROTOCOL.md` § 11, § 12 (pinning), § 13 (Primary + Pool)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: (not yet provisioned — wait for agent_nlp's cache)**

- `git HEAD`: (set on first session)
- Last leaderboard append: (none yet)
- Last checkpoint saved: (none yet)
- Active GPU lock(s): none
- Recent decisions in scope: #1, #4, #6, #7
- In flight: nothing (provisioning pending)

## What I just did (agent owns — overwrite)

(Empty — agent_steer not yet provisioned.)

## Next action (agent owns — overwrite)

**Pre-conditions (Han owns)**:
- Han spawns this agent only **after** agent_nlp's Gemma-IT-L13 cache
  is on HF (~T+3 hr). The agent should not be brought up before then.
- Han ran `bash scripts/bootstrap_runpod.sh` on this pod (interactive
  — prompts for tokens; an agent cannot enter input) AND
  `bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_steer`
  to create your own clone. Because this pod is ephemeral, Han re-runs
  both whenever the pod is recreated. Tokens are in
  `/workspace/.tokens/` and your venv is at
  `/workspace/temp_xc_steer/purified/.venv/`. If the smoke test below
  complains about missing tokens, **ping Han**.

**Your clone path is `/workspace/temp_xc_steer/`** (NOT
`/workspace/temp_xc/` — that's agent_back's primary clone). The
separate clone exists so two agents on the same pod don't collide
on `.git/index.lock` during pull-rebase. Tokens + HF cache are
shared via `/workspace/.tokens/` and `/workspace/hf_cache/`.

**Han launches you via `start_agent.sh`** (not bare `claude`):
```
bash /workspace/temp_xc_steer/purified/scripts/start_agent.sh agent_steer --fresh
```
Re-launches after disconnect drop the `--fresh` so the wrapper passes
`--continue` to claude — you resume your session instead of re-reading
the briefing. The wrapper sources `set_agent_env.sh` in Han's parent
shell so the GPU pin / `AGENT_NAME` / pod mode (ephemeral) propagate
into your process. Bash tool calls don't share shell state, so YOU
sourcing the env in your first action is a no-op for subsequent
commands.

1. `bash scripts/agent_smoke_test.sh` (verifies GPU pin etc.)
2. `bash scripts/sync_from_hf.sh` (mandatory — ephemeral pod;
   pulls agent_nlp's act-cache from `han1823123123/temp-bench-data`)
3. `git pull --rebase origin final`
4. Read `docs/components/c5.md` end-to-end and `papers/temporal_sae.md`
   § 4.4 for the case study definition.
5. Set up Gemini judge (coh + success heads). API key resolves via
   `temp_bench.utils.tokens.get_token("gemini")` from
   `/workspace/.tokens/`.
6. Port `temp_bench.case_studies.steering` from
   `origin/han-phase7-unification` (search experiments/ + src/ for
   the V7 tiled-broadcast steering pipeline; cite phase7's
   `unified-pareto.md` for context).
7. Train T-SAE, TXC-base, TXC-pro on the cached acts (3 archs × 3
   seeds = 9 cells). Use `runner.run_cell(...)` — schema validation
   + auto-push to HF on save.
8. **Pre-test V7 on TXC-pro**: 1 cell at coh threshold 2.0; if
   success rate is degenerate, fall back to per-position (PP) and
   document the switch in c5.md.
9. Run V7 across 5 coherence thresholds {1.5, 1.75, 2.0, 2.25, 2.5} →
   coh-vs-success curves.

## Don't repeat (agent owns — overwrite)

- **Hill-climbing winners** — Galaxy 8/11/18, SoftMaxPool,
  ContrastiveMergeH8 are excluded by decision #1. If TXC-base and
  TXC-pro lose to T-SAE, the paper accepts that — don't chase.
- **Skip the V7 pre-test** — TXC-pro's subseq + multi-distance
  contrastive may not be V7-compatible.
- **Forget HF push on save** — ephemeral pod; checkpoint loss = run loss.
- **Rebuild the act-cache** — agent_nlp built it; pull from HF.
- **Wasteland imports** — `git show` only.

## Open questions for Han (agent owns — overwrite)

(none at provisioning.)
