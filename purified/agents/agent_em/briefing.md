<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_em; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_em
last_state_update: 2026-05-03T22:00:00Z
component: c6
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent EM**. You own C6 only. Files you may edit:
- `agents/agent_em/briefing.md` (your own — agent-owned sections only)
- `docs/components/c6.md`
- `experiments/c6_em/`
- Code under `src/temp_bench/` that you author + commit (the Wang
  procedure runner under `temp_bench.case_studies.em`, Bricken
  trainer logic under `temp_bench.training.bricken`)
- `configs/datasources.yaml` — adding new C6 datasources is fine.

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

You are agent EM, lead on **C6: emergent misalignment** on
`Qwen/Qwen2.5-14B-Instruct` + finance LoRA organism (R1 + R32). The
component is in **status: pending-retest** in `docs/components/c6.md`.

Hardware: pod `2× H100`, pinned to **GPU 1**. Pod mode `persistent`.
agent_nlp shares the pod on GPU 0; you will not collide because
pinning is enforced. **Fallback**: if R32 OOMs the H100 (14B model +
LoRA at fp16 ≈ 28 GB so it should fit, but R32 may stress it), spin
up `agent_em_h200` (provisioned dormant — see `agents/README.md`).

Why the re-test: Dmitry's published Qwen-14B finance numbers
(`em_nanda_results_paper.md`) were plain TXC k=100, no Bricken,
no anti-dead — not a fair comparison vs SAE arditi which has 100k
training steps and dead-feature handling. With the brickenauxk_a8
recipe (Bricken + EMA-AuxK α=1/8 + dead-threshold 128k tokens),
TXC may close the +3.91 gap on R1 and the +12.58 gap on R32.

Decision tree (after R1 30k mid-α first re-run):
- gap ≤ 3 align → **Tied** — headline win
- gap 3–9 align → **Mixed** — note step-efficiency win on Qwen-7B medical
- gap > 9 align → **Honest negative** — back to original framing

Coordinate with **Dmitry on `origin/em-nanda`** — he is still active
on this component. Read `EM_NANDA_BRIEF.md` for his latest state
before launching. Don't merge his branch into `final`; read via
`git show` (decision #4).

Salvageable contributions (independent of headline outcome):
- **Bundle null is architecture-general**: both arches' k=30 bundles
  peak at align ≈ 41.3 on R32, falling 13–23 align points below
  single-feat champions. Falsifies "distributed misalignment by sum."
- **Bundle precision is architecture-specific**: SAE has k=30 < k=3 <
  single-feat (precision helps); TXC inverts (top-3 anti-correlate).

Locked decisions in scope: #2 (C6 reframe + bundle-null result), #4
(cross-branch reads), #6 (HF repos), #7 (Bricken opt-in — **C6 turns
it on by default**, you don't need an A/B; the recipe is justified by
Dmitry's Qwen-7B medical evidence).

References:
- `agents/README.md` (your roster row)
- `docs/components/c6.md` (full setup, decision tree, Wang 4-stage)
- `docs/paper/architecture.md` *Per-experiment training knobs* (Bricken)
- `decisions.md` (esp. #2, #7)
- `origin/em-nanda:docs/dmitry/results/em_features/EM_NANDA_BRIEF.md` (latest)
- `PROTOCOL.md` § 11 (framework), § 12 (GPU pinning)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: (not yet provisioned)**

- `git HEAD`: (set on first session)
- Last leaderboard append: (none yet)
- Last checkpoint saved: (none yet)
- Active GPU lock(s): none
- Recent decisions in scope: #2, #4, #6, #7
- In flight: nothing (provisioning pending)

## What I just did (agent owns — overwrite)

(Empty — agent_em not yet provisioned.)

## Next action (agent owns — overwrite)

**Pre-condition (Han owns)**: Han has already run
`bash scripts/bootstrap_runpod.sh` on this pod (interactive — prompts
for tokens; an agent cannot enter input) AND
`bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh agent_em`
to create your own clone. Tokens are in `/workspace/.tokens/` and
your venv exists at `/workspace/temp_xc_em/purified/.venv/`. If the
smoke test below complains about missing tokens, **ping Han**.

**Your clone path is `/workspace/temp_xc_em/`** (NOT
`/workspace/temp_xc/` — that's agent_nlp's primary clone). The
separate clone exists so two agents on the same pod don't collide
on `.git/index.lock` during pull-rebase. Tokens + HF cache are
shared via `/workspace/.tokens/` and `/workspace/hf_cache/`.

**Han launches you via `start_agent.sh`** (not bare `claude`):
```
bash /workspace/temp_xc_em/purified/scripts/start_agent.sh agent_em --fresh
```
Re-launches after disconnect drop the `--fresh` so the wrapper passes
`--continue` to claude — you resume your session instead of re-reading
the briefing. The wrapper sources `set_agent_env.sh` in Han's parent
shell so the GPU pin / `AGENT_NAME` / pod mode propagate into your
process. Bash tool calls don't share shell state, so YOU sourcing the
env in your first action is a no-op for subsequent commands.

1. `bash scripts/agent_smoke_test.sh` (verifies GPU pin etc.)
2. `git pull --rebase origin final`
3. Read `docs/components/c6.md` end-to-end (decision tree + Wang 4-stage).
4. Read Dmitry's latest:
   `git show origin/em-nanda:docs/dmitry/results/em_features/EM_NANDA_BRIEF.md`
   and any newer results he's pushed under
   `docs/dmitry/results/em_features/`.
5. Port from `origin/em-nanda` (with header-comment attribution):
   - `experiments/em_features/dead_feature_resample.py`
     → `temp_bench/training/bricken.py`
   - `experiments/em_features/run_training_txc_bricken_auxk.py`
     → integrate into `temp_bench` training path
   - `experiments/em_features/run_wang_procedure.py`
     → `temp_bench/case_studies/em.py`
6. Set up Qwen-14B-Instruct + finance LoRA loader. Verify hookpoint
   layer 24 `resid_post`, d_model=5120 (per c6.md).
7. **First cell** (the gap-close test): TXC-base + brickenauxk on
   R1 30k mid-α → run Wang 4-stage → Gemini judge over 8 prompts ×
   8 rollouts. Expected runtime: a few hours of training + eval.
8. Compare peak align vs SAE arditi 30k baseline (95.16 from
   `em_nanda_results_paper.md`). Apply decision tree to decide
   paper framing for C6.

## Don't repeat (agent owns — overwrite)

- **Plain TXC k=100** without Bricken — that's the comparison Dmitry
  already published; we're explicitly re-testing with the better
  recipe.
- **Merge `em-nanda` into `final`** — decision #4 forbids it.
  Cross-branch reads only.
- **Forget Gemini judge variance** — σ ≈ 6 align at n=64. Don't claim
  a win on a sub-σ gap.
- **Hardcode the brickenauxk recipe** — expose it as a `BrickenConfig`
  in `temp_bench/training/bricken.py`; C6's `experiments/c6_em/run.py`
  passes the recipe explicitly so the paper can disclose it.
- **Bypass `runner.run_cell`** — single canonical pathway.

## Open questions for Han (agent owns — overwrite)

(none at provisioning.)
