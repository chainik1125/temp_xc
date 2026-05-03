<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_nlp; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_nlp
last_state_update: 2026-05-03T22:00:00Z
component: c3, c4
---

## Identity + mandate (Han owns — agents do not edit)

You are agent NLP, lead on the language-model components of the paper:
**C3 (sparse probing)** and **C4 (qualitative latents)**. Both are on
the same subject: `google/gemma-2-2b-it` layer 13 residual stream. C4
piggybacks on C3's activation cache.

Hardware: pod `2× H100`, pinned to **GPU 0**. Pod mode `persistent` —
`/workspace` survives stop/start, HF backup is optional but
recommended at session end. agent_em shares the pod on GPU 1.

Your **long-pole task** is the activation cache (~14 GB,
~3 H100-hours): 24K FineWeb sequences × 128 tokens, fp16, layer 13. As
soon as it's on HF (`han1823123123/temp-bench-data`), agent_steer can
unblock — they are gated on this. Push as soon as ready, don't wait
for downstream training.

C3 hypothesis (from `docs/components/c3.md`): TXC-pro matches the best
per-token SAE at k=5 and small seed-significant win at k=20. TXC-base
matches at every k.

C4 hypothesis: TXC-pro matches T-SAE on Top-256 cumulative SEMANTIC
Pareto. **One metric only** — drop pdvar and any paper-style probe
variants (decision pinned in `docs/components/c4.md`).

Open task-suite question (decisions.md "Non-decisions"): the C3 task
suite is Phase 5's 36-task vs Phase 7's 16-task PAPER subset. **Pre-
register a single suite before launch** and document the choice in
`docs/components/c3.md`. Default if undecided: SAEBench standard.

Locked decisions in scope: #1 (two TXCs — no hill-climbing), #4
(cross-branch reads via `git show`), #6 (HF repos), #7 (Bricken
opt-in; for C3/C4 you must run an A/B at 5k×1seed before adopting).

References:
- `agents/README.md` (your roster row + pod specs)
- `docs/components/c3.md` and `docs/components/c4.md`
- `docs/paper/architecture.md` (locked TXC spec)
- `decisions.md` (10 locked policy items)
- `PROTOCOL.md` § 11 (framework discipline), § 12 (GPU pinning)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: (not yet provisioned)**

- `git HEAD`: (set on first session)
- Last leaderboard append: (none yet)
- Last checkpoint saved: (none yet)
- Active GPU lock(s): none
- Recent decisions in scope: #1, #4, #6, #7
- In flight: nothing (provisioning pending)

## What I just did (agent owns — overwrite)

(Empty — agent_nlp not yet provisioned.)

## Next action (agent owns — overwrite)

**Pre-condition (Han owns)**: Han has already run
`bash scripts/bootstrap_runpod.sh` on this pod (interactive — prompts
for tokens; an agent cannot enter input). When you wake up, tokens
are already in `/workspace/.tokens/` and `purified/.venv/` exists. If
the smoke test below complains about missing tokens, **ping Han** —
do not try to populate them yourself.

1. `cd /workspace/temp_xc/purified`
2. `source scripts/set_agent_env.sh agent_nlp`
3. `bash scripts/agent_smoke_test.sh` (46/46 + expected gaps)
4. `git pull --rebase origin final`
5. Read `docs/components/c3.md` + `c4.md` end-to-end. Decide task
   suite (Option A SAEBench / Option B 16-task PAPER) and update
   c3.md before launch.
6. Port `temp_bench.data.nlp.cache_activations` from
   `origin/han-phase7-unification:src/data/` (search for the
   FineWeb activation cache pipeline; copy with header comment +
   commit-hash attribution per PROTOCOL.md § 2).
7. Build the activation cache for datasource
   `gemma_2_2b_it_l13_fineweb_24k128` (from `configs/datasources.yaml`).
   Expected: ~3 H100-hours, ~14 GB on disk.
8. Push the cache to HF `han1823123123/temp-bench-data` —
   **immediately** so agent_steer can unblock.
9. Port the 5 archs needed for C3: `topk_sae`, `tsae_paper`, `mlc`,
   `txc_base`, `txc_pro` (sources listed in `decisions.md` and
   `agent_paper/briefing.md` port table).
10. Train + eval cells via `runner.run_cell(...)`. Schema +
    eval_protocol_version validation will append rows to
    `results/leaderboard.jsonl`.

## Don't repeat (agent owns — overwrite)

- **Two TXCs only** (decision #1) — don't introduce a galaxy steering
  variant or a non-locked TXC; raise it in `docs/components/c3.md`
  first if you genuinely need to.
- **Wasteland imports** — code is on `origin/han-phase7-unification`,
  not in `final`. Use `git show`. Never `from src.architectures...`.
- **Bypass `runner.run_cell`** — it's the only writer to the
  leaderboard. Schema validation is mandatory.
- **Forget the cache push** — agent_steer waits on you. Push as soon
  as the cache is built; don't batch it with downstream training.
- **Hardcode hyperparameters** — anything paper-relevant goes in
  `configs/locked_archs.yaml` and `configs/datasources.yaml`. Edit the
  yaml, not the .py.

## Open questions for Han (agent owns — overwrite)

(none at provisioning — you'll add some after first session.)
