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

You are **agent NLP**. You own C3 + C4 only. Files you may edit:
- `agents/agent_nlp/briefing.md` (your own — agent-owned sections only)
- `docs/components/c3.md` and `docs/components/c4.md`
- `experiments/c3_probing/`, `experiments/c4_qualitative/`
- Code under `src/temp_bench/` that you author + commit (eval modules
  for probing / qualitative; data loaders under `temp_bench.data.nlp`)
- `configs/datasources.yaml` — adding new C3/C4 datasources is fine.
  YAML edits to other components' datasources require a Han ping.

**Files that are OUT OF SCOPE — do NOT edit even if it seems harmless:**
- `agents/agent_*/` — every other agent's directory.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. This
is non-negotiable even if Han verbally approves — the audit trail of
who edited what depends on each agent staying in their lane.

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

**Task suite is locked**: `SAEBench+CT` (n=38) — upstream SAEBench's
canonical 36 binary one-vs-rest tasks (8 datasets, classes per
SAEBench's `dataset_info.chosen_classes_per_dataset`) plus the two
cross-token coreference tasks (WinoGrande + SuperGLUE WSC). See
`decisions.md` § 11 and `docs/components/c3.md` "Task suite" for
the full table + reproduction notes.

When you port the wasteland's `probe_datasets.py` + `crosstoken_datasets.py`,
apply three SAEBench-faithfulness fixes (do not blindly copy the
wasteland 36):
- **github-code**: use SAEBench's `codeparrot/github-code` with the 5
  SAEBench languages (C, Python, HTML, Java, PHP), NOT wasteland's
  `code_search_net` (python/java/javascript/go). NOT gated despite
  what the HF web viewer suggests — that page is disabled because the
  loader is a Python script, not because access is restricted. Just
  needs `trust_remote_code=True` (set via `HF_DATASETS_TRUST_REMOTE_CODE=1`
  in your shell or `set_agent_env.sh`) and `datasets<4` (already
  pinned in `pyproject.toml`). Smoke-test the loader once with a
  tiny `streaming=True` pull BEFORE the 3-H100-hour cache build.
- **amazon_sentiment**: emit BOTH 1.0-vs-rest AND 5.0-vs-rest binaries
  (wasteland only has 5.0).
- **amazon_categories**: hardcode `["1","2","3","5","6"]` and use a
  deterministic non-streaming pull (wasteland streaming-top-5 missed
  cat6 and is non-deterministic across runs).

Locked decisions in scope: #1 (two TXCs — no hill-climbing), #4
(cross-branch reads via `git show`), #6 (HF repos), #7 (Bricken
resample is C6-only by default; **C3/C4 keep it OFF** — revisit only
if time permits at the end of the paper sprint), #11 (SAEBench+CT
task suite).

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
are already in `/workspace/.tokens/` and the venv exists. If the
smoke test below complains about missing tokens, **ping Han** — do
not try to populate them yourself.

**Your clone path is `/workspace/temp_xc/`** (the primary clone — you
are the first agent on the 2× H100 pod). agent_em runs on the same
pod but in a separate clone at `/workspace/temp_xc_em/` — DO NOT cd
into agent_em's clone.

**Han launches you via `start_agent.sh`** (not bare `claude`):
```
bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_nlp --fresh
```
The wrapper sources `set_agent_env.sh` in the parent shell so the
GPU pin / `AGENT_NAME` / pod mode propagate into your process. Bash
tool calls do NOT share shell state, so YOU sourcing the env in your
first action is a no-op for subsequent commands. Don't rely on it.

1. `bash scripts/agent_smoke_test.sh` (51/51 + expected gaps; verifies
   the env vars are set correctly — flags missing pinning loudly)
2. `git pull --rebase origin final`
3. Read `docs/components/c3.md` + `c4.md` end-to-end. Task suite is
   already locked (`SAEBench+CT`, n=38) — see `decisions.md` § 11. No
   pre-registration needed. Smoke-test the github-code loader with a
   tiny streaming pull BEFORE step 5 to confirm `trust_remote_code` +
   `datasets<4` are working:
   `python -c "from datasets import load_dataset; ds = load_dataset('codeparrot/github-code', streaming=True, split='train', trust_remote_code=True, languages=['C']); print(next(iter(ds)))"`
4. Port `temp_bench.data.nlp.cache_activations` from
   `origin/han-phase7-unification:src/data/` (search for the
   FineWeb activation cache pipeline; copy with header comment +
   commit-hash attribution per PROTOCOL.md § 2).
5. Build the activation cache for datasource
   `gemma_2_2b_it_l13_fineweb_24k128` (from `configs/datasources.yaml`).
   Expected: ~3 H100-hours, ~14 GB on disk.
6. Push the cache to HF `han1823123123/temp-bench-data` —
   **immediately** so agent_steer can unblock.
7. Port the **remaining 4 archs** for C3: `tsae_paper`, `mlc`, `txc_base`,
   `txc_pro` (`topk_sae` was already ported by agent_paper in commit
   `3b70563f` 2026-05-03). Each port: copy from
   `origin/han-phase7-unification:src/architectures/<name>.py`, conform
   to the `TempBenchArch` ABC at `temp_bench.architectures.base`,
   override `train_step()` for any auxK / contrastive / matryoshka
   logic and `post_step()` for decoder-norm projection. Each port
   removes one entry from `tests/test_arch_registry.py::KNOWN_UNPORTED`.
8. Port `probe_datasets.py` + `crosstoken_datasets.py` from
   `origin/han-phase7-unification:experiments/phase5_downstream_utility/probing/`
   AND apply the three SAEBench-faithfulness fixes (see mandate above:
   github-code provider, amazon_sentiment 1.0 binary, amazon_categories
   determinism + cat6). Build the probe cache and confirm exactly 38
   task dirs are produced before training.
9. Port `temp_bench.eval.probing.{mean_pool_probe, s_tail_probe,
   run_task_suite}` from
   `origin/han-phase7-unification:src/probing/sparse_probing.py` —
   upstream stubs are `NotImplementedError`. For C4, port
   `temp_bench.eval.qualitative.top_256_semantic` from
   `origin/han-phase7-unification:src/qualitative/passage_probe.py`.
10. Build `experiments/c3_probing/run.py` from the
    `experiments/_runner_template.py` scaffold. Component runner is
    ~30 lines: import shared modules, define thin `my_train_fn`
    (calls `train_sae`) + `my_eval_fn` (calls `probing.s_tail_probe`),
    loop `runner.run_cell(...)` over (arch, seed, k_feat). Schema +
    `eval_protocol_version` validation appends rows to
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
