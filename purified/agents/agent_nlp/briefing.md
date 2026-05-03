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

**Last verified: 2026-05-03T22:00Z (this session)**

- `git HEAD`: 93dcaf9c (after rebase) — `final` branch
- Last leaderboard append: 06afa68f259490a0 (smoke cell — topk_sae
  seed=1 k_feat=5; AUC=0.477 on synthetic labels — pipeline-only
  validation, not a paper number)
- Last checkpoint saved: train_key 45124717553436e6 (topk_sae @ 200
  steps on the FineWeb cache); also one for txc_base
- Active GPU lock(s): none
- Recent decisions in scope: #1, #4, #6, #7, #11
- In flight: nothing — pipeline shipped end-to-end, ready to extend
  with real probe data

## What I just did (agent owns — overwrite)

End-to-end pipeline now shippable for C3. Status:

- ✅ Built `gemma_2_2b_it_l13_fineweb_24k128` activation cache
  (24K seqs × 128 tok × 2304 d_in fp16 → 14.2 GB on H100 in ~2 min).
  Pushed to `han1823123123/temp-bench-data/act_cache/e4916bcae1881963/`
  — agent_steer can sync_from_hf and unblock.
- ✅ `temp_bench.data.nlp.{build_activation_cache, batch_iter_from_act_cache}`
  — clean port from wasteland (no wandb / sweep / CLI cruft).
- ✅ `temp_bench.architectures.tsae.TSAEPaper` — faithful Ye et al.
  port (matryoshka BatchTopK + AuxK + temporal contrastive + threshold
  inference).
- ✅ `temp_bench.architectures.txc_base.TXCBase` — vanilla TopK
  temporal crosscoder + tsae_paper anti-dead stack.
- ✅ `temp_bench.eval.probing.{mean_pool_probe, s_tail_probe, run_task_suite}`
  — implementations replace upstream stubs. Per-token vs window
  aggregation dispatched on `model.T`.
- ✅ `experiments/c3_probing/run.py` — thin runner from the template.
  Verified end-to-end in `--smoke` mode (synthetic probe labels) for
  both per-token (topk_sae) and window (txc_base) archs. Idempotency
  verified (cached re-runs hit `[CACHED]`).
- 🟡 DEFERRED: `txc_pro` port — chain-inheritance is substantial
  (`SubseqH8` → `TXCBareMultiDistanceContrastiveAntidead` → `TXCBareAntidead`).
  Spec: subseq encoder + matryoshka H8 + multi-distance contrastive.
  Source: `origin/han-phase7-unification @ 94119bc0:src/architectures/phase5b_subseq_sampling_txcdr.py`
  (and the two parent files).
- 🟡 DEFERRED: `mlc` port — MLC is a baseline, lower priority.
- 🟡 DEFERRED: `probe_datasets.py` + `crosstoken_datasets.py` ports —
  needed for REAL probing (not just smoke). Without them, the runner
  only works in `--smoke` mode.

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

1. `bash scripts/agent_smoke_test.sh` (51/51 + expected gaps for the
   6 archs that remain unported in `KNOWN_UNPORTED`)
2. `git pull --rebase origin final`
3. Verify the activation cache is intact at
   `results/act_cache/e4916bcae1881963/` (~14 GB). If missing,
   rebuild via the one-liner that worked previously:
   ```
   .venv/bin/python -c "from temp_bench.data import build_activation_cache; build_activation_cache('gemma_2_2b_it_l13_fineweb_24k128', batch_size=64)"
   ```
4. **Port `probe_datasets.py` + `crosstoken_datasets.py` from
   `origin/han-phase7-unification:experiments/phase5_downstream_utility/probing/`**
   into a new `temp_bench.data.nlp.probe_tasks` module. Apply the three
   SAEBench-faithfulness fixes (see mandate above):
   github-code → `codeparrot/github-code` (5 langs), amazon_sentiment
   → both 1.0+5.0 binaries, amazon_categories → hardcode classes
   `["1","2","3","5","6"]` + non-streaming pull. Plus add winogrande +
   wsc from crosstoken_datasets. Confirm **exactly 38 tasks**.
5. Build a `temp_bench.data.nlp.build_probe_cache(datasource_name,
   tasks)` that runs gemma forward over each task's texts and writes
   `results/probe_cache/<datasource_name>/<task_name>/{X_train.npy,
   y_train.npy, X_test.npy, y_test.npy}`.
6. Update `experiments/c3_probing/run.py::my_eval_fn` to load real
   probe-cache for the task list when `smoke=False`. Replace the
   `NotImplementedError` branch with a `temp_bench.eval.probing.run_task_suite`
   call over the cached tasks.
7. Run the real cells: 3 archs (topk_sae, tsae_paper, txc_base) ×
   3 seeds × 2 k_feats = 18 cells. Each cell trains 10K steps on the
   FineWeb cache (~5 min on H100), then probes 38 tasks (~2 min).
   Total: ~2 hours. Verify leaderboard rows are valid + AUCs are sane
   (topk_sae k=20 should be in the 0.85-0.91 range per Phase 7).
8. **Port txc_pro** (3-layer inheritance — pulls in ~250 lines from
   the wasteland). After ports, re-run cells with txc_pro included
   (force_train=False so existing cells skip).
9. **Port mlc** if time permits. Lower priority.
10. Port `temp_bench.eval.qualitative.top_256_semantic` for C4. Use
    cached SAE checkpoints from C3 (same act_cache_key) so no
    retraining needed.
11. Build `experiments/c3_probing/analysis.py` + `experiments/c4_qualitative/analysis.py`
    that aggregate the leaderboard rows and rewrite the AUTO-RESULTS
    blocks of the component docs via `temp_bench.report.render(...)`.

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
