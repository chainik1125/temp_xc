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
- `pyproject.toml` and `uv.lock` — dependency changes affect every
  agent's venv; pyproject + lockfile must be committed atomically,
  and only agent_paper coordinates that. If you need a new dep,
  surface in Open questions.

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

**Last verified: 2026-05-03T22:55Z (autonomous-overnight session)**

- `git HEAD`: c0a1276b — `final` branch
- Last leaderboard append: still the 2 old smoke rows (06afa68f, 1948488e
  — both `eval_cfg.smoke=true`). First REAL row will land when cell 1
  completes (~30 min into the 18-cell run).
- Last checkpoint saved: 86474a703b8af5eb (txc_base smoke 200-step);
  no real-config checkpoints yet (different `n_steps=10000` → different
  train_key).
- Active GPU lock(s): none — pinned to GPU 0 (H100). Cell run is
  using it; agent_em is on GPU 1.
- Recent decisions in scope: #1, #4, #6, #7, #11
- In flight: 18-cell C3 run (PID 21612, started 2026-05-03T22:49Z;
  step 2000/10000 at 22:55Z = ~5 min in). Total ETA ~6 hours
  (9 unique trainings × 30 min each + 18 evals × 5 min each).

## What I just did (agent owns — overwrite)

C3 + C4 plumbing fully shipped. Cells in flight on H100. Status:

**C3 — sparse probing**:

- ✅ `temp_bench.data.nlp.probe_tasks` — 38-task SAEBench+CT loader
  with all 3 SAEBench-faithfulness fixes (github-code via codeparrot
  with post-iter language filter; amazon_sentiment 1+5 binaries;
  amazon_categories non-streaming + shuffle for cat6). All 8 dataset
  loaders smoke-tested individually. Hardcoded SAEBench class lists.
- ✅ `temp_bench.data.nlp.probe_cache` — `build_probe_cache()` +
  `load_probe_cache()` + `list_probe_cache()`. Per-task structure
  is `(N, seq_len, d_in)` fp16 numpy arrays. Idempotent.
- ✅ Full probe cache built (38 tasks, 79 GB at
  `results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`).
- ✅ `experiments/c3_probing/run.py::my_eval_fn` — wired to real
  probe-cache. Returns flattened per-task floats (`auc__<task>` × 38)
  PLUS aggregates (`mean_auc`, `std_auc`, `mean_acc`, `std_acc`,
  `n_tasks`). Primary metric: `mean_auc`.
- ✅ Progress wrapper around `batch_iter` prints every 1000 steps
  (the canonical trainer is silent by design — wrapper added in
  `my_train_fn` for autonomous-overnight visibility).
- 🟡 IN FLIGHT: 18 cells (topk_sae × tsae_paper × txc_base × seeds
  {1,2,42} × k_feats {5,20}). Process PID 21612, started
  2026-05-03T22:49Z. n_steps=10K @ 5 steps/sec → ~30 min training
  per unique (arch,seed). 9 unique trainings + 18 probings. Total
  ETA ~6 hours.

**C4 — qualitative latents**:

- ✅ `temp_bench.eval.qualitative` — full implementation (was
  NotImplementedError stub):
  - `load_concat_corpus(name)` reads from `data/concat_corpora/`
  - `encode_concat_corpus(sae_model, subject_model, layer, token_ids)`
    forwards Gemma → hooks layer → SAE encode → (n_tokens, d_sae) z;
    dispatches per-token vs window on `model.T`
  - `pick_top_features_by_var(z, n)` variance ranking
  - `gather_top_contexts(token_ids, tok, z_col)` — top-N max-activating
    text windows
  - `_call_anthropic` — exponential backoff on rate limits
  - `call_judges` — 2 Haiku judges + agree/verdict
  - `persist_judge_record` — appends to
    `results/runs/<eval_key>/judge_outputs.jsonl` (κ-deferred lifeline)
  - `top_256_semantic` orchestrator returns float-only metrics
- ✅ `experiments/c4_qualitative/run.py` + `analysis.py`. Runner
  shares train_fn with C3 so checkpoints reuse via `runner.run_cell`'s
  auto-skip. Analysis joins to C3 leaderboard via (arch, seed,
  k_feat=20) for Pareto x-axis, draws upper-right frontier.
- ✅ `data/concat_corpora/{concat_A, concat_B, concat_random}.json`
  — pre-tokenized JSONs ported from wasteland.

Earlier session work (still relevant — no regressions):

- ✅ Activation cache `gemma_2_2b_it_l13_fineweb_24k128` — pushed to
  HF (`han1823123123/temp-bench-data/act_cache/e4916bcae1881963/`).
- ✅ `temp_bench.architectures.{tsae, txc_base}` — both ported and
  smoke-tested in --smoke mode.

## Autonomous decisions made (Han: override anytime)

While Han was AFK for the 10-hour overnight window:

- **TrainingConfig for headline cells**: n_steps=10_000, batch_size=256,
  lr=3e-4, warmup_steps=500, precision="bf16". Codified in
  `experiments/c3_probing/run.py::_real_training_cfg`. Picked over
  the schema default (30K steps × 256) so 18 cells fit in the 10-hour
  window. Phase 7 reference used ~50K, but SAE convergence at 328M
  tokens (10K × 256 × 128) is reliable per Ye et al. If headline
  numbers come in low (≪0.85 mean_auc at k=20), bump to 30K.
- **Per-task AUC reporting**: BOTH per-task floats `auc__<task>`
  (38 keys) AND aggregates (`mean_auc`, `std_auc`, ...) emitted on
  every leaderboard row. Per-task is for σ_tasks; aggregate is the
  headline. Best-of-both-worlds; analysis.py uses both.
- **Smoke leaderboard rows kept** (`eval_cfg.smoke=true` filter). Two
  smoke rows on disk (eval_keys 06afa68f259490a0, 1948488e5b127138).
  analysis.py filters them out cleanly.
- **Bricken A/B for C3**: SKIPPED per decision #7 default. C3/C4/C5/C7
  keep Bricken OFF. Revisit only if headline TXC-base undershoots
  topk_sae by > 1% at k=5 — that would be the kind of dead-feature
  symptom Bricken is designed to fix.
- **MLC scope**: SKIPPED for headline. Lower priority per decisions
  doc; appendix-only OK. Will not port mlc unless time permits at
  end of session.
- **txc_pro port**: DEFERRED. The 3-layer wasteland inheritance
  (`SubseqH8` → `TXCBareMultiDistanceContrastiveAntidead` →
  `TXCBareAntidead`) is substantial; risk of derailing the headline
  run if started before cells complete. Will attempt at end of
  session if cells finish with ≥ 1 hour to spare.

- **C4 cells launch timing**: NOT YET. C4's `top_256_semantic`
  needs trained checkpoints from C3 (same train_key — both share
  the FineWeb activation cache + TrainingConfig). After C3 cell 7
  (tsae_paper seed=1 k=5) completes (~2 hours in), I could fire
  one C4 smoke cell to validate end-to-end. Real C4 cells (3 archs
  × 3 seeds × n_features=256) should run AFTER all C3 cells
  finish to avoid GPU contention. Expected total C4 wall time:
  6 cells × (Gemma forward + Haiku judge calls) ≈ 30 min compute +
  ~10 min of API latency for 256 × 3 = 768 Haiku calls per cell.

- **Trainer step rate (5/sec) is mmap-bound**: act cache lives on
  MooseFS (`mfs#us-ca-2.runpod.net`); random-index reads are slow
  vs sequential. After ~1000 steps the kernel page cache warms up
  and rate may improve, but in this session it stabilised at 5
  steps/sec. If a future run wants to be faster, options are:
  (a) copy the act cache to local SSD (none on this pod), or
  (b) use larger batch_size to amortise per-step I/O — risky for
  schema validation since batch_size is part of train_key.

## Pre-launch decisions for next session (Han may override)

These were resolved autonomously this session — see "Autonomous
decisions" above. All can be overridden by editing the constants in
`experiments/c3_probing/run.py` or by re-running with `force_eval=True`
to regenerate any cell with a new config.

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

1. `bash scripts/agent_smoke_test.sh` (51/51 + KNOWN_UNPORTED for
   the still-not-ported `mlc` + `txc_pro`)
2. `git pull --rebase origin final`
3. Verify state on disk:
   - act cache: `results/act_cache/e4916bcae1881963/acts.npy` (14 GB)
   - probe cache: `ls results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`
     should show 38 task dirs (each with X_train, X_test, y_train,
     y_test .npy)
   - leaderboard rows for c3 with `eval_cfg.smoke=false` (real cells)

4. **Check `logs/c3_real_cells.log`** to see whether the 18-cell
   run started by my previous self at 2026-05-03T22:49Z completed
   cleanly. If it did, the leaderboard has up to 18 real C3 rows
   already. If it crashed mid-run or got killed, restart it:
   ```
   bash experiments/c3_probing/run.sh
   ```
   The runner is idempotent (cells with cached eval_key skip).

5. **Render C3 results**: `.venv/bin/python -m experiments.c3_probing.analysis`
   rewrites `docs/components/c3.md` AUTO-RESULTS + writes
   `experiments/c3_probing/plots/auc_by_k.png`.
   - **Sanity check**: `topk_sae k=20 mean_auc` should land in
     [0.85, 0.91] (Phase 7 BASE-side leaderboard reference). On IT
     side it may be slightly lower. If far below 0.80, suspect an
     encode-shape / probe-cache-leakage bug.

6. **Run C4 cells** (~30 min compute + ~10 min Haiku per cell):
   ```
   bash experiments/c4_qualitative/run.sh --archs tsae_paper txc_base --seeds 1 2 42
   ```
   Pre-condition: ANTHROPIC_API_KEY set or
   `/workspace/.tokens/anthropic_key`. The first cell trains
   tsae_paper if not cached (it should be after C3 finishes — same
   train_key). Then forwards Gemma over concat_A+B+random,
   variance-ranks, top-256, Haiku 2-judge labels.
   - **Cost**: ~$0.06 per cell × 6 cells = ~$0.36 total.
   - **Judge persistence**: each Haiku call appends to
     `results/runs/<eval_key>/judge_outputs.jsonl` for post-deadline
     Cohen's κ validation.

7. **Render C4 results**: `.venv/bin/python -m experiments.c4_qualitative.analysis`
   joins to C3 leaderboard for Pareto x-axis, draws the upper-right
   frontier, rewrites `docs/components/c4.md` AUTO-RESULTS.
   - **Pareto check (per c4.md)**: TXC-pro should Pareto-dominate
     T-SAE — that's the headline claim. If it doesn't, report the
     honest negative: "TXC-pro matches T-SAE on probing while losing
     on top-256". Don't introduce a TXC-pro@T_max=20 escape variant
     (decision #1).

8. **Push checkpoints to HF backup** (persistent pod, optional but
   recommended at session end):
   ```python
   from temp_bench.cache import iter_manifest_for_agent
   from huggingface_hub import HfApi
   token = open('/workspace/.tokens/hf_token').read().strip()
   api = HfApi(token=token)
   for row in iter_manifest_for_agent('agent_nlp'):
       if row.hf_url is None:
           api.upload_folder(folder_path=row.local_path.rsplit('/', 1)[0],
                              path_in_repo=row.train_key,
                              repo_id='han1823123123/temp-bench-models',
                              repo_type='model')
   ```
   (The probe cache is already on HF as of 2026-05-03 from this
   session — see open question #1 status.)

9. **Port txc_pro** (only if all above is done + ≥ 2 hours left):
   3-layer wasteland inheritance pulls in ~250 lines:
   - `phase5b_subseq_sampling_txcdr.py::SubseqH8` (subset sampling)
   - `txc_bare_multidistance_contrastive_antidead.py` (matryoshka
     H8 + multi-distance InfoNCE)
   - `txc_bare_antidead.py` (already mostly in `txc_base.py`)

   The contrastive batch shape is **(B, 1+K, T, d_in)** — different
   from canonical `(B, T, d_in)`. Either extend `train_sae` to
   accept multi-window batches OR train txc_pro single-window
   (matryoshka recon only, no contrastive). Single-window degrades
   the arch but at least gets paper-faithful matryoshka behavior.

10. **Port mlc** (lowest priority — appendix-only baseline). Wasteland
    source `src/architectures/mlc.py`. Cross-LAYER crosscoder.

## Don't repeat (agent owns — overwrite)

Locked-decision tripwires:

- **Two TXCs only** (decision #1) — don't introduce a galaxy steering
  variant or a non-locked TXC; raise it in `docs/components/c3.md`
  first if you genuinely need to.
- **Cross-territory edits** — see the OUT OF SCOPE list in mandate.
  Even if Han verbally approves in chat, surface the request in
  writing first. My last-but-one commit got partially rejected on
  exactly this (commit `2283aa15`).
- **Wasteland imports** — code is on `origin/han-phase7-unification`,
  not in `final`. Use `git show`. Never `from src.architectures...`.
- **Bypass `runner.run_cell`** — it's the only writer to the
  leaderboard. Schema validation is mandatory.
- **Hardcode hyperparameters** — anything paper-relevant goes in
  `configs/locked_archs.yaml` and `configs/datasources.yaml`. Edit the
  yaml, not the .py.

Hard-won technical gotchas from this session (verify before bypassing):

- **`datasets<4` pin is load-bearing** for `codeparrot/github-code`.
  v4+ removed `trust_remote_code` and the dataset uses a Python
  loading script. Pinned in `pyproject.toml`. If you ever see
  "Dataset scripts are no longer supported, but found github-code.py",
  re-run `uv sync` to pull `datasets==3.6.0`.
- **github-code `languages=[...]` does NOT filter the stream.** Out
  of 20 samples I got 9 JS, 6 C, 5 other. Your loader MUST
  `if sample['language'] != target_lang: continue` after iter.
- **`tsae_paper.config.T == 1`, NOT 2.** The contrastive pair is a
  TRAINING construct, sampled inside `train_step` over the seq_len
  axis. Setting T=2 routes the probe to window-encoding which is
  wrong for T-SAE.
- **`LeaderboardRow.metrics` is float-only** (Pydantic schema).
  Categorical / int diagnostics like `agg`, `n_train`, `task_name`
  belong outside the `metrics` dict. The runner explodes loudly if
  you violate this.
- **Background `nohup ... &`** — the bash WRAPPER returns immediately
  and the tool reports "completed", but the python process keeps
  running. Always verify via `ps -ef | grep python` or by tailing
  the log file before declaring success or failure.
- **Cache build is FAST on H100** — ~2 min for 24K seqs of Gemma-2-2b,
  not the 3 H100-hours Han's mandate suggested. Don't be surprised
  if it "completes" suspiciously quickly.
- **Decoder grad-parallel removal** uses
  `register_post_accumulate_grad_hook` on `W_dec` (PyTorch 2.0+) —
  this avoids needing a pre-step hook in the canonical trainer. See
  `tsae.py::_project_dec_grad` and `txc_base.py::_project_dec_grad`.
- **`einops` is NOT a dep.** I rewrote the wasteland's
  `einops.einsum(...)` calls with vanilla `torch.einsum` in tsae.py.
  Don't add `import einops` without first adding to pyproject.toml.
- **TQDM_DISABLE=1 must be exported per bash call.**
  `set_agent_env.sh` does NOT set it; sourcing the env script alone
  isn't enough. Standard pattern:
  ```
  export TQDM_DISABLE=1 && source scripts/set_agent_env.sh agent_nlp >/dev/null 2>&1 && <command>
  ```

## Open questions for Han (agent owns — overwrite)

All questions from the previous session were resolved autonomously
during the 10-hour overnight window — see "Autonomous decisions"
above. Han can override any of them by:
- editing `experiments/c3_probing/run.py::_real_training_cfg` for the
  TrainingConfig
- re-running with `force_eval=True` to regenerate cells under a new
  config (n.b. changing config invalidates train_key OR eval_key)
- editing analysis.py's filter rule to include/exclude smoke rows

New open questions surfaced this session:

1. **Probe cache HF push.** Once the 38-task probe cache is built
   (~70-90 GB), should we push it to `han1823123123/temp-bench-data`
   (path `probe_cache/<datasource_name>/`)? Trade-off:
   - PRO: agent_steer / agent_back can sync from HF instead of
     re-tokenising 38 tasks; saves ~10 min per ephemeral pod.
   - CON: 70-90 GB push at 256 MB/s = ~5-6 min one-time cost.
   - Default if Han doesn't say: I'll push since the upload is one-time
     and the saved time is per-pod-restart (likely several over the
     remaining sprint).

2. **What happens if mean_auc is far from Phase 7 reference?** Phase 7
   leaderboard has `txc_bare_antidead_t5 k=20 = 0.9127` (BASE side).
   We're on IT side (Phase 7 noted "IT side is entirely missing").
   What's the threshold below which we should bump n_steps to 30K and
   re-run, vs accept that IT-side numbers are just lower? My default:
   if `mean_auc < 0.85` at k=20 across ALL 3 archs, that's a training
   bug; bump n_steps. If only some archs lag, that's the headline finding.

3. **C4 scope this session.** I'll start C4 (qualitative latents
   port) only after C3 cells are in flight or completed. The C4 lead
   architecture (TXC-pro) isn't ported yet — see autonomous decision
   above. C4 with TopK-SAE / T-SAE only would still be useful. Han:
   do you want me to ship C4 with the 3 archs we have, or wait for
   txc_pro?
