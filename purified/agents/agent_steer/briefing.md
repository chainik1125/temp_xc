<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_steer; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_steer
last_state_update: 2026-05-03T23:30:00Z
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
- `pyproject.toml` and `uv.lock` — dependency changes affect every
  agent's venv; pyproject + lockfile must be committed atomically,
  and only agent_paper coordinates that. If you need a new dep,
  surface in Open questions.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.

### Han decisions 2026-05-04 PM (CRITICAL — TrainingConfig re-issued per Phase 5)

A "batch=256 → 1024 (A40 components)" cross-agent directive was issued
earlier today (commit a9200560) and reverted (commit 0beae2bf). It
selectively bumped contrastive archs — Han caught that as unfair to
non-contrastive baselines. **The new directive is Phase-5-faithful,
applied uniformly across all archs and all pods.** Read `decisions.md`
§ 12 in full before running anything. Gist:

- **`TrainingConfig` defaults are now**: `batch_size=1024`, `n_steps=25_000`,
  `plateau_early_stop=False` (disabled — see below). Just default-construct
  `TrainingConfig()` in your runner — no per-component overrides for these
  knobs.
- **Uniform across pods**: A40 (you) + H100 (agent_nlp/agent_em) both
  at batch=1024. Phase 5's empirically validated batch (summary.md:250)
  plus the SAE-literature-standard fixed-step schedule (T-SAE §4.1, TFA
  App. B.1, GemmaScope). Identical config across all archs being compared.
- **Plateau-stop is OFF; 25K cap is binding for every cell.** The schema's
  plateau detection (absolute max-min over 5K window) is cross-arch unfair
  because archs at different loss scales would trigger at different points.
  Every cell trains to the 25K cap; fairness mechanism is "exactly 25.6M
  tokens per arch."
- **If you observe loss still descending steeply at step 25K** (e.g.,
  final-1K-step drop > 5% of loss value), surface that as a comment on
  your run — the cap may need to be bumped uniformly across all archs.
- **Cache hygiene**: `batch_size` + `n_steps` are in the `train_key` hash
  (`src/temp_bench/config.py:181-193`). New cells get fresh keys
  automatically. Old batch=256 cells stay in `results/leaderboard.jsonl`
  for diff comparison — **when rendering AUTO-RESULTS, filter for new
  rows only** (e.g., `training_cfg.batch_size == 1024`).

**Your specific re-run**: 9-cell C5 sweep (3 archs × 3 seeds). Re-train
TXC-base + TXC-pro + T-SAE at the new defaults, re-run V7 tiled-broadcast
steering protocol, re-run coh-vs-success curves with the new
`peak_success_grade_at_coh_τ` headline metric (already in v1.0.1). VRAM
on A40 at batch=1024 + d_in=2304 (Gemma) should be fine; if you observe
OOM, try `precision="fp16"` first before reducing batch.

### Han decisions 2026-05-04 (NEW — C5 metric mismatch caught)

**The C5 headline metric was wrong** — Han caught it from the c5.md
plot ("all the points horizontally aligned, success rates tiny"
compared to wasteland's `unified-pareto.md` peak15 numbers ~1.5).

Two metrics, different semantics:

- **Old (your implementation, faithful to my buggy spec)**:
  ``success_at_coh_<τ>`` = mean(success_grade ≥ 2 | coh_grade ≥ τ),
  averaged over strengths. **Binary fraction in [0, 1].**
- **New (wasteland-comparable, now headline)**:
  ``peak_success_grade_at_coh_<τ>`` = for each strength, mean
  success_grade (0-3 continuous) over generations with coh ≥ τ;
  take MAX over strengths. **Continuous in [0, 3].**

The old metric collapses across coh thresholds because nearly all
success ≥ 2 events also have coh ≥ 2.0 — your numbers
``success_at_coh_1.75 == success_at_coh_2.0`` for every arch confirm
this. The new metric preserves dynamic range, comparable to
wasteland anchors (1.133 / 0.411 etc.) and the T-SAE paper § B.2 0-3
scale.

**What I (agent_paper) already did this session:**
1. Added ``peak_success_grade_at_coh_<τ>`` to ``coh_success_curves``
   in ``temp_bench.case_studies.steering`` — future cells emit it
   automatically.
2. Added ``mean_success_grade_at_coh_per_strength`` (per-strength
   continuous means) so the per-cell ``metrics.json`` retains the
   data needed for the peak metric.
3. Added a backfill helper:
   ``temp_bench.case_studies.steering.reaggregate_from_judge_outputs(
   judge_outputs_jsonl_path)`` — reads a cell's persisted judge
   calls and returns the new flat metrics dict. Use this to backfill
   existing cells without re-judging.
4. Updated ``experiments/c5_steering/analysis.py`` to render BOTH
   metrics in the AUTO-RESULTS block (peak-grade as headline when
   present; binary fraction always as supplementary). Re-rendered
   ``docs/components/c5.md``.
5. Updated c5.md "Metric" subsection to spec the new headline.

**What you (agent_steer) need to do this session:**

1. **Backfill the existing 9 cells.** Your judge_outputs.jsonl files
   are on the A40 pod (and pushed to HF via push_run_dir.py). For
   each c5 leaderboard row's eval_key, read the local
   ``results/runs/<eval_key>/judge_outputs.jsonl``, run
   ``reaggregate_from_judge_outputs(...)``, and emit a NEW
   leaderboard row with the new metrics + a bumped
   ``EVAL_PROTOCOL_VERSION`` (e.g. "1.0.1") so it doesn't collide
   with the old. The old rows stay in the leaderboard for
   reproducibility / diff.

   Sketch:
   ```python
   from pathlib import Path
   from temp_bench.cache import append_leaderboard, leaderboard_path
   from temp_bench.case_studies.steering import reaggregate_from_judge_outputs
   from temp_bench.report import query_leaderboard
   from temp_bench.schemas import LeaderboardRow
   import json, datetime, hashlib

   for r in query_leaderboard(component="c5"):
       jpath = Path("results/runs") / r.eval_key / "judge_outputs.jsonl"
       if not jpath.exists():
           continue
       new_metrics = reaggregate_from_judge_outputs(jpath)
       new_eval_key = hashlib.sha256(
           f"{r.eval_key}_v1_0_1".encode()).hexdigest()[:16]
       append_leaderboard(LeaderboardRow(
           eval_key=new_eval_key,
           train_key=r.train_key,
           act_cache_key=r.act_cache_key,
           component="c5",
           arch=r.arch,
           arch_version=r.arch_version,
           seed=r.seed,
           datasource=r.datasource,
           eval_protocol_version="1.0.1",
           eval_cfg={**r.eval_cfg, "rebuild_from": r.eval_key,
                     "metric_set": "v1_0_1_with_peak_grade"},
           metrics=new_metrics,
           primary_metric="peak_success_grade_at_coh_1.75",
           agent="agent_steer",
           ts=datetime.datetime.now(datetime.timezone.utc).strftime(
               "%Y-%m-%dT%H:%M:%SZ"),
       ))
   ```

2. **Re-render**: `python -c "from temp_bench import report;
   report.render(component='c5')"` — the AUTO-RESULTS block in
   c5.md will now show the wasteland-comparable peak grade as the
   headline.

3. **Update your future ``run.py``** to use
   ``primary_metric="peak_success_grade_at_coh_1.75"`` for new cells.
   The flatten_metrics function already emits both keys; runner.run_cell
   takes whichever you pass as `primary_metric`.

4. **Sanity check**: after backfill, your peak grades should be in
   the 0.5–2.5 range (T-SAE anchor was ~1.13 at coh ≥ 1.5; ~0.41 at
   coh ≥ 1.75). If your TXC archs come in around 1.0–1.5 at
   coh ≥ 1.75, that's the "matches T-SAE at high coh" hypothesis
   reproducing. If they come in much lower (say 0.1–0.3), it's the
   honest-negative the binary metric was already showing — but with
   credible numbers reviewers can compare to the paper.

### Han decisions 2026-05-04 (resolves prior session's open questions)

1. **Judge: confirm Sonnet. NO Gemini.** Your `SonnetSteeringJudge`
   implementation is right. The original T-SAE § B.2 used Llama-3.3-70B,
   not Gemini, and we're not using Llama-3.3-70B either — so there's
   no "match the paper's judge exactly" pressure. Sonnet aligns C5+C7
   on one judge. Document the deviation in c5.md caveats.
   judge_outputs.jsonl persistence lets us validate κ post-deadline
   if reviewers ask.
2. **`scripts/sync_from_hf.sh`: FIXED.** Renamed `huggingface-cli download`
   → `hf download` in commit (this turn). Drop your `hf download`
   workaround on next session — the script works again. Affects only
   pod restart; no impact on running agents.

You are agent STEER, lead on **C5: RLHF steering** on
`google/gemma-2-2b-it` layer 13 — same subject as C3/C4. The case
study is the T-SAE paper § 4.4 sentiment-steering task.

Hardware: pod `4× A40`, pinned to **GPU 0**. Pod mode **`ephemeral`**:
`/workspace` is wiped on pod stop, HF is the source of truth.
Bootstrap pulls from `han1823123123/temp-bench-{models,data}`;
`cache.save_checkpoint` auto-pushes on save (push failure is fatal —
we cannot risk losing a multi-hour training run).

agent_back shares the pod on GPU 1 (separate component, separate
cache). GPUs 2 + 3 are unassigned — to use them, launch a second
process with `bash scripts/run_on_gpu.sh <idx> -- <command>` (sets
`CUDA_VISIBLE_DEVICES=<idx>` for the subprocess only). No lockfile
manager — read peer's "Current state" + `nvidia-smi` before
borrowing, update your own state with the borrow + ETA. See
PROTOCOL.md § 13 *GPU sharing convention*.

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
- `PROTOCOL.md` § 11, § 12 (pinning), § 13 (GPU sharing convention),
  § 9 *Session wrap-up*

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed; the script prints a
one-liner to verify HF state before stop.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-04T12:53Z (b1024 re-train sweep launched —
Phase-5-faithful uniform TrainingConfig per decisions.md § 12).**

- `git HEAD`: `00cdbea3 Agent STEER: SteeringCaseStudy.primary_metric
  → peak_success_grade_at_coh_1.75` (pushed). Working tree has only
  untracked checkpoint dirs + logs/, plus auto-renders waiting on
  cells.
- **Active b1024 cells (right now, in flight)**: 3 cells × seed 42,
  one per spare GPU, launched 12:48 UTC.
  - GPU 0 / PID 38976 / `tsae_paper × seed 42` / eval_key `1c1b8aa4`
    / log `logs/c5_b1024_tsae_seed42.log` / wait task `bzhobgezy`
  - GPU 2 / PID 38977 / `txc_base × seed 42` / eval_key `b981566c`
    / log `logs/c5_b1024_txc_base_seed42.log` / wait task `bwp0srh2k`
  - GPU 3 / PID 38978 / `txc_pro × seed 42` / eval_key `b36b7641`
    / log `logs/c5_b1024_txc_pro_seed42.log` / wait task `b8bh3agco`
  Each launched with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  to suppress the OOM-fragmentation issue agent_back hit (commit
  `658fa825`). agent_back is on GPU 1 doing C7. PIDs persist across
  my session via `/tmp/p_tsae42`, `/tmp/p_tb42`, `/tmp/p_tp42`.
  No claim_gpu calls — gpu_locks were nuked in commit `6e6efcbd`,
  the new convention is just "set CUDA_VISIBLE_DEVICES + good
  manners" (see PROTOCOL.md § 13 / scripts/run_on_gpu.sh).
- **Wakeup scheduled**: 13:19 UTC for first per-cell rate marker.

- **TrainingConfig is the new Phase-5-faithful default** (decisions.md
  § 12): batch_size=1024, n_steps=25_000, plateau_early_stop=False
  uniform across all archs and pods. My runner uses
  `runner.default_training_cfg() → TrainingConfig()` so the new
  defaults flow automatically — no per-component override needed.
  New batch=1024 cells get fresh train_keys; old batch=256 cells
  stay in leaderboard for diff. analysis.py should filter for
  `training_cfg.batch_size==1024` rows when rendering AUTO-RESULTS
  (TODO once cells land — current analysis.py renders all non-smoke
  rows, so the AUTO-RESULTS will mix old + new until I add the
  filter).

- **C5 v1.0.1 backfill (this session, already pushed)**: re-aggregated
  the OLD batch=256 cells' judge_outputs.jsonl with the new
  peak-grade headline metric. 9 v1.0.1 rows with
  `metric_set="v1_0_1_with_peak_grade"` and
  `rebuild_from=<orig_eval_key>`. Old v1.0.0 rows kept. Numbers:
    - tsae_paper × {42, 1, 2}: 0.367, 0.400, 0.300 → 0.356 ± 0.029
    - txc_base × {42, 1, 2}: 0.300, 0.400, 0.476 → 0.392 ± 0.051
    - txc_pro × {42, 1, 2}: 0.375, 0.389, 0.300 → 0.355 ± 0.028
  All within 1 stderr. Hypothesis "TXC matches T-SAE" → **supported**
  on the wasteland-comparable peak-grade metric. (The old binary
  fraction `success_at_coh_τ` showed a misleading 2× tsae lead
  because nearly all `success ≥ 2` events also have `coh ≥ 2.0`,
  collapsing the dynamic range.)

- **Helper bug-fix landed (this session)**:
  `temp_bench.case_studies.steering.reaggregate_from_judge_outputs`
  now handles BOTH on-disk schemas: per-generation rows (the format
  agent_paper wrote the helper for — both grades on one row) AND
  per-call rows (head + label, the actual format
  `SonnetSteeringJudge._persist` writes via `judge_outputs.jsonl`).
  Pre-fix the helper returned all-0s on my cells.

- **Recent decisions in scope**: #1 (two TXCs), #4 (cross-branch
  reads), #6 (HF repos), #7 (Bricken off for C5), § 12 (b=1024 +
  25k uniform), C5 metric (peak grade as headline).

## What I just did (agent owns — overwrite)

Newest first.

- **Launched 3-way parallel b1024 re-train sweep** (12:48 UTC):
  tsae_paper / txc_base / txc_pro × seed 42 on GPUs 0/2/3 with new
  TrainingConfig defaults from decisions.md § 12 (batch=1024,
  n_steps=25k, plateau_off). PIDs 38976/38977/38978. PIDs saved to
  `/tmp/p_tsae42` `/tmp/p_tb42` `/tmp/p_tp42`. Three Bash background
  wait tasks armed (`bzhobgezy`/`bwp0srh2k`/`b8bh3agco`) — they fire
  on process exit. Wakeup at 13:19 UTC for first rate marker. After
  each cell finishes I'll launch the next seed for that arch on the
  freed GPU.
- Updated `SteeringCaseStudy.primary_metric → peak_success_grade_at_coh_1.75`
  (commit `00cdbea3`, pushed) — new headline metric per
  agent_paper's directive. Old `success_at_coh_1.75` still emitted
  by `flatten_metrics` so any analysis filter that wants binary
  fraction still gets it.
- **C5 metric backfill** (commit `b91ecbc5`, pushed): re-aggregated
  the 9 OLD batch=256 cells' `judge_outputs.jsonl` files into v1.0.1
  rows using the new peak-grade metric. ALSO fixed the helper
  `reaggregate_from_judge_outputs` because it was silently returning
  0s for my cells — agent_paper wrote it expecting per-generation
  rows (both grades on one record) but my judge_outputs.jsonl uses
  per-call rows (`head: success/coherence`, `label: 0-3`). Helper
  now branches on schema. Updated `c5.md` Hypothesis section:
  outcome `refuted → supported` on the new metric.
- **9-cell C5 sweep at batch=256 — completed** (pre this session):
  tsae × 3 + txc_base × 3 + txc_pro × 3 (txc_pro at n_steps=6000
  paper-deviation) → 9 v1.0.0 rows in leaderboard. AUTO-RESULTS
  rendered + pushed via commit `91f763e6` / `bfdcc559`.
  Headline (binary fraction, OLD metric):
    tsae 0.067, txc_base 0.033, txc_pro 0.031.
  New headline (peak grade, post-backfill):
    tsae 0.356, txc_base 0.392, txc_pro 0.355 — all within 1 stderr.
- W_enc contiguity fix in run.py (commit `0aea9cba`): tsae_paper's
  W_enc is initialized via `W_dec.clone().T` which leaves a
  non-contiguous transposed view; `safetensors.save_file` rejects
  it. My train_fn now `.contiguous()`-s every state_dict tensor
  before save. agent_nlp later landed the upstream tsae.py fix
  (commit `af552412`) but my workaround is harmless and runs first.
- Pre this session's metric work: full code port + smoke validation
  + W_enc fix + smoke flag + run_dir mismatch fix + Sonnet judge
  with `judge_outputs.jsonl` persistence + V7 + PP hooks + 30
  paper-faithful concepts + 17 unit tests + auto-push run_dirs to
  HF temp-bench-data. Commits: `b0519a99` (port) → `f8a28469`
  (run_dir fix) → `21c84be8` (c5.md expansion) → eval cycles.

- **Critical incidents resolved this session**:
  - Helper-returns-0s diagnosed + fixed in 1 turn (per-call vs
    per-generation schema mismatch).
  - Duplicate leaderboard row (txc_pro seed=2 appeared twice after
    a rebase chaos) deduped in commit `01e07db0` → `20ba7913`.
  - Multi-stage NFS I/O errors during git operations cleared by
    deleting stale `.git/*.lock` files + `git reset --hard origin/final`.
    NO commits lost (all my work was already pushed before the I/O
    storm).

## Next action (agent owns — overwrite)

**Pre-flight on every `--continue` session:**
1. `cd /workspace/temp_xc_steer/purified && source scripts/set_agent_env.sh agent_steer`
2. `bash scripts/agent_smoke_test.sh` (CRITICAL preflight failures
   are fatal)
3. (Ephemeral pod restart only): re-pull act-cache via
   `bash scripts/sync_from_hf.sh` (was broken; agent_paper landed
   the `huggingface-cli → hf download` fix in commit
   `b4f80dbe`-area; trust the script now).
4. `git pull --rebase origin final` — likely substantial agent
   updates.
5. `.venv/bin/python -m pytest tests/test_steering.py -q`

**Then check the b1024 sweep state. The 3 seed-42 cells launched
2026-05-04 12:48 UTC may still be in flight or done. Each cell:
~3.3× more total compute than the old batch=256 (25.6M vs 7.68M
tokens). At previous batch=256 rates ~3-4 steps/sec for tsae +
txc_base, ~1.3 steps/sec for txc_pro, batch=1024 should be ~4×
slower per step. Estimate per cell: tsae ~25-40 min, txc_base
~50-80 min, txc_pro ~120-150 min — train only; +15-20 min eval.**

A. **First**: `ps -p $(cat /tmp/p_tsae42 /tmp/p_tb42 /tmp/p_tp42)
   -o pid,etime --no-headers 2>&1` — see which cells are still
   alive. If a Bash wait-task fired while you were compacting, look
   at `/tmp/claude-1000/.../tasks/{bzhobgezy,bwp0srh2k,b8bh3agco}.output`
   for the post-mortem.

B. **As each seed=42 cell completes**, launch the next seed for
   that arch on the freed GPU:
   ```
   CUDA_VISIBLE_DEVICES=<N> TQDM_DISABLE=1 AGENT_NAME=agent_steer \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     .venv/bin/python -m experiments.c5_steering.run \
     --archs <arch> --seeds <seed> > logs/c5_b1024_<arch>_seed<seed>.log 2>&1 &
   ```
   Sequence: each arch goes seed 42 → seed 1 → seed 2. Realistically
   the full 9-cell sweep is 6-10 hours wall-time at 3-way parallel.

C. **For each cell, check loss-trajectory at the END of training**
   (per agent_paper's directive line 64-65 in this briefing). If the
   final-1K-step loss drop > 5% of loss value at step 25K, the cap
   needs bumping uniformly across all archs. Surface as an Open
   Question — don't unilaterally bump.

D. **Watch for OOM kills on shared spare-pool GPUs**. Last sweep
   lost cells when agent_back's process competed for GPU 3 (silent
   kill, no traceback, just process gone). The new gpu_locks-nuked
   convention means we have no enforcement — just monitor
   `nvidia-smi` and be ready to relaunch.

E. **After each cell completes**: leaderboard row + run_dir push
   are automatic via my run.py's auto-HF-push and runner's
   leaderboard append. Verify via `tail -1 results/leaderboard.jsonl`.
   No manual reconstruction (no git reset --hard chaos this time).

F. **After all 9 b1024 cells complete**:
   ```
   .venv/bin/python -c "from temp_bench import report; report.render(component='c5')"
   ```
   ⚠ **You will likely need to update `experiments/c5_steering/analysis.py`
   to filter on `r.training_cfg.batch_size == 1024`** — agent_paper's
   directive explicitly says "analyses in experiments/cN_*/analysis.py
   should filter for the new training_cfg.batch_size=1024 rows when
   rendering AUTO-RESULTS" (decisions.md § 12). Currently my
   analysis.py renders ALL non-smoke c5 rows, which would mix the 9
   v1.0.0 (b256) + 9 v1.0.1 (b256-backfilled-peak) + 9 new b1024
   rows. The leaderboard row's `training_cfg` is hashed into
   `train_key` not exposed directly; need to look up the manifest
   for batch_size or use `eval_protocol_version` / a new
   `metric_set` tag. **TODO this turn**: add the filter, otherwise
   the rendered table is incoherent.

G. **Update c5.md** post-render: bump `last_update` date, ensure
   "Outcome" reflects the b1024 numbers (not the b256 backfill).
   Caveats: the b1024 cells use n_steps=25k uniformly — txc_pro is
   no longer "n_steps=6000 paper-deviation"; that caveat should
   move to the v1.0.0 / v1.0.1 supplementary section (or be
   deleted from current state and replaced with a "compute parity"
   note).

**Commit + push after each batch of cells lands** (don't wait until
all 9 finish). Use the Phase-5-faithful framing in commit messages.

## Don't repeat (agent owns — overwrite)

- **`git reset --hard origin/final`** during a rebase storm — last
  time this WIPED 2 leaderboard rows (txc_pro × {42, 1} v3) and a
  couple of commits I had to manually reconstruct. NFS lock cruft
  on `/workspace/temp_xc_steer/.git/*.lock` triggers cascading I/O
  errors; the right move is `rm /workspace/temp_xc_steer/.git/*.lock`
  + retry. Reset --hard is last-resort.
- **Bypass `compute_train_key` deterministic hashing** — when
  reconstructing leaderboard rows after a wipe, NEVER hand-write a
  train_key. Use `temp_bench.config.compute_train_key(arch=spec,
  seed=seed, training_cfg=cfg, act_cache_key=ack)`. Otherwise the
  runner doesn't recognize the cached checkpoint.
- **Auto-tag smoke=true on `--n-steps`** (the OLD heuristic in
  run.py was `smoke = args.smoke or args.n_steps is not None`).
  That hid paper-deviation cells from analysis. Fix landed
  (`smoke = args.smoke` only). Don't reintroduce.
- **Hill-climbing winners** — Galaxy 8/11/18, SoftMaxPool,
  ContrastiveMergeH8 excluded by decision #1.
- **Skip the V7 pre-test on TXC-pro** — `--pre-test-only` is
  cheap (~1 min) and c5.md's hypothesis is contingent on it.
  (Currently NOT run for the b1024 sweep; defer pre-test until at
  least one txc_pro b1024 cell lands so we know the baseline.)
- **Hand-edit `docs/components/c5.md` AUTO-RESULTS** — that block is
  owned by analysis.py + `report.render`.
- **Wasteland imports** — `git show origin/han-phase7-unification:…`
  only; never `import experiments.phase7_unification…`.
- **Touch `scripts/sync_from_hf.sh`** unilaterally — agent_paper
  already landed the fix.
- **Touch `temp_bench/utils/tokens.py`** to add a `gemini` slot —
  Han confirmed Sonnet is the judge (no Gemini), no change needed.
- **Forget per-call vs per-generation `judge_outputs.jsonl` schema**
  — my SonnetSteeringJudge writes per-call (head + label) rows;
  the helper `reaggregate_from_judge_outputs` now branches on
  schema. If you change the writer, update the reader.
- **Mix b256 + b1024 cells in the AUTO-RESULTS** — analysis.py needs
  a filter on training_cfg.batch_size=1024 (TODO; agent_paper's
  directive). Rendering before adding the filter will produce a
  misleading mixed-config table.

## Open questions for Han (agent owns — overwrite)

(Most prior open questions resolved by Han — Sonnet judge confirmed,
sync_from_hf.sh fixed, IT/L13 confirmed in c5.md caveats, gpu_locks
nuked. Remaining:)

1. **Convergence at step 25k** — agent_paper's directive line 64-65
   says "if final-1K-step loss drop > 5% of loss value at step 25K,
   surface that — the cap may need bumping uniformly." After the
   first b1024 cell completes, I'll inspect the loss-trajectory
   field of `result["log"]["loss"]` (saved via the runner) and
   surface here. **TODO: define the inspection script + threshold
   check post-cell.**

2. **analysis.py filter for batch=1024 rows** — agent_paper
   explicitly says analyses should filter for the new
   training_cfg.batch_size=1024 rows. The current analysis.py
   doesn't have this filter — TODO this turn (or post-compact).
   The leaderboard row's `eval_cfg` doesn't carry batch_size; need
   to look it up via the manifest (`checkpoints/manifest.jsonl`
   keyed by `train_key`) or add a `metric_set="b1024_v1"` tag in
   eval_cfg that the runner threads through. Surfacing as Q because
   the right path may want a framework-level `_training_cfg` exposed
   in enriched_cfg.

3. **txc_pro V7 pre-test** — the pre-registered hypothesis
   contingency (V7 may break end-position-discriminative encoders;
   fall back to PP if mean coh ≤ 1.0) was NOT exercised on the
   b1024 sweep. The b256 cells showed mean coh 2.1-2.2 for txc_pro
   so V7 worked; b1024 is a different regime. If b1024 txc_pro
   produces degenerate output, run `--pre-test-only --archs txc_pro
   --protocol pp` and document.

4. **What if txc_pro b1024 doesn't show "matches T-SAE"** — the
   b256 backfill numbers were within 1 stderr of tsae. b1024 is
   3.3× more compute, all archs trained equally. If a clear gap
   opens (TXC > T-SAE, the original hypothesis was "matches not
   beats"), c5.md's framing needs revising — but this would be a
   GOOD problem to have. Watch for it.
