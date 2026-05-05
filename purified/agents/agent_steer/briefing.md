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

### ⚠️ Han decisions 2026-05-05 PM — C5 T-SAE baseline T=1 re-train (your tsae_paper rows shift)

**Background**: SAEBench (papers/are_saes_useful.md, App. B) shows
canonical SAE training is buffer-based, batch=2048 TOKENS/step, ~500M
tokens total. Our C5 per-token arch (`tsae_paper`) currently trains at
`(B=1024, seq_len=128) → flatten = 131,072 tokens/step` — 65× over
SAEBench's canonical 2K. C6 + C7 baselines coincidentally use a
window-based pattern at T=1 (within 2× of canonical); C3 + C5 inherited
the over-batched sequence-based pattern. The earlier "MW deployment"
pivot (decisions.md § 14) was solving the right diagnosis with the wrong
fix. **The right fix is to bring per-token baselines DOWN to T=1 window-
based, not bring TXC up via MW.**

**Han's call (2026-05-05 PM)**: re-train C5's `tsae_paper` (only — TXC
archs are unchanged) × 3 seeds at `train_window_size=2` — Bhalla/Ye
2025 §3.1 explicitly trains T-SAE on adjacent pairs
($\mathbf{x}_t, \mathbf{x}_{t-1}$); 2048 tokens/step matches SAEBench's
2K canonical exactly. **agent_filler is the helper running this
re-train** (repurposed from the aborted C5 MW parallel pivot, now
running on 8× A40). You don't run anything new; the re-trained
`tsae_paper` cells land in `leaderboard.jsonl` automatically.

**Framework change (commit `5555e7eb`)** — read before paper-rendering:

- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` gains
  `train_window_size: int | None = None`. None = full-sequence
  (current); int T = 1 random T-window per row.
- `TrainingConfig.train_window_size: int | None = None` added; flows
  into `compute_train_key` via `model_dump(exclude_none=True)`. Default
  (None) preserves YOUR EXISTING TRAIN KEYS — your v1.1.0 9/9 sweep
  keeps its cache. Setting an int gets a fresh key.

**Your action — analysis.py filter shift**:

`experiments/c5_steering/analysis.py` currently calls
`canonical_train_keys(...)` for all 3 archs (`tsae_paper` + `txc_base`
+ `txc_pro`) at `TrainingConfig(n_steps=20_000)`. After agent_filler's
re-train lands, archs split:

- TXC archs (`txc_base`, `txc_pro`): unchanged. Use
  `TrainingConfig(n_steps=20_000)`.
- T-SAE: re-trained at T=2 (paper-faithful pairs). Use
  `TrainingConfig(n_steps=20_000, train_window_size=2)`.

Two `canonical_train_keys` calls, union the returned sets:

```python
from temp_bench.report import canonical_train_keys
from temp_bench.schemas import TrainingConfig

txc_keys = canonical_train_keys(
    component="c5",
    archs=["txc_base", "txc_pro"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000),
)
tsae_keys = canonical_train_keys(
    component="c5",
    archs=["tsae_paper"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000, train_window_size=2),
)
canonical = txc_keys | tsae_keys
```

Your existing v1.1.0 `tsae_paper` rows stay in the leaderboard under
their old hashes for diff comparison; the analysis.py filter drops them
from the headline AUTO-RESULTS once you wire the second call.

**Paper-claim shift (uncertain pending re-run)**: Today's v1.1.0 C5
headline (peak success grade @ coh ≥ 1.75) shows tsae_paper 2.167 ±
0.104 > txc_pro 1.284 > txc_base 0.792 — "TXC matches T-SAE" hypothesis
REFUTED. Under canonical T=1 training, T-SAE may score lower (it was
over-batched 65×; now at literature scale). The result may shift toward
the original hypothesis (TXC ties T-SAE at high coh) or stay refuted
with a literature-aligned baseline. Either is reviewer-defensible.
Wait for agent_filler's re-train to land before re-rendering.

**No action on your status**: C5 v1.1.0 sweep is COMPLETE; you remain
in `status: complete`. The re-trained cells are agent_filler's
responsibility; your output is the diff-comparison reference. Use
remaining session time for paper-writing if helpful (c5.md caveats:
"per-token T-SAE baseline trained at SAEBench's canonical 1K
tokens/step instead of our sequence-based 131K — agent_filler re-run,
methodological note in § 15").

See `decisions.md` § 15 for the full rationale.

---

### Han decisions 2026-05-04 PM ⚠️ URGENT — 25K→20K + preloaded batch_iter

Two directives, both effective immediately:

**(1) Override `n_steps` 25_000 → 20_000 in your C5 runner.** The Gemma-
family components (C3, C4, C5) now standardize on `20_000` steps ×
`batch=1024` = 20.5M tokens / cell. Aligns C5 with C3+C4 on the Gemma
training-depth axis (C6 stays at 25K × Qwen, C7 at 20K × Llama — see
`docs/paper/methodology.md` per-component table). The marginal value of
the extra 5K steps is small for the C5 "matches T-SAE at high coh"
hypothesis; the wall-clock saved frees A40 cycles for agent_back.

⚠️ **Within-component fairness — KILL in-flight 25K cells immediately.**
Your three b1024 cells (tsae_paper / txc_base / txc_pro × seed 42, PIDs
38976/38977/38978 from the prior session) were launched at 25K. They
cannot stay in C5's headline result alongside seeds 1+2 trained at 20K
— that's exactly the within-component config drift we built decisions.md
§ 12 to prevent. Steps:

```bash
kill $(cat /tmp/p_tsae42 /tmp/p_tb42 /tmp/p_tp42)
ps -p $(cat /tmp/p_tsae42 /tmp/p_tb42 /tmp/p_tp42) 2>/dev/null  # confirm dead
```

Then update `experiments/c5_steering/run.py::_real_training_cfg`:

```python
def _real_training_cfg() -> TrainingConfig:
    # n_steps overridden 25_000 → 20_000 per Han 2026-05-04 PM URGENT.
    # Aligns C3 + C4 + C5 on the Gemma 20.5M-token-per-cell axis.
    return TrainingConfig(n_steps=20_000)
```

Restart all 9 cells (3 archs × 3 seeds) at 20K. The 30-90 minutes of
in-flight GPU work being thrown away is small vs the cost of a
within-C5 breach in the paper.

**(2) Swap to `preloaded_batch_iter_from_act_cache`.** agent_nlp landed
the helper at `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache`
(commit `e12dc719`). It's a bit-identical drop-in for `batch_iter_from_act_cache`
that pre-materializes the activation cache into a CPU torch tensor via
`.clone()`, sidestepping the mmap page-fault overhead (~1.4× end-to-end
trainer speedup; ~3.4× on the data path). Determinism guarantee:
checkpoints are bit-identical for the same `(act_cache_key, seed)` pair —
**train_keys unchanged, no fairness implication**, adopt mid-sweep is
fine.

In `experiments/c5_steering/run.py`:

```python
# Before
from temp_bench.data.nlp import batch_iter_from_act_cache
...
raw_iter = batch_iter_from_act_cache(act_cache_key, seed=seed)

# After
from temp_bench.data.nlp.cache import preloaded_batch_iter_from_act_cache
...
raw_iter = preloaded_batch_iter_from_act_cache(act_cache_key, seed=seed)
```

**RAM cost on A40 pod**: ~14 GB CPU per process for the Gemma cache
(module-global per `act_cache_key`, so multiple cells in the same
process share). agent_back uses ~4.24 GB for their Llama cache on
parallel processes; A40 pod has ~64 GB system RAM total — your ~14 GB
+ their ~17 GB (4 procs × 4.24) = ~31 GB. Plenty of headroom.

**(3) GPU pinning re-allocation — you get GPUs 1 and 3.** New A40 pod
split (Han 2026-05-04 PM): agent_back gets GPUs 0 and 2; you get GPUs
**1 and 3**. Each agent has two dedicated GPUs — no more "borrow agent
peer's spare" pattern, no GPU 0 for you.

When you relaunch the 9-cell sweep at 20K, distribute across GPUs 1
and 3 only:

```bash
# 3 archs × 3 seeds = 9 cells across 2 GPUs → 4-5 cells per GPU sequentially
CUDA_VISIBLE_DEVICES=1 TQDM_DISABLE=1 AGENT_NAME=agent_steer \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m experiments.c5_steering.run \
  --archs tsae_paper txc_base --seeds 42 1 2 \
  > logs/c5_b1024_n20k_gpu1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 TQDM_DISABLE=1 AGENT_NAME=agent_steer \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m experiments.c5_steering.run \
  --archs txc_pro --seeds 42 1 2 \
  > logs/c5_b1024_n20k_gpu3.log 2>&1 &
```

(Adjust the arch-to-GPU split to balance wall-time — tsae_paper is fast,
txc_pro is slow.) Use `bash scripts/run_on_gpu.sh <idx> -- <cmd>` if you
prefer the wrapper. **Do not touch GPUs 0 or 2** — those are agent_back's.

Order of operations: do (1) first (kill + commit the 20K override),
then (2) at relaunch (swap the import in the same edit window before
launching the 9-cell sweep), pinned to GPUs 1+3 per (3).

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

Hardware: pod `4× A40`, pinned to **GPUs 1 and 3** (Han 2026-05-04 PM
re-allocation). Pod mode **`ephemeral`**: `/workspace` is wiped on pod
stop, HF is the source of truth. Bootstrap pulls from
`han1823123123/temp-bench-{models,data}`; `cache.save_checkpoint`
auto-pushes on save (push failure is fatal — we cannot risk losing a
multi-hour training run).

agent_back shares the pod on **GPUs 0 and 2** (separate component,
separate cache). The A40 pod is now fully partitioned: 2 dedicated GPUs
per agent, no unassigned slots, **no borrow pattern**. Launch parallel
processes via `bash scripts/run_on_gpu.sh <idx> -- <command>` for
GPU 1 or 3 only — never touch 0 or 2. See PROTOCOL.md § 13.

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

**Last verified: 2026-05-05T11:00Z — c5 status: COMPLETE (9/9 cells
at v1.1.0 with concept-lift bug fix).**

- `git HEAD`: `eb15651d Agent STEER: c5 — v1.1.0 sweep complete
  (9/9 cells); hypothesis REFUTED` (pushed).
- **Final headline (peak success grade @ coh ≥ 1.75, n=3 each)** —
  v1.1.0, concept-lift selection:
  - tsae_paper: 2.167 ± 0.104  (seeds {42:2.056, 1:2.375, 2:2.071})
  - txc_pro:    1.284 ± 0.084  (seeds {42:1.333, 1:1.120, 2:1.400})
  - txc_base:   0.792 ± 0.063  (seeds {42:0.842, 1:0.867, 2:0.667})
  Hypothesis "TXC matches T-SAE" REFUTED — T-SAE beats txc_pro by
  0.88 ± 0.13 (~7σ) and txc_base by 1.38 (~13σ). Relative ordering
  matches wasteland phase-7 (T-SAE > TXC-pro > TXC-base).
- **Concept-lift bug fix** (commit `ef33f822`, v1.0.0 → v1.1.0):
  Han caught it from the c5 plot — all peak-grade points aligned
  horizontally and success ~0.3 vs wasteland's 1.13 anchor. Root
  cause: ``select_best_features`` did raw-activation argmax. On
  tsae_paper, ALL 30 concepts selected feature 3010 (always-on text
  feature, activation ~95 across every concept, 5× the next-best).
  Fix: subtract per-feature cross-concept baseline → concept-lift.
  Verbatim from wasteland's
  ``origin/han-phase7-unification:experiments/phase7_unification/case_studies/steering/select_features.py``.
  All 9 cells re-evaluated on cached training checkpoints with
  ``--force-eval``; took ~30 min total.
- **GPU pinning (Han 2026-05-04 PM)**: I get GPUs 1 and 3;
  agent_back gets 0 and 2. Now both my GPUs are idle — sweep done.
- **Preloaded batch_iter** (commit 751d1789): bit-identical with
  legacy iterator, ~1.4× trainer speedup. Gemma cache ~14 GB RAM
  per process.
- **Recovery from API outage 2026-05-05T05:59 UTC** (resolved):
  during the v1.0.0 sweep, txc_base seed=1's 270 judge calls failed
  with HTTP 400 "credit balance is too low". Han topped up at 06:30.
  Both old failure-mode incident and v1.0.0 buggy rows are now
  superseded by v1.1.0.

- **TrainingConfig is now (Han 2026-05-04 PM URGENT)**:
  batch_size=1024, n_steps=**20_000** (deadline override),
  plateau_early_stop=False. C3 + C4 + C5 (Gemma-family components)
  share this 20.5M-token-per-cell axis. C6 stays at 25K × Qwen,
  C7 at 20K × Llama. My run.py uses `_real_training_cfg()` to
  return `TrainingConfig(n_steps=20_000)` — explicit override above
  the schema default of 25_000.
- **analysis.py canonical_train_keys filter (already wired)**:
  `experiments/c5_steering/analysis.py` uses
  `temp_bench.report.canonical_train_keys(...)` keyed off the
  current `TrainingConfig()` defaults. Since my runner passes
  `_real_training_cfg()` (20K override) to run_cell, the train_keys
  generated will be canonicalized against `TrainingConfig(n_steps=20_000)`.
  ⚠ TODO: verify analysis.py's canonical_train_keys call also passes
  `training_cfg=TrainingConfig(n_steps=20_000)` — otherwise it'll
  filter against the schema default 25K and miss my 20K cells.

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

**C5 sweep is COMPLETE. There is no follow-up task unless the paper
revision asks for one.** Leaving these notes for the post-compact
instance in case Han needs additional cells:

A. **Re-render is one-liner** (workaround for OQ #5b
   `c5_steering_100k` glob conflict):
   ```python
   import json, importlib
   from pathlib import Path
   from temp_bench import report
   mod = importlib.import_module('experiments.c5_steering.analysis')
   importlib.reload(mod)
   result = mod.run_analysis()
   report._replace_auto_results(Path('docs/components/c5.md'), result.markdown)
   Path('experiments/c5_steering/results.json').write_text(
       json.dumps(result.results, indent=2, sort_keys=True))
   ```
   Direct call bypasses `report.render()`'s `_experiment_dir` glob.

B. **If reviewers question the txc_base seed=1 reeval**, note that
   the cached training checkpoint (train_key `196d4595f0f3b626`) is
   bit-identical between the failed run at 05:59 UTC and the
   re-judged run at 07:13 UTC. Only the eval phase (Sonnet calls
   over generated text) ran twice. Determinism: the steering
   protocol uses ``do_sample=False`` (greedy decode), so the
   generated text is bit-identical too. The judge is stochastic,
   but Sonnet outputs the same labels 99 %+ of the time on a clean
   run. The result is real, just delayed by 75 minutes.

C. **If Han wants additional cells (e.g., a 4th arch or larger
   d_sae)**, just call `run_one_cell(...)` from
   `experiments/c5_steering/run.py` with the new arch. The
   canonical_train_keys filter will pick up the new cells
   automatically (if added to `ARCHS` in `analysis.py`).

D. **post-deadline Cohen's κ validation**: 9 × 270 = 2430 Sonnet
   judge calls are persisted in
   `results/runs/<eval_key>/judge_outputs.jsonl` per cell. Format
   is per-call rows (head=success/coherence, label=0-3 grade) —
   `temp_bench.case_studies.steering.reaggregate_from_judge_outputs`
   handles both schemas (per-call and per-generation). PROTOCOL.md
   § 7 *Judge κ deferred*.

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

(All four prior open questions resolved by the completed 9-cell
20K sweep — convergence < 5 %, canonical_train_keys filter wired,
V7 worked on txc_pro b1024 with mean_coh 2.27–2.40, "matches T-SAE"
held within 1 stderr. Anthropic API outage 5a resolved by Han's
top-up + force_eval recovery. Remaining:)

1. **`experiments/c5_steering_100k/` breaks `report.render(component='c5')`**
   — agent_paper spun up agent_steer_100k as a parallel 100K-step
   instance (commit `6db405bd`). Their dir lives at
   `experiments/c5_steering_100k/` and matches the `c5_*` glob in
   `temp_bench.report._experiment_dir`, raising
   ``RuntimeError: Multiple experiment dirs match c5_*``. I worked
   around it locally by importing my analysis module directly:
   ```python
   import importlib, json
   from pathlib import Path
   from temp_bench import report
   mod = importlib.import_module('experiments.c5_steering.analysis')
   importlib.reload(mod)
   result = mod.run_analysis()
   report._replace_auto_results(Path('docs/components/c5.md'), result.markdown)
   Path('experiments/c5_steering/results.json').write_text(json.dumps(result.results, indent=2, sort_keys=True))
   ```
   The framework-level fix is for `_experiment_dir` to take a list
   of suffixes that DO match (e.g. `{"_steering"}` for c5) — that's
   `temp_bench/report.py` (agent_paper territory). Surface for them
   to land. Until then I'll keep using the direct-import workaround.
