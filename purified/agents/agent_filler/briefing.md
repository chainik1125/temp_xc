<!--
DRAFT — written by agent_paper 2026-05-05 PM. New agent identity for
the 8× A40 filler pod, parallel C5 multi-window deployment.
Section ownership: PROTOCOL.md § 14.
-->

---
agent: agent_filler
last_state_update: 2026-05-05T12:00:00Z
component: c5 (multi-window deployment, parallel)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent FILLER**. You own the **C5 multi-window deployment**
on a fresh 8-GPU A40 pod. Your purpose: run agent_steer's canonical
C5 sweep with the multi-window TXC variants, in parallel across GPUs.
The multi-GPU pod's RAM (401 GB) and CPU (76 cores) make it well-
suited for running 6 cells concurrently.

Files you may edit:

- `agents/agent_filler/briefing.md` (your own — agent-owned sections only)
- `experiments/c5_steering_filler/` (new experiment directory you create
  with a parallel-launch driver — see "First concrete task" below).
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. agent_steer's C5 case-study
  code (V7 steering, Sonnet judge, concept-lift baseline, preloaded
  batch_iter) is already wired and compatible with the multi-window
  TXC archs. Re-use via imports.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory **including
  `agents/agent_steer/` and `agents/agent_steer_100k/`**. Their
  briefings, decisions, and per-cell state are theirs.
- `experiments/c5_steering/` — agent_steer's territory.
- `experiments/c5_steering_mw/` — agent_steer_100k's territory (their
  pivot driver). You write your OWN driver in
  `experiments/c5_steering_filler/`, NOT in their dir.
- `docs/components/c5.md` — agent_steer's territory. agent_paper
  integrates results at paper-render time.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — atomic, agent_paper coordinates.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.
This is non-negotiable — see PROTOCOL.md § 8 + CLAUDE.md Hard Rule #7.

### ⚠️⚠️⚠️ STAND DOWN — MW pivot RESCINDED 2026-05-05 PM ⚠️⚠️⚠️

**Han + agent_paper diagnosed that the MW pivot was solving a misframed
problem.** SAEBench (papers/are_saes_useful.md, App. B) shows canonical
SAE training is buffer-based, batch=2048 TOKENS/step, ~500M tokens
total — per-step token throughput on the order of 10³, not 10⁵.

Our two patterns at this paper:

| Component | Pattern                             | per-token tokens/step |
|---|---|---:|
| C3, C4, C5 | sequence-based (B sentences × all 128 positions) | **131,072** — 5× over SAEBench's 2K canonical |
| C6, C7    | window-based (B sentences × T positions, T=1 for SAE) | **1,024** — close to SAEBench canonical |

C3/C5's 131K is OVER-batched per-step; C6/C7's 1K is near-canonical.
The earlier "TXC has 25× FLOPs disadvantage" framing was directionally
true at C3/C5, but the FIX direction was wrong: bring per-token
baselines DOWN to T=1 window-based (matching SAEBench + matching
C6/C7), NOT bring TXC up via MW.

**Han's call (2026-05-05 PM)**: ABORT all 4 MW pivots. Re-train
per-token baselines at C3 + C5 with the T=1 window-based pattern.

**Your specific abort actions** (do these in this order):

1. **Kill any in-flight C5 MW processes.** If you launched
   `experiments/c5_steering_filler/run_sweep.sh` (6 parallel cells on
   GPUs 0..5), kill all of them. PIDs are in `/tmp/p_filler_gpu{0..5}`
   if the sweep launched.
   ```bash
   for f in /tmp/p_filler_gpu*; do kill -TERM "$(cat $f)" 2>/dev/null; done
   pkill -TERM -f "experiments.c5_steering_filler" || true
   sleep 2
   pkill -KILL -f "experiments.c5_steering_filler" || true
   nvidia-smi --query-gpu=memory.used --format=csv
   # → expect every GPU <500 MB; if not, force-kill stragglers
   ```
   Any landed cells (`txc_base_mw` / `txc_pro_mw` rows in
   `leaderboard.jsonl`) stay — `canonical_train_keys` filters them out
   at paper-render time, harmless.
2. **Set status: idle, awaiting re-purpose.** Do NOT launch any
   further MW work. Update Current state in this briefing to reflect
   "STOOD DOWN — awaiting C5 T-SAE baseline re-train directive."
3. **Your next mission (when re-purposed) will be C5 T-SAE baseline
   T=1 re-train.** agent_paper is landing the framework change
   (`train_window_size: int | None` on `preloaded_batch_iter_from_act_cache`
   + `TrainingConfig`) and will rewrite this briefing to redirect you
   to the C5 T-SAE re-train (tsae_paper × 3 seeds with
   `TrainingConfig(train_window_size=1)`). 3 cells parallel on 3 of
   your 8 A40 GPUs → ~3 hr wall. Wait for the briefing rewrite before
   resuming. ETA on the rewrite: same session — should be fresh by the
   time you read this if you `git pull` again.

**The C5 MW deployment directive below this line is RESCINDED.** Do
not read it as actionable. Left in place for git provenance only.

---

### ⚠️ NEW MISSION 2026-05-05 PM — C5 T-SAE baseline T=2 re-train (decisions § 15)

**You are repurposed.** The MW pivot is dead; the new mission is to
re-train C5's `tsae_paper` baseline at the literature-faithful window
size. agent_paper landed:

- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache(...,
  train_window_size: int | None = None)` — kwarg added; None preserves
  current full-sequence behavior.
- `TrainingConfig.train_window_size: int | None = None` — new field;
  flows into `compute_train_key` via `model_dump(exclude_none=True)`,
  so OLD train_keys preserved (None default) and NEW cells with int set
  get fresh keys (5 new tests, 136/136 green; commit `5555e7eb`).
- agent_steer's `experiments/c5_steering/run.py:my_train_fn` already
  passes through `training_cfg.train_window_size` to the helper, and
  `run_one_cell` accepts a `train_window_size` kwarg (commits landing
  in this push). You inherit both via import.

**Mission scope** (n=3 paired with agent_steer's existing canonical):

| Arch         | seeds      | `train_window_size` | tokens/step | Source |
|---|---|---:|---:|---|
| `tsae_paper` | {42, 1, 2} | **2** | 2048 | Bhalla/Ye 2025 §3.1 paper-faithful adjacent pairs |

**Total: 3 cells.** `batch_size=1024` uniform (Han's call: same B
across archs is cross-arch fairness; T-SAE encodes both anchor and
pair, so its per-step token throughput is 2048 by design).

Other C5 archs (`txc_base`, `txc_pro`) unchanged — agent_steer's
existing v1.1.0 cells stand. **Don't run TXC.** No `topk_sae` either
(it's not a C5 baseline; only T-SAE is the comparison per
`docs/components/c5.md`).

**TrainingConfig**:

```python
TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
    train_window_size=2,          # ← new field, T=2 paper-faithful pairs
)
```

**Per-cell wall-time on A40** (rough): T-SAE at T=2 has ~16× less per-
step encoder work than the over-batched full-sequence version (encoder
runs on 2N tokens vs 2N tokens × 64 positions if it had been per-token-
flattened; matched against agent_steer's existing tsae cells which were
~25-40 min train). At T=2 expect **~10-15 min train + ~30 min judge
eval = ~45 min per cell**. With 3 cells in parallel on 3 of your 8
A40 GPUs, **total wall ~45-60 min**. Plenty of margin.

### First concrete task — kill any MW stragglers, write driver, smoke, launch

Step 0 — verify the STAND DOWN was effective. Confirm no C5 MW
processes still running, all 8 GPUs clean:

```bash
pkill -KILL -f "experiments.c5_steering_filler" || true
sleep 2
nvidia-smi --query-gpu=index,memory.used --format=csv
# → expect every GPU < 500 MB
```

Step 1 — `git pull --rebase origin final` and verify the framework:

```bash
.venv/bin/python -c "
from temp_bench.schemas import TrainingConfig
cfg = TrainingConfig(train_window_size=2)
print(cfg.train_window_size)  # → 2
"
.venv/bin/python -c "
from experiments.c5_steering.run import run_one_cell
import inspect
assert 'train_window_size' in inspect.signature(run_one_cell).parameters, \
    'run_one_cell missing train_window_size kwarg — git pull?'
print('OK')
"
```

Step 2 — pull the Gemma cache (was already pulled for the now-aborted
MW work; should still be on disk):

```bash
ls results/act_cache/*/acts.npy 2>/dev/null
[ ! -f results/act_cache/*/acts.npy ] && bash scripts/sync_from_hf.sh
```

Step 3 — write `experiments/c5_steering_baseline/__init__.py` (empty)
and `experiments/c5_steering_baseline/run.py`. Single-cell driver,
called once per seed pinned to a specific GPU. Different name from
`c5_steering_filler` (which was your aborted MW driver) — pick fresh
to avoid confusion.

```python
"""C5 T-SAE baseline re-train at T=2 (decisions.md § 15).
Single-cell driver; the top-level run_sweep.sh launches 3 in parallel.
"""
from __future__ import annotations
import argparse
import os

# Match the threading defaults agent_steer_100k tuned for parallel runs.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from experiments.c5_steering.run import (
    EVAL_PROTOCOL_VERSION,
    run_one_cell,
)
from temp_bench.case_studies.steering import (
    DEFAULT_COH_THRESHOLDS,
    DEFAULT_STRENGTHS,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 T-SAE baseline re-train at T=2 — single (arch, seed) cell."
    )
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps for smoke tests.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tag the leaderboard row eval_cfg.smoke=True.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    print(
        f"[c5_baseline] tsae_paper seed={args.seed} T=2 "
        f"eval_protocol={EVAL_PROTOCOL_VERSION} smoke={args.smoke}",
        flush=True,
    )

    run_one_cell(
        arch_name="tsae_paper",
        seed=args.seed,
        protocol="v7",
        n_concepts=30,
        strengths=DEFAULT_STRENGTHS,
        coh_thresholds=DEFAULT_COH_THRESHOLDS,
        n_steps=args.n_steps,
        smoke=args.smoke,
        force_train=args.force_train,
        force_eval=args.force_eval,
        train_window_size=2,        # ← Bhalla/Ye 2025 paper-faithful
    )


if __name__ == "__main__":
    main()
```

Step 4 — write `experiments/c5_steering_baseline/run_sweep.sh`:

```bash
#!/usr/bin/env bash
# Launch the 3-cell C5 T-SAE baseline T=2 re-train in parallel.
# 3 seeds × 1 arch (tsae_paper) on GPUs 0..2.

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

declare -A ASSIGN=(
  [0]="42"
  [1]="1"
  [2]="2"
)

for gpu in "${!ASSIGN[@]}"; do
  seed="${ASSIGN[$gpu]}"
  log="logs/c5_baseline_gpu${gpu}_tsae_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → tsae_paper seed=${seed} → ${log}"
  bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c5_steering_baseline.run \
    --seed "${seed}" \
    > "${log}" 2>&1 &
  echo $! > "/tmp/p_baseline_gpu${gpu}"
done

echo "[run_sweep] launched 3 parallel cells; PIDs in /tmp/p_baseline_gpu{0..2}"
echo "[run_sweep] tail -f logs/c5_baseline_gpu*.log to monitor"
wait
echo "[run_sweep] all 3 cells complete"
```

Step 5 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c5_steering_baseline.run \
  --seed 42 --n-steps 200 --smoke 2>&1 | tail -25
```

Verify smoke landed at fresh train_key (distinct from agent_steer's
T=None cells):

```bash
tail -1 results/leaderboard.jsonl | .venv/bin/python -c "
import json, sys
r = json.loads(sys.stdin.read())
print('arch=', r['arch'], 'seed=', r['seed'], 'train_key=', r['train_key'])
print('eval_cfg.smoke=', r['eval_cfg'].get('smoke'))
print('peak_grade@1.75=', r['metrics'].get('peak_success_grade_at_coh_1.75'))
"
```

Step 6 — launch the full 3-cell sweep:

```bash
bash experiments/c5_steering_baseline/run_sweep.sh
```

Step 7 — monitor 3 logs in parallel:

```bash
tail -f logs/c5_baseline_gpu*.log
```

Each cell ETA ~45-60 min. As cells complete, verify rows land at
`arch=tsae_paper`, `eval_protocol_version=1.1.0`, with new train_keys
distinct from agent_steer's T=None cells.

Step 8 — when sweep lands, agent_paper / agent_steer integrate at
paper-render time. agent_steer's `experiments/c5_steering/analysis.py`
will pick up your cells via two `canonical_train_keys()` calls (TXC at
T=None, T-SAE at T=2). Don't render `docs/components/c5.md` yourself.

**Watch-outs**:

- **Don't re-run TXC.** agent_steer's existing v1.1.0 TXC cells stand;
  re-running them is wasted compute.
- **Don't import `my_eval_fn`** from `c5_steering.run` — it's a
  closure built by `_make_eval_fn(seed, workspace, eval_key)`. Use
  `run_one_cell` (the wrapper handles closure plumbing). Same gotcha
  the original C5 MW briefing flagged.
- **Don't pursue Y/W steering hill-climb winners** — Galaxy 8/11/18 /
  SoftMaxPool / ContrastiveMergeH8 are excluded by decision § 1.
- **Don't enable Bricken** — C5 is Bricken-off per decisions § 7.
- **Don't render anything to docs/components/c5.md** — agent_paper
  integrates at paper-render time.
- **Watch for Anthropic API credit issues** — agent_steer hit one
  earlier. If your judge phase produces `n_valid=0`, check
  `results/runs/<eval_key>/judge_outputs.jsonl` for "credit balance is
  too low" errors and surface to Han before re-running. Recovery is
  `--force-eval` on the cached training checkpoint.

---

### Mandate — C5 multi-window deployment, parallel [RESCINDED 2026-05-05 PM — see STAND DOWN above]

agent_paper landed `txc_base_mw` and `txc_pro_mw` as separate arch
identities in `configs/locked_archs.yaml` (decisions.md § 14). They
are YAML aliases of TXCBase / TXCPro respectively, with
`multi_window: true` baked into hparams. The Python classes are
identical to `txc_base` / `txc_pro`; only the per-step sampling
differs (stride-T tiling for TXC-base, stride-(T_max+max_shift)
tiling for TXC-pro). Decision summary:

- Per-token archs (TopK / T-SAE / SAE-arditi / etc.) see ~131K tokens
  per step at our canonical (B=1024, seq_len=128) setting.
- Non-MW TXC archs see only ~5K tokens per step (1 random window per
  row) — a ~25× FLOPs disadvantage per step.
- MW TXC archs match per-token throughput by tiling each sequence into
  N non-overlapping windows.

**Your mission**: run the C5 MW canonical sweep — `txc_base_mw` and
`txc_pro_mw` × seeds {42, 1, 2} = **6 cells total** — in parallel
across 6 of your 8 A40 GPUs. agent_steer's existing v1.1.0 cells stay
as the comparison baseline; your `_mw` cells go alongside them in
`leaderboard.jsonl`. agent_paper toggles which is canonical at
paper-render time by changing the `training_cfg=` argument in
`canonical_train_keys()` in c5_steering's analysis.py.

**Why this pod**: the 8× A40 setup has no per-GPU compute advantage
over agent_steer's 4× A40 pod (same A40 SKU, ~149 BF16 TFLOP/s each).
The advantage is **wall-clock parallelism**: with 6 cells on 6 GPUs,
total wall-time = `max(per-cell time)` not `sum`. agent_steer_100k
on the H100 is independently running C5 MW too, but they're slow per
cell; you're the redundancy + speed-via-parallelism path.

**Why the existing MW work isn't being abandoned**: agent_steer_100k
continues their (slower) H100 C5 MW sweep. Whichever sweep finishes
first (theirs or yours) provides the data; the other is bonus
robustness. Your sweep is expected to land first because of the
parallelism advantage.

Hardware: **8× A40 48GB pod, 401 GB system RAM, 76 vCPU, 1 TB
ephemeral /workspace**. Pod mode `ephemeral`: HF is the source of
truth, auto-push on checkpoint save, fatal on push failure.

Your own python process is pinned to **GPU 0** via
`scripts/set_agent_env.sh agent_filler`. To launch parallel cells
on GPUs 0..7, you spawn subprocesses via
`bash scripts/run_on_gpu.sh <idx> -- <cmd>`, which sets
`CUDA_VISIBLE_DEVICES=<idx>` for the child only. You do NOT use DDP
(framework decision per `docs/paper/hardware.md` § Single-GPU vs
multi-GPU).

VRAM check (per agent_paper's analysis): C5 at Gemma-scale
(d_in=2304, d_sae=18432) MW peaks ~20-25 GB activations, well within
the 48 GB A40 cap. No bf16 forcing or batch reductions needed.

Subject + protocol (replicating agent_steer's setup verbatim with
the multi-window arch swap):

- Datasource: `gemma_2_2b_it_l13_fineweb_24k128`
- Architectures: **`txc_base_mw` + `txc_pro_mw`** (2 archs total).
  These are the YAML aliases per decisions.md § 14. **NOT
  `tsae_paper`** — agent_steer's existing v1.1.0 tsae_paper cells are
  the canonical T-SAE comparison; running it again is wasted compute
  (no MW variant exists for non-TXC archs).
- Per-component d_sae overrides: locked_archs.yaml's
  `per_component_hparams.c5` for both `txc_base_mw` and `txc_pro_mw`
  applies automatically.
- Steering: V7 tiled-broadcast residual-stream protocol (per
  `temp_bench.case_studies.steering` — agent_steer's port).
- Concept set: same 30-concept × 5-example-sentence set agent_steer
  uses.
- **Per-arch best-feature selection**: USE THE FIXED v1.1.0
  concept-lift baseline (commit `ef33f822`, agent_steer's
  `select_best_features` in `temp_bench.case_studies.steering`).
  Your import path automatically picks this up.
- Judge: Sonnet 4.6 (Anthropic; per decisions.md § 12).
- Per-call `judge_outputs.jsonl` persistence for post-deadline κ.
- Headline metric: `peak_success_grade_at_coh_1.75`.
- `EVAL_PROTOCOL_VERSION`: **"1.1.0"**.

Seeds: **{42, 1, 2}** (full n=3). 6 cells total.

`TrainingConfig` for your cells (canonical schedule):

```python
TrainingConfig(
    batch_size=1024,
    n_steps=20_000,
    plateau_early_stop=False,
    # bricken_* defaults (False) — C5 does not use Bricken (decisions.md § 7).
)
```

Per-cell wall-time estimate on A40: agent_steer's non-MW cells took
~25-40 min for tsae_paper, ~50-80 min for txc_base, ~120-150 min for
txc_pro. MW adds ~5-10× per-step compute (matryoshka decode + InfoNCE
all-pairs scaling for txc_pro). Expect:

- `txc_base_mw`: ~3-6 hr per cell
- `txc_pro_mw`: ~10-15 hr per cell (the InfoNCE all-pairs is the hot path)

With 6 cells in parallel on 6 GPUs, **total wall ≈ max(per-cell) ≈ 10-15
hr** (the slowest cell determines completion). Comfortably within the
remaining sprint window.

Locked decisions in scope: #1 (canonical TXCs are `txc_base_mw` /
`txc_pro_mw` per § 1's 2026-05-05 amendment), #4, #6, #7 (Bricken
off for C5), #11, § 12 (canonical training cfg), § 14 (multi-window
deployment).

References:
- `agents/README.md` (your roster row + pod specs)
- `agents/agent_steer/briefing.md` (the canonical C5 setup you
  replicate — read this for context BEFORE launching, especially the
  v1.1.0 concept-lift fix story)
- `docs/components/c5.md` (the canonical C5 writeup; do NOT edit)
- `experiments/c5_steering/{run.py,analysis.py}` (import from)
- `decisions.md` § 7, § 12, § 14
- `papers/temporal_sae.md` § B.2 (T-SAE steering protocol reference)
- `PROTOCOL.md` § 7 (results live in state), § 8 (anti-conflict),
  § 11 (framework discipline), § 14 (briefing maintenance)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed. Push your MW cells'
`results/runs/<eval_key>/judge_outputs.jsonl` and metrics via the
wrap-up script before any pod restart.

### First concrete task — write a parallel-launch driver, then sweep

Step 1 — `git pull --rebase origin final`, verify infra:

```bash
grep "baseline = activation_matrix.mean" src/temp_bench/case_studies/steering.py
# → should match (v1.1.0 concept-lift fix)
.venv/bin/python -c "from experiments.c5_steering.run import EVAL_PROTOCOL_VERSION; print(EVAL_PROTOCOL_VERSION)"
# → expect "1.1.0"
.venv/bin/python -c "from temp_bench.config import load_arch; print(load_arch('txc_base_mw').hparams)"
# → expect a dict containing multi_window=True
nvidia-smi --query-gpu=index,memory.used --format=csv
# → expect 8 GPUs, all <500 MB used
```

Step 2 — pull the activation cache. The Gemma cache is on HF and
~14 GB; pulling once fills the OS page cache for all 6 subprocesses
to share via the preloaded `.clone()` helper:

```bash
bash scripts/sync_from_hf.sh
# → pulls han1823123123/temp-bench-data act-cache for c5
```

Step 3 — write `experiments/c5_steering_filler/__init__.py` (empty)
and `experiments/c5_steering_filler/run.py`. Single-cell driver: take
`--arch` + `--seed` and delegate to agent_steer's `run_one_cell`, which
is the canonical entry point that builds the eval-fn closure (with seed
+ workspace + eval_key) and calls `runner.run_cell`. The top-level
launch script (`run_sweep.sh`) spawns 6 of these in parallel.

⚠ **Do NOT try to import `my_eval_fn` from
`experiments.c5_steering.run` directly** — it doesn't exist at module
level. agent_steer's `my_eval_fn` is built inside the
`_make_eval_fn(seed, workspace, eval_key)` closure factory because the
runner doesn't pass workspace/eval_key into eval_fn arguments, so the
case study persists artifacts (judge_outputs.jsonl, generations.jsonl,
metrics.json, plots) in the right ``run_dir(eval_key)`` only because
the closure has those values bound. `run_one_cell` is the wrapper that
threads those for you. agent_steer_100k hit this same import bug in
`experiments/c5_steering_mw/run.py` and worked around it the same way;
you can read their driver for a working reference (note: don't import
from their dir, just read the pattern).

```python
"""C5 multi-window driver for agent_filler — single cell per invocation,
called once per (arch, seed) pair pinned to a specific GPU. The
top-level launch script (run_sweep.sh) spawns 6 of these in parallel.
"""
from __future__ import annotations
import argparse
import os

# Match the threading defaults agent_steer_100k tuned for parallel runs;
# bit-identical math, just throughput. Tune lower if you see 6 procs
# fighting for the 76-core CPU pool.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from experiments.c5_steering.run import (
    EVAL_PROTOCOL_VERSION,
    run_one_cell,
)
from temp_bench.case_studies.steering import (
    DEFAULT_COH_THRESHOLDS,
    DEFAULT_STRENGTHS,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 multi-window deployment — single (arch, seed) cell."
    )
    ap.add_argument("--arch", required=True,
                    choices=["txc_base_mw", "txc_pro_mw"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps. Default = "
                         "agent_steer's canonical 20_000; use small "
                         "values for smoke tests.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tag the leaderboard row eval_cfg.smoke=True.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    print(
        f"[c5_filler] cell arch={args.arch} seed={args.seed} "
        f"eval_protocol={EVAL_PROTOCOL_VERSION} "
        f"smoke={args.smoke} n_steps_override={args.n_steps}",
        flush=True,
    )

    run_one_cell(
        arch_name=args.arch,
        seed=args.seed,
        protocol="v7",
        n_concepts=30,
        strengths=DEFAULT_STRENGTHS,
        coh_thresholds=DEFAULT_COH_THRESHOLDS,
        n_steps=args.n_steps,
        smoke=args.smoke,
        force_train=args.force_train,
        force_eval=args.force_eval,
    )


if __name__ == "__main__":
    main()
```

Why `run_one_cell` not `runner.run_cell` directly: `run_one_cell`
(defined in `experiments/c5_steering/run.py`) is a thin wrapper that:
(a) calls `_make_eval_fn(seed, workspace, eval_key)` to bind the eval
closure, (b) precomputes `eval_key` so the workspace dir matches what
the runner uses, (c) passes the un-enriched `eval_cfg` to `run_cell`
(the runner re-hashes it; if you pass enriched cfg you get a
different eval_key + the run_dir mismatches metrics.json). All of
that is fragile to redo by hand, and agent_steer already debugged it
(see commit `f8a28469`).

Step 4 — write `experiments/c5_steering_filler/run_sweep.sh`, the
parallel launcher. Pins one cell per GPU; the 6 cells × 6 GPUs leaves
GPUs 6 + 7 idle (reserve them for retries or stretch):

```bash
#!/usr/bin/env bash
# Launch the 6-cell C5 MW sweep in parallel.
# Each subprocess pinned to its own GPU via run_on_gpu.sh.

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

# (arch, seed) → GPU assignment. Slow archs (txc_pro_mw) get GPUs 0..2;
# fast archs (txc_base_mw) get GPUs 3..5. Spreads load + lets
# txc_base_mw potentially finish early and free a GPU for retries.
declare -A ASSIGN=(
  [0]="txc_pro_mw 42"
  [1]="txc_pro_mw 1"
  [2]="txc_pro_mw 2"
  [3]="txc_base_mw 42"
  [4]="txc_base_mw 1"
  [5]="txc_base_mw 2"
)

for gpu in "${!ASSIGN[@]}"; do
  read -r arch seed <<<"${ASSIGN[$gpu]}"
  log="logs/c5_filler_gpu${gpu}_${arch}_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → ${arch} seed=${seed} → ${log}"
  bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c5_steering_filler.run \
    --arch "${arch}" --seed "${seed}" \
    > "${log}" 2>&1 &
  echo $! > "/tmp/p_filler_gpu${gpu}"
done

echo "[run_sweep] launched 6 parallel cells; PIDs in /tmp/p_filler_gpu{0..5}"
echo "[run_sweep] tail -f logs/c5_filler_gpu*.log to monitor"
wait
echo "[run_sweep] all 6 cells complete"
```

Step 5 — smoke-test ONE cell at `n_steps=200` before the full launch
to verify MW + v1.1.0 fix + multi-process safety. Use the driver
from Step 3 (NOT a one-liner; the closure plumbing matters):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c5_steering_filler.run \
  --arch txc_base_mw --seed 42 \
  --n-steps 200 --smoke
```

Expected: cell takes ~3-5 min (200 train steps + brief 30-concept ×
9-strength generation + 540 Sonnet judge calls). On success, look
for `[hf-push] uploaded run_dir → ...` near the end of the log,
and confirm a leaderboard row appended:

```bash
tail -1 results/leaderboard.jsonl | .venv/bin/python -c "
import json, sys
r = json.loads(sys.stdin.read())
print(r['arch'], r['seed'], r['eval_protocol_version'],
      'n_valid=', r['metrics']['n_valid'])
"
# → expect: txc_base_mw 42 1.1.0 n_valid= 270  (or similar)
```

If the smoke row shows `n_valid=0`: judge fail-mode (e.g., Anthropic
credit outage); check `results/runs/<eval_key>/judge_outputs.jsonl`
for HTTP error messages and surface to Han before the full launch.

If smoke passes, launch the full sweep:

```bash
bash experiments/c5_steering_filler/run_sweep.sh
```

Step 6 — monitor. Persistent monitor on the 6 logs:

```bash
tail -f logs/c5_filler_gpu*.log
# OR via the Monitor tool if you prefer event-driven
```

Step 7 — as cells complete: confirm rows land in
`leaderboard.jsonl` with `arch=txc_base_mw` or `txc_pro_mw`,
`eval_protocol_version=1.1.0`. agent_paper integrates at paper-render
time.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: <fill in on first session>**

- `git HEAD`: <sha>
- Pod: 8× A40, ephemeral, 401 GB RAM, 76 vCPU, 1 TB /workspace.
- Active GPU usage: GPU 0 (own process); subprocesses on GPUs 0..5 once
  the sweep launches.
- Last leaderboard append: `(none yet)`.
- Recent decisions in scope: `decisions.md` § 7 (Bricken off for C5),
  § 12 (canonical training cfg), § 14 (multi-window deployment).

## What I just did (agent owns — overwrite)

(none yet — first session)

## Next action (agent owns — overwrite)

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_filler` (pins your own
   process to GPU 0; subprocesses pinned per `run_on_gpu.sh <idx>`).
3. `bash scripts/agent_smoke_test.sh` — expect 131/131 + preflight green.
4. `bash scripts/sync_from_hf.sh` — pulls Gemma activation cache.
5. `git pull --rebase origin final` — stay current with agent_steer's
   v1.1.0 fix and agent_paper's MW arch landings.
6. Write `experiments/c5_steering_filler/{__init__.py, run.py, run_sweep.sh}`
   per Step 3 + Step 4.
7. Smoke-test ONE cell at `n_steps=200` per Step 5.
8. Launch the full 6-cell parallel sweep per Step 5's bottom command.
9. Monitor + verify leaderboard rows land at
   `eval_protocol_version=1.1.0`.

## Don't repeat (agent owns — overwrite)

- **Don't edit `experiments/c5_steering/` or `experiments/c5_steering_mw/`**
  — agent_steer's and agent_steer_100k's territories. Import only.
- **Don't edit `docs/components/c5.md`** — agent_paper integrates at
  paper-render time.
- **Don't bypass `runner.run_cell`** — the call goes through the
  canonical pathway (which appends to `leaderboard.jsonl`).
- **Don't allocate `train_key` / `eval_key` manually** — the runner
  computes them deterministically.
- **Don't include `tsae_paper` in your archs list** — agent_steer's
  v1.1.0 tsae_paper cells are the canonical T-SAE comparison; no MW
  variant exists for non-TXC archs.
- **Don't enable Bricken** — C5 is Bricken-off per decisions.md § 7.
- **Don't pursue Y/W steering hill-climb winners** — Galaxy 8/11/18
  / SoftMaxPool / ContrastiveMergeH8 are excluded by decision #1.
- **Don't push to HF manually** — `cache.save_checkpoint` does it on
  ephemeral pods.
- **Don't use DDP** — single-GPU subprocesses only, per
  `docs/paper/hardware.md`.
- **Don't import `my_eval_fn` from `experiments.c5_steering.run`**
  — it's a closure built by `_make_eval_fn(seed, workspace, eval_key)`,
  not a top-level symbol. Use `run_one_cell` instead (that's what the
  Step 3 driver does); it threads workspace + eval_key + seed for
  you. agent_steer_100k hit this same import bug; their workaround
  is the pattern your driver should follow.
- **Don't pass enriched `eval_cfg` to `runner.run_cell` directly** —
  the runner re-hashes `eval_cfg` to compute `eval_key`. If you add
  `_*` enrichment fields, the resulting `eval_key` won't match the
  workspace your case study writes to, and metrics.json + the
  leaderboard row will live in different `run_dir`s. agent_steer
  debugged this in commit `f8a28469`. Stick with `run_one_cell`.
- **Don't rely on `temp_bench.report.render(component='c5')`** — it
  raises `Multiple experiment dirs match c5_*` because of
  `experiments/c5_steering_100k/`, `experiments/c5_steering_mw/`,
  and now `experiments/c5_steering_filler/`. Render is agent_paper's
  job at paper-time; you don't need to render c5.md yourself. If
  you DO need to compute summary stats, query
  `temp_bench.report.query_leaderboard(component='c5')` directly
  and filter on `arch in ('txc_base_mw', 'txc_pro_mw')` +
  `eval_protocol_version == '1.1.0'`.
- **Don't kill in-flight cells just because etime looks long** —
  txc_pro_mw cells legitimately take 10-15 hr. Check the log for
  step-rate progress (`[TRAIN ... step XXXX/20000 (X.X steps/sec)`)
  before assuming a cell is hung.
- **Don't worry about Anthropic API credits during your sweep** —
  agent_steer hit a credit-exhaustion outage on 2026-05-05 05:59 UTC
  that produced an all-zero metrics row. Han topped up. If your
  judge phase produces `n_valid=0`, check
  `results/runs/<eval_key>/judge_outputs.jsonl` for "credit balance
  is too low" errors and surface to Han before re-running. Recovery
  is `--force-eval` on the cached training checkpoint (not a full
  re-train).

## Open questions for Han (agent owns — overwrite)

(None at briefing-write time. Surface anything that comes up during
the smoke test or parallel launch.)
