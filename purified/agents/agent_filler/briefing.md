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

### Mandate — C5 multi-window deployment, parallel

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
and `experiments/c5_steering_filler/run.py`. Single-cell driver,
identical in logic to agent_steer_100k's `c5_steering_mw/run.py` but
takes `--gpu` to pin its CUDA device explicitly (since you'll launch
6 of these in parallel via `run_on_gpu.sh`):

```python
"""C5 multi-window driver for agent_filler — single cell per invocation,
called once per (arch, seed) pair pinned to a specific GPU. The
top-level launch script (run_sweep.sh) spawns 6 of these in parallel.
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from experiments.c5_steering.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
    _real_training_cfg as _orig_training_cfg,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["txc_base_mw", "txc_pro_mw"])
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    cfg = _orig_training_cfg()    # 20K, batch=1024, plateau_off
    print(f"[c5_filler] cell arch={args.arch} seed={args.seed} "
          f"n_steps={cfg.n_steps} eval_protocol={EVAL_PROTOCOL_VERSION}")

    runner.run_cell(
        component="c5",
        arch_name=args.arch,
        seed=args.seed,
        datasource_name=DATASOURCE,
        training_cfg=cfg,
        eval_cfg={"sweep": "c5_filler_v1"},
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=my_eval_fn,
    )


if __name__ == "__main__":
    main()
```

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
to verify MW + v1.1.0 fix + multi-process safety:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -c "
from temp_bench.schemas import TrainingConfig
from temp_bench import runner
from experiments.c5_steering.run import DATASOURCE, EVAL_PROTOCOL_VERSION, my_train_fn, my_eval_fn
result = runner.run_cell(
    component='c5', arch_name='txc_base_mw', seed=42,
    datasource_name=DATASOURCE,
    training_cfg=TrainingConfig(n_steps=200, batch_size=1024, plateau_early_stop=False),
    eval_cfg={'sweep': 'c5_filler_smoke'},
    eval_protocol_version=EVAL_PROTOCOL_VERSION,
    train_fn=my_train_fn, eval_fn=my_eval_fn,
)
print('smoke result:', result.train_key, result.eval_key, result.cached)
"
```

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

## Open questions for Han (agent owns — overwrite)

(None at briefing-write time. Surface anything that comes up during
the smoke test or parallel launch.)
