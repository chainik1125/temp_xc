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

### ✅ TFA BUG FIX 2026-05-06 — landed (commit `53e63fbb`)

**Your bug report from "Open questions" #3 is FIXED.** The C1 driver's
`_is_valid_cell` no longer treats TFA / TFA-pos as "no constraint".
agent_paper landed the fix at commit `53e63fbb`:

```python
# Before: TFA grouped with topk_sae / tsae_paper (per-token, no constraint).
# After: TFA gets the window-arch constraint at C1.
if arch_name in ("tfa", "tfa_pos"):
    T = int(hp.get("T", 5))
    return k_pos * T <= d_sae         # k_pos × 5 ≤ 40 → k_pos ≤ 8 at C1.
```

**No relaunch needed for the missing high-k TFA cells.** TFA at
`d_sae=40` (C1) is **architecturally bounded to k_pos ≤ 8**: at higher
k_pos the topk over `z_novel` (last dim = `width` = `d_sae` = 40) is
infeasible because `kval_topk = k_pos × T = k_pos × 5 > 40` crashes
`torch.topk` (which is exactly what you saw). The fix just makes the
driver skip them silently instead of crashing — the missing cells
**cannot exist at C1's toy d_sae=40**.

The 21/36 TFA cells you landed (k ∈ {1, 2, 3, 4, 5, 6, 8} × 3 seeds)
are the complete valid set for C1 TFA; high-k tail is genuinely
infeasible (not "missing data"). Pull commit `53e63fbb` to silence
the future re-run warning if you re-launch C1; otherwise no action
needed.

**Bug doesn't affect any other component** — agent_paper checked: at
C3/C5/C6/C7 production scale (d_sae ∈ {18432, 32768}), `k_pos × T = 100`
is well within bounds. Only C1's toy d_sae=40 triggered the crash.

---

### ⚠️ ADDITIONAL MISSION 2026-05-05 PM (URGENT) — Take HALF of agent_steer_100k's BASE C3 load

**Han 2026-05-05 PM**: "agent_filler's 8 A40s will eventually become
idle. this is massive compute we NEED TO UTILIZE." + "Let's make them
take half of agent_steer_100k's load."

**Current state of BASE C3** (after agent_steer_100k's 16/30 cells
checkpoint, commit `0a3b5b95`):
- ✅ `topk_sae`: 7 cells (~complete: 3 seeds × 2 k_feats + 1 extra)
- ✅ `tsae_paper`: 6 cells (complete: 3 seeds × 2 k_feats)
- 🟡 `tfa`: 4 cells (in flight on agent_steer_100k)
- ❌ `txc_base`: 0 cells — **NOT STARTED, your scope**
- ❌ `txc_pro`: 0 cells — **NOT STARTED, your scope**

agent_steer_100k owns the per-token + TFA archs (lighter compute,
their continued workflow). You take the **TXC variants**, which are
the slowest cells and benefit most from your 8-GPU parallelism. This
splits the wall-time roughly 50/50.

Launch this **once your current C5 TopK+TFA sweep + C1+C2 toy sweeps
wrap** (your `c1c2_toy_sweep.sh` is in flight per `5b107f2a`).

### Mission scope (BASE C3 TXC variants, mirrors agent_nlp's IT)

| Arch | T sweep | Seeds | k_feats | Cells |
|---|---|---|---|---:|
| `txc_base` | T=5 (canonical) + T=10 + T=20 (decisions § 17) | {1, 2, 42} | {5, 20} | 9 trainings + 18 evals |
| `txc_pro`  | T_max=10, t_sample=5 (canonical) | {1, 2, 42} | {5, 20} | 3 trainings + 6 evals |

**Total: 12 unique trainings + 24 evals.** All on the BASE datasource
`gemma_2_2b_base_l13_fineweb_24k128`. C4 BASE evals cache-hit on these
checkpoints (agent_steer_100k owns C4 wiring).

### Pre-req: pull the BASE caches from HF

agent_steer_100k built + HF-pushed both the BASE act_cache and BASE
probe_cache for Gemma. Pull them into your pod's `/workspace/`:

```bash
.venv/bin/hf download han1823123123/temp-bench-data \
    --repo-type dataset \
    --include "act_cache/<BASE_KEY>/**" \
    --local-dir results/

.venv/bin/hf download han1823123123/temp-bench-data \
    --repo-type dataset \
    --include "probe_cache/gemma_2_2b_base_l13_fineweb_24k128/**" \
    --local-dir results/
```

(Replace `<BASE_KEY>` with the act_cache_key from
`compute_act_cache_key(load_datasource('gemma_2_2b_base_l13_fineweb_24k128'))`
— derive locally; one-line python.)

### Driver

Reuse `experiments/c3_probing_base/run.py` (agent_steer_100k's BASE
driver, commit `831cd2ea` already supports T=10/T=20 via
`arch_hparams_override`). Filter to TXC archs only:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  .venv/bin/python -m experiments.c3_probing_base.run \
  --archs txc_base txc_pro \
  --seeds 42 1 2 \
  --k-feats 5 20 \
  > logs/c3_base_txc_filler.log 2>&1 &
```

For the txc_base T=10/T=20 sweep, the driver already has the per-arch
TrainingConfig list with the three T values (T=5/T=10/T=20). It
iterates all of them when `--archs txc_base` is passed.

### Wall-time on 8× A40 (parallel)

Per-cell estimates (agent_steer_100k's H100 numbers scaled to A40):
- `txc_base` T=5: ~1.5 hr train + ~30 min eval × 2 k_feats = ~2 hr
- `txc_base` T=10: ~2 hr + ~30 min × 2 = ~2.5 hr
- `txc_base` T=20: ~2.5 hr + ~30 min × 2 = ~3 hr
- `txc_pro`: ~3 hr + ~30 min × 2 = ~3.5 hr (InfoNCE all-pairs is the
  slow part)

12 cells across 8 GPUs in parallel: 8 cells in wall #1 (~2.5 hr),
4 cells in wall #2 (~3 hr) → **~5-6 hr total wall**.

(If your C5 TopK+TFA + C1+C2 sweeps haven't finished by the time you
read this, queue this BASE TXC sweep AFTER they wrap. Don't double-
allocate GPUs.)

### Parallelisation script

```bash
#!/usr/bin/env bash
# experiments/c3_probing_base/run_filler.sh — launches BASE TXC cells
# across 8 A40s. agent_steer_100k still owns per-token + TFA archs.
set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

# 12 cells split across 8 GPUs. Faster archs go to fewer GPUs;
# slower (txc_pro) gets dedicated allocation.
declare -A ASSIGN=(
  [0]="txc_base 42"          # T=5/10/20 × seed=42
  [1]="txc_base 1"
  [2]="txc_base 2"
  [3]="txc_pro 42"
  [4]="txc_pro 1"
  [5]="txc_pro 2"
  # GPUs 6+7 idle / available for retry on OOM
)

for gpu in "${!ASSIGN[@]}"; do
  read -r arch seed <<<"${ASSIGN[$gpu]}"
  log="logs/c3_base_filler_gpu${gpu}_${arch}_seed${seed}.log"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c3_probing_base.run \
    --archs "${arch}" --seeds "${seed}" \
    > "${log}" 2>&1 &
done
wait
```

### Watch-outs

- **Don't duplicate agent_steer_100k's archs.** They own TopK + T-SAE
  + TFA on BASE. You ONLY run `txc_base` (T=5, T=10, T=20) and
  `txc_pro`. Cross-check leaderboard before launching to confirm no
  overlap.
- **HF push is automatic** on ephemeral A40 pods (`cache.save_checkpoint`).
  Verify via `wrap_up_session.sh` before pod stop.
- **C4 BASE evals are agent_steer_100k's** — they cache-hit on your
  txc_base/txc_pro train_keys and run the qualitative judge against
  the concat corpora. You don't run C4.
- **Same `arch_hparams_override` mechanism** for T-sweep that the IT
  side uses (commit `dfd60850`). Different T → fresh train_keys; old
  cells (T=5 default) coexist with T=10 + T=20 cells in the
  leaderboard.
- **Don't render `docs/components/c3.md`** — agent_nlp's territory.
  agent_paper integrates BASE results at paper-render time via
  per-datasource `canonical_train_keys` filter splits.

---

### ⚠️ ADDITIONAL MISSION 2026-05-05 PM — C1 + C2 toy synthetic sweeps (spare A40s)

**Han 2026-05-05 PM**: "agent_filler has 8 A40s, can they leverage all
of them for C1 and C2!" Your C5 sweep below uses 6 of 8 GPUs; the
spare 2 (or all 8 once C5 wraps) can run **two synthetic sweeps**:
- **C1 — Markov-chain TopK sweep**: feature recovery AUC vs k_pos
  for 8 archs (TopK, T-SAE, TFA, TFA-pos, Stacked T={2,5}, TXC-base,
  TXC-pro). Driver in `experiments/c1_synthetic_topk/`.
- **C2 — Coupled HMM gAUC sweep**: global recovery AUC for 7 arch+T
  combos including TXC-pro T-modulation T_max ∈ {2, 5, 12}. Driver
  in `experiments/c2_synthetic_coupled/`.

agent_paper landed both frameworks + drivers in commits `dfd60850`
(C2) + the next commit (C1); you just RUN the sweeps, no driver
writing.

C1 and C2 are small + fast: ~30 sec/cell on A40 (toy d_sae=40,
d_in=40 or 256). Each fits in <1 hr wall on parallel A40s.

### Mission scope (C1)

| Sweep dim | Values |
|---|---|
| Archs × T | 8 combos: topk_sae, tsae_paper, tfa, tfa_pos, stacked_sae T={2,5}, txc_base, txc_pro (canonical T_max=10) |
| `k_pos` | 12 values: {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20} (driver auto-skips invalid k for window archs at toy d_sae=40) |
| Seeds | 3: {1, 2, 42} |
| `n_steps` | 30,000 (canonical for C1 toy training per docs/components/c1.md) |
| **Total** | **~200-220 cells** (after auto-skip of invalid k for txc_base/txc_pro/stacked_T=5 at high k) |

Per-cell metric: feature recovery AUC vs the 20 ground-truth feature
directions. Headline: AUC vs k_pos line per arch.

### Mission scope (C2)

| Sweep dim | Values |
|---|---|
| Archs × T | 7 combos (see `experiments/c2_synthetic_coupled/run.py:ARCH_TS`): topk_sae, stacked_sae T={2,5}, txc_base T=5, txc_pro T_max={2,5,12} (t_sample=2 fixed) |
| `k_pos` | 12 values: {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20} |
| Seeds | 3: {1, 2, 42} |
| `n_steps` | 30,000 (canonical for C2 toy training) |
| **Total** | **7 × 12 × 3 = 252 cells** |

Per-cell metrics: eAUC (vs M=20 emission features), gAUC (vs K=10
hidden features). Headline: gAUC vs k_pos by (arch, T).

### Driver invocations (both smoke-tested by agent_paper on 5090)

```bash
# C1
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  .venv/bin/python -m experiments.c1_synthetic_topk.run \
    [--archs ...] [--seeds ...] [--k-poses ...] [--n-steps 30000]

# C2
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  .venv/bin/python -m experiments.c2_synthetic_coupled.run \
    [--archs ...] [--seeds ...] [--k-poses ...] [--n-steps 30000]
```

Each driver iterates over all (arch, T-override, k_pos, seed) combos
and writes one row per cell to `results/leaderboard.jsonl`. C1 also
auto-skips invalid (arch, k_pos) combinations where k_train would
exceed the matryoshka prefix at toy d_sae=40 (mostly txc_base /
stacked_sae T=5 / txc_pro at k_pos > 8).

### Parallelisation strategy on 8× A40

Toy cells are tiny (~1.8 GB VRAM per cell — Han's MEMORY note from
the 5090 carries over). You can comfortably run 5-10 cells per GPU
in parallel.

**Recommended split for C1+C2 simultaneously on 8× A40**:

```bash
#!/usr/bin/env bash
# experiments/c1c2_toy_sweep.sh — run C1 + C2 across spare A40s.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

# C1 archs across GPUs 0-3 (one process per arch).
declare -A C1=(
  [0]="topk_sae tsae_paper"          # per-token archs (fast)
  [1]="tfa tfa_pos"                  # full-seq attention archs
  [2]="stacked_sae txc_base"         # window archs
  [3]="txc_pro"                      # subseq + matryoshka
)
for gpu in "${!C1[@]}"; do
  archs="${C1[$gpu]}"
  log="logs/c1_gpu${gpu}.log"
  echo "[c1_sweep] GPU ${gpu} → ${archs}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c1_synthetic_topk.run \
    --archs ${archs} \
    > "${log}" 2>&1 &
done

# C2 archs across GPUs 4-7.
declare -A C2=(
  [4]="topk_sae"
  [5]="stacked_sae"
  [6]="txc_base"
  [7]="txc_pro"
)
for gpu in "${!C2[@]}"; do
  arch="${C2[$gpu]}"
  log="logs/c2_gpu${gpu}.log"
  echo "[c2_sweep] GPU ${gpu} → ${arch}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c2_synthetic_coupled.run \
    --archs "${arch}" \
    > "${log}" 2>&1 &
done

wait
echo "[c1c2_sweep] all complete"
```

This uses all 8 GPUs (assuming your C5 sweep is done by then; otherwise
delay C1+C2 until C5 wraps, ~1.5 hr after launch). Expected wall: ~30-
60 min (limited by the slowest single arch — txc_pro at C2 with T=12
sweep × 12 k_pos × 3 seeds = 36 cells × ~30 sec = ~18 min).

### Smoke

agent_paper smoke-tested both drivers on 5090 (~18 cells total at
n_steps=200, all `smoke=True` flagged). Smoke rows are tagged +
filtered by analysis. Verify on your pod with one cell each:

```bash
# C1 smoke
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  bash scripts/run_on_gpu.sh 6 -- \
  .venv/bin/python -m experiments.c1_synthetic_topk.run \
  --archs topk_sae --seeds 42 --k-poses 5 --n-steps 200 --smoke

# C2 smoke
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  bash scripts/run_on_gpu.sh 7 -- \
  .venv/bin/python -m experiments.c2_synthetic_coupled.run \
  --archs txc_pro --seeds 42 --k-poses 5 --n-steps 200 --smoke
```

Each completes in <30 sec; verify rows land at
`component=c{1,2}, smoke=True`.

### Watch-outs (C1 + C2)

- **Toy sweeps run alongside your C5 sweep** — they share the pod's
  CPU/RAM (small, ~few GB combined). Use spare GPUs for C1+C2 if
  C5 still running; otherwise all 8 GPUs once C5 wraps.
- **No HF push needed for C1+C2** — toy data + 40-feature
  dictionaries are small enough that local-only artifacts are fine.
- **Don't render `docs/components/c1.md` or `c2.md`** — agent_paper
  handles rendering via the per-component `analysis.py`.
- **C1 driver auto-skips invalid (arch, k_pos)** combinations where
  k_train > matryoshka prefix at toy d_sae=40. Expect ~200-220 cells
  to actually run (out of 8 archs × 12 k × 3 seeds = 288 nominal).
- **Both sweeps idempotent** — re-running is safe. Cached cells are
  skipped via runner's checkpoint cache (per `train_key`).

---

### ⚠️ NEW MISSION 2026-05-05 PM — C5 TopK + TFA baselines (decisions § 16)

**Your prior C5 T-SAE T=2 mission is COMPLETE** (commit `3a654fab`).
New mission: **add TopK + TFA × 3 seeds each** to the C5 steering
benchmark. Two new baselines, parallel-launchable on your 8× A40 pod.

agent_steer's existing v1.1.0 cells stand:
- `tsae_paper × 3 seeds` (your prior T=2 sweep)
- `txc_base × 3 seeds`, `txc_pro × 3 seeds`

You ADD:
- `topk_sae × 3 seeds at T=1` (vanilla TopK, paper-faithful per § 15)
- `tfa × 3 seeds at B=32 + full seq` (wasteland-faithful, § 16)

### Mission scope (6 cells parallel on 6 of 8 A40s)

| Arch | seeds | TrainingConfig | tokens/step |
|---|---|---|---:|
| `topk_sae` | {42, 1, 2} | `B=1024, n_steps=20_000, train_window_size=1` | 1,024 |
| `tfa` | {42, 1, 2} | `B=32, n_steps=20_000, train_window_size=None` | 4,096 |

**Total: 6 cells.** All on the existing single-layer cache
`gemma_2_2b_it_l13_fineweb_24k128`. No new cache build needed.

### Wall-time on 8× A40 (6 GPUs used; 2 idle)

- TopK at T=1 on Gemma-2-2B: ~10-15 min train + ~30 min judge per cell
  = ~45 min per cell.
- TFA at B=32 + full seq on Gemma-2-2B: ~30-50 min train + ~30 min judge
  per cell = ~1-1.5 hr per cell.
- 6 cells parallel on 6 GPUs → wall = max(per-cell) ≈ **~1.5 hr**.

### TFA paper-faithful background

`origin/han-phase7-unification:experiments/phase7_unification/
train_phase7.py:312-353` shows TFA training uses **batch_size=32 +
full seq=128** specifically because TFA's attention tensor is heavy
(`B × T × d_sae` ~ 9.6 GB fp32 at d_sae=18432). The wasteland convention
gives 4096 tokens/step (close to SAEBench 2K canonical) AND 128 tokens
of context (paper Fig. 2(d) shows ~80% variance explained at 100+
tokens). We adopt that convention here.

### First concrete task — write driver, smoke, launch

Step 0 — `git pull --rebase origin final`. Verify framework:

```bash
.venv/bin/python -c "
from experiments.c5_steering.run import run_one_cell
import inspect
assert 'train_window_size' in inspect.signature(run_one_cell).parameters
print('OK')
"
```

Step 1 — write `experiments/c5_steering_baselines/{__init__.py, run.py}`.
Single-cell driver, takes `--arch <topk_sae|tfa>` + `--seed <N>`:

```python
"""C5 TopK + TFA baselines (decisions § 16) — single (arch, seed) cell.
Top-level run_sweep.sh launches 6 in parallel.
"""
from __future__ import annotations
import argparse, os

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


# Per-arch literature-faithful TrainingConfig overrides via run_one_cell
# kwargs.
ARCH_CFG = {
    "topk_sae": {"batch_size": None, "train_window_size": 1},   # B=1024 default
    "tfa":      {"batch_size": 32,   "train_window_size": None},  # B=32 wasteland
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH_CFG.keys()))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    cfg = ARCH_CFG[args.arch]
    print(
        f"[c5_baseline] {args.arch} seed={args.seed} "
        f"B={cfg['batch_size'] or 1024} T={cfg['train_window_size']} "
        f"smoke={args.smoke} n_steps={args.n_steps}",
        flush=True,
    )

    # run_one_cell now accepts both train_window_size + batch_size kwargs
    # (agent_paper landed the latter in this same push as part of § 16).
    # Both override the canonical TrainingConfig per-cell; different
    # values produce different train_keys.
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
        train_window_size=cfg["train_window_size"],
        batch_size=cfg["batch_size"],
    )


if __name__ == "__main__":
    main()
```

**B=32 plumbing for TFA**: `run_one_cell` accepts a `batch_size`
kwarg as of agent_paper's commit landing in this same push (§ 16).
Setting `batch_size=32` overrides the canonical TrainingConfig
default; different B → fresh `train_key`. No further plumbing needed.

Step 2 — write `experiments/c5_steering_baselines/run_sweep.sh`:

```bash
#!/usr/bin/env bash
# Launch 6-cell C5 TopK + TFA baselines in parallel.
# 3 seeds × 2 archs on GPUs 0..5.

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

declare -A ASSIGN=(
  [0]="topk_sae 42"
  [1]="topk_sae 1"
  [2]="topk_sae 2"
  [3]="tfa 42"
  [4]="tfa 1"
  [5]="tfa 2"
)

for gpu in "${!ASSIGN[@]}"; do
  read -r arch seed <<<"${ASSIGN[$gpu]}"
  log="logs/c5_baseline_gpu${gpu}_${arch}_seed${seed}.log"
  echo "[run_sweep] GPU ${gpu} → ${arch} seed=${seed} → ${log}"
  setsid -f bash scripts/run_on_gpu.sh "${gpu}" -- \
    .venv/bin/python -m experiments.c5_steering_baselines.run \
    --arch "${arch}" --seed "${seed}" \
    > "${log}" 2>&1 &
  echo $! > "/tmp/p_baseline_gpu${gpu}"
done

echo "[run_sweep] launched 6 parallel cells; PIDs in /tmp/p_baseline_gpu{0..5}"
echo "[run_sweep] tail -f logs/c5_baseline_gpu*.log to monitor"
wait
echo "[run_sweep] all 6 cells complete"
```

(Note `setsid -f` per your prior commit `a62e5d6a` so cells survive
session restarts.)

Step 3 — smoke ONE cell (TopK first, TFA after batch_size kwarg
plumbing lands):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_filler \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c5_steering_baselines.run \
  --arch topk_sae --seed 42 --n-steps 200 --smoke 2>&1 | tail -25
```

Step 4 — launch full sweep:

```bash
bash experiments/c5_steering_baselines/run_sweep.sh
```

Step 5 — monitor + verify. Per-cell wall ~45 min (TopK) / ~1-1.5 hr (TFA).
6 cells parallel → wall ~1.5 hr.

### Watch-outs

- **B=32 for TFA** is intentional per § 16. Document in c5.md caveats.
- **Don't re-run T-SAE** — your v1.1.0 cells stand. Only TopK + TFA
  are new.
- **Don't pursue Y/W steering hill-climbers** (decision § 1).
- **Watch for Anthropic API credit issues** — agent_steer hit one
  earlier. Check `judge_outputs.jsonl` if any cell shows `n_valid=0`.
- **Don't render `docs/components/c5.md`** — agent_paper integrates at
  paper-render time. agent_steer's analysis.py will need a 4-call
  `canonical_train_keys` filter (TXC at None, T-SAE at T=2, TopK at
  T=1, TFA at B=32) once your cells land.

---

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

**Last verified: 2026-05-05T17:25Z**

**STATUS: in-flight — C5 TopK + TFA baselines sweep launched (§ 16).**

Prior mission COMPLETE: C5 T-SAE T=2 baseline (3 cells, mean
peak@1.75=1.93). Pushed at commit `dd7ef9b6` (now in history below
the new commits).

Current mission: 6-cell sweep on GPUs 0..5:
- GPUs 0..2: `topk_sae × {42, 1, 2}` at B=1024, train_window_size=1
  (paper-faithful per § 15)
- GPUs 3..5: `tfa × {42, 1, 2}` at B=32, train_window_size=None
  (full-seq, wasteland-faithful per
  `origin/han-phase7-unification:experiments/phase7_unification/train_phase7.py`)

VRAM verified at ~17:25Z (1-2 min in):
- TopK GPUs: 2.5 GB / 88-92% util
- TFA GPUs:  18.5 GB / 100% util (TFA's attention tensor; under 48 GB cap)
- GPUs 6, 7: idle

Smoke topk_sae seed=42 n_steps=200 smoke=True passed first:
train_key=`24bc835e77a4f1b8`, n_valid=270/270, peak@1.75=0.320,
agent=agent_filler, eval_protocol_version=1.1.0. Confirms the
`run_one_cell` framework path with both `train_window_size` +
`batch_size` kwargs (§ 16).

Sweep launcher (`run_sweep.sh`) uses `setsid -f` per the prior MW
sweep's lesson. All 6 PIDs orphaned to init (PPID=1, own SID).

Active monitor: task id `bev83i4r2` (1-hr timeout). Re-arm if it
times out before TFA cells finish (~1.5 hr per cell).

Local `final` == `origin/final` at `28fbbe55` (driver + smoke pushed).

Aborted MW driver still on disk at `experiments/c5_steering_filler/`
with its smoke row at eval_key=`8c6bf97f2de60679` in leaderboard.
Same disposition as before — open question for Han re: removal.

## What I just did (agent owns — overwrite)

The session had THREE halves: (a) the rescinded C5 MW sweep, (b) the
STAND DOWN, (c) the C5 T-SAE T=2 baseline mission. Recorded
chronologically for git provenance.

**Phase A — C5 MW sweep (rescinded):**

1. Wrote `experiments/c5_steering_filler/` driver, smoke-passed
   txc_base_mw seed=42 (n_valid=270, peak@1.75=0.615), launched
   6-cell sweep at 12:21Z.
2. CC restarted, killed the cells. Patched run_sweep.sh to `setsid -f`
   for orphan-to-init. Relaunched at 12:31Z; verified PPID=1.
3. By 13:37Z all 6 cells training healthy.

**Phase B — STAND DOWN execution (13:55Z):**

4. Pulled origin/final, saw `dd5f773e Agent PAPER: STAND DOWN`.
5. `pkill -KILL -f "experiments.c5_steering_filler"` — all 6 procs
   gone, GPUs idle. Stopped Monitor `b25o1m7v8`.
6. Resolved leaderboard.jsonl rebase conflict (kept all rows).
   Briefly went into detached HEAD state during a botched rebase;
   recovered via `rebase --abort` + manual file move.
7. Committed STAND DOWN ack at `ba369f13`.

**Phase C — C5 T-SAE T=2 baseline (current mission, 14:05Z+):**

8. Pulled origin/final again, saw new mandate in briefing
   (decisions § 15: T-SAE re-train at `train_window_size=2`).
9. Verified framework changes: `TrainingConfig.train_window_size=2`
   and `run_one_cell` accepts `train_window_size: int | None = None`
   kwarg.
10. Wrote `experiments/c5_steering_baseline/{__init__.py, run.py,
    run_sweep.sh}` — fresh dir, distinct from rescinded
    c5_steering_filler/. Driver hard-codes `arch_name="tsae_paper"`
    and `train_window_size=2`. Sweep launcher uses `setsid -f`.
11. Smoke launched at ~14:05Z on GPU 0 (tsae_paper seed=42 n_steps=200
    smoke=True). Bash task `bzgkr3t7e`, Monitor `bj2gsw0sv`. Training
    started — GPU 0 hit ~89% util at 2.5 GB VRAM by 14:53Z log
    timestamp; eval_key=`95ff152cb0b31650`.
    - Note: `[train] T=1` in the log refers to `model.T` (T-SAE arch's
      internal T parameter, fixed at 1 — adjacent-pair training with
      anchor + 1-step neighbor). The `train_window_size=2` is the BATCH
      ITERATOR's stride (positions per row per step), not the model's T.
      Both are correct; don't conflate.

## Next action (agent owns — overwrite)

1. **Wait for smoke to finish** (~3-5 min total; training started ~14:53Z).
   Bash poller `bm1769em4` is watching the python PID; Monitor
   `bj2gsw0sv` will fire on `judge [270/270]`, errors, etc.
2. **On smoke success** (n_valid=270, fresh train_key distinct from
   agent_steer's existing tsae_paper T=None train_keys): launch the
   3-cell sweep:
   `bash experiments/c5_steering_baseline/run_sweep.sh`
3. **Sweep monitoring**: tail `logs/c5_baseline_gpu*.log`. Per-cell
   ETA ~45-60 min; total wall = max(per-cell) ≈ 45-60 min.
4. **As cells complete**: confirm leaderboard rows have
   `arch=tsae_paper`, `eval_protocol_version=1.1.0`, fresh train_keys
   (different from agent_steer's existing T=None tsae_paper rows
   `6af5d868f65c4a6c` etc).
5. **When all 3 cells done**: `bash scripts/wrap_up_session.sh` for HF
   push of run_dirs / judge_outputs.jsonl. Surface to agent_paper that
   the T=2 baseline is on the leaderboard for paper-render integration.
6. **Before exit / context compact**: re-overwrite this section.
   The sweep python PIDs are detached via setsid (PPID=1) so they
   survive shell death; check via
   `pgrep -f "experiments.c5_steering_baseline.run"`.

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
  driver does); it threads workspace + eval_key + seed for you.
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
  job at paper-time; you don't need to render c5.md yourself.
- **Don't kill in-flight cells just because etime looks long** —
  txc_pro_mw cells legitimately take 10-15 hr. Check the log for
  active step progress before assuming a cell is hung.
- **Don't `tee` to `logs/...` without `mkdir -p logs` first.** The
  smoke shell exited with code 1 because tee couldn't open the file
  (logs/ didn't exist when the pipeline opened, despite the same Bash
  call doing `mkdir -p logs` later — too late). Either `mkdir -p` in
  a SEPARATE Bash call before launch, or include it in `run_sweep.sh`
  (which already does this).
- **Don't worry about Anthropic API credits during your sweep** —
  agent_steer hit a credit-exhaustion outage on 2026-05-05 05:59 UTC
  that produced an all-zero metrics row. Han topped up; the smoke's
  270/270 calls all returned 200 OK. If a future cell's judge phase
  produces `n_valid=0`, check
  `results/runs/<eval_key>/judge_outputs.jsonl` for "credit balance
  is too low" errors and surface to Han before re-running. Recovery
  is `--force-eval` on the cached training checkpoint (not a full
  re-train).
- **Don't use `tail -F logs/c5_filler_gpu*.log` in a Monitor without
  `-q`.** Without `-q`, tail emits `==>` headers on every file switch
  and they fire as Monitor events constantly. Use `tail -qF` (which is
  what the active monitor uses).
- **Don't launch the sweep with `&` + `wait`.** That makes cells
  children of the shell — when CC restarts they die from SIGHUP/SIGTERM
  propagation. The first launch (12:21Z) was killed exactly this way.
  Use `setsid -f` (current `run_sweep.sh` does this) so each cell is
  reparented to PID 1 and survives shell death.

## Open questions for Han (agent owns — overwrite)

1. **Should the smoke MW row at eval_key=`8c6bf97f2de60679` be
   retracted from leaderboard.jsonl?** It's tagged `smoke=true` so
   `canonical_train_keys` filters it out at paper-render time, but
   if you want a fully clean post-STAND-DOWN leaderboard, agent_paper
   could land a retraction. Default: leave it (append-only convention).
2. **Should `experiments/c5_steering_filler/` be removed entirely?**
   The driver is rescinded along with the MW mandate. Keeping it
   means future runs of the smoke (e.g., to re-validate v1.1.0) work
   out of the box; deleting it tidies the experiments tree. Default:
   keep until the paper ships.
3. **C1 driver's `_is_valid_cell` doesn't auto-skip TFA at high
   `k_pos`.** TFA crashed at `k_pos=10` seed=1 with
   `RuntimeError: selected index k out of range` in
   `_tfa_module.py:216 torch.topk(z_novel, kval, dim=-1)` — `z_novel`'s
   last dim is < 10 at toy d_sae=40. The briefing § C1 says
   "driver auto-skips invalid k for window archs ... mostly txc_base /
   stacked_sae T=5 / txc_pro at k_pos > 8" but TFA wasn't included.
   Workaround: GPU 1 restarted with `--archs tfa_pos stacked_sae`
   (skipping TFA entirely after k≤8 cells were already in). 21/36
   TFA cells landed (k ∈ {1,2,3,4,5,6,8} × 3 seeds); high-k cells
   missing. Fix needs `_is_valid_cell` patched to skip TFA at
   k_pos ≥ 10. Agent_paper's territory (driver in
   `experiments/c1_synthetic_topk/run.py` was landed by them).
4. **CRITICAL: BASE C3 `txc_base` T-override eval is broken.**
   `experiments/c3_probing_base/run.py:my_eval_fn:130` (agent_steer_100k's
   driver) crashes at T=10 eval:
   ```
   RuntimeError: Error(s) in loading state_dict for TXCBase:
     size mismatch for W_enc: shape [5, 2304, 18432] from checkpoint
     vs current [10, 2304, 18432]
   ```
   The eval_fn instantiates a T=10 model (matching the override) but
   loads the T=5 checkpoint's state_dict. Bug = T-override path doesn't
   thread the right state. Affects all T=10 + T=20 evals on txc_base
   for BASE C3.
   **Impact:**
   - T=5 cells: ✅ Done (6 rows in leaderboard from earlier T=5 work).
   - T=10 cells: training works (checkpoints saved + HF-pushed) but
     eval crashes the proc.
   - T=20 cells: same — training will work, eval will crash.
   **Status:** seed=42 already crashed; seeds 1+2 will hit the same
   bug shortly. I'm letting their T=10 trainings finish (so the
   checkpoints land on HF for re-eval after the fix), but they'll
   crash at eval and not advance to T=20.
   **Fix needed:** agent_steer_100k's driver `my_eval_fn` must use the
   freshly-trained state_dict (from train_fn's return) for the
   T-override cells, not the cached T=5 state. Likely involves making
   sure the runner's pipeline threads the per-cell state_dict to the
   eval correctly when arch_hparams_override changes the model shape.
   After fix, re-eval the cached T=10 + T=20 checkpoints with
   `--force-eval`.
   This is **agent_steer_100k's territory** — surfacing to them.

(Surface either as a comment if you want me to take action; otherwise
I'll leave them as-is.)
