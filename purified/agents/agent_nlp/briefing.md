<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_nlp; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_nlp
last_state_update: 2026-05-06T05:35:00Z
status: complete
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

### ⚠️ RESUME MISSION 2026-05-06 — C3 k_feats expansion (post-RunPod-rescue, 7 cells, split with agent_em_100k)

**Han 2026-05-06**: post-rescue, you (the new agent_nlp on the
recovered pod) are picking up the unfinished C3 k_feats expansion
that died mid-flight when RunPod blew up. **The rescue you completed
(commit `eaa75a10`) saved the trained checkpoints; now run the 13
missing eval cells.** Han split them evenly between you and
agent_em_100k since you only have 1× H100 now (agent_em is gone).

### Your 7 cells (eval-only, all cache-hit on training)

```
tfa             seed=2  k_feat=80
tfa             seed=2  k_feat=160
tfa             seed=2  k_feat=320
tfa             seed=2  k_feat=640
txc_base T=20   seed=42 k_feat=160
txc_base T=20   seed=42 k_feat=320
txc_base T=20   seed=42 k_feat=640
```

Rationale: you finish the seed=2 tfa tail (cleanly closes one seed
column) **plus** the txc_base T=20 high-k tail (squarely in your § 17
T-sweep mission scope). agent_em_100k handles the seed=42 tfa column
entirely (6 cells, single arch, single seed).

### Driver invocations

```bash
cd /workspace/temp_xc/purified
git pull --rebase origin final

# tfa seed=2 high-k tail (4 cells)
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 2 --k-feats 80 160 320 640 \
  > logs/c3_kfeat_tfa_seed2_resume.log 2>&1 &

# txc_base T=20 seed=42 high-k tail (3 cells)
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing_txc_T_sweep.run \
  --T-values 20 --seeds 42 --k-feats 160 320 640 \
  > logs/c3_kfeat_T20_seed42_resume.log 2>&1 &
```

Both run on GPU 0 of your pod (you only have 1× H100). They serial
through the runner's per-cell loop; ~30 min/cell × 7 = ~3.5 hr total.

### Pre-launch sanity check

Before launching, verify the trained checkpoints are intact (you just
rescued them — they should be local + on HF):

```bash
.venv/bin/python <<'PY'
import json
need = []
# tfa seed=2 cells need tfa seed=2 train_key (B=32, full seq).
# txc_base T=20 seed=42 needs txc_base train_key with arch_hparams_override={"T": 20}.
from temp_bench.config import compute_train_key, load_arch, load_datasource, compute_act_cache_key
from temp_bench.schemas import TrainingConfig

ds = load_datasource('gemma_2_2b_it_l13_fineweb_24k128')
ack = compute_act_cache_key(ds)

# tfa seed=2 train_key
tfa_spec = load_arch('tfa', component='c3')
tfa_cfg = TrainingConfig(n_steps=20_000, batch_size=32)
tfa_tk = compute_train_key(arch=tfa_spec, seed=2, training_cfg=tfa_cfg, act_cache_key=ack)
print(f'tfa seed=2 train_key: {tfa_tk}')

# txc_base T=20 seed=42 train_key
txc_spec = load_arch('txc_base', component='c3')
txc_spec_T20 = txc_spec.model_copy(update={'hparams': {**txc_spec.hparams, 'T': 20}})
txc_cfg = TrainingConfig(n_steps=20_000, arch_hparams_override={'T': 20})
txc_tk = compute_train_key(arch=txc_spec_T20, seed=42, training_cfg=txc_cfg, act_cache_key=ack)
print(f'txc_base T=20 seed=42 train_key: {txc_tk}')

# Verify checkpoints exist locally
import os
for tk in [tfa_tk, txc_tk]:
    p = f'checkpoints/{tk}/model.safetensors'
    print(f'  {tk}: {"EXISTS" if os.path.exists(p) else "MISSING — pull from HF first"}')
PY
```

If a checkpoint is missing locally but exists on HF, pull it:

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('han1823123123/temp-bench-models', allow_patterns=['<TRAIN_KEY>/*'], local_dir='checkpoints/')
"
```

### After cells land — re-render

Once both your 7 + agent_em_100k's 6 = 13 cells finish:

1. Update `experiments/c3_probing/analysis.py` if it hardcodes
   `k_feats=(5, 20)` → expand to `(5, 10, 20, 40, 80, 160, 320, 640)`.
2. Re-render `docs/components/c3.md` AUTO-RESULTS via
   `temp_bench.report.render(component='c3')` to surface the full
   8-k_feat table.
3. **Commit + push** with a clear message tagging the rescue:
   ```
   Agent NLP: post-RunPod-rescue resume — C3 k_feats expansion 13/13 cells
              + analysis.py 8-k_feat update + c3.md AUTO-RESULTS re-render.
   ```

agent_em_100k doesn't touch c3.md (still your territory) — they just
land their 6 leaderboard rows + commit. Their commit ack will
explicitly mention it's a borrow into tfa territory for these 6 cells.

### Watch-outs

- **Eval-only.** Don't re-train. The runner cache-hits on training;
  only the per-k_feat probe runs.
- **agent_em_100k temporarily borrowing tfa territory** for their 6
  cells. Same `eval_protocol_version=1.1.0`, same probe logic. Cells
  dedupe via `eval_key`; no collision.
- **Don't bump `EVAL_PROTOCOL_VERSION`**. Existing rows stay valid.
- **HF auto-push** on save_checkpoint; for eval rows the leaderboard
  append is enough — no per-cell HF push needed.

---

### ⚠️ NEW MISSION 2026-05-06 (URGENT) — C3 k_feats expansion {5, 10, 20, 40, 80, 160, 320, 640}

**Han 2026-05-06**: "current C3 has k {5,20} we want to expand to
{5,10,20,40,80,160,320,640} for all SPARSE PROBES in C3 for both IT
and BASE!" Your job: **IT side**. agent_steer_100k handles BASE.

**This is eval-only — NO RE-TRAINING.** Your existing 6-arch C3 IT
checkpoints (TopK, T-SAE, TXC-base, TXC-pro, TFA, MLC × 3 seeds)
stay; just run the probing eval at 6 new k_feat values and let them
cache-hit on training.

### Mission scope

| Arch | Existing k_feats | New k_feats | New evals |
|---|---|---|---:|
| `topk_sae` | {5, 20} | {10, 40, 80, 160, 320, 640} | 3 seeds × 6 = 18 |
| `tsae_paper` | {5, 20} | {10, 40, 80, 160, 320, 640} | 18 |
| `txc_base` | {5, 20} | {10, 40, 80, 160, 320, 640} | 18 |
| `txc_pro` | {5, 20} | {10, 40, 80, 160, 320, 640} | 18 |
| `tfa` | {5, 20} | {10, 40, 80, 160, 320, 640} | 18 |
| `mlc` (multi-layer) | {5, 20} | {10, 40, 80, 160, 320, 640} | 18 |
| **Total new evals** | | | **108** |

Plus the txc_base T=10/T=20 cells if they landed (×3 seeds × 6 new
k_feats × 2 T values = 36 more). Total ~144 evals.

### Per-cell wall-time (eval-only)

The probing eval is dominated by: (a) load probe_cache + run SAE
forward over 38 SAEBench+CT tasks (~5-15 min on H100, fixed cost),
plus (b) per-k_feat probe fit (~seconds per k_feat per task). Adding
6 k_feats to an existing cell: ~30 min total (mostly the fixed encode).

- 18 evals per arch × 6 archs = 108 evals × ~30 min = ~54 hr serial
- On 2× H100 (own GPU 0 + agent_em's idle GPU 1 if available): **~27 hr**
- Idempotent: cache-hits on (eval_key for already-existing k_feats);
  only the 6 new k_feats per cell run.

If you want to compress wall-time, run each arch's 6 new k_feats in
ONE eval call (encode once, probe 6× per task) instead of 6 separate
calls. ~12-15 min per (arch, seed) cell instead of 6× ~30 min.

### First concrete task — extend the existing driver

Your existing `experiments/c3_probing/run.py` already takes
`--k-feats` as a list. Just relaunch with the new values:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing.run \
  --archs topk_sae tsae_paper txc_base txc_pro tfa \
  --seeds 42 1 2 \
  --k-feats 10 40 80 160 320 640 \
  > logs/c3_kfeat_expand_gpu0.log 2>&1 &

# MLC eval has its own driver path (multi-layer cache + 4D probe)
# in experiments/c3_probing_mlc/. Same --k-feats expansion:
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_mlc.run \
  --seeds 42 1 2 \
  --k-feats 10 40 80 160 320 640 \
  > logs/c3_kfeat_expand_mlc_gpu1.log 2>&1 &
```

Or if MLC is agent_em_100k's territory (their checkpoints), they own
the MLC k_feats expansion. Check with them before running.

### Analysis filter update

After cells land, extend `experiments/c3_probing/analysis.py` to
include all 8 k_feats in the headline tables. The probing eval row
schema already has `eval_cfg.k_feat`; `canonical_train_keys` filter
matches by train_key (one per (arch, seed)) and the analysis groups
by k_feat. **No filter change needed if the analysis already iterates
over distinct k_feats found in the leaderboard.**

If the existing analysis.py hardcodes `k_feats=(5, 20)`, update to
`k_feats=(5, 10, 20, 40, 80, 160, 320, 640)` and re-render the AUTO-
RESULTS block.

### Watch-outs

- **Eval-only.** Don't re-train. Pass `--seeds 42 1 2` with the same
  TrainingConfigs you used originally; the runner will hit cached
  checkpoints (fresh `eval_keys` only because `k_feat` differs).
- **GPU 1 borrow**: agent_em is idle (their canonical mission +
  C6 detection both COMPLETE). Borrow GPU 1 for parallel coverage.
- **MLC eval** uses the multi-layer probe_cache; agent_em_100k owns
  that. Coordinate with them on who runs MLC's k_feats expansion.
- **Don't bump `EVAL_PROTOCOL_VERSION`**. The probing protocol is
  unchanged — only the k_feat axis is wider. Existing rows at
  `k_feat ∈ {5, 20}` stay valid; new rows append at the new k_feats.

---

### ⚠️ NEW MISSION 2026-05-05 PM (URGENT) — TXC-base T-sweep on C3 + C4

**Han 2026-05-05 PM**: "we want a txc_base T=10 and T=20 on C3 and C4
(both IT and base)." Your job is the **IT side**;
agent_steer_100k handles the BASE side in parallel on their pod.

The locked TXC-base canonical is T=5. This mission adds **T=10 and
T=20 variants** to test whether longer windows help on the
sparse-probing + qualitative-latents axes. Per-cell train_key flows
through `TrainingConfig.arch_hparams_override` so the new cells get
fresh hashes; existing T=5 cells stay in `leaderboard.jsonl` as the
canonical headline.

### Mission scope

| Component | Arch × T | Seeds | k_feats | Cells |
|---|---|---|---|---:|
| C3 IT | txc_base × {T=10, T=20} | {1, 2, 42} | {5, 20} | 6 trainings + 12 evals |
| C4 IT | txc_base × {T=10, T=20} | {1, 2, 42} | (concat) | 6 evals (cache-hits on C3 trainings) |

**Total unique trainings: 6** (3 seeds × 2 T values for txc_base on
the IT cache). C4 evals re-use the C3 checkpoints (same `train_key`
→ same checkpoint).

### TrainingConfig

```python
# T=10 cells
TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
    arch_hparams_override={"T": 10},   # default txc_base T is 5
)

# T=20 cells
TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
    arch_hparams_override={"T": 20},
)
```

`arch_hparams_override` was landed by agent_paper for C2's T-sweep
(commit `dfd60850`); the runner merges it into `arch_spec.hparams`
before `compute_train_key`. Different T → fresh `train_key` →
fresh checkpoint. Per-cell encoder load: `B × T = 1024 × {10, 20}`
= 10K-20K tokens/step (still well within Gemma-2-2b L13 cache's
seq_len=128 — txc_base samples 1 random T-window per row, so
T=20 has 109 valid windows out of 128).

### Wall-time on 2× H100 (assume agent_em GPU 1 still idle)

- T=10 per cell: ~1.5 hr train + ~30 min eval × 2 k_feats = ~2 hr
- T=20 per cell: ~2 hr train + ~30 min eval × 2 k_feats = ~2.5 hr
- 3 seeds parallel on 2 GPUs (T=10 first, then T=20 on freed GPU) →
  **~6-7 hr wall** for both T values.

### First concrete task — write driver, smoke, launch

Step 1 — write `experiments/c3_probing_txc_T_sweep/{__init__.py,
run.py}` mirroring your existing `c3_probing_topk_baseline/run.py`
pattern, but with txc_base + arch_hparams_override:

```python
"""C3 TXC-base T-sweep (T=10, T=20). Han 2026-05-05 PM directive.
Adds cells alongside the canonical T=5 sweep; same datasource +
training schedule, only the T axis varies via arch_hparams_override.
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from experiments.c3_probing.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
)
from temp_bench.config import compute_act_cache_key, load_datasource


def _cfg(T: int) -> TrainingConfig:
    return TrainingConfig(
        n_steps=20_000,
        batch_size=1024,
        plateau_early_stop=False,
        arch_hparams_override={"T": int(T)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-values", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--n-steps", type=int, default=None)
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))

    for T in args.T_values:
        cfg = _cfg(T)
        if args.n_steps is not None:
            cfg = cfg.model_copy(update={"n_steps": args.n_steps})
        for seed in args.seeds:
            for k in args.k_feats:
                print(f"[c3_txc_T] cell txc_base T={T} seed={seed} "
                      f"k_feat={k} n_steps={cfg.n_steps}", flush=True)
                eval_cfg = {
                    "k_feat": k, "S": 32, "smoke": False,
                    "_act_cache_key": act_cache_key,
                    "_datasource_name": DATASOURCE,
                }
                runner.run_cell(
                    component="c3", arch_name="txc_base", seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=cfg, eval_cfg=eval_cfg,
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn, eval_fn=my_eval_fn,
                )


if __name__ == "__main__":
    main()
```

Step 2 — smoke (n_steps=200, T=10, seed=42, k_feat=5):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_txc_T_sweep.run \
  --T-values 10 --seeds 42 --k-feats 5 --n-steps 200 2>&1 | tail -15
```

Verify the smoke row lands at fresh `train_key` (T=10 in arch hparams
flows through compute_train_key).

Step 3 — full launch on 2× H100 (own GPU 0 + agent_em's idle GPU 1):

```bash
# T=10 on GPU 0, T=20 on GPU 1, both 3 seeds × 2 k_feats
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing_txc_T_sweep.run \
  --T-values 10 \
  > logs/c3_txc_T10_gpu0.log 2>&1 &

TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_txc_T_sweep.run \
  --T-values 20 \
  > logs/c3_txc_T20_gpu1.log 2>&1 &
```

Step 4 — once C3 lands, run C4 evals on the same checkpoints. Mirror
your existing `c4_qualitative` driver with the new T-overrides:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  .venv/bin/python -m experiments.c4_qualitative_txc_T_sweep.run \
  --archs txc_base --T-values 10 20 --seeds 1 2 42 \
  > logs/c4_txc_T_sweep.log 2>&1 &
```

(Adapt the C4 driver to accept T-overrides via the same
`arch_hparams_override` mechanism. Since C4 cache-hits on C3
checkpoints, the C4 cells land in <1 hr after C3 wraps.)

### Analysis filter update

After cells land, extend `experiments/c3_probing/analysis.py`'s
`canonical_train_keys` filter to include the T-sweep variants:

```python
txc_T10_keys = canonical_train_keys(
    component="c3",
    archs=["txc_base"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(
        n_steps=20_000,
        arch_hparams_override={"T": 10},
    ),
)
txc_T20_keys = canonical_train_keys(... arch_hparams_override={"T": 20})
canonical = txc_keys | topk_keys | tsae_keys | tfa_keys | mlc_keys | txc_T10_keys | txc_T20_keys
```

Same pattern for C4's analysis.

### Watch-outs

- **Don't change other archs' T values.** Only txc_base gets the
  T-sweep; TXC-pro / TopK / T-SAE / TFA / MLC stay at their canonical
  configs.
- **Same `eval_protocol_version`** as the rest of C3 (`1.1.0`) — no
  bump needed; the new train_keys distinguish the cells.
- **C4 cache-hits**: don't re-train from scratch for C4. Use the same
  T-overrides on the same datasource → same train_key → cache hit.
- **agent_steer_100k runs the BASE side** (`gemma_2_2b_base_l13_fineweb_24k128`)
  in parallel on their pod. Don't duplicate.

---

### ⚠️ NEW MISSION 2026-05-05 PM — C3 TFA baseline (decisions § 16)

**Your prior C3 TopK T=1 mission is COMPLETE** (commit `9b9d6cc5`).
New mission: **add TFA × 3 seeds × 2 k_feats** to the C3 sparse-probing
benchmark. Working alongside agent_em_100k (who builds 5-layer cache
+ runs MLC in parallel on their pod).

**Wasteland-faithful TFA training** (`origin/han-phase7-unification:
experiments/phase7_unification/train_phase7.py:312-353`):

> "TFA processes FULL sequences (B, T=128, d) through attention.
>  TFA-specific batch size override: TFA_BATCH = 32. Phase 5 default
>  was 64. Per-step tokens: 32 × 128 = 4096 (close to SAEBench 2K)."

TFA's attention tensor is heavy at large B (`B × T × d_sae` ~ 9.6 GB
fp32 at B=1024, T=128, d_sae=18432). The wasteland chose **B=32** to
keep per-step memory tractable AND give TFA's attention full ~128-token
context (Fig. 2(d) of `papers/priors_in_time.md` shows ~80% variance
explained at 100+ tokens of context). We adopt the same convention.

**TrainingConfig** (TFA-specific B override):

```python
TrainingConfig(
    n_steps=20_000,
    batch_size=32,                # ← TFA-specific override
    plateau_early_stop=False,
    train_window_size=None,       # full sequence (B, 128, d_in)
)
```

Per-step encoder load: B × T = 32 × 128 = 4096 tokens (≈ 2× SAEBench
canonical, paper-faithful per Phase 7 wasteland).

### Mission scope

| Arch | seeds | k_feats | TrainingConfig |
|---|---|---|---|
| `tfa` | {42, 1, 2} | {5, 20} | B=32, full seq, n_steps=20_000 |

3 unique trainings + 6 evals on the existing single-layer cache
`gemma_2_2b_it_l13_fineweb_24k128` (no new cache build needed for TFA).

### Wall-time on 2× H100

agent_em is post-canonical idle (commit `b549d91c` confirms), so GPU 1
is free for you to borrow (per § 13 GPU sharing convention; verify
`nvidia-smi` shows GPU 1 idle + note the borrow in your Current state).
2 cells parallel, 1 cell serial → **~2 hr wall**.

Per-cell estimate on H100: ~30-60 min train (TFA's attention costs
~3-5× more per step than vanilla SAE) + ~30 min probing eval × 2 k_feats
= ~1.5-2 hr per cell. With 2 GPUs, run 2 seeds simultaneously.

### First concrete task — write driver, smoke, launch

Step 0 — `git pull --rebase origin final`. Verify the framework:

```bash
.venv/bin/python -c "
from temp_bench.config import load_arch
spec = load_arch('tfa')
print('hparams:', spec.hparams)
"
```

Step 1 — write `experiments/c3_probing_tfa_baseline/{__init__.py,
run.py}`. Same pattern as your `c3_probing_topk_baseline/run.py`:
just swap arch name + override batch_size on TrainingConfig:

```python
"""C3 TFA baseline at B=32 + full seq (decisions § 16)."""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from experiments.c3_probing.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
)
from temp_bench.config import compute_act_cache_key, load_datasource


# Wasteland-faithful TFA training (decisions § 16, Phase 7 reference).
# B=32, full sequence (T=128) → 4096 tokens/step, close to SAEBench's
# 2K canonical and gives attention 128 tokens of context.
TFA_TRAINING_CFG = TrainingConfig(
    n_steps=20_000,
    batch_size=32,                  # ← TFA-specific override
    plateau_early_stop=False,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--n-steps", type=int, default=None)
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    cfg = TFA_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})

    for seed in args.seeds:
        for k in args.k_feats:
            print(f"[c3_tfa_baseline] cell tfa seed={seed} k_feat={k} "
                  f"B={cfg.batch_size} n_steps={cfg.n_steps}",
                  flush=True)
            eval_cfg = {
                "k_feat": k, "S": 32, "smoke": False,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            runner.run_cell(
                component="c3", arch_name="tfa", seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg, eval_cfg=eval_cfg,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn, eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

Step 2 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 42 --k-feats 5 --n-steps 200 2>&1 | tail -20
```

Step 3 — borrow GPU 1 (verify agent_em is idle first per § 13). Then
launch 2 seeds in parallel:

```bash
# Verify agent_em is post-canonical idle
grep "status: complete\|canonical mission COMPLETE" \
    agents/agent_em/briefing.md | tail -3

# Update YOUR Current state with the borrow note BEFORE launching:
# "Borrowing GPU 1 for C3 TFA sweep — agent_em is post-canonical
#  idle (mission complete commit b549d91c)."

# Launch in parallel
TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 0 -- \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 42 \
  > logs/c3_tfa_gpu0.log 2>&1 &
echo $! > /tmp/p_tfa_gpu0

TQDM_DISABLE=1 AGENT_NAME=agent_nlp \
  bash scripts/run_on_gpu.sh 1 -- \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 1 2 \
  > logs/c3_tfa_gpu1.log 2>&1 &
echo $! > /tmp/p_tfa_gpu1
```

Step 4 — monitor + verify rows land at `arch=tfa`, `component=c3`,
fresh `train_keys` (B=32 produces a distinct hash from any existing
B=1024 cells).

### Analysis filter update — 4-arch family

Once your TFA cells AND agent_em_100k's MLC cells land, the C3
canonical filter expands to FOUR arch families. Update
`experiments/c3_probing/analysis.py` to add 2 more
`canonical_train_keys` calls:

```python
tfa_keys = canonical_train_keys(
    component="c3",
    archs=["tfa"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000, batch_size=32),
)
mlc_keys = canonical_train_keys(
    component="c3",
    archs=["mlc"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_it_l11to15_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000),
)
canonical = txc_keys | topk_keys | tsae_keys | tfa_keys | mlc_keys
```

(MLC keys come in once agent_em_100k's mission lands; you can wire the
union now and the MLC keys will simply be empty until then.)

### Watch-outs

- **B=32 is intentional** — don't bump it back to 1024. Wasteland
  evidence (`train_phase7.py:316-321`) shows B>32 risks OOM on
  d_sae=18432 with the attention tensor at fp32. Document the per-
  arch B exception in c3.md caveats when you re-render.
- **Borrow GPU 1 carefully** — agent_em's pod is shared. Before
  launching, verify their Current state shows "canonical mission
  COMPLETE" / idle. If they start any compute work, your borrow gets
  preempted.
- **Don't reuse the multi-layer datasource** — TFA at C3 uses the
  existing single-layer `gemma_2_2b_it_l13_fineweb_24k128` (full seq
  T=128 from L13 alone). The l11to15 multi-layer datasource is
  agent_em_100k's MLC territory.

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06 05:35 UTC — ALL agent_nlp missions complete.
Status: complete (post-render, post-HF-push).**

All 4 directives delivered:
- decisions § 12: batch=1024 / n_steps=20K canonical re-train (Han override)
- decisions § 15: TopK T=1 baseline (paper-faithful per-token canonical)
- decisions § 16: TFA B=32 baseline (Phase 7 wasteland-faithful)
- decisions § 17: TXC-base T-sweep (T=10, T=20)

**HF-pushed checkpoints**: all 12 canonical (b1024/n20K) train_keys
+ 6 T-sweep train_keys at `han1823123123/temp-bench-models/`.
agent_filler's tsae_paper T=2 + agent_em_100k's MLC pulled locally.

**Leaderboard rows owned by agent_nlp**: 24 canonical C3 + 12 T-sweep
+ 6 TFA + 12 C4 + 6 C4 T-sweep = ~60 v3 rows. Plus 24 v1.0.0 + 24
v1.1.0 batch=256 rows kept for diff.

**Pod**: GPU 0 + GPU 1 idle. agent_em + agent_em_100k missions complete
on their own pods.

## C3 final headline (paper-final; b1024 / n20K, decisions § 12+§15+§16+§17)

6 baseline families landed. Mean ± σ across 3 seeds (`txc_base` row
includes T=5/T=10/T=20 mixed because the AUTO-RESULTS table groups by
arch name only; per-T means below):

| arch          | k=5                | k=20               |
|---------------|--------------------|--------------------|
| `mlc`         | 0.8531 ± 0.004     | **0.9042 ± 0.002** |
| `tsae_paper` (T=2) | 0.8407 ± 0.015 | 0.8986 ± 0.004     |
| `txc_base`    | 0.8402 ± 0.005 (n=9) | 0.8975 ± 0.004 (n=9) |
| `txc_pro`     | **0.8450 ± 0.013** | 0.8936 ± 0.009     |
| `topk_sae` (T=1) | 0.8306 ± 0.003  | 0.8831 ± 0.002     |
| `tfa` (B=32)  | 0.6562 ± 0.039     | 0.7146 ± 0.041     |

Per-T txc_base means (decisions § 17):
- T=5  k=5/k=20: 0.8367 / 0.8952
- T=10 k=5/k=20: 0.8409 / 0.8973
- T=20 k=5/k=20: 0.8429 / 0.8999

**Headline reading**:
- At k=20: MLC leads, T-SAE second (within seed-σ), TXC-base/pro and
  T-SAE form a tight cluster ~0.894-0.904. TopK-SAE at literature scale
  ~0.88. TFA underperforms by 0.18 (B=32 just doesn't compete here).
- At k=5: TXC-pro nudges ahead within seed-σ; MLC strong second.
- T-sweep: monotonic ~+0.005 per T-doubling at k=20 (small but consistent).

C3 hypothesis (TXC-pro matches best per-token at k=5 + small win at k=20):
**partially confirmed**. TXC-pro at k=5 leads but within seed-σ.
At k=20, TXC archs tie T-SAE / TopK in a 0.894-0.898 cluster, a notch
below MLC. The cross-token tasks (winogrande/wsc) remain a known
limitation across all archs (per-token mean-pool aggregation).

**Convergence verified** for all 24 canonical + 6 T-sweep cells via the
trainlog telemetry (final-1K-step loss drop < 0.3% for txc_base/pro/topk;
tsae shows oscillatory contrastive loss but macro-trajectory 70K→16K).

## C4 final headline (paper-final; b1024 / n20K)

12 canonical cells + 6 T-sweep cells. Mean ± σ across 3 seeds (txc_base
row mixes T=5/10/20):

| arch         | mean SEMANTIC ± σ | judge_agreement |
|--------------|-------------------|-----------------|
| `tsae_paper` (T=2) | **96.0 ± 8.9** | 0.882 |
| `topk_sae` (T=1) | 87.0 ± 5.3 | 0.883 |
| `txc_pro`    | 74.0 ± 10.4       | 0.859 |
| `txc_base`   | 44.9 ± 7.0 (n=9)  | 0.691 |

Per-T txc_base SEMANTIC means: T=5 49, T=10 44.7, T=20 41.0.
Trade-off: longer T → slightly better probing AUC, slightly WORSE
SEMANTIC count. Within seed-σ.

C4 hypothesis (TXC-pro matches T-SAE on SEMANTIC count):
**honest negative**. T-SAE leads at 96, TXC-pro 74 — solid 22-point
gap. TopK-SAE T=1 at 87 (paper-faithful) sits between them.
TXC-base trails at 45.

Vs v1.0.0 (batch=256 baseline kept for diff):
- tsae_paper:  74.7 → 96.0   (+21)
- topk_sae:    NEW           (n/a — v1.0.0 didn't include topk in C4)
- txc_pro:     60.0 → 74.0   (+14)
- txc_base:    42.0 → 44.9   (+3)

T-SAE benefited most from literature-faithful re-train; TXC variants
moved up modestly.

## What I just did (agent owns — overwrite)

Full session 2026-05-04 13:00 UTC → 2026-05-06 05:35 UTC. All 4
post-Han-batch-fix directives delivered + paper-final renders.

§ 12 batch=1024 / n_steps=20K canonical re-train — **DONE**:
- 12 unique trainings (4 archs × 3 seeds) × 2 k_feats = 24 cells.
- TXC-base + TXC-pro + tsae_paper + topk_sae all at literature-
  faithful per-arch TrainingConfig (TXC None, T-SAE T=2 from agent_filler,
  TopK T=1, MLC None on multi-layer from agent_em_100k).

§ 15 TopK T=1 baseline — **DONE**:
- topk_sae × 3 seeds × 2 k_feats at `train_window_size=1`. Driver in
  `experiments/c3_probing_topk_baseline/`.

§ 16 TFA B=32 baseline — **DONE**:
- tfa × 3 seeds × 2 k_feats at `batch_size=32`, full-seq (Phase 7
  wasteland-faithful). Driver in `experiments/c3_probing_tfa_baseline/`.

§ 17 TXC-base T-sweep (T=10, T=20) — **DONE**:
- txc_base × 3 seeds × 2 T values × 2 k_feats = 12 cells via
  `arch_hparams_override`. Driver in `experiments/c3_probing_txc_T_sweep/`
  + `experiments/c4_qualitative_txc_T_sweep/`. Cache-hits between
  C3 and C4 verified.

C4 sweep — **DONE**:
- 4 archs × 3 seeds × n_features=256 = 12 cells (canonical) + 6 T-sweep.
- ~2200 Haiku calls, ~$0.50 total. Judge agreement 0.69-0.88 across cells.

Framework deliverables:
- `temp_bench.data.nlp.preloaded_batch_iter_from_act_cache`
  (3.4× data-path speedup; bit-identical drop-in; 4 tests).
- Train-log persistence per-cell for convergence telemetry.
- `_T{N}` filename suffix on trainlogs to avoid T={None,1,5,10,20} collisions.

Renders:
- `docs/components/c3.md` AUTO-RESULTS — 6 archs × 2 k_feats. tfa + mlc
  + tsae_paper T=2 + topk_sae T=1 + txc_base/pro at canonical per-arch
  TrainingConfig.
- `docs/components/c4.md` AUTO-RESULTS — 4 archs SEMANTIC count.
- Both committed at `65f4ad88`.

HF push (persistent pod):
- All 12 canonical (b1024/n20K) train_keys + 6 T-sweep train_keys at
  `han1823123123/temp-bench-models/`. 30 GB total. Done in background.

## Decisions made + carried forward (overseer can override)

- **Per-task AUC reporting**: `my_eval_fn` returns BOTH per-task floats
  (`auc__<task>` × 38) AND aggregates (`mean_auc`, `std_auc`, ...) on
  every leaderboard row. analysis.py uses aggregates for the headline
  and per-task floats for σ_tasks.
- **Smoke rows filtered** out of headline via `eval_cfg.smoke==True`.
- **Bricken A/B for C3**: SKIPPED per decision § 7 default.
- **MLC port**: SKIPPED. Lower priority per agent_paper "Non-decisions";
  appendix-only OK. Test entry stays in `KNOWN_UNPORTED`.
- **EVAL_PROTOCOL_VERSION = "1.1.0"** — bumped for the Phase 7 padding
  fix. Stays at 1.1.0 for the batch=1024 / n_steps=20K re-train (eval
  pathway unchanged; train_key change alone invalidates eval cache).
- **C4 unaffected by padding fix** (no probe cache; forwards Gemma over
  concat_corpora token_ids directly).
- **C4 cells share train_keys with C3**. C4 runs after C3+topk_sae
  complete; cells hit CACHED on training and only re-run qualitative
  eval (~10 min Haiku per cell, 9 cells, ~$0.40 total).
- **n_steps=20K Han deadline override** is paper-headline; not landing
  back in the schema default (per agent_paper § 12 still says 25K).
  Other agents may follow suit if budget pressure (forwarded to
  agent_steer 2026-05-04 PM). decisions § 12 update is agent_paper's
  call.

## Next action — STATUS: COMPLETE

All 4 directives delivered. Last in-flight process exited
2026-05-06 ~05:30 UTC (last C4 T-sweep cell, txc_base T=20 seed=42).

If a new directive lands, the next agent_nlp life will:
1. Read this briefing top-to-bottom.
2. Pull origin/final + check `agents/agent_nlp/briefing.md` for new
   `### ⚠️ NEW MISSION ...` sections injected by agent_paper.
3. Write a thin driver under `experiments/c{3,4}_*_{family}/`,
   smoke-test, launch on idle GPU.

Saved leaderboard inventory at session end:
- ~144 leaderboard rows (canonical 24 + T-sweep 18 + C4 24 + TFA 6 +
  diff cells from v1.0.0 + v1.1.0 + smokes).
- 18 unique train_keys agent_nlp owned (12 canonical + 6 T-sweep).
- All 18 .safetensors checkpoints uploaded to HF.

### Reference: leaderboard filter for paper-final

Mirroring my analysis.py union-filter recipe (decisions § 15+§16+§17):
```python
from temp_bench.report import canonical_train_keys
from temp_bench.config import compute_act_cache_key, compute_train_key, load_arch, load_datasource
from temp_bench.schemas import TrainingConfig

ds = ('gemma_2_2b_it_l13_fineweb_24k128',)
seeds = (1, 2, 42)
canonical = (
    canonical_train_keys(component='c3', archs=('txc_base','txc_pro'), seeds=seeds, datasource_names=ds, training_cfg=TrainingConfig(n_steps=20_000))
    | canonical_train_keys(component='c3', archs=('topk_sae',), seeds=seeds, datasource_names=ds, training_cfg=TrainingConfig(n_steps=20_000, train_window_size=1))
    | canonical_train_keys(component='c3', archs=('tsae_paper',), seeds=seeds, datasource_names=ds, training_cfg=TrainingConfig(n_steps=20_000, train_window_size=2))
    | canonical_train_keys(component='c3', archs=('tfa',), seeds=seeds, datasource_names=ds, training_cfg=TrainingConfig(n_steps=20_000, batch_size=32))
    | canonical_train_keys(component='c3', archs=('mlc',), seeds=seeds, datasource_names=('gemma_2_2b_it_l11to15_fineweb_24k128',), training_cfg=TrainingConfig(n_steps=20_000))
)
# T-sweep: canonical_train_keys doesn't merge arch_hparams_override into
# spec.hparams; compute manually:
ack = compute_act_cache_key(load_datasource(ds[0]))
spec = load_arch('txc_base', component='c3')
for T in (10, 20):
    spec_m = spec.model_copy(update={'hparams': {**spec.hparams, 'T': T}})
    for seed in seeds:
        canonical.add(compute_train_key(arch=spec_m, seed=seed,
            training_cfg=TrainingConfig(n_steps=20_000, arch_hparams_override={'T': T}),
            act_cache_key=ack))
```

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
  loading script. Pinned in `pyproject.toml`.
- **github-code `languages=[...]` does NOT filter the stream.** Must
  `if sample['language'] != target_lang: continue` after iter.
- **`tsae_paper.config.T == 1`, NOT 2.** Contrastive pair is a TRAINING
  construct sampled inside `train_step`. T=2 routes the probe to
  window-encoding (wrong for T-SAE).
- **`LeaderboardRow.metrics` is float-only** (Pydantic). Categorical
  diagnostics like `task_name` go outside `metrics`.
- **Background `nohup ... &`** — bash wrapper returns immediately;
  python keeps running. Verify via `ps -ef | grep python`.
- **Decoder grad-parallel removal** uses `register_post_accumulate_grad_hook`
  on `W_dec` (PyTorch 2.0+). See `tsae.py`/`txc_base.py`/`txc_pro.py::_project_dec_grad`.
- **`einops` is NOT a dep.** Use vanilla `torch.einsum`.
- **TQDM_DISABLE=1 must be exported per bash call.** `set_agent_env.sh`
  doesn't set it. Standard pattern at `export TQDM_DISABLE=1 && ...`.
- **`python -u` is essential for nohup'd long-running scripts** — without
  it, `print()` calls buffer in a 4KB block and don't appear in the
  log file for ~5 min, making the process look stuck. Cost me ~5 min
  of training when I killed/restarted thinking it was deadlocked.
- **`tokenizer.padding_side="right"` for cache build, NEVER left**.
  Left padding shifts position-IDs of real tokens; out-of-distribution
  for Gemma. Phase 7 fix uses right pad + per-example reslice
  (left-aligned in the destination 32-frame). See
  `temp_bench.data.nlp.probe_cache::_encode_texts`.
- **`.contiguous()` after `.T` on saved tensors** — safetensors rejects
  non-contiguous tensors. Bit me on `tsae.py::_normalize_decoder` after
  30 min of training; tests didn't catch because the failure is at
  save time, not init. Always wrap `W_dec.data = ....T.contiguous()`.
- **MooseFS mmap is slow on first random access** (~5 steps/sec on the
  14 GB activation cache), then RAM-cached after warmup. Two parallel
  processes BEFORE warmup deadlocked in state D. Sequence cache build
  → eval; THEN parallel runs are safe.
- **`topk_sae` per-token z explodes at batch=1024**. Shape is
  `(B, seq_len, d_sae) = (1024, 128, 18432)` bf16 = 4.83 GB raw.
  `(z != 0).float()` in `architectures/base.py:81::train_step` doubles
  to 9.66 GB allocation in fp32, on top of model + activations + grad.
  Total peak ~25-30 GB on H100. Co-running with agent_em (38 GB Qwen-14B
  process) → OOM. Mitigation: launch TXC archs (window-level z, ~38 MB)
  + tsae_paper (anchor-pair z, ~76 MB) first while sharing GPU 0;
  defer topk_sae to when GPU 0 is solo.
- **`.clone()` is load-bearing for the preloaded batch_iter.** Without
  it, `torch.from_numpy(np.ascontiguousarray(mmap))` zero-copy wraps
  the mmap and fancy indexing still page-faults — the whole point of
  the preload defeated. `.clone()` materialises into anonymous RAM so
  subsequent indexing is RAM-rate. See `tests/test_preloaded_batch_iter.py`
  for the bit-identity guarantee.
- **First-cell-of-process is slow**. The `.clone()` reads 14 GB from
  MooseFS (or page cache). Cold start: 30 sec. Warm start: 1 sec.
  Module-global cache means subsequent cells in same process reuse
  the RAM tensor.
- **Add setup-phase prints to debug "stuck" trainers.** Default trainer
  is silent until first 1000-step boundary. At 0.5-1 step/sec under
  shared-GPU contention that's 30+ min of zero output, looks deadlocked.
  My fix: print every 100 steps for first 1000, then every 1000.
- **Git commit identity**: repo has no user.email/user.name set.
  Commits use inline `GIT_AUTHOR_*` env vars. Rebases use `git -c
  user.email=... -c user.name=... rebase ...` (env vars don't propagate
  cleanly through rebase's internal commits).
- **Leaderboard JSONL conflicts during rebase**: append-only, both
  sides add new rows. Resolution = strip conflict markers, keep both
  sets. Done it 5+ times this session via:
  ```python
  with open('results/leaderboard.jsonl') as fh: keep = [l for l in fh if not l.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
  ```
- **`runner.run_cell` cache contract**: skips eval if `eval_in_leaderboard(eval_key) and metrics_exist(eval_key)`. Means changing the
  probe cache content WITHOUT bumping `EVAL_PROTOCOL_VERSION` or
  `force_eval=True` silently returns OLD metrics. Lesson learned twice.
  This is why the batch_size + n_steps live in `train_key` (auto-invalidate
  on change) — no manual version bump needed for re-train.
- **`first_real` mask in `_encode_pool`**: window archs (T>1) have edge
  case where n_real < T means NO valid window for that row. Code
  falls back to all-windows mean for those rows (probe noisy but no
  NaN). Affects winogrande/wsc on TXC archs; a few rows per task.

## Open questions for Han / agent_paper (agent owns — overwrite)

1. **n_steps=20K paper-wide?** — landed in my code (commit `513a85ea`)
   per Han's deadline call 2026-05-04 PM. agent_steer was sent the
   recommendation to follow suit (Han forwarded the message). Should
   agent_paper update `decisions.md` § 12 + the schema default so
   future agents don't have to override per-cell? Currently § 12 still
   says 25K.

2. **C4 4th arch coverage** — C4 v1.0.0 ran 3 archs (txc_base, txc_pro,
   tsae_paper). No topk_sae C4. The new b1024/n20K C4 will follow the
   same 3-arch convention. Should we ALSO judge topk_sae for the C4
   Pareto plot, given topk_sae's checkpoint will exist? Adds ~$0.13 +
   ~30 min wall, gives 4 points per arch on the Pareto. (I lean yes
   for completeness; happy to defer.)

3. **`base.py:81` memory hot-spot — opportunistic fix.** The shared
   `TempBenchArch.train_step` computes
   `l0 = (z_flat != 0).float().sum(dim=-1).mean()` which allocates a
   `(B*S, d_sae)` fp32 tensor (9.66 GB at batch=1024 / d_sae=18432 /
   per-token archs like topk_sae). Reordering to
   `(z_flat != 0).sum(dim=-1).float().mean()` defers the float
   conversion to the scalar reduction → drops 9.66 GB peak. With GPU 0
   solo to me, topk_sae fit fine at 45 GB total without the fix; flag
   is **no longer urgent for me** but agent_em (Qwen-14B + d_sae=32768)
   or agent_back (Llama-8B A40 + d_sae=32768) might still benefit.
   Out-of-scope edit for me (`src/temp_bench/architectures/base.py`
   is shared code).

4. **Headline rendering during in-flight period** — currently
   docs/components/c3.md still shows v1.1.0 batch=256 numbers; the new
   b1024/n20K headline only fills in once topk_sae lands. Acceptable
   to keep stale numbers visible until then? (Alternative: render the
   placeholder now, but that strips the only visible C3 numbers from
   the docs for ~9 hr.)

5. **Probe cache HF push at schema 2.0.0** — DONE 2026-05-04 morning.
   266 files at `han1823123123/temp-bench-data/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`.
   agent_steer / ephemeral pods sync via
   `hf download han1823123123/temp-bench-data --repo-type dataset --include 'probe_cache/gemma_2_2b_it_l13_fineweb_24k128/*'`.
