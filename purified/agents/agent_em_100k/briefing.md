<!--
DRAFT — written by agent_paper 2026-05-04 PM, REWRITTEN 2026-05-05 PM
to reflect the C3 MW pivot (C6 100K mission abandoned, see "Mission
pivot" below).
Section ownership: PROTOCOL.md § 14.
-->

---
agent: agent_em_100k
last_state_update: 2026-05-06T11:35:00Z
component: c3 (tfa borrow eval IN FLIGHT; MLC IT+BASE complete)
status: in_progress (tfa seed=42 k_feats {10,40,80,160,320,640} eval-only)
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent EM 100K** (legacy name; mission has pivoted —
see below). You own the **C3 multi-window deployment** as a helper to
agent_nlp. Your purpose: train `txc_base_mw` and `txc_pro_mw` on the
C3 sparse-probing setup so agent_nlp's headline gets MW data without
agent_nlp needing to re-run cells themselves.

Files you may edit:

- `agents/agent_em_100k/briefing.md` (your own — agent-owned sections only)
- `experiments/c3_probing_mw/` (new experiment directory you create
  with a minimal driver — see "First concrete task" below).
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. agent_nlp's C3 plumbing
  (preloaded batch_iter, training, SAEBench+CT eval) is already wired
  and compatible with the multi-window TXC archs. Re-use via imports.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory **including
  `agents/agent_nlp/` and `agents/agent_em/`**. Their briefings,
  decisions, and per-cell state are theirs.
- `experiments/c3_probing/` — agent_nlp's territory. Their `run.py`,
  `analysis.py`, evaluation pipeline. You import from there without
  modification.
- `experiments/c4_qualitative/` — agent_nlp's territory. C4 cells
  cache-hit on C3 trained checkpoints; agent_nlp will re-eval C4 once
  your MW checkpoints land.
- `experiments/c6_em/` — your previous C6 work, now agent_em's
  territory.
- `experiments/c6_em_100k/` — your OLD driver from the abandoned 100K
  mission. Leave it as-is; agent_paper may delete or repurpose later.
- `docs/components/c3.md` and `docs/components/c4.md` — agent_nlp's
  territory.
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

### ⚠️ NEW MISSION 2026-05-06 — C3 IT TFA k_feats tail (6 cells, helping agent_nlp post-rescue)

**Han 2026-05-06**: agent_nlp's RunPod blew up mid-mission and was
rescued (commit `eaa75a10`). 13 eval cells were left unfinished.
Han split them: **agent_nlp takes 7 (their seed=2 tfa tail + txc_base
T=20 high-k tail), you take 6** (the seed=42 tfa column). Single arch,
single seed, eval-only — fastest possible "borrow into agent_nlp's
territory" arrangement.

This mission is **a brief, well-scoped borrow**. After your 6 cells
land, agent_nlp continues to own tfa territory. You go back to your
own work (BASE MLC parity below if not yet done; otherwise idle).

### Your 6 cells (eval-only, all cache-hit on training)

```
tfa  seed=42  k_feat=10
tfa  seed=42  k_feat=40
tfa  seed=42  k_feat=80
tfa  seed=42  k_feat=160
tfa  seed=42  k_feat=320
tfa  seed=42  k_feat=640
```

Single arch (`tfa`), single seed (42), 6 new k_feat values. The
trained checkpoint (B=32, n_steps=20_000, full-seq) already exists —
agent_nlp's rescue pushed it to HF. The runner cache-hits on training;
only the per-k_feat probe fits run.

### Driver invocation

```bash
cd /workspace/temp_xc/purified
git pull --rebase origin final

# Use agent_nlp's existing tfa baseline driver — IMPORT not EDIT.
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \
  --seeds 42 --k-feats 10 40 80 160 320 640 \
  > logs/c3_kfeat_tfa_seed42_em100k.log 2>&1 &
```

Per-cell ~30 min on H100 → **~3 hr total wall**.

### Pre-launch sanity check

Verify the tfa seed=42 trained checkpoint exists locally or on HF:

```bash
.venv/bin/python <<'PY'
from temp_bench.config import compute_train_key, load_arch, load_datasource, compute_act_cache_key
from temp_bench.schemas import TrainingConfig
import os

ds = load_datasource('gemma_2_2b_it_l13_fineweb_24k128')
ack = compute_act_cache_key(ds)
spec = load_arch('tfa', component='c3')
cfg = TrainingConfig(n_steps=20_000, batch_size=32)
tk = compute_train_key(arch=spec, seed=42, training_cfg=cfg, act_cache_key=ack)
print(f'tfa seed=42 train_key: {tk}')
print(f'  local: {"EXISTS" if os.path.exists(f"checkpoints/{tk}/model.safetensors") else "MISSING"}')

# Pull from HF if missing
if not os.path.exists(f'checkpoints/{tk}/model.safetensors'):
    print(f'  → pulling from HF...')
    from huggingface_hub import snapshot_download
    snapshot_download(
        'han1823123123/temp-bench-models',
        allow_patterns=[f'{tk}/*'],
        local_dir='checkpoints/',
    )
PY
```

### After your 6 cells land — commit + handoff

```bash
git add results/leaderboard.jsonl results/runs/
git commit -m "Agent EM 100K: borrow into tfa territory — C3 IT tfa seed=42 k_feats {10..640} eval-only (6 cells; helping agent_nlp post-rescue per Han 2026-05-06 split)"
git push origin final
```

Then ping agent_nlp that the seed=42 tfa column is complete. They
will:
- Run their 7 cells (seed=2 tfa tail + txc_base T=20 tail).
- Re-render `experiments/c3_probing/analysis.py` + `docs/components/c3.md`
  AUTO-RESULTS with all 8 k_feats.

**Don't touch `docs/components/c3.md`** — that's agent_nlp's
territory. Your 6 cells just land in the leaderboard; agent_nlp
integrates at the analysis step.

### Watch-outs

- **Eval-only, cache-hit on training.** Don't re-train tfa.
- **Pull the trained checkpoint from HF** if not local — it's there
  thanks to agent_nlp's rescue (commit `eaa75a10`).
- **Same `eval_protocol_version=1.1.0`** as agent_nlp's existing tfa
  cells — don't bump.
- **Don't run the txc_base T=20 cells** — those are agent_nlp's
  half of the split.
- **After this brief borrow, return to your own missions** (BASE MLC
  parity below if not done; or idle if done).

---

### ⚠️ NEW MISSION 2026-05-06 (URGENT) — BASE MLC parity + k_feats expansion

**Han 2026-05-06**: "current C3 has k {5,20} we want to expand to
{5,10,20,40,80,160,320,640} for all SPARSE PROBES in C3 for both IT
and BASE!"

**Your IT MLC sweep is COMPLETE** (commit `782c7702`, 7 cells at the
multi-layer datasource `gemma_2_2b_it_l11to15_fineweb_24k128`). Your
H100 is idle. **TWO follow-on missions** to keep paper coverage in
parity between IT and BASE:

### Mission A — IT MLC k_feats expansion (eval-only, fast)

Re-run your `experiments/c3_probing_mlc/run.py` driver with the 6 new
k_feats. Cache-hits on training; eval-only.

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_mlc.run \
  --seeds 42 1 2 \
  --k-feats 10 40 80 160 320 640 \
  > logs/c3_mlc_kfeat_expand.log 2>&1 &
```

3 seeds × 6 new k_feats = 18 evals at ~30 min each → **~9 hr serial**
on 1× H100. (Probe is CPU-bound, much faster; the 30 min is mostly
the encode pass through 38 SAEBench+CT tasks.)

### Mission B — BASE MLC parity (build cache + train + eval, slow)

C3 IT has 6 archs (TopK, T-SAE, TXC-base, TXC-pro, TFA, MLC).
C3 BASE has 5 (no MLC yet). For full IT/BASE parity, build the BASE
multi-layer infrastructure mirroring your IT setup:

| Phase | Work | Wall on H100 |
|---|---|---:|
| 1 | Build `gemma_2_2b_base_l11to15_fineweb_24k128` 5-layer cache (~70 GB) | ~3 hr |
| 2 | Build BASE 5-layer probe_cache (38 tasks) | ~1.5 hr |
| 3 | Run MLC × 3 seeds × 8 k_feats (5+10+20+40+80+160+320+640) on BASE | ~6-7 hr |
| **Total** | | **~10-11 hr** |

The datasource entry was already added by agent_paper:

```bash
.venv/bin/python -c "
from temp_bench.config import load_datasource
ds = load_datasource('gemma_2_2b_base_l11to15_fineweb_24k128')
print('layers:', ds.layers, 'subject:', ds.subject_model)
"
```

Build the BASE caches via the same framework calls as your IT build:

```bash
# 1. BASE multi-layer act cache
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.data.nlp.cache import build_activation_cache
build_activation_cache('gemma_2_2b_base_l11to15_fineweb_24k128')
" 2>&1 | tee logs/c3_mlc_base_cache_build.log

# 2. BASE multi-layer probe_cache
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.data.nlp.probe_cache import build_probe_cache
build_probe_cache('gemma_2_2b_base_l11to15_fineweb_24k128')
" 2>&1 | tee logs/c3_mlc_base_probe_cache_build.log

# 3. HF push both before pod restart
.venv/bin/python -c "
from huggingface_hub import HfApi
from temp_bench.config import compute_act_cache_key, load_datasource
key = compute_act_cache_key(load_datasource('gemma_2_2b_base_l11to15_fineweb_24k128'))
api = HfApi()
api.upload_folder(
    folder_path=f'results/act_cache/{key}',
    path_in_repo=f'act_cache/{key}',
    repo_id='han1823123123/temp-bench-data',
    repo_type='dataset',
)
api.upload_folder(
    folder_path='results/probe_cache/gemma_2_2b_base_l11to15_fineweb_24k128',
    path_in_repo='probe_cache/gemma_2_2b_base_l11to15_fineweb_24k128',
    repo_id='han1823123123/temp-bench-data',
    repo_type='dataset',
)
"

# 4. Run BASE MLC sweep (clone your IT driver, swap datasource)
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_mlc_base.run \
  --seeds 42 1 2 \
  --k-feats 5 10 20 40 80 160 320 640 \
  > logs/c3_mlc_base_full.log 2>&1 &
```

Step 4's driver is a thin clone of `experiments/c3_probing_mlc/run.py`
in a new dir `experiments/c3_probing_mlc_base/` with
`DATASOURCE = "gemma_2_2b_base_l11to15_fineweb_24k128"`. Everything
else is identical (multi-layer batch_iter, MLC encode, custom probe
eval).

### Sequencing recommendation

- **Mission A first** (~9 hr): completes IT MLC k_feats expansion.
- **Mission B second** (~10 hr): BASE MLC parity. Lower priority —
  the headline IT C3 story is already complete; BASE MLC gives the
  cross-model story for MLC specifically.

If you only have time for one, do Mission A (smaller delta,
completes the headline). Mission B is "stretch parity" — the
paper still works without it (5 archs at BASE is acceptable).

### Watch-outs

- **Mission A is eval-only**; don't re-train MLC. Cache-hits on
  existing IT MLC checkpoints.
- **Mission B builds NEW caches** (~70 GB each); don't forget the
  HF push before pod restart.
- **TFA bug fix** landed in commit `53e63fbb` — your work isn't
  affected (you don't run TFA), but FYI for context.

---

### ⚠️ NEW MISSION 2026-05-05 PM — C3 MLC baseline (decisions § 16, paper-faithful L=5)

**Your prior C3 T-SAE T=2 mission is COMPLETE** (commit `82674a75`).
New mission: **build a 5-layer Gemma activation cache + run MLC × 3
seeds × 2 k_feats at C3**. Working alongside agent_nlp (who runs C3
TFA in parallel on their pod). MLC's paper-faithful training requires
multi-layer activations (decisions § 16).

**Framework now available** (commit `d859daef`):

- New datasource `gemma_2_2b_it_l11to15_fineweb_24k128` with
  `layers: [11, 12, 13, 14, 15]`. Inspect:
  ```bash
  .venv/bin/python -c "from temp_bench.config import load_datasource; print(load_datasource('gemma_2_2b_it_l11to15_fineweb_24k128'))"
  ```
- `temp_bench.data.nlp.cache.build_activation_cache(...)` detects
  multi-layer datasource (via `layers: list[int]` field), registers
  N forward hooks in one model pass, captures (N, L=5, seq_len, d_in)
  into a single .npy.
- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_multilayer_cache(
  act_cache_key, seed=...)` returns (B, L=5, d_in) batches for MLC's
  data path.
- `temp_bench.data.nlp.probe_cache.build_probe_cache(...)` extended
  analogously: per-task X arrays at (N, L=5, S=32, d_in) when the
  datasource is multi-layer.

### Mission scope

| Phase | Work | Wall time on H100 |
|---|---|---:|
| 1 | Build 5-layer Gemma activation cache (~70 GB) | ~3 hr |
| 2 | Build 5-layer probe_cache (38 SAEBench+CT tasks) | ~1.5 hr |
| 3 | Smoke MLC at n_steps=200 (verify pipeline) | ~10 min |
| 4 | Run MLC × 3 seeds × 2 k_feats (3 trainings + 6 evals) | ~5 hr |
| **Total** | | **~10 hr** |

**TrainingConfig**:

```python
TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
    train_window_size=None,    # MLC's data shape comes from the
                               #   multi-layer datasource (L axis), not
                               #   from the train_window_size system
)
```

Per-step encoder load: B × L = 1024 × 5 = 5120 tokens (~ same as TXC at T=5).

### First concrete task — build cache, write driver, smoke, launch

Step 0 — `git pull --rebase origin final` and verify framework:

```bash
.venv/bin/python -c "
from temp_bench.data.nlp.cache import preloaded_batch_iter_from_multilayer_cache, build_activation_cache
from temp_bench.config import load_datasource
ds = load_datasource('gemma_2_2b_it_l11to15_fineweb_24k128')
print('layers:', ds.layers)
print('seq_len:', ds.seq_len, 'n_seqs:', ds.n_seqs)
"
```

Step 1 — **build the multi-layer act cache** (~3 hr):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -c "
from temp_bench.data.nlp.cache import build_activation_cache
build_activation_cache('gemma_2_2b_it_l11to15_fineweb_24k128')
" 2>&1 | tee logs/c3_mlc_cache_build.log
```

Verify the output cache is 4D shape (N=24000, L=5, T=128, D=2304):

```bash
.venv/bin/python -c "
from temp_bench.config import compute_act_cache_key, load_datasource, act_cache_dir
import numpy as np
key = compute_act_cache_key(load_datasource('gemma_2_2b_it_l11to15_fineweb_24k128'))
acts = np.load(act_cache_dir(key) / 'acts.npy', mmap_mode='r')
print('shape:', acts.shape, 'dtype:', acts.dtype)
print('size:', acts.nbytes / 2**30, 'GB')
"
```

Step 2 — **build the multi-layer probe_cache** (~1.5 hr):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -c "
from temp_bench.data.nlp.probe_cache import build_probe_cache
build_probe_cache('gemma_2_2b_it_l11to15_fineweb_24k128')
" 2>&1 | tee logs/c3_mlc_probe_cache_build.log
```

This produces 38 task dirs at
`results/probe_cache/gemma_2_2b_it_l11to15_fineweb_24k128/<task>/`
with X_train.npy + X_test.npy at shape (N, L=5, S=32, d_in).

**HF push both caches** (ephemeral pod) before any pod restart:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
from temp_bench.config import compute_act_cache_key, load_datasource
key = compute_act_cache_key(load_datasource('gemma_2_2b_it_l11to15_fineweb_24k128'))
api = HfApi()
api.upload_folder(
    folder_path=f'results/act_cache/{key}',
    path_in_repo=f'act_cache/{key}',
    repo_id='han1823123123/temp-bench-data',
    repo_type='dataset',
)
api.upload_folder(
    folder_path='results/probe_cache/gemma_2_2b_it_l11to15_fineweb_24k128',
    path_in_repo='probe_cache/gemma_2_2b_it_l11to15_fineweb_24k128',
    repo_id='han1823123123/temp-bench-data',
    repo_type='dataset',
)
"
```

Step 3 — **write `experiments/c3_probing_mlc/{__init__.py, run.py}`**.
Key design: import most of agent_nlp's `experiments/c3_probing/run.py`,
override the train_fn (multi-layer batch_iter) and write a custom
eval_fn that handles 4D probe arrays. Sketch:

```python
"""C3 MLC baseline at L=5 (decisions § 16, paper-faithful).
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import (
    compute_act_cache_key, load_datasource, load_arch,
    instantiate_arch, act_cache_dir,
)
from temp_bench.training.sae_trainer import train_sae
from temp_bench.data.nlp.cache import preloaded_batch_iter_from_multilayer_cache

DATASOURCE = "gemma_2_2b_it_l11to15_fineweb_24k128"
EVAL_PROTOCOL_VERSION = "1.1.0"   # match agent_nlp's

MLC_TRAINING_CFG = TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
)


def _d_in_from_act_cache(act_cache_key: str) -> int:
    meta = json.loads((act_cache_dir(act_cache_key) / "meta.json").read_text())
    return int(meta["d_in"])


def my_train_fn_mlc(*, arch_name, arch_hparams, seed, training_cfg,
                    act_cache_key, component):
    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)
    model = instantiate_arch(spec, d_in=d_in)
    model.cuda()
    torch.manual_seed(seed); np.random.seed(seed)
    raw_iter = preloaded_batch_iter_from_multilayer_cache(
        act_cache_key, seed=seed,
    )
    return train_sae(
        model, raw_iter, training_cfg, device="cuda",
    )


def my_eval_fn_mlc(*, _state_dict, _arch_name, _arch_hparams, seed,
                   _act_cache_key, _datasource_name, k_feat, S, smoke):
    """MLC eval: load multi-layer probe_cache, encode each (S, L, d_in)
    frame via MLC, run probe on z. Same s-tail metric as agent_nlp's
    c3 eval but on multi-layer activations.

    Adapt agent_nlp's `experiments/c3_probing/run.py:my_eval_fn` —
    replace the single-layer probe_cache load + per-token SAE encode
    with: load (N, L=5, S=32, d_in) probe arrays, encode via MLC's
    `model.encode(x)` which expects (B, L, d_in) → (B, d_sae). Then
    s-tail pool + select top-k_feat features by mean abs activation
    + train a logistic probe.
    """
    # Implement using agent_nlp's eval as a reference; the only
    # structural change is the input shape (multi-layer) and the
    # encode call.
    ...
```

(Full eval implementation is your work — adapt agent_nlp's existing
s-tail probing pipeline. MLC's encode signature: `encode(x: (B, L, d))
→ z: (B, d_sae)`.)

Step 4 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_mlc.run \
  --seeds 42 --k-feats 5 --n-steps 200 2>&1 | tail -25
```

Step 5 — launch the full sweep:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_mlc.run \
  > logs/c3_mlc_full.log 2>&1 &
echo $! > /tmp/p_c3_mlc
```

Step 6 — monitor + verify rows land at `arch=mlc`, `component=c3`,
fresh `train_keys` (distinct from any single-layer cells).

### Watch-outs

- **Don't reuse the L13 single-layer probe_cache for MLC** — MLC needs
  L11-L15 stacked. Always use the multi-layer probe_cache.
- **HF push the multi-layer caches** before pod stop. Both act_cache
  (~70 GB) and probe_cache (~1 GB).
- **70 GB cache fits on H100 pod** (240 GB system RAM) but verify via
  `free -g` after the .clone() preload completes — the helper takes
  ~30 sec to materialise the tensor on first call.
- **Don't render docs/components/c3.md** — agent_paper integrates at
  paper-render time.

---

### ⚠️ § 15 mission COMPLETE 2026-05-05 PM (commit `82674a75`)

C3 T-SAE T=2 baseline sweep landed: 3 trainings + 6 evals at
`tsae_paper × {42, 1, 2} × {k_feat=5, k_feat=20}` with
`TrainingConfig(n_steps=20_000, train_window_size=2)`. Section below
this point is RESCINDED / completed. Left in place for git provenance.

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
true, but the FIX was wrong: the right move is to bring per-token
baselines DOWN to T=1 window-based (matching SAEBench + matching C6/C7),
NOT to bring TXC up via MW.

**Han's call (2026-05-05 PM)**: ABORT all 4 MW pivots. Re-train per-token
baselines at C3 + C5 with the T=1 window-based pattern.

**Your specific abort actions** (do these in this order):

1. **Kill the in-flight C3 MW sweep.** Last briefing said PID 17963,
   started 2026-05-05T12:16Z, both archs (txc_base_mw + txc_pro_mw) ×
   seed=42 × k_feats {5, 20}. Burning ~4 hr/cell on the wrong direction.
   ```bash
   ps -ef | grep "experiments.c3_probing_mw" | grep -v grep
   # → kill -TERM <PID>; if uncooperative, kill -9
   pkill -TERM -f "experiments.c3_probing_mw" || true
   nvidia-smi --query-gpu=memory.used --format=csv
   # → expect <500 MB; force-kill stragglers via `kill -9 <PID>`
   ```
   Smoke row at `train_key=e0ff471f7ddac586` (n_steps=200) stays in the
   leaderboard — `canonical_train_keys` filters it out at paper-render
   time, harmless.
2. **Set status: idle, awaiting re-purpose.** Do NOT launch any further
   MW work. Update Current state in this briefing to reflect "STOOD
   DOWN — awaiting C3 baseline re-train directive."
3. **Your next mission (when re-purposed) will be C3 baseline T=1
   re-train.** agent_paper is landing the framework change
   (`train_window_size: int | None` on `preloaded_batch_iter_from_act_cache`
   + `TrainingConfig`) and will rewrite this briefing to redirect you
   to the C3 baseline re-train (TopK_SAE + T-SAE × 3 seeds × 2 k_feats
   with `TrainingConfig(train_window_size=1)`). Wait for the briefing
   rewrite before resuming. ETA on the rewrite: same session — should
   be fresh by the time you read this if you `git pull` again.

**The MW pivot directive below this line is RESCINDED.** Do not read
it as actionable. Left in place for git provenance only.

---

### ⚠️ NEW MISSION 2026-05-05 PM — C3 baseline T-window re-train (decisions § 15)

**You are repurposed.** The MW pivot is dead; the new mission is to re-train
C3's per-token baselines at the per-arch literature-faithful window size.
agent_paper landed:

- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache(...,
  train_window_size: int | None = None)` — kwarg added; None preserves
  current full-sequence behavior.
- `TrainingConfig.train_window_size: int | None = None` — new field;
  flows into `compute_train_key` via `model_dump(exclude_none=True)`,
  so OLD train_keys preserved (None default) and NEW cells with int set
  get fresh keys. Confirmed by 5 new tests (136/136 green, commits
  `5555e7eb`).
- agent_nlp's `experiments/c3_probing/run.py:my_train_fn` already
  passes through `training_cfg.train_window_size` to the helper
  (commit landing in this push). You inherit it via import.

**Mission scope** (Han 2026-05-05 PM split between you + agent_nlp):

| Arch         | seeds      | k_feats | `train_window_size` | tokens/step | Source |
|---|---|---|---:|---:|---|
| `tsae_paper` | {42, 1, 2} | {5, 20} | **2** | 2048 | Bhalla/Ye 2025 §3.1 paper-faithful adjacent pairs |

**Total: 3 unique trainings + 6 evals — your scope.**

**agent_nlp takes `topk_sae`** (TopK × 3 seeds × 2 k_feats at T=1) on
their 2× H100 pod (their own GPU 0 + agent_em's now-free GPU 1; agent_em
is idle post-canonical-sweep). They have C3 mastery + the TopK plumbing
already proven, so the natural division is: TopK is theirs, T-SAE is
yours. **Don't re-run topk_sae** — wasted compute.

MLC unported (decision § 11 appendix-only); skipped. TXC archs
(`txc_base`, `txc_pro`) unchanged — agent_nlp's existing canonical
sweep stands.

**TrainingConfig**:

```python
TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
    train_window_size=2,          # ← new field, T=2 paper-faithful pairs
)
```

**Per-cell wall-time on H100** (rough):

- `tsae_paper` at T=2: encoder runs on 2 token-vectors per row
  (`(B, 2, d_in)` → anchor + temporal-pair). ~15-20 min train + ~30
  min probing eval × 2 k_feats. Per-cell: ~1.5 hr.
- 3 cells serial on 1× H100: ~4.5 hr total. Fits in remaining sprint
  window.

### First concrete task — write the C3 baseline driver, smoke, launch

Step 0 — verify the STAND DOWN was effective. Confirm no C3 MW
processes still running, GPU clean:

```bash
ps -ef | grep "experiments.c3_probing_mw" | grep -v grep   # should be empty
nvidia-smi --query-gpu=memory.used --format=csv             # < 500 MB
```

Step 1 — `git pull --rebase origin final` and verify the framework:

```bash
.venv/bin/python -c "
from temp_bench.schemas import TrainingConfig
cfg = TrainingConfig(train_window_size=1)
print(cfg.train_window_size)  # → 1
print(TrainingConfig().model_dump(exclude_none=True).keys())  # → no 'train_window_size'
"
.venv/bin/python -c "
from temp_bench.data.nlp.cache import preloaded_batch_iter_from_act_cache
import inspect
sig = inspect.signature(preloaded_batch_iter_from_act_cache)
assert 'train_window_size' in sig.parameters, 'framework change not present — git pull?'
print('OK')
"
```

Step 2 — verify activation cache + probe cache on disk (reuse from
your prior C3 MW prep, both should still be there):

```bash
ls results/act_cache/e4916bcae1881963/acts.npy
ls results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/ | head -3
```

Step 3 — write `experiments/c3_probing_tsae_baseline/__init__.py`
(empty) and `experiments/c3_probing_tsae_baseline/run.py`:

```python
"""C3 T-SAE baseline re-train at T=2 (decisions.md § 15).
Imports agent_nlp's plumbing verbatim; only the arch + TrainingConfig
change.
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


# Bhalla/Ye 2025 §3.1: T-SAE paper-faithful adjacent pairs → T=2.
TSAE_TRAINING_CFG = TrainingConfig(n_steps=20_000, train_window_size=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps for smoke tests.")
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    cfg = TSAE_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})

    for seed in args.seeds:
        for k in args.k_feats:
            print(
                f"[c3_tsae_baseline] cell tsae_paper seed={seed} k_feat={k} "
                f"T={cfg.train_window_size} n_steps={cfg.n_steps}",
                flush=True,
            )
            # Mirror agent_nlp's eval_cfg shape (run.py:300) so eval
            # cache-keys are stable.
            eval_cfg = {
                "k_feat": k,
                "S": 32,
                "smoke": False,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            runner.run_cell(
                component="c3",
                arch_name="tsae_paper",
                seed=seed,
                datasource_name=DATASOURCE,
                training_cfg=cfg,
                eval_cfg=eval_cfg,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=my_train_fn,
                eval_fn=my_eval_fn,
            )


if __name__ == "__main__":
    main()
```

Step 4 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_tsae_baseline.run \
  --seeds 42 --k-feats 5 --n-steps 200 \
  2>&1 | tail -25
```

Verify the leaderboard row lands with `train_window_size=2` baked
into the train_key (the cell will get a fresh key, distinct from
agent_nlp's existing tsae_paper cells):

```bash
.venv/bin/python -c "
from temp_bench.report import query_leaderboard
rows = [r for r in query_leaderboard(component='c3') if r.arch=='tsae_paper']
print(f'tsae_paper rows: {len(rows)}')
for r in rows[-3:]:
    print(r.train_key, r.eval_key, r.eval_cfg.get('k_feat'), r.metrics.get('mean_auc'))
"
```

Step 5 — launch the full sweep:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_tsae_baseline.run \
  > logs/c3_tsae_baseline_full.log 2>&1 &
echo $! > /tmp/p_c3_tsae_baseline
```

Step 6 — monitor + verify. Per-cell wall ~1.5 hr; full sweep ~4.5 hr.
Use the persistent monitor pattern from your prior C3 MW work; same
log shape (`[SETUP {arch}/seed=...]`, `[TRAIN ... step XXXX/20000]`,
`CELL DONE`).

Step 7 — when sweep lands, re-render is agent_paper / agent_nlp's job
at paper-render time. agent_nlp's `experiments/c3_probing/analysis.py`
will pick up your new cells via three `canonical_train_keys()` calls
(TXC at T=None, TopK at T=1, T-SAE at T=2). Don't render anything to
`docs/components/c3.md` yourself — that's agent_nlp's territory.

**Watch-outs**:

- **Don't override `_real_training_cfg`.** Use `ARCH_TRAINING_CFGS` to
  pick the per-arch config; pass it directly to `runner.run_cell`.
- **eval_cfg shape matters.** Your prior C3 MW work hit two crashes
  (KeyError `_act_cache_key`, FileNotFoundError on probe cache) —
  same plumbing here. Inject `_act_cache_key`, `_datasource_name`,
  `smoke=False` per agent_nlp's `run.py:292-301` pattern.
- **Probe cache on HF**: `sync_from_hf.sh` only pulls `act_cache/**`.
  If the probe_cache dir is missing on a fresh pod, run:
  ```bash
  .venv/bin/hf download han1823123123/temp-bench-data \
      --repo-type dataset \
      --include "probe_cache/gemma_2_2b_it_l13_fineweb_24k128/**" \
      --local-dir results/
  ```
- **GPU is yours alone** on this 1× H100 pod. No GPU-sharing
  considerations.

---

### ⚠️ Mission pivot 2026-05-05 — abandon C6 100K, deploy C3 MW [RESCINDED 2026-05-05 PM — see STAND DOWN above]

**Old mission (abandoned)**: replicate agent_em's C6 sweep at
`n_steps=100_000`. You completed SAE seed=42 14B-finance @ 100K
(peak_align=82.11) and the TXC seed=42 14B-finance @ 100K cell was
running, but the per-step rate was 5.9× slower than SAE on this pod
(your OQ #2 in the previous briefing). Abandon: 100K-pace work
doesn't fit the remaining sprint, and the C6 MW deployment is a
stronger paper headline than 100K convergence-test data. agent_em
themselves are taking on C6 MW after their canonical mission ends.

**Where the existing 100K work survives**:

- **SAE seed=42 14B-finance @ 100K**: row in `leaderboard.jsonl`,
  checkpoint on HF. Stays as a "what does sae_arditi look like at
  field-standard 100K tokens?" reference for the paper's caveats
  section. Useful — sae_arditi went from 78.33 (25K) to 82.11 (100K),
  showing real convergence headroom.
- **TXC seed=42 14B-finance @ 100K**: if Wang completed before this
  pivot (check the leaderboard), the row + checkpoint stay. If still
  in flight, **kill it on session start** — we're done with that mission.
- `experiments/c6_em_100k/run.py`: leave it for now. agent_paper
  may repurpose later if we ever want the convergence data.

**New mission**: train **`txc_base_mw` and `txc_pro_mw` on C3** at
the canonical schedule, write the resulting checkpoints to the
leaderboard, and let agent_nlp's `analysis.py` pick them up via
`canonical_train_keys` at paper-render time.

C3 is agent_nlp's component (sparse probing on Gemma-2-2b-IT L13).
Their canonical sweep is mostly done (topk_sae sweep finishing now);
the `txc_base_mw` and `txc_pro_mw` archs need to be trained and
evaluated to give agent_nlp's C3 headline a multi-window comparison.

**Mission scope** (n=3 paired with agent_nlp's existing canonical):

- 2 archs × 3 seeds = **6 trainings**:
  - txc_base_mw × {seed=42, seed=1, seed=2}
  - txc_pro_mw × {seed=42, seed=1, seed=2}
- Each training cell is evaluated at TWO k_feats ({5, 20}) per the
  canonical C3 protocol — that's 12 eval cells total, but eval is a
  cache-hit on the same checkpoint so only 6 unique trainings.
- C4 evaluation will cache-hit on these C3 checkpoints later;
  agent_nlp re-runs C4 evals when ready (not your job).

**Why this pod is appropriate for C3 MW (despite the C6 slowdown)**:

Your previous slowdown traced to agent_em's `_build_batch_iter` in
`experiments/c6_em/train.py` — a Python for-loop slicing T-windows
per row, ~5-10× slower than vectorized data paths. **C3 uses a
DIFFERENT data path**: agent_nlp's `preloaded_batch_iter_from_act_cache`
from `temp_bench.data.nlp.cache` (the shared helper, vectorized
torch fancy indexing, no Python loop). You inherit it via
`experiments.c3_probing.run` imports. **Expect normal H100
performance on C3** — likely 30-50 min per txc_base_mw cell, 60-100
min per txc_pro_mw cell.

Hardware: **1× H100 80GB pod, ephemeral, 240 GB system RAM, 1 TB
/workspace** (same pod you've been on for the C6 100K mission). Pod
mode `ephemeral`: HF is the source of truth, auto-push on
checkpoint save.

VRAM check: C3 at Gemma-scale (d_in=2304, d_sae=18432) MW peaks
~25-50 GB activations on H100, well within 80 GB. No mitigations
needed.

Subject + protocol (replicating agent_nlp's setup verbatim):

- Datasource: `gemma_2_2b_it_l13_fineweb_24k128`
- Architectures: **`txc_base_mw` + `txc_pro_mw`** (2 archs total).
  These are the YAML aliases per decisions.md § 14. NOT topk_sae,
  NOT tsae_paper — those are agent_nlp's per-token archs (no MW
  variant exists; their canonical cells are the comparison baseline).
- Per-component d_sae overrides: locked_archs.yaml's `txc_base_mw`
  / `txc_pro_mw` use `d_sae=18432` for c3 (mirroring the canonical
  txc_base / txc_pro). Applied automatically.
- Probing: SAEBench+CT (n=38 binary one-vs-rest tasks) per
  decisions.md § 11. agent_nlp's `experiments/c3_probing/run.py`
  pipeline + datasets are inherited via imports.
- Headline metric: probing accuracy / AUC at k_feat ∈ {5, 20}.
- `EVAL_PROTOCOL_VERSION`: inherit from agent_nlp's `c3_probing/run.py`
  (whatever they have set; do not override).

`TrainingConfig` for your cells (canonical schedule):

```python
TrainingConfig(
    batch_size=1024,
    n_steps=20_000,         # canonical, NOT 100K
    plateau_early_stop=False,
    # bricken_* defaults (False) — C3 does not use Bricken (decisions.md § 7).
)
```

**Per-cell wall-time estimate on H100**:

- `txc_base_mw`: ~30-50 min train + ~30 min probing eval × 2 k_feats
  = ~1.5-2 hr per cell
- `txc_pro_mw`: ~60-100 min train + ~30 min probing eval × 2 k_feats
  = ~2.5-3 hr per cell

3 seeds × 2 archs serial: 3 × (2 + 3) hr = ~15 hr total. Fits in
remaining sprint with margin.

Note on TXC-pro MW perf: agent_steer_100k reported MW slowness on
their TXC-pro cells (the all-pairs InfoNCE matrix scales as
(B*N)² = 100× larger at MW). For C3 at Gemma scale, the InfoNCE
matrix is (10240, 10240) × 4 bytes = 400 MB fp32, with BF16
softmax+CE around it. Should be tractable on H100; if you hit OOM
on the InfoNCE compute, reduce batch_size to 512 (effective B*N=5120
still 5× non-MW) and document the deviation in your AUTO-RESULTS
notes.

Locked decisions in scope: #1 (canonical TXCs are `txc_base_mw` /
`txc_pro_mw` per § 1's 2026-05-05 amendment), #4 (cross-branch reads),
#6 (HF repos), #7 (Bricken off for C3), #11 (SAEBench+CT task suite),
§ 12 (canonical training cfg), § 14 (multi-window deployment).

References:
- `agents/README.md` (your roster row)
- `agents/agent_nlp/briefing.md` (the canonical C3 setup you replicate)
- `docs/components/c3.md` (the canonical C3 writeup; do NOT edit)
- `experiments/c3_probing/{run.py,analysis.py}` (import from)
- `decisions.md` § 11, § 12, § 14
- `papers/are_saes_useful.md` (SAEBench reference)
- `PROTOCOL.md` § 7 (results live in state), § 8 (anti-conflict),
  § 11 (framework discipline), § 14 (briefing maintenance)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed. Push artifacts via the
wrap-up script before any pod restart.

### First concrete task — kill C6 100K, write the C3 MW driver, launch

Step 0 — **kill the in-flight C6 100K processes** before doing
anything else (free GPU + RAM):

```bash
ps -ef | grep "experiments.c6_em_100k" | grep -v grep
# → kill any active PIDs
nvidia-smi --query-gpu=memory.used --format=csv
# → expect <500 MB used; if not, force-kill stragglers via `kill -9`
```

Step 1 — `git pull --rebase origin final`, verify the MW arch is
registered + agent_nlp's plumbing is current:

```bash
.venv/bin/python -c "from temp_bench.config import load_arch; print(load_arch('txc_base_mw').hparams)"
# → expect a dict containing multi_window=True
.venv/bin/python -c "from experiments.c3_probing.run import EVAL_PROTOCOL_VERSION; print(EVAL_PROTOCOL_VERSION)"
# → use whatever agent_nlp has set
```

Step 2 — pull the Gemma activation cache from HF if not already on
disk:

```bash
ls results/act_cache/*/acts.npy 2>/dev/null
bash scripts/sync_from_hf.sh   # if not present
```

Step 3 — write `experiments/c3_probing_mw/__init__.py` (empty) and
`experiments/c3_probing_mw/run.py`:

```python
"""C3 multi-window deployment driver — replicates agent_nlp's setup
with txc_base_mw / txc_pro_mw archs.

Re-uses agent_nlp's plumbing verbatim via imports. Only the arch
names differ; the data path, training, and probing eval are
unchanged.
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from experiments.c3_probing.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
    _real_training_cfg as _orig_training_cfg,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+",
                    default=["txc_base_mw", "txc_pro_mw"],
                    choices=["txc_base_mw", "txc_pro_mw"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    args = ap.parse_args()

    cfg = _orig_training_cfg()    # batch=1024, n_steps=20_000, plateau_off

    for arch in args.archs:
        for seed in args.seeds:
            for k in args.k_feats:
                print(f"[c3_mw] cell arch={arch} seed={seed} k_feat={k}")
                runner.run_cell(
                    component="c3",
                    arch_name=arch,
                    seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=cfg,
                    eval_cfg={"k_feat": k, "S": 32},
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                )


if __name__ == "__main__":
    main()
```

(Adapt `eval_cfg` to whatever agent_nlp's `c3_probing/run.py` uses.
Their canonical cells have specific keys; mirror them so your evals
go through the same path.)

Step 4 — smoke ONE cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 .venv/bin/python -c "
from temp_bench.schemas import TrainingConfig
from temp_bench import runner
from experiments.c3_probing.run import DATASOURCE, EVAL_PROTOCOL_VERSION, my_train_fn, my_eval_fn
result = runner.run_cell(
    component='c3', arch_name='txc_base_mw', seed=42,
    datasource_name=DATASOURCE,
    training_cfg=TrainingConfig(n_steps=200, batch_size=1024, plateau_early_stop=False),
    eval_cfg={'k_feat': 5, 'S': 32},
    eval_protocol_version=EVAL_PROTOCOL_VERSION,
    train_fn=my_train_fn, eval_fn=my_eval_fn,
)
print('smoke result:', result.train_key, result.eval_key, result.cached)
"
```

Step 5 — launch the full sweep:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_em_100k \
  .venv/bin/python -m experiments.c3_probing_mw.run \
  > logs/c3_mw_full.log 2>&1 &
echo $! > /tmp/p_c3_mw
```

Step 6 — monitor + verify rows land at `arch=txc_base_mw` /
`txc_pro_mw` with the canonical `eval_protocol_version`. agent_paper
integrates via `canonical_train_keys()` toggle at paper-render time.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-06T11:35Z. NEW MISSION (briefing update
2026-05-06): brief borrow into agent_nlp's tfa territory — 6 eval-
only cells (`tfa seed=42 k_feat ∈ {10,40,80,160,320,640}`) per Han's
13-cell split between agent_nlp + me post-RunPod-rescue (commit
`eaa75a10`).**

**Prior MLC missions A + B COMPLETE** (commit `1da6e2fd`). 49 MLC
cells in leaderboard:
- IT × 3 seeds × 8 k_feats = 24 cells (`gemma_2_2b_it_l11to15_fineweb_24k128`)
- BASE × 3 seeds × 8 k_feats = 24 cells (`gemma_2_2b_base_l11to15_fineweb_24k128`)
- + 1 smoke (IT seed=42 k=5 n_steps=200; canonical-filtered)

### tfa borrow mission state (in_progress)

| Phase | Status | Notes |
|---|---|---|
| Pull tfa seed=42 ckpt from HF | ✅ | `train_key=61da0670ea629ca4` (B=32, n=20K) |
| Launch eval (6 cells) | **IN FLIGHT** | PID 37678, started 11:34Z |
| Monitor → commit + push | pending | Per-cell ~5-30 min; total ETA ~30min-3hr |

Expected leaderboard rows: `arch=tfa, seed=42, eval_cfg.k_feat ∈
{10,40,80,160,320,640}`, `eval_protocol_version=1.1.0`. Cache-hits
on training (no fresh train; just probe fits).

Persistent monitor `bbbq7j4r3` watches for CELL DONE / errors.

### MLC headline numbers (mean across 3 seeds, Mission A + B complete)

**IT** (`gemma_2_2b_it_l11to15_fineweb_24k128`, act_cache=`40a11e1594d9220a`):

| k_feat |    5  |   10  |   20  |   40  |   80  |  160  |  320  |  640  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean_auc | 0.853 | 0.883 | 0.904 | 0.917 | 0.921 | 0.922 | 0.918 | 0.913 |

**BASE** (`gemma_2_2b_base_l11to15_fineweb_24k128`, act_cache=`87b600e76b7ab26d`):

| k_feat |    5  |   10  |   20  |   40  |   80  |  160  |  320  |  640  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean_auc | 0.863 | 0.892 | 0.913 | 0.921 | 0.926 | 0.926 | 0.922 | 0.916 |

BASE consistently ~0.005-0.010 higher than IT across the k_feat
range. Peak at **k=80-160** for both pods (~0.922 IT / 0.926 BASE).
At very high k (640) both decline slightly — top-k selection picks
up noisier features past the saturation point.

### IT train_keys (3 seeds × 8 k_feats; train_key shared across k_feats)

- seed=42: `c5b18a75a0db4994`
- seed=1:  `c4bad817b40f45ac`
- seed=2:  `f07bcad7d9f197d2`

### BASE train_keys (3 seeds × 8 k_feats; same pattern)

- seed=42: `cc0e0ec4a25613e6`
- seed=1:  `0e09f3338c0780dd`
- seed=2:  `468da945c83f2334`

### Mission timing summary

**Mission A — IT k_feats expansion** (eval-only, cache-hit on training):
- 18 cells (3 seeds × 6 new k_feats {10, 40, 80, 160, 320, 640})
- Wall: 1h 34m (06:25:27 → 07:59:38), ~5 min/cell

**Mission B — BASE MLC parity** (build + train + eval):
- Phase 1 (BASE act_cache build): 5 min
- Phase 2 (BASE probe_cache build): 13 min (parallel with Mission A)
- Phase 3 (HF push both BASE caches): ~17 min (parallel)
- Phase 4 (driver write): in parallel with Mission A
- Phase 5 (24-cell sweep): 3h 10m 41s (07:59:51 → 11:10:32)
  - 3 fresh trainings × ~25 min = 75 min
  - 21 cache-hit evals × ~5 min = 105 min

**Total wall (both missions, mostly serial after Mission A finished training)**:
- Mission A start (06:25Z) → Mission B done (11:10Z) = **4h 45m**

Mission A and Mission B Phase 1-3 ran in parallel without contention
issues (free RAM 1.8 TB, free GPU 70 GB). Mission B Phase 5 sweep
serialized after Mission A finished to avoid GPU train-time contention.

### MLC IT vs prior § 15 T-SAE T=2 (same agent, same datasource)

| arch (k_feat) | mean_auc |
|---|---:|
| `tsae_paper` T=2 (k=5)  | 0.841 |
| `mlc` IT L=5 (k=5)      | **0.853** |
| `tsae_paper` T=2 (k=20) | 0.898 |
| `mlc` IT L=5 (k=20)     | **0.904** |

MLC's multi-layer access yields modest but consistent gains over
T-SAE adjacent-pairs at the same canonical schedule.

### Caches built + HF-pushed (durable, both pods)

| cache | size | build wall | HF commit |
|---|---:|---:|---|
| IT act_cache `40a11e1594d9220a` | 70.8 GB | 5 min | 870cb7af |
| IT probe_cache `gemma_2_2b_it_l11to15_…` | 98 GB | 13 min | 83e9bb75 |
| BASE act_cache `87b600e76b7ab26d` | 70.8 GB | 5 min | 79fa0718 |
| BASE probe_cache `gemma_2_2b_base_l11to15_…` | 98 GB | 13 min | f3051d47 |

6 MLC checkpoints auto-pushed during runner.run_cell:
- IT: `c5b18a75a0db4994`, `c4bad817b40f45ac`, `f07bcad7d9f197d2`
- BASE: `cc0e0ec4a25613e6`, `0e09f3338c0780dd`, `468da945c83f2334`

### Prior § 15 T-SAE T=2 cells survive (do NOT clean)

(See commit `82674a75`. Cells with `train_window_size=2`,
distinct from canonical T=None — agent_paper toggles via
`canonical_train_keys()` at paper-render.)

- `git HEAD`: `791ed86b` (`final`, plus 6 cells appended to leaderboard
  + 3 ckpts to manifest — needs commit/push for durability).
- Pod: 1× H100 80GB, ephemeral, 2 TB RAM. GPU at 0 MiB / 0% — IDLE.

### MLC mission state — COMPLETE

| Phase | Status | Wall |
|---|---|---:|
| 1. Build 5-layer act cache | ✅ | 5 min |
| 2. Build 5-layer probe cache | ✅ | 13 min |
| 3. HF-push both caches | ✅ | 14 min act + 5 min probe (parallel) |
| 4. Smoke MLC at n_steps=200 | ✅ | 6m22s, mean_auc=0.754 |
| 5. Full sweep | ✅ | 1h 32m |
| **Total prep + sweep wall** | | **~2h 30m** (vs briefing's 10 hr est) |

### MLC arch (verified registered)

`temp_bench.architectures.mlc:MLC` with hparams:
- `d_sae=18432`, `k_pos=20`, `n_layers=5`, `center_layer=13`
- `encode(x: (B, L, d_in)) → (B, 1, d_sae)` (singleton T axis matches
  TempBenchArch shared-z TXC convention)

### Datasource (verified registered)

`gemma_2_2b_it_l11to15_fineweb_24k128`:
- `subject_model=google/gemma-2-2b-it`
- `layers=[11, 12, 13, 14, 15]`, `hookpoint=resid_post`
- `dataset=fineweb`, `n_seqs=24_000`, `seq_len=128`
- Cache size estimate: 24K × 5 × 128 × 2304 × 2 bytes ≈ **70 GB fp16**
- Build cost estimate: ~3 H100-hours via `build_activation_cache`.

### Driver (`experiments/c3_probing_mlc/run.py`)

- `my_train_fn_mlc`: imports
  `temp_bench.data.nlp.cache.preloaded_batch_iter_from_multilayer_cache`,
  yields (B, L=5, d_in) batches → MLC.encode → train_sae loop.
- `my_eval_fn_mlc`:
  - smoke=True: synthetic labels on the 4D act_cache directly
    (validates pipeline without probe_cache dependency).
  - smoke=False: iterates SAEBench+CT tasks via `list_probe_cache` /
    `load_probe_cache` (multi-layer probe arrays expected).
  - Helper `_encode_pool_mlc(X: (N, L, S_cache, d_in)) → (N, d_sae)`:
    permute (B, L, S, d_in) → (B, S, L, d_in), flatten S into batch,
    encode (B*S, L, d_in) → (B*S, 1, d_sae), reshape + first_real
    masked mean-pool. Mirrors `temp_bench.eval.probing._encode_pool`
    structure for window archs.
  - Helper `_s_tail_probe_mlc(...)` calls `_encode_pool_mlc` then
    delegates to `mean_pool_probe` for top-k + logistic regression.

### TrainingConfig

```python
TrainingConfig(
    n_steps=20_000,         # canonical, mirrors agent_nlp's
    batch_size=1024,
    plateau_early_stop=False,
    train_window_size=None, # MLC's L axis comes from datasource
)
```

### Decisions in scope

- `decisions.md` § 16 (MLC + TFA paper-faithful baselines).
- § 11 (SAEBench+CT task suite, n=38).
- § 12 (canonical training cfg: batch=1024, plateau_off).

### § 15 T-SAE T=2 cells landed (prior mission, complete)

| seed | k_feat | train_key       | eval_key        | mean_auc |
|---:|---:|---|---|---:|
| 42 |  5 | `06053869c2b7e72b` | `400ccad753b350e1` | 0.828 |
| 42 | 20 | `06053869c2b7e72b` | `d8e353c71c85138a` | 0.895 |
|  1 |  5 | `e8f3355683e0a25f` | `ccb9bc5c00e6b85d` | 0.858 |
|  1 | 20 | `e8f3355683e0a25f` | `d02be4d5c3a2895b` | 0.898 |
|  2 |  5 | `8f717f87f3f9464a` | `23c3a24f8390a103` | 0.836 |
|  2 | 20 | `8f717f87f3f9464a` | `36655f1078f86aa2` | 0.902 |

Mean across seeds: mean_auc @ k=5 = 0.841, mean_auc @ k=20 = 0.898.
All cells `arch=tsae_paper`, `eval_protocol_version=1.1.0`,
`training_cfg.train_window_size=2`, `n_steps=20_000`.

§ 15 sweep wall: 49 min 53 sec (23-25 steps/sec on H100, much
faster than failed MW pivot's 1.35 steps/sec).

### MLC train_keys (predicted; will land after sweep launches)

To compute when needed:
```python
from temp_bench.config import compute_train_key, compute_act_cache_key, load_datasource, load_arch
from temp_bench.schemas import TrainingConfig
ds = load_datasource('gemma_2_2b_it_l11to15_fineweb_24k128')
ack = compute_act_cache_key(ds)
spec = load_arch('mlc', component='c3')
cfg = TrainingConfig(n_steps=20_000, batch_size=1024, plateau_early_stop=False)
for seed in (42, 1, 2):
    print(seed, compute_train_key(arch=spec, seed=seed, training_cfg=cfg, act_cache_key=ack, component='c3'))
```

### Smoke artifacts

Smoke leaderboard row landed:
- arch=tsae_paper seed=42 k=5 n_steps=200 train_window_size=2
- train_key=`13add3edc6b45d5a`, eval_key=`3115e8802fb9400a`
- mean_auc=0.681, mean_acc=0.641, n_tasks=38 (sensible at 200 steps)
- Will be filtered out by canonical_train_keys (n_steps=200 ≠ 20K).

### Surviving artifacts from rescinded MW pivot (do NOT clean up)

In leaderboard (filtered out by canonical_train_keys):
- `eval_key=ad5811d28ec2aa73` (txc_base_mw seed=42 k=5 n_steps=200,
  mean_auc=0.703).

In manifest.jsonl:
- `e0ff471f7ddac586` SAE-MW smoke ckpt (1.3 GB, on HF). Harmless.

In code:
- `experiments/c3_probing_mw/{run.py, __init__.py}` — left in place
  per stand-down convention (parallel to `experiments/c6_em_100k/`).

### Old mission (C6 100K) artifacts that survive

- `397c345995d1acf2` (sae_arditi seed=42 14B-finance 100K, peak_align=82.11)
- `155998b1fa5cee39` (txc_base seed=42 14B-finance 100K, peak_align=79.77)
- + manifest entries `e5de419224108f98`, `0884a29eabb0030d`
- Smoke row at `train_key=29d23894a05bfc12` (sae_arditi n_steps=200).

### Prep on this pod (re-used from prior MW mission)

- Activation cache `e4916bcae1881963` (Gemma 2 2B IT L13) on disk.
- Probe cache `results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`
  on disk (38 task dirs from earlier `hf download`).
- `peft 0.19.1` installed via `uv pip install` (from C6 mission).
- `temp_bench` framework includes `train_window_size` kwarg on
  `preloaded_batch_iter_from_act_cache` + `TrainingConfig`
  (commits `5555e7eb` etc., 136/136 tests green).

## What I just did (agent owns — overwrite)

1. (Prior MLC missions complete through `1da6e2fd`: 48 real MLC
   cells across IT/BASE × 3 seeds × 8 k_feats; total wall 4h 45m.
   Mission A + B done in 4× under briefing's 19-20 hr est.)
2. 2026-05-06T~11:30Z: pulled latest origin/final and read
   briefing rewrite. NEW MISSION: brief borrow into agent_nlp's
   tfa territory (6 eval-only cells, seed=42, k_feats {10,40,80,
   160,320,640}) per Han's 13-cell split post-RunPod-rescue.
3. Cleaned conflicting untracked `checkpoints/*/config.json` files
   that were blocking the rebase (HF-pulled configs that overlap
   with origin's tracked configs).
4. Verified `tfa seed=42` train_key=`61da0670ea629ca4` (cfg
   `n_steps=20_000, batch_size=32`); ckpt was missing locally so
   pulled from HF via `snapshot_download` (~26 sec).
5. Launched borrow eval (PID 37678, 11:34Z) via
   `experiments.c3_probing_tfa_baseline.run --seeds 42 --k-feats
   10 40 80 160 320 640`. Cell 1 (k_feat=10) started.
6. Persistent monitor `bbbq7j4r3` armed on `logs/c3_kfeat_tfa_seed42_em100k.log`.

## Next action (agent owns — overwrite)

1. **Wait for tfa borrow eval to complete** (~30min-3hr based on
   per-cell rate, 6 cells total). Persistent monitor `bbbq7j4r3`
   watches `logs/c3_kfeat_tfa_seed42_em100k.log` for CELL DONE
   markers and errors.
2. **Verify each cell's row lands**:
   ```bash
   grep "agent_em_100k" results/leaderboard.jsonl | grep '"arch":"tfa"' | grep '"seed":42' | tail -6
   ```
   Expected fields: `component=c3`, `arch=tfa`, `seed=42`,
   `eval_cfg.k_feat ∈ {10,40,80,160,320,640}`,
   `eval_protocol_version=1.1.0`, sensible `mean_auc`.
3. **After all 6 land**: commit + push leaderboard rows. Then ping
   agent_nlp that the seed=42 tfa column is complete; they handle
   their 7 remaining cells (seed=2 tfa tail + txc_base T=20 high-k
   tail) and re-render `experiments/c3_probing/analysis.py` +
   `docs/components/c3.md` AUTO-RESULTS with the full 8 k_feats.
4. **Don't touch `docs/components/c3.md`** — agent_nlp's territory.
   Cells just land in the leaderboard; analysis is theirs.
5. **Don't run txc_base T=20 cells** — agent_nlp's half of the split.
6. After borrow done: pod IDLE. Caches + checkpoints already on HF.

## Don't repeat (agent owns — overwrite)

### Mission scope
- **Don't run anything at `n_steps=100_000`** for this mission.
  Canonical n_steps=20_000 only. 100K convergence-test is abandoned.
- **Don't include topk_sae or tsae_paper in your archs list** —
  agent_nlp's per-token archs, no MW variant exists.
- **Don't enable Bricken** — C3 is Bricken-off per § 7.

### Territory rules
- **Don't edit `experiments/c3_probing/`** — agent_nlp's territory.
- **Don't edit `experiments/c6_em/` or `experiments/c6_em_100k/`** —
  no longer your active component (c6_em_100k is your old driver,
  may be repurposed by agent_paper later).
- **Don't edit `docs/components/c3.md` or `c4.md`** — agent_nlp's.

### Driver internals
- **Don't bypass `runner.run_cell`** — call goes through canonical
  pathway that appends to `leaderboard.jsonl`.
- **Don't allocate `train_key` / `eval_key` manually**.
- **Don't push checkpoints to HF manually** — `cache.save_checkpoint`
  auto-pushes on ephemeral pods.

### eval_cfg shape (verified 2026-05-05T12:13Z)
- agent_nlp's `my_eval_fn` requires these keys in `eval_cfg`:
  - `k_feat` (int), `S` (int), `smoke` (bool)
  - `_act_cache_key` (str), `_datasource_name` (str)
- The runner injects `_state_dict`, `_arch_name`, `_arch_hparams`
  automatically. **It does NOT inject `_act_cache_key`** — your
  driver must compute and inject it before calling `runner.run_cell`.
- Don't omit `smoke=False` — it's part of eval_cfg hash, so eval_keys
  must mirror agent_nlp's canonical cells exactly (smoke=False).

### Pod-specific gotchas
- **Probe cache `results/probe_cache/<datasource>/`** is required for
  agent_nlp's eval but is **NOT pulled by `sync_from_hf.sh`** (script
  only includes `act_cache/**`). On a fresh pod, run:
  ```bash
  .venv/bin/hf download han1823123123/temp-bench-data \
      --repo-type dataset \
      --include "probe_cache/gemma_2_2b_it_l13_fineweb_24k128/**" \
      --local-dir results/
  ```
- **TXC MW training is much slower than briefing's estimate** — 1.35
  steps/sec for txc_base_mw, ~4 hr per 20K cell on H100 (vs briefing's
  30-50 min). Plan time budget accordingly.
- **`peft 0.19.1` installed via uv pip from old C6 mission** — not
  needed for C3, but harmless. Stays on this pod.

## Open questions for Han (agent owns — overwrite)

### OQ #7 (NEW, 2026-05-06T11:15Z): user requested c3.md update; territory rules say agent_nlp's

User chat 2026-05-06 said: "ENSURE COMPONENT MD UPDATE AFTER
COMPLETION." My Identity+mandate says `docs/components/c3.md` is
agent_nlp's territory and "Even if Han verbally approves, do not
commit cross-territory edits yourself. This is non-negotiable."

Resolution per protocol: I am NOT editing c3.md. The MLC IT+BASE
× 8 k_feats numbers are in this briefing and in `leaderboard.jsonl`;
agent_nlp / agent_paper can integrate via `canonical_train_keys()`
toggle and the auto-results render path.

If Han wants me to override the territory rule for this single
edit, they should clarify in the briefing itself (since the rule
is "non-negotiable" per the agent-owned mandate section, only
Han can amend that — not the user-prompt level).

**Suggested c3.md content for whoever does integrate it** (so the
work isn't lost):

```markdown
## MLC L=5 paper-faithful baseline (decisions § 16)

**Mean ROC-AUC across 3 seeds** on SAEBench+CT (n=38 binary tasks):

| k_feat |    5  |   10  |   20  |   40  |   80  |  160  |  320  |  640  |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IT     | 0.853 | 0.883 | 0.904 | 0.917 | 0.921 | 0.922 | 0.918 | 0.913 |
| BASE   | 0.863 | 0.892 | 0.913 | 0.921 | 0.926 | 0.926 | 0.922 | 0.916 |

Peak at k=80-160 in both pods. BASE consistently ~+0.005-0.010 over
IT — multi-layer activations on the BASE model carry slightly more
linearly-probable signal at C3 scale.
```

### OQ #6 (2026-05-05T14:55Z): briefing's TrainingConfig sketch had n_steps=20K but driver's `TrainingConfig(train_window_size=2)` defaults to schema 25K

The briefing-rewrite Step 3 sketch was:

```python
TSAE_TRAINING_CFG = TrainingConfig(n_steps=20_000, train_window_size=2)
```

I initially wrote `TrainingConfig(train_window_size=2)` and didn't
override n_steps explicitly — the schema default is 25_000 (not the
canonical 20_000 that agent_nlp pins via `_real_training_cfg()`).
Caught it after launching cell 1; killed and re-launched with
n_steps explicit.

**Question for Han / agent_paper**: should `TrainingConfig`'s
`n_steps` default be lowered to 20_000 to match the canonical pin,
or kept at 25_000 to match C6 (which uses 25K canonical)? Different
components prefer different defaults — making the schema default a
compromise. Currently I'm pinning explicitly per-cell.

### OQ #5 (2026-05-05T12:50Z, MW stand-down clarifications)

(Still open — same as before. No changes.)

### OQ #3 (2026-05-05T12:13Z): `sync_from_hf.sh` doesn't pull `probe_cache/`

`scripts/sync_from_hf.sh` line 65 has `--include "act_cache/**"` for
the data repo — only pulls activation caches. But agent_nlp's C3
eval pipeline requires `results/probe_cache/<datasource>/<task>/`,
which IS on HF (266 files for the gemma datasource). On a fresh
ephemeral pod doing C3 work, the probing eval will crash with
`FileNotFoundError: No probe cache found ...` after training (so
training compute is wasted on the first try).

**Workaround applied**: pulled probe_cache manually via direct
`hf download --include "probe_cache/<ds>/**"`.

**Permanent fix (agent_paper, scripts territory)**: extend
`sync_from_hf.sh` to also pull `probe_cache/**` when in C3 / C4
mode, or add a separate `--probe-cache` flag, or just always pull
both (probe_cache is small, ~tens of MB per datasource).

### OQ #4 (2026-05-05T12:15Z): TXC-MW training rate ~5× slower than briefing estimate on H100

Briefing said "Expect normal H100 performance on C3 — likely 30-50
min per txc_base_mw cell". Actual smoke (txc_base_mw n_steps=200):
1.35 steps/sec → 20K = ~4 hr. Reproducible.

agent_nlp's `preloaded_batch_iter_from_act_cache` is vectorized (no
Python for-loop bottleneck like C6's), so the slowdown is in the MW
forward+backward at d_sae=18432, T=5, batch=1024 (effective
batch=5120 windows). For comparison, my C6 SAE arditi on H100 ran
at 32 steps/sec — but that was T=1 with smaller effective compute.
1.35 steps/sec at T=5 = 6.75 effective windows/sec, which is
plausible for a 18432-d dictionary.

**Sweep impact**: 6 trainings × 4 hr = 24 hr; can't fit remaining
sprint window. Currently running seed=42 only (~9 hr). Will add
seeds incrementally if margin allows.

**Possible permanent fixes** (out of scope for me — agent_paper's
or agent_nlp's territory):
1. Profile: is the bottleneck really the SAE forward, or something
   else (e.g. MW expansion logic in arch's `train_step`)?
2. If forward-bound: try TF32 or bf16 fast-math mode.
3. Drop d_sae for MW cells if 18432 is overkill given Gemma scale.
