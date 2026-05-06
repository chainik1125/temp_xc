<!--
DRAFT — written by agent_paper 2026-05-04 PM, REWRITTEN MULTIPLE TIMES
to reflect successive mission pivots:
  - 2026-05-05 AM: 100K convergence-test → C5 MW pivot (rescinded)
  - 2026-05-05 PM: C5 MW → C7 MW pivot (rescinded — § 14 deprecated)
  - 2026-05-05 PM: STAND DOWN, idle (post-§ 15 baselines complete)
  - 2026-05-05 PM: Idle → C6 TFA mission (RESCINDED below — too long
    + judge-API-cost intensive)
  - 2026-05-05 PM: C6 TFA → BASE C3 + C4 replication (CURRENT)
Section ownership: PROTOCOL.md § 14.
-->

---
agent: agent_steer_100k
last_state_update: 2026-05-05T18:00:00Z
component: c3-c4-base
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent STEER 100K** (legacy name; mission has pivoted multiple
times). Your **CURRENT MISSION** is to **replicate C3 + C4 (sparse
probing + qualitative latents) on the BASE Gemma model** —
`google/gemma-2-2b` (NOT the `-it` instruction-tuned variant that
agent_nlp + agent_em_100k use). Same archs, same TrainingConfigs, same
T / sampling / batch_size as the IT replication. The BASE
replication validates that C3/C4 results generalize beyond the
instruction-tuned model and gives reviewers a cross-model
robustness check.

Files you may edit:

- `agents/agent_steer_100k/briefing.md` (your own — agent-owned sections only)
- `experiments/c3_probing_base/` (NEW — your driver dir for BASE C3)
- `experiments/c4_qualitative_base/` (NEW — your driver dir for BASE C4)
- Code under `src/temp_bench/` that you author + commit — but in
  practice you should not need to write any. agent_nlp's C3 + C4
  plumbing handles everything via imports; only the datasource string
  changes.

**Files that are OUT OF SCOPE — do NOT edit:**
- `agents/agent_*/` — every other agent's directory.
- `experiments/c3_probing/`, `experiments/c4_qualitative/` —
  agent_nlp's territory. You import from them without modification.
- `experiments/c3_probing_topk_baseline/`,
  `experiments/c3_probing_tfa_baseline/`,
  `experiments/c3_probing_tsae_baseline/`,
  `experiments/c3_probing_mlc/` — agent_nlp / agent_em_100k drivers.
  You write your OWN drivers in `experiments/c3_probing_base/` and
  `experiments/c4_qualitative_base/`.
- `experiments/c5_steering*/`, `experiments/c6_em*/`, `experiments/c7_*/`
  — other agents' territories.
- `docs/components/c3.md`, `docs/components/c4.md` — agent_nlp's
  territory. agent_paper integrates the BASE replication results at
  paper-render time.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `configs/datasources.yaml` — agent_paper has already added the BASE
  multi-layer + BASE concat entries you need. Don't add more.
- `pyproject.toml` and `uv.lock` — atomic, agent_paper coordinates.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.
This is non-negotiable — see PROTOCOL.md § 8 + CLAUDE.md Hard Rule #7.

### 🔥 BUG FIX 2026-05-06 — re-train + re-eval BASE C3 txc_base T=10/T=20

**agent_filler reported (Open Q #4)**: BASE C3 `txc_base` T=10/T=20
evals crash:

```
RuntimeError: Error(s) in loading state_dict for TXCBase:
  size mismatch for W_enc: shape [5, 2304, 18432] from checkpoint
  vs current [10, 2304, 18432]
```

**Root cause** (NOT in your driver — it was upstream in agent_nlp's
shared `experiments/c3_probing/run.py:my_train_fn`):

The training-time code path was calling `load_arch(arch_name)` and
`instantiate_arch(spec, d_in)` WITHOUT applying the runner-merged
`arch_hparams` (which include `training_cfg.arch_hparams_override`).
So when `arch_hparams_override={"T": 10}` was set, the runner correctly
hashed the train_key with T=10 (different from T=5), but the actual
TRAINED model was instantiated at T=5 (YAML default). The T=5-shape
weights got saved at the T=10-keyed checkpoint dir.

Your `my_eval_fn` was correct (it DID apply `_arch_hparams` from the
runner) — that's why it surfaced as an eval-time crash. The trained
checkpoints are wrong; the eval-time instantiation is right.

**agent_paper landed the fix at commit `<INSERT-HASH>` 2026-05-06**:

```python
# experiments/c3_probing/run.py:my_train_fn (line 97-99 area)
spec = load_arch(arch_name, component=component)
# NEW: apply runner-merged arch_hparams (incl. arch_hparams_override).
spec = spec.model_copy(update={"hparams": dict(arch_hparams)})
d_in = _d_in_from_act_cache(act_cache_key)
model = instantiate_arch(spec, d_in=d_in)
```

Same fix applied symmetrically to
`experiments/c5_steering/run.py:my_train_fn` (defensive — agent_steer
hasn't hit the bug there because they haven't run T-overrides at C5,
but the symmetric fix prevents future surprises).

### What you need to do

**You don't need to write code** — the fix is in the imported function
your driver calls. But you DO need to:

1. **`git pull --rebase origin final`** to pick up the fix.
2. **Re-train** the broken `txc_base` T=10 + T=20 cells on BASE. The
   existing checkpoints at the T=10/T=20 train_keys hold T=5-shape
   weights (incorrect). Re-running with `force_train=True` overwrites
   them with correctly-shaped weights at the same train_key (since
   train_key already had T=10/T=20 in the hash).
3. **Re-run evals** for those cells (they now succeed).
4. **NB: agent_filler already saved the buggy T=10 checkpoints to HF**
   per their Phase-D notes. The re-train will OVERWRITE both local +
   HF — different file content under the same `<train_key>/` HF dir.
   This is fine: the train_key didn't change; only the saved weights
   are corrected.

### Concrete commands

```bash
cd /workspace/temp_xc/purified
git pull --rebase origin final

# Re-train + re-eval the 6 broken cells (txc_base × {T=10, T=20} × 3 seeds).
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c3_probing_base.run \
  --archs txc_base \
  --seeds 42 1 2 \
  --k-feats 5 20 \
  --force-train \
  > logs/c3_base_txc_T_sweep_refix.log 2>&1 &

# Per-cell on H100: ~1.5-2 hr train + ~30 min eval × 2 k_feats = ~2.5 hr.
# 6 cells serial → ~15 hr OR parallel across multiple GPUs if available.
```

Once those land, re-run the **k_feats expansion** for the same cells
at the 6 new k_feats {10, 40, 80, 160, 320, 640} — eval-only after
the re-train, cache-hits on the freshly-correct checkpoints:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c3_probing_base.run \
  --archs txc_base \
  --seeds 42 1 2 \
  --k-feats 10 40 80 160 320 640 \
  > logs/c3_base_txc_T_sweep_kfeat_expand.log 2>&1 &
```

### Watch-outs

- **Don't add `--force-eval` AND `--force-train`** unless your
  driver's CLI supports both at once. If only `--force-train` is
  exposed, that's enough — eval re-runs after train automatically.
- **Don't worry about the T=5 cells** — they were always correct
  (no override → no bug). 6 BASE T=5 cells in leaderboard stand.
- **agent_filler's BASE `txc_pro` cells** (3 seeds × 2 k_feats = 6
  cells) DON'T have this bug — txc_pro doesn't use `arch_hparams_override`
  in agent_steer_100k's driver. They're correct as-is.
- **C4 BASE evals** still pending after this re-train — they
  cache-hit on the freshly-correct C3 BASE checkpoints. Same
  sequencing as before.

---

### ⚠️ NEW MISSION 2026-05-06 (URGENT) — C3 BASE k_feats expansion {5, 10, 20, 40, 80, 160, 320, 640}

**Han 2026-05-06**: "current C3 has k {5,20} we want to expand to
{5,10,20,40,80,160,320,640} for all SPARSE PROBES in C3 for both IT
and BASE!" Your job: **BASE side**. agent_nlp handles IT in parallel.

**This is eval-only — NO RE-TRAINING.** Your existing BASE C3
checkpoints (TopK, T-SAE, TFA + agent_filler's TXC-base/pro × 3 seeds)
stay; just run the probing eval at 6 new k_feat values and let them
cache-hit on training.

### Mission scope (BASE side only)

| Arch | Seeds | New k_feats | New evals |
|---|---|---|---:|
| `topk_sae` | {1, 2, 42} | {10, 40, 80, 160, 320, 640} | 18 |
| `tsae_paper` | {1, 2, 42} | {10, 40, 80, 160, 320, 640} | 18 |
| `tfa` | {1, 2, 42} | {10, 40, 80, 160, 320, 640} | 18 |
| `txc_base` (T=5 + T=10 + T=20) | {1, 2, 42} | {10, 40, 80, 160, 320, 640} | 54 |
| `txc_pro` | {1, 2, 42} | {10, 40, 80, 160, 320, 640} | 18 |
| **Total new BASE evals** | | | **126** |

(Plus BASE MLC if agent_em_100k builds it — coordinate with them.)

### Per-cell wall-time (eval-only)

Eval-only: ~5-15 min encode + ~minute per (k_feat, task) probe fit on
your H100. Per (arch, seed) cell with 6 new k_feats: ~12-15 min total.

- 21 cells (5 archs base configs + 3 T-sweep variants for txc_base
  with 3 seeds each = 21 cells) × ~15 min = **~5-6 hr serial on
  1× H100**.
- TXC cells from agent_filler may not have all landed yet; eval
  those as they finish, in parallel with the per-token archs whose
  checkpoints you already own.

### First concrete task — extend the existing driver

Your existing `experiments/c3_probing_base/run.py` already supports
`--k-feats`. Relaunch with the new values:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c3_probing_base.run \
  --archs topk_sae tsae_paper tfa txc_base txc_pro \
  --seeds 42 1 2 \
  --k-feats 10 40 80 160 320 640 \
  > logs/c3_base_kfeat_expand.log 2>&1 &
```

If TXC cells aren't all in the leaderboard yet (agent_filler still
running), the runner will skip them with a clear message. Re-launch
later for the late-landing TXC cells — idempotent.

### Concurrent work: C4 BASE evals still pending

Your earlier mission (C4 BASE qualitative eval, cache-hit on
agent_filler's TXC checkpoints) is still on the table. Sequence:
- (a) C3 BASE k_feats expansion (this mission, ~5-6 hr).
- (b) C4 BASE qualitative eval once agent_filler's TXC-base + TXC-pro
  checkpoints land (~1-2 hr).

These don't conflict — different drivers, both on your H100.

### Watch-outs

- **Eval-only.** Don't re-train. Pass the same TrainingConfigs you
  used; runner cache-hits on training, fresh eval_keys for new k_feats.
- **Don't bump `EVAL_PROTOCOL_VERSION`**. Existing rows at k_feat ∈
  {5, 20} stay valid; new rows just append at the new k_feats.
- **Don't render `docs/components/c3.md`** — agent_nlp's territory.
  agent_paper integrates IT + BASE results at paper-render time.

---

### ⚠️ LOAD SPLIT 2026-05-05 PM — agent_filler is taking your TXC archs on BASE

**Han 2026-05-05 PM**: "agent_filler's 8 A40s ... NEED TO UTILIZE.
Let's make them take half of agent_steer_100k's load." +
**"inform agent_steer_100k that agent_filler is taking some of their
workload!!"**

**Effective immediately, your BASE C3 mission is split**:

| Owner | Archs on BASE C3 |
|---|---|
| **YOU (agent_steer_100k)** | `topk_sae` (~done: 7 cells), `tsae_paper` (~done: 6 cells), `tfa` (in flight: 4 cells); plus C4 evals once TXC checkpoints land |
| **agent_filler** (8× A40 pod) | `txc_base` (T=5, T=10, T=20) + `txc_pro` × 3 seeds = 12 trainings + 24 evals |

**What you DO**:

1. **Finish your in-flight TFA cells** (~remaining 2 cells × ~5 hr =
   ~10 hr) — already on track.
2. **Skip TXC-base + TXC-pro entirely** on BASE. Don't run them; do
   not start the txc_base T=10/T=20 sweep on your H100. agent_filler
   parallelizes those 12 trainings across 8 A40s in ~5-6 hr wall —
   much faster than your 1× H100 could.
3. **Run C4 BASE evals** once agent_filler's TXC checkpoints land
   (~5-6 hr from now). C4 is eval-only (cache-hit on the C3
   checkpoints via shared `train_key`); ~1-2 hr on your H100.
4. **The § 17 txc_base T-sweep mission below** (T=10, T=20 on BASE)
   is now agent_filler's. Read that section for context but **do
   NOT execute it**.

**What changed for you**:

- ~12-13 hr of compute (txc_base T=10/20 + txc_pro × 3 seeds) is now
  agent_filler's. Your remaining wall drops from ~28 hr → ~15-16 hr.
- Both pods write to the same `leaderboard.jsonl` and `manifest.jsonl`;
  cells deduplicate by `train_key`. agent_filler will pull your BASE
  act_cache + probe_cache from HF (you already pushed both).
- C4 BASE evals stay yours — you have the qualitative-judge plumbing
  cached on your H100 (Anthropic API key, etc).

**Verify before you start anything new**:

```bash
# Confirm agent_filler is queued / running TXC archs on BASE.
grep -A 2 "BASE C3 TXC split" agents/agent_filler/briefing.md | head -10

# Verify your remaining in-flight work is per-token + TFA only.
ps -ef | grep "experiments.c3_probing_base" | grep -v grep
```

If you see yourself running TXC cells on BASE, **stop and stand down**
— agent_filler has it.

---

### ⚠️ ADDITIONAL MISSION 2026-05-05 PM (URGENT) — TXC-base T-sweep on C3 + C4 BASE [REASSIGNED 2026-05-05 PM TO agent_filler — see LOAD SPLIT above; section preserved for context]

**Han 2026-05-05 PM**: "we want a txc_base T=10 and T=20 on C3 and C4
(both IT and base)." Your job is the **BASE side**;
agent_nlp handles the IT side in parallel on their pod.

**This stacks on top of your existing BASE C3+C4 replication** (the
mission below). Same datasource (`gemma_2_2b_base_l13_fineweb_24k128`),
same training schedule, just adds two new T values for `txc_base` via
`arch_hparams_override`.

### Mission scope (additive)

| Component | Arch × T | Seeds | k_feats | Cells |
|---|---|---|---|---:|
| C3 BASE | txc_base × {T=10, T=20} | {1, 2, 42} | {5, 20} | 6 trainings + 12 evals |
| C4 BASE | txc_base × {T=10, T=20} | {1, 2, 42} | (concat) | 6 evals (cache-hits on C3 trainings) |

**Total unique trainings: 6** (3 seeds × 2 T values for txc_base on
the BASE cache). C4 evals re-use the C3 checkpoints.

### TrainingConfig

```python
TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
    arch_hparams_override={"T": 10},   # or 20
)
```

`arch_hparams_override` flows into `compute_train_key` (commit
`dfd60850`); fresh hashes for each T value.

### Wall-time on 1× H100

- T=10 per cell: ~1.5 hr train + ~30 min eval × 2 k_feats = ~2 hr
- T=20 per cell: ~2 hr train + ~30 min eval × 2 k_feats = ~2.5 hr
- 6 trainings serial on 1 H100: **~12-13 hr** for both T values.

### Sequencing with your other BASE work

Two options depending on where you are in the BASE replication:

- **Option A (recommended)**: finish your existing BASE C3 sweep
  for the canonical 5 archs (TopK + T-SAE + TFA + TXC-base T=5 +
  TXC-pro), then launch the txc_base T=10/T=20 sweep. Adds ~12-13
  hr to your total.
- **Option B**: launch the T=10/T=20 sweep in parallel from the start
  if you have the BASE cache already built. Same compute total,
  just interleaved.

Both fit comfortably in the remaining sprint window.

### First concrete task — extend your driver

Your existing `experiments/c3_probing_base/run.py` driver already
has the `arch_hparams_override` mechanism wired in (you copy-pasted
agent_nlp's plumbing). Just add T=10 + T=20 entries to your
ARCH_TRAINING_CFGS:

```python
ARCH_TRAINING_CFGS: dict[str, list[TrainingConfig]] = {
    # existing canonical 5 archs (one cfg each)
    "topk_sae":   [TrainingConfig(n_steps=20_000, train_window_size=1)],
    "tsae_paper": [TrainingConfig(n_steps=20_000, train_window_size=2)],
    "tfa":        [TrainingConfig(n_steps=20_000, batch_size=32)],
    "txc_base":   [
        TrainingConfig(n_steps=20_000),                                       # T=5 default
        TrainingConfig(n_steps=20_000, arch_hparams_override={"T": 10}),      # NEW
        TrainingConfig(n_steps=20_000, arch_hparams_override={"T": 20}),      # NEW
    ],
    "txc_pro":    [TrainingConfig(n_steps=20_000)],
}
```

Then iterate over `(arch, cfg)` pairs in your sweep loop. Same C4
extension via the qualitative driver.

### Analysis filter update (after cells land)

Add txc_base T=10 + T=20 to the canonical filter (mirroring
agent_nlp's IT analysis):

```python
txc_T10_keys_base = canonical_train_keys(
    component="c3",
    archs=["txc_base"],
    seeds=(1, 2, 42),
    datasource_names=("gemma_2_2b_base_l13_fineweb_24k128",),
    training_cfg=TrainingConfig(n_steps=20_000, arch_hparams_override={"T": 10}),
)
txc_T20_keys_base = canonical_train_keys(... arch_hparams_override={"T": 20})
```

### Watch-outs

- **Don't add T-sweep for other archs.** Only `txc_base` gets T=10
  and T=20 in this mission.
- **agent_nlp runs the IT side**; don't duplicate. You only run the
  BASE-side cells (`gemma_2_2b_base_l13_fineweb_24k128`).
- **C4 cache-hits on C3.** When you run C4 BASE evals, use the same
  T-overrides → same `train_key` → same checkpoint → eval-only.

---

### ⚠️ NEW MISSION 2026-05-05 PM — BASE C3 + C4 replication (decisions § 16)

**Mandate**: replicate the C3 sparse-probing benchmark AND C4
qualitative-latents Pareto on the BASE Gemma-2-2B model. Match
agent_nlp + agent_em_100k's per-arch TrainingConfigs **exactly** —
this is the load-bearing constraint Han emphasized: "incredibly
important to use the same choice of T, sampling etc that agent_nlp
and agent_em_100k are using."

**Subject model**: `google/gemma-2-2b` (BASE, NOT `-it`).
**Datasource for C3**: `gemma_2_2b_base_l13_fineweb_24k128` (already in
`configs/datasources.yaml`; cache must be built — ~3 H100-hr).
**Datasource for C4**: `gemma_2_2b_base_l13_concat_v1` (just added to
`configs/datasources.yaml` in this push by agent_paper; small cache,
fast build).
**Layer**: L13 (mirrors agent_nlp's IT side; direct comparison).

### Per-arch TrainingConfig — MUST MATCH IT exactly

| Arch | TrainingConfig (BASE = IT, only datasource differs) |
|---|---|
| `topk_sae` | `n_steps=20_000, batch_size=1024, train_window_size=1` |
| `tsae_paper` | `n_steps=20_000, batch_size=1024, train_window_size=2` |
| `tfa` | `n_steps=20_000, batch_size=32, train_window_size=None` |
| `txc_base` | `n_steps=20_000, batch_size=1024, train_window_size=None` (internal T=5 sampling) |
| `txc_pro` | `n_steps=20_000, batch_size=1024, train_window_size=None` (internal T=5+max_shift sampling) |
| `mlc` (Tier 2 stretch) | `n_steps=20_000, batch_size=1024, train_window_size=None` (multi-layer datasource) |

**Same eval_protocol_version** as agent_nlp: `EVAL_PROTOCOL_VERSION = "1.1.0"`.

### Mission scope (Tier 1: 5 archs without MLC)

| Phase | Work | Wall time on H100 |
|---|---|---:|
| 0 | Build BASE single-layer act cache (`...l13_fineweb_24k128`) | ~3 hr |
| 1 | Build BASE single-layer probe_cache (38 SAEBench+CT tasks) | ~1.5 hr |
| 2 | TopK × 3 seeds × 2 k_feats (T=1) | ~3.6 hr |
| 3 | T-SAE × 3 seeds × 2 k_feats (T=2) | ~4.5 hr |
| 4 | TFA × 3 seeds × 2 k_feats (B=32, full seq) | ~5 hr |
| 5 | TXC-base × 3 seeds × 2 k_feats (T=5 internal) | ~3.6 hr |
| 6 | TXC-pro × 3 seeds × 2 k_feats (T=8 internal, slower InfoNCE) | ~6 hr |
| 7 | C4 eval on concat_v1 BASE × 5 archs × 3 seeds | ~1.5 hr |
| **Tier 1 total** | | **~28-29 hr** |

Tier 2 stretch (only if Tier 1 wraps with margin):
- Build BASE multi-layer act cache (`...l11to15_fineweb_24k128`): +3 hr
- Build BASE multi-layer probe_cache: +1.5 hr
- MLC × 3 seeds × 2 k_feats: +5 hr
- → Adds ~10 hr

If sprint window is tight, **drop TXC-pro first** (saves ~6 hr) since
agent_nlp's IT TXC-pro vs TXC-base were within noise at C3. Headline
4 archs (TopK, T-SAE, TFA, TXC-base) = ~17 hr training+eval + ~6 hr
caches + ~1.5 hr C4 = **~24 hr total minimum-viable** for the BASE
replication.

### First concrete task — verify, build caches, write drivers, launch

Step 0 — `git pull --rebase origin final`. Verify framework + datasources:

```bash
.venv/bin/python -c "
from temp_bench.config import load_datasource, compute_act_cache_key
ds = load_datasource('gemma_2_2b_base_l13_fineweb_24k128')
print('subject:', ds.subject_model)
print('layer:', ds.layer)
print('act_cache_key:', compute_act_cache_key(ds))
ds2 = load_datasource('gemma_2_2b_base_l13_concat_v1')
print('C4 BASE: subject=', ds2.subject_model, 'layer=', ds2.layer)
"
```

Step 1 — **build the BASE single-layer activation cache** (~3 hr):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -c "
from temp_bench.data.nlp.cache import build_activation_cache
build_activation_cache('gemma_2_2b_base_l13_fineweb_24k128')
" 2>&1 | tee logs/c3_base_cache_build.log
```

Verify:

```bash
.venv/bin/python -c "
from temp_bench.config import compute_act_cache_key, load_datasource, act_cache_dir
import numpy as np
key = compute_act_cache_key(load_datasource('gemma_2_2b_base_l13_fineweb_24k128'))
acts = np.load(act_cache_dir(key) / 'acts.npy', mmap_mode='r')
print('shape:', acts.shape, 'dtype:', acts.dtype, 'size:', acts.nbytes/2**30, 'GB')
"
```

**HF push the cache** (ephemeral pod) before any pod restart:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
from temp_bench.config import compute_act_cache_key, load_datasource
key = compute_act_cache_key(load_datasource('gemma_2_2b_base_l13_fineweb_24k128'))
api = HfApi()
api.upload_folder(
    folder_path=f'results/act_cache/{key}',
    path_in_repo=f'act_cache/{key}',
    repo_id='han1823123123/temp-bench-data',
    repo_type='dataset',
)
"
```

Step 2 — **build the BASE single-layer probe_cache** (~1.5 hr):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -c "
from temp_bench.data.nlp.probe_cache import build_probe_cache
build_probe_cache('gemma_2_2b_base_l13_fineweb_24k128')
" 2>&1 | tee logs/c3_base_probe_cache_build.log
```

HF push:

```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='results/probe_cache/gemma_2_2b_base_l13_fineweb_24k128',
    path_in_repo='probe_cache/gemma_2_2b_base_l13_fineweb_24k128',
    repo_id='han1823123123/temp-bench-data',
    repo_type='dataset',
)
"
```

Step 3 — **write `experiments/c3_probing_base/{__init__.py, run.py}`**.
Same pattern as agent_nlp's per-arch baseline drivers
(`experiments/c3_probing_topk_baseline/`,
`experiments/c3_probing_tsae_baseline/`,
`experiments/c3_probing_tfa_baseline/`) but ONE driver that handles
all 5 archs in a sweep, parameterised by --arch + --seed:

```python
"""C3 BASE replication — sparse probing on google/gemma-2-2b L13.
Mirrors agent_nlp's IT setup arch-for-arch, only the datasource
differs.

Per-arch TrainingConfig MUST match agent_nlp + agent_em_100k's IT
conventions exactly (decisions § 15 + § 16):
  topk_sae:   B=1024, train_window_size=1
  tsae_paper: B=1024, train_window_size=2
  tfa:        B=32,   train_window_size=None
  txc_base:   B=1024, train_window_size=None (internal T=5)
  txc_pro:    B=1024, train_window_size=None (internal T=5+shift)
"""
from __future__ import annotations
import argparse
from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from experiments.c3_probing.run import (
    EVAL_PROTOCOL_VERSION,
    my_train_fn,
    my_eval_fn,
)
from temp_bench.config import compute_act_cache_key, load_datasource


# BASE datasource — only thing that differs vs agent_nlp's IT setup.
DATASOURCE = "gemma_2_2b_base_l13_fineweb_24k128"


# Per-arch TrainingConfig overrides. Same as agent_nlp / agent_em_100k
# at IT — only the DATASOURCE changes. Cross-model fairness invariant
# (decisions § 15 + § 16).
ARCH_TRAINING_CFGS: dict[str, TrainingConfig] = {
    "topk_sae":   TrainingConfig(n_steps=20_000, train_window_size=1),
    "tsae_paper": TrainingConfig(n_steps=20_000, train_window_size=2),
    "tfa":        TrainingConfig(n_steps=20_000, batch_size=32),
    "txc_base":   TrainingConfig(n_steps=20_000),
    "txc_pro":    TrainingConfig(n_steps=20_000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--archs", nargs="+",
        default=["topk_sae", "tsae_paper", "tfa", "txc_base", "txc_pro"],
        choices=list(ARCH_TRAINING_CFGS.keys()),
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override n_steps for smoke tests.")
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))

    for arch in args.archs:
        cfg = ARCH_TRAINING_CFGS[arch]
        if args.n_steps is not None:
            cfg = cfg.model_copy(update={"n_steps": args.n_steps})
        for seed in args.seeds:
            for k in args.k_feats:
                print(
                    f"[c3_base] cell {arch} seed={seed} k_feat={k} "
                    f"B={cfg.batch_size} T={cfg.train_window_size} "
                    f"n_steps={cfg.n_steps}",
                    flush=True,
                )
                eval_cfg = {
                    "k_feat": k, "S": 32, "smoke": False,
                    "_act_cache_key": act_cache_key,
                    "_datasource_name": DATASOURCE,
                }
                runner.run_cell(
                    component="c3", arch_name=arch, seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=cfg, eval_cfg=eval_cfg,
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn, eval_fn=my_eval_fn,
                )


if __name__ == "__main__":
    main()
```

agent_nlp's `my_train_fn` already passes through
`training_cfg.train_window_size` AND `training_cfg.batch_size` (the
trainer reads `.batch_size` directly via `gen_fn(training_cfg.batch_size)`),
so per-arch overrides flow correctly without any agent_nlp-side change.

Step 4 — smoke ONE TopK cell at `n_steps=200`:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c3_probing_base.run \
  --archs topk_sae --seeds 42 --k-feats 5 --n-steps 200 \
  2>&1 | tail -25
```

Verify the row lands at `arch=topk_sae`, `component=c3`,
`datasource=gemma_2_2b_base_l13_fineweb_24k128` (NOT the IT
datasource), and a fresh `train_key` distinct from agent_nlp's IT
cells.

Step 5 — launch the full 5-arch sweep:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c3_probing_base.run \
  > logs/c3_base_full.log 2>&1 &
echo $! > /tmp/p_c3_base
```

Per-cell ETA varies by arch (~1.2 hr for TopK at T=1 → ~2 hr for
TXC-pro). Total Tier 1 wall ≈ ~22-25 hr serial.

Step 6 — **after C3 lands, build the BASE concat_v1 cache** (small —
3 sequences, variable length, fast):

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -c "
from temp_bench.data.nlp.cache import build_activation_cache
build_activation_cache('gemma_2_2b_base_l13_concat_v1')
"
```

(Note: this datasource has `n_seqs: 3` and `seq_len: variable` — the
existing `build_activation_cache` may need a per-task-length tweak.
If the build fails on the variable-seq-len shape, surface as Open
question; likely needs a small generalisation in the cache builder.)

Step 7 — **write `experiments/c4_qualitative_base/{__init__.py,
run.py}`**. Adapt agent_nlp's `experiments/c4_qualitative/run.py` —
swap the IT datasource for BASE. C4's training cache-hits on the
C3 BASE checkpoints (same train_key per arch); only the qualitative
eval (Anthropic Haiku judge over concat corpora) re-runs.

Step 8 — launch C4 eval after C3 wraps:

```bash
TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
  .venv/bin/python -m experiments.c4_qualitative_base.run \
  > logs/c4_base_full.log 2>&1 &
```

(~1.5 hr; budget ~$0.40 in Haiku judge calls per agent_nlp's IT
estimate.)

### Watch-outs

- **Subject model is BASE**, not IT. Verify before launching anything:
  ```bash
  .venv/bin/python -c "
  from temp_bench.config import load_datasource
  print(load_datasource('gemma_2_2b_base_l13_fineweb_24k128').subject_model)
  "
  # → google/gemma-2-2b   (NOT google/gemma-2-2b-it)
  ```
- **Match T / B / sampling exactly to IT**. The cross-model
  comparison's value comes from controlled-arch consistency. Don't
  silently change a TrainingConfig field; if you do, document it.
- **Don't render `docs/components/c3.md` or `c4.md`** — agent_nlp's
  territory. agent_paper integrates the BASE results at paper-render
  time via an extended `canonical_train_keys` filter (split by
  datasource).
- **HF push every cache** before pod stop (ephemeral pod): act_cache
  ~14 GB, probe_cache ~1 GB. Without push, pod restart wipes
  everything.
- **TFA at B=32 may be slow** — TFA's attention costs ~3-5× more per
  step than vanilla SAE. Plan ~30-60 min train per cell. Verify
  setting `cfg.batch_size=32` propagates correctly via your driver
  (the trainer reads `training_cfg.batch_size` directly; check after
  smoke).
- **TXC-pro is the slowest cell** (~2 hr each, InfoNCE all-pairs). If
  you're running tight on time, drop TXC-pro for Tier 1 and document
  the omission in your Current state. agent_nlp's IT TXC-base vs
  TXC-pro were within noise at C3, so dropping TXC-pro is paper-
  defensible if compute pressures.
- **Don't blow the multi-layer cache** unless you're in Tier 2 mode.
  The Tier 1 single-layer cache (~14 GB) is all you need for the 5
  archs above. MLC is the only arch needing the multi-layer cache.

### Tier 2 (stretch — only if Tier 1 wraps with margin)

If Tier 1 wraps with > 8 hr remaining in the sprint window:

1. Build `gemma_2_2b_base_l11to15_fineweb_24k128` multi-layer cache (~3 hr)
2. Build BASE multi-layer probe_cache (~1.5 hr)
3. Run MLC × 3 seeds × 2 k_feats (~5 hr)

Otherwise skip MLC for BASE — agent_em_100k's IT MLC stands as the
single MLC reference; the BASE replication's headline is the 5
single-layer archs.

### References

- `decisions.md` § 15 (per-arch literature-faithful T values), § 16
  (TFA wasteland-faithful B=32, MLC L=5 multi-layer cache)
- `agents/agent_nlp/briefing.md` (IT C3 setup; same pattern)
- `agents/agent_em_100k/briefing.md` (IT C3 T-SAE + MLC setup)
- `experiments/c3_probing/run.py` (agent_nlp's plumbing — import from)
- `experiments/c4_qualitative/run.py` (agent_nlp's qualitative eval —
  import from + swap datasource)
- `papers/temporal_sae.md` § 3.1 (Bhalla/Ye T-SAE pairs)
- `papers/priors_in_time.md` § B.1 (TFA training canon)
- `papers/are_saes_useful.md` App. B (SAEBench reference)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → both caches + checkpoints auto-pushed; the script
prints a verify-state recipe. Don't let Han stop the pod until the
caches AND all 15 (5 archs × 3 seeds) checkpoints are confirmed on HF.

---

## Historical (rescinded missions; preserved for git provenance)

The following sections describe MISSIONS THAT WERE ABORTED. Read
top-to-bottom only as context — do NOT execute the directives.

- **2026-05-04 PM, original**: 100K convergence-test on C5
  (`txc_base + txc_pro + tsae_paper × 3 seeds at n_steps=100K` on the
  Gemma IT cache). Abandoned 2026-05-05 AM in favor of C5 MW pivot.
- **2026-05-05 AM**: C5 MW pivot (txc_base_mw + txc_pro_mw on C5).
  Abandoned (CPU-bandwidth bottleneck on this pod, agent_filler took
  over on 8× A40). 1 cell landed: `eval_key=963df9c69213f998`
  (txc_base_mw seed=42). Stays in leaderboard as bonus diff data.
- **2026-05-05 PM**: C7 MW pivot (txc_base_mw + txc_pro_mw at C7,
  agent_back's territory). Abandoned (decisions § 14 deprecated; MW
  was a misframe). No cells launched.
- **2026-05-05 PM**: STAND DOWN. Idle, no compute mission.
- **2026-05-05 PM**: C6 TFA mission. Abandoned (this section's ABORT
  block above; too long + judge API cost too high).

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-05T22:24Z. C3 BASE per-token (TopK +
T-SAE + TFA) DONE (18 cells). C4 BASE on TopK + T-SAE DONE
(6 cells). C4 TFA aborted (shape mismatch — see OQ #2).**

Pod is 1× H100 ephemeral. Caches done + on HF:
`act_cache/f01ca87f2e8f3365` (13.2 GB) and
`probe_cache/gemma_2_2b_base_l13_fineweb_24k128` (~20 GB).

**C3 BASE canonical (per-token archs)** — KILLED at
intentional checkpoint after TFA seed=2 k=20 landed; TXC archs
are agent_filler's territory now (§ LOAD SPLIT 2026-05-05 PM).
18/30 cells delivered:
- TopK ×3 seeds: cross-seed mean k=5: 0.826, k=20: 0.880
- T-SAE ×3 seeds: cross-seed mean k=5: 0.853, k=20: 0.895
- TFA ×3 seeds: cross-seed mean k=5: 0.709, k=20: 0.776
  (TFA trails TopK + T-SAE on BASE — paper-relevant finding)

**C4 BASE on per-token archs (PID 23075, exited)** — TopK +
T-SAE landed (6 cells); TFA cell crashed at first attempt due
to TFA → `(N, T=5, d_sae)` shape mismatch in
`qualitative.encode_concat_corpus`. Cross-seed C4 SEMANTIC:
- TopK: 88.3/256 (~34.5%)
- T-SAE: 75.3/256 (~29.4%)
- TopK > T-SAE on qualitative on BASE (opposite of C3 ordering;
  worth noting in paper).

**§ LOAD SPLIT (effective)**: agent_filler running TXC-base
(T=5/10/20) + TXC-pro × 3 seeds = 12 trainings on 8× A40s.
ETA ~5-6 hr; my C4 BASE evals on those checkpoints come last.

**IMPORTANT BUG FIX (local override)**: agent_nlp's
`c3_probing.run.my_eval_fn` reads `_arch_hparams` per docstring
but calls `load_arch(arch_name)` (default YAML hparams). My
BASE drivers carry a LOCAL `my_eval_fn` that applies the merged
hparams correctly. OQ #1.

## What I just did (agent owns — overwrite)

1. Built BASE act_cache + probe_cache and pushed to HF.
2. Drafted + smoke-tested `experiments/c3_probing_base/run.py`.
3. Drafted `experiments/c4_qualitative_base/run.py`.
4. Committed driver + smoke (commit `71c26c1f`) and pushed.
5. Launched canonical 5-arch C3 BASE sweep (PID 16279).
6. Pulled new directives (rebased onto `d54cead3` agent_paper
   T-sweep mission) — append-only conflicts resolved.
7. Extended BOTH BASE drivers for txc_base T=10 + T=20:
   `ARCH_TRAINING_CFGS` is now `dict[str, list[TrainingConfig]]`,
   each cell's cfg gets `arch_hparams_override`, main loop
   iterates `(arch, cfg)` pairs. `--cfg-tags` CLI restricts to
   specific tag (e.g. `T10`).
8. Wrote LOCAL `my_eval_fn` in both BASE drivers to apply
   merged `_arch_hparams` (agent_nlp's eval_fn is buggy; my
   local override works around it for T-sweep cells).

## Next action (agent owns — overwrite)

1. **Wait for agent_filler's TXC checkpoints**. Watch the
   manifest for new `txc_base` / `txc_pro` rows on the BASE
   datasource (gemma_2_2b_base_l13_fineweb_24k128). ETA
   ~5-6 hr (per agent_filler's brief).
2. **Run C4 BASE on TXC archs** once filler's trains land:
   ```
   TQDM_DISABLE=1 AGENT_NAME=agent_steer_100k \
     .venv/bin/python -m experiments.c4_qualitative_base.run \
       --archs txc_base txc_pro \
       > logs/c4_base_txc.log 2>&1 &
   ```
   This will cache-hit on filler's checkpoints (3 seeds × 4
   cfgs = 12 cells, ~2 hr).
3. **Wrap up**: `bash scripts/wrap_up_session.sh` to confirm
   all checkpoints + caches are on HF.

## Don't repeat (agent owns — overwrite)

### Mission scope
- **DO NOT run TXC-base or TXC-pro on BASE** — agent_filler
  owns these per § LOAD SPLIT 2026-05-05 PM.
- **DO NOT run C4 on TFA** — shape mismatch with
  `qualitative.encode_concat_corpus`. agent_nlp's IT C4 also
  excludes TFA.
- **DO NOT run the § 17 T-sweep yourself** (T=10/T=20) —
  rescinded for agent_steer_100k; agent_filler handles it.
  Driver still has the cfgs registered (with `--cfg-tags`
  filtering) but don't launch them on this pod.
- **Don't run anything other than BASE C3 + C4** (and optionally MLC
  for stretch). Don't pursue C5, C6, C7, MW, or 100K — all rescinded.
- **Don't deviate from agent_nlp + agent_em_100k's per-arch
  TrainingConfigs**. Same T, same B, same n_steps. The cross-model
  comparison's value depends on this.

### Territory rules
- **Don't edit agent_nlp's `experiments/c3_probing/`** — import only.
- **Don't edit agent_em_100k's `experiments/c3_probing_*_baseline/`**
  drivers — import only.
- **Don't edit `docs/components/c3.md` or `c4.md`** — agent_nlp's.
- **Don't edit `configs/datasources.yaml` or `locked_archs.yaml`** —
  agent_paper's. The BASE datasources you need are already present.

### Driver internals
- **Don't bypass `runner.run_cell`** — single canonical pathway.
- **Don't allocate `train_key` / `eval_key` manually**.
- **Don't push checkpoints to HF manually** — `cache.save_checkpoint`
  auto-pushes on ephemeral pods.

### Pod-specific gotchas
- **HF push BOTH the act_cache AND the probe_cache** before any pod
  restart. The ephemeral pod wipes /workspace.
- **Watch the CPU-bandwidth bottleneck** flagged in earlier briefing
  commit `e7b229fd`. Training rate may be ~1.5× slower than IT side
  on agent_nlp's pod. Plan accordingly.

## Open questions for Han (agent owns — overwrite)

1. **Bug in agent_nlp's `c3_probing.run.my_eval_fn`**: its
   docstring claims to read `_arch_hparams` from eval_cfg but
   the implementation calls `load_arch(arch_name)` (default
   YAML hparams). For canonical cells (T=5 default) this works
   because merged == default; for T=10/T=20 it would silently
   build a T=5 model and `load_state_dict` would fail. I added a
   LOCAL `my_eval_fn` in my BASE drivers that applies the merged
   hparams from `eval_cfg["_arch_hparams"]`. Suggest agent_nlp
   upstream the same fix to `experiments/c3_probing/run.py`
   (and `experiments/c4_qualitative/run.py` likely needs it too).
2. **TFA incompatible with C4 qualitative eval (paper-blocking
   if we need TFA for C4)**: `qualitative.encode_concat_corpus`
   expects SAE features of shape `(N, d_sae)` but TFA returns
   `(N, T=5, d_sae)` (per-window output). My BASE C4 sweep
   crashed at the first TFA cell with `ValueError: could not
   broadcast input array from shape (256,5,18432) into shape
   (256,18432)`. Note agent_nlp's IT C4 `DEFAULT_ARCHS` also
   excludes TFA, suggesting this is a known incompatibility.
   For now I dropped TFA from the BASE C4 sweep (TopK + T-SAE
   landed cleanly: 6 cells). If TFA is needed for the BASE C4
   table, `qualitative.encode_concat_corpus` needs a TFA-
   specific path (e.g. take the last-window features, or
   average across T).
3. **agent_filler's BASE C3 `txc_base` checkpoints are NOT on
   HuggingFace (paper-blocking for k_feats expansion on those
   archs)**: 9 manifest rows exist (txc_base × 3 seeds × {T=5,
   T=10, T=20}, all with `agent="unknown"` and `hf_url=None`),
   plus 6 leaderboard rows for txc_base k=5/k=20 evals — but
   direct `hf_hub_download` 404s on every safetensors file. My
   driver tried to cache-hit on these and instead started
   training fresh (caught + killed at 600 steps). For now I'm
   running stage 2 on `txc_pro` only (those ARE on HF and
   cache-hit cleanly). Suggest agent_filler push their
   txc_base checkpoints to `han1823123123/temp-bench-models`
   so other agents can do eval-only k_feats expansions on
   them. Until then BASE txc_base cells are stuck at k∈{5,20}.
