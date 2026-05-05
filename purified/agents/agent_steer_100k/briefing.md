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

### ⚠️ ABORT 2026-05-05 PM — C6 TFA mission RESCINDED

**Han 2026-05-05 PM**: C6 TFA was too long (~11 hr serial) and the
Sonnet judge API cost was projected too high. **ABORT before any cell
launches.**

```bash
# Kill any C6 TFA processes (likely none if you hadn't launched yet)
pkill -KILL -f "experiments.c6_em_tfa_baseline" || true
nvidia-smi --query-gpu=memory.used --format=csv  # expect <500 MB
```

If you didn't get a chance to launch, no cleanup is needed. The MW
artifacts (1 C5 MW cell at `eval_key=963df9c69213f998`) stay in the
leaderboard as bonus diff data — DO NOT delete.

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

**Last verified: 2026-05-05T18:00Z. NEW MISSION — BASE C3 + C4
replication. Pod is 1× H100 ephemeral, idle.** Active checkpoints + 1
C5 MW cell + Llama BASE caches all on HF; safe to start fresh.

(Overwrite this section with your own state when you start.)

## What I just did (agent owns — overwrite)

(Overwrite when you start — newest first.)

## Next action (agent owns — overwrite)

1. `cd /workspace/temp_xc/purified` (or your local equivalent)
2. `source scripts/set_agent_env.sh agent_steer_100k`
3. `bash scripts/agent_smoke_test.sh` (CRITICAL preflight failures
   are fatal)
4. `git pull --rebase origin final`
5. Verify the BASE datasources resolve (Step 0 above).
6. Kick off Phase 0: build the BASE single-layer act cache (~3 hr).
   The build runs in the background; monitor + HF-push when done.
7. While the cache builds, draft `experiments/c3_probing_base/run.py`
   per the Step 3 sketch.
8. After cache builds, build probe_cache (Step 2).
9. Smoke + launch the full 5-arch sweep (Step 4 + 5).
10. After C3 wraps, run C4 (Step 6-8).
11. Optionally proceed to Tier 2 (MLC + multi-layer cache) if margin
    permits.

## Don't repeat (agent owns — overwrite)

### Mission scope
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

(None at briefing-rewrite time. Surface anything that comes up
during the BASE cache build or smoke test.)
