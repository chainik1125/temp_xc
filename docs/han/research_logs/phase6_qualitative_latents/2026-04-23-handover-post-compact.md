---
author: Han
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Handover: Phase 6.1 follow-ups for the next agent

**Audience**: a fresh agent picking up Phase 6.1 cold after a context
compact + disk bump + pod restart.

**Start here**: read this doc, then [[2026-04-23-agentic-log]]
(per-cycle hypothesis → result → takeaway for 5 cycles),
then [[summary]] §9.5 (the new Phase 6.1 update section).

### Current state (end of 2026-04-23 session)

Phase 6.1 ran 5 agentic cycles on top of `agentic_txc_02` (the Phase
5.7 winner that lost Phase 6 qualitative 2 / 8). The goal was to
push TXC autointerp on concat_A + concat_B to ≥ 5 / 8 semantic
labels while preserving Phase 5 sparse-probing AUC within 0.01.

**Outcome: 7 / 8 semantic, alive 0.80, new champion.**

| arch | sparsity | anti-dead | alive | /8 | role |
|---|---|---|---|---|---|
| `agentic_txc_02` | TopK | — | 0.37 | 2 | Phase 5 baseline |
| `agentic_txc_09_auxk` (A) | TopK | AuxK only | 0.37 | 3 | null effect |
| `agentic_txc_10_bare` (Track 2) | TopK | full (AuxK + unit-norm + grad-⊥ + geom-med) | 0.62 | 6 | ties `tsae_paper`! |
| **`agentic_txc_02_batchtopk` (F)** | BatchTopK | — | **0.80** | **7** 🏆 | **champion** |
| `agentic_txc_11_stack` (H) | BatchTopK | AuxK | 0.79 | 5 | stacking regresses |

**Three load-bearing findings** (full reasoning in the agentic-log):

1. **BatchTopK sparsity is the single biggest lever** (+5 labels vs
   baseline). Variable per-sample sparsity lets rare concept
   features fire on the contexts where they matter without
   displacing an incumbent top-500 winner.
2. **The anti-dead stack (unit-norm decoder + decoder-parallel grad
   removal + geom-median `b_dec` init + AuxK) is an alternate path**
   that almost matches BatchTopK at TopK sparsity. Matryoshka +
   multi-scale contrastive is NOT required for qualitative — only
   for Phase 5 probing AUC.
3. **The two mechanism classes don't stack additively** — Cycle H
   (BatchTopK + AuxK) regressed 7 → 5 /8. AuxK's residual gradient
   promoted structural / format features into top-by-variance.

### Everything is committed and pushed

- Branch: `han-phase6` → `origin/han-phase6` (9 Phase 6.1 commits,
  latest `8764126` is the summary.md update).
- **Uncommitted**: 6 UMAP PNGs under
  `experiments/phase6_qualitative_latents/results/umap/concat_C_v2__umap_high__tsae_ours__*.png`
  — these are unrelated to Phase 6.1 (pre-existing UMAP
  regeneration output, sat uncommitted through the whole session).
  Either commit them as a separate piece or discard — do NOT roll
  into a Phase 6.1 commit.

### Environment — verified on 2026-04-23 18:51 UTC

Assuming `/workspace/` persists across the disk bump + restart:

| Token | File | Status at handover |
|---|---|---|
| GitHub PAT | `/workspace/.github-token` | ✅ 40 bytes, push works |
| HF token | `/workspace/.hf-token` (+ `/workspace/hf_cache/token` mirror) | ✅ 37 bytes, whoami = `han1823123123` |
| Anthropic key | `/workspace/.anthropic-key` | ✅ **restored** during handover (user had deleted it; re-extracted from `/workspace/temp_xc/.env`) |

`~/.bashrc` at `/home/appuser/.bashrc` holds the exports:

```bash
export HF_HOME=/workspace/hf_cache
export UV_LINK_MODE=copy
export HF_TOKEN=$(cat /workspace/.hf-token)
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
```

`~/.bashrc` may NOT persist across pod stop / start on some RunPod
configs — re-run the export block from `RUNPOD_INSTRUCTIONS.md` if
`HF_HOME` or `ANTHROPIC_API_KEY` are empty after restart. Git
credential helper is wired at the repo level (`git config --local`),
which persists with the repo.

**First-15-minute checklist on restart:**

```bash
cd /workspace/temp_xc
source ~/.bashrc 2>/dev/null                  # or re-export manually
echo "$HF_HOME $UV_LINK_MODE"                 # both should be non-empty
echo "${ANTHROPIC_API_KEY:0:12}..."           # should print sk-ant-api...
ls -la /workspace/.github-token /workspace/.hf-token /workspace/.anthropic-key
uv sync && uv sync                            # second pass should only audit
git status -sb                                # expect "han-phase6...origin/han-phase6"
git log --oneline -3                          # latest should be 8764126 (summary update)
TQDM_DISABLE=1 uv run python -c "
import torch
from src.architectures._tfa_module import TemporalSAE
print(torch.cuda.get_device_name(0), torch.cuda.is_available())
"
```

If any of the above fails, fix it before any experiment work.

### GPU + disk

- Last observed: A40, 48 GB VRAM. User is bumping the disk to
  400 GB — so plenty of headroom for ckpts + probe_cache.
- 5 Phase 6.1 ckpts on disk under
  `experiments/phase5_downstream_utility/results/ckpts/`:
  - `agentic_txc_09_auxk__seed42.pt` (1.36 GB, Cycle A)
  - `agentic_txc_10_bare__seed42.pt` (850 MB, Track 2)
  - `agentic_txc_02_batchtopk__seed42.pt` (1.36 GB, Cycle F — also
    already on HF)
  - `agentic_txc_11_stack__seed42.pt` (1.36 GB, Cycle H)
  - (`agentic_txc_02__seed42.pt`, `agentic_mlc_08__seed42.pt`,
    `tsae_paper__seed42.pt`, `tsae_ours__seed42.pt`, `tfa_big` —
    from earlier work, reference checkpoints)

### What to do next (ordered by expected impact × cost)

**#1 — Seed variance on Cycle F (HIGH priority).** The 7 / 8 result
is single-seed. Phase 5.7 found that `agentic_txc_02`'s headline
gain halved across seeds (0.035 → 0.022 mean). Confirm 7 / 8 is
real by training seeds {1, 2} and evaluating:

```bash
TQDM_DISABLE=1 PYTHONPATH=. nohup .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[1, 2], max_steps=25000, archs=['agentic_txc_02_batchtopk'])
" > logs/phase6.1_cycleF_seedvar.log 2>&1 &

# After both finish (~80 min), eval each with the harness:
bash experiments/phase6_qualitative_latents/run_cycle_eval.sh agentic_txc_02_batchtopk  # seed=42 already done
SEED=1 bash experiments/phase6_qualitative_latents/run_cycle_eval.sh agentic_txc_02_batchtopk
SEED=2 bash experiments/phase6_qualitative_latents/run_cycle_eval.sh agentic_txc_02_batchtopk
```

**Caveat**: `run_cycle_eval.sh` hard-codes `seed=42` unless `SEED=` is
exported. Also, `run_probing.py` will fail without `probe_cache`
(see #5). You can skip that step in the harness by commenting out
STEP 4a/4b.

Expected: 3-seed variance on 7 / 8 likely falls in 5 / 8 – 7 / 8.
A mean of ≥ 6 / 8 is sufficient for the paper claim (beats
`tsae_paper` in mean at matched spec).

**#2 — Missing 2×2 cell: bare + BatchTopK + full anti-dead stack.**
We did:

|  | TopK | BatchTopK |
|---|---|---|
| matryoshka + contrastive | (baseline) 2 / 8 | **Cycle F 7 / 8** |
| bare (no matryoshka / contrastive) | **Track 2** 6 / 8 | **⬜ not tested** |

The empty cell is the simplest architecture with both anti-dead
axes engaged. If it beats 7 / 8, the paper story simplifies to
"BatchTopK + anti-dead stack, nothing fancy required". If it ties
or loses, BatchTopK + anti-dead doesn't stack (similar interaction
to Cycle H).

Implementation: subclass `TXCBareAntidead`
([`src/architectures/txc_bare_antidead.py`](../../../src/architectures/txc_bare_antidead.py))
and replace its TopK-scatter with a `BatchTopK(k)` module from
[`src/architectures/_batchtopk.py`](../../../src/architectures/_batchtopk.py).
~20-line addition. Arch name suggestion: `agentic_txc_12_bare_batchtopk`.

Dispatcher: add branch in `train_primary_archs.py` modeled on the
existing `agentic_txc_10_bare` branch (line ~1284 before edit).
Wire `encode_archs.py`, `run_probing.py`, `arch_health.py` the same
way Cycle A / Track 2 / Cycle F / Cycle H were wired.

One training run ~30 min + eval ~5 min.

**#3 — Longer training past the plateau-stop artefact.** All Phase
6.1 cycles converged at step 4000 – 5600 / 25000 due to the 2 %
plateau threshold. Phase 5.7 `agentic_txc_02` went to ~16 205 steps
before plateau. The anti-dead mechanisms may shape decoder
directions more effectively over the full window.

Try bumping `min_steps` from 3000 → 10000 for Cycle F:

```python
# In train_primary_archs.py TrainCfg:
@dataclass
class TrainCfg:
    ...
    min_steps: int = 10_000        # was 3_000
```

Or pass a custom cfg at call time. Re-train Cycle F, eval. Cheap
(~40 min). Expected: modest improvement, probably not enough alone
to push to 8 / 8.

**#4 — Corpus-aware qualitative metric.** Top-8-by-variance gets
biased toward tokens that activate intermittently but strongly —
punctuation, MMLU answer-option formatting, hyphens, ligatures.
These appear in Cycle F's remaining 1 non-semantic feature and in
most of Cycle H's non-semantic slots.

Alternative ranking: **passage-discriminative variance**, i.e.
rank features by `Var_over_passages(mean_activation_in_passage)`
instead of `Var_over_tokens(activation)`. This suppresses features
that fire on isolated tokens regardless of passage and lifts
features that respond to passage-level semantics.

Implementation: modify `_pick_top_features` in
[`experiments/phase6_qualitative_latents/run_autointerp.py`](../../../experiments/phase6_qualitative_latents/run_autointerp.py)
line 70. Compute per-feature (n_passages,)-shaped vector of mean
activations per passage (passages are recoverable from concat_A
and concat_B provenance JSONs in `z_cache/*/provenance.json`), then
rank by that vector's variance.

**No retraining needed** — this works on the existing Cycle F z
cache. If the ranking reveals 8 / 8 on current ckpts, that's the
cheapest win.

**#5 — Sparse-probing regression on Cycle F (PAPER-CRITICAL).**
The paper claim is "sparsity is a one-knob trade-off". That needs a
numeric trade-off curve, not just a qualitative gain.

Required: download `probe_cache` from
`han1823123123/txcdr-data` on HF.

```bash
HF_HOME=/workspace/hf_cache HF_TOKEN=$(cat /workspace/.hf-token) \
  uv run huggingface-cli download --repo-type dataset han1823123123/txcdr-data \
  --include "experiments/phase5_downstream_utility/results/probe_cache/**" \
  --local-dir .
```

Size: **70 GB**, 144 files, probably 30–60 min on RunPod network.
After the disk bump to 400 GB this is fine.

Then run probing:

```bash
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation last_position --run-ids agentic_txc_02_batchtopk__seed42 \
  --skip-baselines
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase5_downstream_utility/probing/run_probing.py \
  --aggregation mean_pool --run-ids agentic_txc_02_batchtopk__seed42 \
  --skip-baselines
```

Compare Δ AUC vs `agentic_txc_02` (0.775 last_pos / 0.799 mean_pool).
Phase 5.7 experiment (ii) found a regression; quantify the exact
magnitude for the paper.

**#6 — HF upload of the 4 new Phase 6.1 ckpts.** For cross-pod
reproduction. Target repo: `han1823123123/txcdr`.

```python
from huggingface_hub import upload_file
for arch in ('agentic_txc_09_auxk', 'agentic_txc_10_bare', 'agentic_txc_11_stack'):
    upload_file(
        path_or_fileobj=f'experiments/phase5_downstream_utility/results/ckpts/{arch}__seed42.pt',
        path_in_repo=f'ckpts/{arch}__seed42.pt',
        repo_id='han1823123123/txcdr',
    )
# agentic_txc_02_batchtopk already on HF
```

### Dead ends — do NOT re-run these

- **More AuxK on BatchTopK.** Cycle H showed BatchTopK + AuxK
  regressed 7 → 5 /8. Adding AuxK gradient on top of BatchTopK
  shifts decoder directions toward residual-reconstruction
  specificity, which surfaces structural / format features at
  top-by-variance. Any "stack more anti-dead on BatchTopK" idea
  has this as prior art.
- **Orthogonality penalty on scale-1 prefix** (Phase 5.7 Cycle 01
  regressed −0.018 vs reference). Unlikely to help qualitative
  either — it perturbs decoder columns away from the probing-useful
  local optimum.
- **Hard negatives in contrastive** (Phase 5.7 Cycle 06 no benefit
  on top of multi-scale at B=1024). The 1023 in-batch negatives are
  already sufficient.
- **Feature-diversity Gram-matrix penalty** (briefing flagged as
  small effect; Cycle 01's orth penalty is a close analogue that
  regressed).
- **Cosine consistency instead of InfoNCE** (Phase 5.7 Cycle 07
  LOST by ~0.018; pull-only weaker than push-together + push-apart).

### File map — where things live

- **Code**:
  - `src/architectures/matryoshka_txcdr_contrastive_multiscale_auxk.py` — Cycle A
  - `src/architectures/txc_bare_antidead.py` — Track 2
  - `src/architectures/matryoshka_txcdr_contrastive_multiscale_batchtopk_auxk.py` — Cycle H
  - `src/architectures/_batchtopk.py` + `_batchtopk_variants.py` — BatchTopK infra (cherry-picked from `origin/han`)
- **Dispatcher entries**:
  `experiments/phase5_downstream_utility/train_primary_archs.py`
  grep for `agentic_txc_09_auxk`, `agentic_txc_10_bare`,
  `agentic_txc_11_stack`, `agentic_txc_02_batchtopk`.
- **Probing dispatch**:
  `experiments/phase5_downstream_utility/probing/run_probing.py`
  same four arch names.
- **Encoding dispatch**:
  `experiments/phase6_qualitative_latents/encode_archs.py`
  same four arch names.
- **Eval harness**: `experiments/phase6_qualitative_latents/run_cycle_eval.sh`
  runs arch_health → encode → autointerp → probe (last/mean).
- **Results**:
  `experiments/phase6_qualitative_latents/results/arch_health.json`,
  `results/autointerp/<arch>__labels.json`,
  `results/autointerp/summary.md`.
- **Docs**:
  - [[brief]] — original Phase 6 brief
  - [[plan]] — pre-registered methodology
  - [[summary]] — end-of-phase writeup, §9.5 is the Phase 6.1 update
  - [[2026-04-22-encoding-protocol]] — per-arch encoding conventions
  - [[2026-04-23-handover-txc-qualitative]] — original 6.1 briefing
  - [[2026-04-23-agentic-log]] — 5-cycle detailed log
  - this doc — post-compact handover

### Working conventions (from user, persistent)

- **Push after each milestone**, not batched. Each cycle's
  commit + push is one milestone; don't accumulate.
- **Explore minimal-intervention alternatives** alongside the
  briefing's cycles — that's how Track 2 got discovered.
- **Disable tqdm**: `TQDM_DISABLE=1` on every Python invocation.
- **Use `python -u`** (unbuffered) when running long background
  training jobs, otherwise stdout stays empty until the process
  exits and the Monitor tool can't see progress.

### One-paragraph restart

On restart, verify tokens + venv + git state with the checklist
above. If everything is green, start with **follow-up #1 (seed
variance on Cycle F)** — it gates the paper claim and runs
autonomously for ~80 minutes. In parallel, launch **#5 (probe_cache
download)** which is CPU/network-only. By the time both finish,
you'll have the full paper-ready result: mean 3-seed Cycle F
qualitative + probing regression curve. **#2 and #4 are
orthogonal ideas for pushing past 7 / 8** if time permits.
