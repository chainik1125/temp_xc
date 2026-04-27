---
author: Han
date: 2026-04-27
tags:
  - guide
---

## Collaborator onboarding: barebones TXC T=5 (seed=42)

You're picking up `txc_bare_antidead_t5` (row 9 of Phase 7's
canonical 49-arch set; the "Track 2 reference"). This is the
**vanilla TemporalCrosscoder + tsae-paper anti-dead stack only** —
no matryoshka head, no temporal contrastive loss, no multi-distance
shifts. The simplest TXC variant that still has the auxiliary-K
dead-feature recovery from the T-SAE paper.

### TL;DR (commands)

```bash
# 1. Clone repo + pick up Phase 7's branch as your starting point
git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc
git fetch origin
git checkout -b <your-branch> origin/han-phase7-unification

# 2. Set up env (uv-managed venv)
curl -LsSf https://astral.sh/uv/install.sh | sh
export HF_HOME=/workspace/hf_cache
export UV_LINK_MODE=copy
uv sync

# 3. Pull the trained seed=42 ckpt + per-run training log from HF
huggingface-cli download han1823123123/txcdr-base \
    ckpts/txc_bare_antidead_t5__seed42.pt \
    training_logs/txc_bare_antidead_t5__seed42.json \
    --local-dir experiments/phase7_unification/results/

# 4. Sanity-check load
.venv/bin/python -c "
import torch, json
sd = torch.load('experiments/phase7_unification/results/ckpts/txc_bare_antidead_t5__seed42.pt', weights_only=True, map_location='cpu')
print('state_dict keys:', sorted(sd.keys()))
log = json.load(open('experiments/phase7_unification/results/training_logs/txc_bare_antidead_t5__seed42.json'))
print('elapsed_s:', log['elapsed_s'], 'final_step:', log['final_step'], 'converged:', log['converged'])
"
```

### What's in the ckpt

Saved as `torch.save(state_dict_fp16)` after fp32 → fp16 cast. Cast
back to fp32 on load (Phase 7's `_load_phase7_model` in
`experiments/phase7_unification/run_probing_phase7.py` does this
automatically). State dict has:

- `W_enc`: shape (T=5, d_in=2304, d_sae=18432) — encoder
- `b_enc`: shape (d_sae,)
- `W_dec`: shape (d_sae, T=5, d_in)
- `b_dec`: shape (T, d_in)
- `num_tokens_since_fired`, `last_auxk_loss`, `last_dead_count`,
  `b_dec_initialized`: anti-dead training state buffers (safe to
  ignore at inference time)

### Architecture details

| field | value |
|---|---|
| arch class | `src.architectures.txc_bare_antidead.TXCBareAntidead` |
| T (window length) | 5 |
| k (window-level TopK) | 500 (= k_pos × T = 100 × 5) |
| d_in | 2304 |
| d_sae | 18432 |
| anti-dead aux_k | 512 |
| anti-dead dead_threshold_tokens | 10_000_000 |
| anti-dead auxk_alpha | 1/32 |
| trained on | google/gemma-2-2b BASE (NOT -it) |
| anchor layer | **L12 (0-indexed)** = `model.model.layers[12]` |

To instantiate:

```python
from src.architectures.txc_bare_antidead import TXCBareAntidead
import torch

model = TXCBareAntidead(d_in=2304, d_sae=18432, T=5, k=500)
sd = torch.load('experiments/phase7_unification/results/ckpts/txc_bare_antidead_t5__seed42.pt',
                weights_only=True, map_location='cuda')
sd = {k: (v.float() if v.dtype == torch.float16 else v) for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to('cuda').eval()
```

### Subject model + activation tap

Use the **base** Gemma-2-2b (NOT the -it model). The L12 anchor is
deliberate — matches T-SAE (Ye et al. 2025 §4.1) and TFA (Lubana et
al. 2025 App. B.1), both of which train on Gemma-2-2b base at
0-indexed L12 (≈50% model depth).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained('google/gemma-2-2b')
gemma = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2-2b', torch_dtype=torch.bfloat16, device_map='cuda',
)
ANCHOR_LAYER = 12

# To capture L12 residual stream during forward pass:
captured = {}
def hook(module, inp, out):
    captured['L12'] = (out[0] if isinstance(out, tuple) else out).detach()
handle = gemma.model.layers[ANCHOR_LAYER].register_forward_hook(hook)
```

Then `captured['L12']` shape `(B, seq_len, 2304)` — feed slices of
that into `model.encode(window)` where `window` has shape
`(B, T=5, 2304)`.

### Activation cache (if you want to rerun training from scratch)

If you want to **re-train** rather than just analyse the existing
ckpt, you'll need the cached activations:

- Path: `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy`
- Shape: `(24_000, 128, 2304)` fp32 (~28 GB on disk)
- Source: 24K FineWeb sequences forwarded through Gemma-2-2b base
- Build script: `experiments/phase7_unification/build_act_cache_phase7.py`
  (single-layer, resumable; ~30 min on H200)
  OR canonical builder: `python -m src.data.nlp.cache_activations
  --model gemma-2-2b --dataset fineweb --mode forward
  --num-sequences 24000 --seq-length 128
  --layer_indices 12 --components resid`
- Not on HF (Agent A's pod local). Easiest to rebuild on your pod.

### Training script (if you want to re-train)

```bash
HF_HOME=/workspace/hf_cache TQDM_DISABLE=1 PHASE7_REPO=/workspace/temp_xc \
  PYTHONUNBUFFERED=1 \
  .venv/bin/python -u -m experiments.phase7_unification.train_phase7 \
    --arch txc_bare_antidead_t5 --seed 42 --max_steps 8000
```

Per-run training log written to
`experiments/phase7_unification/results/training_logs/txc_bare_antidead_t5__seed42.json`.
Original training: 19.2 min on H200 at batch=4096 (per
`docs/han/research_logs/phase7_unification/2026-04-27-training-times.md`).

### Phase 7 context (the bigger picture)

Read `docs/han/research_logs/phase7_unification/brief.md` for the
overall Phase 7 design + 3-agent execution model.

The key conventions you should know:

- **Subject model**: gemma-2-2b BASE.
- **Anchor layer**: 0-indexed L12.
- **Sparsity convention**: k_win = 500 fixed across all archs
  (k_pos × T for window archs; k for per-token).
- **Aggregation S** for sparse probing: **S=32** (corrected after
  bug + speed pivot — see
  `2026-04-27-S-decision-revised.md` and
  `2026-04-27-URGENT-probing-cache-fix.md`).
- **(T, S) validity rule**: kept windows = `S − T + 1`; cell valid
  iff `S ≥ T`. Earlier `S − 2T + 2` rule was wrong; ignore old docs
  that mention it.

### Branch hygiene + push protocol

Per Phase 7's branch convention (per `agent_c_brief.md`):

- Work on **your own branch** (e.g., `<your-name>-txc-bare-t5`)
  branched off `han-phase7-unification`.
- Pull `origin/han-phase7-unification` periodically to integrate
  Agent A's evolving infra.
- DO NOT push directly to `han-phase7-unification` (Agent A
  occasionally force-pushes amends; would clobber your commits).
- DO NOT modify shared framework files
  (`experiments/phase7_unification/_paths.py`,
  `train_phase7.py`, `run_probing_phase7.py`, `canonical_archs.json`).
  If you need a new field added to meta or a new aggregation cell,
  open a PR comment.
- Add new files under your own scratch sub-dir, e.g.,
  `experiments/phase7_unification/<your-name>_txc_bare_t5_analysis/`.
- Merge back via PR `<your-branch>` → `han-phase7-unification` when
  ready. Mention which Phase 7 commits you depend on.

### Useful pointers

| topic | doc / file |
|---|---|
| Phase 7 brief (motivation + 3-agent split) | `docs/han/research_logs/phase7_unification/brief.md` |
| Pre-registered methodology | `docs/han/research_logs/phase7_unification/plan.md` |
| Canonical 49 arch list | `experiments/phase7_unification/canonical_archs.json` |
| Phase 5 → Phase 7 carryover (Track 2 lineage) | `docs/han/research_logs/phase5_downstream_utility/` |
| Anti-dead loss reference (paper) | `papers/temporal_sae.md` (T-SAE paper §3) |
| Phase 7 training driver | `experiments/phase7_unification/train_phase7.py` (`train_txc_bare_antidead`) |
| Phase 7 probing driver | `experiments/phase7_unification/run_probing_phase7.py` (uses `_load_phase7_model` to load any arch incl. yours) |
| HF model repo (all Phase 7 ckpts) | `han1823123123/txcdr-base` |
| HF dataset repo (caches if you need them) | `han1823123123/txcdr-base-data` (some files; activation cache mostly local-only) |

### Ask-Han questions

- "Should the `txc_bare_antidead_t5` analysis stay in Phase 7
  scope or branch into its own phase?" — depends on what you find.
- "Can I push to HF txcdr-base?" — coordinate with Han first;
  uploads should be tagged with seed/source so they don't collide
  with Agent A/B/C ckpts.
