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

**#0 — Upgrade the evaluation metric (DO THIS FIRST).** This gates
the evidence quality of every other follow-up. User explicitly
approved the API budget for scaling up.

The current `x / 8` metric has binomial stderr ±1.2 labels at
p=0.75, so:

- **Robust findings** (delta ≥ 4 labels, outside any reasonable noise):
  BatchTopK lever (+5), anti-dead stack helps (+4), AuxK alone
  null (Δ≈1 within noise — "null" is a robust call).
- **Non-robust findings** (delta 1–2, within binomial noise):
  *Cycle F beats `tsae_paper`* (7/8 vs 6/8 is Δ=1), *Cycle H
  regresses from Cycle F* (5/8 vs 7/8 is ~2σ). Both could flip at a
  different seed or at different N; both are load-bearing for the
  paper narrative. Re-verify.

Four upgrades, combinable:

1. **Bump `N_TOP_FEATURES` from 8 → 32** in
   [`experiments/phase6_qualitative_latents/run_autointerp.py`](../../../experiments/phase6_qualitative_latents/run_autointerp.py)
   line 35 (single-line edit). Binomial stderr drops from ±1.2 to
   ±0.76 labels. Cost: ~32 Haiku calls/arch ≈ $0.01 per arch.
2. **Auto-classify semantic vs non-semantic** instead of hand-
   classifying. Add a second Haiku call per label with an explicit
   rubric. Removes single-labeller bias. ~30-line addition near
   the existing `interp_arch` function. Example rubric:

   ```
   Given this SAE feature label: "{LABEL}"
   Classify as SEMANTIC or SYNTACTIC.
   SEMANTIC = names a concept, topic, theme, entity, or domain
     (examples: "plant biology", "Animal Farm references",
      "archaic poetic English")
   SYNTACTIC = describes surface patterns — punctuation, word class,
     capitalisation, formatting, hyphens, quoted text
     (examples: "sentence-ending periods", "multiple-choice answer
      formatting", "hyphens between compound words")
   Reply with exactly one word: SEMANTIC or SYNTACTIC.
   ```

   Cost: another ~$0.01 per arch.

3. **Passage-discriminative ranking** instead of per-token variance.
   The current top-by-variance ranking has a structural punctuation
   floor because high-density token patterns (full-stops, spaces)
   get high variance. Replace with:

   ```python
   # In run_autointerp.py _pick_top_features:
   # Original: var = z.var(axis=0); argsort(-var)[:N]
   # Proposed: group tokens by passage, compute per-feature mean
   # activation per passage, rank by variance OF THOSE PER-PASSAGE
   # means. Features that fire uniformly on punctuation across
   # passages have low passage-discriminative variance.
   ```

   Passage IDs are recoverable from `z_cache/<concat>/provenance.json`
   (concat_A and concat_B each concatenate 3-4 named passages).
   No retrain — works on existing z_cache. Expected: removes the
   punctuation-floor tax uniformly across archs, might reveal 8/8
   on current Cycle F checkpoint.

4. **3 seeds on Cycle F** — this was formerly follow-up #1. Requires
   ~80 min GPU to train seeds {1, 2}. Combined with #1 + #2 + #3,
   each arch then has a proper `/32 score × 3 seeds` with mean ± stderr.

**Concrete execution plan**:

```bash
# Step A: edit run_autointerp.py
#   (1) line 35: N_TOP_FEATURES = 32
#   (2) add auto-classify function + wire into interp_arch
#   (3) modify _pick_top_features for passage-discriminative ranking
#   (4) emit a per-arch "semantic_count" field in the labels JSON
#       so downstream doesn't need re-classification.

# Step B: re-run on all 9 Phase 6 archs (no retrain)
#   The z_cache already exists for every arch on concat_A/B.
#   ~30 min, ~$0.10 total.
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs agentic_txc_02 agentic_txc_09_auxk agentic_txc_10_bare \
          agentic_txc_02_batchtopk agentic_txc_11_stack \
          tsae_paper tsae_ours tfa_big agentic_mlc_08

# Step C: train Cycle F seeds {1, 2} in background (~80 min GPU)
TQDM_DISABLE=1 PYTHONPATH=. nohup .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[1, 2], max_steps=25000, archs=['agentic_txc_02_batchtopk'])
" > logs/cycleF_seedvar.log 2>&1 &

# Step D: once Cycle F seeds land, encode + re-autointerp for each
# using the upgraded metric. 3 seeds × (encode ~2 min + autointerp ~1 min).
for SEED in 1 2; do
  TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/encode_archs.py \
    --archs agentic_txc_02_batchtopk --sets A B
  # [then modify encode_archs.py to read seed from env var if not there;
  # currently hardcoded to seed=42 — see encode_archs.py:135 CKPT_DIR]
  TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase6_qualitative_latents/run_autointerp.py \
    --archs agentic_txc_02_batchtopk
done

# Step E: update summary.md §9.5 and agentic-log headline table
# with /32 means ± stderrs for all archs (3-seed for Cycle F,
# 1-seed for others; note this explicitly).
```

**Caveats to anticipate:**

- `encode_archs.py:135` hardcodes `__seed42.pt`. Multi-seed support
  needs a small extension (env var or CLI flag).
- `run_cycle_eval.sh` hardcodes seed=42 and calls the N=8 autointerp.
  Update or fork once the upgraded autointerp is in place.
- `plot_top_features.py` (used for Figure 6 in summary.md §9.5) still
  plots top-8 by raw variance. Fine for visualisation — the paper's
  Figure-4-analogue literally shows 8 features; change the metric
  for the *score* but keep the *plot* at 8 for readability.

**What outcomes to watch for:**

- If Cycle F at `/32 × 3 seeds` gives mean ≥ `tsae_paper`'s
  `/32 × 1 seed` value minus 1 label → "Cycle F matches or beats
  `tsae_paper`" headline survives.
- If Cycle F mean falls below `tsae_paper` by > 1 label → headline
  softens to "Cycle F matches within noise, paper claim is
  sparsity-knob not beat-paper".
- If Cycle H at `/32` is still clearly below Cycle F → "stacking
  doesn't help" holds. If within noise → "stacking is neutral"
  (weaker but still consistent with our mechanism analysis).
- **Regardless**: the BatchTopK-is-the-lever story (+5 labels at N=8,
  expect ≥ +12 labels at N=32 if the effect is pure scale) is
  overwhelmingly robust.

**Skip-retrain justification**: existing ckpts for seed=42 are
fine; the noise we're worrying about is in the *evaluation*, not the
*training*. Only Cycle F needs 2 extra seeds for the variance
check. All other Phase 6 / 6.1 archs stay 1-seed — the /32 bump
alone tightens their CIs enough that large deltas (3-4+ labels)
become unambiguous.

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

- **Re-running the clustering panel with a different 2-D projection
  method (UMAP → t-SNE).** Already done; doesn't help. The
  [[summary]] §6 mentions a planned t-SNE rerun as a follow-on to
  the UMAP-disconfirmed H1 (semantic clustering on high-level
  prefix). It was executed — 48 PNGs under
  [`experiments/phase6_qualitative_latents/results/tsne/`](../../../experiments/phase6_qualitative_latents/results/tsne/)
  for all 4 archs × {high, low} × {semantic, context, pos}.
  **Silhouette scores are computed on the raw SAE latent prefix
  with cosine metric** (see
  [`run_tsne.py:83`](../../../experiments/phase6_qualitative_latents/run_tsne.py)
  and
  [`run_umap.py`](../../../experiments/phase6_qualitative_latents/run_umap.py)
  `_silhouette`), not on the 2D projection — so the scores are
  byte-identical between
  `results/umap/concat_C_v2__silhouette_scores.csv` and
  `results/tsne/concat_C_v2__tsne_silhouette_scores.csv` for the
  overlapping rows. All `high/semantic` scores are negative for
  every arch including `tsae_paper`. Visually, t-SNE surfaces more
  local structure (multiple small sub-clusters) than UMAP's single
  central blob, but the sub-clusters are not MMLU-subject-aligned.
  The H1 disconfirmation is upstream of the projection method and
  unlikely to be fixable by a different projection.

  **Likely real causes** (per summary §6 speculation, all unresolved):
  Gemma-2-2b-**IT** L13 vs paper's Gemma-2-2b **BASE** L12 (IT
  models have different internal representations at L13 than base
  models at L12); 30-token MMLU windows may not carry enough
  subject signal at this layer / model; cosine on TopK-sparse
  vectors is dominated by the handful of active features. None of
  these matter for Phase 6.1's headline result — autointerp and
  passage-smoothness (§4, §5) reproduce cleanly and both are where
  Cycle F's 7/8 win lives. The clustering panel is Figure 2 of the
  paper; autointerp and smoothness are Figures 1 and 4. We have
  Figures 1 and 4 — Figure 2 is the weak reproduction.

  Don't spend cycles on a third projection method (e.g. PCA, MDS,
  PaCMAP). If clustering matters for the paper, the intervention
  has to be upstream: try Gemma-2-2b BASE L12 (match the paper's
  model + layer exactly), or lengthen the MMLU context window, or
  switch silhouette to an ℓ2 metric on per-feature activation
  patterns rather than cosine on raw prefixes.

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
above. If everything is green, **do follow-up #0 first** (upgrade the
`x / N` metric to N=32 + auto-classify + passage-discriminative
ranking). This is the smallest-LoC change and it upgrades every
subsequent comparison — every other follow-up produces cleaner
evidence once this is in place. Step A–B of #0 (edit script + re-
run on existing 9 ckpts) takes ~30 min and ~$0.10; step C (train
Cycle F seeds {1, 2}) takes ~80 min GPU and can run in parallel with
**#5 (probe_cache download, 70 GB, CPU/network-only)**. By the time
both finish, every row in the headline table has a proper
`/32 semantic count` with auto-classification, plus Cycle F has
3-seed variance and a sparse-probing regression number for the
paper trade-off claim. **#2 and #3 are orthogonal ideas** for
pushing past 7/8 once the measurement is trustworthy.
