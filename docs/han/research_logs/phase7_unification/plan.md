---
author: Han
date: 2026-04-26
tags:
  - design
  - in-progress
---

## Phase 7 plan — pre-registered protocol for the unified leaderboard

See `brief.md` for the *why* and the agent-execution model. This doc
pre-registers the *what* and *how*, addressed to the two execution
agents (sparse-probing agent + autointerp agent).

---

## Resource environment

Each agent runs on its own RunPod with persistent volume.

| | Agent A — sparse probing | Agent B — autointerp | Agent C — case studies |
|---|---|---|---|
| GPU | NVIDIA **H200**, 141 GB HBM3e | NVIDIA **H100**, 80 GB HBM3 | NVIDIA **H100**, 80 GB HBM3 |
| vCPUs | **12** | 8 | 8 |
| System RAM | **188 GB** | 125 GB | 125 GB |
| Persistent volume | **5 TB** at /workspace | **1 TB** at /workspace | **2 TB** at /workspace |

> Agent C is a Phase 7 addition for the T-SAE §4.5-style case studies
> (HH-RLHF dataset understanding + AxBench-style steering). Reuses
> Phase 7's HF ckpts (`txcdr-base`); no independent training. Full
> scope in `agent_c_brief.md`.

> Agent A's 5 TB volume (bumped from the original 1 TB plan) is sized
> generously: ~923 GB working footprint (140 GB activation cache fp32
> + 488 GB probe cache S=128 anchor + S=128 mlc_tail + 250 GB local
> ckpts + 45 GB HF cache, venv, misc) leaves ~80% headroom. Plenty
> of room for retries, debugging artefacts, multiple cache rebuilds
> if needed, autointerp pre-staging, etc. The earlier 1 TB plan
> required a HF-push-then-delete-local dance on each ckpt to fit;
> 5 TB drops that complexity entirely — all 147 ckpts stay on disk
> for fast probing-pass access (no HF re-download).

### How Agent A should leverage the H200 pod

1. **Pre-load the anchor probe cache into RAM at process startup.**
   ~80 GB total (36 tasks × ~2.2 GB anchor each) fits comfortably
   in 188 GB. All 36 tasks' `acts_anchor` tensors held simultaneously
   in a dict; downstream loops iterate over the dict instead of
   re-loading per-task. **Saves ~50-80 hours of disk I/O across the
   probing pass for per-token SAE / TXC / H8 / SubseqH8 archs (46 of
   the 49).** MLC-tail (~400 GB total) streams per-task only when
   probing the 3 MLC archs (rows 4-6) — those iterations pay one
   I/O hit per task. Drop Phase 5's universal streaming pattern
   (which was sized for ~5 GB RAM caps); use the hybrid pattern
   above instead.

2. **Parallel probe fitting via joblib (12 workers).** sklearn's
   `LogisticRegression(solver="liblinear")` is single-threaded. Wrap
   the (k_feat × S × task) probe-fit loop in
   `joblib.Parallel(n_jobs=12, backend="loky")`. **Set
   `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`** in the worker env to
   prevent nested-parallelism contention with sklearn.

3. **Pipeline GPU encoding with CPU probe-fitting.** While GPU encodes
   ckpt[i] task[t+1], CPU pool fits probes for ckpt[i] task[t]. CPU
   work effectively becomes free; pipeline wall-clock is GPU-bound.
   Reduces probing pass from ~12 hr (single-threaded) to ~5-8 hr.

4. **Bump training batch size.** Phase 5's `batch=1024` was set for
   32 GB 5090s. H200's 141 GB has plenty of headroom: at the worst
   case (T=32, k_win=500, full H8 stack with multi-distance pair
   tensor) the active footprint is ~40 GB, leaving ~100 GB free.
   Bump to **batch=4096** for ~1.5-2× speedup per training. Verify on
   a smoke-test cell first that loss curves match batch=1024
   convergence; if they don't, fall back to batch=2048.

5. **Run SubseqH8 at T_max ∈ {32, 64, 128}.** These cells are added
   to the canonical table as rows 48–50 specifically because H200's
   memory headroom unlocks them (T_max=128 weights+Adam ≈ 95 GB —
   would OOM H100 80 GB). Tests the "subseq sampling unlocks
   T-scaling beyond what TXC alone can" claim; positive result is
   paper-strong.

6. **Push ckpts to HF incrementally as they complete.** Don't batch
   all uploads to the end. Agent B is polling HF; faster pushes =
   earlier autointerp start = parallel-execution win.

### How Agent B should leverage the H100 pod

1. **Concurrent Haiku API calls.** 8 vCPUs; use
   `concurrent.futures.ThreadPoolExecutor(max_workers=16)` for
   autointerp scoring. Anthropic API rate limits typically allow
   ~50-100 req/s — the bottleneck is per-request latency, not
   throughput. Expect 10-20× speedup over sequential calls.

2. **Cache hot ckpts in RAM.** As Agent A pushes new ckpts, keep the
   last ~10 in RAM (each ~1-2 GB; 20 GB total). With 125 GB RAM
   this is no constraint. Feature-variance computation on a hot
   ckpt is sub-second.

3. **Pre-build qualitative passages once at startup.** concat-A,
   concat-B, concat-random tokenizations + activations through the
   base model. Reuse for every arch (the activations DON'T depend on
   the SAE; only on the model). Cache at
   `/workspace/temp_xc/experiments/phase7_unification/results/qualitative_cache/`.

4. **GPU-batch the top-256-feature encoding.** When extracting
   per-feature activations on top-10 contexts, batch all
   256 × 10 = 2560 contexts into a single SAE encode call. Single
   GPU pass instead of 2560 serial calls.

5. **HF polling cadence**. Poll HF every 60s for new ckpts (use
   `huggingface_hub.list_repo_files()`). Don't busy-wait. Track
   which ckpts have already been autointerp'd in a local
   `processed_ckpts.json` to make polling idempotent across pod
   restarts.

6. **Prefetch next ckpt during autointerp scoring.** While Haiku is
   scoring features for ckpt[N], start downloading ckpt[N+1] from
   HF in a background thread. Hides the ~1-2 minute HF download
   behind the ~5-10 minute Haiku scoring.

### Phase 7 results layout — keeping the index clean

The pre-Phase-7 branches each maintained their own
`results/training_index.jsonl` and `probing_results.jsonl` in their
phase-specific directories. After the prior-phase merge,
`han-phase7-unification` now carries forward all of these as
**read-only historical references**:

| path | belongs to | Phase 7 access |
|---|---|---|
| `experiments/phase5_downstream_utility/results/training_index.jsonl` | Phase 5 | read-only |
| `experiments/phase5_downstream_utility/results/probing_results.jsonl` | Phase 5 | read-only |
| `experiments/phase5b_t_scaling_explore/results/training_index.jsonl` | Phase 5B | read-only |
| `experiments/phase5b_t_scaling_explore/results/probing_results.jsonl` | Phase 5B | read-only |
| `experiments/phase6_2_autoresearch/results/phase62_results.jsonl` | Phase 6.2 | read-only |
| `experiments/phase6_qualitative_latents/results/...` | Phase 6.1, 6.3 | read-only |

**Phase 7 writes ONLY to** `experiments/phase7_unification/results/`,
which Agent A creates fresh:

```
experiments/phase7_unification/results/
├── ckpts/                          # gitignored, sync to HF txcdr-base
├── training_logs/                  # per-ckpt JSON logs
├── training_index.jsonl            # one row per (arch_id, seed) trained
├── probing_results.jsonl           # one row per (run_id, task, agg, k_feat, S)
├── t_sweep_results.jsonl           # one row per T-sweep cell with alive_fraction etc.
├── probe_cache/                    # rebuilt cache, sync to HF txcdr-base-data
├── plots/                          # phase7_*.png
└── autointerp/                     # Agent B's outputs
    └── <run_id>/
        ├── concat_A_labels.json
        ├── concat_B_labels.json
        ├── concat_random_labels.json
        └── cumulative_semantic.json
```

**Path-discipline mechanism — `_paths.py`**: Agent A creates
`experiments/phase7_unification/_paths.py` mirroring Phase 5B's
pattern. All Phase 7 drivers (train + probe + analyzer) import path
constants from `_paths.py`; **no bare-string paths anywhere**. This
prevents fork-drift from Phase 5's `train_primary_archs.py` /
`run_probing.py` accidentally writing to Phase 5 dirs:

```python
# experiments/phase7_unification/_paths.py — Phase 7 canonical paths
from pathlib import Path
import os

REPO = Path(os.environ.get("PHASE7_REPO", os.getcwd())).resolve()
ANCHOR_LAYER = 12                      # 0-indexed
MLC_LAYERS = (10, 11, 12, 13, 14)
CACHE_DIR  = REPO / "data/cached_activations/gemma-2-2b/fineweb"
OUT_DIR        = REPO / "experiments/phase7_unification/results"
CKPT_DIR       = OUT_DIR / "ckpts"
LOGS_DIR       = OUT_DIR / "training_logs"
INDEX_PATH     = OUT_DIR / "training_index.jsonl"
PROBING_PATH   = OUT_DIR / "probing_results.jsonl"
T_SWEEP_PATH   = OUT_DIR / "t_sweep_results.jsonl"
PROBE_CACHE    = OUT_DIR / "probe_cache"
PLOTS_DIR      = OUT_DIR / "plots"
AUTOINTERP_DIR = OUT_DIR / "autointerp"
# Pre-Phase-7 results — READ-ONLY references; don't write here
PHASE5_RESULTS  = REPO / "experiments/phase5_downstream_utility/results"
PHASE5B_RESULTS = REPO / "experiments/phase5b_t_scaling_explore/results"
PHASE6_RESULTS  = REPO / "experiments/phase6_qualitative_latents/results"
```

**Startup banner**: every Phase 7 driver should print on entry:
```
Phase 7 driver — writing to:
  experiments/phase7_unification/results/training_index.jsonl
  experiments/phase7_unification/results/probing_results.jsonl
```
so any misconfigured path is immediately visible. Mismatched paths
are an error, not a silent corruption.

**Analyzer convention**: Agent A's headline-leaderboard analyzer
loads ONLY `experiments/phase7_unification/results/probing_results.jsonl`.
It does NOT glob across `experiments/*/results/probing_results.jsonl`
— that would mix in Phase 5 / 5B IT-trained results. If a
cross-phase comparison is needed (e.g. "did Phase 7's TXCDR T=5
match Phase 5's at the same arch?"), that's a separate ad-hoc
analysis explicitly named, not the headline.

### Hugging Face repository layout

Phase 7 uses **two new HF repos** to avoid any collision with the
existing IT-trained ckpts and caches:

| repo | purpose | Phase 7 access |
|---|---|---|
| `han1823123123/txcdr` | pre-Phase-7 IT ckpts (Phase 5, 5b) | **read-only** if needed for cross-check |
| `han1823123123/txcdr-data` | pre-Phase-7 IT-derived caches | **read-only** if needed |
| `han1823123123/txcdr-base` (NEW) | base-Gemma-2-2b ckpts (Phase 7) | **READ + WRITE** for both agents |
| `han1823123123/txcdr-base-data` (NEW) | base-Gemma-2-2b activation + probe caches | **READ + WRITE** for both agents |

**Anti-confusion rules baked into Phase 7 code:**

- Hardcoded repo names in `scripts/hf_upload_phase7_ckpts.py` and
  `scripts/hf_upload_phase7_data.py` — no env-var override paths.
  Agents physically cannot push to the wrong repo without editing
  the script.
- Subject-model verification: each ckpt's training metadata is
  checked for `meta["subject_model"] == "google/gemma-2-2b"` before
  pushing to `txcdr-base` (or aborting if mismatched).
- README on each new repo points to the other three with the
  read/write table above.

---

### Subject model

`google/gemma-2-2b` (base, NOT instruct).

- **Layer 12 anchor (0-indexed)** = `model.model.layers[12]`.
- **L10–L14 for MLC** (5-layer window centred on L12; 0-indexed).
- HF model id: `google/gemma-2-2b`.
- Tokenizer: `google/gemma-2-2b` (matches base model).

> **Layer choice — deliberate shift from Phase 5/5b/6 (which used
> 0-indexed L13):** T-SAE (Ye et al. 2025 §4.1) trains Gemma SAEs at
> "layer 12" for "comparability with Neuronpedia"; TFA (Lubana et
> al. 2025 App. B.1) explicitly says "Layer 12 for Gemma … both
> 0-indexed". Phase 7's L12 anchor + L10–L14 MLC stack therefore
> matches *exactly* the residual-stream tap used by both reference
> papers. Phase 5 used 0-indexed L13 (= 14/26 ≈ 53.8% depth);
> Phase 7 uses 0-indexed L12 (= 13/26 = 50.0% depth, "around 50%
> model depth" per TFA's wording). Cache rebuild from the model
> switch is zero-marginal-cost so this layer fix is essentially free.

### Activation cache (rebuild) — owned by Agent A

- Source: 24 000 sequences from FineWeb (consistent with prior
  Phase 5/5b pipeline; document the FineWeb-vs-Pile difference vs
  T-SAE/TFA in the limitations section).
- Context length: 128 tokens.
- Storage: fp16, layer-major. ~3 GB per layer × 5 layers ≈ 15 GB.
  Path: `data/cached_activations/gemma-2-2b/fineweb/resid_L<n>.npy`.
- Sync to HF for cross-agent access:
  `han1823123123/txcdr/phase7_activation_cache/`.

### Probe cache (built) — owned by Agent A

- Tasks: same 36-task set as Phase 5 (8 dataset families).
- Per-task storage:
  - `acts_anchor.npz`: train_acts, test_acts at L12, **shape
    (N, 128, d)** — full 128-token tail. Used by per-token SAE,
    TXC family, H8, SubseqH8. Probing aggregates over the LAST
    S=32 of these per the new headline S decision (revised
    2026-04-26).
  - `acts_mlc.npz`: train_acts, test_acts at L10-L14, **(N, 5, d)** —
    last real token only. Used for the no-context MLC reference.
  - `acts_mlc_tail.npz`: train_acts, test_acts at L10-L14, **shape
    (N, 128, 5, d)** — full 128-token tail. Used by MLC mean-pool
    probing. Cache stays at S=128 even though headline S is now 32
    — costs nothing extra (already built) and keeps the door open
    for future S-sweep ablations without rebuilding.
  - `meta.json`: dataset_key, task_name, n_train, n_test, etc.
- Splits: same as Phase 5 (n_train=3040, n_test=760).
- Storage budget per task (fp16): anchor 2.24 GB + mlc_last 0.087 GB +
  mlc_tail 11.2 GB ≈ **13.5 GB/task × 36 = ~488 GB total**.
  Comfortably under the 5 TB persistent volume.
- RAM strategy at probe time: anchor pre-loads into RAM once (~80 GB;
  fits in H200's 188 GB). MLC-tail streams per-task only when probing
  the 3 MLC archs (rows 4-6 in `canonical_archs.json`).
- Sync to HF for cross-agent access:
  `han1823123123/txcdr-base-data/probe_cache/`.

### Sparsity convention

`k_win = 500` across all archs and all T values, fixed. Per-arch
k_pos derivation is in [§Canonical architecture set](#canonical-architecture-set)
below — that's the single source of truth for the arch list.

Justification (against k_win=100 alternative):
- Switching to Gemma2B base means everything is retrained from
  scratch regardless, so "preserve existing TXC ckpts" is moot —
  but **preserving the regime where TXC recipes were tuned is
  not.** Recipes (Track 2 anti-dead, H8 multi-distance, B2/B4
  subseq) showed working trade-offs at k_win=500 in Phase 5/5B.
  Pushing to k_win=100 introduces a "did the recipe transfer to a
  sparser regime?" question on top of the model-switch question.
  Less risk to keep one variable fixed.
- 500/18432 = 2.7% density — well within the "sparse-probing regime"
  used by the SAEBench protocol.
- For per-token SAEs (topk_sae, mlc, tsae_paper at our convention), it
  is denser than the literature norm (k≤100 typical), but this is the
  same density TXC uses, which is what fairness requires.
- `tsae_paper` reported additionally at native k=20 as a paper-
  faithful reference and the regime difference is documented.

### Probing protocol — pre-registered

Single S-parameterized aggregation:

```python
def probe_aggregate(model, anchor_acts, T, S=128):
    """anchor_acts: (N, S, d_in) — the S-token tail per example.
    Returns (N, d_sae) per-example representations."""
    if T == 1:
        # per-token SAE
        z_per = encode_per_token(model, anchor_acts)   # (N, S, d_sae)
        z = z_per[:, T-1:].mean(axis=1)                # drop first T-1 (=0), mean
    else:
        # window arch
        K = S - T + 1
        windows = slide_windows(anchor_acts, T)        # (N, K, T, d_in)
        z_per_win = encode(model, windows)             # (N, K, d_sae)
        # Drop windows whose left edge < T-1 (would bleed beyond tail).
        z = z_per_win[:, T-1:].mean(axis=1)            # (N, d_sae)
    return z
```

Probe: SAEBench protocol. Top-`k_feat`-by-`|mean_pos − mean_neg|` +
L1 LR (C=1.0, max_iter=2000, `with_mean=False` standardization).
Headline `k_feat ∈ {5, 20}`; ablation at `k_feat ∈ {1, 2}`.

#### Reported S values

**Decision (revised 2026-04-26): S = 32 across the board.**

- Compatible with all 48 of 49 canonical archs (cell-validity rule
  is `S ≥ T`; only row 49 SubseqH8 T_max=64 invalid, since 32 < 64).
- ~6× faster per-task probing than the earlier S=128 plan (encode
  cost scales linearly in the number of windows, S − T + 1).
- Earlier S=128 / 64 / 20 sweep was dropped after the (T, S)
  validity rule was found to be over-aggressive (see correction
  below).

Sanity checks (e.g. the subseq_h8 vs txcdr_t5 vs mlc ordering
verification) use **S = 20** specifically: matches Phase 5's tail
length so cross-phase numerical sanity checks are direct, and
valid for all 47 archs with T ≤ 20.

NOT reported: S = T (per-window comparison) — explicitly dropped
because comparing across architectures with different T at S=T is
structurally confusing.

#### FLIP convention

Apply `max(AUC, 1−AUC)` per-task on `winogrande_correct_completion`
and `wsc_coreference` (cross-token tasks with arbitrary label
polarity). Same as Phase 5.

#### (T, S) validity + short-sentence handling

**Correction (2026-04-26)**: The earlier "drop first T−1 windows"
rule was over-aggressive. The corrected rule (in the live code at
`run_probing_phase7.aggregate_s` + `cell_is_valid`):

  Kept windows per example = `effective_tail − T + 1`
  (= number of T-windows fully inside the effective tail)
  Cell validity rule: `S ≥ T`

Why the earlier rule was wrong: it claimed windows starting near
the tail's left boundary "had less preceding-tail context", but
window archs encode the window IN ISOLATION (no recurrent state,
no cross-window dependency at probe time). The window IS the
encoder input; it doesn't need additional preceding context.

Two failure modes still guarded:

1. **Cell-level**: any cell with `S < T` has zero windows that fit
   inside the tail for any example, regardless of length. Skipped
   with `{"skipped": true, "reason": "S_lt_T"}` in the jsonl so
   downstream analysis sees the absence explicitly. At the chosen
   headline S=32, this only skips row 49 (SubseqH8 T_max=64).

2. **Example-level**: real probing examples are tokenized to ≤ 128
   tokens and right-padded. For an example of true length `L` tokens,
   the cached `last_idx` records the position of the last real token
   in the 128-tail. The probing driver computes
   `effective_tail_i = min(S, last_idx_i + 1)` per example and keeps
   only windows whose left-edge ≥ first_real_position AND right-edge
   ≤ last_idx_i. Examples where `effective_tail_i < T` contribute no
   kept windows for that (T, S) cell and are filtered. Per-cell drop
   counts `n_drop_train` / `n_drop_test` are written to the jsonl
   so a cell that drops the majority of its examples (likely
   indicating a short-sentence-heavy task) is auditable post-hoc.

Avoided:
- **Padding artifacts**: aggregating over pad tokens contaminates
  the mean with whatever activation the model produces on PAD. The
  per-example `last_idx` filter restricts the average to real tokens.
- **Silent garbage**: a (T, S) cell that yields `kept ≤ 0` would
  otherwise produce `mean of empty array = nan` and pollute the AUC
  computation. The hard skip + jsonl marker makes the failure
  inspectable.

Tasks dominated by short sentences (e.g. WSC coreference) may have
large `n_drop` at high T and small S — that's expected and
documented per-cell rather than averaged into a misleading headline.

---

## Canonical architecture set {#canonical-architecture-set}

**The machine-readable source of truth is
`experiments/phase7_unification/canonical_archs.json`.**

Agents should load that JSON directly rather than parsing the
markdown table below. Example:

```python
import json
canonical = json.load(open("experiments/phase7_unification/canonical_archs.json"))
for seed in canonical["seeds"]:                      # [42, 1, 2] in this order
    for arch in canonical["archs"]:                  # 49 entries
        train(arch, seed)                            # arch dict has all params
```

The JSON contains: subject_model + anchor_layer + mlc_layers +
d_in/d_sae + k_win_default + seeds (in training-loop order) +
groups dict + 49 arch records. Each arch record includes:
`row`, `arch_id`, `group`, `T`/`T_max`/`t_sample`, `k_win`,
`k_pos`, `shifts`, `src_module`, `src_class`, `recipe`, `purpose`,
plus arch-specific extras (alpha, gamma, n_scales, n_layers).

The markdown table below is a human-readable view of the same data.
**If the JSON and the markdown ever diverge, the JSON is canonical.**

Every row is one architecture trained at 3 seeds (seed ∈ {1, 2, 42},
trained outer-loop seed-first, see §Deliverable A.i). Total rows:
**49**. Total trainings: 49 × 3 = **147** on H200. Estimated
wall-clock at ~5 min/training (with batch=4096 on H200): **~12 hr**.

T-sweep entries (rows 14–45) double as leaderboard entries — the
Agent A leaderboard and the Agent A T-sweep are subsets of this
table, not separate runs. Rows 48–49 are SubseqH8 T_max-sweep cells
that exist specifically to test the "subseq sampling unlocks
T-scaling beyond what TXC alone can" claim — they're enabled by the
H200's 141 GB memory which fits T_max=64 but H100 80 GB cannot.

Notes on the table format:
- **k_pos** is derived as `k_win / n_active_positions` where
  n_active_positions is T (for window archs), 1 (per-token), L
  (for MLC, where L=5 layers), or `t_sample` (for B2/B4 subseq).
- **shifts** column is filled only for H8 rows (multi-distance
  InfoNCE recipe).
- **group**: 1 = per-token/non-TXC; 2 = fixed-T TXC variant;
  3 = TXCDR T-sweep; 4 = H8 T-sweep; 5 = anchor cell (k_pos=100);
  6 = SubseqH8 T_max-sweep (H200-only, fits because of 141 GB GPU).

| # | arch_id | grp | T (or T_max,t_sample) | k_win | k_pos | shifts | recipe / purpose |
|---|---|---|---|---|---|---|---|
| 1 | `topk_sae` | 1 | 1 | 500 | 500 | — | per-token TopK SAE; baseline |
| 2 | `tsae_paper_k500` | 1 | 1 | 500 | 500 | — | per-token Matryoshka BatchTopK + temporal InfoNCE (α=0.1) + AuxK; T-SAE port at our k convention |
| 3 | `tsae_paper_k20` | 1 | 1 | 20 | 20 | — | same as #2 at native k=20; paper-faithful baseline (Ye et al. 2025) |
| 4 | `mlc` | 1 | 5 layers (L10–L14) | 500 | 100/layer | — | per-token TopK over 5 layers; layer-crosscoder baseline |
| 5 | `mlc_contrastive_alpha100_batchtopk` | 1 | 5 layers | 500 | 100/layer | shift=1 | MLC + Matryoshka H/L + temporal InfoNCE (α=1.0) + BatchTopK; Phase 5 lp leader |
| 6 | `agentic_mlc_08` | 1 | 5 layers | 500 | 100/layer | shifts (1,2,3) γ=0.5 | MLC + multi-scale InfoNCE (n_scales=3); Phase 5 multi-scale MLC |
| 7 | `tfa_big` | 1 | n/a | 500 | n/a | — | TFA full (predictive + novel codes), full-size attention; predictive-coding reference |
| 8 | `agentic_txc_02` | 2 | T=5 | 500 | 100/slab | shifts (1,2,3) γ=0.5 over matryoshka prefixes | TXC + multi-scale matryoshka InfoNCE; Phase 5 multi-scale matryoshka winner |
| 9 | `txc_bare_antidead_t5` (Track 2) | 2 | T=5 | 500 | 100/slab | — | TXC + tsae-paper anti-dead stack only (no matryoshka, no contrastive); Phase 6.1 `agentic_txc_10_bare` |
| 10 | `txc_bare_antidead_t10` (Track 2 T=10) | 2 | T=10 | 500 | 50/slab | — | same recipe as #9 at T=10 |
| 11 | `txc_bare_antidead_t20` (Track 2 T=20) | 2 | T=20 | 500 | 25/slab | — | same recipe as #9 at T=20 |
| 12 | `phase5b_subseq_track2` (B2) | 2 | T_max=10, t_sample=5 | 500 | 100/active-slab | — | Track 2 + subseq sampling (random t_sample of T_max positions feed gradient per step); Phase 5B Track 2 winner |
| 13 | `phase5b_subseq_h8` (B4) | 2 | T_max=10, t_sample=5 | 500 | 100/active-slab | (1,2,5), inv-dist weights | H8 stack (= matryoshka H/L + multi-distance) + subseq sampling; Phase 5B mp champion |
| 14 | `txcdr_t3` | 3 | T=3 | 500 | 167 | — | vanilla TemporalCrosscoder, no matryoshka/contrastive/anti-dead |
| 15 | `txcdr_t4` | 3 | T=4 | 500 | 125 | — | same |
| 16 | `txcdr_t5` | 3 | T=5 | 500 | 100 | — | same |
| 17 | `txcdr_t6` | 3 | T=6 | 500 | 83 | — | same |
| 18 | `txcdr_t7` | 3 | T=7 | 500 | 71 | — | same |
| 19 | `txcdr_t8` | 3 | T=8 | 500 | 62 | — | same |
| 20 | `txcdr_t9` | 3 | T=9 | 500 | 56 | — | same |
| 21 | `txcdr_t10` | 3 | T=10 | 500 | 50 | — | same |
| 22 | `txcdr_t12` | 3 | T=12 | 500 | 42 | — | same |
| 23 | `txcdr_t14` | 3 | T=14 | 500 | 36 | — | same |
| 24 | `txcdr_t16` | 3 | T=16 | 500 | 31 | — | same |
| 25 | `txcdr_t18` | 3 | T=18 | 500 | 28 | — | same |
| 26 | `txcdr_t20` | 3 | T=20 | 500 | 25 | — | same |
| 27 | `txcdr_t24` | 3 | T=24 | 500 | 21 | — | same |
| 28 | `txcdr_t28` | 3 | T=28 | 500 | 18 | — | same |
| 29 | `txcdr_t32` | 3 | T=32 | 500 | 16 | — | same |
| 30 | `phase57_partB_h8_bare_multidistance_t3` | 4 | T=3 | 500 | 167 | (1) | H8 recipe = TXC + anti-dead + matryoshka H/L (H=0.2·d_sae) + multi-distance InfoNCE; auto-scaled shifts (1, max(1,T//4), max(1,T//2)) deduped; inverse-distance weights w_s = 1/(1+s) |
| 31 | `phase57_partB_h8_bare_multidistance_t4` | 4 | T=4 | 500 | 125 | (1, 2) | H8 |
| 32 | `phase57_partB_h8_bare_multidistance_t5` | 4 | T=5 | 500 | 100 | (1, 2) | H8 |
| 33 | `phase57_partB_h8_bare_multidistance_t6` | 4 | T=6 | 500 | 83 | (1, 3) | H8 |
| 34 | `phase57_partB_h8_bare_multidistance_t7` | 4 | T=7 | 500 | 71 | (1, 3) | H8 |
| 35 | `phase57_partB_h8_bare_multidistance_t8` | 4 | T=8 | 500 | 62 | (1, 2, 4) | H8 |
| 36 | `phase57_partB_h8_bare_multidistance_t9` | 4 | T=9 | 500 | 56 | (1, 2, 4) | H8 |
| 37 | `phase57_partB_h8_bare_multidistance_t10` | 4 | T=10 | 500 | 50 | (1, 2, 5) | H8 |
| 38 | `phase57_partB_h8_bare_multidistance_t12` | 4 | T=12 | 500 | 42 | (1, 3, 6) | H8 |
| 39 | `phase57_partB_h8_bare_multidistance_t14` | 4 | T=14 | 500 | 36 | (1, 3, 7) | H8 |
| 40 | `phase57_partB_h8_bare_multidistance_t16` | 4 | T=16 | 500 | 31 | (1, 4, 8) | H8 |
| 41 | `phase57_partB_h8_bare_multidistance_t18` | 4 | T=18 | 500 | 28 | (1, 4, 9) | H8 |
| 42 | `phase57_partB_h8_bare_multidistance_t20` | 4 | T=20 | 500 | 25 | (1, 5, 10) | H8 |
| 43 | `phase57_partB_h8_bare_multidistance_t24` | 4 | T=24 | 500 | 21 | (1, 6, 12) | H8 |
| 44 | `phase57_partB_h8_bare_multidistance_t28` | 4 | T=28 | 500 | 18 | (1, 7, 14) | H8 |
| 45 | `phase57_partB_h8_bare_multidistance_t32` | 4 | T=32 | 500 | 16 | (1, 8, 16) | H8 |
| 46 | `txcdr_t20_kpos100` | 5 | T=20 | 2000 | 100/slab | — | anchor cell: vanilla TXCDR at fix-k_pos=100 (= row 26 with denser k); disentangles "context limit" vs "per-slab sparsity collapse" |
| 47 | `phase57_partB_h8_bare_multidistance_t20_kpos100` | 5 | T=20 | 2000 | 100/slab | (1, 5, 10) | anchor cell: H8 at fix-k_pos=100 (= row 42 with denser k); same disentanglement |
| 48 | `phase5b_subseq_h8_T32_s5` | 6 | T_max=32, t_sample=5 | 500 | 100/active-slab | auto (1, 8, 16) | SubseqH8 at T_max=32 — extended T-scaling test for subseq family; H8 multi-distance shifts auto-scaled |
| 49 | `phase5b_subseq_h8_T64_s5` | 6 | T_max=64, t_sample=5 | 500 | 100/active-slab | auto (1, 16, 32) | SubseqH8 at T_max=64 — extreme T-scaling test; would OOM on H100 80GB at fp32, fits comfortably on H200 141GB. Tests whether subseq enables T-scaling beyond what any other arch can reach |

### What's NOT in this table (and why)

Phase 5 / 5B / 6 had many more architectures. The following are
explicitly excluded from Phase 7 retraining; they remain on their
historical branches as code, mentioned in the negative-results /
appendix sections of the paper if relevant:

- H7 (`phase57_partB_h7_bare_multiscale`) — multi-scale rather than
  multi-distance contrastive. H8 covered the better-performing
  variant of this family.
- Phase 5B negatives (D1 strided, C-family token-level encoder,
  F SubsetEncoderTXC).
- BatchTopK paired variants beyond `mlc_contrastive_alpha100_batchtopk`.
- Stacked SAE family.
- TXCDR weight-sharing ablations (`txcdr_shared_*`, `txcdr_tied_*`,
  `txcdr_pos_*`, `txcdr_causal_*`, `txcdr_block_sparse_*`,
  `txcdr_lowrank_dec_*`, `txcdr_rank_k_dec_*`, `txcdr_basis_*`,
  `txcdr_rotational_*`).
- `time_layer_crosscoder_t5`, `mlc_temporal_t3`, `temporal_contrastive`
  and other Phase 5 exploratory archs.
- `tsae_ours` (Phase 6's crude T-SAE port — superseded by `tsae_paper`).
- Most of the 8 H8 shift-ablation variants (`phase57_partB_h8a_shifts*`)
  except as appendix-only references.

---

## Agent A — sparse-probing leaderboard

### Deliverable A.i — definitive leaderboard

For all 49 archs in [§Canonical architecture set](#canonical-architecture-set),
3 seeds, headline at S=128 and `k_feat ∈ {5, 20}`. Per-arch: mean
± σ over 3 seeds. Output:

- `experiments/phase7_unification/results/probing_results.jsonl` —
  full per-task per-seed per-`k_feat` per-S rows.
- `experiments/phase7_unification/results/headline_S128_k5.json`
  + `headline_S128_k20.json` — aggregated mean ± σ per arch.
- `experiments/phase7_unification/results/plots/phase7_headline_bar_S128_k5.png`
  + `..._k20.png` — paired bar charts.

#### Training loop ordering — OUTER LOOP IS SEED, not arch

The training driver MUST iterate seeds on the OUTER loop, archs on
the INNER loop:

```
for seed in (42, 1, 2):
    for arch_id in canonical_set_49_archs:
        train(arch_id, seed)
        push_ckpt_to_hf(arch_id, seed)   # incremental, per-ckpt
```

**Why this matters**:

- After ~6-10 hr Agent A completes the **full seed=42 batch** (one
  ckpt per arch, all 49 archs). At that point Agent B has all 49
  seed=42 ckpts on HF and can start autointerp on the COMPLETE
  Pareto x-axis (it just lacks σ data). Per the cost-saving rule,
  autointerp is seed=42-only anyway.
- If the outer loop were instead per-arch, Agent B would have
  seed=42 ckpts trickling in for some archs but not others,
  delaying when it can compute the full Pareto.
- Probing-side σ analysis can wait until seed=1 and seed=2 finish.

**Per-seed signaling to Agent B**: after the seed=42 outer-loop
iteration completes, Agent A pushes a marker file
`han1823123123/txcdr-base/seed42_complete.json` listing all 49
arch_ids of completed seed=42 ckpts. Agent B polls for this marker
and starts the Pareto-x-axis run when it appears (rather than
polling for individual ckpts). Same marker pattern for seed=1 and
seed=2 if Agent B wants to incorporate σ.

### Deliverable A.ii — T-sweep at fix k_win=500

For arch ∈ {`txcdr_t<T>`, `H8_t<T>`} (rows 14–45 of the canonical
table), all 3 seeds. Anchor cells (rows 46–47) ALSO trained, 3 seeds.

Per-cell metrics in the JSON output:

- `test_auc` at S=128, k_feat=5
- `test_auc` at S=128, k_feat=20
- `test_auc` at S=20, k_feat=5 (continuity check)
- `alive_fraction`: fraction of d_sae features that fired ≥ once on
  a 5K-token held-out batch
- `final_recon_loss`: last logged reconstruction loss
- `final_step`, `converged`, `plateau_last`: convergence sanity

Output:

- `experiments/phase7_unification/results/t_sweep_results.jsonl` —
  per-cell rows.
- `experiments/phase7_unification/results/plots/phase7_t_sweep_S128_k5.png`
  — line plot AUC vs T for {txcdr, H8}, with anchor cells overlaid.
- `experiments/phase7_unification/results/plots/phase7_t_sweep_alive_fraction.png`
  — alive_fraction vs T (sanity-check plot to flag sparsity
  collapse at large T).

**Caveat documented in the writeup**: at fix k_win=500, per-slab
k_pos = 500/T shrinks with T. By T=32, k_pos ≈ 16 per slab. If
T=32's AUC regression coincides with alive_fraction collapse,
the regression is partly attributable to per-slab under-training,
not architecture-intrinsic context limits. The anchor cell at
T=20 fix-k_pos=100 (rows 46–47) tests this disentanglement directly.

### Hypotheses (pre-registered)

- **H1 (Gemma-IT vs base)**: per-task AUCs may change by ~0.01-0.02
  on most tasks; ARCH RANKINGS likely preserve. If rankings flip
  significantly, this is itself a finding worth reporting.
- **H2 (k_win=500 fixed across families)**: bumping per-token SAE
  and MLC families up to k_win=500 from Phase 5's k=100 should
  improve their absolute AUC moderately. Whether the cross-family
  ranking (TXC vs MLC vs SAE) preserves is the key question.
- **H3 (T-sweep at fix k_win)**: Phase 5's T=5 peak was at fix
  k_pos=100 (k_win = 100·T). At fix k_win=500 the peak may shift,
  flatten, or persist. Sub-hypotheses:
    - **H3a (sparsity-dilution)**: peak shifts to larger T, T-scaling
      becomes near-monotone.
    - **H3b (context-mismatch)**: peak stays near T=5–6, regardless
      of k convention.
    - **H3c (sparsity-collapse at large T)**: T ≥ 24 underperforms
      due to per-slab k_pos < 20 being too sparse. Anchor cells at
      T=20 fix-k_pos=100 (rows 46–47) disentangle this from H3b.
- **H4 (TXC vs SAE at fix k_win=500 + S=128)**: TXC family wins by
  ≥ 0.005 mp at headline. If not, the entire "TXC > SAE" claim
  weakens significantly.

### Figures Agent A produces

- `phase7_headline_bar_S128_k5.png` — paired bar chart at headline.
- `phase7_headline_bar_S128_k20.png` — same at k_feat=20.
- `phase7_t_sweep_S128_k5.png` — line plot (txcdr, H8) AUC vs T.
- `phase7_t_sweep_alive_fraction.png` — alive_fraction vs T.
- `phase7_seed_variance.png` — error bars on top archs.
- (Optional) `phase7_S_sweep.png` — AUC vs S for top archs.

---

## Agent B — qualitative autointerp

### Deliverable B.i — Top-256 cumulative semantic Pareto

Single Pareto plot, replicating
`experiments/phase6_qualitative_latents/results/phase63_pareto_top256.png`
from `han-phase6` but with Phase 7 archs from the canonical set.
**Skip the other 3 Pareto plots from Phase 6** (top-32, top-64,
top-128 cumulative are not part of Phase 7's deliverable).

For each arch in the canonical set:

1. Take the top-256 features by per-token activation variance over
   `concat_A + concat_B + concat_random` (Phase 6 protocol).
2. For each of those 256 features, send the top-10 activating
   20-token contexts to Claude Haiku 4.5 with the Bills-et-al-style
   labelling prompt. Get a one-line label.
3. Hand-classify each label as **semantic** or **non-semantic**
   (Phase 6's protocol; semantic = names a concept/topic/theme, not
   punctuation/whitespace/syntax/format pattern).
4. Plot **cumulative semantic count** vs **rank** (rank = position
   in variance-sorted top-256), giving a curve per arch.
5. The headline figure is the FINAL cumulative count at rank 256
   (i.e., total semantic features in top-256), plotted on the y-axis
   against the arch's sparse-probing AUC at S=128, k_feat=5 on the
   x-axis. Arches in the upper-right are Pareto-better.

Output:

- `experiments/phase7_unification/results/autointerp/<run_id>/concat_<A|B|random>_labels.json`
  — per-arch per-feature labels.
- `experiments/phase7_unification/results/autointerp/<run_id>/cumulative_semantic.json`
  — per-arch cumulative-count curves.
- `experiments/phase7_unification/results/plots/phase7_qualitative_pareto_top256.png`
  — the headline Pareto figure.

### Cost-saving: autointerp at seed=42 only

To keep Claude Haiku spend reasonable, **autointerp scoring uses
seed=42 only** for each arch (not 3 seeds). Sparse probing on the
x-axis still uses 3-seed mean. This is a documented compromise.

Scale check at seed=42 only:

- 49 archs × 256 features × 10 contexts × 1 seed ≈ 125K Haiku calls.
- At ~5K input tokens / call with prompt caching, expected cost
  ~$50-150. Tractable.

### Coordination dependency

Agent B's Pareto plot REQUIRES Agent A's `probing_results.jsonl` to
populate the x-axis (the sparse-probing AUC). Coordination:

- Agent A pushes ckpts to HF as they complete (incremental, per arch).
- Agent A pushes a partial `probing_results.jsonl` to HF after
  probing each batch of archs.
- Agent B polls HF for ckpts; for each new ckpt, runs autointerp.
- Agent B reads Agent A's `probing_results.jsonl` (latest snapshot)
  to compose the Pareto plot.
- Final Pareto plot is generated AFTER Agent A finishes the full
  3-seed sparse-probing pass.

### Figures Agent B produces

- `phase7_qualitative_pareto_top256.png` — the single deliverable.
- (Optional) `phase7_top8_panel.png` — top-8-by-variance feature
  activation panel for each canonical arch on concat-A and concat-B.
  Phase 6.1's standard qualitative figure, useful for paper but not
  required.

---

## Branch hygiene — both agents

- All Phase 7 work on `han-phase7-unification`, branched off
  `origin/han`.
- Cherry-picked arch files only (NOT phase5b/6 experiment infra).
- New ckpts in `experiments/phase7_unification/results/ckpts/`
  (gitignored, sync to HF at `han1823123123/txcdr/phase7_ckpts/`).
- Probing results jsonl at
  `experiments/phase7_unification/results/probing_results.jsonl`.
- Autointerp results jsonl at
  `experiments/phase7_unification/results/autointerp/<arch>/<concat>_labels.json`.
- Commits: one per arch family + ablation + each writeup
  (`<DATE>-<topic>.md`).
- No force-push to `han`. PR or fast-forward only after manual
  human verification.

### Files Phase 7 MUST NOT modify

To preserve historical reproducibility:

- Phase 5's `experiments/phase5_downstream_utility/results/ckpts/`
  (Phase 5 agent's untouched ckpt set).
- Phase 5's `probing_results.jsonl`, `training_index.jsonl`.
- Phase 6's `experiments/phase6_qualitative_latents/results/`.
- Phase 5B's `experiments/phase5b_t_scaling_explore/results/`.
- The 36-task probe cache for Gemma-IT (read-only reference; new
  Gemma-base cache lives at a separate path).

Phase 7 writes ONLY to `experiments/phase7_unification/`.

### What this phase will NOT do

- Train new architectures beyond the canonical set.
- Modify the SAEBench-style probing protocol (top-k-by-class-sep +
  L1 LR is unchanged).
- Run autointerp protocol changes beyond Phase 6's protocol (rerun
  on new ckpts, but no protocol changes).
- Cross-token tasks (winogrande, wsc) get the FLIP convention as
  before — no change.
- Train on layer ≠ 12 (anchor) or != L10–L14 (MLC).
- Use `last_position` as a separate metric in the headline. Reported
  only as a caveat-laden footnote in the paper, if at all.
- Reproduce Phase 6's other 3 Pareto plots (top-32, top-64, top-128).
  Top-256 is the deliverable.

### Coordination protocol between Agent A and Agent B

- **Day 1**: both agents branch `han-phase7-unification` independently.
  Both pull arch files via cherry-pick. Agent A starts cache build;
  Agent B starts passage build + Phase 6 pipeline port.
- **Day 2-3**: Agent A pushes activation cache + probe cache to HF.
  Agent B pulls activation cache (for passage encoding).
- **Day 3-5**: Agent A starts pushing trained ckpts to HF as they
  complete. Agent B polls HF for new ckpts; for each new ckpt at
  seed=42, runs the autointerp pipeline immediately.
- **Day 5-6**: Agent A finishes all 3-seed trainings + sparse-
  probing pass. Agent A pushes final `probing_results.jsonl`.
- **Day 6**: Agent B catches up on autointerp for any seed=42 ckpts
  not yet processed. Pulls Agent A's final probing jsonl. Generates
  Pareto plot.
- **Day 7-9**: write-up phase. Both agents draft their respective
  sections of the unified summary; merge into one
  `phase7_unification/summary.md` near deadline.
