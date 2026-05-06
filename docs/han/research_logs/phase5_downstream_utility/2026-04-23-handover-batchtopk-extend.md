---
author: Han
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Handover: extend BatchTopK to the remaining 14 bench archs

**Audience**: post-compact agent finishing Phase 5.7 experiment (ii).

**Context**: the initial BatchTopK run covered the minimum 4-arch
scope (`txcdr_t5`, `mlc`, `agentic_txc_02`, `agentic_mlc_08`) — enough
for the recipe-composition finding. Han wants the full bench to have
a clean "TopK vs BatchTopK" column: every key arch that appears in
Figure 1/2 should have a BatchTopK counterpart row. This extension
adds 14 more archs.

**Commits on `han` you're building on** (all pushed to GitHub):

- `185a7e8` — T-sweep (iii) partial (T=2,3,8 done; T≥10 OOMs)
- `61fa553` — BatchTopK minimum 4 archs + summary.md section
- `104734c` — full-size TFA (i) complete
- `0f71e14` — TXC/MLC router (iv) complete
- `b87f561` — ORDERED_ARCHS extended for all new archs

**Current Figure 1/2 state**: 38 TopK archs + 4 BatchTopK archs. After
this extension, Figure 1/2 will have 38 TopK + 18 BatchTopK = 56 rows
(14 new BatchTopK archs added this round).

### Scope — 14 archs to add

Grouped by existing class family so you know which subclass to reuse:

**Reuse `TemporalCrosscoderBatchTopK`** (already exists in
[`_batchtopk_variants.py`](../../../../src/architectures/_batchtopk_variants.py)):

- `txcdr_t2_batchtopk`, `txcdr_t3_batchtopk`, `txcdr_t8_batchtopk`,
  `txcdr_t10_batchtopk`, `txcdr_t15_batchtopk`, `txcdr_t20_batchtopk`
  (6 archs — just dispatch with different T)

**Reuse `MatryoshkaTXCDRContrastiveMultiscaleBatchTopK`**:

- `agentic_txc_02_t2_batchtopk`, `agentic_txc_02_t3_batchtopk`,
  `agentic_txc_02_t8_batchtopk` (3 archs — just set T + n_contr_scales)
- **DO NOT attempt `agentic_txc_02_t{10,15,20}_batchtopk`** — the
  underlying TopK variant already OOMs on matryoshka decoder memory
  (documented in summary.md §T-sweep). BatchTopK adds no headroom.

**New subclasses needed** (5, each ~10-15 lines — pattern in
[`_batchtopk_variants.py`](../../../../src/architectures/_batchtopk_variants.py)):

1. **`TopKSAEBatchTopK(TopKSAE)`** → `topk_sae_batchtopk`
   - `encode`: `pre = x_c @ W_enc.T + b_enc`; replace TopK-scatter with
     `self.sparsity(pre)`.
2. **`PositionMatryoshkaTXCDRBatchTopK(PositionMatryoshkaTXCDR)`** →
   `matryoshka_t5_batchtopk`
   - `encode`: same einsum as TXCDR but parent uses the matryoshka
     prefix structure — just swap the TopK step.
3. **`MLCContrastiveBatchTopK(MLCContrastive)`** →
   `mlc_contrastive_batchtopk` (and its α=1.0 variant; both use the
   same class, passed α at train time)
4. **`TimeLayerCrosscoderBatchTopK(TimeLayerCrosscoder)`** →
   `time_layer_crosscoder_t5_batchtopk`
   - NB: TimeLayerCrosscoder uses a **global** TopK over the flat
     (T·L·d_sae) grid. BatchTopK should pool across (B, T·L·d_sae)
     for consistency. Flatten pre-act to `(B, T*L*d_sae)` before
     calling `BatchTopK(k_total)`.
5. **`StackedBatchTopK`** → `stacked_t5_batchtopk`, `stacked_t20_batchtopk`
   - Stacked SAE is T independent SAEs sharing a buffer. Simplest:
     subclass + BatchTopK on each per-position flattened pre-act.

**Part-B α=1.0 winners**:

- `matryoshka_txcdr_contrastive_t5_alpha100_batchtopk` (subclass of
  `MatryoshkaTXCDRContrastive` — needs a 6th new subclass, or reuse
  the multi-scale one with α=1.0 and n_contr_scales=1)
- `mlc_contrastive_alpha100_batchtopk` (reuse `MLCContrastiveBatchTopK`
  from item 3 above — same class, α=1.0 at train time)

**Not doing** (documented in summary.md Caveats, skip):

- 8 discarded TXCDR variants (`txcdr_{tied, shared_dec, shared_enc,
  pos, causal, block_sparse, lowrank, rank_k}_t5`) — these were
  shelved as DISCARD/AMBIGUOUS; adding BatchTopK rows wouldn't change
  any paper claim.
- TFA variants (`tfa_{big,pos_big,small,pos_small}`) — TopK is
  hard-coded inside `_tfa_module.py` (`sae_diff_type="topk"`). Would
  need a new `sae_diff_type="batchtopk"` branch at line ~232, which is
  invasive (not a subclass job). Explicitly defer.
- `temporal_contrastive` — single-token SAE that lost the 25-arch
  bench (last_position 0.73); not competitive, skip.

### Implementation plan

**Step 1 — Add subclasses** to
[`src/architectures/_batchtopk_variants.py`](../../../../src/architectures/_batchtopk_variants.py).
5-6 new classes, each ~10-15 lines. Pattern:

```python
class PositionMatryoshkaTXCDRBatchTopK(PositionMatryoshkaTXCDR):
    def __init__(self, d_in, d_sae, T, k):
        super().__init__(d_in, d_sae, T, k)
        if k is not None:
            self.sparsity = BatchTopK(k)
    def encode(self, x):
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is None: return F.relu(pre)
        return self.sparsity(pre)
```

CPU smoke test each new class (pattern in earlier session's commit
`f896dbf`): instantiate with small d_in/d_sae, forward a random batch
in train() mode, check `z.nonzero().item() == B*k`. Took <30s for the
first 4 subclasses.

**Step 2 — Add dispatcher branches** in
[`experiments/phase5_downstream_utility/train_primary_archs.py`](../../../../experiments/phase5_downstream_utility/train_primary_archs.py).
Copy the existing `txcdr_t5_batchtopk` / `mlc_batchtopk` dispatchers
(around line 1296) as templates. Each branch:

1. Builds the BatchTopK variant via new subclass.
2. Uses the same gen fn as the TopK base (e.g. `make_window_gen_gpu`
   for TXCDR, `make_pair_window_gen_gpu` for contrastive, etc.).
3. Sets `meta["sparsity"] = "batchtopk"`.

For the 6 `txcdr_t{N}_batchtopk`: wrap in a single `elif` with
`T = int(arch.removeprefix("txcdr_t").removesuffix("_batchtopk"))`.

For the 3 `agentic_txc_02_t{N}_batchtopk`: same prefix-strip pattern.

**Step 3 — Add probe routing** in
[`experiments/phase5_downstream_utility/probing/run_probing.py`](../../../../experiments/phase5_downstream_utility/probing/run_probing.py).
Two places to edit:

1. `_load_model_for_run` (~line 357): one branch per new arch family
   that instantiates the BatchTopK variant.
2. `_encode_for_probe` (~line 611): add the new arch names to the
   existing `arch in (...)` tuples — encoder API is identical, so
   same `_encode_*` function works.

For the 6 `txcdr_t{N}_batchtopk`: add to the
`txcdr_t2, txcdr_t3, ...` tuple in `_load_model_for_run`, and make
`_encode_txcdr` match on `arch.startswith("txcdr_t")` (already does).

**Step 4 — Add `BASE_OF` entries** in
[`run_autoresearch.sh`](../../../../experiments/phase5_downstream_utility/run_autoresearch.sh).
Pattern: each `*_batchtopk` maps to its TopK counterpart.

**Step 5 — Write a launcher script** —
`run_batchtopk_extend.sh`, copy `run_batchtopk.sh` template,
replace `CANDS` with the 14 new arch names (grouped small-to-large
to fail-fast on any OOM). Use `max_steps=25000` with plateau-stop.

**Step 6 — Launch + monitor**:

```bash
nohup bash experiments/phase5_downstream_utility/run_batchtopk_extend.sh \
    > logs/overnight/run_batchtopk_extend_main.log 2>&1 &
```

Arm a background waiter on the wrapper PID (pattern used in prior
session) so you get notified on pipeline completion.

**Step 7 — Update `summary.md`**:

- Insert 14 new rows into Figure 1 (last_position) + Figure 2
  (mean_pool) tables using the sort-by-AUC pattern. Mark each with 🆕
  and suffix-note the base arch it compares against.
- Extend the "BatchTopK apples-to-apples" section with the extended
  result table. The current section shows 4 Δ rows; grow it to 14
  for the full comparison.
- Key question to answer in the writeup: does the "MLC multi-scale
  composes / TXC multi-scale doesn't" finding generalise? Specifically,
  check whether `matryoshka_t5_batchtopk` vs `matryoshka_txcdr_contrastive_t5_alpha100_batchtopk`
  preserves the matryoshka-contrastive gain. If it does, the
  TopK-dependence is specific to the multi-scale mechanism, not to
  contrastive more broadly. If it doesn't, the story is "contrastive
  TXC generally needs TopK".

**Step 8 — Regenerate plots** via
`.venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py`.
The 14 new arch names are already in `ORDERED_ARCHS` from commit
`b87f561`... actually no — only the 4 first BatchTopK variants are
there. Add the 14 new names to `ORDERED_ARCHS` before regenerating.

**Step 9 — Commit per milestone** (train-half, probe-half, writeup-half)
to keep git history granular. Then `HF_HOME=/workspace/hf_cache
.venv/bin/python scripts/hf_upload_ckpts.py` to sync the 14 new ckpts
(skip-by-hash, re-runnable).

### Time estimate

Per-arch average: ~45 min train + ~5 min probe = ~50 min.
14 archs × 50 min = **~12 hr wall-clock**.

Engineering: ~1-1.5 hr for subclasses + dispatchers + probe routing
(6 new subclasses, but most are ~10 lines).

Total: **~13-14 hr**, doable overnight.

### Known risks

1. **`agentic_txc_02_t{10,15,20}_batchtopk` are off-limits.** The
   underlying matryoshka OOMs at T≥10 at d_sae=18432 (documented in
   summary.md §T-sweep). BatchTopK doesn't help — model params
   dominate memory, not activations.

2. **BatchTopK threshold calibration**. The inference-time EMA
   threshold takes a few 100 steps to stabilize. If any arch's
   BatchTopK probing result looks *way* off its TopK baseline (>5 pp
   regression), sanity-check: run the encoder forward in eval mode
   on a training batch and confirm average per-sample sparsity is
   close to `k/d_sae`. If it's drifting (<50% of expected sparsity),
   either extend training, or manually set `model.sparsity.threshold`
   from a calibration pass on fresh data before saving the ckpt.

3. **`time_layer_crosscoder_t5_batchtopk`**. The base arch uses a
   *global* TopK over (T·L·d_sae). BatchTopK should pool across (B,
   T·L·d_sae) — flatten before `BatchTopK(k_total)`. Double-check the
   forward shapes match; the `BatchTopK` module expects 2D input.

4. **Non-determinism ~0.005 on mean-of-36**. When comparing a new
   BatchTopK entry to its TopK counterpart, treat diffs < 0.005 as
   noise, not signal.

### Resume checklist (first 15 min)

1. `git log --oneline | head -5` — confirm at `185a7e8` or later.
2. Read
   [`summary.md`](summary.md) §"BatchTopK apples-to-apples" to see the
   current 4-arch finding you're extending.
3. Read
   [`2026-04-22-handover-batchtopk-tsweep.md`](2026-04-22-handover-batchtopk-tsweep.md)
   §"Experiment (ii)" for the original scope + BatchTopK module code.
4. Inspect
   [`src/architectures/_batchtopk_variants.py`](../../../../src/architectures/_batchtopk_variants.py)
   and see the 4 existing subclasses as templates.
5. `ps -ef | grep -E "(train_primary|run_probing)"` — expect empty.
6. `nvidia-smi` — GPU should be idle.
7. Start with the 6 easy `txcdr_t{N}_batchtopk` (reuse existing class),
   then the 3 `agentic_txc_02_t{N}_batchtopk` (also reuse), then write
   the 5-6 new subclasses for the remaining archs.

### Deliverable

14 new rows in Figure 1/2 tables, extended "BatchTopK apples-to-apples"
section with the full 18-arch comparison table, regenerated plots,
all committed + pushed + HF-synced.
