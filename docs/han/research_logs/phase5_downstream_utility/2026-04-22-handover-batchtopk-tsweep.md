---
author: Han
date: 2026-04-22
tags:
  - proposal
  - in-progress
---

## Handover: full-size TFA + BatchTopK + T-sweep + TXC/MLC router

**Audience**: post-compact agent picking up four deferred follow-on
experiments after Phase 5.7 closed (morning brief + agentic log
committed at `326394b`).

**Ordering constraint from Han**: full-size TFA **MUST** be done
**before** the BatchTopK experiment (experiment (ii) below). The
reason: the current bench uses `tfa_small` (d_sae=4096, seq_len=32)
which is a capacity-underpowered comparison against the d_sae=18 432,
seq_len=128 archs. Reviewers (and `summary.md` Caveats) already
flag this; closing the gap is a prerequisite before adding new
cross-sparsity comparisons that would otherwise inherit the same
unfairness.

**Latest relevant commits on `han`** (all pushed to github):

- `326394b` — bar charts sorted by score + coloured by family
- `a0eb6b6` — docs: close loop on agentic winners + test-set follow-up + HF URLs
- `c12e3d9` — dual TFA probing + regenerated plots
- `3abb1a5` — agentic loop complete + morning brief + seed variance

**Paper state**: Phase 5.7 bench has 29 entries (25 original + 2
agentic winners + 2 Part-B winners). Multi-scale contrastive with
γ=0.5 decay is the family-agnostic recipe (works on TXC & MLC).
Numbers in `summary.md` Figure 1 (last_position) + Figure 2 (mean_pool)
are apples-to-apples (same probe era, last-write-wins dedup).

**Your job**: two independent experiments that shore up the paper's
defensibility. Do them sequentially on single GPU. Pick the order
based on which reviewer critique worries you more.

### Context you'll want to read first

- [`summary.md`](summary.md) — 29-entry bench, dual TFA, agentic winners subsection
- [`2026-04-22-morning-brief.md`](2026-04-22-morning-brief.md) — what happened in the overnight agentic loop + afternoon test-set follow-up
- [`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md) — all 8 agentic cycles, per-cycle hypotheses + results
- [`2026-04-21-phase5_7-architectures.md`](2026-04-21-phase5_7-architectures.md) — architecture reference including agentic cycles section at bottom

### Winning architectures (for reference)

Both use **α=1.0 outer contrastive weight, γ=0.5 decay, 3 contrastive scales**:

- **TXC**: `src/architectures/matryoshka_txcdr_contrastive_multiscale.py`
  → `MatryoshkaTXCDRContrastiveMultiscale(d_in, d_sae, T, k_eff, n_contr_scales=3, gamma=0.5)`
  - Applies InfoNCE at the first 3 matryoshka sub-window scales
- **MLC**: `src/architectures/mlc_contrastive_multiscale.py`
  → `MLCContrastiveMultiscale(d_in, d_sae, n_layers, k, prefix_lens=(d_sae//4, d_sae//2, d_sae), gamma=0.5)`
  - Applies InfoNCE at 3 d_sae prefix lengths

Dispatchers:
- `train_primary_archs.py` branches for `agentic_txc_02` and `agentic_mlc_08`
- `run_probing.py` encode routing mirrors

### Test-set AUC 3-seed baseline (for reproducibility checks)

If you retrain these at seed=42, you should land within ~0.005 of:

| arch | last_position | mean_pool |
|---|---|---|
| agentic_txc_02 | 0.7775 | 0.8007 |
| agentic_mlc_08 | 0.8047 | 0.7890 |

---

## Experiment (i): Full-size TFA [DO BEFORE EXPERIMENT (ii)]

**Purpose**: the current bench lists `tfa_small` / `tfa_pos_small` at
d_sae=4096, seq_len=32 — significantly smaller than every other bench
arch at d_sae=18 432, seq_len=128. This is flagged explicitly in
[`summary.md`](summary.md) Caveats section:

> **TFA "small" variants (d_sae = 4096, seq_len = 32).** The full-size
> TFA would not fit the A40 wall-clock budget without a significant
> refactor… SAEBench numbers are therefore not a like-for-like
> comparison against the d_sae = 18 432 archs.

Full-size TFA closes this gap. After the overnight audit showed TFA
probing conventions matter (dual entries `tfa_*` vs `tfa_*_full`),
the fair comparison is full-size TFA at both probing variants, then
compare against the matched-capacity (d_sae=18 432) bench archs.

**Scope**: 2 new archs:

1. `tfa_big` — use_pos=False, d_sae=18 432, seq_len=128
2. `tfa_pos_big` — use_pos=True, d_sae=18 432, seq_len=128

Plus the `*_full` dual probing is automatic via the arch-suffix
routing added on 2026-04-22 (see `run_probing.py::_load_model_for_run`
alias logic). Creating symlinks
`tfa_big_full__seed42.pt -> tfa_big__seed42.pt` after training gives
you the dual probe out of the box — same pattern as the small
variants.

**Implementation plan**:

1. **Confirm `train_tfa` helper already supports d_sae and seq_len kwargs.**
   Inspect `experiments/phase5_downstream_utility/train_primary_archs.py::train_tfa`.
   If it already accepts `d_sae` and `seq_len`, you only need dispatcher
   branches. If not, extend the helper to pass through to
   `src/architectures/_tfa_module.py::TemporalSAE` constructor.

2. **Add 2 dispatcher branches** in `train_primary_archs.py`:

   ```python
   elif arch == "tfa_big":
       model, log = train_tfa(cfg, device, k=100, use_pos=False,
                              d_sae=18432, seq_len=128,
                              buf=get_anchor())
       meta = dict(seed=seed, k_pos=100, k_win=None, T=128,
                   use_pos=False, d_sae=18432, layer=13,
                   scale=log.get("input_scale", 1.0))
   elif arch == "tfa_pos_big":
       model, log = train_tfa(cfg, device, k=100, use_pos=True,
                              d_sae=18432, seq_len=128,
                              buf=get_anchor())
       meta = dict(seed=seed, k_pos=100, k_win=None, T=128,
                   use_pos=True, d_sae=18432, layer=13,
                   scale=log.get("input_scale", 1.0))
   ```

3. **Probe routing extension** in `run_probing.py`:
   - `_load_model_for_run` tfa branch already handles
     `d_sae_eff = meta.get("d_sae", d_sae)` — no change.
   - Add `tfa_big` + `tfa_pos_big` (plus `_full` aliases) to the
     `arch in (...)` tuples in BOTH `_load_model_for_run` and
     `_encode_for_probe`.

4. **Launch training**. First check VRAM headroom — at d_sae=18 432
   and seq_len=128, the TFA model's pre-activation + bottleneck attention
   tensors are ~4.5× larger in d_sae and ~4× larger in seq_len than
   the small variant, so roughly **~18× more memory per forward**
   than tfa_small. The small variant used ~6 GB peak; the big variant
   projects to ~25-35 GB peak on A40 (46 GB total). Should fit but
   check with batch_size=256 first, scale down if OOM:

   ```bash
   PYTHONPATH=/workspace/temp_xc TQDM_DISABLE=1 PYTHONUNBUFFERED=1 \
     .venv/bin/python -u -c "
   from experiments.phase5_downstream_utility.train_primary_archs import run_all
   run_all(seeds=[42], max_steps=25000, archs=['tfa_big','tfa_pos_big'])
   " > logs/overnight/tfa_big_train.log 2>&1 &
   ```

5. **If OOM**, drop d_sae to 8192 first (halfway between small=4096
   and big=18432). Retrain, probe, document the compromise in
   summary.md Caveats.

6. **After training, create symlinks for the dual-probing `_full`
   variants** (matches the pattern already used for tfa_small):

   ```bash
   cd experiments/phase5_downstream_utility/results/ckpts
   ln -sf tfa_big__seed42.pt tfa_big_full__seed42.pt
   ln -sf tfa_pos_big__seed42.pt tfa_pos_big_full__seed42.pt
   ```

7. **Probe all 4 run_ids** (`tfa_big`, `tfa_big_full`, `tfa_pos_big`,
   `tfa_pos_big_full`) at both `last_position` and `mean_pool`:

   ```bash
   for AGG in last_position mean_pool; do
     PYTHONPATH=/workspace/temp_xc TQDM_DISABLE=1 \
       .venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
       --aggregation $AGG --skip-baselines \
       --run-ids tfa_big__seed42 tfa_big_full__seed42 \
                 tfa_pos_big__seed42 tfa_pos_big_full__seed42
   done
   ```

8. **Update `summary.md`**:
   - Add 4 new rows to Figure 1 (last_position) + Figure 2 (mean_pool),
     keeping the dual `_full` convention established for tfa_small.
   - Rewrite the "TFA small variants" Caveats entry to note that a
     full-size comparison is now available.
   - Keep the small-TFA rows for continuity (historical reference).

9. **Regenerate plots**: add `tfa_big`, `tfa_big_full`, `tfa_pos_big`,
   `tfa_pos_big_full` to `ORDERED_ARCHS` in `plots/make_headline_plot.py`
   (the `flavor_of` function will map them to "tfa" flavor
   automatically via the `tfa_` prefix). Regenerate via
   `.venv/bin/python plots/make_headline_plot.py`.

10. **Commit per-arch** via the orchestrator (`run_autoresearch.sh`),
    or manually if you skipped the orchestrator.

**Time estimate**: `tfa_small` took 155s to plateau at 4000 steps.
At d_sae=18 432 × seq_len=128 the per-step cost scales ~4.5× × 4× ≈
18×, so roughly **45 min per run** (2 archs → 90 min).
Probing 4 run_ids × 2 aggregations at ~8 min each = **~65 min**.
Total: **~2.5 hr wall-clock**. Add 1-2 hr buffer for VRAM debugging.

**Known risk**: VRAM. TFA's n_attn_layers=1 bottleneck attention scales
quadratically in seq_len. If batch_size=256 OOMs, drop to 128. If
still OOMing, drop d_sae to 8192 and document.

**Deliverable**: 4 new rows in both bench tables; updated plots; Caveats
entry rewrite. This is what Han asked for: proper-sized TFA in the
bench before any new capacity-sensitive experiments (i.e. BatchTopK).

---

## Experiment (ii): BatchTopK apples-to-apples

**Purpose**: the bench currently uses TopK sparsity everywhere. SAEBench
reviewers will ask about BatchTopK (Bussmann et al. 2024) — which picks
top B·k across the pooled (batch, d_sae) tensor rather than top k
per sample. Known to give better reconstruction at matched average
sparsity because samples can flex their per-sample budgets.

**Question**: does our headline ranking (multi-scale recipe wins across
families) survive under BatchTopK? Or is the TXC/MLC story a
TopK-specific artefact?

**Minimum scope** (defensible): 4 archs at seed=42 × {last_position, mean_pool}:

1. `txcdr_t5_batchtopk` — BatchTopK applied to vanilla TXCDR
2. `agentic_txc_02_batchtopk` — BatchTopK applied to the multi-scale TXC winner
3. `mlc_batchtopk` — BatchTopK applied to vanilla MLC
4. `agentic_mlc_08_batchtopk` — BatchTopK applied to the multi-scale MLC winner

Compare the pair-wise deltas:
- Δ_TXC_batchtopk = agentic_txc_02_batchtopk − txcdr_t5_batchtopk
- Δ_TXC_topk = agentic_txc_02 − txcdr_t5 (already measured: 0.7775 − 0.7752 = +0.0023 at last_position;
  0.8007 − 0.7991 = +0.0016 at mean_pool)

If Δ_batchtopk ≈ Δ_topk, the multi-scale recipe composes with BatchTopK.
If Δ_batchtopk > Δ_topk, it *composes synergistically* — even stronger paper story.
If Δ_batchtopk < 0 (BatchTopK breaks the recipe), that's important to know.

**Full scope** (for comprehensive Figure 1 update): 7 archs — add
`topk_sae_batchtopk`, `matryoshka_t5_batchtopk`, `mlc_contrastive_batchtopk`
so the bench has a clean "TopK vs BatchTopK" column for every key arch.
But this triples the comparison work without adding much over the minimum.

**Implementation plan**:

1. **Implement BatchTopK module** (~50-80 lines, new file
   `src/architectures/_batchtopk.py`):
   ```python
   class BatchTopK(nn.Module):
       """BatchTopK sparsity: pick top B*k across the (B, d_sae) pre-activation.

       At training time: compute pre-activation, flatten, keep top B*k,
       zero the rest. Track running quantile of activation magnitudes
       for inference-time thresholding.
       At inference time: apply a threshold derived from the running
       quantile (JumpReLU-style) to produce a sparse activation pattern
       that matches training average sparsity.

       Args:
           k: per-sample sparsity budget (pooled across batch)
           momentum: EMA for inference threshold (default 0.99)
       """
       def __init__(self, k, momentum=0.99):
           super().__init__()
           self.k = k
           self.momentum = momentum
           self.register_buffer("threshold", torch.tensor(0.0))

       def forward(self, pre_act):
           # pre_act: (B, d_sae)
           if self.training:
               B, d = pre_act.shape
               flat = pre_act.reshape(-1)
               top_vals, _ = flat.topk(B * self.k)
               cutoff = top_vals.min()
               out = torch.where(pre_act >= cutoff, F.relu(pre_act), torch.zeros_like(pre_act))
               with torch.no_grad():
                   self.threshold = self.momentum * self.threshold + (1 - self.momentum) * cutoff
               return out
           else:
               return torch.where(pre_act >= self.threshold, F.relu(pre_act), torch.zeros_like(pre_act))
   ```

   Unit test: on a known tensor, verify that `BatchTopK(k=5)` produces
   approximately B·5 non-zeros and the threshold tracks sensibly.

2. **Factor sparsity into the SAE classes.** Easiest: add a
   `sparsity_mechanism` parameter to the `__init__` of:
   - `TopKSAE` in `src/architectures/topk_sae.py`
   - `MultiLayerCrosscoder` in `src/architectures/mlc.py`
   - `TemporalCrosscoder` in `src/architectures/txcdr.py`
   - `PositionMatryoshkaTXCDR` in `src/architectures/matryoshka_txcdr.py`
   When `sparsity_mechanism="batchtopk"`, use `BatchTopK(k)` in place of
   the in-line `topk + scatter` pattern. Default remains `"topk"`.

3. **Dispatcher branches** in `train_primary_archs.py` for
   `txcdr_t5_batchtopk`, `mlc_batchtopk`, `agentic_txc_02_batchtopk`,
   `agentic_mlc_08_batchtopk`. Each passes `sparsity_mechanism="batchtopk"`
   to the underlying constructor.

4. **Probe routing** in `run_probing.py` for the 4 new archs.
   `_load_model_for_run` reads `meta["sparsity_mechanism"]` and passes
   to constructor. `_encode_for_probe` is unchanged (encoder API same).

5. **Launch via `run_autoresearch.sh`** as usual — add `BASE_OF` entries:
   - `BASE_OF[txcdr_t5_batchtopk]=txcdr_t5` (same-mechanism check — should
     be close to txcdr_t5's 0.7752)
   - `BASE_OF[agentic_txc_02_batchtopk]=txcdr_t5_batchtopk` (measure the
     recipe's gain on top of the BatchTopK baseline — the key comparison)
   - Ditto for MLC.

6. **Probe at `last_position` and `mean_pool`** (test-set, not `_val`).

7. **Add rows to `summary.md` Figure 1 + Figure 2** and regenerate plots.

**Time estimate**: 3-4 hr engineering + 3 hr train+probe + 1 hr
write-up = **~7-8 hr wall-clock for the minimum 4-arch version**.
Double if you add seed variance on the agentic winners (recommended
for paper quality — pattern in `experiments/phase5_downstream_utility/agentic/seed_variance.sh`).

**Known risk**: BatchTopK's inference threshold calibration. The
training-to-inference gap (batch stats → per-sample threshold) is the
published weakness. If the JumpReLU-style threshold gives bad
reconstruction at inference, probe AUCs will look artificially low.
Sanity check: run forward on a training batch in eval mode and
confirm the sparsity rate is close to `k/d_sae`. If not, bump the
momentum up or use a quantile-based threshold (pick the `(d_sae - k)/d_sae`
quantile of training activations).

---

## Experiment (iii): T-sweep on agentic_txc_02

**Purpose**: extend the existing Figure 3 T-sweep (`txcdr_t{2, 3, 5, 8, 10, 15, 20}`)
to the multi-scale winner. The vanilla sweep showed T=5 is the peak
for `txcdr_*` at last_position and mean_pool. Does multi-scale
contrastive shift that peak? Answers: "is T=5 a property of the base
arch, or a property of our specific multi-scale config at that T?"

**Scope**: 7 new archs:
`agentic_txc_02_t{2, 3, 8, 10, 15, 20}` (+ `agentic_txc_02` itself
already exists at T=5). Note: the base `agentic_txc_02` has
`n_contr_scales=3`, `γ=0.5` hard-coded at T=5. For the sweep, keep
`γ=0.5` fixed and set `n_contr_scales = min(3, T)` so we gracefully
handle T=2 (only 2 scales available).

**Training cost per T** (rough, based on the existing T-sweep + cycle-02
runtime of ~37 min at T=5):
- T=2: ~20 min
- T=3: ~25 min
- T=5: existing (agentic_txc_02) — skip
- T=8: ~45 min
- T=10: ~55 min
- T=15: ~75 min
- T=20: ~95 min

**Implementation plan**:

1. **Add 6 dispatcher branches in `train_primary_archs.py`**:

   ```python
   elif arch in ("agentic_txc_02_t2", "agentic_txc_02_t3",
                 "agentic_txc_02_t8", "agentic_txc_02_t10",
                 "agentic_txc_02_t15", "agentic_txc_02_t20"):
       T = int(arch.removeprefix("agentic_txc_02_t"))
       n_scales = min(3, T)
       model, log = train_matryoshka_txcdr_contrastive_multiscale(
           cfg, device, k=100, T=T, alpha=1.0,
           n_contr_scales=n_scales, gamma=0.5, buf=get_anchor(),
       )
       meta = dict(seed=seed, k_pos=100, k_win=100 * T, T=T,
                   match_budget=True, layer=13, alpha=1.0,
                   n_contr_scales=n_scales, gamma=0.5,
                   variant="agentic_txc_02_t_sweep")
   ```

2. **Add the same 6 names to the `agentic_txc_02` branch in `run_probing.py`**
   (both the `_load_model_for_run` and `_encode_for_probe` sections that
   already list `agentic_txc_02`).

3. **Add `BASE_OF` entries in `run_autoresearch.sh`** mapping each to
   `matryoshka_t5` (the vanilla family base). For cross-T comparison
   against the original T-sweep, the comparison arch is `txcdr_t{N}`
   at matched T — you can compute this cross-comparison in the summariser
   by hand; the autoresearch_index.jsonl default just needs ONE base.

4. **Launch all 6 via `run_autoresearch.sh`**:
   ```bash
   bash experiments/phase5_downstream_utility/run_autoresearch.sh \
       agentic_txc_02_t2 agentic_txc_02_t3 agentic_txc_02_t8 \
       agentic_txc_02_t10 agentic_txc_02_t15 agentic_txc_02_t20 \
       > logs/overnight/tsweep_agentic.log 2>&1 &
   ```

5. **Probe at last_position + mean_pool**. (The orchestrator probes
   `last_position_val` by default; you need to extend to both test-set
   aggregations explicitly, patterning after
   `experiments/phase5_downstream_utility/agentic/test_set_eval.sh`.)

6. **Update `summary.md` T-sweep table** (around line 216, "T-sweep
   ladder (TXCDR at T ∈ {2, 3, 5, 8, 10, 15, 20})"). Add a parallel
   row for `agentic_txc_02` at each T. Or side-by-side columns for
   `txcdr_t{N}` vs `agentic_txc_02_t{N}` to make the recipe-vs-vanilla
   delta immediate.

7. **Regenerate plots**: add T-sweep archs to `ORDERED_ARCHS` in
   `plots/make_headline_plot.py` if you want them in Figure 1/2 (they
   share the `agentic_txc_*` flavor prefix, so the family-colour
   logic will handle them automatically). Regenerate via
   `.venv/bin/python plots/make_headline_plot.py`.

**Time estimate**: 30 min engineering + ~5 hr training + ~1.5 hr
probing = **~7 hr wall-clock for the full sweep at seed=42**.

**Expected outcome (my guess)**: the multi-scale recipe's peak will
match vanilla TXCDR's T=5 peak. Why: the γ=0.5 decay + n=3 scales
config was tuned at T=5 where matryoshka scales {1, 2, 3} cover 60%
of the sub-windows. At T=20 with n=3 fixed, only 15% of scales get
contrastive — the tail scales {4..20} train via reconstruction only,
same as vanilla. So at large T, the gain should shrink; the *optimal
T for the recipe* may be slightly smaller than vanilla's T=5 if
multi-scale pushes info into earlier scales. Cycle 04 already showed
n=5 (all scales) hurts at T=5, so expanding n proportionally with T
is unlikely to help.

**Known risk**: low. The recipe is well-tested at T=5, and T-variants
of vanilla `txcdr_t*` are well-established. If a specific T fails to
train (OOM, NaN), check the per-window k_eff — at T=20, k_eff = 100 × 20
= 2000 which is 11% of d_sae and might interact weirdly with matryoshka's
nested prefixes. If so, reduce k to 50 for T=20 runs and document.

---

## Experiment (iv): TXC ↔ MLC task router

**Purpose**: the bench currently shows `agentic_txc_02` and
`agentic_mlc_08` each win at their "home" aggregation — TXC at
mean_pool, MLC at last_position (see `summary.md` Figure 1/2). The
implicit "they're complementary" claim is soft: it's based on
different overall rankings and the earlier Jaccard error-overlap
number (0.34). This experiment *concretizes* the complementarity
story: build a simple learned classifier that routes each task to
either TXC or MLC, and show this router dominates either arch alone.

**The headline claim to test**: on the 36-task bench,
`route(task) → {TXC, MLC}` achieves a mean AUC greater than *either
arch's individual mean AUC on all 36 tasks*. Three numbers to report:

1. **Best individual**: max(mean(agentic_txc_02), mean(agentic_mlc_08))
   per aggregation.
2. **Oracle router** (upper bound): for each task, pick whichever arch
   has higher test_auc on that task. This is the ceiling for any
   router given these two archs.
3. **Learned router**: predict winner from task features via k-fold
   CV; report effective AUC = mean across folds of "use the
   predicted-winner arch's test_auc per held-out task".

If (3) ≳ (1), you've concretized the complementarity. If (3) is near
(2), routing is learnable and close to oracle. If (3) ≈ (1), routing
from task features alone isn't learnable and we fall back to the
Jaccard error-overlap argument for complementarity.

**Why this is fast**: no retraining, no GPU. Pure post-hoc analysis
of the existing `probing_results.jsonl` + task metadata. Implementable
in a single afternoon.

**Implementation plan**:

1. **Extract per-task AUCs** for both archs at both aggregations
   (seed=42, k=5, last-write-wins) — reuse the pattern in
   `experiments/phase5_downstream_utility/agentic/compute_bench_table.py`
   or adapt the `_val_per_task` helper from `partB_summarise.py`.
   Store as a 36×2 table per aggregation.

2. **Compute oracle upper bound**:
   ```python
   oracle_auc = per_task[["agentic_txc_02", "agentic_mlc_08"]].max(axis=1).mean()
   ```

3. **Define task features** — keep simple, since n=36 tasks. Three
   feature sets to try in order of simplicity:
   - **F1 (metadata)**: dataset source one-hot (15 sources: ag_news,
     amazon_reviews, bias_in_bios_set{1,2,3}, europarl, github_code,
     language_id, mmlu, winogrande, wsc, …), label-type indicator
     (binary / multi-class), average example length. 15-20 features
     per task.
   - **F2 (activation summary)**: per-task mean + std of scale-1
     agentic_txc_02 latents and of MLC agentic_mlc_08 latents on the
     task's train set (768-dim — 2 × 2 × d_sae/T //? actually keep
     small: just a few summary statistics like L0, mean activation
     magnitude, top-k sparsity). Lets the router see "how sparse /
     active is each arch on this task".
   - **F3 (difficulty proxy)**: the per-arch AUC at train time minus
     a simple baseline (e.g., majority-class AUC). This leaks target
     information if used naively; only use for a sanity check.
   Start with F1. If F1 gives a learnable signal, stop.

4. **Fit a classifier** with 6-fold CV on 36 tasks (6 tasks per fold).
   Start with `sklearn.linear_model.LogisticRegression(C=1.0, penalty="l2")`.
   Target = `argmax_arch(test_auc_per_task)`. Report:
   - CV accuracy (how often the router picks the right arch on
     held-out tasks)
   - Effective AUC = mean-across-folds of
     `per_task[predicted_arch, held_out_task]`

5. **Also compute the concat-latent probing baseline** — a separate
   way to "use both archs at once", complementary to routing:
   - Concatenate agentic_txc_02 and agentic_mlc_08 latents per task
     into a (n, 2·d_sae) tensor.
   - Run the standard top-k-by-class-separation probe (k=5) on the
     concatenated tensor.
   - Compare resulting mean AUC to (1), (2), (3). If concat-probing
     > routing, the "use both" flavor with shared probe is the
     stronger claim. If routing > concat, task-level assignment is
     the right granularity.

   The `run_probing.py` pipeline supports arbitrary latent
   concatenation — most effort is in constructing the concat tensor
   from two separate encoded outputs. See the existing error-overlap
   analysis in `experiments/phase5_downstream_utility/analysis/`
   (specifically `error_overlap_summary_last_position_k5.json`) for
   the join-keys pattern on (task, sample) pairs.

6. **Run for both aggregations** (last_position + mean_pool). Expect
   each aggregation to give a different routing pattern since TXC
   dominates mean_pool and MLC dominates last_position.

7. **Write up the numbers as a small table in `summary.md`** —
   something like:

   | aggregation | best individual | oracle router | learned router | concat probe |
   |---|---|---|---|---|
   | last_position | 0.8047 (MLC) | TBD | TBD | TBD |
   | mean_pool | 0.8007 (TXC) | TBD | TBD | TBD |

   Add a 1-paragraph interpretation: which claim survives.

**Time estimate**: 30 min feature extraction + 15 min CV + 1 hr
concat-probing + 1-2 hr writeup/figures = **~3 hr wall-clock**.
No GPU required for routing; concat-probing wants ~15 min on GPU.

**Known risks**:
- **Only 36 samples**. Logistic regression with 15-20 features + 36
  samples has high variance. CV accuracy may be near 50% even if
  oracle is meaningfully above best individual. That's an *honest*
  finding: routing needs more tasks to be learnable.
- **Dataset-source one-hot could overfit**. If a single source has
  only 1-2 tasks and one arch wins there, the router memorizes that
  signal. Guard by per-source stratified CV if feasible.
- **Oracle may be only marginally above best individual**. If
  agentic_mlc_08 wins 30/36 tasks at last_position, oracle pick is
  close to agentic_mlc_08 alone. Check the per-task win distribution
  before over-investing — if one arch dominates 80%+ of tasks, the
  routing story is weak for THAT aggregation but may still work at
  the other.

**Deliverable**: a new section in `summary.md` titled
"Complementarity: TXC/MLC routing and concat probing" with the
numbers table, a per-task bar plot showing winner breakdown, and a
1-paragraph conclusion.

**Pointers**:
- Existing error-overlap analysis:
  `experiments/phase5_downstream_utility/results/error_overlap_summary_last_position_k5.json`
- Per-task AUC extraction pattern:
  `experiments/phase5_downstream_utility/agentic/compute_bench_table.py::load_last_write`
- Task list + dataset sources:
  `experiments/phase5_downstream_utility/results/probe_cache/` (36 task dirs)

---

## What to do first

**Hard constraint from Han**: experiment (i) full-size TFA **must
precede** experiment (ii) BatchTopK. Rationale in the opening
ordering note — you can't do capacity-sensitive cross-sparsity
experiments while leaving an under-capacity TFA in the bench.

**Recommended order**:

1. **(iv) TXC/MLC router** (~3 hr) — shortest, no GPU, no training,
   concretizes the "they're complementary" story that's currently
   soft in the paper. Do this early while longer experiments queue.
2. **(i) Full-size TFA** (~2.5 hr + buffer) — closes the
   d_sae-mismatch unfairness, enables every subsequent comparison
   to include fair TFA numbers.
3. **(iii) T-sweep on agentic_txc_02** (~7 hr) — lowest-risk
   exploration, extends the existing Figure 3 cleanly, independent
   of TFA work.
4. **(ii) BatchTopK** (~7-9 hr) — highest engineering cost, defensive
   against SAEBench reviewers; run last when (i)'s full-TFA numbers
   are in the bench so BatchTopK arch comparisons are built on fair
   capacity.

**Alternative ordering if time-pressed**: (iv) → (i) → (ii), skip (iii).
You'd still have a fully fair bench with BatchTopK for reviewer
defense plus the complementarity story; the T-sweep is "nice to
have" completeness for the paper's recipe claim.

**Total wall-clock for all four**: ~20-22 hr (≈ one full overnight
session + morning). If you do (iv) + (i) + (iii): ~13 hr. If you do
(iv) + (i) + (ii): ~13-15 hr.

## Gotchas

1. **Pgrep defensive check** in `run_autoresearch.sh` requires Python
   commands NOT shells. Fixed at commit `009292a` — don't regress.
2. **Probing non-determinism ~0.005 on mean-of-36**. When comparing
   new BatchTopK entries to existing TopK ones, the diff of 0.005 is
   within noise. Trust only diffs ≥ 0.01 without re-probing the
   comparison arch in the same era.
3. **Orchestrator commits per-candidate** — each arch trains, probes,
   summarises, commits+pushes. Don't batch commits manually.
4. **Don't touch `tfa_*` / `tfa_*_full`** unless testing TFA. The
   current split convention (novel vs novel+pred) is load-bearing.
5. **Seed variance pattern**: see `experiments/phase5_downstream_utility/agentic/seed_variance.sh`
   for the template if you want 3-seed mean on BatchTopK winners.
6. **HF sync** after ckpt additions: run
   `HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_ckpts.py`.
   Safe to re-run — upload_folder skips unchanged files by hash.

## Resume checklist (first 15 min)

1. `git log --oneline | head -5` — confirm at `326394b` or later.
2. Read this doc + the 4 referenced docs above (order: summary.md,
   morning-brief, agentic-log, architectures).
3. `ls experiments/phase5_downstream_utility/results/ckpts/ | wc -l`
   — should be ~65 ckpts. If not, something's wrong.
4. `ps -ef | grep -E "(train_primary|run_probing|run_autoresearch)"`
   — should be empty (nothing in flight).
5. `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv`
   — GPU should be idle.
6. Start with **(iv) TXC/MLC router** (fast, CPU-only, concretizes the
   complementarity story) or **(i) full-size TFA** (hard ordering
   constraint — must precede (ii) BatchTopK). Can do (iv) first on
   CPU while (i) trains on GPU. Then (iii) T-sweep or (ii) BatchTopK,
   per the "what to do first" ordering.
7. Implement, launch, monitor, update summary.md + regenerate plots,
   commit+push per experiment.
