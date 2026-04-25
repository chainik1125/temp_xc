---
author: Han
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Handover: push TXC qualitative performance via agentic autoresearch

**Audience**: Phase 6 agent (you) picking up where
[[summary]] left off.

**The problem to solve**: our proposed TXC variant
(`agentic_txc_02`) is **the paper's headline architecture** and
**top-of-class on Phase 5 downstream sparse probing** (0.80 AUC at
mean_pool). But on Phase 6's qualitative axes it **lost to everything
else**:

| arch | autointerp semantic / 8 |
|---|---|
| `tsae_paper` | 6 |
| `tfa_big` | 6 |
| `agentic_mlc_08` | 5 |
| `tsae_ours` | 3 |
| **`agentic_txc_02`** | **2** |

TXC's top features read as punctuation / delimiter-only. Passage
smoothness absent (`concat_B__agentic_txc_02__top8.png`). Alive
fraction 0.39 vs tsae_paper's 0.73.

For the paper's story ("a new TXC arch that's best at sparse probing
AND qualitatively interpretable"), we need to close the gap — or at
least get TXC to ≥ 5 / 8 semantic labels while preserving its Phase 5
AUC. This briefing lays out an agentic autoresearch loop, modelled on
Phase 5.7's 8-cycle loop (see
[`2026-04-21-agentic-log.md`](../phase5_downstream_utility/2026-04-21-agentic-log.md)),
to push TXC as far as possible on the qualitative metric.

### Diagnosis — why tsae_paper wins

From Phase 6's `summary.md`, `tsae_paper` doesn't win because its
*loss function* is better. It wins because of **a stack of anti-dead-
feature machinery** (see
[`src/architectures/tsae_paper.py`](../../../src/architectures/tsae_paper.py)):

1. **Auxiliary-K loss** (paper App. B.1): parallel reconstruction from
   the top `aux_k` DEAD features — this is what sustains alive
   fraction ≳ 0.73. Without it, `tsae_ours` (same loss family, no
   AuxK) gets 30 %.
2. **Geometric-median `b_dec` init**: non-zero init centred on a data
   sample (not the origin). Avoids initial dead-feature cascade.
3. **Unit-norm decoder constraint** enforced each step by `set_decoder_norm_to_unit_norm`.
4. **Decoder-parallel gradient removal** (`remove_gradient_parallel_to_decoder_directions`):
   project out the component of `∇W_dec` parallel to each column
   before step, so the unit-norm constraint doesn't shrink the update.
5. **BatchTopK + EMA-threshold inference**: per-sample sparsity is
   variable, which seems to help feature diversity.
6. **Lower k (20 vs our 100)**: sparser representations → more
   feature-per-concept.

All six of these are orthogonal to the contrastive loss structure.
**Most of them are also directly applicable to TXC and MLC.** The
agentic loop below ports them one at a time, measuring each cycle's
autointerp delta.

### The secondary hypothesis — contrastive window / chain length

Han's intuition: the cycle-02 recipe on TXC uses pair (length-2,
shift-by-1) contrastive windows, matching T-SAE. The T-SAE paper
*speculated* longer chains but didn't find gain on single-token SAEs.
**TXC is structurally different**: at T=5 the same token appears in
5 different windows, so "semantic continuity across windows" has
richer chain structure than "semantic continuity across tokens".

Plausible reason TXC failed qualitatively: with only pair contrastive,
scale-1 latents learn to be *shift-invariant* (because adjacent
T-windows share T−1 positions), but they don't learn to be
*concept-coded* — any feature that's roughly constant across the
window satisfies the InfoNCE objective, including position-indexed
or punctuation-indexed features. Longer chains (3, 5 windows)
progressively constrain scale-1 latents to be passage-level rather
than position-level.

Include this as one of the agentic cycles (H-QUAL5 below).

### Metric: autointerp semantic score

**Primary**: number of the top-8 features (by activation variance on
`concat_A` + `concat_B` combined, 1819 tokens) that the Claude Haiku
autointerp pipeline assigns a **semantic concept label**
(non-punctuation, non-position). Pipeline:
[`run_autointerp.py`](../../../experiments/phase6_qualitative_latents/run_autointerp.py).
Cost ≈ $1 per arch per run.

**Secondary**:

- **Passage smoothness** on `concat_B` — qualitative inspection of
  `plot_top_features.py` output. Do top features show clear passage-
  boundary transitions? Figure-4-style.
- **Alive fraction** on 16 k held-out fineweb tokens. Current
  `agentic_txc_02` = 0.39; target ≥ 0.65.
- **Paper Table 1 metrics** via
  [`arch_health.py`](../../../experiments/phase6_qualitative_latents/arch_health.py):
  FVE, CosSim, S(H).

**Sparse-probing guard**: run the trained candidate through
[`experiments/phase5_downstream_utility/probing/run_probing.py`](../../../experiments/phase5_downstream_utility/probing/run_probing.py)
at test-set `last_position` + `mean_pool` with `seed=42, k_feat=5`.
Candidate must hold within 0.01 of `agentic_txc_02`'s 0.78 / 0.80 to
count as a paper-defensible win. If a cycle's autointerp +3 comes
with a sparse-probing −0.02, that's net useful only if we reframe
the paper's claim.

### Phase 6.1 agentic autoresearch — proposed cycles

Ordered by **expected impact × ease of implementation**. Each cycle
is a ~45-min train + 5-min autointerp cost. Run up to 10 cycles
total; pattern: 1 hypothesis → 1 code change → 1 eval → 1 takeaway.

#### Cycle A — Port AuxK loss onto TXC (HIGHEST PRIORITY)

Subclass [`MatryoshkaTXCDRContrastiveMultiscale`](../../../src/architectures/matryoshka_txcdr_contrastive_multiscale.py)
as `MatryoshkaTXCDRContrastiveMultiscaleAuxK`:

- Track per-feature `last_fired_step` (as in [`tsae_paper.py`](../../../src/architectures/tsae_paper.py)).
- At forward, compute dead mask (last_fired > threshold steps ago);
  gate pre-activation with dead mask, keep top `aux_k` (e.g. 512) of
  those DEAD features, reconstruct the RESIDUAL `x - x_hat`,
  mse-loss it, add to total.
- Expected delta: **+30 % alive fraction, +2 semantic labels**.
  AuxK is the single most impactful thing tsae_paper has.

#### Cycle B — Unit-norm decoder + grad-parallel removal

Add `set_decoder_norm_to_unit_norm()` + `remove_gradient_parallel_to_decoder_directions()`
hooks to the TXC training loop (copy from `tsae_paper.py`). Apply to
every `W_decs[t]` slice in the matryoshka hierarchy.

- Expected: +1 autointerp label, ~same AUC. Keeps decoder atoms
  disentangled.
- Coupled with Cycle A (AuxK implicitly assumes unit-norm decoders).

#### Cycle C — Geometric-median `b_dec` init

One-shot at step 0: compute geometric median of 1k fineweb
activations, init `b_dec[t] = median` for each t. Copy from
`tsae_paper.py` init path.

- Expected: +5 % alive at step 500, small long-run delta.

#### Cycle D — Lower k

Retrain cycle-02 recipe at `k=20` (matching `tsae_paper`). If AUC
holds, sparser representation likely yields more concept-coded
features.

- Caveat: k=20 × T=5 = window L0 of 100 vs current 500. Big drop
  in reconstruction capacity. Likely AUC regression; keep if the
  autointerp delta is > +2 labels.

#### Cycle E — Contrastive chain length > 2

Extend pair (length-2, shift-1) to **triple** (length-3, shift-1):
generate (W_{t−2}, W_{t−1}, W_t) as `(B, 3, T, d)`. Apply InfoNCE
with 3 anchors pairwise, γ-weighted:

    L_contr = InfoNCE(z_{t−1}, z_t) + γ · InfoNCE(z_{t−2}, z_t)
            + γ² · InfoNCE(z_{t−2}, z_{t−1})

Also consider:

- **Multi-distance positives**: use (W_t, W_{t+1}) and (W_t, W_{t+5})
  as two positive pairs — shift-1 enforces short-range consistency,
  shift-5 enforces passage-level consistency.
- **Triple with only length-3 positives** (no shift-1 intermediate):
  tests whether long-range consistency alone matters.

Pair-gen helper will need to be extended — see existing
[`make_pair_window_gen_gpu`](../../../experiments/phase5_downstream_utility/train_primary_archs.py)
line 341 as template.

- Expected: +1 to +3 autointerp labels if chain length really
  matters; 0 delta if the "pair is enough" T-SAE result transfers.

#### Cycle F — BatchTopK + threshold inference on TXC

Already know from Phase 5.7 experiment (ii) that BatchTopK regresses
TXC sparse probing. But it may help qualitative — per-sample variable
sparsity lets rare features fire when needed. Test autointerp only.

- Existing class: [`MatryoshkaTXCDRContrastiveMultiscaleBatchTopK`](../../../src/architectures/_batchtopk_variants.py)
  (from `han` branch, need merge).

#### Cycle G — Feature diversity penalty

Add L2 penalty on off-diagonal decoder Gram matrix
`‖W_decᵀW_dec − I‖²` across each matryoshka scale. Pushes features
apart in feature space.

- Expected: +1 autointerp label. Small effect, cheap.

#### Cycle H — All-of-the-above stack

Combine winners from Cycles A, B, C into
`MatryoshkaTXCDRContrastiveMultiscalePlus`. If cumulative autointerp
delta ≥ +4 labels (reaching `tsae_paper`'s 6/8), this becomes the
new paper headline TXC.

### Expected trajectory

| cycle | expected Δ autointerp | expected Δ alive |
|---|---|---|
| A: AuxK | +2 labels | +0.30 |
| B: decoder norm | +0.5 labels | +0.05 |
| C: geom-median init | ~0 (improved stability) | +0.05 |
| D: k=20 | +1 label (big AUC risk) | +0.10 |
| E: chain > 2 | +1 label | 0 |
| F: BatchTopK | +0 to +1 | +0.20 |
| G: diversity penalty | +0.5 | +0.05 |
| H: stack | +3-4 labels cumulative | +0.40 |

Realistic headline **target: 5/8 semantic labels at alive ≥ 0.6** —
matches `agentic_mlc_08` and closes 80 % of the gap to `tsae_paper`.

Stretch target: 6/8 semantic labels → ties `tsae_paper` / `tfa_big`.

### How to run each cycle

Template (mirror of Phase 5.7's loop):

1. **Write new class** in `src/architectures/`. Subclass
   `MatryoshkaTXCDRContrastiveMultiscale` whenever possible so the
   encode API stays identical and Phase 5 probing still works.

2. **Add dispatcher branch** for a new arch name like
   `agentic_txc_09_auxk` in
   [`train_primary_archs.py`](../../../experiments/phase5_downstream_utility/train_primary_archs.py).

3. **Train it** (same seed=42, 25k steps, plateau-stop):
   ```bash
   PYTHONPATH=. TQDM_DISABLE=1 .venv/bin/python -c "
   from experiments.phase5_downstream_utility.train_primary_archs import run_all
   run_all(seeds=[42], max_steps=25000, archs=['agentic_txc_09_auxk'])
   "
   ```

4. **Encode on concat-A + concat-B** via
   [`encode_archs.py`](../../../experiments/phase6_qualitative_latents/encode_archs.py)
   (add the new arch to its dispatch map).

5. **Run autointerp**:
   ```bash
   PYTHONPATH=. TQDM_DISABLE=1 .venv/bin/python \
     experiments/phase6_qualitative_latents/run_autointerp.py \
     --archs agentic_txc_09_auxk
   ```
   Reads per-arch z caches, selects top-8 by variance, sends top-10
   activating contexts to Claude Haiku, parses semantic/syntactic
   label, writes `results/autointerp/<arch>__labels.json`.

6. **Compute autointerp semantic count** + write the cycle result
   row:
   ```
   agentic_txc_09_auxk:  semantic 4/8 (+2 vs agentic_txc_02),
     alive 0.62 (+0.23), sparse-probe AUC 0.7750 (−0.0025) at mean_pool
   ```

7. **Log to `2026-04-23-agentic-log.md`** (create if missing) with
   hypothesis → change → result → takeaway, same format as
   [`2026-04-21-agentic-log.md`](../phase5_downstream_utility/2026-04-21-agentic-log.md).

8. **Commit + push per cycle** — don't batch.

9. **Update `summary.md`** incrementally as the autointerp count for
   each cycle lands.

### Implementation pointers

- Copy AuxK helper + dead-feature tracking from
  [`tsae_paper.py`](../../../src/architectures/tsae_paper.py) lines
  ~200-280 (look for `aux_k`, `last_k_fired`). That file is the
  load-bearing reference.
- The decoder norm hook is at
  `set_decoder_norm_to_unit_norm` (line ~54) + `remove_gradient_parallel_to_decoder_directions`
  (line ~66). Apply after `opt.step()` in the TXC train loop; call
  before the norm projection.
- For pair-gen extensions (Cycle E), pattern is in
  [`train_primary_archs.py:341`](../../../experiments/phase5_downstream_utility/train_primary_archs.py)
  — `make_pair_window_gen_gpu` is the template; a `make_triple_window_gen_gpu`
  is a ~20-line variant.

### What's already available from Phase 5 / Phase 6

On the `han` branch (needs merge into `han-phase6`):

- `MatryoshkaTXCDRContrastiveMultiscaleBatchTopK` class (Cycle F)
- All BatchTopK infrastructure for TXC / MLC.

On `han-phase6`:

- `tsae_paper.py` — the reference stack of anti-dead machinery.
- `tsae_ours.py` — the failure-mode control; instructive comparison.
- Full autointerp pipeline + concat corpora + z_cache.

### Note on evaluation cost

- Training a cycle: ~45 min × ~0.5/hr compute = $0.40 if you're
  paying for A40.
- Autointerp: ~$1 per arch (160 features × 10 context × 2 call =
  ~3200 Haiku calls; Haiku 4.5 is ~$0.0003/call).
- 10 cycles: ~$5 + ~8 hr wall-clock if run serially.

Budget is not the constraint here; the constraint is how many
qualitative labels we can push TXC up the table.

### Deliverable

Updated [[summary]] Section 4 autointerp table with
`agentic_txc_XX` rows demonstrating the cumulative gain, a new
`2026-04-23-agentic-log.md` Phase 6.1 log, and (if the stacked
recipe works) a new final-winner arch
`agentic_txc_09_plus__seed42.pt` trained to plateau, synced to HF,
and added to the Phase 5 bench (to confirm the paper story: same
arch wins both sparse probing AND qualitative).

### Resume checklist (first 15 min)

1. `git log --oneline | head -10` — confirm at `437bcca` (final 5-arch
   silhouette table).
2. Read this doc + Phase 6 [[summary]] §3-5 (training metrics,
   autointerp, passage smoothness) to calibrate on current TXC failure.
3. Read [[2026-04-22-encoding-protocol]] (per-arch encoding conventions
   — your new TXC variants inherit this).
4. Read Phase 5 [`2026-04-21-agentic-log.md`](../phase5_downstream_utility/2026-04-21-agentic-log.md)
   for the hypothesis → change → result → takeaway format.
5. Run `arch_health.py --arch agentic_txc_02` to sanity-check that
   its reported alive=0.39 is reproducible on your pod.
6. Start with Cycle A (AuxK) — highest expected gain.
