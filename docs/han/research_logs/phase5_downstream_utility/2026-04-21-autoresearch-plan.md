---
author: Han
date: 2026-04-21
tags:
  - proposal
  - in-progress
---

## Phase 5.7 autoresearch — brave archs first, then tune winners

Handover document for the post-compact agent. Replaces T17 (seed
variance) with **agent-driven architectural exploration** on the top-3
SAEs (TXCDR-T5, MLC, time_layer_crosscoder_t5). Deadline: NeurIPS is
**~14 days out** from 2026-04-21.

### Key decision — do brave ideas FIRST, tune afterwards

Earlier draft put hyperparam tuning ahead of new-arch exploration.
Reversed after user feedback: the upside of a new architectural win
(e.g., Ye et al. InfoNCE gave +0.8 pp, beat our entire weight-sharing
ladder) dwarfs the upside of tuning an existing arch. Brave
exploration has asymmetric payoff.

### Why scrap T17

T17 was a 3-seed variance study on 5 existing archs — useful for
publishing "we ran multiple seeds" but did not test new ideas.
Phase 5.7 instead uses the compute budget to *discover* architectural
improvements. Seed variance can be added as a final sanity step on
whatever winners emerge.

### Current state (2026-04-21 ~14:00)

- **Committed up to `b976ee6`** on branch `han` (includes this doc's
  first draft).
- **T17 seed 1 fully complete** (5 archs trained + probed, committed
  at `9b429ec`). Ckpts retained for reference.
- **T17 seed 2 training**: all 5 archs' state dicts saved to disk
  (`{time_layer,mlc,txcdr_rank_k,txcdr_t5,txcdr_tied}_crosscoder_t5__seed2.pt`),
  probing rows not yet emitted. Ckpts retained.
- **Memory**: cgroup sits at ~45 GB (page cache at limit; 7–8 GB
  actual Python RSS; failcnt=0 — not a concern).

### Step 0 — kill T17 before starting

```bash
pkill -f "run_overnight_phase5.sh"
pkill -f "train_primary_archs"
pkill -f "run_probing.py"
# Verify:
ps -ef | grep run_ | grep -v grep
```

### CRITICAL: fairness rules

Scope: these rules apply **within the Phase-5.7 autoresearch cohort**
(TXCDR-T5, MLC, time_layer_crosscoder_t5). The existing 25-arch
canonical benchmark (summary.md) stays FROZEN at matched defaults —
we do NOT re-train the other 22 archs at any new setting. The paper
reports two tables side-by-side:

- **Canonical benchmark** (25 archs × matched defaults) — apples-to-
  apples architecture comparison, unchanged from summary.md.
- **Tuned leaders** (each top arch × its best config after auto-
  research) — shows how far each top arch can be pushed. Each row
  in this table discloses its full hyperparameter block
  (d_sae, k, steps, lr, special losses).

Within-cohort rules:

1. **d_sae is FIXED at per-arch defaults** during Part A:
   - TXCDR-T5: 18432
   - MLC: 18432
   - time_layer_crosscoder_t5: 8192

   d_sae moves only in the **Part-B capacity study** (one-shot 3-way
   fair bump).
2. **Sparsity `k` and training length CAN be bumped per-arch** in
   Part B. New arch candidates in Part A inherit the base arch's
   d_sae and k unless the arch has its own legitimate reason to
   differ (e.g., a rotational decoder with K extra params).
3. **Arch-internal hyperparams have no fairness issue** — internal
   to that arch. Examples: `mlc_contrastive`'s `α` contrastive
   weight; Matryoshka H/L partition sizes; rotational-decoder's Lie
   algebra rank.
4. **Disclose everything** in the leader table: d_sae, k,
   training_steps, lr, special losses.

### Val / test split (no peeking rule)

Per task: 3040 train → **train' = 2432** + **val = 608** (80/20,
deterministic seed from `dataset_key`). Test (760) untouched.

- **During autoresearch**: probe fits on train' (2432), reports
  **val AUC** in autoresearch_index.jsonl. Agent iterates on val.
- **For finalists**: probe fits on full train (3040), reports **test
  AUC** at BOTH `last_position` + `mean_pool`.
- Tag new JSONL rows with `aggregation=last_position_val` so they
  don't collide with existing `last_position` rows (full-train).

Implementation: extend `run_probing.py` with
`--aggregation last_position_val` and `mean_pool_val`.

### Part A — brave architectural exploration (priority; ~8-10 days)

Tier 1 ideas (highest upside, build first):

| # | name | idea | est. train |
|---|---|---|---|
| A1 | `txcdr_rotational_t5` | Rotational / Lie-group decoder: `W_dec^(t) = exp(t·A) W_base` with A skew-symmetric. Cayley transform avoids matrix exp. **Directly operationalises "feature direction rotates across time"** — interpretable angular-velocity parameter. brief.md §3.4 called this "most novel candidate, publishable on its own if it works". | ~30 min |
| A2 | `txcdr_contrastive_t5` | TXCDR-T5 + Ye et al. InfoNCE on (t−1, t) paired windows + Matryoshka H/L. Combines our two strongest wins: TXCDR-T5 (best at mean_pool) + mlc_contrastive's +0.8 pp contrastive lift. Highest empirical prior. | ~35 min |
| A3 | `matryoshka_txcdr_contrastive_t5` | Position-nested Matryoshka TXCDR + InfoNCE. Current `matryoshka_t5` (0.749) was plain Matryoshka without contrastive — **re-explore with contrastive added**; the nested-prefix structure gives contrastive an obvious H/L assignment. | ~35 min |
| A4 | `mlc_temporal_t3` | MLC extended across 3 adjacent tokens: input `(B, 3, L, d)`, shared encoder weights across tokens, joint decode. Novel combination of MLC's layer-axis and TXCDR's temporal-axis sharing. | ~45 min |
| A5 | `txcdr_basis_expansion_t5` | `W_dec^(t) = Σ_k α_k(t) · W_base_k` with K<<T shared basis matrices and learned (or fixed-sinusoidal) time coefficients. Smoother than full per-position decoder; more capacity than shared decoder. brief.md §3.4. | ~30 min |

Tier 2 (clever variants, build as Tier 1 results come in):

| # | name | idea | est. train |
|---|---|---|---|
| A6 | `txcdr_film_t5` | FiLM per-position modulation: `W_dec^(t) = diag(g_t) W_base diag(h_t)`. Cheapest parameterized decoder — captures scale-only variation. If this matches vanilla TXCDR-T5, "rotation" isn't needed. | ~25 min |
| A7 | `txcdr_smoothness_t5` | Vanilla TXCDR + soft decoder smoothness penalty `Σⱼ Σₜ (1 − cos(W_dec[j,t,:], W_dec[j,t+1,:]))`. One coefficient. brief.md §3.4. Cheap test of "do we need slowly-varying decoders?". | ~30 min |
| A8 | `txcdr_dynamics_t5` | Dynamics-based encoder: `z_{t+1} = g(z_t, x_{t+1})` — features evolve via learned latent dynamical system instead of independent re-computation. brief.md §4. Novel and tricky; save until Tier 1 signals direction. | ~50 min |
| A9 | `matryoshka_feature_idx_t5` | Feature-index Matryoshka (prefix of latents reconstructs at successive quality levels — standard Matryoshka-SAE). Brief's §1 lists it as the other Matryoshka axis; we only tried position-nested so far. | ~30 min |
| A10 | `time_layer_contrastive_t5` | `time_layer_crosscoder_t5` + InfoNCE on (t−1, t) layer-slabs. Tests whether contrastive generalises to time×layer joint latents. Direct parallel to mlc_contrastive. | ~50 min |

Tier 3 (wildcards, build only if Tier 1/2 plateau):

| # | name | idea | est. train |
|---|---|---|---|
| A11 | `txcdr_jumprelu_t5` | JumpReLU activation (per-feature learnable threshold) instead of TopK. Different sparsity family; Gated-SAE-adjacent. | ~30 min |
| A12 | `txcdr_batchtopk_t5` | BatchTopK: sparsity over `(batch × position)` jointly instead of per-window. Different inductive bias — rare features can activate in bursts. | ~30 min |
| A13 | `txcdr_group_topk_t5` | Partition d_sae into G groups, TopK within each. Prevents all-active-features-in-one-semantic-cluster. | ~30 min |
| A14 | `mlc_aux_ortho` | MLC with decoder-orthogonality regularizer `Σⱼ≠ₖ (cos(W_dec[j], W_dec[k]))²`. Encourages feature diversity. | ~12 min |
| A15 | `txcdr_hybrid_tfa_t5` | TFA pred head (dense, long-range) feeding into TXCDR sparse per-window decoder. brief.md §4. Most composable of the "hybrid" ideas. | ~50 min |

**Part A execution**: build A1 first, then A2 (both can reuse
`temporal_contrastive_sae.py` + `mlc_contrastive.py` as references).
Probe on val. Commit + push after each. Once results are in, pick the
most promising 2-3 directions and build Tier 2 members that
complement them. Don't commit to building all 15 — the brief-inspired
spirit is "follow your nose".

Each candidate ships as a **triple** when the axis is arch-agnostic
(e.g., A2's contrastive-addition idea makes sense for all 3 top
archs, so we'd probe mlc_contrastive (existing), txcdr_contrastive,
time_layer_contrastive together to keep within-cohort fairness). Tier
1 items flagged as TXCDR-specific (rotational, basis-expansion) only
apply to TXCDR because they modify its temporal decoder structure.

**Candidate scoring**: at plateau, compute `Δ_val = val_AUC(new) −
val_AUC(base)`. Flag as finalist if `Δ_val > +0.010` (1 pp). Discard
if `Δ_val < -0.020` (2 pp loss). Retry at 50k steps without
plateau-stop if `-0.020 < Δ_val < +0.010` (ambiguous zone). No more
than 2 retries per candidate.

### Part B — tune winners + final eval (2-4 days)

Once Part A identifies 2-4 finalist archs, Part B does targeted
hyperparam tuning. **NOT** an exhaustive grid — we only need enough
to convince ourselves the candidate isn't constrained by a bad
default.

For each finalist, in order:

1. **Training-budget check**: 25k → 50k steps without plateau-stop.
   If no further improvement, mark 25k as sufficient.
2. **Sparsity check**: k → 2×. If val AUC improves, try 4×. Stop on
   first non-improving point.
3. **Learning-rate check**: lr → 1e-4 and 1e-3 (vs default 3e-4). Pick
   best.
4. **Arch-internal micro-tune** (if applicable): mlc_contrastive α ∈
   {0.05, 0.1, 0.5}; rotational A rank ∈ {4, 8, 16}; etc.

This is ~4 hyperparam points per finalist × ~30-50 min each = ~3-4 h
per finalist. With 3 finalists, Part B = ~12 h of compute + finalisation.

**Fair d_sae capacity study** (one shot, 3 archs): `txcdr_t5`, `mlc`,
`time_layer_crosscoder_t5` all at 2× their defaults (36864 / 36864 /
16384). Report as standalone 3-row comparison. ~2–3 h. Do this
alongside Part B, only if it adds to the story.

**Final test-set evaluation**: for each finalist at its best config,
probe on full-train (3040) + held-out test (760) at BOTH
`last_position` + `mean_pool`. Compare against the canonical bench's
entries for the base arch. Tuned-leaders table in summary.md.

### Infrastructure

Files to touch:

- `experiments/phase5_downstream_utility/probing/run_probing.py`:
  add val/test split support. New `--aggregation` choices
  `last_position_val`, `mean_pool_val`. After loading the task
  cache, split `train_acts` / `train_labels` / `train_last_idx`
  deterministically seeded from `dataset_key`. Use train' to fit
  the probe, evaluate on val.
- `src/architectures/<new_arch>.py`: one file per Part-A class.
  Models to copy as templates:
  - `src/architectures/crosscoder.py` (vanilla TXCDR) for A1, A5, A6, A7, A8.
  - `src/architectures/mlc_contrastive.py` for A2, A3, A10 (contrastive pattern).
  - `src/architectures/mlc.py` for A4.
  - `src/architectures/txcdr_variants.py` for weight-sharing-like variants.
  - `src/architectures/matryoshka_txcdr.py` (PositionMatryoshkaTXCDR) for A3, A9.
- `experiments/phase5_downstream_utility/train_primary_archs.py`: add
  dispatcher branches for each new arch (roughly 10 lines each:
  import, build model, make generator, call `_iterate_train`, save).
- `experiments/phase5_downstream_utility/run_autoresearch.sh`: simple
  sequential orchestrator. `pgrep`-wait to serialize. Commit+push
  after each candidate.
- `experiments/phase5_downstream_utility/results/autoresearch_index.jsonl`:
  one row per candidate with `(name, hyperparams, val_auc_mean,
  val_auc_std, delta_vs_base, elapsed_s, notes)`.
- `experiments/phase5_downstream_utility/plots/plot_autoresearch.py`:
  val AUC over candidates (ordered by run time).

### Tracking / output files

Each candidate emits:
- `results/ckpts/<cand>__seed42.pt` (~850 MB).
- `results/training_logs/<cand>__seed42.json`.
- `probing_results.jsonl` rows tagged `aggregation=last_position_val`
  (during autoresearch) and later `last_position` + `mean_pool`
  (for finalists on test set).
- `autoresearch_index.jsonl` summary row.
- `logs/overnight/autoresearch_<cand>.log`.

### Tier 1 results (run 2026-04-21, seed 42, last_position_val, k=5)

All five Tier 1 candidates completed. Baseline probes for txcdr_t5,
matryoshka_t5, mlc now present at `last_position_val` (full 36 tasks).

| candidate | verdict | Δ_val | t | wins/losses |
|---|---|---|---|---|
| A3 matryoshka_txcdr_contrastive_t5 | **FINALIST** | +0.0155 | +1.46 | 22/12 |
| A2 txcdr_contrastive_t5 | **FINALIST** | +0.0120 | +1.17 | 24/10 |
| A1 txcdr_rotational_t5 | DISCARD | −0.0332 | −3.12 | 12/23 |
| A5 txcdr_basis_expansion_t5 | DISCARD | −0.0448 | −4.15 | 9/26 |
| A4 mlc_temporal_t3 | DISCARD | −0.0615 | −3.54 | 7/27 |

**Clean pattern (5/5 agreement)**:
- InfoNCE on adjacent-latent pairs (A2, A3) → both FINALIST.
- Decoder-constraint / weight-sharing (A1, A5, A4) → all DISCARD.

Working hypothesis: at 25 k steps with d_sae=18432 we are
capacity-limited, not structure-limited. Adding soft structural
priors to the decoder hurts; adding an auxiliary contrastive signal
that the model is free to shape helps.

### Tier 2 shelving decision (2026-04-21)

Given the Tier 1 pattern, the 5 plan items are reprioritised:

**Tried next:**
- **A10 `time_layer_contrastive_t5`** — adds InfoNCE to
  `time_layer_crosscoder_t5` (already top-5 at last_position).
  Direct generality test of the InfoNCE win. Highest prior.
- **A8 `txcdr_dynamics_t5`** — recurrent sparse latent
  `z_{t+1} = TopK(γ·z_t + W_enc·x_{t+1})`. Constrains the ENCODER /
  LATENT TRAJECTORY rather than decoder, so orthogonal to A1/A5/A4.
  Shares family with contrastive (both enforce "adjacent latents
  should agree") — contrastive does it as a loss, dynamics as an
  architecture. Aligned with the winning signal. Plan had it tagged
  "save until Tier 1 signals direction" — signal now present.

**Shelved (predicted-loss or low-prior, reasoning):**
- **A6 `txcdr_film_t5`** — FiLM is a *more* restrictive
  decoder-parameterization than A1 (rotational) or A5 (basis-K=3).
  3/3 decoder-constraint ideas already DISCARD; skipping saves ~25 min
  of predictable negative result.
- **A7 `txcdr_smoothness_t5`** — soft cosine penalty on adjacent
  `W_dec[j,t]` columns. Same axis as A1/A5, softer knob. Same
  capacity argument applies. If latent-level smoothness (A2/A3
  InfoNCE) is the right knob, weight-level smoothness is probably
  still the wrong one.
- **A9 `matryoshka_feature_idx_t5` (standalone)** — feature-nested
  Matryoshka is same capacity as existing `matryoshka_t5` (position-
  nested), just a different partition. Standalone version ≈
  matryoshka_t5. Interesting compound is A9+contrastive, which we'd
  revisit only if we wanted to test the "feature-nested axis"
  independently; but A3 already validates the
  "Matryoshka+contrastive" pattern, so the marginal value is low.
  Shelved unless A10/A8 motivate revisiting.

**Tier 3 remains fully parked.**

### Open questions / decisions for the next agent

1. **Start with A1 (`txcdr_rotational_t5`) or A2 (`txcdr_contrastive_t5`)?**
   A2 has higher empirical prior (InfoNCE already gave +0.8 pp on
   MLC); A1 has higher novelty. Recommendation: **A2 first** (lower
   implementation risk), **A1 second**.
2. **Building new arch classes**: most are 100–250 line files. Use
   `mlc_contrastive.py` as reference for the contrastive-loss
   pattern; use `txcdr_variants.py` for the decoder-parameterization
   pattern. Test each on a single task first before launching full
   36-task probe.
3. **Memory**: cgroup stays near limit due to page cache; safe.
   Python RSS 7–8 GB on probing.
4. **Budget allocation**:
   - Part A Tier 1 (5 candidates): ~4 h training + 40 min probing
     each = ~20–25 h compute.
   - Part A Tier 2 (5 more): ~25 h.
   - Part B (3 finalists): ~12 h.
   - Part B capacity study: ~3 h.
   - Final eval: ~2 h.
   - Buffer: ~40 h.
   - **Total: ~100 h = ~4–6 days** of serial GPU. Plenty of buffer
     before deadline.
5. **Parallel execution**: single A40. Serialize via pgrep-wait.

### Resume checklist (for the post-compact agent)

1. `git log --oneline -n 10` — confirm at `b976ee6` or later.
2. Read this doc fully.
3. Read [`summary.md`](summary.md) (status of the 25-arch bench) and
   [`brief.md`](brief.md) §3.4 ("Decoder smoothness / parameterized
   decoder") and §4 ("Invent your own — open design axes").
4. Run Step 0 above to kill T17.
5. Implement val/test split in `run_probing.py` (~30 min).
6. Implement A2 (`txcdr_contrastive_t5`) — highest empirical prior.
   Test on 1-2 tasks first. Commit the arch class.
7. Launch A2 training + probing via `run_autoresearch.sh`. Commit + push
   result.
8. Based on A2 outcome: A1 (rotational) + A3 (matryoshka+contrastive)
   in parallel implementation, then launch.
9. Iterate through Part A Tier 1 (A1-A5), then Tier 2 as the tree
   branches.
10. When 2-4 finalists emerge, move to Part B.

### Hard resource constraints (unchanged)

- **Pod volume `/workspace`**: `df -h`; don't let free drop under
  5 GB. Each new ckpt ~850 MB; 15 candidates × 0.85 = ~13 GB. Fine.
- **Container disk** (`/tmp`, `/home`): 150 GB free; fine for scratch.
- **cgroup memory limit: 46 GB.** Page cache hugs limit; failcnt=0
  means safe. Python RSS 7–9 GB during probing.
- **Claude Code state**: /home/appuser/.claude symlinked to
  /workspace/claude_home. If pod is rebuilt, run
  `bash scripts/bootstrap_claude.sh` to restore the symlink before
  launching Claude.

### Notes on re-exploring Matryoshka

The current `matryoshka_t5` (0.7494 last_position, 0.7747 mean_pool)
is **position-nested** (first m₁ latents reconstruct each position
alone; next m₂ extend to T=2 windows; etc.) without contrastive.
brief.md §3.1 explicitly flags this as "recommended primary" for 5.3
but the plain Matryoshka under-delivered. Three unexplored
combinations to try (A3, A9, and a dual variant):

1. **Position-nested Matryoshka + contrastive** (A3) — the nested
   prefix-structure gives a natural H/L assignment for Ye et al.
   InfoNCE.
2. **Feature-index Matryoshka** (A9) — prefix of latents
   reconstructs at successive quality. Different from our current
   position-nested version.
3. **Dual Matryoshka** (feature × position) — if A3 and A9 both
   show promise, combine.

Position-nested + contrastive (A3) is the brave-idea top pick.
