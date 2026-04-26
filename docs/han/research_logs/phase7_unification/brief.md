---
author: Han
date: 2026-04-25
tags:
  - design
  - in-progress
---

## Phase 7 brief — unification and methodological consolidation

### Why this phase exists

Across Phase 5, 5B, 6, 6.2, 6.3 we have accumulated ~60+ trained
architectures, ~3 leaderboards, two branches diverged from `han`
(`han-phase5b` and `han-phase6`), a deprecated aggregation
(`full_window`), and three known methodological weaknesses:

1. **Subject model is `google/gemma-2-2b-it`**, not `google/gemma-2-2b`
   (base). The base model is what T-SAE (Ye et al. 2025) and TFA
   (Lubana et al. 2025) train on. Using IT introduces a confound that
   reviewers will flag and that prevents direct comparison to
   pretrained Neuronpedia / Gemma-Scope SAEs.

2. **Sparsity convention varies across families.** Phase 5's
   convention has been `k_pos = 100, k_win = k_pos × n_active_positions`
   for window archs but MLC was an anomaly at `k_win = 100` globally.
   The probe sees representations of differing sparsity, biasing
   comparisons in ways that conflate "feature quality" with "feature
   count."

3. **Cluttered leaderboard with two aggregations and ambiguous
   framing.** `last_position` and `mean_pool` were treated as
   independent metrics, but lp implicitly assumes a "the z represents
   the last token" framing that is architecturally unjustified for
   TXC (the encoder is symmetric over T positions; nothing privileges
   the last one). After working through this with the user, we
   consolidate to ONE leaderboard.

NeurIPS deadline is 2026-05-05 (10 days from now). Phase 7 is the
final pass: re-run the most informative archs on a sanitized setup
that addresses (1)–(3), present a single unified leaderboard, and
write the paper.

### Execution model: two parallel RunPod agents

This brief is written for the agents (the human who designed it is
not executing). Phase 7 is split into two parallel workstreams, each
run by an autonomous agent on its own RunPod:

- **Agent A — sparse-probing leaderboard.** Runs on an **H200 pod**
  (141 GB GPU, 188 GB RAM, 12 vCPUs, 1 TB volume). Owns the
  activation cache rebuild, training of all 47 archs × 3 seeds,
  ckpt upload to HF, and the long-tail sliding mean-pool sparse-
  probing leaderboard. The H200 enables larger batch sizes plus the
  T_max ∈ {64, 128} SubseqH8 cells that wouldn't fit on H100 80GB.
- **Agent B — qualitative autointerp.** Runs on an **H100 pod**
  (80 GB GPU, 125 GB RAM, 8 vCPUs, 1 TB volume). Owns the Phase
  6-style qualitative pipeline (concat-A/B/random passages, top-k
  feature selection, autointerp via Claude Haiku, Pareto analysis).
  Pulls trained ckpts from HF (no independent training).

Per-agent guidance on how to maximally leverage the hardware
(probe-cache pre-loading, joblib parallelism, GPU/CPU pipelining,
batch size, concurrent Haiku API calls, etc.) is in
`plan.md` §Resource environment.

**Shared resources:**

- HuggingFace repo `han1823123123/txcdr` at `phase7_ckpts/` —
  Agent A pushes, Agent B pulls.
- HF for the rebuilt Gemma2B-base activation cache and probe cache
  (Agent A pushes; Agent B can pull if useful, or rebuild
  independently for the qualitative passages).

**Sequential dependency:** Agent B can't start autointerp until
Agent A's ckpts are on HF. Agent B can do day-1 prep (passage
building, Phase 6 pipeline port) in parallel with Agent A's
training.

### Where to find prior-phase code and docs

The Phase 7 branch (`han-phase7-unification`) carries forward the
**research logs and phase-specific experiment directories** from
prior phases so agents do NOT need to switch branches to read prior
work. Available directly on this branch:

| dir | from phase | what's there |
|---|---|---|
| `docs/han/research_logs/phase5_downstream_utility/` | Phase 5 | brief, plan, dated experiments, summary, agentic logs, groundbreaking handover |
| `docs/han/research_logs/phase5b_t_scaling_explore/` | Phase 5B | brief, plan, message-to-phase5-agent, t-encoder writeup |
| `docs/han/research_logs/phase6_qualitative_latents/` | Phase 6.1, 6.3 | qualitative interpretability + Pareto analysis |
| `docs/han/research_logs/phase6_2_autoresearch/` | Phase 6.2 | TXC-cannot-close-the-qualitative-gap negative result |
| `docs/han/research_logs/phase7_unification/` | Phase 7 (this) | brief, plan |
| `experiments/phase5_downstream_utility/` | Phase 5 | training driver, probing pipeline, results jsonls (read-only reference) |
| `experiments/phase5b_t_scaling_explore/` | Phase 5B | subseq sampling + 2D sweep code, results jsonls |
| `experiments/phase6_qualitative_latents/` | Phase 6 | qualitative pipeline + Pareto plot generation |
| `experiments/phase6_2_autoresearch/` | Phase 6.2 | autoresearch loop scripts |

Plus all the arch class files (`src/architectures/*.py`) including
those that aren't in the Phase 7 canonical 49 (e.g. dead-end Phase
5B negative results: `phase5b_strided_txcdr.py`,
`phase5b_token_subseq_sae.py`, `phase5b_subset_encoder_txc.py`,
`phase5b_per_pos_scale_matryoshka.py`).

The full historical branches (`origin/han`, `origin/han-phase5b`,
`origin/han-phase6`) ARE preserved and contain phase-specific
experiment infrastructure (training loops, ckpts metadata, etc.)
that wasn't migrated. If you find yourself needing something not on
Phase 7's branch, that's where to look — e.g. Phase 5's
`run_partB_h8.sh` orchestrator script lives only on `origin/han`.
But for documentation and per-phase experiment code, this branch is
self-contained.

### Strategy: fresh phase, historical branches preserved

After deliberation, **we do NOT merge `han-phase5b` or `han-phase6`
into `han`**. The branches stay as historical artefacts; if we need
to revisit Phase 5B or Phase 6 results, the original branches and
their full infrastructure are preserved exactly.

Phase 7 is a **clean re-implementation** of just the parts we need
from prior phases:

- Branch `han-phase7-unification` off latest `origin/han`.
- Cherry-pick ONLY the architecture class files that Phase 7 uses
  (not the experimental drivers, not the result indices).
- Fork Phase 5's `train_primary_archs.py` and `run_probing.py` as
  *starting points* for Phase 7 drivers; modify in-place to bake in
  the new conventions.
- Phase 7 maintains its own `training_index.jsonl`,
  `probing_results.jsonl`, and ckpt directory. Existing phase5/5b/6
  results are NOT migrated.

This avoids the 92-commit-divergent merge of `han-phase6` and the
infrastructure-sprawl problem of carrying ~5 phase-shaped silos
forward.

### The four decisions

#### (i) Subject model: `google/gemma-2-2b` (base)

Hard requirement. Rebuild the activation cache from scratch using
the base model. **Do not mention `gemma-2-2b-it` in the paper.**

Practical implications for Agent A:

- Re-extract activations for FineWeb sequences using base model.
  **Layer 12 anchor (0-indexed)** + L10–L14 for MLC.

  > Note: this is a deliberate shift from Phase 5/5b/6 (which used
  > 0-indexed L13). T-SAE (Ye et al. 2025 §4.1) and TFA (Lubana et
  > al. 2025 App. B.1) both train at 0-indexed L12 on Gemma-2-2b —
  > chosen for "comparability with pretrained Neuronpedia /
  > Gemma-Scope SAEs" (T-SAE) and "around 50% model depth" (TFA).
  > Switching Phase 7 to L12 puts our results on the *exact same
  > residual-stream tap* as both reference papers. Cache rebuild
  > from the model switch is zero-marginal-cost.

- Rebuild the 36-task probe cache against the new model + new layer.

Practical implications for Agent B:

- Rebuild the 4-passage Phase 6 concat-A/B/random qualitative cache
  against the new model + new layer.

#### (ii) Convention: fix `k_win = 500` across all archs

Every arch's per-window/per-token z output has 500 active features
after TopK. This matches Phase 5's existing TXC convention
(`k_pos × T = 100 × 5 = 500` at T=5) — preserves the regime where
TXC recipes were tuned, while standardizing per-token SAE and MLC
families up to the same `k_win`. The single exception is
`tsae_paper_k20`, kept at native k=20 as a paper-faithful baseline
to Ye et al. 2025.

**Per-arch k_pos derivation is in `plan.md` §Canonical architecture
set — that's the single source of truth for the arch list.**

`k_win=500` justification (against k_win=100 alternative):
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

#### (iii) Probing protocol: long-tail sliding mean-pool, S=128

Single S-parameterized aggregation:

- **TXC**: slide T-window with stride 1 across the S=128-token tail.
  Drop the first T−1 windows so every averaging unit covers tokens
  entirely within the considered tail. Mean the remaining
  (S − 2T + 2) per-window z's into a single (d_sae,) vector.
- **per-token SAE**: encode each of the 128 tail tokens. Drop the
  first T−1 to keep coverage strictly aligned with TXC. Mean the
  remaining per-token z's.
- **MLC**: same as per-token SAE with multi-layer extraction.

Probe: SAEBench-style top-`k_feat`-by-class-sep + L1 LR (existing
protocol unchanged). Headline `k_feat = 5`. Ablation
`k_feat ∈ {1, 2, 20}`.

This subsumes the previous lp / mp / full_window distinctions —
all three are special cases of S-parameterized sliding mean-pool. We
DO NOT report a separate "last T tokens" leaderboard: comparing
window archs with different T's at S=T is structurally confusing
(scope differs per arch).

The FLIP convention (max(AUC, 1-AUC) on `winogrande_correct_completion`
and `wsc_coreference`) is preserved — same as Phase 5.

#### (iv) Architecture set (49 archs, 3 seeds = 147 trainings)

The canonical set is grouped into 6 families:

| group | count | what's in it |
|---|---|---|
| Group 1: per-token / non-TXC | 7 | `topk_sae`, `tsae_paper` at k=500 + at k=20, `mlc`, `mlc_contrastive_alpha100_batchtopk`, `agentic_mlc_08`, `tfa_big` |
| Group 2: fixed-T TXC variants | 6 | `agentic_txc_02`, `txc_bare_antidead` (Track 2) at T ∈ {5, 10, 20}, B2 (`phase5b_subseq_track2`), B4 (`phase5b_subseq_h8`) |
| Group 3: TXCDR T-sweep | 16 | `txcdr_t<T>` for T ∈ {3,4,5,6,7,8,9,10,12,14,16,18,20,24,28,32} |
| Group 4: H8 T-sweep | 16 | `phase57_partB_h8_bare_multidistance_t<T>` for same T set |
| Group 5: anchor cells (fix k_pos=100) | 2 | `txcdr_t20_kpos100`, `phase57_partB_h8_bare_multidistance_t20_kpos100` |
| Group 6: SubseqH8 T_max-sweep (H200-only) | 2 | `phase5b_subseq_h8_T32_s5`, `phase5b_subseq_h8_T64_s5` |

T-sweep entries (Groups 3 + 4) double as leaderboard entries — same
seeds, no duplicate training. Group 6 cells exist specifically to
exploit the H200's 141 GB memory: T_max=64 wouldn't fit at fp32 on
H100 80GB.

**Full per-arch table (k_win, k_pos derivation, recipe, purpose,
src_module/src_class) lives at
`experiments/phase7_unification/canonical_archs.json` — the
machine-readable single source of truth.** A human-readable
markdown view is in `plan.md` §Canonical architecture set. Do not
duplicate the table here.

### What's dropped from the leaderboard

Phase 5 / 5B / 6 trained dozens of architectures beyond the canonical
47. They stay in `src/architectures/` as historical code on their
respective branches; they are NOT retrained for Phase 7. Categories:
H7 multi-scale contrastive; Phase 5B negatives (D1 strided, C-family
token-level, F SubsetEncoderTXC); BatchTopK paired variants beyond
`mlc_contrastive_alpha100_batchtopk`; stacked SAE family; TXCDR
weight-sharing ablations; `tsae_ours`; H8 shift-ablation variants;
`time_layer_crosscoder`, `mlc_temporal_t3`, `temporal_contrastive`,
and other Phase 5 exploratory archs.

**Full exclusion list with reasons lives in `plan.md` §Canonical
architecture set / "What's NOT in this table". Single source of
truth.**

### Branch / repo strategy

1. Branch `han-phase7-unification` off `origin/han` (latest, includes
   Phase 5 agent's shift-ablation completion).
2. Cherry-pick ONLY arch class files from `han-phase5b` and
   `han-phase6` into the Phase 7 branch:
   - From `han-phase5b`: `src/architectures/phase5b_subseq_sampling_txcdr.py`
     (with the mask-then-einsum bug fix).
   - From `han-phase6`: `src/architectures/tsae_paper.py`.
3. Phase 7 docs (`brief.md`, `plan.md`) live at
   `docs/han/research_logs/phase7_unification/`.
4. Phase 7 experiment infrastructure lives at
   `experiments/phase7_unification/`:
   - `train_phase7.py` — fork of Phase 5's `train_primary_archs.py`,
     adapted for Gemma2B-base + k_win=500 convention + ~12 archs.
   - `run_probing_phase7.py` — fork of Phase 5's `run_probing.py`,
     adapted for the new long-tail mean-pool + first-T-1 drop +
     S-parameterization.
   - `results/training_index.jsonl`, `probing_results.jsonl`, ckpts.
5. New activation cache at `data/cached_activations/gemma-2-2b/fineweb/`
   (separate from existing `gemma-2-2b-it/`).
6. Existing phase5/5b/6 result dirs and ckpts stay untouched —
   historical artefacts.

### 10-day timeline (target NeurIPS 2026-05-05)

Two-agent parallel execution. Day numbers are wall-clock days.

#### Agent A (sparse-probing leaderboard)

| day | task |
|---|---|
| 1 | Spin up **H200 RunPod (1 TB volume, 12 vCPUs, 188 GB RAM)**. Run `scripts/runpod_phase7_bootstrap.sh` to set up GH/HF/Anthropic tokens. Pull `han-phase7-unification` branch (already on origin). Fork `train_primary_archs.py` and `run_probing.py` into Phase 7 drivers. Strip dispatchers down to the canonical 47 archs. Set up Gemma2B-base path config. Smoke imports. **Verify the H200's hardware leverage hooks** (joblib parallelism, large batch, T_max=128 capacity) before starting compute. |
| 2 | Build Gemma2B-base activation cache (5 layers × 24k seqs × 128 tokens × fp16; ~70 GB). Build new probe cache with S=128 tail (~140 GB). Push caches to HF (`han1823123123/txcdr-base-data`) for cross-agent access. |
| 3 | Smoke-train each canonical arch (200 steps) on the new cache. Verify k_win=500 enforced, no OOMs. **Begin the seed=42 batch — outer loop is SEED, inner loop is ARCH** (see plan.md §Training loop ordering). Use batch=4096 if smoke-test convergence matches batch=1024. |
| 4 | **Complete the seed=42 batch (all 49 archs, ~6-10 hr on H200).** Push ckpts to `txcdr-base` incrementally as each completes. **At end of seed=42 batch, push the `seed42_complete.json` marker file to HF — this signals Agent B to start the Pareto-x-axis autointerp work.** Begin seed=1 batch (49 archs again, outer loop = seed). |
| 5 | Complete the seed=1 batch. Push `seed1_complete.json` marker. Begin seed=2 batch. |
| 6 | Complete the seed=2 batch. Push `seed2_complete.json` marker. Run the sparse-probing pass at S=128 (headline) + S=20 (continuity), k_feat ∈ {1, 2, 5, 20}. Push final `probing_results.jsonl` to HF. |
| 7 | Generate sparse-probing figures (headline bar, T-sweep, alive_fraction, seed-variance). Draft sparse-probing section of summary. |
| 8-10 | Buffer for bug-fixes / re-runs / paper draft. |

#### Agent B (qualitative autointerp)

| day | task |
|---|---|
| 1 | Spin up **H100 RunPod (1 TB volume, 8 vCPUs, 125 GB RAM)**. Run `scripts/runpod_phase7_bootstrap.sh`. Pull `han-phase7-unification` branch. Build Phase 6-style concat-A/B/random passages against Gemma-2-2b base at L12 (passages need new tokenization + activations through the base model). Port Phase 6.1's autointerp pipeline. Smoke-test on a stub ckpt. |
| 2 | Continue passage building. **Set up concurrent Haiku API calls** (ThreadPoolExecutor max_workers=16). Verify pipeline runs end-to-end on a stub. **Poll HF for `seed42_complete.json` marker** on `txcdr-base` (don't run autointerp on partial ckpt sets — Agent A's outer loop is seed, so the seed=42 batch arrives all-at-once). Estimated end of Agent A's day 4. |
| 3 | (Continues passage prep + waiting for marker.) When `seed42_complete.json` appears, pull all 49 seed=42 ckpts from HF in parallel batches. Begin autointerp scoring with concurrent Haiku calls + ckpt prefetching. |
| 4 | Continue autointerp scoring across all 49 archs at seed=42. |
| 5 | Complete autointerp scoring on all 49 archs at seed=42 (per the cost-saving in plan.md — autointerp is seed=42-only). |
| 6 | Pull Agent A's final `probing_results.jsonl` from `txcdr-base` (Agent A pushes after their full 3-seed sparse-probing pass). Compute the **Top-256 cumulative SEMANTIC Pareto plot**. |
| 7 | Generate qualitative figures + draft qualitative section of summary. |
| 8-10 | Buffer for re-runs / paper draft. |

### Risks

1. **TXC at k_win=500 + Gemma2B base may not give same rankings as
   Phase 5.** New model = new activations, possibly different
   feature structure. Honest-report.
2. **TFA at our convention** (k=500 novel head) may need verification
   that TFA's training stack is compatible — currently uses k=100.
3. **MLC family at k_win=500 vs Phase 5 historical k=100**: 5× looser,
   may change MLC rankings. Phase 5's mlc_contrastive_alpha100_batchtopk
   was the lp leader at k=100; that may not survive the bump.
4. **Storage on RunPod**: probe cache at S=128 fp16 needs ~140 GB
   plus ~280-370 GB for ckpts plus caches and HF cache. Each agent's
   RunPod uses a persistent 1 TB volume (matches the spec above)
   (cache + ckpts + workspace).
5. **HF rate limits**: pushing 12 × 3 = 36 ckpts (each ~1.7 GB) is
   ~60 GB to upload. Should be fine but may take 1-2 hours total.
6. **Autointerp Claude Haiku cost**: 12 archs × 3 seeds × 8
   features × 10 contexts ≈ 2880 Haiku calls. At Haiku pricing this
   is < $5. Negligible.

### Out of scope

- New architectures (anything beyond the canonical 12).
- Pruning `src/architectures/` of dead-weight files (post-deadline cleanup).
- Migrating phase5/5b/6 result jsonls (kept as historical artefacts).
- Long-context training beyond 128 tokens.
- T_max=128 t_sample=low ("infinite-T low-t").
- Autointerp protocol changes (re-run on new ckpts but same protocol
  as Phase 6.1).
