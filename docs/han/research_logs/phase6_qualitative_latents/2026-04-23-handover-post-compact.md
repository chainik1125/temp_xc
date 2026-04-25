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
5.7 TXC winner that lost Phase 6 qualitative 2 / 8 to `tsae_paper`
6 / 8 and `tfa_big` 6 / 8). The goal was to push a **TXC-based**
architecture to qualitative parity with the T-SAE and TFA baselines
on concat_A + concat_B, while preserving the TXC family's Phase 5
sparse-probing utility.

**Outcome (single-seed, N=8, label-count axis only): TXC-based Cycle F
reaches 7 / 8 semantic, alive 0.80 — at or above both baselines on the
x/N label-count metric. Coverage axis is unmeasured and is load-bearing
— see framing note below.**

| arch | family | alive | /8 | role |
|---|---|---|---|---|
| **`agentic_txc_02_batchtopk` (F)** | **TXC** | **0.80** | **7** 🏆 | **TXC-based qualitative winner** |
| `tsae_paper` | T-SAE | 0.73 | 6 | paper baseline |
| `tfa_big` | TFA | 1.00 | 6 | TFA baseline |
| `agentic_txc_10_bare` (Track 2) | TXC | 0.62 | 6 | ties baselines w/o BatchTopK |
| `agentic_mlc_08` | MLC | 0.13 | 5 | Phase 5.7 MLC winner |
| `agentic_txc_11_stack` (H) | TXC | 0.79 | 5 | Cycle F + AuxK, regresses |
| `tsae_ours` | T-SAE (naive port) | 0.42 | 3 | control |
| `agentic_txc_09_auxk` (A) | TXC | 0.37 | 3 | null effect |
| `agentic_txc_02` | TXC | 0.37 | 2 | Phase 5 baseline |

**Framing (see also the `project_phase6_framing` memory):** the
paper-narrative target is **TXC-family qualitative parity with
T-SAE and TFA**, not a generic BatchTopK-vs-TopK sparsity-function
ablation. Cycle F is evidence that the TXC family can be pushed to
parity; BatchTopK is the mechanism that got us there, not the
contribution. Frame summary / paper text as "TXC-based Cycle F
matches or beats both baselines qualitatively while retaining
TXC-family probing utility" (probing number pending #3 below),
**not** as "BatchTopK is the lever".

**Qualitative parity is a two-axis claim.** The paper's Figures 1
and 4 implicitly require BOTH (a) the top-N features describe
concepts (what x/N SEMANTIC label-count tests) AND (b) the top-N
features collectively span the distinct passage types in the
concat, each firing preferentially in its home passage. Our x/N
only tests (a). Skimming Cycle F's 7/8 labels, (b) is *uneven* —
top-8 features fire on Animal Farm / Darwin / "archaic English"
multiple times but **neither MMLU-bio nor MMLU-math gets a
top-8 feature peaking on it** (2 of the 7 passages in concat_A+B
uncovered). So the "7/8" headline is label-count-only; coverage is
unmeasured. #2's new sub-item (5) adds the coverage diagnostic.

**Three mechanism findings from the 5 cycles** (full reasoning in the
agentic-log; these are context for the paper's ablation section, not
the headline):

1. **Switching TXC from TopK to BatchTopK alone gives the biggest
   single jump** (+5 labels vs baseline, 0.37 → 0.80 alive). Variable
   per-sample sparsity lets rare concept features fire on the
   contexts where they matter without displacing incumbents.
2. **The anti-dead stack (unit-norm decoder + decoder-parallel grad
   removal + geom-median `b_dec` init + AuxK) on bare TXC (Track 2)
   gets most of the way there at TopK sparsity** (6 / 8, alive 0.62).
   Matryoshka + multi-scale contrastive is not load-bearing for
   qualitative — only for Phase 5 probing AUC.
3. **The two mechanism classes don't stack additively** at single
   seed — Cycle H (BatchTopK + AuxK) scored 5 / 8 vs Cycle F's 7 / 8
   at the same seed. Δ=2 at single seed and single concat is within
   plausible seed/corpus noise (Phase 5.7 precedent: ~1 label σ on
   seed alone), so treat the mechanism story ("AuxK's residual
   gradient promotes format features") as a **post-hoc explanation
   of a suggestive effect**, not an established finding, until the
   multi-seed re-run lands (#2 below).

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

### What to do next (ordered by information gain toward the TXC-parity claim)

The target the paper needs to land is **TXC-family qualitative parity
with T-SAE and TFA baselines, while retaining TXC-family probing
utility**. Follow-ups below are ranked by how directly they support
(or could overturn) that claim. See the `project_phase6_framing` and
`feedback_uncertainty_framing` memories for the motivating framing.

**#0 — Fix `encode_archs.py:135` seed hardcode (PREREQUISITE).**
The script reads `__seed42.pt` unconditionally. Every multi-seed
downstream step blocks on this. Add a CLI flag / env var (~10 LoC).
Audit `run_cycle_eval.sh` for the same assumption before re-running
it on non-42 seeds. Nothing below works until this is done.

**#1 — Seed variance on the comparison triangle: Cycle F,
`tsae_paper`, `tfa_big`.** This is the headline experiment — the
parity claim is a comparison between three arch families at three
seeds each. The current 7 / 6 / 6 single-seed headline rests on
seed-42 snapshots only; Phase 5.7 precedent had single-seed gains
halving across 3 seeds, so the effect is unmeasured on **both
directions** of the comparison.

Training cost: 3 archs × 2 extra seeds × ~30 min ≈ 3 hr GPU total,
trivially parallelisable. If the `tfa_big` per-seed cost is too high
(~3 hr/seed at plateau), you can scope it back to 1-seed and note
the asymmetry in the summary — but Cycle F and `tsae_paper` seeds
{1, 2} are non-negotiable, since they're the direct head-to-head.

**Why the triangle, not just Cycle F:** the original plan retrained
2 extra seeds of Cycle F only. That leaves the "Cycle F ≥ tsae_paper"
claim asymmetric — the baseline's variance is unmeasured. If
tsae_paper seed variance is ±1 label, a single-seed 7 vs 6 gap is
within noise and the parity claim must be hedged accordingly.

**#2 — Upgrade the evaluation metric: N=32 throughout, multi-judge,
random-corpus control.** Applies on top of #1 to every
(arch × seed × concat) cell. Sharpens every comparison in the
triangle. User explicitly approved **N=32 throughout** (despite my
suggestion to defer), so apply it globally. But note that the
dominant uncertainty is **not** within-arch sampling noise — it's
seed + corpus + judge. N=32 is a useful power boost for fine-grained
ranking; treat it as one of four complementary improvements below,
not as the primary fix for uncertainty.

Four upgrades, combine cleanly:

1. **N=32 throughout.** Bump `N_TOP_FEATURES` from 8 → 32 in
   [`run_autointerp.py`](../../../experiments/phase6_qualitative_latents/run_autointerp.py)
   line 35. Applies to every arch, every seed, every concat.
   ~32 Haiku calls / cell ≈ $0.01 per cell.

2. **Multi-judge auto-classify** semantic vs syntactic — **NOT
   single-Haiku**. Single-Haiku just swaps hand-labeller bias for
   LLM-labeller bias; the whole point is tightening judge variance.
   Implementation: 2 judge models (Haiku + Sonnet) × 2–3 prompt
   variants; take majority vote; emit `judge_disagreement_rate` as
   a per-arch field. If disagreement is > ~15 % on any arch, flag
   the /32 number as judge-sensitive in the summary. ~50 LoC. Cost:
   ~$0.03/cell. Rubric (commit edge-case resolutions in the code
   comment so they're pre-registered, not decided after seeing data):

   ```
   Given this SAE feature label: "{LABEL}"
   Classify as SEMANTIC or SYNTACTIC.
   SEMANTIC = names a concept, topic, theme, entity, or domain
     (examples: "plant biology", "Animal Farm references",
      "archaic poetic English", "historical dates in biography")
   SYNTACTIC = describes surface patterns — punctuation, word class,
     capitalisation, formatting, hyphens, quoted text
     (examples: "sentence-ending periods", "multiple-choice answer
      formatting", "hyphens between compound words")
   Edge cases (pre-registered):
     - "historical dates"                     → SEMANTIC (concept)
     - "quoted text from Animal Farm"         → SEMANTIC (corpus ref)
     - "text in quotation marks"              → SYNTACTIC (surface)
     - "MMLU answer-option formatting"        → SYNTACTIC (surface)
     - "multi-layer/cross-domain terminology" → SEMANTIC (concept)
   Reply with exactly one word: SEMANTIC or SYNTACTIC.
   ```

3. **Random-FineWeb concat control (generalisation test).** Build a
   ~1800-token concat from random FineWeb passages (NOT curated)
   and re-run the whole pipeline on it. If Cycle F at 7/8 on
   concat_A/B drops disproportionately more than tsae_paper on the
   random corpus, the parity claim is curated-concat-specific — a
   load-bearing caveat the paper needs to own. If all archs drop
   proportionally, the ranking generalises. ~30 min to build the
   corpus (new `build_concat_random.py`); no retrain, same
   autointerp cost. Commit the FineWeb passage IDs for reproducibility.

4. **Passage-discriminative ranking — DIAGNOSTIC ONLY, NOT PRIMARY.**
   The original briefing proposed this as a replacement ranking.
   **Do not swap it in as the primary metric**: ranking features by
   "how well they discriminate passages" and then measuring "how
   many have passage-concept labels" conditions on the outcome.
   Keep top-by-variance as the primary metric (matches the paper's
   Fig 1/4 construction); emit passage-discriminative
   `semantic_count_pdvar` alongside `semantic_count_var` as a
   secondary diagnostic with an explicit caveat. Same z_cache,
   no retrain.

5. **Passage-coverage diagnostic (k/P) — THE SECOND PRIMARY AXIS,
   alongside x/N SEMANTIC.** This directly tests the "span the
   passage types" half of the paper's Figure 1/4 argument, which
   x/N does not. For each top-N feature, compute its **peak
   passage** = `argmax` over passages of `mean_activation_in_passage`
   (passage IDs recoverable from `z_cache/<concat>/provenance.json`).
   Emit two numbers per (arch × seed × concat):

   - `passage_coverage_count` = number of *distinct* passages
     covered by any top-N feature's peak (max = min(N, P); P=7 for
     concat_A+B, variable for random-FineWeb)
   - `passage_coverage_entropy` = Shannon entropy of the peak-passage
     distribution over the N features (nats; high = spread across
     passages; low = concentrated on a few)

   Does not select on the outcome of interest (keeps top-by-variance
   ranking; just describes where the top-by-variance features peak),
   so it's a clean complement to x/N SEMANTIC rather than a
   re-ranking that conditions on the construct.

   **The paper-parity claim requires both axes to hold**: a TXC arch
   with 8/8 SEMANTIC labels but 3/7 passage coverage has NOT
   reproduced Figure 4. Current Cycle F x/8 = 7 but passage coverage
   on top-8 is uneven (skips MMLU-bio + MMLU-math per skim). Report
   both axes in the summary §9.5 table.

   Implementation: ~20 LoC in `run_autointerp.py` after `_pick_top_features`.
   No new API calls — works on existing z_cache + labels. Also add
   to the outcome criteria below (coverage thresholds in addition
   to x/N thresholds).

**#3 — Sparse-probing regression on Cycle F (PAPER-CRITICAL for the
"retains TXC probing utility" half of the claim).** The parity story
has two halves: (a) qualitative parity with T-SAE/TFA baselines (what
Cycle F establishes at single seed — #1 tightens it), and (b) TXC
probing utility preserved. Without (b), the contribution collapses
to "use T-SAE" — there's no reason to have invented a TXC variant.
Currently (b) is unmeasured on the BatchTopK variant.

Required: download `probe_cache` from `han1823123123/txcdr-data` on
HF (70 GB, 144 files, ~30–60 min on RunPod network; fine after disk
bump to 400 GB). Run non-GPU, parallel to #1 training.

```bash
HF_HOME=/workspace/hf_cache HF_TOKEN=$(cat /workspace/.hf-token) \
  uv run huggingface-cli download --repo-type dataset han1823123123/txcdr-data \
  --include "experiments/phase5_downstream_utility/results/probe_cache/**" \
  --local-dir .
```

Then probe on all Cycle F seeds at both aggregations:

```bash
for AGG in last_position mean_pool; do
  for SEED in 42 1 2; do
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
      experiments/phase5_downstream_utility/probing/run_probing.py \
      --aggregation $AGG \
      --run-ids agentic_txc_02_batchtopk__seed${SEED} \
      --skip-baselines
  done
done
```

Compare to `agentic_txc_02` 3-seed means (Phase 5 summary §Agentic
recipe: last_position 0.7749 ± 0.0038, mean_pool 0.7987 ± 0.0020).
**Acceptable for the paper claim**: Δ AUC ≤ ~0.02 (within the
Phase 5.7 agentic-recipe gain over baseline) → "TXC probing utility
preserved". Δ AUC in (0.02, 0.05] → soften to "modest regression,
family still competitive". Δ AUC > 0.05 → the parity story needs
a different TXC variant (try the #4 2×2 cell) that retains probing.

**#4 — Missing 2×2 cell: bare + BatchTopK + full anti-dead stack
(mechanism context; paper ablation section, not headline).**
Secondary to the triangle comparison, but cheap and informative.

|  | TopK | BatchTopK |
|---|---|---|
| matryoshka + contrastive | (baseline) 2 / 8 | **Cycle F 7 / 8** |
| bare (no matryoshka / contrastive) | **Track 2** 6 / 8 | **⬜ not tested** |

Outcome readings:
- Beats Cycle F (≥ 8 / 8 or significantly above): the simpler
  recipe — "BatchTopK + anti-dead, no matryoshka or contrastive"
  — is sufficient; matryoshka + contrastive is ornamental for
  qualitative. Promote to candidate headline TXC arch.
- Ties Cycle F: matryoshka + contrastive is orthogonal to
  qualitative gain (already suspected from Track 2 at 6/8).
- Regresses to ~5 / 8 (like Cycle H): anti-dead + BatchTopK don't
  compose in general; Cycle H's regression generalises beyond
  matryoshka. That IS a mechanism finding, just a different one.

Implementation: subclass `TXCBareAntidead`
([`src/architectures/txc_bare_antidead.py`](../../../src/architectures/txc_bare_antidead.py))
and swap its TopK-scatter for `BatchTopK(k)` from
[`src/architectures/_batchtopk.py`](../../../src/architectures/_batchtopk.py).
~20 LoC. Arch name: `agentic_txc_12_bare_batchtopk`. Dispatcher:
add branch in `train_primary_archs.py` modelled on Track 2's branch
(~line 1284). Wire `encode_archs.py`, `run_probing.py`,
`arch_health.py` the same way Cycle A / Track 2 / Cycle F / Cycle H
were wired. One training run ~30 min + eval ~5 min.

If it lands competitively (≥ 6 / 8 at seed 42), add it to #1's
multi-seed retrain and #3's probing run — it may be a better
headline arch than Cycle F.

**#5 — Longer training past the plateau-stop artefact.** All Phase
6.1 cycles converged at step 4000 – 5600 / 25000 due to the 2 %
plateau threshold. Phase 5.7 `agentic_txc_02` went to ~16 205 steps.
The anti-dead mechanisms may shape decoder directions better over
the full window.

```python
# In train_primary_archs.py TrainCfg:
@dataclass
class TrainCfg:
    ...
    min_steps: int = 10_000        # was 3_000
```

Cheap (~40 min per re-train). Expected: modest improvement, probably
not enough alone to push past 7 / 8.

**#6 — HF upload of Phase 6.1 ckpts.** Cross-pod reproduction. Target
repo: `han1823123123/txcdr`. Cycle F seed 42 is already up; upload
the rest (Cycle A, Track 2, Cycle H, the #4 2×2 cell if trained, and
the new seeds {1, 2} for Cycle F + tsae_paper once they land).

```python
from huggingface_hub import upload_file
# for arch in ('agentic_txc_09_auxk', 'agentic_txc_10_bare',
#              'agentic_txc_11_stack', 'agentic_txc_12_bare_batchtopk'):
#     for seed in (42, 1, 2):
#         upload_file(
#             path_or_fileobj=f'experiments/phase5_downstream_utility/results/ckpts/{arch}__seed{seed}.pt',
#             path_in_repo=f'ckpts/{arch}__seed{seed}.pt',
#             repo_id='han1823123123/txcdr',
#         )
```

**Concrete execution plan (parallelisable):**

```bash
# Step 0: PREREQ — fix encode_archs.py:135 seed hardcode + audit
# run_cycle_eval.sh for the same assumption. Commit before anything
# below.

# Step A (non-GPU, no API key): start probe_cache download for #3.
HF_HOME=/workspace/hf_cache HF_TOKEN=$(cat /workspace/.hf-token) \
  nohup uv run huggingface-cli download --repo-type dataset \
  han1823123123/txcdr-data \
  --include "experiments/phase5_downstream_utility/results/probe_cache/**" \
  --local-dir . > logs/probe_cache_dl.log 2>&1 &

# Step B (GPU, parallel to A): triangle seed-variance training.
# Cycle F + tsae_paper at seeds {1, 2}; add tfa_big if time permits.
# Also add agentic_txc_12_bare_batchtopk (#4) here if implemented.
TQDM_DISABLE=1 PYTHONPATH=. nohup .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[1, 2], max_steps=25000,
        archs=['agentic_txc_02_batchtopk', 'tsae_paper'])
" > logs/phase61_triangle_seedvar.log 2>&1 &

# Step C (non-GPU, parallel to A+B): edit run_autointerp.py for #2
#   (1) N_TOP_FEATURES = 32
#   (2) multi-judge auto-classify (Haiku + Sonnet, 2-3 prompts)
#   (3) passage-discriminative ranking as a SECONDARY emit
#   (4) passage-coverage diagnostic (k/P + entropy) — SECOND PRIMARY
#       axis alongside x/N; computed on top-by-variance features
#       (no re-ranking)
#   (5) per-cell {semantic_count_var, semantic_count_pdvar,
#       passage_coverage_count, passage_coverage_entropy,
#       judge_disagreement_rate} in labels JSON.

# Step D (non-GPU, parallel to A+B): build the random-FineWeb concat.
# New script: experiments/phase6_qualitative_latents/build_concat_random.py
# Commit the passage IDs for reproducibility.

# Step E (GPU, after B): encode every (arch × seed × concat) cell.
for SEED in 42 1 2; do
  for ARCH in agentic_txc_02_batchtopk tsae_paper tfa_big; do
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
      experiments/phase6_qualitative_latents/encode_archs.py \
      --archs $ARCH --sets A B random --seed $SEED
  done
done
# Also encode the remaining 6 archs at seed=42 on the random concat
# so every row of §9.5 gets a random-corpus number.

# Step F (after C+E): upgraded autointerp on every cell.
export ANTHROPIC_API_KEY=$(cat /workspace/.anthropic-key)
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py

# Step G (after A+B): probing on Cycle F (+ #4 cell if trained).
# See #3 above for commands.

# Step H: update summary.md §9.5 and agentic-log with:
#   - Triangle 3-seed means ± stderrs at /32, both concat_A/B and
#     random-FineWeb, for each arch
#   - Judge-disagreement rates per arch
#   - Cycle F probing Δ AUC at both aggregations (+ #4 cell if run)
#   - Passage-discriminative diagnostic as a footnote, not headline
# Reframe the summary's §9.5 headline from "BatchTopK is the lever"
# / "one-knob trade-off" to "TXC-family qualitative parity with T-SAE
# and TFA, probing utility retained" per project_phase6_framing.
```

**Outcomes that support the TXC-parity claim** (**BOTH qualitative
axes should hold**: label-count AND passage-coverage):

- **x/N label-count axis** (triangle at /32 × 3 seeds, both concats):
  Cycle F mean within ±0.5 labels of both tsae_paper and tfa_big →
  "matches both baselines". Cycle F strictly ≥ both → "matches or
  beats".
- **Passage-coverage axis** (new): Cycle F top-N coverage count ≥
  baselines' count, and coverage entropy within 0.1 nats of
  baselines'. Cycle F at 5/7 coverage is acceptable ONLY if
  tsae_paper is also ~5/7; if tsae_paper is 7/7 and Cycle F is 5/7,
  the parity claim fails on coverage regardless of label-count.
- **Random-FineWeb concat:** all archs drop, but Cycle F's rank vs
  baselines holds **on both axes** → generalisation confirmed. If
  Cycle F drops disproportionately on either → "curated-concat-
  specific", hedge the claim.
- **Cycle F probing Δ AUC ≤ ~0.02:** TXC probing utility preserved;
  full parity-plus-utility claim stands.
- **Judge disagreement < 15 %:** /32 numbers are not judge-sensitive;
  claims robust.

**Outcomes that would overturn the claim:**

- `tsae_paper` seed variance ≥ 1.5 labels and Cycle F ties it within
  noise → demote x/N axis to "TXC matches within noise".
- **Cycle F passage coverage materially below tsae_paper's** (e.g.
  3/7 vs 6/7) while x/N matches → **this is the failure mode that a
  skim of §9.5's top-8 labels already suggests.** The current top-8
  concentrates on Animal Farm / Darwin / archaic passages and skips
  MMLU-bio + MMLU-math. If the N=32 expansion doesn't recover those
  passages, the paper-parity claim fails on the coverage axis even
  though the label-count axis looks good. Acknowledge explicitly
  rather than burying; the coverage gap is then a load-bearing
  follow-up (longer training? more scale in contrastive? different
  Phase 5 recipe?).
- Cycle F drops disproportionately on the random concat → the
  qualitative gain is corpus-specific; the paper has to own this.
- Cycle F probing Δ AUC > 0.05 vs `agentic_txc_02` → probing utility
  significantly degraded. The parity-plus-utility claim fails for
  Cycle F specifically; shift to the #4 2×2 cell as the headline
  arch if it gets both.

**Caveats:**
- `plot_top_features.py` (Figure 6 in §9.5) plots top-8 by raw
  variance. Keep that for visualisation (paper Fig 4 literally shows
  8 features) — change only the *score* metric to N=32, not the
  plot.
- `run_cycle_eval.sh` hardcodes seed=42 and calls the N=8
  autointerp. Update or fork in Step 0.

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
above. Then: **#0 is a hard prereq** — fix `encode_archs.py:135`
seed hardcode before touching anything else. After that, the work
splits into three parallelisable tracks. **GPU track (#1):** kick
off the triangle seed-variance training — Cycle F + `tsae_paper`
(+ optionally `tfa_big` + the #4 2×2 cell) at seeds {1, 2}, ~3 hr
background. **Non-GPU track (#3):** start the 70 GB `probe_cache`
download in parallel, then probe Cycle F at both aggregations once
the ckpts from #1 land. **Script-edit track (#2):** upgrade
`run_autointerp.py` with **five** changes — N=32, multi-judge
auto-classify (Haiku + Sonnet × 2–3 prompts), random-FineWeb concat,
passage-discriminative as a secondary diagnostic only, and the
**passage-coverage k/P diagnostic as a second primary axis** (this
directly tests whether top-N features span the distinct passage
types, which the x/N label-count does not — and a skim of current
§9.5 top-8 labels suggests Cycle F may fail on this axis, skipping
MMLU-bio and MMLU-math). Once tracks finish, every row in §9.5
gets `/32` means ± seed stderrs **plus coverage k/P** on both curated
and random concats, Cycle F gets a probing Δ AUC number, and the
summary can be reframed from "BatchTopK is the lever" to the intended
"TXC-family qualitative parity with T-SAE and TFA (both label-count
and passage-coverage axes), probing utility retained".
#4 (missing 2×2 cell) and #5 (longer training) are orthogonal
ideas to explore if time remains or if the parity claim needs a
better-scoring TXC variant.
