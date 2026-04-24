---
author: Han
date: 2026-04-24
tags:
  - proposal
  - todo
---

## Post-compact agent priorities (READ THIS FIRST)

**Target audience**: a fresh agent picking up after the 2026-04-24
context compact. Read this doc in full, then
[[../EXPERIMENT_INDEX]] in `docs/han/`, then the specific priority
handover you're picking up.

### 0 — What's in flight and what just broke (pre-compact state)

**Nothing is actively running.** GPU is idle. No background processes
pending. All ckpts that should exist do exist. Branch is pushed.

**One unfinished piece** (non-blocking but should be cleaned up
early): **`phase62_c2_track2_contrastive__seed2` eval**. The training
completed successfully (ckpt at
`experiments/phase5_downstream_utility/results/ckpts/phase62_c2_track2_contrastive__seed2.pt`,
~850 MB) but the chained `encode → autointerp → probe` step crashed
somewhere after it started (last log line in
`logs/phase62_c2_seedvar.log` is "=== C2 seed=2 encode + autointerp
===" with no subsequent lines). Recovery: run ~15 min's worth of
work:

```bash
source /workspace/temp_xc/.envrc && cd /workspace/temp_xc
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/encode_archs.py \
  --archs phase62_c2_track2_contrastive --sets A B random --seed 2
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs phase62_c2_track2_contrastive --seeds 2 --concats A B random
for AGG in last_position mean_pool; do
  TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python -u \
    experiments/phase5_downstream_utility/probing/run_probing.py \
    --aggregation $AGG --run-ids phase62_c2_track2_contrastive__seed2 \
    --skip-baselines
done
.venv/bin/python scripts/hf_sync.py --go
```

After this, C2 has 3-seed data on both metrics. Update
`EXPERIMENT_INDEX.md §1` to remove the "(seed 2 ckpt only, eval
pending)" note.

**Environment check on pod restart** (in case the compact coincides
with a pod bump — defensive):

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/temp_xc && uv sync 2>&1 | tail -3
source .envrc
echo "$HF_HOME ${ANTHROPIC_API_KEY:0:14}"   # both non-empty
.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"
git status -sb | head -3                    # clean on han-phase6
```

### 1 — Priority 1: T-sweep (user's first-ask)

**Hypothesis**: "increasing TXC window T trades probe AUC for more
interpretable latents".

**Status**: fully scoped. Handover doc is
[[phase6_2_autoresearch/2026-04-24-handover-t-sweep]]. Has
execution plan, dispatcher code templates, decision tree, risks.

**What the agent has to do**:

1. Add 3 dispatcher branches (`phase63_track2_t3`, `_t10`, `_t20`)
   in `train_primary_archs.py` around line 1470 (template in
   handover doc).
2. Wire them into `encode_archs.py` load_arch + encode dispatch,
   `run_probing.py` load_model + encoder dispatch, `arch_health.py`
   TXC list (4 edits total, templates in handover).
3. Train 3 models at seed=42 (~2.5 hr on A40 — T=20 is the long
   one).
4. Encode + autointerp + probe them.
5. Plot "AUC vs T" and "random-concept /32 vs T" overlaid.
6. Commit + HF sync.

**Decision rules** (pre-registered):

- If **T=10 or T=20 scores ≥ 7/32 random**: this breaks the Phase
  6.2 plateau. Retrain at seeds {1, 2} for 3-seed variance.
  Update §9.5 claim from "structural gap" → "larger T closes part
  of the gap at probing cost".
- If **T=3/10/20 all at 2-4/32 random**: strengthens the
  structural-gap claim. Add as an appendix ablation row and move on.
- If **T=3 alone > T=5 qualitative**: interesting, tsae_paper-ward.
  Try T=2 next.

**Budget**: ~2.5 hr GPU + <$0.50 API. Well within $10 remaining.

### 2 — Priority 2: rethink the x/32 metric (user's methodological concern)

**User's concern (verbatim)**: *"Just because the top 32 features
don't contain the semantic features we expect it doesn't mean the
latents are 'worse' for interp. It might be the case that we really
need to expand 32 to something larger. From an interp perspective,
faithfulness is important, and the TXC's superior probe AUC suggests
that it is more faithful. So wouldn't it be better for the
qualitative metric to be the 'number of distinct semantic features
that appear in the top X' rather than 'the fraction of the top 32
which are semantic'."*

**Why the metric might be systematically unfair to TXC**: the TXC
encoder sees 5-token windows, which have a lot of variance in
boundary-pattern activations (sentence starts, punctuation, etc.).
These are *genuine statistical regularities* the model learns about,
and a faithful SAE SHOULD represent them — which means they occupy
top-by-variance slots. Semantic features may exist further down the
ranking. The x/32 ratio penalises TXC for being faithful; it's the
wrong success measure.

Three concrete redesigns, ordered cheapest-to-most-informative:

#### 2a — Passage-discriminative ranking (already-computed; $0.90 API)

`run_autointerp.py` already emits `top_feat_indices_pdvar` — the
top-32 feature indices when ranked by
`Var_over_passages(mean_act_in_passage)` instead of per-token
variance. These indices are in every
`{arch}__seed{S}__concat{C}__labels.json` under the key
`top_feat_indices_pdvar`. Currently we compute
`semantic_count_pdvar_overlap` = "how many pdvar-top-32 overlap with
var-top-32" — a diagnostic, not a proper metric.

**To get a true pdvar semantic count**: label the features in
pdvar-top-32 that are NOT already in var-top-32. Per cell that's
~12-20 additional features (overlap is usually ~12-20 out of 32).

**Implementation**:

```python
# experiments/phase6_qualitative_latents/run_autointerp_pdvar.py
# — new script, small. For each existing labels JSON:
#   1. Load top_feat_indices_pdvar from the JSON
#   2. Identify which pdvar indices are NOT in top_feat_indices_var
#   3. For each such feature: gather contexts, Haiku label, judge
#   4. Recompute semantic count over the union {var ∩ pdvar} (existing
#      labels) + {pdvar only} (new labels)
#   5. Write metrics["semantic_count_pdvar"] = that count
```

**Scope**: re-run for the 5 primary archs × 3 seeds × 3 concats = 45
cells × ~15 new features × 3 calls = 2025 Haiku calls ≈ $1.50.
**Cheap enough to run today.**

**Decision rule**: if TXC-family archs gain > 3 labels under pdvar
ranking (e.g., Track 2 3.3/32 → 8/32 on random), the structural-gap
claim softens substantially. Regenerate the Pareto figure with this
new axis.

#### 2b — Top-N sweep (N ∈ {32, 64, 128, 256}; ~$5 API)

Extend the top-N set beyond 32. Same variance ranking, just more
features. Plot SEMANTIC count as a function of N for 3 target archs
(Track 2, tsae_paper, Cycle F) at seed=42 on concat_random (the
concat that showed the dramatic collapse).

**Implementation**:

```python
# One-line edit in run_autointerp.py:
N_TOP_FEATURES = 256    # was 32

# Then RE-RUN only for the 3 target archs × concat_random at seed=42
# (re-running A/B is cheap but lower signal — concat_random is where
# the plateau was observed).
```

**Scope**: 3 archs × 1 concat × 1 seed × (256 - 32) new features × 3
calls = 2016 Haiku calls ≈ $1.50. If interesting, extend to 3 seeds
= another $3.

**Decision rule**:

- **Track 2 curve flattens at ~5-8/32** by N=256: TXC genuinely
  caps out. `tsae_paper` likely reaches 30-50+ by N=256.
  Structural-gap claim holds, and we have a cleaner figure:
  "TXC caps at 8 semantic features, tsae_paper keeps going".
- **Track 2 curve rises to > 15/256 by N=128 or N=256**: the
  plateau was a top-rank artefact. TXC has plenty of semantic
  features at lower ranks. PAPER STORY CHANGES: the gap isn't
  structural, it's about where semantic features sit in the variance
  ranking. This RECOVERS the TXC-qualitative claim in a
  methodologically defensible way.

**Budget warning**: if agent wants N=512 on 3 archs × 3 seeds × 3
concats that's ~$30 — over the remaining $10 Haiku budget. Stick to
N=256 and 1 seed × 1 concat.

#### 2c — Distinct-concept dedup (most faithful to user's request, ~$5 API)

User asked for "number of **distinct** semantic features in the top
X", not total. Needs label deduplication:

**Implementation** (after labelling N=256):

```python
# Use an embedding model (cached sentence-transformers or similar)
# to cluster the 256 label strings into distinct concepts.
# Alternatively: ask Claude Sonnet to cluster them ("here are 256
# SAE feature labels, group semantically-identical ones. Return N
# groups, one label per group").
# Count N.
```

**Caveat**: clustering is judgement-dependent. Strong paper claim
would need: (i) inter-coder agreement, (ii) published clustering
method, (iii) multi-seed robustness. Probably out of scope for the
NeurIPS deadline unless Priority 2b shows a striking effect.

**Suggested ordering within Priority 2**: run 2a first (cheapest,
fastest decision). If 2a shows TXC catching up to tsae_paper under
pdvar ranking, write up the result and consider the paper-narrative
implications. If 2a doesn't move the needle, run 2b before 2c. 2c
only if 2b shows the gap closing.

### 3 — Priority 3: faithfulness ablation (user's other methodological concern)

**User's suggestion**: *"To test faithfulness, we can take
inspiration from what the MatryoshkaSAE paper did: ablate a latent
and see its effect on a probe."*

**Experimental design**:

1. Pick the top-5 SEMANTIC-labelled features from each arch's
   top-32 on concat_A+B (we have labels already).
2. For each such feature: run the Phase 5 probing pipeline with the
   feature ZEROED at encode time (intervention).
3. Compare AUC drop on: (a) the task whose domain matches the
   feature's label, vs (b) average of the other 35 tasks. Faithful
   feature: big (a) drop, small (b) drop. Non-faithful: uniform drop.

**Matching features to tasks** is the hard part. "Plant biology"
label → no direct task in our 36-task set. "Animal Farm references"
→ no direct task. This is a mismatch between the Phase 6 concepts
(archaic literature, biology) and the Phase 5 tasks (amazon
reviews, bias in bios, github code). Two ways to handle it:

a. **Expand the task set** to include Phase 6-concept-aligned tasks
   (political ideology classification, historical-vs-modern text
   detection, literature style). That's a significant task-building
   effort.
b. **Use whatever alignment exists** — e.g., a "biographical
   dates" feature should damage bias_in_bios tasks more than
   ag_news. Weak signal but runnable today.

**Scope** if option (b): ~20 probing runs × 5 min = ~2 hr. Implement
a `--zero-feature-idx N` flag in `run_probing.py` (~20 LoC hook in
`_encode_for_probe`).

**Deferred to Phase 6.4** unless Priority 2 doesn't rehabilitate TXC
qualitative — in which case this becomes the backup evidence.

### 4 — Ordering recommendation + critical path

If there's time for only ONE priority before the NeurIPS submission:

1. **Do Priority 2a first (pdvar metric, ~$1.50 API, ~15 min
   agent time)**. It's cheap, fast, and can fundamentally
   reframe the paper story. This is the user's methodological
   concern that has the highest expected information.
2. If 2a shows nothing interesting, run Priority 1 (T-sweep).
3. Priority 2b if 2a was negative and you want one more shot.
4. Priority 3 is Phase 6.4 — don't do it unless we need it.

**Critical path to the paper** (assuming Priority 2a + T-sweep both
land):

- Priority 2a (pdvar) finishes in ~15 min → decide whether to
  rewrite the "structural gap" narrative.
- Priority 1 (T-sweep) finishes in ~3 hr → add T-axis figure to
  §9.5.
- Update `docs/han/research_logs/phase6_qualitative_latents/summary.md`
  §9.5 with new findings.
- Update `experiments/phase6_qualitative_latents/plot_pareto_robust.py`
  to include pdvar-ranked points.
- Commit + push + HF sync.
- Hand off to paper-writing pass.

### 5 — What NOT to do

- **Don't retrain anything already in
  [[../EXPERIMENT_INDEX]] §1**. 14+ archs × up-to-3 seeds each
  already exist. Check the index before any `train_primary_archs.py`
  call.
- **Don't touch Phase 5 benchmarks** (Phase 5 agent owns that on
  `han` branch).
- **Don't conflate "top-32 semantic count" with "qualitative
  quality"** — Priority 2 exists precisely because we're unsure the
  metric measures what we want. Write paper prose carefully.
- **Don't commit training-log JSONs or per-run loss curves to git**
  — they live in `experiments/phase5_downstream_utility/results/training_logs/`,
  are NOT in the HF-sync plan by default. If you need them pushed,
  extend `scripts/hf_sync.py`'s `_plan_entries()`.
- **Don't re-run autointerp cells that already have labels at
  temp=0** without a good reason — the data is deterministic (mostly;
  Haiku has small drift) and regenerating wastes API budget.

### 6 — Resources + checkpoints

- **Master index**: [[../EXPERIMENT_INDEX]] (incl. branch-sharing
  protocol — safe to cherry-pick to `han` without conflict).
- **Phase 6.1 handover** (pre-6.2, historical reference):
  [[phase6_qualitative_latents/2026-04-23-handover-post-compact]].
- **Phase 6.2 summary** (completed 6.2 analysis):
  [[phase6_2_autoresearch/summary]].
- **Phase 5 probing summary** (for baseline AUC reference):
  [[phase5_downstream_utility/summary]].
- **Paper narrative anchor**: §9.5 of
  [[phase6_qualitative_latents/summary]] — this is the section the
  paper writeup will draw from.
- **HF repos**:
  - Ckpts: `han1823123123/txcdr`
  - Data: `han1823123123/txcdr-data`
- **Sync**: `scripts/hf_sync.py --go` (idempotent, manifest-indexed).
  Call after every experiment lands.
- **Memory** (persists across compacts):
  `/home/appuser/.claude/projects/-workspace-temp-xc/memory/MEMORY.md`
  — pointer to 3 project memories: Phase 6 framing, uncertainty
  framing, HF-sync rule, Phase 6.2 queued status.

### 7 — Final sanity: before you start any experiment

1. Run the env-check in §0.
2. `git log --oneline -3` — HEAD should be `36c4af4` (or later).
3. `cat .hf_sync_manifest.json | wc -l` — should be > 200 entries.
4. Read [[../EXPERIMENT_INDEX]] to confirm what's already done.
5. Kill off the C2 seed 2 eval pending issue (§0 script).
6. Start with Priority 2a (pdvar metric).
