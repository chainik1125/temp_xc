---
author: Han
date: 2026-04-26
tags:
  - design
  - in-progress
---

## Agent C — case studies on Phase 7 ckpts

### Purpose

Agent A produces a sparse-probing leaderboard. Agent B produces the
top-256 cumulative-semantic Pareto. **Neither directly answers
"are TXCs more useful than T-SAE / MLC / TopK SAE for *downstream*
interpretability tasks."** That's the gap Agent C fills, by
reproducing T-SAE's §4.5 case-study protocol on Phase 7 ckpts.

Two case studies, both reusing Phase 7's trained ckpt set:

1. **Dataset understanding (HH-RLHF)** — decompose Anthropic's
   helpful-harmless preference dataset with each arch's features.
   Reproduce T-SAE's findings (length spurious correlation; safety
   features) and check whether TXC families recover those signals
   at least as well.
2. **AxBench-style steering** — for ~30 interpretable features,
   intervene at varying strengths, grade output success + coherence
   with Claude Sonnet 4.6 (substituted for the paper's
   Llama-3.3-70b grader). T-SAE's Pareto-domination claim is the
   benchmark to match or beat.

### Pod spec (provisioned)

| | Agent C — case studies |
|---|---|
| GPU | NVIDIA **H100**, 80 GB HBM3 |
| vCPUs | **8** |
| System RAM | **125 GB** |
| Persistent volume | **2 TB** at /workspace |

Disk budget at 2 TB:
- 6 ckpts pulled from `txcdr-base` (~10 GB)
- HF cache (gemma-2-2b base + tokenizer + datasets): ~15 GB
- HH-RLHF cache + tokenized: ~5 GB
- Steering generations + grader logs: ~10 GB
- Phase 7 activation cache (optional, if Agent C does its own
  per-token feature mining): ~140 GB
- venv + misc: ~30 GB
- **Total ~210 GB / 2000 GB ≈ 10% utilisation.** Plenty of margin.

### Reusable Phase 7 infrastructure

Agent C does NOT re-train. Pulls everything else from Phase 7:

| Phase 7 asset | Agent C use |
|---|---|
| `han1823123123/txcdr-base` HF model repo | Pull selected ckpts by `run_id` |
| `experiments/phase7_unification/canonical_archs.json` | Same `arch_id` → `src_class` registry |
| `experiments/phase7_unification/_paths.py` | Path-discipline pattern; Agent C extends |
| `experiments/phase7_unification/run_probing_phase7.py:_load_phase7_model` | **Reuse directly** — covers all 12 src_class types Phase 7 trains |
| `/workspace/.tokens/anthropic_key` | Claude Sonnet 4.6 grader (substituted for paper's Llama-3.3-70b) |
| `/workspace/.tokens/hf_token` | Pull ckpts; pull HH-RLHF / FineWeb |

### Selected arch shortlist — staged

**Stage 1 (framework-debug pass): 3 archs only.** Build the C.i + C.ii
pipelines end-to-end, fix infra issues, validate the apples-to-apples
protocol. Picking the 3 most conceptually distinct designs so we
expose any framework bugs that depend on arch family:

| arch_id | row | family | rationale |
|---|---|---|---|
| `topk_sae` | 1 | per-token SAE | literature baseline (TopK SAE); the "naive" comparison every reviewer wants to see |
| `tsae_paper_k500` | 2 | T-SAE port | direct head-to-head with Ye et al. 2025's flagship method (the paper we're reproducing the case study from) |
| `agentic_txc_02` | 8 | TXC + multi-scale matryoshka | TXC contribution representative (Phase 5 mp winner); covers the window-arch family |

**Stage 2 (expansion, after Stage 1 works): add 3 more.** Only do
this once the Stage-1 protocol is locked in and producing
publication-quality results.

| arch_id | row | family | rationale |
|---|---|---|---|
| `mlc_contrastive_alpha100_batchtopk` | 5 | MLC | Phase 5 lp leader; multi-layer crosscoder reference (different design philosophy from TXC) |
| `phase5b_subseq_h8` | 13 | SubseqH8 | Phase 5B mp champion (subsequence sampling) |
| `phase57_partB_h8_bare_multidistance_t5` | 32 | H8 multi-distance | Phase 5 mp peak at IT regime; vanilla H8 reference |

If time allows after Stage 2, add `tfa_big` (row 7).

**Why staged**: case-study infra (HH-RLHF preprocessing, AxBench
intervention loop, Sonnet grader prompt design) needs iteration.
Discovering a protocol bug after running 6 archs costs 6× more than
catching it after running 3. Stage 1 is the iteration phase; Stage 2
is the production run.

### Reference code status (T-SAE repo)

Cloned to `references/temporal_saes/`. **What's there**:

- `dictionary_learning/dictionary_learning/trainers/temporal_sequence_top_k.py`:
  `TemporalMatryoshkaBatchTopKSAE` — same class Phase 7 ports at
  `src/architectures/tsae_paper.py`. Confirms our port is faithful.
- `dictionary_learning/dictionary_learning/utils.py:60 load_dictionary`:
  loads THEIR ckpt format. **Phase 7 ckpts are saved in OUR format**
  (raw `state_dict.pt` via `torch.save`); use Phase 7's
  `_load_phase7_model` instead.
- `src/data.py`, `src/activations.py`: dataset + activation utilities
  using `nnsight.LanguageModel`. Useful patterns; Agent C can keep
  using HF `AutoModelForCausalLM` (matches Phase 7 convention).
- `src/experiments/probing.py`: their probing entrypoint on FineFineWeb
  10-domain classification. Different from Phase 7's SAEBench-style
  probing — *not* what we need (Agent A handles probing).

**What's NOT there**:

- `src/experiments/alignment_study.py` (HH-RLHF understanding) —
  documented in their README but absent from the repo. **Agent C
  builds this from scratch using the paper as protocol spec.**
- Any steering code — the README literally says "Under construction!"
  Agent C builds the steering pipeline from the AxBench paper
  (Wu et al. 2025, arXiv:2501.17148), which Phase 7 cites in §4.5.

So Agent C's deliverable is largely *new* code, with the T-SAE repo
serving mainly as a sanity-check on the trainer + a reference for
data loading patterns.

### Deliverables

#### C.i — HH-RLHF dataset understanding

For each of the 6 selected archs:

1. Load Phase 7 ckpt via `_load_phase7_model`.
2. Forward Anthropic/hh-rlhf chosen + rejected pairs through Gemma-2-2b
   base; capture L12 activations at every token.
3. Encode through arch → top-K activating features per text.
4. Compute per-feature **t-statistic of (mean_chosen − mean_rejected)**
   to find features that discriminate chosen from rejected.
5. **Reproduce T-SAE's length-spurious-correlation finding**: project
   chosen vs rejected onto length, t-test, report.
6. Hand-label the top-20 features per arch with their top-activating
   contexts (autointerp protocol from Agent B).

Output:
- `experiments/phase7_unification/case_studies/hh_rlhf/<arch_id>/feature_stats.json`
- `experiments/phase7_unification/case_studies/hh_rlhf/<arch_id>/top_features.json`
- `experiments/phase7_unification/results/plots/phase7_hh_rlhf_summary.png`

#### C.ii — AxBench-style steering (apples-to-apples protocol)

**The fairness gap and how we close it**: T-SAE paper says they
"average across 30 different features" but doesn't specify HOW the
30 are chosen per arch. Different archs have different feature
spaces; "feature 17" in T-SAE is not "feature 17" in TXC. Naively
picking 30 features per arch by, say, "highest-variance" gives
each arch a different set, and the comparison loses meaning.

**Our protocol**: pick a fixed set of N **target concepts** (not
features). For each (arch, concept) pair, find the BEST feature in
that arch's space for that concept (highest autointerp-label match
score on Agent B's labels, OR highest activation on a held-out
text annotated with that concept). Steer THAT feature.

The comparison axis becomes: "given the same target concept, how
well does each arch's best feature steer the model?" That's the
AxBench-style apples-to-apples question.

**Concept set (suggested)**: 30 concepts spanning safety-relevant +
domain-specific + style-defining categories. Initial proposal:
- Safety/alignment: harmful_content, deception, refusal_pattern,
  helpfulness_marker, jailbreak_pattern (5)
- Domain: medical, legal, mathematical, programming, scientific,
  literary, religious, financial, historical, geographical (10)
- Style: formal_register, casual_register, instructional,
  narrative, dialogue, poetic, technical_jargon (7)
- Sentiment: positive_emotion, negative_emotion, neutral_factual,
  question_form, imperative_form (5)
- Other: code_context, list_format, citation_pattern (3)
Total: 30. Refine after Stage-1 spike.

Per (arch, concept):
1. Look up best feature for this concept in this arch (via Agent B's
   autointerp labels at `txcdr-base/autointerp/<arch>/`; if no
   matching label, score-rank by encoded activation on a 100-text
   concept-annotated sample).
2. For each of 8 strengths ∈ {0.5, 1, 2, 4, 8, 12, 16, 24} of the
   feature's unit-norm decoder direction:
   - Steer-decode 60 tokens via Gemma-2-2b base from a fixed neutral
     prompt ("We find").
3. Grade each generation with Claude Sonnet 4.6 along two axes
   (per AxBench protocol):
   - **Steering success**: does output contain the concept's
     semantics? 0–3.
   - **Coherence**: is output coherent text? 0–3.
4. Per-arch summary: mean (success, coherence) across 30 concepts ×
   8 strengths = 240 cells per arch. Plot Pareto: each arch is a
   curve; Pareto-dominance = both higher success AND higher
   coherence at any operating point.

**Total grader calls**: 3 archs × 30 concepts × 8 strengths = 720
generations + grades for Stage 1; expand to 1440 in Stage 2.

Output:
- `experiments/phase7_unification/case_studies/steering/<arch_id>/feature_selection.json` — chosen feature per concept + score
- `experiments/phase7_unification/case_studies/steering/<arch_id>/generations.jsonl`
- `experiments/phase7_unification/case_studies/steering/<arch_id>/grades.jsonl`
- `experiments/phase7_unification/results/plots/phase7_steering_pareto.png`

### Time budget (9 days to NeurIPS)

| day | task |
|---|---|
| 1 | Bootstrap pod (RUNPOD_INSTRUCTIONS + restart_recovery.sh). Pull 6 ckpts from `txcdr-base`. Smoke-test `_load_phase7_model` on each. Pull HH-RLHF dataset. |
| 2-3 | C.i implementation + run on 6 archs. Write up. |
| 4-7 | C.ii implementation (AxBench protocol from scratch); run 6 archs × 30 features × 8 strengths = 1440 generations + grader calls. |
| 8-9 | Cross-arch comparison plots, draft Phase 7 case-studies section, merge into `summary.md`. |

If C.ii blows the budget, the C.i write-up alone is paper-strong as a
"can SAE-family methods recover known dataset structure?" study. C.ii
moves to appendix.

### Note for Agent C: S decision changed (2026-04-26)

Phase 7's sparse-probing headline `S` value was changed from a
multi-S sweep `(128, 64, 20)` to a **single value S = 32 across the
board**. This affects how Agent C should interpret the sparse-probing
leaderboard when correlating against the case-study results.

Why: at S=128 the probing pass would have taken ~60 hr (incompatible
with deadline); also the original `S ≥ 2T − 1` validity rule was
found to be a bug (correct: `S ≥ T`). Full rationale at
`docs/han/research_logs/phase7_unification/2026-04-26-S-decision-revised.md`.

What this means for Agent C:
- The sparse-probing AUCs Agent C reads from
  `experiments/phase7_unification/results/probing_results.jsonl` will
  all be at `S=32, k_feat ∈ {5, 20}`. Old rows at S=128 / 64 / 20
  (from earlier sanity-check experiments) may also be present in the
  jsonl — filter explicitly by `S=32` for the headline.
- The shortlist of 6 archs unchanged: all 6 are valid at S=32
  (max T in shortlist = 13 = SubseqH8.T_max, kept = 32 − 13 + 1 = 20
  windows per example).
- Case studies (HH-RLHF understanding, AxBench steering) are
  unaffected — they don't use the sparse-probing aggregation; they
  forward Gemma-2-2b on case-study text directly.

### Branch strategy + DO-NOT-DEVIATE list

**Agent C works on its own branch `han-phase7-agent-c`**, not directly
on `han-phase7-unification`. Reasons:

- Agent A has done multiple force-pushes to `han-phase7-unification`
  during this session (commit-message amends, author-email fix). A
  force-push from Agent A would clobber Agent C's commits if both
  pushed to the same branch.
- Even without force-push, simultaneous `git push` from two agents
  races; one always loses.
- File-level conflicts are unlikely (different sub-dirs) but
  branch-level state-machine conflicts ARE likely.

**Workflow**:

```bash
# Agent C bootstrap (after pod setup):
cd /workspace/temp_xc
git fetch origin
git checkout -b han-phase7-agent-c origin/han-phase7-unification
# ... Agent C work ...
git push -u origin han-phase7-agent-c

# Periodic sync from Agent A's work:
git fetch origin han-phase7-unification
git merge origin/han-phase7-unification    # OR rebase if no merge artefacts yet

# When Agent C results are ready for integration:
# Open a PR han-phase7-agent-c → han-phase7-unification, request human review.
```

**DO NOT** (preserves the shared Phase 7 framework):

- DO NOT modify `experiments/phase7_unification/_paths.py`,
  `_train_utils.py`, `train_phase7.py`, `run_probing_phase7.py`,
  `build_act_cache_phase7.py`, `build_probe_cache_phase7.py`,
  or `canonical_archs.json`. These are Agent A's. If you need a
  field added to a meta dict, propose it via PR comment instead of
  editing.
- DO NOT touch any file under `experiments/phase5_*`,
  `experiments/phase5b_*`, `experiments/phase6_*`. Read-only.
- DO NOT write outside
  `experiments/phase7_unification/results/case_studies/` for
  outputs, and `experiments/phase7_unification/case_studies/` for
  code.
- DO NOT load Phase 7 ckpts via T-SAE's `load_dictionary()` — use
  `experiments.phase7_unification.run_probing_phase7._load_phase7_model`.
  Phase 7's ckpt format is `torch.save(state_fp16)`, not the T-SAE
  format, and a wrong loader silently returns the wrong tensors.
- DO NOT push to `origin/han` or `origin/han-phase7-unification`.
  PR-only.
- DO NOT change the canonical arch metadata (rows, k_pos, k_win,
  shifts, etc.) — those are pre-registered in `canonical_archs.json`
  and any change invalidates Agent A's leaderboard.

**OK to**:

- Read freely from `references/temporal_saes/`,
  `references/TemporalFeatureAnalysis/`, `papers/*.md`.
- Pull ckpts and meta from `han1823123123/txcdr-base`.
- Add new files under `experiments/phase7_unification/case_studies/`.
- Extend `case_studies/_paths.py` with Agent-C-specific constants.
- Add Agent-C-specific dependencies via `uv add` if needed (commit
  the lockfile change, document why).

### Coordination with Agents A and B

- **A → C dependency (per-ckpt, not seed-batch-level)**: Agent C
  doesn't train; it just needs the specific ckpts it'll analyse to
  exist on `han1823123123/txcdr-base`. Check status via
  `huggingface_hub.HfApi().list_repo_files('han1823123123/txcdr-base')`.
  As of 2026-04-26 ~11:20 UTC:
  - **Stage 1 (all 3 archs)**: ALREADY on HF.
    `topk_sae`, `tsae_paper_k500`, `agentic_txc_02` all pushed by
    Agent A — Agent C can start Stage 1 immediately.
  - **Stage 2**: 2 of 3 on HF (`mlc_contrastive_alpha100_batchtopk`,
    `phase5b_subseq_h8`); waiting on
    `phase57_partB_h8_bare_multidistance_t5` (Agent A's row 32,
    expected within 6-8h of Agent A's seed=42 batch start).
  Agent C should poll HF for the missing Stage 2 ckpt every ~30 min;
  no need to wait on a batch-level marker.
- **B → C dependency**: C.ii's "best feature per concept" lookup uses
  Agent B's autointerp labels at `txcdr-base/autointerp/<arch>/`.
  If Agent B hasn't labelled a given arch yet, C.ii falls back to
  picking by max activation on a concept-annotated text sample
  (slower but no dependency). Don't block on Agent B.
- **Anthropic API budget**: Agents B (autointerp) and C.ii (steering
  grader) both burn Anthropic credit. Coordinate to avoid rate-limit
  collisions: B uses Haiku 4.5 (small), C uses Sonnet 4.6 (grader).

### Risks

1. **AxBench protocol-on-TXC adapter**: AxBench was designed for
   per-token SAE intervention. TXC encodes T-token windows; "set
   feature i high during decode" needs translation. Likely answer:
   intervene on the per-window z, then the decoder output. Worth a
   careful 1-day spike before launching the full 1440-call run.
2. **Sonnet 4.6 vs Llama-3.3-70b grader skew**: different graders have
   different biases. Document the substitution as a deliberate
   methodology choice (cheaper, integrated auth). If a reviewer asks,
   we can re-grade a small sample with Llama-3.3-70b on the cheap as a
   sanity check.
3. **HH-RLHF dataset has version drift**: there are several variants
   on HF. Pick the canonical one used by AxBench / T-SAE
   (`Anthropic/hh-rlhf`) and document.
4. **Ckpt format compatibility**: Phase 7 ckpts use `torch.save(state_fp16)`,
   not the T-SAE repo's `load_dictionary` format. Use `_load_phase7_model`.
   Don't try to load Phase 7 ckpts via T-SAE's loader — it'll silently
   fail or load the wrong tensors.
