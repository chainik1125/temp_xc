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

### Selected arch shortlist (6 archs)

Running case studies on all 49 archs would take ~6 days of GPU; with
the NeurIPS deadline 2026-05-05 we can't. Pick 6 representatives that
span the design space:

| arch_id | row | family | rationale |
|---|---|---|---|
| `topk_sae` | 1 | per-token SAE | literature baseline; the "naive" comparison |
| `tsae_paper_k500` | 2 | T-SAE port | direct head-to-head with the paper's flagship method |
| `mlc_contrastive_alpha100_batchtopk` | 5 | MLC | Phase 5 lp leader; multi-layer crosscoder reference |
| `agentic_txc_02` | 8 | TXC + multi-scale matryoshka | Phase 5 mp winner |
| `phase5b_subseq_h8` | 13 | SubseqH8 | Phase 5B mp champion |
| `phase57_partB_h8_bare_multidistance_t5` | 32 | H8 multi-distance | Phase 5 mp peak at IT regime; vanilla H8 reference |

If time allows, add `tfa_big` (row 7) and one anchor cell for completeness.

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

#### C.ii — AxBench-style steering

Per-arch:

1. Pick 30 features that the autointerp protocol (Agent B's pipeline,
   reused) labels as semantic (not punctuation/whitespace/syntax).
2. For each feature, generate **8 variants** at steering strengths
   ∈ {0.5, 1, 2, 4, 8, 12, 16, 24} of the unit-norm-decoder direction.
3. Generate ~60 tokens per variant.
4. Grade with Claude Sonnet 4.6 along two axes (per AxBench protocol):
   - **Steering success**: does output contain feature semantics? 0–3
   - **Coherence**: is output coherent text? 0–3
5. Plot Pareto: per-arch (success, coherence) means across 30 features.

Output:
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

### Coordination with Agents A and B

- **A → C dependency**: Agent C waits for Agent A's `seed42_complete.json`
  marker on `txcdr-base` (signals all 6 selected ckpts are uploaded).
  Until then Agent C does Day-1 setup work.
- **B → C dependency**: C.ii's "30 semantic features" selection benefits
  from Agent B's autointerp labels at `txcdr-base/autointerp/<arch>/`.
  Agent B's pipeline already runs autointerp on every arch's top-256
  features; Agent C filters those for `label.semantic == True`.
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
