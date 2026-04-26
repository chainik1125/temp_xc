# Phase 7 case studies (Agent C)

Reproduces T-SAE §4.5's two case studies on Phase 7's trained ckpt set:

1. **HH-RLHF dataset understanding** (`hh_rlhf/`)
2. **AxBench-style steering** (`steering/`)

Run on a 6-arch shortlist (see `_paths.SELECTED_ARCHS_FOR_CASE_STUDIES`).

## Setup

This sub-package runs on **Agent C's pod** — H100 80GB, 2 TB volume,
8 vCPU, 125 GB RAM. See `docs/han/research_logs/phase7_unification/agent_c_brief.md`
for the full plan + 9-day timeline.

```bash
# After bootstrap_claude.sh + uv sync (per RUNPOD_INSTRUCTIONS.md):
huggingface-cli download han1823123123/txcdr-base \
    ckpts/topk_sae__seed42.pt \
    ckpts/tsae_paper_k500__seed42.pt \
    ckpts/mlc_contrastive_alpha100_batchtopk__seed42.pt \
    ckpts/agentic_txc_02__seed42.pt \
    ckpts/phase5b_subseq_h8__seed42.pt \
    ckpts/phase57_partB_h8_bare_multidistance_t5__seed42.pt \
    --local-dir /workspace/temp_xc/experiments/phase7_unification/results/
```

## Reusable Phase 7 helpers

- **Loading any ckpt**: import
  `experiments.phase7_unification.run_probing_phase7._load_phase7_model`.
  Covers all 12 src_class types Phase 7 supports.
- **Path constants**: import from `case_studies._paths`.
- **Canonical arch metadata**: read `experiments/phase7_unification/canonical_archs.json`.

## Driver layout (planned)

- `hh_rlhf/build_hh_rlhf_cache.py` — tokenize + forward Gemma-2-2b base on
  HH-RLHF chosen + rejected pairs; cache per-token L12 activations.
- `hh_rlhf/decompose_hh_rlhf.py` — for each arch: encode HH-RLHF activations,
  rank features by t-stat (chosen vs rejected), produce per-arch top-K
  feature stats + length-correlation analysis.
- `hh_rlhf/label_top_features.py` — autointerp on top-K features per arch
  using top-activating contexts from HH-RLHF.
- `steering/select_features.py` — load Agent B's autointerp labels from
  HF; pick 30 semantic features per arch (or seed if unavailable).
- `steering/intervene_and_generate.py` — for each (arch, feature, strength)
  cell: steer-decode 60 tokens via Gemma-2-2b base; cache to JSONL.
- `steering/grade_with_sonnet.py` — call Claude Sonnet 4.6 grader on
  each generation; store success + coherence scores.
- `steering/plot_pareto.py` — per-arch (success, coherence) Pareto plot.

## Outputs

Per `_paths`:
- `results/case_studies/hh_rlhf/<arch_id>/feature_stats.json`
- `results/case_studies/hh_rlhf/<arch_id>/top_features.json`
- `results/case_studies/steering/<arch_id>/generations.jsonl`
- `results/case_studies/steering/<arch_id>/grades.jsonl`
- `results/plots/phase7_hh_rlhf_summary.png`
- `results/plots/phase7_steering_pareto.png`

## Reference code (T-SAE)

Cloned to `references/temporal_saes/`. Useful entries:
- `dictionary_learning/dictionary_learning/trainers/temporal_sequence_top_k.py`:
  the original `TemporalMatryoshkaBatchTopKSAE`. Confirms our Phase 7 port
  at `src/architectures/tsae_paper.py` is faithful.
- `src/data.py`, `src/activations.py`: dataset + activation utility patterns
  (Agent C can keep using HF AutoModelForCausalLM, doesn't need nnsight).

What's NOT in the T-SAE repo (must build from scratch):
- `alignment_study.py` (HH-RLHF understanding) — README mentions it but
  source is missing.
- Steering — README literally says "Under construction!" Build from the
  AxBench paper protocol (Wu et al. 2025, arXiv:2501.17148).
