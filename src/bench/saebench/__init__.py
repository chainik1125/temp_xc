"""SAEBench sparse-probing integration.

Wraps EleutherAI's SAEBench (https://github.com/adamkarvonen/SAEBench)
to run sparse-probing evals on our TopKSAE / MLC / TempXC architectures.

Public surface:
  - configs: single source of truth for protocols, k-values, T-values,
    aggregation names, Gemma-2-2B L12 target.
  - aggregation: four strategies for collapsing (T, d_sae) per-window
    activations into the (L, d_sae) shape SAEBench expects.
  - matching_protocols: protocol A vs B, parameterized by T.
  - saebench_wrapper.SAEBenchAdapter: ArchSpec → SAEBench BaseSAE bridge.
  - probing_runner.run_probing: call SAEBench's sparse_probing eval and
    emit one JSONL record per (arch, T, protocol, aggregation, task, k).

Pre-registration: docs/aniket/experiments/sparse_probing/plan.md
Exploration notes: docs/aniket/experiments/sparse_probing/saebench_notes.md
"""
