"""temp_bench v2 — paper-ready framework for temporal crosscoder evaluation.

Read ``docs/framework.md`` before touching code. That document is
the framework's spec.

Public layout (matches docs/framework.md "Directory layout"):

    temp_bench.core          ──  runner, cache, schemas, config, trainer, code_version
    temp_bench.interfaces    ──  TempBenchArch, BatchIter, Evaluator ABCs
    temp_bench.archs         ──  locked architectures (registry-driven)
    temp_bench.evals         ──  paper-section evaluators (one per § 4 / § 5.x)
    temp_bench.data          ──  buffers + synthetic generators + real_lm cache
    temp_bench.training      ──  bricken (anti-dead resample plug-in)
    temp_bench.plotting      ──  save_figure helper
    temp_bench.utils         ──  seed, gpu_locks, shuffles, tokens
"""

__version__ = "2.0.0"
