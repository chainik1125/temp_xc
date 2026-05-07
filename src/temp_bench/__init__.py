"""temp-bench: paper-ready framework for temporal crosscoder evaluation.

This package is the only code supporting the paper. Everything outside
```` (i.e. the wasteland) is reference-only and must not be imported.

The framework is built on the principles in ``docs/paper/framework.md``.
**Read that document before writing any experiment code.**

Public API:

    temp_bench.config          ──  yaml loaders + cache-key computation
    temp_bench.schemas         ──  Pydantic models (LeaderboardRow, …)
    temp_bench.cache           ──  checkpoint + leaderboard ops (the only writers)
    temp_bench.runner          ──  ``run_cell`` — the canonical pathway
    temp_bench.architectures   ──  locked arch implementations
    temp_bench.data            ──  toy generators, NLP activation cache
    temp_bench.training        ──  shared training loop, BrickenConfig
    temp_bench.eval            ──  metrics + the CaseStudy abstract base
    temp_bench.case_studies    ──  C5 steering, C6 EM, C7 backtracking
    temp_bench.plotting        ──  save_figure helper
    temp_bench.utils           ──  seeding helpers
"""

from temp_bench import cache, config, runner, schemas  # noqa: F401

__version__ = "0.2.0"
