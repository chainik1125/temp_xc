"""temp-bench: paper-ready framework for temporal crosscoder evaluation.

This package is the only code supporting the paper. Everything outside
``purified/`` (i.e. the wasteland) is reference-only and must not be imported.

Public API (still skeletal — agents fill in as they go):

    temp_bench.architectures   ──  TopK-SAE, T-SAE, TXC-base, TXC-pro
    temp_bench.data            ──  toy generators, NLP activation cache
    temp_bench.training        ──  shared training loop
    temp_bench.eval            ──  metrics + the CaseStudy abstract base
    temp_bench.case_studies    ──  C5 steering, C6 EM, C7 backtracking
    temp_bench.plotting        ──  save_figure helper
    temp_bench.utils           ──  seeding, device, run-id, leaderboard append
"""

__version__ = "0.1.0"
