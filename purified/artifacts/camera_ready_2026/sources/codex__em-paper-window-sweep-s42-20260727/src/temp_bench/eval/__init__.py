"""Evaluation API.

- :mod:`temp_bench.eval.synthetic` — toy-data NMSE / AUC / gAUC (C1, C2)
- :mod:`temp_bench.eval.probing`   — sparse probing (C3)
- :mod:`temp_bench.eval.qualitative` — passage probe + var/pdvar (C4)
- :mod:`temp_bench.eval.case_study` — :class:`CaseStudy` ABC (C5/C6/C7)

Parallelism contract (all eval modules):

- Eval modules accept ``n_jobs`` in their config (default ``-1`` = all
  cores). Used inside ``joblib.Parallel`` for embarrassingly-parallel
  inner loops (probing tasks, judge calls, etc.).
- Cell-level parallelism (concurrent (arch, seed) cells across multiple
  GPUs on a shared pod) flows through
  ``temp_bench.runner.run_concurrent_cells`` (TODO).
- The pod-wide vCPU and VRAM budgets are documented in
  ``docs/paper/hardware.md``.
"""

from temp_bench.eval.case_study import CaseStudy  # noqa: F401
from temp_bench.eval import qualitative, probing, steering, synthetic  # noqa: F401
