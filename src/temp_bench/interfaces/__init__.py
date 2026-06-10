"""Contracts that every plugin in ``temp_bench`` honors.

Three ABCs / protocols define the surface area:

- :class:`temp_bench.interfaces.architecture.TempBenchArch` — every
  architecture under ``temp_bench/archs/`` subclasses this.
- :class:`temp_bench.interfaces.batch_iter.BatchIter` — the training-data
  contract (token-level or window-level depending on
  ``arch.consumes``).
- :class:`temp_bench.interfaces.evaluator.Evaluator` — every evaluation
  under ``temp_bench/evals/`` subclasses this.

The runner / trainer / cache only depend on these contracts. Concrete
archs, evals, and data sources are loaded via the YAML registry.
"""

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch
from temp_bench.interfaces.batch_iter import BatchIter, TokenBatchIter, WindowBatchIter
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator

__all__ = [
    "ArchConfig",
    "TempBenchArch",
    "BatchIter",
    "TokenBatchIter",
    "WindowBatchIter",
    "EvalSpec",
    "Evaluator",
]
