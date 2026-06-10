"""§ 5.1 — Sparse probing on Gemma-2-2B-IT layer 13.

For each (arch, seed) trained on the Gemma activation cache, run the
36-task SAEBench probing suite. Headline = mean ROC AUC over tasks.

PORT STATUS: stub — interface in place, full logic awaits port from
``origin/final:experiments/c3_probing/run.py`` +
``origin/final:src/temp_bench/data/nlp/probe_cache.py``.

Steps to complete the port (~3-4 hr work):
1. Port ``data/nlp/probe_cache.py`` and ``probe_tasks.py`` from
   origin/final into ``temp_bench/data/probe_cache.py`` (probe cache
   build + 36-task loaders).
2. Port the canonical probing logic (top-k feature selection by class-
   mean diff + L1 logistic regression) from ``c3_probing/run.py``.
3. Wire :meth:`ProbingEval.eval` to call those + return ``mean_auc`` +
   per-task AUC dict + ``std_auc`` (across tasks).
"""

from __future__ import annotations

from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator


class ProbingEval(Evaluator):
    """SAEBench 36-task sparse probing on Gemma."""

    name = "probing"
    protocol_version = "1.0.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        raise NotImplementedError(
            "ProbingEval.eval is not yet ported. "
            "Reference: origin/final:experiments/c3_probing/run.py "
            "(my_eval_fn) + origin/final:src/temp_bench/data/nlp/"
            "{probe_cache,probe_tasks}.py. Port both into temp_bench/data/ "
            "and replace this stub with the canonical probing protocol."
        )

    def primary_metric(self) -> str:
        return "mean_auc"
