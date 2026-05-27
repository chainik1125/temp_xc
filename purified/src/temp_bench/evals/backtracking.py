"""§ 5.2 — Backtracking detection + inducement on DeepSeek-R1-Distill.

Two-axis evaluation (paper's "headline real-world win"):

- **Detection**: PR-AUC of a sparse-feature probe on labeled rollouts;
  measures how well the SAE features distinguish backtracking from
  non-backtracking spans. Uses
  :func:`temp_bench.evals.legacy.detection.detect_case_study` with
  within-window shuffle ablation.

- **Inducement**: a steering hook (encoder pre-image of the strongest
  backtracking feature) applied to the subject model, then measuring
  the increase in backtracking rate vs the unsteered baseline. Uses
  :mod:`temp_bench.evals.legacy.steering_hooks` +
  :mod:`temp_bench.evals.legacy.steering_protocols`.

PORT STATUS: skeleton — the legacy detection + steering modules are
preserved (under ``temp_bench/evals/legacy/``). Remaining port:

1. Rollout data + labels — port from
   ``origin/final:purified/src/temp_bench/data/nlp/ward.py`` (Ward
   stage B labels) into ``temp_bench/data/``.
2. The driver that maps (model, rollouts, labels) → headline metrics
   was in ``origin/final:purified/experiments/c7_backtracking/run.py``
   (+ ``analysis.py``). Port the orchestration into :meth:`BacktrackingEval.eval`.
3. Inducement: ``origin/final:purified/experiments/det_steer/run_c7_locked.py``
   has the inducement protocol; port into ``temp_bench/evals/backtracking.py``.
"""

from __future__ import annotations

from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator


class BacktrackingEval(Evaluator):
    """§ 5.2 — backtracking detection + inducement."""

    name = "backtracking"
    protocol_version = "1.0.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        raise NotImplementedError(
            "BacktrackingEval.eval is not yet ported. References:\n"
            "  - detection skeleton: temp_bench/evals/legacy/detection.py "
            "(detect_case_study with shuffle ablation)\n"
            "  - steering hooks:    temp_bench/evals/legacy/steering_*.py\n"
            "  - canonical driver:  origin/final:purified/experiments/"
            "c7_backtracking/run.py + analysis.py\n"
            "  - rollout data:      origin/final:purified/src/temp_bench/"
            "data/nlp/ward.py\n"
            "Port the rollout loader + the orchestration that combines "
            "detection PR-AUC + inducement gap-recovery into a single "
            "dict of metrics."
        )

    def primary_metric(self) -> str:
        return "detection_pr_auc"
