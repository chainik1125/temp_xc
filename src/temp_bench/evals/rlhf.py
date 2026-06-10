"""§ 5.4 — HH-RLHF preference data decomposition.

Rank-based decomposition: for each (chosen, rejected) HH-RLHF pair,
encode both sequences with the SAE, find the K features whose mean
difference most correlates with the preference label. Headline =
rank-based AUC of preference recovery + per-feature top-K precision.

PORT STATUS: stub — preference loader + steering hooks need porting.

References:
- Steering primitives: ``temp_bench/evals/legacy/steering_*.py``
- Canonical driver:    ``origin/final:experiments/c5_steering/run.py``
- Preference data:     standard HH-RLHF release on HuggingFace.
"""

from __future__ import annotations

from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator


class RLHFEval(Evaluator):
    """§ 5.4 — HH-RLHF preference decomposition."""

    name = "rlhf"
    protocol_version = "1.0.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        raise NotImplementedError(
            "RLHFEval.eval is not yet ported. References:\n"
            "  - canonical driver: origin/final:experiments/"
            "c5_steering/run.py\n"
            "  - steering primitives: temp_bench/evals/legacy/steering_*.py\n"
            "Port the (chosen, rejected) loader + the rank-based "
            "decomposition + headline metric. Single seed sufficient for "
            "the paper headline; multi-seed nice-to-have."
        )

    def primary_metric(self) -> str:
        return "preference_auc"
