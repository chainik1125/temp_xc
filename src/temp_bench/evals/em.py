"""§ 5.3 — Emergent misalignment via Wang procedure.

Multi-stage screening protocol on Qwen-2.5-7B/14B with medical/finance
LoRA organisms:

- Stage 1: Δz̄ ranking (top-100 candidates)
- Stage 2: causal screen at α=±1 (top-20 survivors)
- Stage 3: per-survivor coh-aware sweep (top-3 finalists)
- Stage 4: 27-α frontier on top-3

Headline = peak alignment-injection score at coherence ≥ 30 (judge-
graded). Paper uses Anthropic Claude Haiku as the alignment + coherence
judge.

PORT STATUS: stub — judge calls cost ~$0.50/cell, deferred.

Reference: ``origin/final:experiments/c6_em/run.py`` has the
Wang driver. Port the multi-stage screening, judge client (with
caching), and result aggregation. Per-cell wall: ~3-5 hr at 30K steps
on Qwen-7B; ~$0.50 judge cost per cell.
"""

from __future__ import annotations

from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator


class EmergentMisalignmentEval(Evaluator):
    """§ 5.3 — Wang procedure on Qwen + medical/finance LoRA."""

    name = "em"
    protocol_version = "1.0.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        raise NotImplementedError(
            "EmergentMisalignmentEval.eval is not yet ported. "
            "Reference: origin/final:experiments/c6_em/run.py "
            "(Wang 4-stage driver). Required infra to port:\n"
            "  - Judge client with caching (Anthropic Haiku)\n"
            "  - LoRA-organism loader for Qwen subject model\n"
            "  - Per-cell ~3-5 hr training + ~$0.50 judge spend.\n"
            "Run with caution; budget the API spend."
        )

    def primary_metric(self) -> str:
        return "peak_align"
