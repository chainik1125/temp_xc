"""Steering evaluation — C5 (RLHF sentiment, T-SAE paper § 4.5 + B.2).

Public API for the C5 case study. The implementation lives in
:mod:`temp_bench.case_studies.steering` (one file, single source of
truth) — this module re-exports the user-facing types so callers
import them by their evaluation purpose:

    from temp_bench.eval import steering
    cs = steering.SteeringCaseStudy(workspace=Path("results/runs/<eval_key>"))
    cs.setup()
    result = cs.evaluate(arch=my_arch, seed=42)
    metrics = result.metrics                # success_at_coh_<tau>, …

Two protocols, dispatched by :class:`SteeringConfig.protocol`:

- **V7 tiled-broadcast** — non-overlapping T-blocks, single uniform δ
  per block. Default for arch-uniformity.
- **PP per-position** — sliding T-window stride 1, full per-position
  delta written and averaged at overlaps. Fallback when V7 produces a
  degenerate success rate (e.g., subseq-encoder + multi-distance
  contrastive); see :meth:`SteeringCaseStudy.pre_test_v7`.

Judge: Sonnet 4.6 (default) for the paper's two heads (success,
coherence) on a 0–3 scale. Every call is persisted to
``judge_outputs.jsonl`` for post-deadline Cohen's κ validation —
:class:`SonnetSteeringJudge` enforces this.

The metrics returned to ``run_cell`` are flat
``dict[str, float]`` — see :func:`flatten_metrics` in
:mod:`temp_bench.case_studies.steering`. Aggregate (3-seed mean ±
stderr) lives in ``experiments/c5_steering/analysis.py``.
"""

from __future__ import annotations

from temp_bench.case_studies.steering import (  # noqa: F401
    ANCHOR_LAYER,
    CONCEPTS,
    DEFAULT_COH_THRESHOLDS,
    DEFAULT_STRENGTHS,
    SUBJECT_MODEL,
    FeatureSelection,
    Generation,
    Grade,
    SonnetSteeringJudge,
    SteeringCaseStudy,
    SteeringConfig,
    coh_success_curves,
    flatten_metrics,
    generate_steered_continuations,
    get_concept,
    select_best_features,
)


__all__ = [
    "ANCHOR_LAYER",
    "CONCEPTS",
    "DEFAULT_COH_THRESHOLDS",
    "DEFAULT_STRENGTHS",
    "SUBJECT_MODEL",
    "FeatureSelection",
    "Generation",
    "Grade",
    "SonnetSteeringJudge",
    "SteeringCaseStudy",
    "SteeringConfig",
    "coh_success_curves",
    "flatten_metrics",
    "generate_steered_continuations",
    "get_concept",
    "select_best_features",
]
