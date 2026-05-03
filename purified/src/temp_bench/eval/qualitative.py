"""Qualitative latent evaluation — C4 (Top-256 cumulative SEMANTIC Pareto).

**Single metric** by decision: Top-256 cumulative SEMANTIC Pareto. We
do NOT use pdvar, paper-style probe variants, or anything else.
See ``docs/components/c4.md`` and ``decisions.md``.

**Judge-output persistence is mandatory.** :func:`top_256_semantic`
must append every judge call to
``results/runs/<eval_key>/judge_outputs.jsonl`` with fields
``{feature_id, context_id, judge_id, label, prompt_hash, judge_model,
ts}``. This is what lets us defer Cohen's κ validation to a paper-end
stretch task: post-deadline, ``pandas.read_json(judge_outputs.jsonl)``
+ ``scipy.stats.cohen_kappa_score(human_labels, judge_labels)`` is
all that's needed to land κ — no re-judging, no re-training, no
re-evaluating.

Public API (worker fills in):

- :func:`top_256_semantic(model, datasource, judges, n_jobs=-1) -> dict`
- :func:`pareto_frontier(metrics_per_arch_seed) -> list[(arch, x, y)]`

The judge is an ensemble of 2 Haiku judges + majority vote. Judge
calls go through ``temp_bench.utils.judge_dispatch`` (not yet ported).
"""

from __future__ import annotations

from typing import Any

from temp_bench.architectures.base import TempBenchArch


def top_256_semantic(
    model: TempBenchArch,
    *,
    datasource_name: str,
    n_features: int = 256,
    judge_model: str = "claude-haiku-4-5-20251001",
    n_jobs: int = -1,
) -> dict[str, float]:
    """Top-256 cumulative SEMANTIC Pareto.

    Procedure (T-SAE paper § 4.2-style, single metric per c4.md):

    1. Variance-rank features by activation variance on ``datasource``.
    2. Take top-256 features.
    3. For each, mine ~10 max-activating passages.
    4. Send to judge ensemble (2× Haiku + majority).
    5. Cumulative SEMANTIC = fraction of judged features that are
       'semantic' rather than syntactic / lexical.

    Returns dict with ``"top_256_semantic"`` (the headline number),
    ``"n_features_judged"``, ``"judge_agreement"``.

    TODO — port from
    ``origin/han-phase7-unification:src/qualitative/passage_probe.py``.
    """
    raise NotImplementedError(
        "top_256_semantic — port from Phase 6 qualitative pipeline."
    )


def pareto_frontier(points: list[tuple[Any, float, float]]) -> list[tuple[Any, float, float]]:
    """Compute the upper-right Pareto frontier from a list of
    ``(label, x, y)`` triples.

    Used by C4 to draw a frontier through (probing AUC, top-256 SEMANTIC)
    points. Pure utility, no external dependency.
    """
    if not points:
        return []
    by_x = sorted(points, key=lambda p: p[1], reverse=True)
    frontier: list[tuple[Any, float, float]] = []
    best_y = float("-inf")
    for p in by_x:
        if p[2] > best_y:
            frontier.append(p)
            best_y = p[2]
    return frontier
