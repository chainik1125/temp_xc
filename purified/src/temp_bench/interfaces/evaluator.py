"""Evaluator ABC — the contract every evaluation module honors.

An Evaluator takes a trained :class:`TempBenchArch` plus an ``EvalSpec``
(arbitrary kwargs the evaluator interprets) and returns a flat
``dict[str, float]`` of metrics that lands directly in the leaderboard
row.

Non-float diagnostics (per-task arrays, judge transcripts, intermediate
plots) belong in a per-run directory (``results/runs/<eval_key>/``).
The metric dict is the *headline*; everything else is supplementary.

There is one Evaluator per paper section:

- ``synthetic_recovery`` (§ 4) — eAUC, gAUC, NMSE vs synthetic
  ground-truth features.
- ``probing`` (§ 5.1) — 36-task SAEBench probing AUC on Gemma.
- ``backtracking`` (§ 5.2) — detection PR-AUC + inducement gap-recovery
  on DeepSeek-R1.
- ``em`` (§ 5.3) — Wang procedure on Qwen + medical/finance LoRA.
- ``rlhf`` (§ 5.4) — HH-RLHF preference decomposition.

Each evaluator owns its protocol_version. Bump it to invalidate cached
eval rows (forces re-eval of every train_key under this evaluator).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from temp_bench.interfaces.architecture import TempBenchArch


@dataclass
class EvalSpec:
    """Argument bundle passed to ``Evaluator.eval(model, spec)``.

    Generic by design: each evaluator interprets its own keys. The
    runner does not introspect ``extra`` — it only ensures the spec is
    JSON-serialisable (so it hashes into ``eval_key``).

    Fields:
      - ``datasource``: registry key (mirrors what was used at training
        time). Evaluators that need to reconstruct the training data
        (e.g. synthetic feature recovery) look this up.
      - ``data_key``: hash of the datasource spec (a registry-key-stable
        identifier for the data the model was trained on).
      - ``smoke``: bool; if True, run a fast/small validation pass.
      - ``extra``: free-form per-task knobs (k_feat, S, alpha grid,
        judge model, …). Goes into ``eval_key`` so each tuple is a
        distinct cache cell.
    """

    datasource: str
    data_key: str
    smoke: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """Subclass for every paper-section evaluation.

    Subclasses set ``name`` (stable, lowercase, hyphenated — used in
    ``eval_key``) and ``protocol_version`` (semver — bump to invalidate
    cached rows).
    """

    name: str = "BASE"
    protocol_version: str = "0.0.0"

    @abstractmethod
    def eval(
        self,
        model: TempBenchArch,
        spec: EvalSpec,
    ) -> dict[str, float]:
        """Compute the leaderboard-row metrics for one trained model.

        MUST be deterministic given the same ``(model state_dict,
        spec)``. The runner pre-loads the model in eval mode and pins
        the device; the evaluator only computes.
        """

    def primary_metric(self) -> str:
        """Name of the metric in the returned dict that is the headline
        for this evaluator. Used by leaderboard sorting + paper render.

        Default: 'mean_auc'. Subclasses override if appropriate.
        """
        return "mean_auc"
