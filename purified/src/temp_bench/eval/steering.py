"""Steering evaluation — C5 (RLHF sentiment, T-SAE paper § 4.4).

Per-token decoder-row addition at the residual stream, two steering
protocols:

- **V7 tiled-broadcast** — stride-T blocks, single uniform δ per block.
  Default for arch-uniformity.
- **PP per-position** — fallback when V7 breaks an arch (TXC-pro's
  subseq encoder + multi-distance contrastive may not survive V7;
  agent_steer pre-tests at coh threshold 2.0 before locking V7).

Judge: Gemini, two heads (coherence, success). Judge calls go through
``temp_bench.utils.judge_dispatch`` (not yet ported).

Public API (worker fills in):

- :func:`steer_v7_tiled(model, prompts, magnitude, T_block) -> list[str]`
- :func:`steer_pp(model, prompts, magnitude_per_pos) -> list[str]`
- :func:`coh_success_curve(model, prompts, judge, magnitudes) -> dict`
"""

from __future__ import annotations

from typing import Any

from temp_bench.architectures.base import TempBenchArch


def steer_v7_tiled(
    model: TempBenchArch,
    *,
    prompts: list[str],
    feature_idx: int,
    magnitude: float,
    T_block: int | None = None,
) -> list[str]:
    """V7 tiled-broadcast steering. Returns generated continuations.

    TODO — port from
    ``origin/han-phase7-unification:src/case_studies/steering_v7.py``.
    """
    raise NotImplementedError(
        "steer_v7_tiled — port from Phase 7 unified-pareto steering pipeline."
    )


def steer_pp(
    model: TempBenchArch,
    *,
    prompts: list[str],
    feature_idx: int,
    magnitude_per_pos: list[float],
) -> list[str]:
    """Per-position steering (fallback when V7 breaks an arch). TODO."""
    raise NotImplementedError("steer_pp — port from Phase 5 steering baseline.")


def coh_success_curve(
    model: TempBenchArch,
    *,
    prompts: list[str],
    judge,
    feature_idx: int,
    magnitudes: list[float],
    coh_thresholds: tuple[float, ...] = (1.5, 1.75, 2.0, 2.25, 2.5),
    protocol: str = "v7",
) -> dict[str, list[float]]:
    """Coherence-vs-success curve at multiple coherence thresholds.

    Returns dict with ``"success_at_coh"`` (one list per threshold),
    plus per-magnitude raw judge scores. The agent pre-tests V7 on
    each arch (1 cell at coh threshold 2.0) before running the full
    sweep — fall back to ``protocol='pp'`` if V7 produces degenerate
    success rates.
    """
    raise NotImplementedError(
        "coh_success_curve — port from Phase 7 unified-pareto steering."
    )
