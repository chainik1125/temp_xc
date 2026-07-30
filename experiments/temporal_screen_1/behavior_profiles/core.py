"""Shared, model-agnostic summaries for temporal behaviour profiles.

The primary coordinate is normalized progress in an observed input or
rollout.  Event-aligned behaviours may additionally use signed token offset.
These helpers deliberately avoid assuming that every behaviour has a discrete
onset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


DEFAULT_REVEAL_FRACTIONS = (
    0.0,
    0.125,
    0.25,
    0.375,
    0.5,
    0.625,
    0.75,
    0.875,
    1.0,
)


@dataclass(frozen=True)
class TurnOnSummary:
    """Compact summary of a monotone-ish response over normalized progress."""

    start: float
    end: float
    half_rise_fraction: float | None
    effective_fraction_95: float | None
    normalized_area: float

    def to_dict(self) -> dict[str, float | None]:
        return {
            "start": self.start,
            "end": self.end,
            "half_rise_fraction": self.half_rise_fraction,
            "effective_fraction_95": self.effective_fraction_95,
            "normalized_area": self.normalized_area,
        }


@dataclass(frozen=True)
class SpatialMediationSummary:
    """Causal contribution of sequence positions beyond the current token.

    Every scalar must use the same behavior metric, oriented so larger values
    mean more of the target behavior.  Necessity is the clean primary result:
    it compares current-token-only ablation with sequence-wide ablation.
    Sufficiency is optional because repeated all-position addition generally
    injects more total intervention energy than current-token-only addition.
    """

    baseline_target: float
    current_token_ablated: float
    all_positions_ablated: float
    current_token_necessity: float
    all_positions_necessity: float
    sequence_support_gap: float
    baseline_neutral: float | None = None
    current_token_added: float | None = None
    all_positions_added: float | None = None
    current_token_sufficiency: float | None = None
    all_positions_sufficiency: float | None = None
    sequence_sufficiency_gap: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def reveal_counts(
    n_tokens: int,
    fractions: Sequence[float] = DEFAULT_REVEAL_FRACTIONS,
) -> list[int]:
    """Map normalized reveal fractions to unique token-prefix lengths."""

    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")
    counts: list[int] = []
    for fraction in fractions:
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("reveal fractions must lie in [0, 1]")
        count = int(np.ceil(float(fraction) * n_tokens))
        if not counts or count != counts[-1]:
            counts.append(count)
    if not counts or counts[-1] != n_tokens:
        counts.append(n_tokens)
    return counts


def nearest_length_matching(
    positive_lengths: Sequence[int],
    neutral_lengths: Sequence[int],
) -> list[int]:
    """Return a one-to-one neutral index for every positive item.

    In one dimension, pairing the sorted samples is optimal when both sets
    have the same size.  When the neutral pool is larger, this deterministic
    greedy nearest-neighbour pass gives a transparent length match without
    importing a task-specific semantic ontology.
    """

    if len(neutral_lengths) < len(positive_lengths):
        raise ValueError("neutral pool must be at least as large as positive pool")
    available = sorted(
        ((int(length), index) for index, length in enumerate(neutral_lengths)),
        key=lambda item: (item[0], item[1]),
    )
    result = [-1] * len(positive_lengths)
    for length, positive_index in sorted(
        ((int(length), index) for index, length in enumerate(positive_lengths)),
        key=lambda item: (item[0], item[1]),
    ):
        insertion = int(
            np.searchsorted(
                np.asarray([item[0] for item in available]),
                length,
                side="left",
            )
        )
        candidates = []
        if insertion < len(available):
            candidates.append(insertion)
        if insertion > 0:
            candidates.append(insertion - 1)
        selected = min(
            candidates,
            key=lambda idx: (
                abs(available[idx][0] - length),
                available[idx][0],
                available[idx][1],
            ),
        )
        _, neutral_index = available.pop(selected)
        result[positive_index] = neutral_index
    return result


def binary_auc(positive: Iterable[float], negative: Iterable[float]) -> float:
    """AUC with exact half credit for ties, implemented without sklearn."""

    pos = np.asarray(list(positive), dtype=float)
    neg = np.asarray(list(negative), dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(
        (
            (pos[:, None] > neg[None, :]).sum()
            + 0.5 * (pos[:, None] == neg[None, :]).sum()
        )
        / (pos.size * neg.size)
    )


def turn_on_summary(
    fractions: Sequence[float],
    values: Sequence[float],
    *,
    endpoint_tolerance_fraction: float = 0.05,
) -> TurnOnSummary:
    """Summarize a response curve without fitting a decay law.

    ``half_rise_fraction`` is the first observed point crossing half of the
    signed endpoint change. ``effective_fraction_95`` is the earliest point
    after which every remaining observation stays within 5% of the endpoint
    change from the full-input value.  Either can be ``None`` for a flat or
    non-settling curve.
    """

    x = np.asarray(fractions, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("fractions and values must be equal-length 1D arrays")
    if np.any(np.diff(x) <= 0):
        raise ValueError("fractions must be strictly increasing")
    start = float(y[0])
    end = float(y[-1])
    delta = end - start
    area = float(np.trapezoid(y, x) / (x[-1] - x[0]))
    if abs(delta) <= 1e-12:
        return TurnOnSummary(start, end, None, None, area)

    signed_progress = (y - start) / delta
    half_indices = np.flatnonzero(signed_progress >= 0.5)
    half = float(x[half_indices[0]]) if half_indices.size else None

    tolerance = max(abs(delta) * endpoint_tolerance_fraction, 1e-12)
    effective = None
    for index in range(len(x)):
        if np.all(np.abs(y[index:] - end) <= tolerance):
            effective = float(x[index])
            break
    return TurnOnSummary(start, end, half, effective, area)


def paired_bootstrap_curve(
    values: np.ndarray,
    *,
    n_bootstrap: int = 2_000,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Mean and paired-bootstrap intervals for ``(sample, progress)`` values."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("values must have shape (sample, progress)")
    rng = np.random.default_rng(seed)
    means = array.mean(axis=0)
    boot = np.empty((n_bootstrap, array.shape[1]), dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, array.shape[0], size=array.shape[0])
        boot[index] = array[sample].mean(axis=0)
    return {
        "mean": means.tolist(),
        "ci_low": np.quantile(boot, 0.025, axis=0).tolist(),
        "ci_high": np.quantile(boot, 0.975, axis=0).tolist(),
    }


def spatial_mediation_summary(
    *,
    baseline_target: float,
    current_token_ablated: float,
    all_positions_ablated: float,
    baseline_neutral: float | None = None,
    current_token_added: float | None = None,
    all_positions_added: float | None = None,
) -> SpatialMediationSummary:
    """Summarize current-token versus sequence-wide causal mediation.

    ``sequence_support_gap`` is

    ``metric(current-token ablation) - metric(all-position ablation)``.

    It is in the behavior metric's natural units and avoids entropy or
    cross-entropy proxies.  A positive value says that removing the feature
    outside the current token has additional causal effect.  It does not by
    itself distinguish diffuse support from one earlier bottleneck; lag-band
    localization is the next stage.
    """

    values = [
        baseline_target,
        current_token_ablated,
        all_positions_ablated,
    ]
    optional = [baseline_neutral, current_token_added, all_positions_added]
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("mediation metrics must be finite")
    supplied = [value is not None for value in optional]
    if any(supplied) and not all(supplied):
        raise ValueError(
            "baseline_neutral and both addition metrics must be supplied together"
        )
    if all(supplied) and not np.isfinite(np.asarray(optional, dtype=float)).all():
        raise ValueError("addition metrics must be finite")

    current_necessity = float(baseline_target - current_token_ablated)
    all_necessity = float(baseline_target - all_positions_ablated)
    sequence_gap = float(current_token_ablated - all_positions_ablated)
    if not all(supplied):
        return SpatialMediationSummary(
            baseline_target=float(baseline_target),
            current_token_ablated=float(current_token_ablated),
            all_positions_ablated=float(all_positions_ablated),
            current_token_necessity=current_necessity,
            all_positions_necessity=all_necessity,
            sequence_support_gap=sequence_gap,
        )

    assert baseline_neutral is not None
    assert current_token_added is not None
    assert all_positions_added is not None
    current_sufficiency = float(current_token_added - baseline_neutral)
    all_sufficiency = float(all_positions_added - baseline_neutral)
    return SpatialMediationSummary(
        baseline_target=float(baseline_target),
        current_token_ablated=float(current_token_ablated),
        all_positions_ablated=float(all_positions_ablated),
        current_token_necessity=current_necessity,
        all_positions_necessity=all_necessity,
        sequence_support_gap=sequence_gap,
        baseline_neutral=float(baseline_neutral),
        current_token_added=float(current_token_added),
        all_positions_added=float(all_positions_added),
        current_token_sufficiency=current_sufficiency,
        all_positions_sufficiency=all_sufficiency,
        sequence_sufficiency_gap=float(
            all_positions_added - current_token_added
        ),
    )
