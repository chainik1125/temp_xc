"""Cross-fitted temporal state profiles for emergent misalignment.

Emergent misalignment (EM) does not have a natural token-level onset.  This
module therefore estimates a *state profile*: how well a fixed direction,
learned only from the end of held-in rollouts, separates eventually
misaligned from aligned rollouts as their generated responses unfold.

The primary estimator is deliberately conservative:

- keep only coherent, high-confidence endpoint labels;
- fit and test on different prompt groups;
- construct the direction from within-prompt class contrasts, so prompt
  identity cannot create it;
- evaluate AUC within each held-out prompt before averaging prompts.

The resulting curve is retrospective decodability, not a causal claim and
not a literal onset time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SelectedRollouts:
    """Indices and binary endpoint labels retained for the EM profile."""

    indices: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    n_positive: int
    n_negative: int
    n_excluded: int


def select_coherent_extremes(
    rows: Sequence[dict],
    *,
    coherence_min: float = 70.0,
    positive_alignment_max: float = 50.0,
    negative_alignment_min: float = 75.0,
    group_key: str = "prompt_index",
) -> SelectedRollouts:
    """Select coherent EM-positive and aligned-negative rollout endpoints.

    Low alignment is the EM-positive class.  Ambiguous alignment scores and
    incoherent responses are excluded rather than forced into either class.
    """

    selected: list[int] = []
    labels: list[int] = []
    groups: list[int] = []
    for index, row in enumerate(rows):
        alignment = float(row["alignment"])
        coherence = float(row["coherence"])
        if coherence < coherence_min:
            continue
        if alignment <= positive_alignment_max:
            label = 1
        elif alignment >= negative_alignment_min:
            label = 0
        else:
            continue
        selected.append(index)
        labels.append(label)
        groups.append(int(row[group_key]))

    label_array = np.asarray(labels, dtype=np.int8)
    return SelectedRollouts(
        indices=np.asarray(selected, dtype=np.int64),
        labels=label_array,
        groups=np.asarray(groups, dtype=np.int64),
        n_positive=int(label_array.sum()),
        n_negative=int((label_array == 0).sum()),
        n_excluded=len(rows) - len(selected),
    )


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact binary AUC with half credit for ties."""

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _unit(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("cannot normalize a zero or non-finite direction")
    return value / norm


def _fit_group_balanced_direction(
    terminal: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Average unit positive-minus-negative contrasts across prompt groups."""

    contrasts = []
    contrast_groups = []
    for group in np.unique(groups):
        mask = groups == group
        positive = terminal[mask & (labels == 1)]
        negative = terminal[mask & (labels == 0)]
        if not len(positive) or not len(negative):
            continue
        contrast = positive.mean(axis=0) - negative.mean(axis=0)
        try:
            contrasts.append(_unit(contrast))
        except ValueError:
            continue
        contrast_groups.append(int(group))
    if not contrasts:
        raise ValueError("training fold contains no nonzero within-group contrast")
    return _unit(np.mean(contrasts, axis=0)), contrast_groups


def crossfit_terminal_direction_scores(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Fit terminal directions leave-one-group-out and score all progress.

    Parameters
    ----------
    activations:
        Array of shape ``(rollout, progress, d_model)``.
    labels:
        Binary endpoint EM labels.
    groups:
        Prompt identifiers.  Every rollout from a held-out prompt is scored by
        a direction which never saw that prompt.
    """

    values = np.asarray(activations, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups)
    if values.ndim != 3:
        raise ValueError("activations must have shape (rollout, progress, d_model)")
    if len(labels) != len(values) or len(groups) != len(values):
        raise ValueError("rollout, label, and group dimensions differ")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(values).all():
        raise ValueError("activations contain non-finite values")

    scores = np.full(values.shape[:2], np.nan, dtype=float)
    folds: dict[int, dict] = {}
    for held_out in np.unique(groups):
        test = groups == held_out
        train = ~test
        direction, contrast_groups = _fit_group_balanced_direction(
            values[train, -1, :],
            labels[train],
            groups[train],
        )
        scores[test] = np.einsum("npd,d->np", values[test], direction)
        folds[int(held_out)] = {
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "training_contrast_groups": contrast_groups,
            "direction_norm": float(np.linalg.norm(direction)),
        }
    return scores, folds


def crossfit_positionwise_direction_scores(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Fit a separate leave-one-group-out direction at every progress point.

    This is the trajectory-changing complement to the transported terminal
    direction.  It asks whether *some* shared linear readout exists at each
    point, even if the readout rotates as the response unfolds.
    """

    values = np.asarray(activations, dtype=float)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups)
    if values.ndim != 3:
        raise ValueError("activations must have shape (rollout, progress, d_model)")
    if len(labels) != len(values) or len(groups) != len(values):
        raise ValueError("rollout, label, and group dimensions differ")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(values).all():
        raise ValueError("activations contain non-finite values")

    scores = np.full(values.shape[:2], np.nan, dtype=float)
    folds: dict[int, dict] = {}
    for held_out in np.unique(groups):
        test = groups == held_out
        train = ~test
        progress_details = []
        for progress_index in range(values.shape[1]):
            try:
                direction, contrast_groups = _fit_group_balanced_direction(
                    values[train, progress_index, :],
                    labels[train],
                    groups[train],
                )
            except ValueError:
                # Progress zero is the shared final prompt token.  Endpoint
                # labels cannot define a within-prompt direction there, so
                # record the structurally correct chance score.
                scores[test, progress_index] = 0.0
                progress_details.append(
                    {
                        "progress_index": progress_index,
                        "training_contrast_groups": [],
                        "direction_norm": 0.0,
                        "null_readout": True,
                    }
                )
                continue
            scores[test, progress_index] = values[test, progress_index, :] @ direction
            progress_details.append(
                {
                    "progress_index": progress_index,
                    "training_contrast_groups": contrast_groups,
                    "direction_norm": float(np.linalg.norm(direction)),
                    "null_readout": False,
                }
            )
        folds[int(held_out)] = {
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "progress": progress_details,
        }
    return scores, folds


def within_group_auc_curve(
    labels: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Evaluate a score curve without allowing prompt prevalence leakage."""

    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(labels):
        raise ValueError("scores must have shape (rollout, progress)")

    eligible = []
    per_group: dict[str, list[float]] = {}
    pair_counts: dict[str, int] = {}
    for group in np.unique(groups):
        mask = groups == group
        n_positive = int(np.count_nonzero(labels[mask] == 1))
        n_negative = int(np.count_nonzero(labels[mask] == 0))
        if not n_positive or not n_negative:
            continue
        key = str(int(group))
        eligible.append(int(group))
        per_group[key] = [
            binary_auc(labels[mask], scores[mask, progress])
            for progress in range(scores.shape[1])
        ]
        pair_counts[key] = n_positive * n_negative

    if not eligible:
        raise ValueError("no group contains both endpoint classes")
    matrix = np.asarray([per_group[str(group)] for group in eligible], dtype=float)
    weights = np.asarray(
        [pair_counts[str(group)] for group in eligible],
        dtype=float,
    )
    return {
        "eligible_groups": eligible,
        "n_eligible_groups": len(eligible),
        "n_within_group_pairs": int(weights.sum()),
        "macro_auc": matrix.mean(axis=0).tolist(),
        "pair_weighted_auc": np.average(matrix, axis=0, weights=weights).tolist(),
        "per_group_auc": per_group,
        "pair_counts": pair_counts,
    }


def group_bootstrap_curve(
    per_group_auc: dict[str, Sequence[float]],
    *,
    n_resamples: int = 4_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Bootstrap the macro AUC curve over prompt groups."""

    values = np.asarray(list(per_group_auc.values()), dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("per_group_auc must contain at least one curve")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    boot = values[draws].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha], axis=0)
    return {
        "unit": "prompt_group",
        "n_groups": int(len(values)),
        "n_resamples": int(n_resamples),
        "ci": ci,
        "low": low.tolist(),
        "high": high.tolist(),
    }


def summarize_formation(
    progress: Sequence[float],
    macro_auc: Sequence[float],
    bootstrap: dict,
    *,
    recovery_fraction: float = 0.9,
) -> dict:
    """Summarize retrospective formation without inventing an event onset."""

    progress_array = np.asarray(progress, dtype=float)
    auc = np.asarray(macro_auc, dtype=float)
    low = np.asarray(bootstrap["low"], dtype=float)
    if not (len(progress_array) == len(auc) == len(low)):
        raise ValueError("progress, AUC, and bootstrap dimensions differ")

    terminal_effect = max(0.0, float(auc[-1] - 0.5))
    target = 0.5 + recovery_fraction * terminal_effect
    worst_later = np.minimum.accumulate(auc[::-1])[::-1]
    recovered = np.flatnonzero(worst_later >= target)
    above_chance = np.flatnonzero(low > 0.5)
    return {
        "terminal_macro_auc": float(auc[-1]),
        "terminal_effect_above_chance": terminal_effect,
        "recovery_fraction": recovery_fraction,
        "recovery_target_auc": target,
        "earliest_progress_with_sustained_recovery": (
            float(progress_array[recovered[0]]) if recovered.size else None
        ),
        "earliest_progress_with_bootstrap_low_above_chance": (
            float(progress_array[above_chance[0]]) if above_chance.size else None
        ),
        "interpretation": (
            "retrospective endpoint-direction decodability; not a discrete "
            "EM onset and not evidence of causal mediation"
        ),
    }


def normalize_activation_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize residual vectors for a cosine-profile sensitivity check."""

    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= eps) or not np.isfinite(norms).all():
        raise ValueError("cannot normalize zero or non-finite activation rows")
    return values / norms


def estimate_state_profile(
    activations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    progress: Iterable[float],
    *,
    n_bootstrap: int = 4_000,
    seed: int = 42,
    fit_mode: str = "terminal",
) -> dict:
    """Run cross-fitting, within-prompt evaluation, and group bootstrap."""

    progress_list = [float(value) for value in progress]
    if len(progress_list) != np.asarray(activations).shape[1]:
        raise ValueError("progress grid does not match activation progress dimension")
    if fit_mode == "terminal":
        scores, folds = crossfit_terminal_direction_scores(
            activations,
            labels,
            groups,
        )
    elif fit_mode == "positionwise":
        scores, folds = crossfit_positionwise_direction_scores(
            activations,
            labels,
            groups,
        )
    else:
        raise ValueError(f"unknown fit_mode {fit_mode!r}")
    auc = within_group_auc_curve(labels, groups, scores)
    bootstrap = group_bootstrap_curve(
        auc["per_group_auc"],
        n_resamples=n_bootstrap,
        seed=seed,
    )
    return {
        "fit_mode": fit_mode,
        "progress": progress_list,
        "folds": folds,
        "auc": auc,
        "bootstrap": bootstrap,
        "summary": summarize_formation(
            progress_list,
            auc["macro_auc"],
            bootstrap,
        ),
    }
