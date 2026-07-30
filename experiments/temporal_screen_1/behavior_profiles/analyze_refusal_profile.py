"""Reproduce the prompt-level analysis of the refusal prefix profile.

The analysis distinguishes three quantities:

1. cohort separation by the published refusal direction;
2. the first stable, neutral-calibrated direction expression per prompt; and
3. the first generated lexical refusal per prompt.

Run from the repository root:

    python experiments/temporal_screen_1/behavior_profiles/analyze_refusal_profile.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "refusal_prefix_profile.json"
DEFAULT_JSON = HERE / "refusal_analysis.json"
DEFAULT_FIGURE = HERE / "refusal_analysis.png"

DIRECTION_DECISION = "direction_projection_decision_position"
DIRECTION_SOURCE = "direction_projection_source_position"
REFUSAL_LOG_ODDS = "refusal_log_odds"


def empirical_auc(positive: list[float], negative: list[float]) -> float:
    """Return rank AUC with half credit for exact ties."""

    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if not len(pos) or not len(neg):
        return float("nan")
    comparisons = pos[:, None] - neg[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 20_000,
) -> tuple[float, float]:
    """Return a paired-bootstrap percentile interval for a mean effect."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(array),
        size=(n_bootstrap, len(array)),
    )
    boot_means = array[samples].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975])
    return float(low), float(high)


def paired_condition_effect(
    baseline: dict[str, dict[str, Any]],
    condition: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Summarize a condition against baseline on the same prompt IDs."""

    prompt_ids = sorted(set(baseline) & set(condition))
    if set(prompt_ids) != set(baseline) or set(prompt_ids) != set(condition):
        raise ValueError("baseline and condition prompt IDs must match")
    baseline_refusal = np.asarray(
        [int(baseline[prompt_id]["generated_refusal"]) for prompt_id in prompt_ids]
    )
    condition_refusal = np.asarray(
        [
            int(condition[prompt_id]["generated_refusal"])
            for prompt_id in prompt_ids
        ]
    )
    refusal_reduction = baseline_refusal - condition_refusal
    log_odds_reduction = np.asarray(
        [
            baseline[prompt_id][REFUSAL_LOG_ODDS]
            - condition[prompt_id][REFUSAL_LOG_ODDS]
            for prompt_id in prompt_ids
        ],
        dtype=float,
    )
    refusal_ci = bootstrap_mean_interval(refusal_reduction, seed=seed)
    log_odds_ci = bootstrap_mean_interval(
        log_odds_reduction,
        seed=seed + 10_000,
    )
    transitions = Counter(
        zip(
            baseline_refusal.tolist(),
            condition_refusal.tolist(),
            strict=True,
        )
    )
    return {
        "n_paired_prompts": len(prompt_ids),
        "baseline_refusal_rate": float(baseline_refusal.mean()),
        "condition_refusal_rate": float(condition_refusal.mean()),
        "paired_refusal_rate_reduction": float(refusal_reduction.mean()),
        "paired_refusal_rate_reduction_ci95": list(refusal_ci),
        "baseline_refuses_condition_does_not": int(
            transitions[(1, 0)]
        ),
        "baseline_does_not_condition_refuses": int(
            transitions[(0, 1)]
        ),
        "paired_mean_refusal_log_odds_reduction": float(
            log_odds_reduction.mean()
        ),
        "paired_mean_refusal_log_odds_reduction_ci95": list(log_odds_ci),
        "changed_response_count": int(
            sum(
                baseline[prompt_id]["response_sha256"]
                != condition[prompt_id]["response_sha256"]
                for prompt_id in prompt_ids
            )
        ),
    }


def prompt_index(row: dict[str, Any]) -> int:
    return int(row["prompt_id"].split("-")[1])


def rows_for(
    rows: list[dict[str, Any]],
    cohort: str,
    condition: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["cohort"] == cohort and row["condition"] == condition
    ]


def by_prompt(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt_id"]].append(row)
    return {
        prompt_id: sorted(group, key=lambda row: row["reveal_fraction"])
        for prompt_id, group in grouped.items()
    }


def first_stable_index(flags: list[bool]) -> int | None:
    """Return the first index from which every later flag remains true."""

    for index in range(len(flags)):
        if all(flags[index:]):
            return index
    return None


def serialize_distribution(values: list[float | None]) -> dict[str, int]:
    counts = Counter("never" if value is None else f"{value:g}" for value in values)
    return dict(sorted(counts.items()))


def curve(
    rows: list[dict[str, Any]],
    fractions: list[float],
    key: str,
) -> list[float]:
    return [
        float(
            np.mean(
                [
                    float(row[key])
                    for row in rows
                    if row["reveal_fraction"] == fraction
                ]
            )
        )
        for fraction in fractions
    ]


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["rows"]
    harmful = rows_for(rows, "harmful", "baseline")
    harmless = rows_for(rows, "harmless", "baseline")
    ablated = rows_for(rows, "harmful", "direction_ablation")
    current_ablated = rows_for(
        rows, "harmful", "current_token_ablation"
    )
    current_added = rows_for(
        rows, "harmless", "current_token_addition"
    )
    added = rows_for(rows, "harmless", "direction_addition")
    fractions = sorted({float(row["reveal_fraction"]) for row in harmful})

    expected = len(fractions)
    harmful_prompts = by_prompt(harmful)
    harmless_prompts = by_prompt(harmless)
    assert all(len(group) == expected for group in harmful_prompts.values())
    assert all(len(group) == expected for group in harmless_prompts.values())

    auc_curves: dict[str, list[float]] = {}
    for metric in (DIRECTION_DECISION, DIRECTION_SOURCE, REFUSAL_LOG_ODDS):
        auc_curves[metric] = [
            empirical_auc(
                [
                    float(row[metric])
                    for row in harmful
                    if row["reveal_fraction"] == fraction
                ],
                [
                    float(row[metric])
                    for row in harmless
                    if row["reveal_fraction"] == fraction
                ],
            )
            for fraction in fractions
        ]

    length_by_index_h = {
        prompt_index(rows[0]): int(rows[0]["total_instruction_tokens"])
        for rows in harmful_prompts.values()
    }
    length_by_index_n = {
        prompt_index(rows[0]): int(rows[0]["total_instruction_tokens"])
        for rows in harmless_prompts.values()
    }
    length_differences = np.asarray(
        [
            length_by_index_n[index] - length_by_index_h[index]
            for index in sorted(length_by_index_h)
        ],
        dtype=int,
    )
    close_indices = {
        index
        for index in length_by_index_h
        if abs(length_by_index_n[index] - length_by_index_h[index]) <= 5
    }
    close_match_auc: dict[str, list[float]] = {}
    for metric in (DIRECTION_DECISION, DIRECTION_SOURCE, REFUSAL_LOG_ODDS):
        close_match_auc[metric] = [
            empirical_auc(
                [
                    float(row[metric])
                    for row in harmful
                    if row["reveal_fraction"] == fraction
                    and prompt_index(row) in close_indices
                ],
                [
                    float(row[metric])
                    for row in harmless
                    if row["reveal_fraction"] == fraction
                    and prompt_index(row) in close_indices
                ],
            )
            for fraction in fractions
        ]

    neutral_thresholds = {
        fraction: float(
            np.quantile(
                [
                    row[DIRECTION_DECISION]
                    for row in harmless
                    if row["reveal_fraction"] == fraction
                ],
                0.95,
            )
        )
        for fraction in fractions
    }

    prompt_rows = []
    direction_onsets = []
    refusal_onsets = []
    fraction_lags = []
    token_lags = []
    refusal_flickers = 0
    for prompt_id, group in harmful_prompts.items():
        refusal_flags = [bool(row["generated_refusal"]) for row in group]
        if any(
            refusal_flags[index] and not refusal_flags[index + 1]
            for index in range(len(refusal_flags) - 1)
        ):
            refusal_flickers += 1
        refusal_index = next(
            (index for index, flag in enumerate(refusal_flags) if flag),
            None,
        )
        direction_flags = [
            row["reveal_fraction"] > 0
            and row[DIRECTION_DECISION]
            > neutral_thresholds[float(row["reveal_fraction"])]
            for row in group
        ]
        direction_index = first_stable_index(direction_flags)
        direction_fraction = (
            float(group[direction_index]["reveal_fraction"])
            if direction_index is not None
            else None
        )
        direction_tokens = (
            int(group[direction_index]["prefix_tokens"])
            if direction_index is not None
            else None
        )
        refusal_fraction = (
            float(group[refusal_index]["reveal_fraction"])
            if refusal_index is not None
            else None
        )
        refusal_tokens = (
            int(group[refusal_index]["prefix_tokens"])
            if refusal_index is not None
            else None
        )
        if direction_fraction is not None and refusal_fraction is not None:
            fraction_lags.append(refusal_fraction - direction_fraction)
            token_lags.append(refusal_tokens - direction_tokens)
        direction_onsets.append(direction_fraction)
        refusal_onsets.append(refusal_fraction)
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "total_instruction_tokens": int(
                    group[0]["total_instruction_tokens"]
                ),
                "stable_direction_onset_fraction": direction_fraction,
                "stable_direction_onset_tokens": direction_tokens,
                "lexical_refusal_onset_fraction": refusal_fraction,
                "lexical_refusal_onset_tokens": refusal_tokens,
            }
        )

    fraction_lags_array = np.asarray(fraction_lags)
    token_lags_array = np.asarray(token_lags)
    full_harmful = [
        row for row in harmful if float(row["reveal_fraction"]) == 1.0
    ]
    full_harmless = [
        row for row in harmless if float(row["reveal_fraction"]) == 1.0
    ]
    full_ablated = [
        row for row in ablated if float(row["reveal_fraction"]) == 1.0
    ]
    full_current_ablated = [
        row
        for row in current_ablated
        if float(row["reveal_fraction"]) == 1.0
    ]
    full_current_added = [
        row
        for row in current_added
        if float(row["reveal_fraction"]) == 1.0
    ]
    empty_harmful = [
        row for row in harmful if float(row["reveal_fraction"]) == 0.0
    ]
    empty_harmless = [
        row for row in harmless if float(row["reveal_fraction"]) == 0.0
    ]
    harmful_refusal_curve = curve(
        harmful, fractions, "generated_refusal"
    )
    harmless_refusal_curve = curve(
        harmless, fractions, "generated_refusal"
    )
    ablated_refusal_curve = curve(
        ablated, fractions, "generated_refusal"
    )
    baseline_full_by_prompt = {
        row["prompt_id"]: row for row in full_harmful
    }
    lag_effects = []
    for index, band in enumerate(
        payload["aggregate"]["prompt_lag_localization"]["bands"]
    ):
        condition_rows = rows_for(
            rows,
            "harmful",
            band["condition"],
        )
        condition_by_prompt = {
            row["prompt_id"]: row
            for row in condition_rows
            if float(row["reveal_fraction"]) == 1.0
        }
        lag_effects.append(
            {
                "condition": band["condition"],
                "low_lag_inclusive": band["low_lag_inclusive"],
                "high_lag_inclusive": band["high_lag_inclusive"],
            }
            | paired_condition_effect(
                baseline_full_by_prompt,
                condition_by_prompt,
                seed=20260729 + index,
            )
        )
    finite_lag_effects = [
        effect
        for effect in lag_effects
        if effect["high_lag_inclusive"] is not None
    ]
    all_prior_effect = next(
        effect
        for effect in lag_effects
        if effect["high_lag_inclusive"] is None
    )

    def first_auc_at_least(values: list[float], cutoff: float) -> float | None:
        return next(
            (
                fraction
                for fraction, value in zip(fractions, values, strict=True)
                if fraction > 0 and value >= cutoff
            ),
            None,
        )

    return {
        "source": {
            "model": payload["model"],
            "method": payload["method"],
            "n_harmful_prompts": len(harmful_prompts),
            "n_harmless_prompts": len(harmless_prompts),
            "reveal_fractions": fractions,
        },
        "length_matching": {
            "harmful_token_length_min_median_max": [
                int(min(length_by_index_h.values())),
                float(np.median(list(length_by_index_h.values()))),
                int(max(length_by_index_h.values())),
            ],
            "mean_absolute_difference": float(
                np.mean(np.abs(length_differences))
            ),
            "max_absolute_difference": int(
                np.max(np.abs(length_differences))
            ),
            "pairs_within_5_tokens": len(close_indices),
            "n_pairs": len(length_differences),
        },
        "cohort_curves": {
            "harmful_refusal_rate": harmful_refusal_curve,
            "harmless_refusal_rate": harmless_refusal_curve,
            "ablated_harmful_refusal_rate": ablated_refusal_curve,
            "harmful_direction_decision_mean": curve(
                harmful, fractions, DIRECTION_DECISION
            ),
            "harmless_direction_decision_mean": curve(
                harmless, fractions, DIRECTION_DECISION
            ),
            "harmful_refusal_log_odds_mean": curve(
                harmful, fractions, REFUSAL_LOG_ODDS
            ),
            "harmless_refusal_log_odds_mean": curve(
                harmless, fractions, REFUSAL_LOG_ODDS
            ),
        },
        "empty_prefix_calibration": {
            "note": (
                "Both cohorts render the same empty user message. Nominal "
                "class separation is therefore an implementation artifact."
            ),
            "harmful_unique_response_hashes": len(
                {row["response_sha256"] for row in empty_harmful}
            ),
            "harmless_unique_response_hashes": len(
                {row["response_sha256"] for row in empty_harmless}
            ),
            "shared_response_hashes": len(
                {row["response_sha256"] for row in empty_harmful}
                & {row["response_sha256"] for row in empty_harmless}
            ),
            "nominal_auc": {
                metric: values[0] for metric, values in auc_curves.items()
            },
        },
        "separability": {
            "auc_by_fraction": auc_curves,
            "first_fraction_auc_at_least_0.8": {
                metric: first_auc_at_least(values, 0.8)
                for metric, values in auc_curves.items()
            },
            "within_5_token_match_sensitivity_n_pairs": len(close_indices),
            "within_5_token_match_auc_by_fraction": close_match_auc,
        },
        "prompt_onsets": {
            "direction_definition": (
                "first non-empty prefix from which decision-position "
                "projection stays above the same-fraction 95th percentile "
                "of length-matched harmless prompts"
            ),
            "neutral_95_percentile_threshold_by_fraction": {
                f"{fraction:g}": value
                for fraction, value in neutral_thresholds.items()
            },
            "stable_direction_onset_distribution": serialize_distribution(
                direction_onsets
            ),
            "lexical_refusal_onset_distribution": serialize_distribution(
                refusal_onsets
            ),
            "n_refusal_flickers": refusal_flickers,
            "n_with_both_onsets": len(fraction_lags),
            "direction_at_or_before_refusal_count": int(
                np.sum(fraction_lags_array >= 0)
            ),
            "direction_strictly_before_refusal_count": int(
                np.sum(fraction_lags_array > 0)
            ),
            "median_fraction_lag_refusal_minus_direction": float(
                np.median(fraction_lags_array)
            ),
            "mean_fraction_lag_refusal_minus_direction": float(
                np.mean(fraction_lags_array)
            ),
            "median_token_lag_refusal_minus_direction": float(
                np.median(token_lags_array)
            ),
            "mean_token_lag_refusal_minus_direction": float(
                np.mean(token_lags_array)
            ),
            "prompt_rows": prompt_rows,
        },
        "prompt_lag_localization": {
            "coordinate": (
                "absolute token lag from the final rendered assistant "
                "decision-state token; lag zero is not ablated"
            ),
            "scope": (
                "all-layer directional ablation at selected prompt-pass "
                "positions; cached generation-token states are untouched"
            ),
            "coordinate_caveat": (
                "Rendered lags include chat-template and instruction tokens; "
                "high-lag bands can also include left padding for shorter "
                "prompts."
            ),
            "paired_effects": lag_effects,
            "largest_finite_band_refusal_rate_reduction": float(
                max(
                    effect["paired_refusal_rate_reduction"]
                    for effect in finite_lag_effects
                )
            ),
            "sum_finite_band_marginal_refusal_rate_reductions": float(
                sum(
                    effect["paired_refusal_rate_reduction"]
                    for effect in finite_lag_effects
                )
            ),
            "all_prior_refusal_rate_reduction": float(
                all_prior_effect["paired_refusal_rate_reduction"]
            ),
            "interpretation": (
                "No tested finite band is individually sufficient to remove "
                "refusal, whereas ablating all prior prompt positions removes "
                "every baseline refusal. This is compatible with redundant "
                "distributed support or non-additive interactions across "
                "positions, not a uniquely localized lag."
            ),
        },
        "causal_checks": {
            "intervention_scope": {
                "current_token": (
                    "last token in prompt prefill and the one-token input "
                    "on each cached generation step"
                ),
                "all_positions": (
                    "every prompt position in prefill and the one-token "
                    "input on each cached generation step"
                ),
            },
            "full_prompt_cells": {
                "baseline_harmful": {
                    "refusal_count": int(
                        sum(
                            row["generated_refusal"]
                            for row in full_harmful
                        )
                    ),
                    "n": len(full_harmful),
                },
                "current_token_ablation_harmful": {
                    "refusal_count": int(
                        sum(
                            row["generated_refusal"]
                            for row in full_current_ablated
                        )
                    ),
                    "n": len(full_current_ablated),
                    "direction_decision_mean": float(
                        np.mean(
                            [
                                row[DIRECTION_DECISION]
                                for row in full_current_ablated
                            ]
                        )
                    ),
                    "direction_source_mean": float(
                        np.mean(
                            [
                                row[DIRECTION_SOURCE]
                                for row in full_current_ablated
                            ]
                        )
                    ),
                    "refusal_log_odds_mean": float(
                        np.mean(
                            [
                                row[REFUSAL_LOG_ODDS]
                                for row in full_current_ablated
                            ]
                        )
                    ),
                },
                "all_position_ablation_harmful": {
                    "refusal_count": int(
                        sum(
                            row["generated_refusal"]
                            for row in full_ablated
                        )
                    ),
                    "n": len(full_ablated),
                },
                "baseline_harmless": {
                    "refusal_count": int(
                        sum(
                            row["generated_refusal"]
                            for row in full_harmless
                        )
                    ),
                    "n": len(full_harmless),
                },
                "current_token_addition_harmless": {
                    "refusal_count": int(
                        sum(
                            row["generated_refusal"]
                            for row in full_current_added
                        )
                    ),
                    "n": len(full_current_added),
                    "direction_decision_mean": float(
                        np.mean(
                            [
                                row[DIRECTION_DECISION]
                                for row in full_current_added
                            ]
                        )
                    ),
                    "direction_source_mean": float(
                        np.mean(
                            [
                                row[DIRECTION_SOURCE]
                                for row in full_current_added
                            ]
                        )
                    ),
                    "refusal_log_odds_mean": float(
                        np.mean(
                            [
                                row[REFUSAL_LOG_ODDS]
                                for row in full_current_added
                            ]
                        )
                    ),
                },
                "all_position_addition_harmless": {
                    "refusal_count": int(
                        sum(row["generated_refusal"] for row in added)
                    ),
                    "n": len(added),
                },
            },
            "supported_claim": (
                "The direction is a sequence-wide causal effector, but its "
                "presence at only the current autoregressive token is "
                "neither necessary nor sufficient in this intervention."
            ),
            "localization_limit": (
                "Current-token and finite prompt-band interventions leave "
                "other prompt-position states and their cached keys/values "
                "intact. The controls reject a unique bottleneck in any "
                "tested band, but cannot distinguish redundant distributed "
                "support from non-additive interactions across positions."
            ),
        },
    }


def make_figure(
    payload: dict[str, Any],
    analysis: dict[str, Any],
    output: Path,
) -> None:
    fractions = np.asarray(analysis["source"]["reveal_fractions"])
    curves = analysis["cohort_curves"]
    aggregate = payload["aggregate"]["curves"]
    harmful_aggregate = aggregate["baseline:harmful"]["metrics"]
    harmless_aggregate = aggregate["baseline:harmless"]["metrics"]

    fig, axes = plt.subplot_mosaic(
        [
            ["direction", "refusal", "spatial"],
            ["auc", "onset", "lags"],
        ],
        figsize=(15.5, 8.4),
        width_ratios=[1.05, 1.05, 0.9],
    )
    ax = axes["direction"]
    for label, source, color in (
        ("harmful", harmful_aggregate, "#b33a3a"),
        ("length-matched harmless", harmless_aggregate, "#3a70b3"),
    ):
        metric = source[DIRECTION_DECISION]
        mean = np.asarray(metric["mean"])
        low = np.asarray(metric["ci_low"])
        high = np.asarray(metric["ci_high"])
        ax.plot(fractions, mean, marker="o", label=label, color=color)
        ax.fill_between(fractions, low, high, color=color, alpha=0.15)
    ax.set_title("Published direction at assistant decision state")
    ax.set_ylabel("projection onto direction")
    ax.legend(frameon=False)

    ax = axes["refusal"]
    ax.plot(
        fractions,
        curves["harmful_refusal_rate"],
        marker="o",
        label="harmful",
        color="#b33a3a",
    )
    ax.plot(
        fractions,
        curves["harmless_refusal_rate"],
        marker="o",
        label="harmless",
        color="#3a70b3",
    )
    ax.plot(
        fractions,
        curves["ablated_harmful_refusal_rate"],
        marker="o",
        label="harmful + direction ablation",
        color="#777777",
    )
    ax.set_title("Generated lexical refusal")
    ax.set_ylabel("refusal rate")
    ax.set_ylim(-0.04, 1.04)
    ax.legend(frameon=False)

    ax = axes["auc"]
    auc = analysis["separability"]["auc_by_fraction"]
    for metric, label, color in (
        (DIRECTION_DECISION, "direction: decision", "#b33a3a"),
        (DIRECTION_SOURCE, "direction: source -5", "#d68b32"),
        (REFUSAL_LOG_ODDS, "refusal-token log odds", "#4b8a57"),
    ):
        ax.plot(fractions, auc[metric], marker="o", label=label, color=color)
    ax.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
    ax.axhline(0.8, color="#bbbbbb", linestyle=":", linewidth=1)
    ax.set_title("Harmful versus harmless separation")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.35, 1.03)
    ax.legend(frameon=False, fontsize=8)

    ax = axes["onset"]
    onset_rows = [
        row
        for row in analysis["prompt_onsets"]["prompt_rows"]
        if row["lexical_refusal_onset_tokens"] is not None
        and row["stable_direction_onset_tokens"] is not None
    ]
    direction_tokens = np.asarray(
        [row["stable_direction_onset_tokens"] for row in onset_rows]
    )
    refusal_tokens = np.asarray(
        [row["lexical_refusal_onset_tokens"] for row in onset_rows]
    )
    limit = max(direction_tokens.max(), refusal_tokens.max()) + 2
    ax.scatter(
        direction_tokens,
        refusal_tokens,
        alpha=0.75,
        color="#744c9e",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.plot([0, limit], [0, limit], color="#888888", linestyle="--", linewidth=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Prompt-level onset (31 refusing prompts)")
    ax.set_xlabel("stable direction onset (prefix tokens)")
    ax.set_ylabel("lexical refusal onset (prefix tokens)")

    ax = axes["spatial"]
    cells = analysis["causal_checks"]["full_prompt_cells"]
    labels = [
        "harmful: baseline",
        "harmful: current ablate",
        "harmful: all-position ablate",
        "harmless: baseline",
        "harmless: current add",
        "harmless: all-position add",
    ]
    keys = [
        "baseline_harmful",
        "current_token_ablation_harmful",
        "all_position_ablation_harmful",
        "baseline_harmless",
        "current_token_addition_harmless",
        "all_position_addition_harmless",
    ]
    rates = [
        cells[key]["refusal_count"] / cells[key]["n"]
        for key in keys
    ]
    y = np.asarray([5, 4, 3, 2, 1, 0])
    colors = [
        "#b33a3a",
        "#d68b32",
        "#777777",
        "#3a70b3",
        "#6d94c6",
        "#744c9e",
    ]
    ax.barh(y, rates, color=colors, alpha=0.88)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("full-prompt refusal rate")
    ax.set_title("One vector ≠ one-token bottleneck")
    ax.axhline(2.5, color="#bbbbbb", linewidth=1)
    ax.tick_params(axis="y", labelsize=8)
    for y_value, rate, key in zip(y, rates, keys, strict=True):
        cell = cells[key]
        ax.text(
            min(rate + 0.025, 1.02),
            y_value,
            f"{cell['refusal_count']}/{cell['n']}",
            va="center",
            fontsize=9,
        )
    ax = axes["lags"]
    effects = analysis["prompt_lag_localization"]["paired_effects"]
    lag_labels = [
        (
            "all prior"
            if effect["high_lag_inclusive"] is None
            else (
                f"{effect['low_lag_inclusive']}"
                f"–{effect['high_lag_inclusive']}"
            )
        )
        + (
            "  ("
            f"{round(effect['condition_refusal_rate'] * effect['n_paired_prompts'])}"
            f"/{effect['n_paired_prompts']})"
        )
        for effect in effects
    ]
    reductions = np.asarray(
        [effect["paired_refusal_rate_reduction"] for effect in effects]
    )
    interval_low = np.asarray(
        [
            effect["paired_refusal_rate_reduction_ci95"][0]
            for effect in effects
        ]
    )
    interval_high = np.asarray(
        [
            effect["paired_refusal_rate_reduction_ci95"][1]
            for effect in effects
        ]
    )
    lag_y = np.arange(len(effects))[::-1]
    lag_colors = ["#d68b32"] * (len(effects) - 1) + ["#744c9e"]
    ax.barh(
        lag_y,
        reductions,
        color=lag_colors,
        alpha=0.88,
        xerr=np.vstack(
            [reductions - interval_low, interval_high - reductions]
        ),
        error_kw={"capsize": 3, "elinewidth": 1},
    )
    ax.set_yticks(lag_y, lag_labels)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("paired refusal-rate reduction")
    ax.set_title("No finite prompt-lag band is sufficient")

    for ax in axes.values():
        ax.set_xlabel(
            ax.get_xlabel() or "fraction of instruction tokens revealed"
        )
        ax.grid(alpha=0.18)
    fig.suptitle(
        "Refusal-direction readout precedes behavior; "
        "causal support spans prior prompt positions"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    analysis = analyze(payload)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    make_figure(payload, analysis, args.output_figure)

    onset = analysis["prompt_onsets"]
    separation = analysis["separability"]["first_fraction_auc_at_least_0.8"]
    print(
        "direction AUC >= 0.8 at",
        separation[DIRECTION_DECISION],
        "; refusal-log-odds AUC >= 0.8 at",
        separation[REFUSAL_LOG_ODDS],
    )
    print(
        "stable direction at/before lexical refusal:",
        f"{onset['direction_at_or_before_refusal_count']}/"
        f"{onset['n_with_both_onsets']};",
        "strictly before:",
        f"{onset['direction_strictly_before_refusal_count']}/"
        f"{onset['n_with_both_onsets']};",
        "median lag:",
        f"{onset['median_token_lag_refusal_minus_direction']:.1f} tokens",
    )
    cells = analysis["causal_checks"]["full_prompt_cells"]
    print(
        "harmful refusal baseline/current-token/all-position ablation:",
        *[
            cells[key]["refusal_count"] / cells[key]["n"]
            for key in (
                "baseline_harmful",
                "current_token_ablation_harmful",
                "all_position_ablation_harmful",
            )
        ],
    )
    print(
        "harmless refusal baseline/current-token/all-position addition:",
        *[
            cells[key]["refusal_count"] / cells[key]["n"]
            for key in (
                "baseline_harmless",
                "current_token_addition_harmless",
                "all_position_addition_harmless",
            )
        ],
    )
    lag_effects = analysis["prompt_lag_localization"]["paired_effects"]
    print(
        "prompt-lag refusal rates:",
        {
            effect["condition"].removeprefix("prompt_lag_ablation_"):
            effect["condition_refusal_rate"]
            for effect in lag_effects
        },
    )
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_figure}")


if __name__ == "__main__":
    main()
