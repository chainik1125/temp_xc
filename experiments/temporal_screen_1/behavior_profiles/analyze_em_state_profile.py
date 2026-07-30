"""Summarize and plot the SAE-free Medical EM temporal state profile."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "em_state_profile_paper7b.json"
DEFAULT_JSON = HERE / "em_state_profile_analysis.json"
DEFAULT_FIGURE = HERE / "em_state_profile_analysis.png"

REPRESENTATIONS = {
    "instantaneous_residual": "Instantaneous residual",
    "prefix_mean_residual": "Prefix-mean residual",
}
PROJECTIONS = {
    "raw_projection": "Raw projection",
    "cosine_projection": "Cosine projection",
}
FIT_FAMILIES = {
    "profiles": "Terminal-transported direction",
    "positionwise_profiles": "Positionwise-refit direction",
}


def canonicalize_progress_zero(profile: dict) -> tuple[dict, dict]:
    """Enforce the exact within-prompt chance baseline at progress zero.

    Every rollout for a prompt has exactly the same token history before the
    first sampled response token. Its within-prompt AUC is therefore exactly
    0.5. The canonical rerun enforces this in the activation array; this
    analysis keeps the same guard and records whether any correction was
    needed.
    """

    result = copy.deepcopy(profile)
    progress = np.asarray(result["progress"], dtype=float)
    if not len(progress) or not np.isclose(progress[0], 0.0):
        raise ValueError("the profile must start at progress zero")

    audit = {
        "raw_macro_auc": float(result["auc"]["macro_auc"][0]),
        "raw_pair_weighted_auc": float(result["auc"]["pair_weighted_auc"][0]),
        "raw_bootstrap_interval": [
            float(result["bootstrap"]["low"][0]),
            float(result["bootstrap"]["high"][0]),
        ],
        "canonical_value": 0.5,
        "correction_needed": not np.isclose(
            float(result["auc"]["macro_auc"][0]),
            0.5,
        ),
        "reason": (
            "within a prompt, all rollouts have identical prompt-only token "
            "history; deviations are floating-point tie artifacts"
        ),
    }
    result["auc"]["macro_auc"][0] = 0.5
    result["auc"]["pair_weighted_auc"][0] = 0.5
    result["bootstrap"]["low"][0] = 0.5
    result["bootstrap"]["high"][0] = 0.5
    for curve in result["auc"]["per_group_auc"].values():
        curve[0] = 0.5
    return result, audit


def _variant_summary(profile: dict, audit: dict) -> dict:
    progress = np.asarray(profile["progress"], dtype=float)
    macro = np.asarray(profile["auc"]["macro_auc"], dtype=float)
    pair = np.asarray(profile["auc"]["pair_weighted_auc"], dtype=float)
    low = np.asarray(profile["bootstrap"]["low"], dtype=float)
    high = np.asarray(profile["bootstrap"]["high"], dtype=float)
    terminal_groups = {
        group: float(values[-1])
        for group, values in profile["auc"]["per_group_auc"].items()
    }
    peak_index = int(np.argmax(macro))
    above = progress[low > 0.5]
    terminal = float(macro[-1])
    terminal_low = float(low[-1])
    terminal_high = float(high[-1])
    return {
        "terminal_macro_auc": terminal,
        "terminal_pair_weighted_auc": float(pair[-1]),
        "terminal_prompt_bootstrap_interval": [terminal_low, terminal_high],
        "terminal_interval_contains_chance": bool(
            terminal_low <= 0.5 <= terminal_high
        ),
        "terminal_effect_above_chance": terminal - 0.5,
        "peak_macro_auc": float(macro[peak_index]),
        "peak_progress": float(progress[peak_index]),
        "peak_prompt_bootstrap_interval": [
            float(low[peak_index]),
            float(high[peak_index]),
        ],
        "progress_points_with_pointwise_low_above_chance": above.tolist(),
        "terminal_per_prompt_auc": terminal_groups,
        "terminal_prompts_above_chance": sum(
            value > 0.5 for value in terminal_groups.values()
        ),
        "terminal_prompts_below_chance": sum(
            value < 0.5 for value in terminal_groups.values()
        ),
        "terminal_prompt_range": [
            min(terminal_groups.values()),
            max(terminal_groups.values()),
        ],
        "n_eligible_prompt_groups": int(profile["auc"]["n_eligible_groups"]),
        "n_within_prompt_pairs": int(profile["auc"]["n_within_group_pairs"]),
        "progress_zero_audit": audit,
    }


def analyze(payload: dict) -> dict:
    """Build an uncertainty-forward summary of both direction-fit families."""

    if payload.get("status") != "complete":
        raise ValueError("EM profile is not complete")
    families = {}
    plot_profiles = {}
    for family in FIT_FAMILIES:
        if family not in payload:
            raise ValueError(f"result is missing {family}")
        families[family] = {}
        plot_profiles[family] = {}
        for representation in REPRESENTATIONS:
            families[family][representation] = {}
            plot_profiles[family][representation] = {}
            for projection in PROJECTIONS:
                corrected, audit = canonicalize_progress_zero(
                    payload[family][representation][projection]
                )
                families[family][representation][projection] = _variant_summary(
                    corrected,
                    audit,
                )
                plot_profiles[family][representation][projection] = {
                    "progress": corrected["progress"],
                    "macro_auc": corrected["auc"]["macro_auc"],
                    "pair_weighted_auc": corrected["auc"]["pair_weighted_auc"],
                    "bootstrap_low": corrected["bootstrap"]["low"],
                    "bootstrap_high": corrected["bootstrap"]["high"],
                }

    terminal_values = [
        variant["terminal_macro_auc"]
        for family in families.values()
        for representations in family.values()
        for variant in representations.values()
    ]
    terminal_supported = [
        not variant["terminal_interval_contains_chance"]
        for family in families.values()
        for representations in family.values()
        for variant in representations.values()
    ]
    paired_support = {}
    for family in FIT_FAMILIES:
        paired_support[family] = {}
        for representation in REPRESENTATIONS:
            raw = families[family][representation]["raw_projection"][
                "progress_points_with_pointwise_low_above_chance"
            ]
            cosine = families[family][representation]["cosine_projection"][
                "progress_points_with_pointwise_low_above_chance"
            ]
            paired_support[family][representation] = sorted(set(raw) & set(cosine))

    positionwise = families["positionwise_profiles"]
    return {
        "source": {
            "task": payload["task"],
            "model": payload["model"],
            "hook": payload["teacher_forcing"]["hook"],
            "frozen_input_sha256": payload["frozen_input_sha256"],
            "elapsed_seconds": payload["elapsed_seconds"],
        },
        "sample": {
            **payload["endpoint_labels"],
            **payload["length_audit"],
            "n_eligible_prompt_groups": next(
                iter(families["profiles"]["instantaneous_residual"].values())
            )["n_eligible_prompt_groups"],
            "n_within_prompt_pairs": next(
                iter(families["profiles"]["instantaneous_residual"].values())
            )["n_within_prompt_pairs"],
        },
        "estimand": payload["definition"],
        "fit_families": {
            "profiles": (
                "one direction fit at terminal response progress and "
                "transported unchanged across the whole trajectory"
            ),
            "positionwise_profiles": (
                "a separate leave-one-prompt-out direction fit at each "
                "response-progress point"
            ),
        },
        "families": families,
        "cross_variant": {
            "terminal_macro_auc_range": [
                min(terminal_values),
                max(terminal_values),
            ],
            "n_terminal_intervals_excluding_chance": sum(terminal_supported),
            "n_variants": len(terminal_supported),
            "headline": (
                "The fixed terminal direction transports weakly: all terminal "
                "macro AUCs are 0.604 and their prompt-bootstrap intervals "
                "contain chance. Positionwise refitting finds isolated "
                "mid-response separation, not a persistent trajectory."
            ),
            "trajectory": (
                "Positionwise prefix-mean raw and cosine profiles both peak "
                "at AUC 0.806 at 30% progress, with pointwise bootstrap lows "
                "0.583 and 0.611. Instantaneous profiles peak near 0.72 at "
                "60%. All curves then fall to terminal AUC 0.604."
            ),
            "paired_raw_cosine_pointwise_support": paired_support,
            "positionwise_prefix_mean_peak": {
                projection: {
                    "auc": positionwise["prefix_mean_residual"][projection][
                        "peak_macro_auc"
                    ],
                    "progress": positionwise["prefix_mean_residual"][projection][
                        "peak_progress"
                    ],
                    "pointwise_interval": positionwise["prefix_mean_residual"][
                        projection
                    ]["peak_prompt_bootstrap_interval"],
                }
                for projection in PROJECTIONS
            },
            "interpretation": (
                "The positionwise result is compatible with a rotating or "
                "transient shared signal, but uses a different direction at "
                "each point. With six prompt groups and many pointwise "
                "comparisons, it is exploratory and is not an onset estimate."
            ),
        },
        "progress_zero_policy": (
            "The canonical rerun makes prompt-only within-prompt AUC exactly "
            "0.5 in the source result; the analysis retains that invariant."
        ),
        "plot_profiles": plot_profiles,
    }


def plot_analysis(analysis: dict, output: Path) -> None:
    """Render the four curves with prompt-bootstrap uncertainty."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "raw_projection": "#0072B2",
        "cosine_projection": "#D55E00",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), sharex=True, sharey=True)
    for row, (family, family_label) in enumerate(FIT_FAMILIES.items()):
        for column, (representation, representation_label) in enumerate(
            REPRESENTATIONS.items()
        ):
            axis = axes[row, column]
            for projection, label in PROJECTIONS.items():
                profile = analysis["plot_profiles"][family][representation][projection]
                progress = np.asarray(profile["progress"], dtype=float)
                macro = np.asarray(profile["macro_auc"], dtype=float)
                low = np.asarray(profile["bootstrap_low"], dtype=float)
                high = np.asarray(profile["bootstrap_high"], dtype=float)
                axis.fill_between(
                    progress,
                    low,
                    high,
                    color=colors[projection],
                    alpha=0.13,
                    linewidth=0,
                )
                axis.plot(
                    progress,
                    macro,
                    color=colors[projection],
                    marker="o",
                    markersize=3.8,
                    linewidth=1.7,
                    label=label,
                )
            axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
            axis.set_title(
                f"{family_label}\n{representation_label}",
                fontsize=10.5,
            )
            axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(0.0, 1.02)
            axis.grid(axis="y", alpha=0.2)
            if row == 1:
                axis.set_xlabel("Fraction of response observed")
            if column == 0:
                axis.set_ylabel("Within-prompt AUC")
            if row == 0:
                note = "terminal AUC = 0.604; CI includes 0.5"
            elif representation == "prefix_mean_residual":
                note = "peak AUC = 0.806 at 30%; not sustained"
            else:
                note = "peak AUC ≈ 0.72 at 60%; not sustained"
            axis.text(
                0.02,
                0.035,
                note,
                transform=axis.transAxes,
                fontsize=8.2,
                va="bottom",
            )
    axes[0, 1].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "Medical EM: fixed-direction transport versus positionwise separation",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.005,
        (
            "Six eligible prompt groups; 95% pointwise prompt-bootstrap bands "
            "are uncorrected for progress-point multiplicity. No curve is an onset."
        ),
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    analysis = analyze(payload)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    plot_analysis(analysis, args.output_figure)
    print(
        json.dumps(
            {
                "headline": analysis["cross_variant"]["headline"],
                "trajectory": analysis["cross_variant"]["trajectory"],
                "output_json": str(args.output_json),
                "output_figure": str(args.output_figure),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
