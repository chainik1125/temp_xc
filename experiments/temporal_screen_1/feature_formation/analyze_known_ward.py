"""Make the corrected Ward feature-formation summary and figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_NATURAL = HERE / "results" / "ward_known_feature_formation.json"
DEFAULT_CAUSAL = HERE / "results" / "ward_offset_causal_efficacy.json"
DEFAULT_SUMMARY = HERE / "results" / "ward_known_feature_summary.json"
DEFAULT_FIGURE = HERE / "results" / "ward_known_feature_summary.png"


def _xy(curve: list[dict], key: str = "auc") -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["offset"] for row in curve]),
        np.asarray([row[key] for row in curve]),
    )


def _causal_offset(name: str) -> int | None:
    prefix = "base_derived_off"
    if not name.startswith(prefix) or not name.endswith("_raw"):
        return None
    return int(name[len(prefix) : -len("_raw")])


def _bootstrap_causal(
    causal: dict,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260729,
) -> dict:
    """Prompt-paired bootstrap intervals for each intervention and baseline lift."""

    by_source = {}
    for row in causal["rows"]:
        by_source.setdefault(row["source"], {})[row["prompt_id"]] = float(
            row["keyword_rate"]
        )
    baseline_name = "stage_a_magnitude_zero_baseline"
    baseline = by_source[baseline_name]
    prompt_ids = sorted(baseline)
    rng = np.random.default_rng(seed)
    sample = rng.integers(
        0,
        len(prompt_ids),
        size=(n_bootstrap, len(prompt_ids)),
    )
    baseline_values = np.asarray([baseline[qid] for qid in prompt_ids])
    rows = {}
    for source, source_map in by_source.items():
        if set(source_map) != set(prompt_ids):
            raise ValueError(f"prompt mismatch for causal source {source}")
        values = np.asarray([source_map[qid] for qid in prompt_ids])
        boot_mean = np.mean(values[sample], axis=1)
        boot_delta = np.mean((values - baseline_values)[sample], axis=1)
        rows[source] = {
            "mean_keyword_rate": float(np.mean(values)),
            "mean_keyword_rate_ci95": [
                float(np.quantile(boot_mean, 0.025)),
                float(np.quantile(boot_mean, 0.975)),
            ],
            "paired_lift_over_baseline": float(
                np.mean(values - baseline_values)
            ),
            "paired_lift_ci95": [
                float(np.quantile(boot_delta, 0.025)),
                float(np.quantile(boot_delta, 0.975)),
            ],
        }
    return {
        "method": "10,000 prompt-paired nonparametric bootstrap replicates",
        "seed": seed,
        "sources": rows,
    }


def analyze(natural: dict, causal: dict | None) -> dict:
    known = natural["known_directions"]
    sae = natural["conventional_sae"]
    base = known["summaries"]["base_union"]
    distributed = sae["distributed_projection"]["summary"]
    single_name = f"f{sae['best_positive_feature']}"
    single = sae["summaries"][single_name]
    summary = {
        "n_pairs": natural["n_pairs"],
        "natural": {
            "primary_direction": "base_union",
            "base_union": base,
            "sae_distributed_projection": distributed,
            "sae_best_single_feature": {
                "feature": sae["best_positive_feature"],
                "decoder_cosine": sae["best_positive_cosine"],
                "summary": single,
            },
            "ward_band_mean_auc": {
                "residual_base_union": base["bands"]["ward"]["mean_auc"],
                "sae_distributed": distributed["bands"]["ward"]["mean_auc"],
                "sae_best_single": single["bands"]["ward"]["mean_auc"],
            },
            "post_band_mean_auc": {
                "residual_base_union": base["bands"]["post"]["mean_auc"],
                "sae_distributed": distributed["bands"]["post"]["mean_auc"],
                "sae_best_single": single["bands"]["post"]["mean_auc"],
            },
        },
        "causal": None,
    }
    if causal is not None:
        aggregates = causal["aggregates"]
        bootstrap = _bootstrap_causal(causal)
        band_rows = []
        for source, values in aggregates.items():
            offset = _causal_offset(source)
            if offset is None or offset == 0:
                continue
            band_rows.append(
                {
                    "offset": offset,
                    "source": source,
                    **values,
                }
            )
        band_rows.sort(key=lambda row: row["offset"])
        baseline = aggregates["stage_a_magnitude_zero_baseline"]
        best = max(
            band_rows,
            key=lambda row: row["mean_keyword_rate"],
        )
        matched_auc_by_offset = {
            int(row["offset"]): float(row["auc"])
            for row in known["cross_offset"]["base"]["matched_offset"]
        }
        causal_rates = np.asarray(
            [row["mean_keyword_rate"] for row in band_rows]
        )
        matched_aucs = np.asarray(
            [matched_auc_by_offset[row["offset"]] for row in band_rows]
        )
        summary["causal"] = {
            "magnitude": causal["magnitude"],
            "baseline": baseline,
            "band_offsets": band_rows,
            "best_band_offset": best["offset"],
            "best_band_mean_keyword_rate": best["mean_keyword_rate"],
            "union": aggregates["base_derived_union_raw"],
            "offset_zero_raw": aggregates["base_derived_off+0_raw"],
            "offset_zero_normmatched": aggregates[
                "base_derived_off+0_normmatched"
            ],
            "sae_single_normmatched": aggregates[
                next(
                    name
                    for name in aggregates
                    if name.startswith("sae_f")
                )
            ],
            "bootstrap_95": bootstrap,
            "matched_observational_auc_vs_causal_rate": {
                "n_offsets": len(band_rows),
                "pearson_r": float(
                    np.corrcoef(matched_aucs, causal_rates)[0, 1]
                ),
                "warning": (
                    "Exploratory correlation across only six highly "
                    "cosine-similar directions; not an independent "
                    "validation set."
                ),
            },
        }
    return summary


def plot(natural: dict, causal: dict | None, output: Path) -> None:
    known = natural["known_directions"]
    sae = natural["conventional_sae"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    ax = axes[0, 0]
    curves = [
        ("Residual: base-derived union", known["curves"]["base_union"], "#2667ff", 2.2),
        (
            "SAE: distributed reconstruction",
            sae["distributed_projection"]["curve"],
            "#f28e2b",
            2.0,
        ),
        (
            f"SAE: nearest single latent f{sae['best_positive_feature']}",
            sae["curves"][f"f{sae['best_positive_feature']}"],
            "#8b5cf6",
            1.5,
        ),
    ]
    for label, curve, color, width in curves:
        x, y = _xy(curve)
        ax.plot(x, y, marker=".", ms=3, lw=width, color=color, label=label)
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.axvspan(-13, -8, color="#54a24b", alpha=0.15, label="Ward window")
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set(xlabel="Tokens relative to backtracking sentence", ylabel="Event vs neutral AUC")
    ax.set_title("Prespecified feature presence")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[0, 1]
    for arm, color in (("base", "#2667ff"), ("reasoning", "#e15759")):
        rows = known["cross_offset"][arm]["matched_offset"]
        ax.plot(
            [row["offset"] for row in rows],
            [row["auc"] for row in rows],
            marker="o",
            color=color,
            label=f"{arm}-derived direction",
        )
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.axvspan(-13, -8, color="#54a24b", alpha=0.15)
    ax.set(xlabel="Direction derivation/observation offset", ylabel="Matched AUC")
    ax.set_title("Matched-offset observational contrast")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    cross = known["cross_offset"]["base"]
    observation_offsets = np.asarray(cross["observation_offsets"])
    keep = (observation_offsets >= -24) & (observation_offsets <= 8)
    matrix = np.asarray(cross["auc"])[:, keep]
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=0.5,
        vmax=max(0.7, float(np.max(matrix))),
        cmap="viridis",
        extent=[
            observation_offsets[keep][0] - 0.5,
            observation_offsets[keep][-1] + 0.5,
            -0.5,
            matrix.shape[0] - 0.5,
        ],
    )
    ax.set_yticks(
        np.arange(len(cross["direction_source_offsets"])),
        labels=cross["direction_source_offsets"],
    )
    ax.axvspan(-13, -8, color="white", alpha=0.12)
    ax.axvline(0, color="white", lw=0.8, alpha=0.8)
    ax.set(xlabel="Observation offset", ylabel="Direction derivation offset")
    ax.set_title("Cross-offset AUC: base-derived directions")
    figure.colorbar(image, ax=ax, label="AUC", fraction=0.046)

    ax = axes[1, 1]
    if causal is None:
        ax.text(0.5, 0.5, "Causal run not available", ha="center", va="center")
        ax.set_axis_off()
    else:
        aggregates = causal["aggregates"]
        rows = [
            (_causal_offset(name), values["mean_keyword_rate"])
            for name, values in aggregates.items()
            if _causal_offset(name) is not None
            and _causal_offset(name) != 0
        ]
        rows.sort()
        ax.plot(
            [row[0] for row in rows],
            [row[1] for row in rows],
            marker="o",
            lw=2,
            color="#e15759",
            label="Raw per-offset DoM",
        )
        baseline = aggregates["stage_a_magnitude_zero_baseline"][
            "mean_keyword_rate"
        ]
        ax.axhline(
            baseline,
            color="black",
            ls=":",
            lw=1,
            label="Magnitude-zero baseline",
        )
        extras = [
            ("union", aggregates["base_derived_union_raw"], "#2667ff"),
            (
                "offset 0\nnorm matched",
                aggregates["base_derived_off+0_normmatched"],
                "#59a14f",
            ),
            (
                "single SAE\nnorm matched",
                aggregates[
                    next(
                        name
                        for name in aggregates
                        if name.startswith("sae_f")
                    )
                ],
                "#8b5cf6",
            ),
        ]
        for index, (label, values, color) in enumerate(extras):
            ax.scatter(
                [-6.5 + index * 1.2],
                [values["mean_keyword_rate"]],
                s=50,
                color=color,
                label=label,
                zorder=3,
            )
        ax.axvspan(-13, -8, color="#54a24b", alpha=0.15)
        ax.set(xlabel="Direction derivation offset", ylabel="Mean keyword rate")
        ax.set_title(f"Held-out causal efficacy at magnitude {causal['magnitude']:g}")
        ax.legend(fontsize=7, ncol=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", type=Path, default=DEFAULT_NATURAL)
    parser.add_argument("--causal", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    natural = json.loads(args.natural.read_text())
    causal = json.loads(args.causal.read_text()) if args.causal.exists() else None
    summary = analyze(natural, causal)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    plot(natural, causal, args.figure)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
