"""Summarize and plot the fresh two-arm steering calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ARM_LABELS = {"topk_sae": "Final-token SAE", "txc_base": "TXC-base"}
ARM_COLORS = {"topk_sae": "#3569b7", "txc_base": "#d1495b"}
NEGATIVE_LOBE_MAGNITUDES = [-12.0, -10.0, -8.0, -7.0, -6.0, -5.0]


def metric_curve(metrics: dict[str, Any], magnitudes: list[float]) -> list[float]:
    return [float(metrics[f"delta_gc_mag_{float(magnitude):+.1f}"]) for magnitude in magnitudes]


def summarize(workspace: Path, historical_path: Path, output_dir: Path) -> dict[str, Any]:
    historical = json.loads(historical_path.read_text())
    magnitudes = [float(value) for value in historical["magnitudes"]]
    fresh = {
        arm: json.loads((workspace / f"{arm}_result.json").read_text())
        for arm in ARM_LABELS
    }

    summary: dict[str, Any] = {
        "magnitudes": magnitudes,
        "negative_lobe_magnitudes": NEGATIVE_LOBE_MAGNITUDES,
        "arms": {},
    }
    for arm, result in fresh.items():
        metrics = result["metrics"]
        curve = metric_curve(metrics, magnitudes)
        reference = historical["arms"][arm]
        fresh_by_magnitude = dict(zip(magnitudes, curve, strict=True))
        historical_by_magnitude = dict(
            zip(magnitudes, reference["delta_gc"], strict=True)
        )
        negative_lobe = [fresh_by_magnitude[value] for value in NEGATIVE_LOBE_MAGNITUDES]
        historical_negative_lobe = [
            historical_by_magnitude[value] for value in NEGATIVE_LOBE_MAGNITUDES
        ]
        decoder_norm = float(result["feature"]["decoder_norm"])
        summary["arms"][arm] = {
            "train_key": result["train_key"],
            "feature_id": result["feature"]["feature_id"],
            "feature_selectivity": result["feature"]["selectivity"],
            "decoder_norm": decoder_norm,
            "checkpoint_sha256": result["checkpoint_sha256"],
            "delta_gc": curve,
            "delta_gc_peak": float(metrics["delta_gc_peak"]),
            "delta_gc_peak_magnitude": float(metrics["delta_gc_peak_magnitude"]),
            "peak_residual_l2": abs(float(metrics["delta_gc_peak_magnitude"]))
            * decoder_norm,
            "negative_lobe_mean": sum(negative_lobe) / len(negative_lobe),
            "historical_negative_lobe_mean": sum(historical_negative_lobe)
            / len(historical_negative_lobe),
            "historical_delta_gc_peak": float(reference["delta_gc_peak"]),
            "peak_change": float(metrics["delta_gc_peak"] - reference["delta_gc_peak"]),
            "successful_judge_keys": int(result["successful_judge_keys"]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for arm, label in ARM_LABELS.items():
        color = ARM_COLORS[arm]
        ax.plot(
            magnitudes,
            historical["arms"][arm]["delta_gc"],
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            color=color,
            label=f"{label} (May reference)",
        )
        ax.plot(
            magnitudes,
            summary["arms"][arm]["delta_gc"],
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=color,
            label=f"{label} (fresh)",
        )
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Steering magnitude")
    ax.set_ylabel("Backtracking inducement, Δgc")
    ax.set_title("C7 cut25 steering: matched 20k baselines")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "steering_comparison.png", dpi=200)
    svg_path = output_dir / "steering_comparison.svg"
    fig.savefig(svg_path)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.workspace, args.historical, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
