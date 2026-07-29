"""Summarize and plot the two Ward feature-formation pilots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


BANDS = {
    "far_pre": (-64, -32),
    "ward": (-13, -8),
    "immediate_pre": (-2, -1),
    "expression": (0, 4),
}


def _band_mean(rows: list[dict], key: str, lo: int, hi: int) -> float:
    values = [row[key] for row in rows if lo <= row["offset"] <= hi]
    return float(np.mean(values)) if values else float("nan")


def summarize() -> dict:
    sae_free = json.loads((RESULTS / "ward_sae_free.json").read_text())
    sae = json.loads((RESULTS / "ward_sae_features.json").read_text())
    sources = {
        "sae_free": sae_free,
        "sae": sae,
    }
    summary = {}
    for source_name, payload in sources.items():
        summary[source_name] = {}
        for curve_name, rows in payload["curves"].items():
            summary[source_name][curve_name] = {
                band: {
                    "auc": _band_mean(rows, "auc", lo, hi),
                    "log_loss_gain_nats": _band_mean(
                        rows,
                        "log_loss_gain_nats",
                        lo,
                        hi,
                    ),
                }
                for band, (lo, hi) in BANDS.items()
            }
    summary["metadata"] = {
        "n_pairs": sae_free["pairing"]["n_pairs"],
        "model": sae_free["model"],
        "layer": sae_free["layer"],
        "ward_band": [-13, -8],
        "sae_reconstruction": sae["sae"]["reconstruction"],
        "modal_wall_seconds": sae["runtime"]["wall_seconds"],
    }
    return summary


def plot() -> Path:
    sae_free = json.loads((RESULTS / "ward_sae_free.json").read_text())
    sae = json.loads((RESULTS / "ward_sae_features.json").read_text())
    curves = [
        (
            "SAE-free, positionwise local",
            sae_free["curves"]["positionwise_local"],
            "#1f77b4",
            "-",
        ),
        (
            "SAE-free, fixed Ward-band local",
            sae_free["curves"]["transported_local"],
            "#1f77b4",
            "--",
        ),
        (
            "SAE, positionwise top-16",
            sae["curves"]["positionwise_top16"],
            "#d62728",
            "-",
        ),
        (
            "SAE, fixed Ward-band single feature",
            sae["curves"]["transported_single_feature"],
            "#d62728",
            "--",
        ),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, rows, color, linestyle in curves:
        offset = [row["offset"] for row in rows]
        axes[0].plot(
            offset,
            [row["auc"] for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
        axes[1].plot(
            offset,
            [row["log_loss_gain_nats"] for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    for axis in axes:
        axis.axvspan(-13, -8, color="#ffbf00", alpha=0.18)
        axis.axvline(0, color="black", linewidth=1, alpha=0.7)
        axis.grid(alpha=0.2)
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Held-out ROC-AUC")
    axes[1].set_ylabel("Log-loss gain (nats)")
    axes[1].set_xlabel("Tokens relative to genuine backtracking sentence")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title(
        "Ward Backtracking feature-formation calibration\n"
        "gold = Ward precursor band; vertical line = labelled sentence onset"
    )
    figure.tight_layout()
    output = RESULTS / "ward_feature_formation_curves.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> None:
    summary = summarize()
    summary_path = RESULTS / "ward_feature_formation_band_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    plot_path = plot()
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "plot": str(plot_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

