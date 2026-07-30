"""Join the published Figure 2 baselines to the new Spectral-v1 sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINES = POWER_ROOT / "results" / "paper_synthetic_baselines.json"
DEFAULT_SPECTRAL = (
    POWER_ROOT / "results" / "paper_synthetic_v1_remote" / "summary.json"
)
DEFAULT_OUTPUT = (
    POWER_ROOT / "results" / "paper_synthetic_v1_comparison.json"
)
ARCHITECTURES = ("topk_sae", "tsae_paper", "txc_base", "spectral_v1")
LABELS = {
    "topk_sae": "TopK SAE",
    "tsae_paper": "T-SAE",
    "txc_base": "TXC-base",
    "spectral_v1": "Spectral v1",
}


def build_comparison(
    baselines: dict[str, Any],
    spectral: dict[str, Any],
) -> dict[str, Any]:
    if not spectral.get("complete"):
        raise RuntimeError("refusing to compare an incomplete Spectral-v1 sweep")
    tasks: dict[str, Any] = {}
    for task_name in ("denoising", "coupling"):
        baseline_task = baselines["tasks"][task_name]
        rows = []
        for architecture in ARCHITECTURES[:-1]:
            value = baseline_task["architectures"][architecture]
            rows.append(
                {
                    **value,
                    "architecture": architecture,
                    "label": LABELS[architecture],
                    "source": "published Figure 2 snapshot",
                    "metric": baseline_task["metric"],
                }
            )
        spectral_value = spectral["best_cells"][task_name]
        rows.append(
            {
                **spectral_value,
                "architecture": "spectral_v1",
                "label": LABELS["spectral_v1"],
                "t_label": f"T={spectral_value['T']}",
                "source": "new paper-recipe sweep",
                "metric": baseline_task["metric"],
            }
        )
        txc = next(row for row in rows if row["architecture"] == "txc_base")
        spectral_row = rows[-1]
        spectral_row["delta_vs_txc"] = (
            float(spectral_row["mean"]) - float(txc["mean"])
        )
        tasks[task_name] = {
            "metric": baseline_task["metric"],
            "rows": rows,
        }
    return {
        "schema_version": 1,
        "architectures": list(ARCHITECTURES),
        "baseline_source": baselines["source"],
        "spectral_run": spectral["run_name"],
        "tasks": tasks,
        "notes": {
            "selection": (
                "Each bar is the best seed-mean hyperparameter cell. "
                "Published bars retain the historical renderer's duplicate rows; "
                "Spectral v1 requires all three configured seeds."
            ),
            "coupling": spectral["coupling_metric_note"],
            "backbone": (
                "Spectral v1 is BatchTopK plus AuxK; the published TopK SAE "
                "and TXC-base use the legacy TopK-to-ReLU backbone."
            ),
        },
    }


def write_csv(payload: dict[str, Any], path: Path) -> None:
    rows = []
    for task_name, task in payload["tasks"].items():
        for row in task["rows"]:
            rows.append(
                {
                    "task": task_name,
                    "metric": task["metric"],
                    "architecture": row["architecture"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "min": row["min"],
                    "max": row["max"],
                    "n": row["n"],
                    "t_label": row["t_label"],
                    "k_pos": row["k_pos"],
                    "source": row["source"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--spectral", type=Path, default=DEFAULT_SPECTRAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_comparison(
        json.loads(args.baselines.read_text()),
        json.loads(args.spectral.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(payload, args.output.with_suffix(".csv"))
    print(args.output)


if __name__ == "__main__":
    main()
