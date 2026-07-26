"""Render frozen-dictionary C7 order controls as a figure and table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARCH_COLORS = {"txc_base": "#5E81AC", "topk_sae": "#D08770"}


def load_dictionary(path: Path) -> tuple[dict, list[dict]]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    metadata = next(row for row in records if row.get("record_type") == "metadata")
    rows = [row for row in records if row.get("record_type") != "metadata"]
    return metadata, rows


def summarize(inputs: list[Path]) -> dict:
    summaries = []
    provenance = []
    for path in inputs:
        metadata, rows = load_dictionary(path)
        provenance.append(metadata)
        for n_features in sorted({row["n_features"] for row in rows}):
            group = [row for row in rows if row["n_features"] == n_features]
            ordered = np.asarray([row["ordered"]["pr_auc"] for row in group])
            controls = {
                name: np.asarray([row["controls"][name]["pr_auc"] for row in group])
                for name in ("shuffle", "reverse", "circular")
            }
            summaries.append(
                {
                    "arch": metadata["arch"],
                    "n_features": n_features,
                    "n_folds": len(group),
                    "ordered_fold_values": ordered.tolist(),
                    "ordered_mean": float(ordered.mean()),
                    "ordered_std": float(ordered.std(ddof=1)),
                    "control_means": {
                        name: float(values.mean()) for name, values in controls.items()
                    },
                    "gap_means": {
                        name: float(np.mean(ordered - values))
                        for name, values in controls.items()
                    },
                    "gap_fold_values": {
                        name: (ordered - values).tolist()
                        for name, values in controls.items()
                    },
                }
            )
    return {"provenance": provenance, "summaries": summaries}


def markdown_table(payload: dict) -> str:
    lines = [
        "| architecture | probe S | ordered PR-AUC | shuffled | reversed | circular | "
        "shuffle gap | reversal gap | circular gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(payload["summaries"], key=lambda x: (x["arch"], x["n_features"])):
        lines.append(
            "| {arch} | {features} | {ordered:.3f} | {shuffle:.3f} | {reverse:.3f} | "
            "{circular:.3f} | {shuffle_gap:+.3f} | {reverse_gap:+.3f} | "
            "{circular_gap:+.3f} |".format(
                arch=row["arch"],
                features=row["n_features"],
                ordered=row["ordered_mean"],
                shuffle=row["control_means"]["shuffle"],
                reverse=row["control_means"]["reverse"],
                circular=row["control_means"]["circular"],
                shuffle_gap=row["gap_means"]["shuffle"],
                reverse_gap=row["gap_means"]["reverse"],
                circular_gap=row["gap_means"]["circular"],
            )
        )
    return "\n".join(lines) + "\n"


def plot_dictionary(payload: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), constrained_layout=True)
    for arch in sorted({row["arch"] for row in payload["summaries"]}):
        rows = sorted(
            [row for row in payload["summaries"] if row["arch"] == arch],
            key=lambda row: row["n_features"],
        )
        x = np.asarray([row["n_features"] for row in rows])
        mean = np.asarray([row["ordered_mean"] for row in rows])
        sem = np.asarray([row["ordered_std"] / np.sqrt(row["n_folds"]) for row in rows])
        axes[0].errorbar(
            x,
            mean,
            yerr=1.96 * sem,
            marker="o",
            capsize=3,
            label=arch,
            color=ARCH_COLORS.get(arch),
        )
        gap = np.asarray([row["gap_means"]["shuffle"] for row in rows])
        axes[1].plot(
            x,
            gap,
            marker="o",
            label=arch,
            color=ARCH_COLORS.get(arch),
        )
    axes[0].set(
        xlabel="probe feature count S",
        ylabel="grouped-fold PR-AUC",
        title="Frozen dictionary detection",
    )
    axes[1].axhline(0.0, color="#4C566A", linewidth=1)
    axes[1].set(
        xlabel="probe feature count S",
        ylabel="ordered minus shuffled PR-AUC",
        title="Same-probe order dependence",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.input)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plot_dictionary(payload, args.figure)
    args.markdown.write_text(markdown_table(payload))


if __name__ == "__main__":
    main()
