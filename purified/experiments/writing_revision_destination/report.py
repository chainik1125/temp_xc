"""Render reviewer-ready KLiCKe deletion-destination figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROTOCOL_VERSION = "klicke-deletion-raw-activation-gate-v1"
EXPECTED_WINDOWS = tuple(range(1, 11))
TARGETS = {
    "capped_token_label": {
        "slug": "token_distance",
        "title": "Predicting deletion distance from pre-deletion activations",
        "description": "capped model-token deletion distance (2, 3, 4, 5, 6+)",
    },
    "lexical_label": {
        "slug": "lexical_destination",
        "title": "Predicting lexical deletion destination from pre-deletion activations",
        "description": "deleted-word count (2, 3, 4, 5+)",
    },
}
SERIES = (
    ("ordered", "Ordered history", "#2F6BFF", "o", "-"),
    ("best_offset", "Best single offset", "#343A46", "P", "--"),
    ("endpoint", "Last token", "#737A86", "s", "--"),
    (
        "invariant_mean_std_max",
        "Order-invariant summary",
        "#14866D",
        "v",
        "-.",
    ),
    (
        "ordered_retrained_shuffle",
        "Shuffled history (refit)",
        "#9B4DCA",
        "D",
        ":",
    ),
    ("second_difference", "Second differences", "#C06A22", "^", "-."),
)
GAP_SERIES = tuple(series for series in SERIES if series[0] != "ordered")


def _load_raw_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"{path}: expected protocol {PROTOCOL_VERSION}, got "
            f"{payload.get('protocol_version')!r}"
        )
    target = payload.get("target")
    if target not in TARGETS:
        raise ValueError(f"{path}: unsupported target {target!r}")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{path}: results must be a mapping")
    observed_windows = tuple(sorted(int(window) for window in results))
    if observed_windows != EXPECTED_WINDOWS:
        raise ValueError(
            f"{path}: expected windows {EXPECTED_WINDOWS}, got "
            f"{observed_windows}"
        )
    configured_windows = tuple(
        int(window)
        for window in payload.get("configuration", {}).get(
            "window_sizes", ()
        )
    )
    if configured_windows != EXPECTED_WINDOWS:
        raise ValueError(
            f"{path}: configuration window grid is {configured_windows}"
        )
    if int(payload.get("rows", 0)) < 1 or int(payload.get("writers", 0)) < 1:
        raise ValueError(f"{path}: rows and writers must be positive")

    always_required = {
        "ordered",
        "best_offset",
        "endpoint",
        "invariant_mean_std_max",
        "ordered_retrained_shuffle",
    }
    for window in EXPECTED_WINDOWS:
        record = results[str(window)]
        if int(record.get("window_tokens", -1)) != window:
            raise ValueError(f"{path}: window record {window} is mislabeled")
        metrics = record.get("metrics", {})
        missing = always_required.difference(metrics)
        if missing:
            raise ValueError(
                f"{path}: window {window} is missing metrics {sorted(missing)}"
            )
        if window >= 3 and "second_difference" not in metrics:
            raise ValueError(
                f"{path}: window {window} is missing second_difference"
            )
        for name, values in metrics.items():
            if "log_loss" not in values or "balanced_accuracy" not in values:
                raise ValueError(
                    f"{path}: incomplete metrics for T={window}, {name}"
                )
            if not np.isfinite(float(values["log_loss"])):
                raise ValueError(
                    f"{path}: non-finite log loss for T={window}, {name}"
                )
        bootstrap = record.get("equal_writer_bootstrap", {})
        for name in always_required - {"ordered"}:
            gap = bootstrap.get(f"{name}_minus_ordered")
            if not isinstance(gap, dict):
                raise ValueError(
                    f"{path}: missing paired gap for T={window}, {name}"
                )
            lower = float(gap["ci95_lower"])
            mean = float(gap["equal_writer_mean_log_loss_difference"])
            upper = float(gap["ci95_upper"])
            if not lower <= mean <= upper:
                raise ValueError(
                    f"{path}: invalid CI ordering for T={window}, {name}"
                )
    return payload


def _metric(payload: dict, window: int, name: str) -> dict | None:
    return payload["results"][str(window)]["metrics"].get(name)


def _gap(payload: dict, window: int, name: str) -> dict | None:
    return payload["results"][str(window)][
        "equal_writer_bootstrap"
    ].get(f"{name}_minus_ordered")


def _publication_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for window in EXPECTED_WINDOWS:
        for name, label, *_ in SERIES:
            metric = _metric(payload, window, name)
            if metric is None:
                continue
            gap = None if name == "ordered" else _gap(payload, window, name)
            rows.append(
                {
                    "target": payload["target"],
                    "window_tokens": window,
                    "view": name,
                    "view_label": label,
                    "log_loss": float(metric["log_loss"]),
                    "balanced_accuracy": float(
                        metric["balanced_accuracy"]
                    ),
                    "control_minus_ordered_equal_writer": (
                        ""
                        if gap is None
                        else float(
                            gap[
                                "equal_writer_mean_log_loss_difference"
                            ]
                        )
                    ),
                    "ci95_lower": (
                        "" if gap is None else float(gap["ci95_lower"])
                    ),
                    "ci95_upper": (
                        "" if gap is None else float(gap["ci95_upper"])
                    ),
                    "writers_positive": (
                        "" if gap is None else int(gap["writers_positive"])
                    ),
                    "writers_total": (
                        "" if gap is None else int(gap["writers_total"])
                    ),
                }
            )
    return rows


def _best_window(payload: dict) -> int:
    return min(
        EXPECTED_WINDOWS,
        key=lambda window: float(
            _metric(payload, window, "ordered")["log_loss"]
        ),
    )


def _plot(payload: dict, output_png: Path, output_pdf: Path) -> None:
    spec = TARGETS[payload["target"]]
    figure, (loss_axis, gap_axis) = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.7),
        gridspec_kw={"width_ratios": (1.18, 1.0)},
    )

    for name, label, color, marker, style in SERIES:
        windows = [
            window
            for window in EXPECTED_WINDOWS
            if _metric(payload, window, name) is not None
        ]
        losses = [
            float(_metric(payload, window, name)["log_loss"])
            for window in windows
        ]
        loss_axis.plot(
            windows,
            losses,
            label=label,
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=2.5 if name == "ordered" else 1.7,
            markersize=5.5,
            zorder=4 if name == "ordered" else 2,
        )

    best_window = _best_window(payload)
    best_loss = float(_metric(payload, best_window, "ordered")["log_loss"])
    loss_axis.annotate(
        f"best ordered: T={best_window}, {best_loss:.3f}",
        xy=(best_window, best_loss),
        xytext=(12, 18),
        textcoords="offset points",
        fontsize=9,
        color="#2F6BFF",
        arrowprops={"arrowstyle": "-", "color": "#2F6BFF", "lw": 1.0},
    )
    loss_axis.set_xlabel("Strict pre-deletion history length $T$ (tokens)")
    loss_axis.set_ylabel("Multiclass log loss\n(lower is better)")
    loss_axis.set_xticks(EXPECTED_WINDOWS)
    loss_axis.set_title("Absolute predictive performance", loc="left")
    loss_axis.grid(axis="y", color="#D8DCE3", linewidth=0.7, alpha=0.8)
    loss_axis.spines[["top", "right"]].set_visible(False)
    loss_axis.legend(frameon=False, fontsize=8.3, ncols=2)

    for name, label, color, marker, style in GAP_SERIES:
        windows = [
            window
            for window in EXPECTED_WINDOWS
            if _gap(payload, window, name) is not None
        ]
        records = [_gap(payload, window, name) for window in windows]
        means = np.asarray(
            [
                float(record["equal_writer_mean_log_loss_difference"])
                for record in records
            ]
        )
        lower = np.asarray(
            [float(record["ci95_lower"]) for record in records]
        )
        upper = np.asarray(
            [float(record["ci95_upper"]) for record in records]
        )
        gap_axis.errorbar(
            windows,
            means,
            yerr=np.vstack((means - lower, upper - means)),
            label=label,
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=1.7,
            markersize=5.0,
            capsize=2.5,
        )
    gap_axis.axhline(0.0, color="#555B66", linewidth=1.0)
    gap_axis.set_xlabel("Strict pre-deletion history length $T$ (tokens)")
    gap_axis.set_ylabel(
        "Control − ordered log loss\n(positive = order helps)"
    )
    gap_axis.set_xticks(EXPECTED_WINDOWS)
    gap_axis.set_title(
        "Equal-writer paired gaps with 95% CIs",
        loc="left",
    )
    gap_axis.grid(axis="y", color="#D8DCE3", linewidth=0.7, alpha=0.8)
    gap_axis.spines[["top", "right"]].set_visible(False)
    gap_axis.legend(frameon=False, fontsize=8.3, ncols=1)

    figure.suptitle(
        spec["title"],
        fontsize=13,
        fontweight="semibold",
        x=0.055,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, dpi=300, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _summary(payload: dict, files: dict[str, str]) -> dict:
    best_window = _best_window(payload)
    ordered = _metric(payload, best_window, "ordered")
    controls = {}
    for name, label, *_ in GAP_SERIES:
        gap = _gap(payload, best_window, name)
        if gap is None:
            continue
        controls[name] = {
            "label": label,
            "mean_control_minus_ordered": float(
                gap["equal_writer_mean_log_loss_difference"]
            ),
            "ci95": [
                float(gap["ci95_lower"]),
                float(gap["ci95_upper"]),
            ],
        }
    return {
        "status": "complete",
        "protocol_version": payload["protocol_version"],
        "target": payload["target"],
        "target_description": TARGETS[payload["target"]]["description"],
        "rows": int(payload["rows"]),
        "writers": int(payload["writers"]),
        "windows": list(EXPECTED_WINDOWS),
        "best_ordered": {
            "window_tokens": best_window,
            "log_loss": float(ordered["log_loss"]),
            "balanced_accuracy": float(ordered["balanced_accuracy"]),
        },
        "paired_controls_at_best_window": controls,
        "files": files,
    }


def _write_markdown(summary: dict, path: Path) -> None:
    best = summary["best_ordered"]
    lines = [
        f"# {TARGETS[summary['target']]['title']}",
        "",
        f"![{summary['target']}]({summary['files']['png']})",
        "",
        (
            f"Protocol `{summary['protocol_version']}`; "
            f"{summary['rows']:,} events from {summary['writers']:,} writers. "
            f"The best ordered model is T={best['window_tokens']} "
            f"(log loss {best['log_loss']:.3f}, balanced accuracy "
            f"{best['balanced_accuracy']:.3f}). Positive paired gaps mean the "
            "control is worse than ordered history."
        ),
        "",
        "| Control at best T | Control − ordered log loss | 95% CI |",
        "|---|---:|---:|",
    ]
    for record in summary["paired_controls_at_best_window"].values():
        lower, upper = record["ci95"]
        lines.append(
            f"| {record['label']} | "
            f"{record['mean_control_minus_ordered']:.3f} | "
            f"[{lower:.3f}, {upper:.3f}] |"
        )
    lines.extend(
        [
            "",
            (
                "The right panel uses an equal-writer bootstrap, so prolific "
                "writers cannot dominate the uncertainty estimate."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_publication(input_paths: list[Path], output_dir: Path) -> list[dict]:
    payloads = [_load_raw_result(path) for path in input_paths]
    targets = [payload["target"] for payload in payloads]
    if len(set(targets)) != len(targets):
        raise ValueError(f"duplicate targets: {targets}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for payload in payloads:
        slug = TARGETS[payload["target"]]["slug"]
        files = {
            "png": f"{slug}.png",
            "pdf": f"{slug}.pdf",
            "csv": f"{slug}.csv",
            "markdown": f"{slug}.md",
            "json": f"{slug}.summary.json",
        }
        _plot(
            payload,
            output_dir / files["png"],
            output_dir / files["pdf"],
        )
        rows = _publication_rows(payload)
        _write_csv(rows, output_dir / files["csv"])
        summary = _summary(payload, files)
        (output_dir / files["json"]).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_markdown(summary, output_dir / files["markdown"])
        summaries.append(summary)
    (output_dir / "publication_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summaries


def _render_legacy(payload: dict, output: Path) -> None:
    sweep = payload["fixed_cohort_window_sweep"]
    windows = sorted(int(value) for value in sweep)
    views = {
        "ordered": ("Ordered history", "#2F6BFF", "o", "-"),
        "endpoint": ("Last word only", "#555B66", "s", "--"),
        "bag": ("Order-invariant bag", "#14866D", "v", "-."),
        "canonical": ("Canonical multiset", "#C06A22", "^", "-."),
        "reverse": ("Reversed at test", "#9B4DCA", "D", ":"),
    }
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for key, (label, color, marker, style) in views.items():
        values = [
            sweep[str(window)][key]["log_loss"] for window in windows
        ]
        axis.plot(
            windows,
            values,
            label=label,
            color=color,
            marker=marker,
            linestyle=style,
            linewidth=2.0,
            markersize=5.5,
        )
    axis.set_xlabel("Strict pre-deletion lexical history length $T$")
    axis.set_ylabel("Held-out multiclass log loss (lower is better)")
    axis.set_xticks(windows)
    axis.grid(axis="y", color="#D8DCE3", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=2, loc="upper left")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    first = json.loads(args.input[0].read_text(encoding="utf-8"))
    if "fixed_cohort_window_sweep" in first:
        if len(args.input) != 1 or args.output is None:
            parser.error("legacy input requires one --input and --output")
        _render_legacy(first, args.output)
        return
    if args.output_dir is None:
        parser.error("raw-activation inputs require --output-dir")
    summaries = render_publication(args.input, args.output_dir)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
