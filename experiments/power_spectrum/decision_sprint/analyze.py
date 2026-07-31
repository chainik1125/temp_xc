"""Apply the frozen decision rule and make the comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent


def bootstrap_mean(
    values: np.ndarray, *, seed: int = 20260731, draws: int = 100_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return np.quantile(sampled.mean(axis=1), [0.025, 0.975]).tolist()


def analyze(raw: dict, config: dict) -> dict:
    if raw.get("status") != "complete":
        raise ValueError("refusing to analyze an incomplete run")
    if raw.get("protocol") != config["protocol"]:
        raise ValueError("raw result protocol does not match config")
    rows = raw["results"]
    seeds = [int(row["seed"]) for row in rows]
    if (
        len(seeds) != len(set(seeds))
        or sorted(seeds) != sorted(config["seeds"])
    ):
        raise ValueError(
            f"expected unique seeds {config['seeds']}, got {seeds}"
        )
    txc = np.asarray(
        [
            row["canonical"]["txc"]["metrics"]["lambda_recovery_v2"]
            for row in rows
        ],
        dtype=float,
    )
    candidates = {}
    for rank in config["adapter_ranks"]:
        name = f"adapter_rank{rank}"
        candidates[name] = np.asarray(
            [row["evaluations"][name]["lambda_recovery_v2"] for row in rows]
        )
    primary_rank = int(config["primary_adapter_rank"])
    best_name = f"adapter_rank{primary_rank}"
    best = candidates[best_name]
    arrays = [txc, *candidates.values()]
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError("non-finite primary metric")
    txc_minus = txc - best
    best_reverse = np.asarray(
        [
            row["evaluations"][f"{best_name}_reverse"]["lambda_recovery_v2"]
            for row in rows
        ]
    )
    best_untrained = np.asarray(
        [
            row["evaluations"][f"{best_name}_untrained"][
                "lambda_recovery_v2"
            ]
            for row in rows
        ]
    )
    txc_reverse = np.asarray(
        [
            row["evaluations"]["txc_reverse"]["lambda_recovery_v2"]
            for row in rows
        ]
    )
    txc_untrained = np.asarray(
        [
            row["canonical"]["txc_untrained"]["metrics"][
                "lambda_recovery_v2"
            ]
            for row in rows
        ]
    )
    txc_l0 = np.asarray(
        [
            row["canonical"]["txc"]["metrics"]["l0_per_window"]
            for row in rows
        ]
    )
    adapter_l0 = np.asarray(
        [
            row["evaluations"][best_name]["l0_per_window"] for row in rows
        ]
    )
    realized_l0_gap = float(abs(txc_l0.mean() - adapter_l0.mean()))
    txc_order_drop = float((txc - txc_reverse).mean())
    paired_ci = bootstrap_mean(txc_minus)
    noninferior = bool(
        paired_ci[1] <= config["noninferiority_margin_r"]
    )
    sign_of_life = bool(
        float(txc_minus.mean()) >= config["sign_of_life_margin_r"]
        and paired_ci[0] > 0
        and txc_order_drop >= config["order_drop_margin_r"]
        and float(txc.mean()) > float(txc_untrained.mean())
        and float(best.mean()) > float(best_untrained.mean())
        and realized_l0_gap <= config["max_realized_l0_gap"]
    )
    if noninferior:
        verdict = "STOP_GENERAL_TXC"
    elif sign_of_life:
        verdict = "INITIAL_REAL_TASK_SIGN_OF_LIFE"
    else:
        verdict = "INCONCLUSIVE"
    models = {
        "TXC": txc.tolist(),
        **{name: values.tolist() for name, values in candidates.items()},
        "SAE last": [
            row["evaluations"]["sae_last"]["lambda_recovery_v2"]
            for row in rows
        ],
        "SAE mean top-8": [
            row["evaluations"]["sae_mean_top8"]["lambda_recovery_v2"]
            for row in rows
        ],
        "SAE max top-8": [
            row["evaluations"]["sae_max_top8"]["lambda_recovery_v2"]
            for row in rows
        ],
    }
    return {
        "verdict": verdict,
        "best_adapter": best_name,
        "models": {
            name: {
                "seed_values": values,
                "mean": float(np.mean(values)),
                "std_sample": float(np.std(values, ddof=1)),
            }
            for name, values in models.items()
        },
        "paired_txc_minus_best": {
            "seed_values": txc_minus.tolist(),
            "mean": float(txc_minus.mean()),
            "bootstrap_ci95": paired_ci,
        },
        "best_adapter_order_drop": {
            "seed_values": (best - best_reverse).tolist(),
            "mean": float((best - best_reverse).mean()),
        },
        "best_adapter_untrained": {
            "seed_values": best_untrained.tolist(),
            "mean": float(best_untrained.mean()),
        },
        "txc_order_drop": {
            "seed_values": (txc - txc_reverse).tolist(),
            "mean": txc_order_drop,
        },
        "txc_untrained": {
            "seed_values": txc_untrained.tolist(),
            "mean": float(txc_untrained.mean()),
        },
        "realized_l0": {
            "txc_seed_values": txc_l0.tolist(),
            "txc_mean": float(txc_l0.mean()),
            "adapter_seed_values": adapter_l0.tolist(),
            "adapter_mean": float(adapter_l0.mean()),
            "mean_gap": realized_l0_gap,
        },
        "gates": {
            "adapter_noninferior_ci_upper_within_0.03": noninferior,
            "txc_real_task_sign_of_life": sign_of_life,
            "txc_order_receipt_available": True,
            "realized_l0_gap_within_tolerance": bool(
                realized_l0_gap <= config["max_realized_l0_gap"]
            ),
        },
    }


def plot(summary: dict, path: Path) -> None:
    labels = list(summary["models"])
    means = [summary["models"][label]["mean"] for label in labels]
    stds = [summary["models"][label]["std_sample"] for label in labels]
    colors = [
        "#d95f02" if label == "TXC" else "#1b9e77" if "adapter" in label else "#777777"
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.9)
    for index, label in enumerate(labels):
        values = summary["models"][label]["seed_values"]
        ax.scatter(
            np.full(len(values), index) + np.linspace(-0.09, 0.09, len(values)),
            values,
            color="black",
            s=18,
            zorder=3,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Grouped-dialogue ridge Pearson r")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("DailyDialog turn-length trend, fresh paired seeds")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("/workspace/txc_decision_sprint/results/raw_results.json"),
    )
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/txc_decision_sprint/results"),
    )
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    config = json.loads(args.config.read_text())
    summary = analyze(raw, config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    plot(summary, args.output / "comparison.png")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
