"""Paired question-bootstrap audit for the pooled-SAE steering controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.power_spectrum.backtracking_sae_pooling.steering_baselines import (
    run_baselines,
)


ARMS = ("topk_sae", "pooled_sae_mean", "pooled_sae_max", "txc_base")
NEGATIVE_LOBE = (-12.0, -10.0, -8.0, -7.0, -6.0, -5.0)
POSITIVE_LOBE = (5.0, 6.0, 7.0, 8.0, 10.0, 12.0)


def successful_labels(
    rows: list[dict[str, Any]], *, seed: int = run_baselines.SEED
) -> dict[tuple[str, float, str], int]:
    """Keep the last successful label for each question, dose, and arm."""
    labels: dict[tuple[str, float, str], int] = {}
    for row in rows:
        arm = str(row.get("arch", ""))
        if arm not in ARMS or int(row.get("seed", -1)) != seed:
            continue
        label = int(row.get("label", -1))
        if label >= 0:
            labels[(str(row["transcript_id"]), float(row["magnitude"]), arm)] = label
    return labels


def per_question_lobe(
    labels: dict[tuple[str, float, str], int], arm: str, magnitudes: tuple[float, ...]
) -> dict[str, float]:
    qids = sorted(qid for qid, magnitude, name in labels if name == arm and magnitude == 0.0)
    out: dict[str, float] = {}
    for qid in qids:
        baseline = labels[(qid, 0.0, arm)]
        out[qid] = float(
            np.mean([labels[(qid, magnitude, arm)] - baseline for magnitude in magnitudes])
        )
    return out


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int = 42, draws: int = 50_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = successful_labels(rows)
    per_sign = {
        "negative": {
            arm: per_question_lobe(labels, arm, NEGATIVE_LOBE) for arm in ARMS
        },
        "positive": {
            arm: per_question_lobe(labels, arm, POSITIVE_LOBE) for arm in ARMS
        },
    }
    qid_sets = {
        f"{sign}:{arm}": set(values)
        for sign, per_arm in per_sign.items()
        for arm, values in per_arm.items()
    }
    if len({frozenset(qids) for qids in qid_sets.values()}) != 1:
        raise ValueError(f"arm qid mismatch: {qid_sets}")
    qids = sorted(next(iter(qid_sets.values())))
    if len(qids) != 61:
        raise ValueError(f"expected 61 questions, found {len(qids)}")

    result: dict[str, Any] = {
        "questions": len(qids),
        "negative_lobe_magnitudes": list(NEGATIVE_LOBE),
        "positive_lobe_magnitudes": list(POSITIVE_LOBE),
        "lobes": {},
        "exploratory_sign_aligned_differences": {},
    }
    comparisons = (
        ("pooled_sae_mean", "topk_sae"),
        ("pooled_sae_max", "topk_sae"),
        ("txc_base", "pooled_sae_mean"),
        ("txc_base", "pooled_sae_max"),
    )
    for sign, per_arm in per_sign.items():
        lobe_result: dict[str, Any] = {"arms": {}, "paired_differences": {}}
        for arm, values_by_qid in per_arm.items():
            values = np.asarray([values_by_qid[qid] for qid in qids], dtype=np.float64)
            lobe_result["arms"][arm] = {
                "mean": float(values.mean()),
                "bootstrap_95_ci": list(bootstrap_mean_ci(values)),
            }
        for left, right in comparisons:
            difference = np.asarray(
                [per_arm[left][qid] - per_arm[right][qid] for qid in qids],
                dtype=np.float64,
            )
            lobe_result["paired_differences"][f"{left}_minus_{right}"] = {
                "mean": float(difference.mean()),
                "bootstrap_95_ci": list(bootstrap_mean_ci(difference)),
            }
        result["lobes"][sign] = lobe_result

    for pooled_arm in ("pooled_sae_mean", "pooled_sae_max"):
        difference = np.asarray(
            [
                per_sign["positive"][pooled_arm][qid]
                - per_sign["negative"]["txc_base"][qid]
                for qid in qids
            ],
            dtype=np.float64,
        )
        result["exploratory_sign_aligned_differences"][
            f"{pooled_arm}_positive_minus_txc_base_negative"
        ] = {
            "mean": float(difference.mean()),
            "bootstrap_95_ci": list(bootstrap_mean_ci(difference)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run_baselines.read_jsonl(args.judge_jsonl)
    result = audit(rows)
    run_baselines.atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
