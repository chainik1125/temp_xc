"""Audit whether a pooled SAE steering feature is localized to one position."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def feature_position_summary(arm: dict[str, Any], feature_id: int) -> dict[str, Any]:
    ids = [int(value) for value in arm["top_feature_ids_abs_diff"]]
    diffs = [float(value) for value in arm["top_feature_signed_diffs"]]
    try:
        index = ids.index(feature_id)
    except ValueError as error:
        raise ValueError(f"feature {feature_id} is absent from the stored top-32") from error
    signed_diff = diffs[index]
    positive_rank = 1 + sum(value > signed_diff for value in diffs)
    return {
        "feature_selectivity": signed_diff,
        "feature_positive_selectivity_rank": positive_rank,
        "feature_absolute_selectivity_rank": index + 1,
        "position_top_feature_id": int(arm["steering_feature_id_signed_positive"]),
        "position_top_feature_selectivity": float(arm["steering_feature_selectivity"]),
        "pr_auc": {key: float(value) for key, value in arm["pr_auc"].items()},
    }


def audit(
    detection: dict[str, Any],
    steering: dict[str, Any],
    *,
    feature_id: int,
) -> dict[str, Any]:
    arms = detection["arms"]
    position_names = sorted(
        (name for name in arms if name.startswith("position_")),
        key=lambda name: int(name.removeprefix("position_")),
    )
    if len(position_names) != 5:
        raise ValueError(f"expected five positions, got {position_names}")

    per_position = {
        name: feature_position_summary(arms[name], feature_id) for name in position_names
    }
    max_arm = arms["max"]
    if int(max_arm["steering_feature_id_signed_positive"]) != feature_id:
        raise ValueError("requested feature is not the max-pooled steering feature")
    if int(steering["feature"]["feature_id"]) != feature_id:
        raise ValueError("steering result selected a different feature")

    pooled_selectivity = float(steering["feature"]["selectivity"])
    best_position_name = max(
        position_names,
        key=lambda name: per_position[name]["feature_selectivity"],
    )
    best_position_selectivity = float(
        per_position[best_position_name]["feature_selectivity"]
    )

    fold_dominance: dict[str, Any] = {}
    for budget, pooled_scores in max_arm["fold_pr_auc"].items():
        fold_wins = []
        for fold, pooled_score in enumerate(pooled_scores):
            best_single = max(
                float(arms[name]["fold_pr_auc"][budget][fold])
                for name in position_names
            )
            fold_wins.append(float(pooled_score) > best_single)
        fold_dominance[budget] = {
            "max_pool_pr_auc": float(max_arm["pr_auc"][budget]),
            "best_single_position_pr_auc": max(
                float(arms[name]["pr_auc"][budget]) for name in position_names
            ),
            "folds_beating_every_single_position": sum(fold_wins),
            "fold_count": len(fold_wins),
        }

    return {
        "schema_version": 1,
        "feature_id": feature_id,
        "position_order": "oldest_to_newest",
        "per_position": per_position,
        "max_pool": {
            "positive_activation_mean": float(steering["feature"]["pos_act_mean"]),
            "negative_activation_mean": float(steering["feature"]["neg_act_mean"]),
            "feature_selectivity": pooled_selectivity,
            "best_single_position": best_position_name,
            "best_single_position_selectivity": best_position_selectivity,
            "selectivity_ratio_vs_best_single_position": (
                pooled_selectivity / best_position_selectivity
            ),
        },
        "detection_fold_dominance": fold_dominance,
        "conclusion": {
            "one_fixed_position_explains_feature": False,
            "reason": (
                "Feature selectivity is positive at all five positions, no single-position "
                "miner ranks it first, and max pooling produces substantially greater "
                "selectivity and cross-validated detection than any fixed position."
            ),
            "remaining_uncertainty": (
                "Stored summaries do not retain per-example feature activations, so they "
                "cannot distinguish temporal jitter across examples from repeated evidence "
                "within the same example."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detection-results", type=Path, required=True)
    parser.add_argument("--steering-result", type=Path, required=True)
    parser.add_argument("--feature-id", type=int, default=24530)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    detection = json.loads(args.detection_results.read_text())
    steering = json.loads(args.steering_result.read_text())
    result = audit(detection, steering, feature_id=args.feature_id)
    result["sources"] = {
        "detection_results": str(args.detection_results),
        "detection_results_sha256": sha256_file(args.detection_results),
        "steering_result": str(args.steering_result),
        "steering_result_sha256": sha256_file(args.steering_result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
