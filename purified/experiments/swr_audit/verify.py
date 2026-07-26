"""Post-hoc integrity checks for persisted E1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.swr_audit.aggregate import conservative_swr


EXPECTED_ARTIFACT_SHA256 = "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--swr", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matched = json.loads(args.matched.read_text())
    records = [json.loads(line) for line in args.swr.read_text().splitlines()]
    metadata = next(row for row in records if row.get("record_type") == "metadata")
    rows = [row for row in records if row.get("record_type") != "metadata"]
    summary = json.loads(args.summary.read_text())
    audited = summary["summaries"][0]

    checks = {
        "artifact_sha256": matched["artifact_sha256"] == EXPECTED_ARTIFACT_SHA256,
        "matched_rows": matched["n_rows"] == 25_204,
        "matched_groups": matched["n_groups"] == 300,
        "swr_rows": metadata["n_rows"] == 25_204,
        "swr_groups": metadata["n_groups"] == 300,
        "five_unique_folds": sorted(row["fold"] for row in rows) == list(range(5)),
        "window_offsets": all(
            row["window_offsets"] == [-13, -12, -11, -10, -9, -8] for row in rows
        ),
        "parameter_counts": all(
            row["parameter_counts"]["ordered"] == 3_881
            and row["parameter_counts"]["mean_pool_param_matched"] == 4_081
            and row["parameter_counts"]["mean_pool_same_rank"] == 681
            for row in rows
        ),
        "conservative_gap_recomputed": np.allclose(
            [conservative_swr(row) for row in rows],
            audited["swr_pr_auc_conservative"]["fold_values"],
        ),
        "interval_present": (
            audited["swr_pr_auc_conservative"]["grouped_fold_t_95"] is not None
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"E1 verification failed: {checks}")
    result = {
        "status": "ok",
        "checks": checks,
        "note": (
            "post-hoc persisted-artifact verification; the original tmux wrapper "
            "did not record its process exit code"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
