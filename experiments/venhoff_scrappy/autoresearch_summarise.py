"""Autoresearch Δ-reporter for venhoff_scrappy.

Reads a candidate's grade_results.json, looks up the declared baseline's
grade_results.json, computes Δ Gap Recovery, emits a verdict, and
appends a v1 schema row to results/autoresearch_index.jsonl.

Pattern adopted from Han's experiments/phase5_downstream_utility/autoresearch_summarise.py.

Schema v1 additions (web-claude review 2026-04-23):
  - schema_version
  - per_task_outcomes (20-element bool array, enables paired post-hoc)
  - hybrid_acc, absolute_gap_{cand,base}  (co-metrics)
  - best_cell (winning coefficient × window)
  - gap_recovery_per_gpu_minute (cost-adjusted Δ)
  - phase0_cache_hash (provenance stamp)

Usage:
    .venv/bin/python experiments/venhoff_scrappy/autoresearch_summarise.py \\
        --candidate tempxc_sum_layer10 --write-index
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SCRAPPY = REPO / "experiments/venhoff_scrappy"
CANDIDATES_DIR = SCRAPPY / "candidates"
CYCLES_DIR = SCRAPPY / "results/cycles"
INDEX_JSONL = SCRAPPY / "results/autoresearch_index.jsonl"
DEFAULTS_YAML = SCRAPPY / "config.yaml"

SCHEMA_VERSION = 1

# Required keys in grade_results.json before we'll score.
REQUIRED_GRADE_KEYS = {
    "candidate", "arch", "n_tasks",
    "thinking_acc", "base_acc", "hybrid_acc",
    "gap_recovery",
}


def load_grade(candidate: str) -> dict | None:
    p = CYCLES_DIR / candidate / "grade_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_candidate_cfg(candidate: str) -> dict:
    return yaml.safe_load((CANDIDATES_DIR / f"{candidate}.yaml").read_text())


def verdict_for(delta_pp: float, defaults: dict) -> str:
    t = defaults["autoresearch"]
    if delta_pp > t["finalist_threshold_pp"]:
        return "FINALIST"
    if delta_pp < t["discard_threshold_pp"]:
        return "DISCARD"
    promising = t.get("promising_threshold_pp", 3.0)
    if delta_pp > promising:
        return "PROMISING"
    return "AMBIGUOUS"


def validate_grade_schema(grade: dict, candidate: str) -> None:
    """Reject partial grade outputs early. A missing key here usually
    means a downstream dispatch step crashed after writing some but
    not all fields."""
    missing = REQUIRED_GRADE_KEYS - set(grade.keys())
    if missing:
        raise ValueError(
            f"grade_results.json for {candidate!r} missing keys: {sorted(missing)}. "
            f"Either run_cycle.py crashed mid-write, or the schema drifted."
        )
    # Scaffold-mode grades have gap_recovery=None — treated as INSUFFICIENT
    # downstream, not an error here.


def summarise(candidate: str, write_index: bool = False) -> dict:
    cand_cfg = load_candidate_cfg(candidate)
    defaults = yaml.safe_load(DEFAULTS_YAML.read_text())
    baseline = cand_cfg.get("baseline", defaults["autoresearch"]["default_baseline"])

    cand_grade = load_grade(candidate)
    if cand_grade is None:
        raise FileNotFoundError(f"no grade_results.json for candidate={candidate}")
    validate_grade_schema(cand_grade, candidate)

    base_grade = load_grade(baseline) if baseline != candidate else cand_grade
    if base_grade is None:
        raise FileNotFoundError(
            f"baseline {baseline!r} has no grade_results.json — run it first"
        )
    validate_grade_schema(base_grade, baseline)

    cand_gr = cand_grade.get("gap_recovery")
    base_gr = base_grade.get("gap_recovery")

    if cand_gr is None or base_gr is None:
        delta_pp = None
        verdict = "INSUFFICIENT"
    else:
        delta_pp = 100.0 * (cand_gr - base_gr)
        verdict = verdict_for(delta_pp, defaults)

    wall = cand_grade.get("wall_time_s") or 1.0
    gpu_minutes = wall / 60.0
    gr_per_gpu_min = (cand_gr / gpu_minutes) if (cand_gr is not None and gpu_minutes > 0) else None

    row = {
        "schema_version": SCHEMA_VERSION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": candidate,
        "baseline": baseline,
        "arch_cand": cand_grade.get("arch"),
        "arch_base": base_grade.get("arch"),
        "n_tasks": cand_grade.get("n_tasks"),

        # Absolute accuracies
        "thinking_acc": cand_grade.get("thinking_acc"),
        "base_acc": cand_grade.get("base_acc"),
        "hybrid_acc_cand": cand_grade.get("hybrid_acc"),
        "hybrid_acc_base": base_grade.get("hybrid_acc"),

        # Gap Recovery + Δ
        "gap_recovery_cand": cand_gr,
        "gap_recovery_base": base_gr,
        "delta_pp": delta_pp,

        # Co-metrics
        "absolute_gap_cand": (
            cand_grade["hybrid_acc"] - cand_grade["base_acc"]
            if cand_grade.get("hybrid_acc") is not None and cand_grade.get("base_acc") is not None
            else None
        ),
        "absolute_gap_base": (
            base_grade["hybrid_acc"] - base_grade["base_acc"]
            if base_grade.get("hybrid_acc") is not None and base_grade.get("base_acc") is not None
            else None
        ),
        "best_cell": cand_grade.get("best_cell"),

        # Paired analysis support (web-claude review)
        "per_task_outcomes_cand": cand_grade.get("per_task_outcomes"),
        "per_task_outcomes_base": base_grade.get("per_task_outcomes"),

        # Provenance
        "phase0_cache_hash": cand_grade.get("phase0_cache_hash"),

        # Cost
        "wall_time_s": wall,
        "gap_recovery_per_gpu_minute": gr_per_gpu_min,

        "verdict": verdict,
        "scaffold": bool(cand_grade.get("scaffold")),
    }

    # Compact print — drop the per-task vector in stdout for readability.
    pretty = {k: v for k, v in row.items() if k not in ("per_task_outcomes_cand", "per_task_outcomes_base")}
    print(json.dumps(pretty, indent=2))

    if write_index:
        INDEX_JSONL.parent.mkdir(parents=True, exist_ok=True)
        with INDEX_JSONL.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[info] appended row to {INDEX_JSONL}")

    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--write-index", action="store_true")
    args = ap.parse_args()
    summarise(args.candidate, write_index=args.write_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
