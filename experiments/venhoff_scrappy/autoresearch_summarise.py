"""Autoresearch Δ-reporter for venhoff_scrappy.

Reads a candidate's grade_results.json, looks up the declared baseline's
grade_results.json, computes Δ Gap Recovery, emits a verdict, and
appends a row to results/autoresearch_index.jsonl.

Pattern adopted from Han's experiments/phase5_downstream_utility/autoresearch_summarise.py.

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
    return "AMBIGUOUS"


def summarise(candidate: str, write_index: bool = False) -> dict:
    cand_cfg = load_candidate_cfg(candidate)
    defaults = yaml.safe_load(DEFAULTS_YAML.read_text())
    baseline = cand_cfg.get("baseline", defaults["autoresearch"]["default_baseline"])

    cand_grade = load_grade(candidate)
    if cand_grade is None:
        raise FileNotFoundError(f"no grade_results.json for candidate={candidate}")

    base_grade = load_grade(baseline) if baseline != candidate else cand_grade
    if base_grade is None:
        raise FileNotFoundError(
            f"baseline {baseline!r} has no grade_results.json — run it first"
        )

    cand_gr = cand_grade.get("gap_recovery")
    base_gr = base_grade.get("gap_recovery")

    # Scaffold grade files emit None; propagate that upstream.
    if cand_gr is None or base_gr is None:
        delta_pp = None
        verdict = "INSUFFICIENT"
    else:
        delta_pp = 100.0 * (cand_gr - base_gr)
        verdict = verdict_for(delta_pp, defaults)

    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": candidate,
        "baseline": baseline,
        "arch_cand": cand_grade.get("arch"),
        "arch_base": base_grade.get("arch"),
        "n_tasks": cand_grade.get("n_tasks"),
        "gap_recovery_cand": cand_gr,
        "gap_recovery_base": base_gr,
        "delta_pp": delta_pp,
        "verdict": verdict,
        "wall_time_s": cand_grade.get("wall_time_s"),
        "scaffold": bool(cand_grade.get("scaffold")),
    }

    print(json.dumps(row, indent=2))

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
