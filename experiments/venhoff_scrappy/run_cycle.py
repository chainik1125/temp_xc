"""Run a single venhoff_scrappy cycle: Phase 0 (cached) → 2 → 3 → grade.

Reads the scrappy defaults from config.yaml and merges per-candidate
overrides from candidates/<name>.yaml. Dispatches to the existing
src/bench/venhoff/ entry points with the scrappy budget, writes the
grade output to results/cycles/<name>/grade_results.json.

The autoresearch_summarise.py step (run separately by the orchestrator)
reads that json and computes Δ vs baseline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SCRAPPY = REPO / "experiments/venhoff_scrappy"
SCRAPPY_DEFAULTS = SCRAPPY / "config.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_candidate(cand_cfg_path: Path) -> dict:
    base = yaml.safe_load(SCRAPPY_DEFAULTS.read_text())
    override = yaml.safe_load(cand_cfg_path.read_text())
    merged = _deep_merge(base, override)
    # Stamp identity fields from the candidate so they're non-overridable.
    for k in ("name", "arch", "baseline", "hypothesis"):
        if k in override:
            merged[k] = override[k]
    return merged


def run_cycle(cfg: dict, candidate: str, result_dir: Path) -> dict:
    """Run the scrappy Phase 0→3 pipeline + grading.

    Returns a dict: {thinking_acc, base_acc, hybrid_acc, gap_recovery,
    n_tasks, wall_time_s}.

    NOTE: this is a scaffold. The actual dispatch to src/bench/venhoff/
    entry points (run_activations.py, run_steering.py, run_hybrid.py,
    grade.py) is TODO — wired up when the first real scrappy cycle runs.
    Until then, this emits a placeholder grade_results.json so the
    ledger + summariser pipeline can be smoke-tested end-to-end.
    """
    t0 = time.time()
    result_dir.mkdir(parents=True, exist_ok=True)

    # TODO: replace this placeholder block with real dispatch:
    #   1. Phase 0: reuse cached activations at configured path
    #   2. Phase 2: subprocess.run([.venv/bin/python, -m, src.bench.venhoff.run_steering,
    #                               --arch, cfg["arch"], --max-iters, cfg["phase2_steering"]["max_iters"],
    #                               --n-train, cfg["phase2_steering"]["n_training_examples"],
    #                               --reuse-venhoff-vectors (if cfg["reuse_venhoff_vectors"])])
    #   3. Phase 3: subprocess.run([... run_hybrid with --n-tasks, --coefficients, --token-windows])
    #   4. Grade: subprocess.run([... grade.py --judge-model, ...])
    #   5. Parse the resulting grade output into the dict below.
    print(f"[scaffold] would run Phase 0→3 for candidate={candidate}")
    print(f"[scaffold] merged config: {json.dumps(cfg, indent=2)}")

    placeholder = {
        "candidate": candidate,
        "arch": cfg.get("arch", "unknown"),
        "n_tasks": cfg["phase3_hybrid"]["n_tasks"],
        "thinking_acc": None,
        "base_acc": None,
        "hybrid_acc": None,
        "gap_recovery": None,
        "wall_time_s": time.time() - t0,
        "scaffold": True,
        "note": "run_cycle.py is a scaffold; real dispatch TBD on first use",
    }
    (result_dir / "grade_results.json").write_text(json.dumps(placeholder, indent=2))
    (result_dir / "merged_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return placeholder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--result-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_candidate(args.config)
    run_cycle(cfg, args.candidate, args.result_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
