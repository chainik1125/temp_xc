"""Phase 0 bootstrap — one-time trace + activation collection for scrappy.

Reads scrappy config.yaml for model/dataset/n_traces/layer/seed, invokes
the same generate_traces + activation_collection modules the paper-budget
pipeline uses, and writes the artifacts under:

    experiments/venhoff_scrappy/results/phase0/<identity_slug>/

where <identity_slug> matches ArtifactPaths so cycles can symlink the
directory in and Phase 2 + 3 find everything at the expected ArtifactPaths
locations.

Subsequent `run_cycle.py` invocations symlink this directory rather than
regenerate. For candidates that change `steering_layer`, re-run this
script with --layer override.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SCRAPPY = REPO / "experiments/venhoff_scrappy"
CONFIG_YAML = SCRAPPY / "config.yaml"
PHASE0_ROOT = SCRAPPY / "results/phase0"


def _run(cmd: list[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=None,
                    help="Override config steering_layer (e.g. to cache for L8/L16 candidates).")
    ap.add_argument("--n-traces", type=int, default=None,
                    help="Override config n_traces (default reads from config.yaml).")
    ap.add_argument("--force", action="store_true", help="Re-run even if artifacts exist.")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    # generate_traces.py takes the short slug (e.g. 'deepseek-r1-distill-llama-8b')
    # which it resolves via model_registry.py; full HF path is used
    # elsewhere (run_steering/run_hybrid).
    thinking_full = cfg["model"]["thinking"]
    thinking_slug = thinking_full.split("/")[-1].lower()
    dataset = cfg["phase3_hybrid"]["dataset"]
    split = cfg["phase3_hybrid"]["dataset_split"]
    seed = cfg["autoresearch"]["seed"]
    layer = args.layer if args.layer is not None else cfg["model"]["steering_layer"]
    n_traces = args.n_traces if args.n_traces is not None else cfg["phase0_activations"]["n_traces"]

    PHASE0_ROOT.mkdir(parents=True, exist_ok=True)
    main_python = str((REPO / ".venv" / "bin" / "python").absolute())

    # Phase 0a — generate traces.
    traces_cmd = [
        main_python, "-m", "src.bench.venhoff.generate_traces",
        "--root", str(PHASE0_ROOT),
        "--model", thinking_slug,
        "--dataset", dataset,
        "--split", split,
        "--n-traces", str(n_traces),
        "--layer", str(layer),
        "--seed", str(seed),
    ]
    if args.force:
        traces_cmd.append("--force")
    _run(traces_cmd)

    # Phase 0b — activation collection.
    act_cmd = [
        main_python, "-m", "src.bench.venhoff.activation_collection",
        "--root", str(PHASE0_ROOT),
        "--model", thinking_slug,
        "--dataset", dataset,
        "--split", split,
        "--n-traces", str(n_traces),
        "--layer", str(layer),
        "--seed", str(seed),
    ]
    if args.force:
        act_cmd.append("--force")
    _run(act_cmd)

    print(f"[ok] Phase 0 complete at {PHASE0_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
