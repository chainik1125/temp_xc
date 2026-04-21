"""Part B finalization: pick best config per finalist + run test-set eval.

Runs in two phases:

1. **Auto-pick best config** per finalist by reading
   `results/partB_summary.json` (produced by `partB_summarise.py`).
   The best config is the variant with the highest
   `mean_val_AUC` inside each family (A2, A3). Ties are broken by
   smaller α first (more conservative contrastive weight).

2. **Test-set eval** at `last_position` and `mean_pool` aggregations
   on the best-config ckpt. These aggregations fit the probe on the
   FULL train set (3040) and evaluate on the held-out test (760).
   This is the first and only time the test split gets touched for
   Phase 5.7.

Usage:
    # Dry-run: print which config would be picked, don't probe.
    .venv/bin/python -m experiments.phase5_downstream_utility.partB_finalize

    # Run probing at last_position + mean_pool on the test set.
    .venv/bin/python -m experiments.phase5_downstream_utility.partB_finalize --run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO = Path("/workspace/temp_xc")
RESULTS = REPO / "experiments/phase5_downstream_utility/results"
SUMMARY_JSON = RESULTS / "partB_summary.json"
CKPT_DIR = RESULTS / "ckpts"


def _pick_best(family_data):
    """Return the variant with highest mean_val_AUC. None if no data."""
    variants = family_data.get("variants", [])
    available = [v for v in variants if v.get("mean") is not None]
    if not available:
        return None
    # Sort by (−mean, alpha). Descending mean, then smaller alpha first
    # (more conservative regularisation) on ties.
    available.sort(key=lambda v: (-v["mean"], v.get("alpha", 0.1)))
    return available[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="Execute probing at last_position + mean_pool "
                         "on the chosen ckpts. Without this flag, prints "
                         "the chosen configs only (dry-run).")
    ap.add_argument("--aggs", nargs="+",
                    default=["last_position", "mean_pool"],
                    help="Aggregations to probe. Default: both.")
    args = ap.parse_args()

    if not SUMMARY_JSON.exists():
        print(f"FATAL: {SUMMARY_JSON} missing. Run partB_summarise.py first.")
        return

    summary = json.loads(SUMMARY_JSON.read_text())
    picks = {}
    print("Part B finalization — auto-pick best config per finalist:")
    print()
    for fam_name, fam in summary["families"].items():
        best = _pick_best(fam)
        if best is None:
            print(f"  {fam_name}: NO DATA yet, skipping.")
            continue
        picks[fam_name] = best
        print(f"  {fam_name} best = {best['arch']}")
        print(f"      config = {best['label']}")
        print(f"      mean_val_AUC = {best['mean']:.4f}")
        print(f"      Δ vs vanilla {fam['vanilla_base']} = "
              f"{best['vs_vanilla']['mean']:+.4f} ±"
              f"{best['vs_vanilla']['stderr']:.4f}")
        if best["arch"] != fam["ref"]:
            d = best.get("vs_ref")
            if d is not None:
                print(f"      Δ vs α=0.1 ref = {d['mean']:+.4f} ±"
                      f"{d['stderr']:.4f}")
        print()

    if not args.run:
        print("(Dry-run. Pass --run to execute test-set probing.)")
        return

    # Verify checkpoints exist for the picks
    missing = [p["arch"] for p in picks.values()
               if not (CKPT_DIR / f"{p['arch']}__seed42.pt").exists()]
    if missing:
        print(f"FATAL: missing ckpts: {missing}")
        return

    # Kick off probing for each pick at each aggregation. This is the
    # FIRST time the test split is read for Phase 5.7 — log it
    # prominently.
    print("─" * 60)
    print("NOTICE: about to probe the TEST split (held-out since Phase 5.7 began).")
    print("This writes rows tagged aggregation=last_position or mean_pool")
    print("into probing_results.jsonl, next to the existing 25-arch bench.")
    print("─" * 60)
    print()

    run_ids = [f"{p['arch']}__seed42" for p in picks.values()]
    for agg in args.aggs:
        print(f"### probing {run_ids} @ aggregation={agg}")
        cmd = [
            str(REPO / ".venv/bin/python"), "-u",
            str(REPO / "experiments/phase5_downstream_utility/probing/run_probing.py"),
            "--aggregation", agg, "--skip-baselines",
            "--run-ids", *run_ids,
        ]
        print(f"  $ {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=REPO)

    print()
    print("Test-set probing complete. Next steps:")
    print("  - re-run partB_summarise.py to regenerate the table")
    print("  - insert a 'Tuned leaders' section into summary.md")


if __name__ == "__main__":
    main()
