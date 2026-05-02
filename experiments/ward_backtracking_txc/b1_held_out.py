"""Exp 5 — held-out B1 prompt set.

Stage A's 300 prompts are split: 280 'dom' (used for DoM derivation
+ as the corpus for activation caching) and 20 'eval' (used for all
our B1 / metric / hill-climb work to date). We've been selecting
metrics + thresholds against the 20-eval-split prompts, so there's an
overfitting concern.

This script samples a NEW 20 prompts from the 280-dom-split (disjoint
from the eval set), re-runs B1 on the 5 headline cells, and reports
per-arch ordering on the held-out set vs the original 20-eval-split.
If TXC's lead inverts, we've been overfitting the metric.

Usage:
    python -m experiments.ward_backtracking_txc.b1_held_out
"""

from __future__ import annotations
import argparse, json, logging, sys, random
from pathlib import Path

import yaml
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.b1_held_out")

CELLS_TO_EVAL = [
    "txc__resid_L10__k16__s42",
    "txc_h13__resid_L10__k16__s42",
    "txc_h8__resid_L10__k16__s42",
    "topk_sae__ln1_L10__k64__s42",
    "stacked_sae__resid_L10__k16__s42",
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--n-held-out", type=int, default=20)
    p.add_argument("--seed", type=int, default=137)
    p.add_argument("--cells", nargs="+", default=CELLS_TO_EVAL)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())

    # Sample held-out prompts from Stage A's dom-split (280 prompts).
    all_prompts = json.loads(Path(cfg["paths"]["stageA_prompts"]).read_text())
    dom_pool = [p for p in all_prompts if p.get("split") == "dom"]
    rng = random.Random(args.seed)
    rng.shuffle(dom_pool)
    held_out = dom_pool[:args.n_held_out]
    log.info("[held-out] %d prompts sampled from dom-split (seed=%d)",
             len(held_out), args.seed)

    # Save the held-out prompt manifest so the per-cell B1 can reference it.
    held_out_path = Path(cfg["paths"]["root"]) / "b1_held_out" / "held_out_prompts.json"
    held_out_path.parent.mkdir(parents=True, exist_ok=True)
    held_out_path.write_text(json.dumps([{"id": p["id"], "category": p.get("category"),
                                           "prompt": p.get("question") or p.get("prompt"),
                                           "answer": p.get("answer")} for p in held_out], indent=2))
    log.info("[held-out] saved manifest %s", held_out_path)

    # Now run B1 on each cell. We'll reuse the existing b1_steer_eval but
    # need to override the eval-prompts source. Easiest: monkey-patch by
    # writing a temp prompts file and pointing config at it.
    import shutil
    held_out_full = held_out_path.parent / "held_out_full_prompts.json"
    # Build same shape as Stage A prompts.json: list of dicts with split, question, etc.
    full_format = []
    for p in held_out:
        p2 = dict(p); p2["split"] = "eval"   # mark as eval so existing _eval_prompts picks them up
        full_format.append(p2)
    held_out_full.write_text(json.dumps(full_format, indent=2))
    log.info("[held-out] saved override prompts %s", held_out_full)

    # For each cell, override config.paths.stageA_prompts to held_out_full,
    # run b1_steer_eval --cell with --no-dom (to skip DoM rerun), output to
    # b1_held_out/<cell>.json.
    import subprocess
    for cell in args.cells:
        out_path = Path(cfg["paths"]["root"]) / "b1_held_out" / f"b1__{cell}.json"
        if out_path.exists():
            log.info("[skip-b1] %s exists", out_path)
            continue
        # Write a temp config that overrides stageA_prompts
        tmp_cfg = held_out_path.parent / f"config_held_out.yaml"
        cfg_override = dict(cfg)
        cfg_override["paths"] = dict(cfg["paths"])
        cfg_override["paths"]["stageA_prompts"] = str(held_out_full)
        # Override the steering-output to land in our held-out dir
        cfg_override["paths"]["root"] = str(Path(cfg["paths"]["root"]) / "b1_held_out_run")
        Path(cfg_override["paths"]["root"]).mkdir(parents=True, exist_ok=True)
        cfg_override["paths"]["steering"] = str(Path(cfg_override["paths"]["root"]) / "b1_steering_results.json")
        tmp_cfg.write_text(yaml.safe_dump(cfg_override))
        cmd = [sys.executable, "-m", "experiments.ward_backtracking_txc.b1_steer_eval",
               "--config", str(tmp_cfg), "--cell", cell, "--no-dom"]
        log.info("[run] %s", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            log.error("[FAIL] %s rc=%d", cell, rc); continue
        # Move per-cell B1 from b1_held_out_run/steering_per_cell/b1__<cell>.json
        # to b1_held_out/<cell>.json
        produced = Path(cfg_override["paths"]["root"]) / "steering_per_cell" / f"b1__{cell}.json"
        if produced.exists():
            shutil.copy(produced, out_path)
            log.info("[ok] saved %s", out_path)
        else:
            log.warning("[warn] expected %s missing", produced)
    log.info("[held-out] all cells done. Results at %s/b1__<cell>.json",
             held_out_path.parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
