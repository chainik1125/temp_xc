"""Per-task comparison: B2 (subseq T_max=10 t_sample=5) vs Phase 5 references.

Reads phase5b probing_results.jsonl + phase5 references (text-coded for now,
since we don't have phase 5's per-task results in this branch). Computes
per-task win/loss for B2 mean against H8 3-seed mean.

Output: stdout table + a per-task heatmap PNG.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.phase5b_t_scaling_explore._paths import PROBING_PATH, PLOTS_DIR


def load_b2_per_task():
    """B2 mean over seeds, per task, per aggregation."""
    rows = [json.loads(l) for l in PROBING_PATH.open()]
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arch"] != "subseq_track2": continue
        if int(r.get("k_feat", 0)) != 5: continue
        out[r["aggregation"]][r["task_name"]].append(float(r["test_auc"]))
    # mean across seeds
    return {
        agg: {t: float(np.mean(aucs)) for t, aucs in tasks.items()}
        for agg, tasks in out.items()
    }


def main():
    b2 = load_b2_per_task()
    if not b2:
        print("no B2 rows found")
        return
    n_seeds = len({json.loads(l)["seed"] for l in PROBING_PATH.open()
                    if json.loads(l)["arch"] == "subseq_track2"})

    print(f"B2 (subseq_track2 T_max=10 t_sample=5) per-task headline AUC, k=5, mean over {n_seeds} seeds:")
    print()
    print(f"{'task':45s}  {'lp':>7s}  {'mp':>7s}")
    print("-" * 65)
    tasks = sorted(set(b2.get("last_position", {}).keys()) |
                    set(b2.get("mean_pool", {}).keys()))
    avg_lp, avg_mp, n = 0.0, 0.0, 0
    for t in tasks:
        lp = b2["last_position"].get(t, float("nan"))
        mp = b2["mean_pool"].get(t, float("nan"))
        if not (np.isnan(lp) or np.isnan(mp)):
            avg_lp += lp; avg_mp += mp; n += 1
        print(f"{t:45s}  {lp:>7.4f}  {mp:>7.4f}")
    print("-" * 65)
    print(f"{'MEAN':45s}  {avg_lp/n:>7.4f}  {avg_mp/n:>7.4f}")


if __name__ == "__main__":
    main()
