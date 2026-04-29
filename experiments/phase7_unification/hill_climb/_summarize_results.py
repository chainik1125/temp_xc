"""Aggregate hill-climb probing results vs Phase 7 leaderboard.

Reads `probing_results.jsonl`, computes mean AUC per (arch_id, k_feat) at
seed=42 base side, and prints a markdown table comparing each
hill-climb cell to the current leaderboard #1 at k_feat=5 and k_feat=20.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.hill_climb._summarize_results
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from experiments.phase7_unification._paths import PROBING_PATH


LEADERBOARD_K5 = ("phase57_partB_h8_bare_multidistance_t8", 0.8989)
LEADERBOARD_K20 = ("txc_bare_antidead_t5", 0.9358)


def load(seed: int = 42, S: int = 32):
    """Return dict[(arch_id, k_feat)] -> list[test_auc_flip]."""
    agg = defaultdict(list)
    with PROBING_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("seed") != seed:
                continue
            if r.get("S") != S:
                continue
            if r.get("skipped"):
                continue
            agg[(r["arch_id"], r["k_feat"])].append(r["test_auc_flip"])
    return agg


def verdict(auc: float, target: float) -> str:
    delta = auc - target
    if delta > 0.001:
        return f"WIN  (Δ=+{delta:.4f})"
    if delta < -0.001:
        return f"LOSE (Δ={delta:.4f})"
    return f"NEUT (Δ={delta:+.4f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--S", type=int, default=32)
    ap.add_argument("--prefix", default="hill_",
                    help="arch_id prefix to highlight (default 'hill_')")
    args = ap.parse_args()
    agg = load(seed=args.seed, S=args.S)

    # Headline: hill-climb cells vs leaderboard
    hill_cells = sorted({k for k in agg if k[0].startswith(args.prefix)})
    print(f"\n=== hill-climb cells (seed={args.seed} S={args.S}) ===")
    print(f"{'arch_id':50s} {'k':>3s} {'mean':>8s} {'n':>4s} {'verdict':>22s}")
    for arch_id, k_feat in hill_cells:
        target = LEADERBOARD_K5[1] if k_feat == 5 else (
            LEADERBOARD_K20[1] if k_feat == 20 else None)
        v = agg[(arch_id, k_feat)]
        m = sum(v) / len(v) if v else 0.0
        msg = verdict(m, target) if target is not None else ""
        print(f"{arch_id:50s} {k_feat:>3d} {m:>8.4f} {len(v):>4d} {msg:>22s}")

    # Top 5 (overall) at k_feat=5 and k_feat=20 for context
    for k in (5, 20):
        ranked = sorted(
            [(a, sum(v) / len(v), len(v)) for (a, kf), v in agg.items() if kf == k],
            key=lambda x: -x[1],
        )[:8]
        print(f"\n=== top 8 archs at k_feat={k} (seed={args.seed} S={args.S}) ===")
        print(f"{'arch_id':50s} {'mean':>8s} {'n':>4s}")
        for a, m, n in ranked:
            marker = " *" if a.startswith(args.prefix) else ""
            print(f"{a:50s} {m:>8.4f} {n:>4d}{marker}")


if __name__ == "__main__":
    main()
