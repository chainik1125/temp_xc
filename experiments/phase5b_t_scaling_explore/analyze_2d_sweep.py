"""2D sweep analyzer: (T_max, t_sample) → AUC heatmap for subseq_track2.

Aggregates per (T_max, t_sample) the seed=42 mean test_auc at k=5 across
the 36-task probe set, both aggregations. Includes B2 3-seed (T=10, s=5)
result for the reference cell.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.phase5b_t_scaling_explore._paths import PROBING_PATH, PLOTS_DIR


def load_grid(arch_filter: str = "subseq_h8", k_win_filter: int = 500):
    """Load (T_max, t_sample, agg) cells for the 2D sweep.

    Filters to a single arch + k_win to keep cells comparable.
    Defaults to subseq_h8 with k_win=500 (matches the run_b4_2d_sweep.sh
    convention that fixes k for clean t_sample interpretation).
    """
    rows = [json.loads(l) for l in PROBING_PATH.open()]
    grid = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arch"] != arch_filter: continue
        if int(r.get("k_feat", 0)) != 5: continue
        if k_win_filter is not None and int(r.get("k_win", 0)) != k_win_filter:
            continue
        T_max = int(r.get("T_max", 10))
        t_sample = int(r.get("t_sample", 5))
        seed = int(r.get("seed", 42))
        # 2D grid is seed=42-only; aggregate (mean) across seeds if available.
        key = (T_max, t_sample, r["aggregation"])
        grid[key]["aucs"].append(float(r["test_auc"]))
        grid[key]["seeds"].append(seed)
    return grid


def aggregate_grid(grid):
    out = {}
    for key, data in grid.items():
        a = np.asarray(data["aucs"])
        out[key] = {"mean": float(a.mean()), "n": int(a.size)}
    return out


def print_table(table):
    Ts = sorted(set(k[0] for k in table.keys()))
    print()
    print(f"2D sweep: subseq_track2 seed=42 mean AUC k=5 (36 tasks)")
    print()
    for agg in ("last_position", "mean_pool"):
        print(f"--- {agg} ---")
        Ss = sorted(set(k[1] for k in table.keys() if k[2] == agg))
        header = "T_max\\t_sample".ljust(16) + " ".join(f"{s:>8}" for s in Ss)
        print(header)
        for T in Ts:
            row = f"{T:>16}" + " ".join(
                f"{table[(T, s, agg)]['mean']:>8.4f}"
                if (T, s, agg) in table else "    -   "
                for s in Ss
            )
            print(row)
        print()


def plot_heatmap(table):
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    Ts = sorted(set(k[0] for k in table.keys()))
    Ss = sorted(set(k[1] for k in table.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3 + 0.4 * len(Ts)))
    for ax, agg in zip(axes, ("last_position", "mean_pool")):
        M = np.full((len(Ts), len(Ss)), np.nan)
        for i, T in enumerate(Ts):
            for j, s in enumerate(Ss):
                if (T, s, agg) in table:
                    M[i, j] = table[(T, s, agg)]["mean"]
        im = ax.imshow(M, vmin=0.78, vmax=0.83, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(np.arange(len(Ss))); ax.set_xticklabels(Ss)
        ax.set_yticks(np.arange(len(Ts))); ax.set_yticklabels(Ts)
        ax.set_xlabel("t_sample"); ax.set_ylabel("T_max")
        ax.set_title(f"{agg} AUC")
        for i in range(len(Ts)):
            for j in range(len(Ss)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Phase 5B 2D sweep: subseq_track2 (T_max, t_sample) seed=42, k=5")
    fig.tight_layout()
    p = PLOTS_DIR / "phase5b_2d_sweep_heatmap.png"
    fig.savefig(p, dpi=150)
    fig.savefig(PLOTS_DIR / "phase5b_2d_sweep_heatmap.thumb.png", dpi=48,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"saved: {p}")


def main():
    grid = load_grid()
    if not grid:
        print(f"no rows in {PROBING_PATH}")
        return
    table = aggregate_grid(grid)
    print_table(table)
    try:
        plot_heatmap(table)
    except Exception as e:
        print(f"[plot] skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
