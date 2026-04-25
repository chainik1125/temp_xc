"""Tier-1 results aggregator + headline plot.

Reads phase5b probing_results.jsonl, computes per-(run_id, aggregation,
k_feat) mean test_auc across the 36 tasks, prints a sorted table, and
saves a paired bar plot (lp vs mp) at d_sae=18432, k=5.

Compares against the canonical Phase 5 references:
    H8 3-seed         lp 0.8005 ± 0.0030, mp 0.8126 ± 0.0030
    H8 T=6 seed=42    lp ?      , mp 0.8188
    agentic_txc_02 3s lp 0.7749 ± 0.0038, mp 0.7987 ± 0.0020
    txcdr_t5          lp 0.7829, mp 0.8064
    vanilla T=10      lp 0.7671, mp 0.7754

Run from repo root:
    .venv/bin/python -m experiments.phase5b_t_scaling_explore.analyze_tier1
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.phase5b_t_scaling_explore._paths import (
    PROBING_PATH, PLOTS_DIR,
)


PHASE5_REFS = {
    "H8 3-seed @ T=5":        {"lp": 0.8005, "mp": 0.8126, "lp_se": 0.003, "mp_se": 0.003},
    "H8 seed=42 @ T=6":       {"lp": None,   "mp": 0.8188, "lp_se": None,  "mp_se": None},
    "H8 seed=42 @ T=10":      {"lp": 0.7931, "mp": 0.8040, "lp_se": None,  "mp_se": None},
    "agentic_txc_02 3-seed":  {"lp": 0.7749, "mp": 0.7987, "lp_se": 0.004, "mp_se": 0.002},
    "txcdr_t5 seed=42":       {"lp": 0.7829, "mp": 0.8064, "lp_se": None,  "mp_se": None},
    "txcdr_t10 seed=42":      {"lp": 0.7671, "mp": 0.7754, "lp_se": None,  "mp_se": None},
}


def load_rows(path: Path = PROBING_PATH):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows, k_feat: int = 5):
    """Group by (run_id, arch, aggregation); compute mean+stderr over tasks."""
    buckets = defaultdict(list)
    meta_by_run = {}
    for r in rows:
        if int(r.get("k_feat", 0)) != k_feat:
            continue
        key = (r["run_id"], r["arch"], r["aggregation"])
        buckets[key].append(float(r["test_auc"]))
        meta_by_run.setdefault(r["run_id"], r)
    out = []
    for (run_id, arch, agg), aucs in buckets.items():
        a = np.asarray(aucs)
        out.append({
            "run_id": run_id, "arch": arch, "aggregation": agg,
            "mean_auc": float(a.mean()),
            "std_auc":  float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "se_auc":   float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
            "n_tasks":  int(a.size),
            "meta":     meta_by_run.get(run_id, {}),
        })
    return out


def print_table(agg, baselines: dict | None = None):
    baselines = baselines or PHASE5_REFS
    by_run = defaultdict(dict)
    for row in agg:
        by_run[row["run_id"]][row["aggregation"]] = row

    # Sort by max(lp, mp) descending
    def best_auc(d):
        return max((d.get(a, {}).get("mean_auc", 0.0) for a in ("last_position", "mean_pool")),
                    default=0.0)

    runs_sorted = sorted(by_run.items(), key=lambda kv: -best_auc(kv[1]))

    print()
    print(f"{'run_id':50s}  {'arch':22s}  {'lp_auc':>8s}  {'mp_auc':>8s}  n_tasks")
    print("-" * 110)
    for run_id, by_agg in runs_sorted:
        lp = by_agg.get("last_position", {})
        mp = by_agg.get("mean_pool", {})
        print(
            f"{run_id:50s}  {(lp.get('arch') or mp.get('arch') or '?'):22s}  "
            f"{lp.get('mean_auc', float('nan')):>8.4f}  "
            f"{mp.get('mean_auc', float('nan')):>8.4f}  "
            f"{lp.get('n_tasks') or mp.get('n_tasks') or 0}"
        )
    print()
    print(f"--- Phase 5 reference points (k=5, 36 tasks) ---")
    for name, d in baselines.items():
        lp = "{:.4f}".format(d["lp"]) if d["lp"] is not None else "    -   "
        mp = "{:.4f}".format(d["mp"]) if d["mp"] is not None else "    -   "
        print(f"{name:50s}  {'reference':22s}  {lp:>8s}  {mp:>8s}")


def plot_headline(agg, save_dir: Path = PLOTS_DIR):
    import matplotlib.pyplot as plt
    save_dir.mkdir(parents=True, exist_ok=True)

    by_run = defaultdict(dict)
    for row in agg:
        by_run[row["run_id"]][row["aggregation"]] = row

    rows = sorted(
        by_run.items(),
        key=lambda kv: -max(
            kv[1].get("last_position", {}).get("mean_auc", 0.0),
            kv[1].get("mean_pool", {}).get("mean_auc", 0.0),
        ),
    )
    labels = [r[0].replace("phase5b_", "").replace("__seed42", "") for r in rows]
    lp = [r[1].get("last_position", {}).get("mean_auc", float("nan")) for r in rows]
    mp = [r[1].get("mean_pool", {}).get("mean_auc", float("nan")) for r in rows]

    # Append references
    for name, d in PHASE5_REFS.items():
        labels.append(f"[ref] {name}")
        lp.append(d["lp"] if d["lp"] is not None else float("nan"))
        mp.append(d["mp"] if d["mp"] is not None else float("nan"))

    fig, ax = plt.subplots(figsize=(11, 0.36 * len(labels) + 1.5))
    y = np.arange(len(labels))
    h = 0.4
    ax.barh(y - h / 2, lp, h, label="last_position", color="#4477AA")
    ax.barh(y + h / 2, mp, h, label="mean_pool",     color="#EE6677")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUC (k=5, 36 tasks, seed=42)")
    ax.axvline(0.8126, color="#EE6677", linestyle="--", alpha=0.5,
                label="H8 3-seed mp baseline")
    ax.axvline(0.8005, color="#4477AA", linestyle="--", alpha=0.5,
                label="H8 3-seed lp baseline")
    ax.set_xlim(0.5, 0.86)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Phase 5B tier-1 headline AUC vs Phase 5 references")
    fig.tight_layout()

    png_path = save_dir / "phase5b_tier1_headline_bar_lp_mp.png"
    fig.savefig(png_path, dpi=150)

    # Thumbnail
    thumb_path = save_dir / "phase5b_tier1_headline_bar_lp_mp.thumb.png"
    fig.savefig(thumb_path, dpi=48,
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"saved: {png_path}")
    print(f"saved: {thumb_path}")


def main():
    rows = load_rows()
    if not rows:
        print(f"no rows in {PROBING_PATH}")
        return
    agg = aggregate(rows, k_feat=5)
    print_table(agg)
    try:
        plot_headline(agg)
    except Exception as e:
        print(f"[plot] skipped: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
