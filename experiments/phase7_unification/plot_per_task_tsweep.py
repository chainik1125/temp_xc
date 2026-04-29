"""Per-task T-sweep plots — barebones TXCDR (`txcdr_t<T>`) at every T,
36 small multiples (6×6 grid).

Also identifies tasks with the best / worst T-scaling — measured by
linear-regression slope of (T → mean_AUC) over T ∈ {3..32}.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.plot_per_task_tsweep
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.phase7_unification._paths import OUT_DIR, PLOTS_DIR


PROBING_PATH = OUT_DIR / "probing_results.jsonl"
SEEDS = (1, 42)
S_FILTER = 32
BAREBONES_RE = re.compile(r"^txcdr_t(\d+)$")
ALL_T = (3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32)


def save_figure(fig, path: str, dpi: int = 150, thumb_max_width: int = 288, thumb_dpi: int = 48):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    w_in, _ = fig.get_size_inches()
    thumb_dpi_eff = min(thumb_dpi, int(thumb_max_width / max(w_in, 0.1)))
    fig.savefig(p.with_suffix(".thumb.png"), dpi=thumb_dpi_eff, bbox_inches="tight")


def task_cluster(name: str) -> str:
    if name.startswith("bias_in_bios"): return "bias_in_bios"
    if name.startswith("ag_news"):       return "ag_news"
    if name.startswith("amazon_reviews_cat"): return "amazon_cat"
    if "amazon_reviews_sentiment" in name: return "amazon_sentiment"
    if name.startswith("europarl"):     return "europarl"
    if name.startswith("github_code"):  return "github_code"
    if name in ("winogrande_correct_completion", "wsc_coreference"):
        return "coreference"
    return "other"


CLUSTER_COLOR = {
    "bias_in_bios":     "#4472c4",
    "europarl":         "#70ad47",
    "amazon_cat":       "#ed7d31",
    "amazon_sentiment": "#e8a629",
    "ag_news":          "#a5a5a5",
    "github_code":      "#7030a0",
    "coreference":      "#c00000",
}


def load_per_task_per_T() -> dict:
    """Returns dict[k_feat][task][T] -> mean_AUC across seeds."""
    by_seed = defaultdict(dict)
    with PROBING_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("S") != S_FILTER or r.get("seed") not in SEEDS: continue
            if r.get("k_feat") not in (5, 20): continue
            if "skipped" in r: continue
            m = BAREBONES_RE.match(r.get("arch_id", ""))
            if not m: continue
            T = int(m.group(1))
            key = (r["k_feat"], r["task_name"], T, r["seed"])
            by_seed[key] = r.get("test_auc_flip", r["test_auc"])
    out = defaultdict(lambda: defaultdict(dict))
    accum = defaultdict(list)
    for (kf, task, T, seed), auc in by_seed.items():
        accum[(kf, task, T)].append(auc)
    for (kf, task, T), v in accum.items():
        out[kf][task][T] = float(np.mean(v))
    return out


def plot_grid(data, kf, out_path: Path):
    tasks = sorted({t for kf2, taskdict in data.items()
                    if kf2 == kf for t in taskdict})
    n = len(tasks)
    ncol = 6
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 3 * nrow), constrained_layout=True)
    axes = axes.flatten()

    for ax, task in zip(axes, tasks):
        cluster = task_cluster(task)
        color = CLUSTER_COLOR.get(cluster, "#888888")
        d = data[kf][task]
        Ts = sorted([T for T in d if T in ALL_T])
        ys = [d[T] for T in Ts]
        ax.plot(Ts, ys, marker="o", color=color, markersize=4, linewidth=1.2)
        # Linear regression slope for ranking
        if len(Ts) >= 3:
            slope, _ = np.polyfit(Ts, ys, 1)
            ax.text(0.02, 0.98, f"slope={slope:+.4f}/T", transform=ax.transAxes,
                    fontsize=7, va="top")
        # Best-T marker
        if Ts:
            best_idx = int(np.argmax(ys))
            ax.scatter([Ts[best_idx]], [ys[best_idx]], color=color, s=36,
                       edgecolor="black", zorder=10)
        ax.set_title(task, fontsize=8)
        ax.set_xticks([3, 5, 8, 14, 24, 32])
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlabel("T", fontsize=7)
        ax.set_ylabel(f"k_feat={kf} AUC", fontsize=7)

    # Hide unused panels
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Per-task T-sweep — barebones TXCDR (txcdr_t<T>), "
                 f"2-seed mean, k_feat={kf}\n"
                 f"colour = task cluster; filled dot = best-T per task",
                 fontsize=12, weight="bold")
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"  Wrote {out_path}")


def rank_by_t_scaling(data, kf):
    rows = []
    for task, d in data[kf].items():
        Ts = sorted([T for T in d if T in ALL_T])
        if len(Ts) < 5: continue
        ys = [d[T] for T in Ts]
        slope, intercept = np.polyfit(Ts, ys, 1)
        rng = max(ys) - min(ys)
        best_T = Ts[int(np.argmax(ys))]
        worst_T = Ts[int(np.argmin(ys))]
        rows.append({
            "task": task, "cluster": task_cluster(task),
            "slope": slope, "range": rng,
            "best_T": best_T, "best_auc": max(ys),
            "worst_T": worst_T, "worst_auc": min(ys),
            "mean_auc": float(np.mean(ys)),
        })
    return sorted(rows, key=lambda r: -r["slope"])


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_per_task_per_T()

    # Plots
    for kf in (5, 20):
        plot_grid(data, kf,
                  PLOTS_DIR / f"phase7_per_task_tsweep_k{kf}.png")

    # Rankings
    for kf in (5, 20):
        print()
        print("=" * 110)
        print(f"Per-task T-scaling (txcdr_t<T> family), 2-seed mean, k_feat={kf}")
        print("=" * 110)
        rows = rank_by_t_scaling(data, kf)
        print(f"\nTOP 8 — most positive T-slope (T helps):")
        print(f"  {'task':40s}  {'cluster':18s}  {'slope':>10s}  {'best_T':>6s}  {'best_AUC':>9s}  {'mean':>8s}")
        for r in rows[:8]:
            print(f"  {r['task']:40s}  {r['cluster']:18s}  {r['slope']:>+10.5f}  {r['best_T']:>6d}  {r['best_auc']:>9.4f}  {r['mean_auc']:>8.4f}")
        print(f"\nBOTTOM 8 — most negative T-slope (T hurts):")
        print(f"  {'task':40s}  {'cluster':18s}  {'slope':>10s}  {'best_T':>6s}  {'best_AUC':>9s}  {'mean':>8s}")
        for r in rows[-8:]:
            print(f"  {r['task']:40s}  {r['cluster']:18s}  {r['slope']:>+10.5f}  {r['best_T']:>6d}  {r['best_auc']:>9.4f}  {r['mean_auc']:>8.4f}")
        # Per-cluster mean slope
        print(f"\nPer-cluster mean T-slope:")
        c_sums = defaultdict(list)
        for r in rows: c_sums[r["cluster"]].append(r["slope"])
        for c in sorted(c_sums.keys(), key=lambda c: -np.mean(c_sums[c])):
            print(f"  {c:18s}  n={len(c_sums[c]):>2d}  mean_slope={np.mean(c_sums[c]):>+10.5f}")


if __name__ == "__main__":
    main()
