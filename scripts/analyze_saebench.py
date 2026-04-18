"""Aggregate + plot SAEBench sparse-probing JSONL records.

Emits:
  - Console tables: headline, aggregation ablation, T-sweep, per-task
  - Plots in results/saebench/plots/

Run: python scripts/analyze_saebench.py
"""

from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = Path("results/saebench/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ARCH_COLOR = {"sae": "#1f77b4", "mlc": "#ff7f0e", "tempxc": "#2ca02c"}
AGG_ORDER = ["last", "mean", "max", "full_window"]


def load_records():
    records = []
    for p in sorted(glob.glob("results/saebench/results/*.jsonl")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    # Dedupe SAE identical-aggregation duplicates: keep one record per
    # (arch, protocol, T, aggregation, task, k).
    seen = {}
    for r in records:
        key = (r["architecture"], r["matching_protocol"], r["t"],
               r["aggregation"], r["task"], r["k"])
        # Take mean if multiple — should be identical for SAE anyway
        seen.setdefault(key, []).append(r["accuracy"])
    deduped = []
    for (arch, proto, t, agg, task, k), accs in seen.items():
        deduped.append({
            "architecture": arch, "matching_protocol": proto, "t": t,
            "aggregation": agg, "task": task, "k": k,
            "accuracy": statistics.mean(accs),
            "n_dupes": len(accs),
        })
    return deduped


def table_headline(records):
    """Protocol A × full_window × k=5, averaged across tasks."""
    by = defaultdict(list)
    for r in records:
        if (r["matching_protocol"] == "A"
                and r["aggregation"] == "full_window"
                and r["k"] == 5):
            by[(r["architecture"], r["t"])].append(r["accuracy"])
    print("\n=== HEADLINE: protocol A × full_window × k=5 (averaged over 8 tasks) ===")
    print(f"{'arch':<8} {'T':<4} {'mean_acc':<10} {'stddev':<10} {'n_tasks':<8}")
    rows = []
    for (arch, t), accs in sorted(by.items()):
        m = statistics.mean(accs)
        s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"{arch:<8} {t:<4} {m:<10.4f} {s:<10.4f} {len(accs):<8}")
        rows.append({"arch": arch, "t": t, "mean": m, "std": s, "n": len(accs)})
    return rows


def table_aggregation_ablation(records):
    """T=5, protocol A, k=5 — how much aggregation matters."""
    by = defaultdict(list)
    for r in records:
        if (r["matching_protocol"] == "A"
                and r["t"] == 5
                and r["k"] == 5):
            by[(r["architecture"], r["aggregation"])].append(r["accuracy"])
    print("\n=== AGGREGATION ABLATION: T=5, protocol A, k=5 ===")
    print(f"{'arch':<8} {'agg':<14} {'mean_acc':<10}")
    rows = []
    for (arch, agg), accs in sorted(by.items(), key=lambda kv: (kv[0][0], AGG_ORDER.index(kv[0][1]))):
        m = statistics.mean(accs)
        print(f"{arch:<8} {agg:<14} {m:<10.4f}")
        rows.append({"arch": arch, "agg": agg, "mean": m})
    return rows


def table_tsweep(records, protocol="A"):
    """TempXC T-sweep at protocol × full_window × k=5."""
    by = defaultdict(list)
    for r in records:
        if (r["architecture"] == "tempxc"
                and r["matching_protocol"] == protocol
                and r["aggregation"] == "full_window"
                and r["k"] == 5):
            by[r["t"]].append(r["accuracy"])
    print(f"\n=== T-SWEEP: TempXC, protocol {protocol}, full_window, k=5 ===")
    print(f"{'T':<4} {'mean_acc':<10} {'n_tasks':<8}")
    rows = []
    for t, accs in sorted(by.items()):
        m = statistics.mean(accs)
        print(f"{t:<4} {m:<10.4f} {len(accs):<8}")
        rows.append({"t": t, "mean": m, "n": len(accs)})
    return rows


def _normalize_task(task_name):
    """SAE/TempXC (SAEBench stock) append '_results'; MLC (our fork) doesn't."""
    return task_name.replace("_results", "").split("/")[-1]


def table_per_task(records):
    """Per-task accuracy across architectures at T=5 × protA × full_window × k=5."""
    by = defaultdict(dict)
    for r in records:
        if (r["t"] == 5
                and r["matching_protocol"] == "A"
                and r["aggregation"] == "full_window"
                and r["k"] == 5):
            by[_normalize_task(r["task"])][r["architecture"]] = r["accuracy"]
    print("\n=== PER-TASK (T=5, protA, full_window, k=5) ===")
    print(f"{'task':<50} {'sae':<8} {'mlc':<8} {'tempxc':<8}")
    rows = []
    for task in sorted(by.keys()):
        row = by[task]
        sae = row.get("sae", float("nan"))
        mlc = row.get("mlc", float("nan"))
        txc = row.get("tempxc", float("nan"))
        print(f"{task:<50} {sae:<8.4f} {mlc:<8.4f} {txc:<8.4f}")
        rows.append({"task": task, "sae": sae, "mlc": mlc, "tempxc": txc})
    return rows


def plot_headline(headline_rows):
    """Bar chart: SAE / MLC / TempXC at T=5 × full_window × k=5 × protA."""
    at_t5 = {r["arch"]: r for r in headline_rows if r["t"] == 5}
    archs = ["sae", "mlc", "tempxc"]
    means = [at_t5[a]["mean"] for a in archs]
    stds = [at_t5[a]["std"] for a in archs]
    colors = [ARCH_COLOR[a] for a in archs]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.arange(len(archs))
    bars = ax.bar(xs, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["SAE", "MLC", "TempXC"])
    ax.set_ylabel("Sparse-probing accuracy (k=5, averaged over 8 tasks)")
    ax.set_title("Headline: architecture comparison at T=5\nprotocol A × full_window × k=5")
    ax.set_ylim(0.75, 1.0)
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    path = PLOTS_DIR / "fig1_headline_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_aggregation_ablation(records):
    """Grouped bar chart: aggregation × architecture at T=5 × protA × k=5."""
    archs = ["sae", "mlc", "tempxc"]
    aggs = AGG_ORDER
    data = {a: {} for a in archs}
    for r in records:
        if (r["matching_protocol"] == "A"
                and r["t"] == 5
                and r["k"] == 5):
            data[r["architecture"]].setdefault(r["aggregation"], []).append(r["accuracy"])
    means = {a: [statistics.mean(data[a][agg]) for agg in aggs] for a in archs}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(aggs))
    w = 0.27
    for i, a in enumerate(archs):
        ax.bar(x + (i - 1) * w, means[a], w, label=a.upper(),
               color=ARCH_COLOR[a], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(aggs)
    ax.set_ylabel("Sparse-probing accuracy (k=5, mean over 8 tasks)")
    ax.set_title("Aggregation ablation at T=5 × protocol A × k=5")
    ax.set_ylim(0.6, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = PLOTS_DIR / "fig2_aggregation_ablation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_tsweep(records):
    """TempXC T-sweep: protA (scales with T) vs protB (fixed window budget)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for protocol, style in [("A", "-o"), ("B", "--s")]:
        by = defaultdict(list)
        for r in records:
            if (r["architecture"] == "tempxc"
                    and r["matching_protocol"] == protocol
                    and r["aggregation"] == "full_window"
                    and r["k"] == 5):
                by[r["t"]].append(r["accuracy"])
        ts = sorted(by.keys())
        means = [statistics.mean(by[t]) for t in ts]
        stds = [statistics.stdev(by[t]) if len(by[t]) > 1 else 0 for t in ts]
        ax.errorbar(ts, means, yerr=stds, fmt=style, capsize=4,
                    label=f"TempXC (protocol {protocol})", linewidth=2, markersize=8)

    # SAE + MLC baselines at T=5 for reference
    sae_baseline = []
    mlc_baseline = []
    for r in records:
        if (r["matching_protocol"] == "A" and r["aggregation"] == "full_window"
                and r["k"] == 5 and r["t"] == 5):
            if r["architecture"] == "sae":
                sae_baseline.append(r["accuracy"])
            elif r["architecture"] == "mlc":
                mlc_baseline.append(r["accuracy"])
    if sae_baseline:
        ax.axhline(statistics.mean(sae_baseline), color=ARCH_COLOR["sae"],
                   linestyle=":", label=f"SAE baseline (T=5)", alpha=0.8)
    if mlc_baseline:
        ax.axhline(statistics.mean(mlc_baseline), color=ARCH_COLOR["mlc"],
                   linestyle=":", label=f"MLC baseline (T=5)", alpha=0.8)

    ax.set_xlabel("T (temporal window size)")
    ax.set_ylabel("Sparse-probing accuracy (k=5, mean over 8 tasks)")
    ax.set_title("T-sweep: TempXC full_window × protocol A vs B × k=5")
    ax.set_xticks([5, 10, 20])
    ax.set_ylim(0.6, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = PLOTS_DIR / "fig3_tsweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_per_task(records):
    """Per-task grouped bar chart at T=5 × protA × full_window × k=5."""
    by = defaultdict(dict)
    for r in records:
        if (r["t"] == 5
                and r["matching_protocol"] == "A"
                and r["aggregation"] == "full_window"
                and r["k"] == 5):
            short = _normalize_task(r["task"])
            # Truncate very long task names
            short = short.replace("bias_in_bios_class_set", "bios")
            short = short.replace("amazon_reviews_mcauley_1and5", "amazon")
            short = short.replace("github-code", "github")
            by[short][r["architecture"]] = r["accuracy"]
    tasks = sorted(by.keys())
    archs = ["sae", "mlc", "tempxc"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(tasks))
    w = 0.27
    for i, a in enumerate(archs):
        accs = [by[t].get(a, 0) for t in tasks]
        ax.bar(x + (i - 1) * w, accs, w, label=a.upper(),
               color=ARCH_COLOR[a], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("Sparse-probing accuracy (k=5)")
    ax.set_title("Per-task accuracy at T=5 × protocol A × full_window × k=5")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = PLOTS_DIR / "fig4_per_task.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_k_sweep(records):
    """Accuracy vs k (probing top-k features) at T=5 × protA × full_window."""
    archs = ["sae", "mlc", "tempxc"]
    ks = [1, 2, 5, 20]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in archs:
        by_k = defaultdict(list)
        for r in records:
            if (r["architecture"] == a
                    and r["t"] == 5
                    and r["matching_protocol"] == "A"
                    and r["aggregation"] == "full_window"):
                by_k[r["k"]].append(r["accuracy"])
        means = [statistics.mean(by_k[k]) for k in ks if k in by_k]
        valid_ks = [k for k in ks if k in by_k]
        ax.plot(valid_ks, means, "-o", color=ARCH_COLOR[a], label=a.upper(),
                linewidth=2, markersize=8)
    ax.set_xlabel("Top-k features selected for probe")
    ax.set_ylabel("Sparse-probing accuracy (mean over 8 tasks)")
    ax.set_title("k-sweep: probing capacity vs accuracy\nT=5 × protocol A × full_window")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 20])
    ax.set_xticklabels([1, 2, 5, 20])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = PLOTS_DIR / "fig5_k_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    records = load_records()
    print(f"Loaded {len(records)} deduped records from "
          f"{len(glob.glob('results/saebench/results/*.jsonl'))} JSONL files")

    headline = table_headline(records)
    table_aggregation_ablation(records)
    table_tsweep(records, protocol="A")
    table_tsweep(records, protocol="B")
    table_per_task(records)

    print("\n=== PLOTS ===")
    plot_headline(headline)
    plot_aggregation_ablation(records)
    plot_tsweep(records)
    plot_per_task(records)
    plot_k_sweep(records)

    print(f"\nAll plots in {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
