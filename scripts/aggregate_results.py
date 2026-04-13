#!/usr/bin/env python3
"""Aggregate per-run bench JSONs into a Slack-ready comparison report.

Ingests every JSON under `results/` (or a supplied --root), joins on
(model, dataset, layer, arch, k, T), and emits:

  - report.md       Comparison table + embedded plot links
  - nmse_l0.png     Pareto frontier (reconstruction vs sparsity) per arch
  - max_cos_hist.png  Distribution of max-cosine feature overlap
  - temporal_mi.png Mean temporal MI by lag, per arch
  - span_hist.png   Mean activation span distribution, per arch
  - umap_*.png      Side-by-side UMAPs per arch (one file per dataset/layer)

Every plot is a single PNG that pastes directly into Slack. The report.md
fences paths as markdown images.

Usage:
    python scripts/aggregate_results.py --root results/ --out reports/latest/
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_runs(root: Path) -> list[dict]:
    runs = []
    for p in root.rglob("*.json"):
        try:
            with open(p) as f:
                d = json.load(f)
            if "nmse" not in d and "result" in d:
                d = d["result"]
            if "nmse" in d:
                d.setdefault("_path", str(p))
                runs.append(d)
        except Exception as e:
            print(f"  skipped {p}: {e}")
    return runs


def _group(runs: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict = defaultdict(list)
    for r in runs:
        k = tuple(r.get(kk, "?") for kk in keys)
        out[k].append(r)
    return out


def _pareto_plot(runs: list[dict], out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    by_arch = _group(runs, ("arch",))
    for (arch,), rs in sorted(by_arch.items()):
        xs = [r["l0"] for r in rs if r.get("l0") is not None]
        ys = [r["nmse"] for r in rs if r.get("nmse") is not None]
        if not xs:
            continue
        ax.scatter(xs, ys, label=str(arch), alpha=0.75)
    ax.set_xlabel("L0 (sparsity)")
    ax.set_ylabel("NMSE")
    ax.set_title("Pareto: reconstruction vs sparsity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _temporal_mi_plot(runs: list[dict], out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    by_arch = _group(runs, ("arch",))
    for (arch,), rs in sorted(by_arch.items()):
        curves = []
        lags = None
        for r in rs:
            tmi = r.get("temporal_mi")
            if not tmi or "lags" not in tmi:
                continue
            lags = tmi["lags"]
            curves.append(tmi["mean_mi_per_lag"])
        if not curves:
            continue
        mean = np.mean(curves, axis=0)
        ax.plot(lags, mean, marker="o", label=str(arch))
    ax.set_xlabel("lag k")
    ax.set_ylabel("mean temporal MI (nats)")
    ax.set_title("Temporal MI by lag")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _span_plot(runs: list[dict], out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    by_arch = _group(runs, ("arch",))
    for (arch,), rs in sorted(by_arch.items()):
        vals = [r["span_stats"]["mean_span"] for r in rs
                if r.get("span_stats") and "mean_span" in r["span_stats"]]
        if not vals:
            continue
        ax.hist(vals, bins=20, alpha=0.5, label=str(arch))
    ax.set_xlabel("mean activation span (tokens)")
    ax.set_ylabel("# runs")
    ax.set_title("Activation span distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _max_cos_plot(runs: list[dict], out_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    by_arch = _group(runs, ("arch",))
    for (arch,), rs in sorted(by_arch.items()):
        vals = [r["mean_max_cos"] for r in rs if r.get("mean_max_cos") is not None]
        if not vals:
            continue
        ax.hist(vals, bins=20, alpha=0.5, label=str(arch))
    ax.set_xlabel("mean max-cosine (feature overlap)")
    ax.set_ylabel("# runs")
    ax.set_title("Feature overlap distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _table_md(runs: list[dict]) -> str:
    cols = ["arch", "subject_model", "layer_key", "k", "T", "nmse", "l0",
            "auc", "mean_max_cos"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in sorted(runs, key=lambda x: (str(x.get("arch", "")),
                                         str(x.get("subject_model", "")),
                                         str(x.get("layer_key", "")))):
        row = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("reports/latest"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.root)
    print(f"Loaded {len(runs)} runs from {args.root}")
    if not runs:
        print("No runs found.")
        return

    _pareto_plot(runs, args.out / "nmse_l0.png")
    _max_cos_plot(runs, args.out / "max_cos_hist.png")
    _temporal_mi_plot(runs, args.out / "temporal_mi.png")
    _span_plot(runs, args.out / "span_hist.png")

    report = args.out / "report.md"
    with open(report, "w") as f:
        f.write("## Sprint comparison report\n\n")
        f.write(f"- root: `{args.root}`\n- n_runs: {len(runs)}\n\n")
        f.write("### Comparison table\n\n")
        f.write(_table_md(runs))
        f.write("\n\n### Plots\n\n")
        for png in ("nmse_l0.png", "max_cos_hist.png", "temporal_mi.png", "span_hist.png"):
            f.write(f"![{png}]({png})\n\n")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
