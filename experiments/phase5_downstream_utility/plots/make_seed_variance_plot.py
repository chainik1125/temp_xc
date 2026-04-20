"""Render Phase-5 headline bar with cross-seed mean ± std.

Reads `results/probing_results.jsonl` and aggregates `test_auc` over
seeds {1, 2, 3, 42} for each (arch, task). Keeps only archs that have
>=2 seeds. Emits `results/plots/headline_seed_variance_k{k}_last_position_{metric}.png`.

Usage:
    .venv/bin/python experiments/phase5_downstream_utility/plots/make_seed_variance_plot.py \\
        [--metric auc] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.save_figure import save_figure


REPO = Path("/workspace/temp_xc")
RESULTS_DIR = REPO / "experiments/phase5_downstream_utility/results"
JSONL = RESULTS_DIR / "probing_results.jsonl"
PLOTS_DIR = RESULTS_DIR / "plots"

FLIP_TASKS = {"winogrande_correct_completion", "wsc_coreference"}

ARCHS_WITH_SEED_VARIANCE = [
    "mlc", "time_layer_crosscoder_t5", "txcdr_rank_k_dec_t5",
    "txcdr_t5", "txcdr_tied_t5",
]

RUN_ID_RE = re.compile(r"^(?P<arch>.+)__seed(?P<seed>\d+)$")


def load(metric: str, k: int) -> dict[str, dict[int, dict[str, float]]]:
    """{arch: {seed: {task: value}}}"""
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    key = f"test_{metric}"
    with JSONL.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("error") or r.get(key) is None:
                continue
            if r.get("aggregation") != "last_position":
                continue
            if r.get("k_feat") != k:
                continue
            rid = r.get("run_id", "")
            m = RUN_ID_RE.match(rid)
            if not m:
                continue
            arch = m.group("arch")
            seed = int(m.group("seed"))
            task = r.get("task_name")
            v = float(r[key])
            if task in FLIP_TASKS:
                v = max(v, 1.0 - v)
            out[arch][seed][task] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="auc", choices=["auc", "acc"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load(args.metric, args.k)

    rows = []
    for arch in ARCHS_WITH_SEED_VARIANCE:
        seeds_present = sorted(data.get(arch, {}).keys())
        if len(seeds_present) < 2:
            print(f"[seed_variance] skip {arch} — only {len(seeds_present)} seeds")
            continue
        per_seed_means = [
            float(np.mean(list(data[arch][s].values())))
            for s in seeds_present
        ]
        rows.append({
            "arch": arch,
            "n_seeds": len(seeds_present),
            "seeds": seeds_present,
            f"mean_{args.metric}": float(np.mean(per_seed_means)),
            f"std_{args.metric}": float(np.std(per_seed_means)),
            "per_seed_means": per_seed_means,
        })

    if not rows:
        print(f"[seed_variance] no archs have >=2 seeds for {args.metric} k={args.k}")
        return

    rows.sort(key=lambda r: -r[f"mean_{args.metric}"])

    out_json = RESULTS_DIR / f"seed_variance_summary_last_position_{args.metric}_k{args.k}.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"[seed_variance] wrote {out_json}")

    names = [r["arch"] for r in rows]
    means = [r[f"mean_{args.metric}"] for r in rows]
    stds = [r[f"std_{args.metric}"] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(names)), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color="C0")
    for xi, (m, s, r) in enumerate(zip(means, stds, rows)):
        ax.text(xi, m + s + 0.005, f"{m:.3f}±{s:.3f}\n(n={r['n_seeds']})",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(f"mean {args.metric.upper()} across seeds (k={args.k})")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Phase 5 — last_position seed variance [k={args.k}, metric={args.metric}]")
    out_png = PLOTS_DIR / f"headline_seed_variance_k{args.k}_last_position_{args.metric}.png"
    save_figure(fig, str(out_png))
    plt.close(fig)
    print(f"[seed_variance] wrote {out_png}")


if __name__ == "__main__":
    main()
