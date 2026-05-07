"""Render paper-ready bundles for the C1 + C2 synthetic sweeps.

Inputs: rows from ``results/leaderboard.jsonl`` with
``component in {c1, c2}`` and ``eval_cfg._full_suite = true`` (the
flag set by the ``c{1,2}_synthetic_*_full`` drivers, isolating us
from agent_filler's collisions).

Outputs (under ``--output-dir``, default ``docs/components/``):
- ``c1_paper_results.md`` + ``c1_paper_assets/`` (AUC vs k_pos)
- ``c2_paper_results.md`` + ``c2_paper_assets/`` (gAUC + eAUC vs k_pos)

Aggregation: rows are grouped by ``(arch, t_label, k_pos)``; multiple
seeds within a group become mean ± SE error bars. Per-arch headline
is the peak metric across the k_pos sweep.

Usage::

    .venv/bin/python -m scripts.synthetic_paper_renderer \\
        --output-dir figs/<component>
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PURIFIED = Path(__file__).resolve().parents[1]
LEADERBOARD = PURIFIED / "results" / "leaderboard.jsonl"

# Stable color per (arch, t_label) — same arch with different T_max
# gets a distinct shade so curves are visually separable.
_ARCH_COLORS = {
    "topk_sae":     "#1f77b4",
    "tsae_paper":   "#9467bd",
    "tfa":          "#2ca02c",
    "tfa_pos":      "#17becf",
    "stacked_sae":  "#8c564b",
    "mlc":          "#e377c2",
    "txc_base":     "#d62728",
    "txc_pro":      "#ff7f0e",
}
_T_SHADE = {"default": 1.0, "T=2": 0.55, "T=5": 0.85, "T=12": 1.15}


def _adjust(c: str, scale: float) -> tuple[float, float, float]:
    rgb = matplotlib.colors.to_rgb(c)
    return tuple(min(1.0, max(0.0, x * scale)) for x in rgb)


def color(arch: str, t_label: str) -> tuple[float, float, float]:
    base = _ARCH_COLORS.get(arch, "#444444")
    return _adjust(base, _T_SHADE.get(t_label, 1.0))


def label_for(arch: str, t_label: str) -> str:
    if t_label == "default":
        return arch
    return f"{arch} ({t_label})"


def load_rows(component: str) -> list[dict]:
    if not LEADERBOARD.exists():
        return []
    rows: list[dict] = []
    with LEADERBOARD.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("component") != component:
                continue
            if not r.get("eval_cfg", {}).get("_full_suite"):
                continue
            rows.append(r)
    return rows


def aggregate(rows: list[dict], metric: str
              ) -> dict[tuple[str, str], dict[int, tuple[float, float, int]]]:
    """{(arch, t_label): {k_pos: (mean, sem, n_seeds)}}."""
    by_group: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        arch = r["arch"]
        t_label = r.get("eval_cfg", {}).get("t_label", "default")
        k_pos = r.get("eval_cfg", {}).get("k_pos")
        if k_pos is None:
            continue
        v = r.get("metrics", {}).get(metric)
        if v is None:
            continue
        by_group[(arch, t_label, int(k_pos))].append(float(v))
    out: dict[tuple[str, str], dict[int, tuple[float, float, int]]] = defaultdict(dict)
    for (arch, t_label, k), vals in by_group.items():
        arr = np.asarray(vals, dtype=float)
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        out[(arch, t_label)][k] = (float(arr.mean()), sem, int(len(arr)))
    return out


def _arch_sort_key(arch: str, t_label: str) -> tuple[int, str, str]:
    order = {"topk_sae": 0, "tsae_paper": 1, "tfa": 2, "tfa_pos": 3,
             "stacked_sae": 4, "mlc": 5, "txc_base": 6, "txc_pro": 7}
    return (order.get(arch, 9), arch, t_label)


def plot_metric_vs_k(agg: dict[tuple[str, str], dict[int, tuple[float, float, int]]],
                     metric_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    keys = sorted(agg.keys(), key=lambda kt: _arch_sort_key(*kt))
    for arch, t_label in keys:
        per_k = agg[(arch, t_label)]
        if not per_k:
            continue
        ks = sorted(per_k.keys())
        means = [per_k[k][0] for k in ks]
        sems = [per_k[k][1] for k in ks]
        c = color(arch, t_label)
        ax.errorbar(ks, means, yerr=sems, marker="o", linewidth=1.6,
                    markersize=4.5, capsize=3, color=c, label=label_for(arch, t_label))
    ax.set_xlabel("k_pos")
    ax.set_ylabel(metric_label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    ax.set_title(f"{metric_label} vs sparsity (mean ± SE across seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def best_per_arch(agg: dict[tuple[str, str], dict[int, tuple[float, float, int]]]
                  ) -> list[tuple[str, str, int, float, float, int]]:
    """For each (arch, t_label), pick the k_pos with the highest mean.
    Returns rows of (arch, t_label, best_k, best_mean, best_sem, n_seeds)."""
    out = []
    for (arch, t_label), per_k in agg.items():
        if not per_k:
            continue
        best_k, (mean, sem, n) = max(per_k.items(), key=lambda kv: kv[1][0])
        out.append((arch, t_label, best_k, mean, sem, n))
    out.sort(key=lambda x: -x[3])  # by mean desc
    return out


def headline_table(best: list[tuple[str, str, int, float, float, int]],
                   metric_label: str) -> str:
    lines = [
        f"| arch | T | k\\* | {metric_label} (mean ± SE) | n seeds |",
        "|------|---|----|---------------------------|---------|",
    ]
    for arch, t_label, k, mean, sem, n in best:
        t_pretty = "—" if t_label == "default" else t_label
        lines.append(f"| `{arch}` | {t_pretty} | {k} | {mean:.3f} ± {sem:.3f} | {n} |")
    return "\n".join(lines)


def per_cell_table(agg: dict[tuple[str, str], dict[int, tuple[float, float, int]]],
                   metric_label: str) -> str:
    keys = sorted(agg.keys(), key=lambda kt: _arch_sort_key(*kt))
    all_k = sorted({k for per_k in agg.values() for k in per_k.keys()})
    if not all_k:
        return "_no rows yet_"
    head = "| arch (T) | " + " | ".join(f"k={k}" for k in all_k) + " |"
    sep = "|" + "---|" * (len(all_k) + 1)
    lines = [head, sep]
    for arch, t_label in keys:
        per_k = agg[(arch, t_label)]
        cells = []
        for k in all_k:
            if k in per_k:
                m, s, n = per_k[k]
                cells.append(f"{m:.3f}")
            else:
                cells.append("—")
        lines.append(f"| `{label_for(arch, t_label)}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_c1(out_dir: Path) -> Path | None:
    rows = load_rows("c1")
    if not rows:
        return None
    assets = out_dir / "c1_paper_assets"
    assets.mkdir(parents=True, exist_ok=True)
    agg = aggregate(rows, "auc")
    plot_path = assets / "c1_auc_vs_k.png"
    plot_metric_vs_k(agg, "feature-recovery AUC", plot_path)
    best = best_per_arch(agg)
    n_seeds_set = sorted({n for *_, n in best})
    md = [
        "# C1 Synthetic TopK — paper-ready results (auto-generated)",
        "",
        f"_Last refreshed: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "## Setup",
        "",
        "- Datasource: `toy_markov_n20_d40_full` (Markov-chain support, "
        "`n_features=20`, `d_in=40`, `seq_len=8`, `n_seqs=4096`, `pi=0.10`).",
        "- Sweep: 8 archs × 12 `k_pos` values (1–20) × 3 seeds.",
        "- Metric: feature-recovery AUC against the 20 ground-truth orthogonal "
        "directions (per-feature max cosine ≥ τ, integrated over τ ∈ [0, 1]).",
        "- Driver: `experiments/c1_synthetic_topk_full/run.py` (routes through "
        "`temp_bench.data.toy_full.api`, isolated from agent_filler's "
        "`toy_markov_n20_d40` cache via distinct `act_cache_key`).",
        "",
        f"Currently {len(rows)} cells across {len({(r['arch'], r.get('eval_cfg',{}).get('t_label','default')) for r in rows})} (arch, T) configurations; "
        f"seeds-per-cell range = {n_seeds_set or '?'}.",
        "",
        "## Headline (peak AUC over k_pos)",
        "",
        headline_table(best, "AUC"),
        "",
        "## AUC vs k_pos",
        "",
        "![c1_auc_vs_k](c1_paper_assets/c1_auc_vs_k.png)",
        "",
        "## Per-cell mean (averaged across seeds)",
        "",
        per_cell_table(agg, "AUC"),
        "",
    ]
    out_path = out_dir / "c1_paper_results.md"
    out_path.write_text("\n".join(md))
    return out_path


def render_c2(out_dir: Path) -> Path | None:
    rows = load_rows("c2")
    if not rows:
        return None
    assets = out_dir / "c2_paper_assets"
    assets.mkdir(parents=True, exist_ok=True)
    g_agg = aggregate(rows, "gauc")
    e_agg = aggregate(rows, "eauc")
    g_plot = assets / "c2_gauc_vs_k.png"
    e_plot = assets / "c2_eauc_vs_k.png"
    plot_metric_vs_k(g_agg, "global-recovery gAUC (vs hidden)", g_plot)
    plot_metric_vs_k(e_agg, "emission-recovery eAUC (vs M emissions)", e_plot)
    g_best = best_per_arch(g_agg)
    e_best = best_per_arch(e_agg)
    n_seeds_set = sorted({n for *_, n in g_best})
    md = [
        "# C2 Synthetic Coupled — paper-ready results (auto-generated)",
        "",
        f"_Last refreshed: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
        "",
        "## Setup",
        "",
        "- Datasource: `toy_coupled_K10_M20_d256_full` (factorial-HMM hidden "
        "→ OR-gate emissions; `K_hidden=10`, `M_emissions=20`, `d_in=256`).",
        "- Sweep: 7 (arch, T) variants × 12 `k_pos` values × 3 seeds.",
        "- Headline metric: gAUC — recovery of the K=10 hidden directions "
        "(the latents the SAE *should* find if it bypasses the OR-gate).",
        "- Companion: eAUC — recovery of the M=20 emission directions.",
        "- Driver: `experiments/c2_synthetic_coupled_full/run.py` (isolated "
        "`act_cache_key` via `temp_bench.data.toy_full.api:coupled_hmm`).",
        "",
        f"Currently {len(rows)} cells across {len({(r['arch'], r.get('eval_cfg',{}).get('t_label','default')) for r in rows})} (arch, T) configurations; "
        f"seeds-per-cell range = {n_seeds_set or '?'}.",
        "",
        "## gAUC headline (peak gAUC over k_pos)",
        "",
        headline_table(g_best, "gAUC"),
        "",
        "![c2_gauc_vs_k](c2_paper_assets/c2_gauc_vs_k.png)",
        "",
        "## eAUC headline (peak eAUC over k_pos)",
        "",
        headline_table(e_best, "eAUC"),
        "",
        "![c2_eauc_vs_k](c2_paper_assets/c2_eauc_vs_k.png)",
        "",
        "## Per-cell mean — gAUC",
        "",
        per_cell_table(g_agg, "gAUC"),
        "",
        "## Per-cell mean — eAUC",
        "",
        per_cell_table(e_agg, "eAUC"),
        "",
    ]
    out_path = out_dir / "c2_paper_results.md"
    out_path.write_text("\n".join(md))
    return out_path


def main(*, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    c1 = render_c1(output_dir)
    c2 = render_c2(output_dir)
    rows1 = len(load_rows("c1"))
    rows2 = len(load_rows("c2"))
    print(f"[synthetic_paper] c1 rows={rows1} → "
          f"{c1 if c1 else '(no cells yet)'}")
    print(f"[synthetic_paper] c2 rows={rows2} → "
          f"{c2 if c2 else '(no cells yet)'}")


def cli():
    ap = argparse.ArgumentParser(description=(
        "Synthetic (C1 + C2) paper figure renderer. Reads "
        "purified/results/leaderboard.jsonl in-repo."
    ))
    ap.add_argument(
        "--output-dir", type=Path,
        default=PURIFIED / "figs" / "synthetic",
        help="Output directory (default: purified/figs/synthetic/).",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(output_dir=args.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    cli()
