"""Reviewer-robust Pareto trade-off figure: 3-seed mean ± stderr on
both axes for the 5 paper-load-bearing archs (baseline TXC, Track 2,
2×2 cell, Cycle F, tsae_paper). Phase 6.2 ablations (C1/C2/C3/C5/C6)
shown as small grey markers (no labels) in the TXC cluster region.

Clean two-panel layout:
  Main panel  — full probing-vs-qualitative Pareto plane, 5 primary
                archs with 3-seed error bars on both axes.
  Right panel — zoom into the TXC cluster showing Phase 6.2 ablation
                points (C1/C2/C3/C5/C6) annotated individually.

Axes:
    x = mean_pool sparse-probing AUC (Phase 5 benchmark, k=5)
    y = concat_random SEMANTIC label count (rigorous metric, N=32)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
AUTOIN = REPO / "experiments/phase6_qualitative_latents/results/autointerp"

# Primary archs with 3-seed data. Tuple = (display_name, color, marker, label_offset)
PRIMARY = {
    "tsae_paper":                  ("T-SAE (paper)",             "#d62728", "*", (10, -2)),
    "agentic_txc_10_bare":         ("Track 2 (TXC+anti-dead)",   "#08519c", "D", (10, 6)),
    "agentic_txc_12_bare_batchtopk": ("2×2 cell",                "#4292c6", "s", (10, -18)),
    "agentic_txc_02":              ("TXC baseline",              "#3182bd", "o", (10, -18)),
    "agentic_txc_02_batchtopk":    ("Cycle F",                    "#9ecae1", "v", (-45, -18)),
    "agentic_mlc_08":              ("MLC (multi-layer)",          "#2ca02c", "p", (10, 6)),
    "txcdr_t5":                    ("TXCDR T=5 (no anti-dead)",   "#6baed6", "^", (10, -14)),
    "mlc":                         ("MLC baseline",               "#74c476", "<", (10, -14)),
    "mlc_contrastive_alpha100":    ("MLC contrastive α=1",        "#238b45", ">", (-55, -10)),
    "agentic_mlc_08_batchtopk":    ("MLC (+BatchTopK)",           "#41ab5d", "H", (10, 6)),
    "phase57_partB_h8_bare_multidistance": ("H8 (multidistance)",  "#fd8d3c", "X", (10, 6)),
}

SECONDARY = {
    "phase62_c1_track2_matryoshka":               ("C1  Track 2 + matryoshka",        "#9e9ac8"),
    "phase62_c2_track2_contrastive":              ("C2  Track 2 + contrastive",       "#756bb1"),
    "phase62_c3_track2_matryoshka_contrastive":   ("C3  Track 2 + matr + contr",      "#54278f"),
    "phase62_c5_track2_longer":                   ("C5  Track 2 longer train",        "#bcbddc"),
    "phase62_c6_bare_batchtopk_longer":           ("C6  2×2 cell longer",             "#dadaeb"),
}

# Phase 6.3 T-sweep: Track 2 recipe at T ∈ {3, 10, 20}.
# Plotted as a trajectory through the primary Track 2 (T=5) point
# on the zoom panel, showing the T axis explicitly.
T_SWEEP = {
    "phase63_track2_t3":   (3,  "T=3"),
    "phase63_track2_t10": (10, "T=10"),
    "phase63_track2_t20": (20, "T=20"),
}


def load_probing(agg="mean_pool", k=5):
    per_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    with PROBE.open() as f:
        for line in f:
            d = json.loads(line)
            if "aggregation" not in d or "k_feat" not in d:
                continue  # skip error rows from failed probes
            if d["aggregation"] != agg or d["k_feat"] != k:
                continue
            rid = d["run_id"]
            if "__seed" not in rid:
                continue
            arch, sd = rid.rsplit("__seed", 1)
            try:
                sd = int(sd)
            except ValueError:
                continue
            key = (rid, d["task_name"])
            if key in seen:
                continue
            seen.add(key)
            per_seed[arch][sd].append(d["test_auc"])
    out = {}
    for arch, seed_dict in per_seed.items():
        seed_means = [sum(v) / len(v) for v in seed_dict.values() if len(v) == 36]
        if seed_means:
            out[arch] = seed_means
    return out


def load_random_sem(metric_key: str = "semantic_count"):
    """Load per-arch list of random-concat SEMANTIC counts.

    metric_key: "semantic_count" for var-based ranking (original Phase 6.1
    metric) or "semantic_count_pdvar" for passage-discriminative-variance
    ranking (Priority 2a). Cells missing the key are skipped.
    """
    out: dict[str, list[int]] = defaultdict(list)
    for p in AUTOIN.glob("*__seed*__concatrandom__labels.json"):
        d = json.loads(p.read_text())
        metrics = d.get("metrics", {})
        if metric_key in metrics:
            out[d["arch"]].append(metrics[metric_key])
    return dict(out)


def load_passage_probe(k: int = 5):
    """Load per-arch list of per-seed mean passage-ID probe accuracy
    across the 3 concats (A, B, random). Returns {arch: [seed_means...]}.

    Source file: `results/passage_probe_results.jsonl` emitted by
    run_passage_probe.py. Each row has per_k[str(k)]["acc_mean"] for
    one (arch, seed, concat) cell. We average over the 3 concats per
    seed, then collect across seeds.
    """
    probe_path = REPO / "experiments/phase6_qualitative_latents/results/passage_probe_results.jsonl"
    if not probe_path.exists():
        return {}
    per_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with probe_path.open() as f:
        for line in f:
            d = json.loads(line)
            k_str = str(k)
            if k_str not in d.get("per_k", {}):
                continue
            per_seed[d["arch"]][d["seed"]].append(d["per_k"][k_str]["acc_mean"])
    out = {}
    for arch, seed_dict in per_seed.items():
        # Average over concats per seed (only keep seeds with all 3 concats present)
        seed_means = [sum(v) / len(v) for v in seed_dict.values() if len(v) == 3]
        if seed_means:
            out[arch] = seed_means
    return out


def _mean_se(vals):
    if not vals:
        return float("nan"), 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def _plot_primary(ax, aucs, sems, show_labels=True, show_errorbars=True):
    for arch, (label, color, marker, _off) in PRIMARY.items():
        if arch not in aucs or arch not in sems:
            continue
        a_mean, a_se = _mean_se(aucs[arch])
        s_mean, s_se = _mean_se(sems[arch])
        ax.errorbar(
            [a_mean], [s_mean],
            xerr=[a_se] if show_errorbars and a_se > 0 else None,
            yerr=[s_se] if show_errorbars and s_se > 0 else None,
            fmt=marker, markersize=14, color=color,
            markeredgecolor="black", markeredgewidth=1.2,
            ecolor="black", elinewidth=1.0, capsize=3,
            zorder=5,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="mean_pool",
                    choices=["last_position", "mean_pool"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--metric", default="semantic_count",
                    choices=["semantic_count", "semantic_count_pdvar",
                             "probe_acc_passage"],
                    help="which qualitative metric to put on the y-axis. "
                         "probe_acc_passage = k-sparse multinomial probe "
                         "predicting passage ID across A/B/random (T-SAE §4.2 style).")
    ap.add_argument("--out", type=str,
                    default=str(REPO / "experiments/phase6_qualitative_latents/results/phase61_pareto_robust.png"))
    args = ap.parse_args()

    aucs = load_probing(agg=args.agg, k=args.k)
    if args.metric == "probe_acc_passage":
        sems = load_passage_probe(k=args.k)
        metric_display = "paper-style passage probe"
    elif args.metric == "semantic_count_pdvar":
        sems = load_random_sem(metric_key=args.metric)
        metric_display = "pdvar-ranked"
    else:
        sems = load_random_sem(metric_key=args.metric)
        metric_display = "var-ranked"

    # 2-panel: (A) full Pareto plane, (B) TXC-cluster zoom with Phase 6.2 labels
    fig, (ax, axz) = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.55},
    )

    # Primary points (no text labels yet — add via legend)
    _plot_primary(ax, aucs, sems, show_labels=False, show_errorbars=True)

    # Place primary labels with manual offsets to avoid overlap
    for arch, (label, color, marker, offset) in PRIMARY.items():
        if arch not in aucs or arch not in sems:
            continue
        a_mean, _ = _mean_se(aucs[arch])
        s_mean, _ = _mean_se(sems[arch])
        ax.annotate(
            label, (a_mean, s_mean), xytext=offset,
            textcoords="offset points", fontsize=10, fontweight="bold",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, linewidth=1.0, alpha=0.9),
        )

    # Pareto frontier — compute non-dominated set over PRIMARY arches.
    # Non-dominated: no other arch has higher AUC AND higher qualitative.
    primary_pts = []
    for arch in PRIMARY:
        if arch in aucs and arch in sems:
            primary_pts.append((arch, _mean_se(aucs[arch])[0],
                                _mean_se(sems[arch])[0]))
    def _dominated(p, pts):
        return any(q[1] >= p[1] and q[2] >= p[2] and q != p for q in pts)
    frontier = [(x, y) for (_, x, y) in primary_pts
                if not _dominated((None, x, y), primary_pts)]
    frontier.sort(key=lambda p: p[0])
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                "--", color="#333333", linewidth=1.8, alpha=0.7, zorder=2)

    # Gap arrows between tsae_paper and Track 2 (minimalist)
    if "tsae_paper" in aucs and "agentic_txc_10_bare" in aucs:
        tsp_x = _mean_se(aucs["tsae_paper"])[0]
        tsp_y = _mean_se(sems["tsae_paper"])[0]
        tr2_x = _mean_se(aucs["agentic_txc_10_bare"])[0]
        tr2_y = _mean_se(sems["agentic_txc_10_bare"])[0]

        # Horizontal (probing) gap: drawn at y = mid-low, scaled by metric
        if args.metric == "probe_acc_passage":
            mid_y = 0.55
        elif args.metric == "semantic_count_pdvar":
            mid_y = 15.0
        else:
            mid_y = 7.0
        ax.annotate(
            "", xy=(tsp_x, mid_y), xytext=(tr2_x, mid_y),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3),
        )
        ax.text((tsp_x + tr2_x) / 2, mid_y - 0.7,
                f"Δ AUC = {tr2_x - tsp_x:+.3f}",
                ha="center", fontsize=10, color="#333333",
                bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                          boxstyle="round,pad=0.2"))

        # Vertical (qualitative) gap: at x just right of tsae_paper
        mid_x = tsp_x + 0.012
        ax.annotate(
            "", xy=(mid_x, tsp_y), xytext=(mid_x, tr2_y),
            arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.3),
        )
        ax.text(mid_x + 0.002, (tsp_y + tr2_y) / 2,
                f"Δ = {tsp_y - tr2_y:+.1f} labels",
                ha="left", va="center", fontsize=10, color="#333333",
                bbox=dict(facecolor="white", edgecolor="#aaaaaa",
                          boxstyle="round,pad=0.2"))

    # Shaded regions — y-axis scale depends on metric
    xlim_lo = min(0.71, _mean_se(aucs.get("tsae_paper", [0.72]))[0] - 0.015)
    xlim_hi = 0.815
    ax.set_xlim(xlim_lo, xlim_hi)
    if args.metric == "probe_acc_passage":
        y_lo, y_hi = 0.4, 1.0
        y_split = 0.7
    elif args.metric == "semantic_count_pdvar":
        y_lo, y_hi = -1, 28
        y_split = 14
    else:
        y_lo, y_hi = -1, 17
        y_split = 6
    ax.set_ylim(y_lo, y_hi)
    ax.axhspan(y_lo, y_split, xmin=0, xmax=1, color="#08519c", alpha=0.04, zorder=0)
    ax.axhspan(y_split, y_hi, xmin=0, xmax=1, color="#d62728", alpha=0.04, zorder=0)

    if args.metric == "probe_acc_passage":
        top_label_y = y_hi - 0.06
        bot_label_y = y_lo + 0.02
    else:
        top_label_y = y_hi - 1.5
        bot_label_y = 0.2
    ax.text(xlim_lo + 0.003, top_label_y,
            "T-SAE (paper) region\n— wins qualitative",
            fontsize=9, color="#d62728", alpha=0.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    ax.text(xlim_hi - 0.003, bot_label_y,
            "TXC family region\n— wins probing",
            fontsize=9, color="#08519c", alpha=0.8, ha="right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    ax.set_xlabel(f"Probing mean AUC  ({args.agg}, k=5)  "
                  f"— Phase 5 SAEBench protocol, 36 binary tasks",
                  fontsize=11)
    if args.metric == "probe_acc_passage":
        ax.set_ylabel(
            f"Paper-style passage-ID probe accuracy  (k={args.k})\n"
            f"— mean across concat_A/B/random, 5-fold stratified CV",
            fontsize=11,
        )
    else:
        ylabel_detail = ("pdvar ranking (passage-discriminative variance)"
                         if args.metric == "semantic_count_pdvar"
                         else "top-32 per-token-variance ranking")
        ax.set_ylabel(
            f"SEMANTIC label count on concat_random  (/32)\n"
            f"— {ylabel_detail}, multi-Haiku temp=0",
            fontsize=11,
        )
    ax.set_title(
        f"TXC vs T-SAE: Pareto trade-off  ({metric_display})\n"
        "Error bars: stderr across 3 training seeds on both axes",
        fontsize=12,
    )
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # Main-plot legend (primary archs + Pareto frontier + ablation inset caption)
    from matplotlib.lines import Line2D
    legend_handles = []
    for arch, (label, color, marker, _) in PRIMARY.items():
        legend_handles.append(Line2D([], [], marker=marker, color=color,
                                     markeredgecolor="black", markersize=10,
                                     linestyle="", label=label))
    legend_handles.append(Line2D([], [], linestyle="--", color="#333333",
                                 label="Pareto frontier"))
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=9,
              framealpha=0.95, borderaxespad=0)

    # --- Right panel: TXC-cluster zoom with Phase 6.2 labels ---
    # Redraw the TXC-cluster primary archs
    inset_archs = (
        "agentic_txc_02", "agentic_txc_10_bare",
        "agentic_txc_12_bare_batchtopk", "agentic_txc_02_batchtopk",
    )
    # Use manual offsets tuned for the zoom panel (different from main panel)
    ZOOM_OFFSETS = {
        "agentic_txc_02":                ( 8, -15),
        "agentic_txc_10_bare":            ( 8,  5),
        "agentic_txc_12_bare_batchtopk":  ( 8, -12),
        "agentic_txc_02_batchtopk":       (-70, -5),
    }
    for arch in inset_archs:
        if arch not in aucs:
            continue
        label, color, marker, _ = PRIMARY[arch]
        a_mean, a_se = _mean_se(aucs[arch])
        s_mean, s_se = _mean_se(sems[arch])
        axz.errorbar(
            [a_mean], [s_mean],
            xerr=[a_se] if a_se > 0 else None,
            yerr=[s_se] if s_se > 0 else None,
            fmt=marker, markersize=12, color=color,
            markeredgecolor="black", markeredgewidth=1.0,
            ecolor="black", elinewidth=0.8, capsize=3,
        )
        axz.annotate(
            label, (a_mean, s_mean), xytext=ZOOM_OFFSETS[arch],
            textcoords="offset points", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, linewidth=1.0, alpha=0.9),
        )
    # Phase 6.2 secondary points, labelled individually
    C_OFFSETS = {
        "phase62_c1_track2_matryoshka":                ( 6,  5),
        "phase62_c2_track2_contrastive":               ( 6, -9),
        "phase62_c3_track2_matryoshka_contrastive":    ( 6, -5),
        "phase62_c5_track2_longer":                    (-38,  4),
        "phase62_c6_bare_batchtopk_longer":            ( 6, -5),
    }
    for arch, (label, color) in SECONDARY.items():
        if arch not in aucs or arch not in sems:
            continue
        a_mean, _ = _mean_se(aucs[arch])
        s_mean, _ = _mean_se(sems[arch])
        axz.scatter([a_mean], [s_mean], s=70, color=color, alpha=0.9,
                    marker="o", edgecolor="#333333", linewidth=0.8,
                    zorder=4)
        cid = label.split()[0]  # "C1", "C2", …
        axz.annotate(cid, (a_mean, s_mean), xytext=C_OFFSETS[arch],
                     textcoords="offset points", fontsize=9,
                     color="#333333", fontweight="bold")

    # T-sweep trajectory: connect T=3 → Track 2 (T=5) → T=10 → T=20.
    # Only draw if at least one phase63_track2_t* arch has data.
    t_sweep_pts = []
    if "agentic_txc_10_bare" in aucs and "agentic_txc_10_bare" in sems:
        t_sweep_pts.append((5, _mean_se(aucs["agentic_txc_10_bare"])[0],
                            _mean_se(sems["agentic_txc_10_bare"])[0]))
    for arch, (T, label) in T_SWEEP.items():
        if arch not in aucs or arch not in sems:
            continue
        a_mean, _ = _mean_se(aucs[arch])
        s_mean, _ = _mean_se(sems[arch])
        t_sweep_pts.append((T, a_mean, s_mean))
    t_sweep_pts.sort(key=lambda p: p[0])
    if len(t_sweep_pts) >= 2:
        xs = [p[1] for p in t_sweep_pts]
        ys = [p[2] for p in t_sweep_pts]
        axz.plot(xs, ys, "-", color="#08519c", lw=1.4, alpha=0.6, zorder=3)
        for arch, (T, label) in T_SWEEP.items():
            if arch not in aucs or arch not in sems:
                continue
            a_mean, _ = _mean_se(aucs[arch])
            s_mean, _ = _mean_se(sems[arch])
            axz.scatter([a_mean], [s_mean], s=70, color="#08519c",
                        alpha=0.9, marker="D",
                        edgecolor="black", linewidth=0.8, zorder=5)
            axz.annotate(label, (a_mean, s_mean), xytext=(6, 4),
                         textcoords="offset points", fontsize=8,
                         color="#08519c", fontweight="bold")

    axz.set_xlim(0.772, 0.813)
    if args.metric == "probe_acc_passage":
        axz.set_ylim(0.60, 0.90)
    elif args.metric == "semantic_count_pdvar":
        axz.set_ylim(-1.0, 15.0)
    else:
        axz.set_ylim(-0.6, 7.0)
    axz.set_title("TXC cluster (zoom) — Phase 6.1 arches +\n"
                  "Phase 6.2 ablation (C1–C6)", fontsize=10)
    axz.grid(alpha=0.3)
    axz.set_axisbelow(True)
    axz.tick_params(labelsize=9)
    axz.set_xlabel("probing mean AUC (mean_pool, k=5)", fontsize=9)
    axz.set_ylabel("SEMANTIC labels on concat_random  (/32)", fontsize=9)
    # Short legend for the C-codes on the right panel
    from matplotlib.patches import Patch
    c_legend = [
        Patch(color=color, label=name)
        for _, (name, color) in SECONDARY.items()
    ]
    axz.legend(handles=c_legend, loc="upper left",
               bbox_to_anchor=(1.02, 1.0), fontsize=7,
               framealpha=0.95, borderaxespad=0)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    thumb = Path(args.out).with_suffix(".thumb.png")
    fig.savefig(thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out} + {thumb}")


if __name__ == "__main__":
    main()
