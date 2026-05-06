"""Phase 2 + Phase 4 plotting — gAUC vs k_pos line plots.

Two outputs:
  - ``c2_txc_win_gauc_vs_k.png`` — Phase 2 ZOOM result on the winning
    (p_B, n_parents) datasource: 6 archs, gAUC vs k_pos, error bars
    over seeds. The headline figure for the hunt.
  - ``c2_headline_2panel.png``  — Phase 4 combined figure: ZOOM panel
    side-by-side with the hierarchical bench panel (read from the
    hierarchical leaderboard rows).

Reads ``results/leaderboard.jsonl``; filters by component=c2 +
hunt_phase=zoom (Phase 2) or bench=hierarchical (Phase 3).

Usage (from purified/):
    .venv/bin/python -m experiments.c2_synthetic_coupled.plot_headline
or
    .venv/bin/python -m experiments.c2_synthetic_coupled.plot_headline \\
        --winner-datasource toy_coupled_noisy_K10_M20_d256_pB05_np5
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

LEADERBOARD = Path("results/leaderboard.jsonl")
NOISY_PLOT_DIR = Path("experiments/c2_synthetic_coupled/plots")
HIER_PLOT_DIR = Path("experiments/c2_hierarchical/plots")

ARCH_COLORS = {
    "topk_sae":         ("#888888", "TopK-SAE", "o"),
    "stacked_sae_T2":   ("#9467bd", "Stacked-SAE T=2", "s"),
    "stacked_sae_T5":   ("#c5b0d5", "Stacked-SAE T=5", "s"),
    "txc_base":         ("#1f77b4", "TXC-base T=5", "^"),
    "txc_pro_T2":       ("#d62728", "TXC-pro T=2", "v"),
    "txc_pro_T5":       ("#ff9896", "TXC-pro T=5", "v"),
    "stacked_sae_default": ("#c5b0d5", "Stacked-SAE T=5", "s"),
    "txc_pro_default":     ("#ff9896", "TXC-pro T=5", "v"),
}

# Order in which we draw archs in the legend (top → bottom).
ARCH_ORDER = [
    "txc_pro_T5", "txc_pro_T2",
    "txc_base",
    "stacked_sae_T5", "stacked_sae_T2",
    "topk_sae",
]


def _arch_label(arch_name: str, t_label: str) -> str:
    """Map (arch, t_label) to one of ARCH_COLORS keys."""
    if arch_name == "topk_sae":
        return "topk_sae"
    if arch_name == "stacked_sae":
        return "stacked_sae_T2" if t_label == "T=2" else "stacked_sae_T5"
    if arch_name == "txc_base":
        return "txc_base"
    if arch_name == "txc_pro":
        if "T=2" in t_label or "T_max=2" in t_label:
            return "txc_pro_T2"
        return "txc_pro_T5"
    return arch_name


ZOOM_CUTOFF_TS = "2026-05-06T22:54:30Z"


def _load_panel(
    *,
    component: str = "c2",
    filter_fn,
    metric: str = "gauc",
) -> dict[str, dict[int, list[float]]]:
    """Group leaderboard rows by arch label → k_pos → list of metric over seeds.

    Dedupe by eval_key first — two subprocesses may compute the same
    (arch, seed, k_pos) on different GPUs (cache-collision); keep the
    latest row per eval_key so we don't artificially shrink error bars.

    For zoom rows (hunt_phase=zoom), only keep rows after
    ``ZOOM_CUTOFF_TS`` — the early zoom cells used n_steps=30000 which
    didn't have time to finish; the post-cutoff cells are at n_steps=
    8000 and form a coherent batch.
    """
    # Pass 1: dedupe by eval_key (latest wins).
    latest: dict[str, dict] = {}
    with LEADERBOARD.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["component"] != component:
                continue
            if not filter_fn(d):
                continue
            ec = d.get("eval_cfg") or {}
            if ec.get("smoke") is True:
                continue
            # For zoom rows, drop early n_steps=30k cells.
            if ec.get("hunt_phase") == "zoom" and d["ts"] < ZOOM_CUTOFF_TS:
                continue
            latest[d["eval_key"]] = d
    # Pass 2: group.
    by_arch_kpos: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in latest.values():
        ec = d.get("eval_cfg") or {}
        k_pos = ec.get("k_pos")
        t_label = ec.get("t_label", "default")
        arch_label = _arch_label(d["arch"], t_label)
        val = d["metrics"].get(metric)
        if val is None or k_pos is None:
            continue
        by_arch_kpos[arch_label][int(k_pos)].append(float(val))
    return by_arch_kpos


def _plot_panel(ax, data, title: str):
    for arch_label in ARCH_ORDER:
        if arch_label not in data:
            continue
        kpos_list = sorted(data[arch_label].keys())
        means = [mean(data[arch_label][k]) for k in kpos_list]
        stds = [
            stdev(data[arch_label][k]) if len(data[arch_label][k]) > 1 else 0.0
            for k in kpos_list
        ]
        color, lab, marker = ARCH_COLORS.get(arch_label, ("#000", arch_label, "o"))
        ax.errorbar(
            kpos_list, means, yerr=stds, label=lab, color=color,
            marker=marker, markersize=6, linewidth=1.6, capsize=3,
            elinewidth=0.8, alpha=0.95,
        )
    ax.set_xlabel("k_pos (per-token TopK)")
    ax.set_ylabel("gAUC (global feature recovery)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)


def render_zoom(winner_ds: str, out_path: Path):
    NOISY_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    filter_fn = lambda d: (
        d.get("datasource") == winner_ds
        and (d.get("eval_cfg") or {}).get("hunt_phase") == "zoom"
    )
    gauc_data = _load_panel(filter_fn=filter_fn, metric="gauc")
    eauc_data = _load_panel(filter_fn=filter_fn, metric="eauc")
    if not gauc_data:
        print(f"[plot] no zoom data yet for {winner_ds}")
        return None
    # Build descriptive title from the datasource fields embedded in any row.
    p_B = None
    n_par = None
    rho = None
    with LEADERBOARD.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("datasource") == winner_ds:
                ec = d.get("eval_cfg") or {}
                p_B = ec.get("p_B")
                n_par = ec.get("n_parents")
                rho = ec.get("rho")
                if p_B is not None:
                    break
    # Two-panel: gAUC + eAUC side by side (global vs local divide).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    title_g = (f"Global recovery (gAUC) — p_B={p_B}, n_parents={n_par}, ρ={rho}\n"
               f"6 archs × 3 seeds × 8 k_pos")
    _plot_panel(axes[0], gauc_data, title_g)
    title_e = "Local recovery (eAUC) — same data"
    _plot_panel(axes[1], eauc_data, title_e)
    axes[1].set_ylabel("eAUC (local emission recovery)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".thumb.png"), dpi=64, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return gauc_data


def render_hierarchical(out_path: Path,
                        datasource: str = "toy_hierarchical_Kg10_Kl30_d256"):
    HIER_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    filter_fn = lambda d: (
        d.get("datasource") == datasource
        and (d.get("eval_cfg") or {}).get("bench") == "hierarchical"
    )
    gauc_data = _load_panel(filter_fn=filter_fn, metric="gauc")
    eauc_data = _load_panel(filter_fn=filter_fn, metric="eauc")
    if not gauc_data:
        print(f"[plot] no hierarchical data yet for {datasource}")
        return None
    # Two-panel: gAUC + eAUC. The hierarchical bench is engineered for
    # the divide; show both axes.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    title_g = ("Global recovery (gAUC) — 10 slow globals × 30 fast locals\n"
               "6 archs × 3 seeds × 7 k_pos (n_steps=20k)")
    _plot_panel(axes[0], gauc_data, title_g)
    title_e = "Local recovery (eAUC) — same data"
    _plot_panel(axes[1], eauc_data, title_e)
    axes[1].set_ylabel("eAUC (local emission recovery)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".thumb.png"), dpi=64, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return gauc_data


def render_2panel_headline(
    *,
    winner_ds: str,
    hier_ds: str = "toy_hierarchical_Kg10_Kl30_d256",
    out_path: Path,
):
    """Phase 4 — combined headline figure for the paper."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    # Left: zoom on noisy + overlap (gAUC only).
    zoom_data = _load_panel(
        filter_fn=lambda d: (
            d.get("datasource") == winner_ds
            and (d.get("eval_cfg") or {}).get("hunt_phase") == "zoom"
        ),
        metric="gauc",
    )
    p_B, n_par, rho = None, None, None
    with LEADERBOARD.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("datasource") == winner_ds:
                ec = d.get("eval_cfg") or {}
                p_B, n_par, rho = ec.get("p_B"), ec.get("n_parents"), ec.get("rho")
                if p_B is not None:
                    break
    title_left = f"Noisy + overlap (Dmitry-style)\np_B={p_B}, n_parents={n_par}, ρ={rho}"
    _plot_panel(axes[0], zoom_data or {}, title_left)

    # Right: hierarchical (gAUC only).
    hier_data = _load_panel(
        filter_fn=lambda d: (
            d.get("datasource") == hier_ds
            and (d.get("eval_cfg") or {}).get("bench") == "hierarchical"
        ),
        metric="gauc",
    )
    title_right = "Hierarchical (engineered)\n10 slow globals × 30 fast locals"
    _plot_panel(axes[1], hier_data or {}, title_right)

    fig.suptitle("TXC dictionaries recover global features that per-token SAEs miss",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".thumb.png"), dpi=64, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner-datasource", default=None,
                    help="Phase 2 winner ds. If omitted, read from "
                         "hunt_summary.json.")
    ap.add_argument("--hier-datasource", default="toy_hierarchical_Kg10_Kl30_d256")
    ap.add_argument("--zoom-only", action="store_true")
    ap.add_argument("--hier-only", action="store_true")
    ap.add_argument("--two-panel-only", action="store_true")
    args = ap.parse_args()

    winner = args.winner_datasource
    if winner is None:
        summary = Path("experiments/c2_synthetic_coupled/hunt_summary.json")
        if summary.exists():
            winner = json.loads(summary.read_text()).get("overall_winner_datasource")

    # Render zoom plots for both top regimes if we have data for them.
    secondary = "toy_coupled_noisy_K10_M20_d256_pB05_np10"

    if not args.hier_only and not args.two_panel_only:
        if winner:
            render_zoom(
                winner,
                NOISY_PLOT_DIR / "c2_txc_win_gauc_vs_k.png",
            )
        # Also render the pB05_np10 zoom (most-robust-gap regime).
        render_zoom(
            secondary,
            NOISY_PLOT_DIR / "c2_txc_win_gauc_vs_k_np10.png",
        )

    if not args.zoom_only and not args.two_panel_only:
        render_hierarchical(HIER_PLOT_DIR / "c2_hierarchical_gauc_vs_k.png",
                            datasource=args.hier_datasource)

    if not args.zoom_only and not args.hier_only and winner:
        render_2panel_headline(
            winner_ds=winner,
            hier_ds=args.hier_datasource,
            out_path=NOISY_PLOT_DIR / "c2_headline_2panel.png",
        )
        # Also render the alternate 2-panel using pB05_np10 left.
        render_2panel_headline(
            winner_ds=secondary,
            hier_ds=args.hier_datasource,
            out_path=NOISY_PLOT_DIR / "c2_headline_2panel_np10.png",
        )


if __name__ == "__main__":
    main()
