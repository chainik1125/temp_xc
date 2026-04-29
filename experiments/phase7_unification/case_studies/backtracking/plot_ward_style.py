#!/usr/bin/env python3
"""Ward-et-al-2025-style line plots: keyword fraction (Wait/Hmm %) vs steering
magnitude, multiple lines for different feature/baseline directions.

Mirrors the paper's Fig 3 (per-direction bar/line) and Fig 4 (intervention
magnitude sweep with SEM error bars). Each curve = one steering direction
(an SAE feature, raw DoM, etc.).

Outputs:
  plots_summary/ward_fig4_lines.{png,thumb.png}
        Single panel: kw% vs α with all curves overlaid. Equivalent to
        Ward Fig 4 ('Effect of Interventions on Wait Token Percentage').

  plots_summary/ward_fig4_lines_zoomed.{png,thumb.png}
        Same but zoomed to α∈[0, 12] where coherence is mostly preserved
        across architectures.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.plotting.save_figure import save_figure  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    RESULTS_DIR,
)


# Curated list of (intervene_dir, mode, target, label, color, marker).
# Intentionally similar to Ward's Fig 4 baselines (backtracking direction,
# raw DoM, etc.) — we substitute the SAE-feature directions for the
# 'self/noise/deduction/initializing' baselines.
DEFAULT_LINES = [
    ("intervene_32x",                 "sae_additive", "feat_71839",  "Llama-Scope 32x feat_71839",          "tab:purple", "o"),
    ("intervene",                     "sae_additive", "feat_7792",   "Llama-Scope 8x feat_7792",            "tab:blue",   "s"),
    ("intervene_llama_topk_30k",      "sae_additive", "feat_28417",  "Llama-trained TopKSAE 30k",           "tab:orange", "^"),
    ("intervene_llama_txc_resid_30k", "sae_additive", "feat_5228",   "Llama-trained TXC@resid 30k",         "tab:red",    "D"),
    ("intervene_llama_txc_attn_30k_d8k_v2", "sae_additive", "feat_8013", "Llama-trained TXC@attn 30k (cross-hook)", "tab:cyan", "v"),
    ("intervene",                     "raw_dom",      "raw_dom",     "raw DoM (paper baseline)",            "black",      "x"),
]


def load_curve(intervene_dir: Path, mode: str, target: str):
    p = intervene_dir / "keyword_rates.csv"
    if not p.exists():
        return None
    rows = []
    with p.open() as f:
        for r in csv.DictReader(f):
            if r["mode"] != mode or r["target"] != target:
                continue
            try:
                a = float(r["magnitude"])
                kw = float(r["mean_keyword_fraction"])
                sem = float(r.get("sem_keyword_fraction") or 0.0)
                coh_raw = r.get("mean_coherence", "")
                coh = float(coh_raw) if coh_raw not in ("", None) else None
            except (ValueError, KeyError):
                continue
            rows.append({"alpha": a, "kw": kw, "sem": sem, "coh": coh})
    rows.sort(key=lambda r: r["alpha"])
    return rows


def plot_lines(curves, out_path: Path, *, alpha_max: float | None = None,
               show_coh_marker: bool = False, title: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, color, marker, pts in curves:
        if alpha_max is not None:
            pts = [r for r in pts if r["alpha"] <= alpha_max]
        if not pts:
            continue
        xs = [r["alpha"] for r in pts]
        ys = [r["kw"] * 100 for r in pts]  # convert to percentage like Ward
        es = [r["sem"] * 100 for r in pts]
        ax.errorbar(xs, ys, yerr=es, label=label,
                    color=color, marker=marker, linewidth=1.6, markersize=6,
                    capsize=3)
        if show_coh_marker:
            # Outline points where coherence dropped below 1.5 (model degenerate)
            for r in pts:
                if r["coh"] is not None and r["coh"] < 1.5:
                    ax.scatter(r["alpha"], r["kw"] * 100, s=130, facecolors="none",
                               edgecolors=color, linewidths=1.5, zorder=5)
    ax.set_xlabel("steering magnitude α (or clamp strength)")
    ax.set_ylabel(r"Wait/Hmm token fraction (%)  —  Ward et al. Eq. 1, B = {wait, hmm}")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Effect of intervention magnitude on backtracking-keyword fraction")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    if show_coh_marker:
        ax.text(0.98, 0.02,
                "○ outlined point = mean coherence < 1.5 (model degenerate at that α)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"[ward] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / "plots_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = []
    for sub, mode, target, label, color, marker in DEFAULT_LINES:
        d = RESULTS_DIR / sub
        pts = load_curve(d, mode, target)
        if not pts:
            print(f"[skip] no data for {label} ({sub} / {mode} / {target})")
            continue
        curves.append((label, color, marker, pts))

    plot_lines(curves, out_dir / "ward_fig4_lines.png", show_coh_marker=True,
               title="Wait/Hmm token fraction vs intervention magnitude (Ward 2025 Fig 4-style)")
    plot_lines(curves, out_dir / "ward_fig4_lines_zoomed.png", alpha_max=12.0,
               show_coh_marker=True,
               title="Wait/Hmm fraction vs α (zoom to α ≤ 12 — useful steering range)")


if __name__ == "__main__":
    main()
