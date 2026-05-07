"""Render paper-ready C6 emergent-misalignment plots in c7-paper style.

Two plots per organism (only 7B-medical implemented for the Overleaf
snippet today):

  c6_em_alignment_delta_<org>.{png,pdf}  — bar chart of $\\Delta\\text{align}$
                                            per arch, computed as
                                            $\\max\\text{align} - \\min\\text{align}$
                                            over all (finalist, $\\alpha$) cells
                                            with mean coherence ≥ 70 in
                                            each cell's wang_full.json
                                            stage-4 frontier.
  c6_em_detection_prauc_<org>.{png,pdf}   — bar chart of sparse-probe
                                            PR-AUC at $S=16$ per arch,
                                            from leaderboard rows with
                                            eval_protocol_version=3.0.0.

Both plots: bars = mean across the 2 paired seeds {1,42}; error bars =
min/max range across seeds.

Aggregation matches experiments/c6_em/analysis.py on origin/final:
  - canonical (arch, datasource, seed) cells for c6
  - 7B-medical = ``qwen_2_5_7b_instruct_medical_l15_resid_post``
  - HEADLINE_PROTOCOL = "2.0.0" (full Wang) — drives wang_full.json input
  - DETECTION_PROTOCOL = "3.0.0", DETECTION_S_HEADLINE = 16

Usage (defaults resolve to in-repo canonical paths)::

    cd purified
    .venv/bin/python -m scripts.c6_paper_renderer
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# c7-paper style.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
})

PAPER_ARCH_COLOR = {
    "sae_arditi": "#355C7D",  # deep slate blue (per-token SAE family,
                              # distinct from topk_sae #4C72B0)
    "txc_base":   "#8172B2",  # purple (matches c3/c7 conventions)
    "tsae_paper": "#55A868",  # green
    "txc_pro":    "#CCB974",  # gold
}

ARCH_DISPLAY = {
    "sae_arditi": "SAE-arditi",
    "txc_base":   "TXC-base",
    "tsae_paper": "T-SAE",
    "txc_pro":    "TXC-pro",
}

# Plot order (left → right). Matches main.tex caption.
ARCH_ORDER = ("sae_arditi", "txc_base", "tsae_paper", "txc_pro")

# 7B-medical canonical datasource, eval-protocol versions.
ORG_TAG = "7bmed"
ORG_DATASOURCE = "qwen_2_5_7b_instruct_medical_l15_resid_post"
HEADLINE_PROTOCOL = "2.0.0"   # full Wang (wang_full.json source)
DETECTION_PROTOCOL = "3.0.0"  # sparse-probe PR-AUC
DETECTION_S = 16
COH_THRESHOLD = 70.0


def _save_png_pdf(fig, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))


def load_rows(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def alignment_delta_per_seed(wang: dict, *, coh_threshold: float) -> float | None:
    """Δalign = max(mean_align) − min(mean_align) over (finalist, α)
    cells with mean_coh ≥ ``coh_threshold``. Reads stage-4 finalists."""
    rows: list[dict] = []
    for fin in wang.get("stage4", {}).get("finalists", []):
        rows.extend(fin.get("rows", []))
    valid = [r for r in rows
             if r.get("mean_align") is not None
             and r.get("mean_coh") is not None]
    kept = [r for r in valid if r["mean_coh"] >= coh_threshold]
    if not kept:
        return None
    aligns = [r["mean_align"] for r in kept]
    return float(max(aligns) - min(aligns))


def find_wang_path(wang_dir: Path, train_key: str) -> Path | None:
    """Look for wang_<train_key>.json (renderer convention) or
    c6_<train_key>/wang_full.json (origin/final on-disk layout)."""
    p1 = wang_dir / f"wang_{train_key}.json"
    if p1.exists():
        return p1
    p2 = wang_dir / f"c6_{train_key}" / "wang_full.json"
    if p2.exists():
        return p2
    return None


def collect_alignment_delta(rows: list[dict], wang_dir: Path,
                            *, organism_datasource: str
                            ) -> dict[str, list[float]]:
    """Per-arch list of Δalign values (one per seed) for the given
    organism. Reads each cell's wang_full.json from ``wang_dir``."""
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("component") != "c6":
            continue
        if r.get("eval_protocol_version") != HEADLINE_PROTOCOL:
            continue
        if r.get("datasource") != organism_datasource:
            continue
        arch = r.get("arch")
        if arch not in PAPER_ARCH_COLOR:
            continue
        wang_path = find_wang_path(wang_dir, r["train_key"])
        if wang_path is None:
            print(f"[c6_paper] missing wang_full for {arch} seed={r.get('seed')} "
                  f"train_key={r['train_key']}")
            continue
        wang = json.loads(wang_path.read_text())
        d = alignment_delta_per_seed(wang, coh_threshold=COH_THRESHOLD)
        if d is not None:
            out[arch].append(d)
    return out


def collect_detection_prauc(rows: list[dict], *, organism_datasource: str
                            ) -> dict[str, list[float]]:
    """Per-arch list of PR-AUC@S=DETECTION_S values (one per seed)."""
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("component") != "c6":
            continue
        if r.get("eval_protocol_version") != DETECTION_PROTOCOL:
            continue
        if r.get("datasource") != organism_datasource:
            continue
        arch = r.get("arch")
        if arch not in PAPER_ARCH_COLOR:
            continue
        v = r.get("metrics", {}).get(f"pr_auc_S{DETECTION_S}")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[arch].append(float(v))
    return out


def _bar_chart(per_arch: dict[str, list[float]], *,
               out_stem: Path, ylabel: str, ymin: float | None = None,
               ymax: float | None = None,
               annotate_decimals: int = 2,
               value_offset_frac: float = 0.04) -> None:
    """Single-panel bar chart, c7-paper style. Bars = mean across seeds,
    error bars = min/max range. Numeric value annotated above each bar."""
    archs = [a for a in ARCH_ORDER if per_arch.get(a)]
    means = [float(mean(per_arch[a])) for a in archs]
    lows = [min(per_arch[a]) for a in archs]
    highs = [max(per_arch[a]) for a in archs]
    err_lo = [m - lo for m, lo in zip(means, lows)]
    err_hi = [hi - m for m, hi in zip(means, highs)]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(archs))
    bars = ax.bar(
        x, means,
        yerr=[err_lo, err_hi],
        color=[PAPER_ARCH_COLOR[a] for a in archs],
        edgecolor="black", linewidth=0.7, alpha=0.92,
        error_kw=dict(ecolor="#222", capsize=4, elinewidth=1.0),
        width=0.6,
    )

    # Axis range. If user passed (ymin, ymax) honor it; otherwise auto.
    if ymin is None:
        lo_seen = min(lows) if lows else 0.0
        ymin = max(0.0, lo_seen - 0.1 * max(abs(lo_seen), 1.0))
    if ymax is None:
        hi_seen = max(highs) if highs else 1.0
        ymax = hi_seen + 0.18 * max(abs(hi_seen), 1.0)
    ax.set_ylim(ymin, ymax)

    # Annotate value above each bar.
    fmt = f"{{:.{annotate_decimals}f}}"
    yspan = ymax - ymin
    for i, (m, hi) in enumerate(zip(means, highs)):
        ax.text(i, hi + value_offset_frac * yspan,
                fmt.format(m),
                ha="center", va="bottom",
                fontsize=10, color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_DISPLAY.get(a, a) for a in archs])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", which="both", bottom=False, length=0)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)

    fig.tight_layout()
    _save_png_pdf(fig, out_stem)
    plt.close(fig)


def main(*, leaderboard: Path, wang_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(leaderboard)

    # Δalign per (arch, seed) for 7B-medical.
    align_delta = collect_alignment_delta(
        rows, wang_dir, organism_datasource=ORG_DATASOURCE,
    )
    if not align_delta:
        print(f"[c6_paper] no full-Wang rows for {ORG_DATASOURCE}")
    else:
        out_stem = output_dir / f"c6_em_alignment_delta_{ORG_TAG}"
        _bar_chart(
            align_delta, out_stem=out_stem,
            ylabel=r"$\Delta\,\mathrm{align}$ "
                   rf"(coh $\geq$ {int(COH_THRESHOLD)})",
            annotate_decimals=2,
        )
        print("[c6_paper] alignment_delta per arch (mean, min, max):")
        for arch in ARCH_ORDER:
            if arch in align_delta:
                vs = align_delta[arch]
                print(f"  {arch:12s} n={len(vs)}  mean={mean(vs):.2f}  "
                      f"range=[{min(vs):.2f}..{max(vs):.2f}]")

    # Detection PR-AUC@S=16 per (arch, seed) for 7B-medical.
    detect = collect_detection_prauc(rows, organism_datasource=ORG_DATASOURCE)
    if not detect:
        print(f"[c6_paper] no detection (3.0.0) rows for {ORG_DATASOURCE}")
    else:
        out_stem = output_dir / f"c6_em_detection_prauc_{ORG_TAG}"
        _bar_chart(
            detect, out_stem=out_stem,
            ylabel=rf"Sparse-probe PR-AUC at $S\!=\!{DETECTION_S}$",
            ymin=0.0, ymax=1.0,
            annotate_decimals=3,
        )
        print(f"[c6_paper] detection PR-AUC@S={DETECTION_S} per arch:")
        for arch in ARCH_ORDER:
            if arch in detect:
                vs = detect[arch]
                print(f"  {arch:12s} n={len(vs)}  mean={mean(vs):.3f}  "
                      f"range=[{min(vs):.3f}..{max(vs):.3f}]")


def _purified_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cli() -> None:
    root = _purified_root()
    ap = argparse.ArgumentParser(description=(
        "C6 (emergent misalignment) paper figure renderer. "
        "Defaults to in-repo canonical paths."
    ))
    ap.add_argument(
        "--leaderboard", type=Path,
        default=root / "results" / "leaderboard.jsonl",
        help="Leaderboard jsonl (default: results/leaderboard.jsonl).",
    )
    ap.add_argument(
        "--wang-dir", type=Path,
        default=root / "results" / "runs",
        help="Directory containing wang_full.json files (default: results/runs/). "
             "Either as wang_<train_key>.json (script convention) or as "
             "c6_<train_key>/wang_full.json (run-dir layout).",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=root / "figs",
        help=("Output directory (default: figs/, matching the "
              "paper's \\includegraphics{figs/c6_em_*} paths)."),
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(
        leaderboard=args.leaderboard,
        wang_dir=args.wang_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cli()
