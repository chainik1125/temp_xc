"""BTK re-run analysis — d(perf)/dT per arm, the re-run gate figure.

Consumes leaderboard rows produced by ``driver.py`` (the uniform
``eval_window_L == 40`` protocol; archs ``txc_base`` = paper-match
composition vs ``txc_base_btk`` = btk-only). Produces:

1. ``btk_rerun_dperf_dT.{png,pdf}`` — per bench x metric: metric vs T
   (log2 x), one line per (arm, k_pos in {1,2}), seeds mean ± std;
   per-arm slope d(metric)/d log2 T in the legend (least squares over
   pooled non-degenerate cells, k_pos in {1,2,5}); frozen tsae reference
   overlaid at its T=1 value (gauc/eauc only — decoder-direction
   metrics, protocol-stable; NEVER nmse, which is eval-window-bound).
2. ``btk_rerun_fingerprint.{png,pdf}`` — realized l0_per_window /
   nominal k_win vs T per arm: the mixing fingerprint.
3. ``btk_rerun_summary.json`` — slopes, per-cell table, clipped-cell
   mask, baseline references, verdict inputs.

Degenerate cells: k_pos*T >= d_sae=20 clips the paper budget in BOTH
arms (dense code). Clipped cells are excluded from slopes and marked x.

Colors: black = paper-match, Okabe-Ito blue = btk-only, Okabe-Ito
orange = frozen tsae reference (palette CVD-validated). k_pos carries a
linestyle (solid k=1, dashed k=2), never a color.

Usage:
    python -m experiments.explorations.btk_rerun.analysis [--out-dir plots/]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# txc_base_btk (v1.0.0, pre-convention name) and txc_base_btkonly
# (v1.1.0, canonical) share a bit-identical TRAINING path; the rename +
# threshold-flag/EMA changes touch only the eval encode path, which
# gauc/eauc never use. Rows from both names fold into the btk-only arm.
ARM_OF = {"txc_base": "paper-match", "txc_base_btk": "btk-only",
          "txc_base_btkonly": "btk-only", "txc_base_relumix": "relu-mix"}
ARM_LABEL = {"paper-match": "paper-match (per-window TopK→ReLU)",
             "btk-only": "btk-only (batch pool, no ReLU)",
             "relu-mix": "relu-mix control (ReLU→batch pool)"}
ARM_COLOR = {"paper-match": "#000000", "btk-only": "#0072B2",
             "relu-mix": "#009E73"}
BASE_COLOR = "#E69F00"          # frozen tsae reference
K_STYLE = {1: "-", 2: "--"}     # k_pos plotted in the headline figure
K_SLOPE = (1, 2, 5)            # k_pos pooled into slopes (non-clipped only)
METRICS = ["gauc", "eauc", "nmse"]
D_SAE_SYNTH = 20
EVAL_L = 40
BASELINE_CUTOVER = "2026-05-31T22:30:00Z"


def _ov(r: dict) -> dict:
    return (r.get("training_cfg") or {}).get("arch_hparams_override") or {}


def cell_T(r: dict) -> int:
    return int(_ov(r).get("T", 5))


def cell_k(r: dict) -> int:
    return int(_ov(r).get("k_pos", (r.get("eval_cfg") or {}).get("k_pos")))


def cell_dsae(r: dict) -> int:
    return int(_ov(r).get("d_sae", D_SAE_SYNTH))


def is_clipped(k_pos: int, T: int, d_sae: int = D_SAE_SYNTH) -> bool:
    return k_pos * T >= d_sae


def load_rows(leaderboard: Path) -> list[dict]:
    rows = []
    for line in leaderboard.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("experiment") != "synthetic":
            continue
        if r.get("arch") not in ARM_OF:
            continue
        ec = r.get("eval_cfg") or {}
        if ec.get("smoke") or ec.get("eval_window_L") != EVAL_L:
            continue
        rows.append(r)
    return rows


def load_baseline_refs(leaderboard: Path) -> dict:
    """(bench, k_pos) -> {metric: mean} for frozen tsae rows (T=1).

    gauc/eauc only — decoder-direction metrics documented as stable
    across the 1.1→1.3 protocol bumps. Baselines were NOT rerun; they
    had n_steps=10_000 (vs the sweep's 6_000), which favors them.
    """
    acc = defaultdict(lambda: defaultdict(list))
    for line in leaderboard.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("experiment") != "synthetic" or r.get("arch") != "tsae"
                or (r.get("eval_cfg") or {}).get("smoke")
                or r.get("ts", "") < BASELINE_CUTOVER
                or (r.get("training_cfg") or {}).get("n_steps") != 10_000):
            continue
        k = cell_k(r)
        for m in ("gauc", "eauc"):
            v = (r.get("metrics") or {}).get(m)
            if v is not None:
                acc[(r["datasource"], k)][m].append(float(v))
    return {k: {m: float(np.mean(v)) for m, v in md.items()}
            for k, md in acc.items()}


def aggregate(rows: list[dict]):
    # Within the btk-only arm, prefer canonical txc_base_btkonly rows over
    # pre-convention txc_base_btk at the same cell: gauc/eauc are identical
    # (verified max |Δ| 0.004 over 108 matched cells) but nmse/l0 follow
    # the eval threshold convention, which the canonical rename changed.
    best: dict = {}
    for r in rows:
        cell = (r["datasource"], ARM_OF[r["arch"]], cell_k(r), cell_T(r),
                cell_dsae(r), r["seed"])
        cur = best.get(cell)
        if cur is None or (cur["arch"] == "txc_base_btk"
                           and r["arch"] == "txc_base_btkonly"):
            best[cell] = r
    acc = defaultdict(lambda: defaultdict(list))
    for (ds, arm, k, t, dsae, _seed), r in best.items():
        for m, v in (r.get("metrics") or {}).items():
            acc[(ds, arm, k, t, dsae)][m].append(float(v))
    return {key: {m: (float(np.mean(v)), float(np.std(v)), len(v))
                  for m, v in md.items()}
            for key, md in acc.items()}


def slope_d_dlogT(points: list[tuple[int, float]]) -> float | None:
    pts = [(math.log2(t), v) for t, v in points if not math.isnan(v)]
    if len(pts) < 3:
        return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    if np.allclose(x.var(), 0):
        return None
    return float(np.polyfit(x, y, 1)[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--leaderboard", type=Path,
                   default=Path("results/leaderboard.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("plots/btk_rerun"))
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(args.leaderboard)
    refs = load_baseline_refs(args.leaderboard)
    print(f"[analysis] {len(rows)} canonical rows (eval_window_L={EVAL_L}); "
          f"{len(refs)} baseline ref cells")
    agg_all = aggregate(rows)
    # Headline figure = paper d_sae (20). The d_sae=50 wing feeds
    # summary["slopes"] under a |d50 suffix only.
    agg = {(b, a, k, t): md for (b, a, k, t, ds), md in agg_all.items()
           if ds == D_SAE_SYNTH}
    wing = {(b, a, k, t): md for (b, a, k, t, ds), md in agg_all.items()
            if ds != D_SAE_SYNTH}
    benches = sorted({k[0] for k in agg})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"n_rows": len(rows), "slopes": {}, "cells": {},
                     "baseline_refs": {f"{b}|k{k}": v
                                       for (b, k), v in sorted(refs.items(),
                                                               key=str)}}

    # ── Fig 1: metric vs T ──
    fig, axes = plt.subplots(len(METRICS), max(len(benches), 1),
                             figsize=(6.4 * max(len(benches), 1),
                                      3.9 * len(METRICS)), squeeze=False)
    for col, bench in enumerate(benches):
        for mrow, metric in enumerate(METRICS):
            ax = axes[mrow][col]
            handles, labels = [], []
            for arch in ARM_LABEL:
                pooled = []
                for k_pos in K_SLOPE:
                    pts = []
                    for (b, a, kk, T), md in agg.items():
                        if ((b, a, kk) != (bench, arch, k_pos)
                                or metric not in md):
                            continue
                        pts.append((T, *md[metric][:2],
                                    is_clipped(kk, T)))
                    pts.sort()
                    pooled += [(t, m) for t, m, _, c in pts if not c]
                    if k_pos not in K_STYLE or not pts:
                        continue
                    Ts = [t for t, *_ in pts]
                    mu = [m for _, m, _, _ in pts]
                    sd = [s for _, _, s, _ in pts]
                    (ln, _, _) = ax.errorbar(
                        Ts, mu, yerr=sd, color=ARM_COLOR[arch],
                        linestyle=K_STYLE[k_pos], marker="o",
                        markersize=3.5, linewidth=1.6, capsize=2)
                    for t, m, c in zip(Ts, mu, [c for *_, c in pts]):
                        if c:
                            ax.plot([t], [m], marker="x", color="#888888",
                                    markersize=8, zorder=5)
                s = slope_d_dlogT(pooled)
                summary["slopes"][f"{bench}|{metric}|{arch}"] = s
                stxt = "n/a" if s is None else f"{s:+.3f}"
                handles.append(plt.Line2D([], [], color=ARM_COLOR[arch],
                                          linewidth=1.8))
                labels.append(f"{ARM_LABEL[arch]} — slope {stxt}")
            if metric in ("gauc", "eauc"):
                for k_pos, ls in K_STYLE.items():
                    ref = refs.get((bench, k_pos), {}).get(metric)
                    if ref is None:
                        continue
                    ax.axhline(ref, color=BASE_COLOR, linestyle=ls,
                               linewidth=1.2, alpha=0.9)
                if any((bench, k) in refs for k in K_STYLE):
                    handles.append(plt.Line2D([], [], color=BASE_COLOR,
                                              linewidth=1.4))
                    labels.append("tsae frozen ref (T=1, 10k steps)")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 5, 8, 10])
            ax.set_xticklabels(["1", "2", "4", "5", "8", "10"])
            ax.set_xlabel("window size T")
            ax.set_ylabel(metric + (" (lower is better)"
                                    if metric == "nmse" else ""))
            ax.set_title(f"{bench.replace('toy_', '')} — {metric}",
                         fontsize=10)
            ax.grid(alpha=0.2)
            if mrow == 0 and col == 0:
                handles.append(plt.Line2D([], [], color="#555", linestyle="-"))
                labels.append("k_pos = 1 (solid) / 2 (dashed)")
                handles.append(plt.Line2D([], [], color="#888888", marker="x",
                                          linestyle="none"))
                labels.append("clipped cell (k·T ≥ d_sae) — not in slopes")
            ax.legend(handles, labels, fontsize=7, loc="best", framealpha=0.9)
    fig.suptitle(
        "Paper TXC, composite vs btk-only: performance vs window size T\n"
        "seeds mean±std; slopes pooled over non-clipped k∈{1,2,5} cells",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"btk_rerun_dperf_dT.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: mixing fingerprint ──
    fig, axes = plt.subplots(1, len(benches),
                             figsize=(6.4 * len(benches), 3.9), squeeze=False)
    for col, bench in enumerate(benches):
        ax = axes[0][col]
        for arch in ARM_LABEL:
            pts = defaultdict(list)
            for (b, a, kk, T), md in agg.items():
                if b != bench or a != arch or "l0_per_window" not in md:
                    continue
                nominal = min(kk * T, D_SAE_SYNTH)
                pts[T].append(md["l0_per_window"][0] / max(nominal, 1))
            Ts = sorted(pts)
            if Ts:
                ax.plot(Ts, [float(np.mean(pts[t])) for t in Ts],
                        color=ARM_COLOR[arch], marker="o", linewidth=1.6,
                        label=ARM_LABEL[arch])
        ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 5, 8, 10])
        ax.set_xticklabels(["1", "2", "4", "5", "8", "10"])
        ax.set_xlabel("window size T")
        ax.set_ylabel("realized l0 / nominal k_win")
        ax.set_title(f"{bench.replace('toy_', '')} — mixing fingerprint",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"btk_rerun_fingerprint.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)

    # Wing slopes (d_sae=50): de-clipped grid, all metrics.
    for metric in METRICS:
        for bench in sorted({k[0] for k in wing}):
            for arch in ARM_LABEL:
                pooled = [(t, md[metric][0])
                          for (b, a, k, t), md in wing.items()
                          if b == bench and a == arch and metric in md
                          and not is_clipped(k, t, 50)]
                s_w = slope_d_dlogT(pooled)
                if s_w is not None:
                    summary["slopes"][f"{bench}|{metric}|{arch}|d50"] = s_w

    for key, md in sorted(agg.items(), key=str):
        b, a, k, t = key
        summary["cells"][f"{b}|{a}|k{k}|T{t}"] = {
            "clipped": is_clipped(k, t),
            **{m: {"mean": v[0], "std": v[1], "n": v[2]}
               for m, v in md.items()},
        }

    # Arm deltas vs paper-match at matched (bench, k, T) — the anchor-free
    # view of whether the gap closes as T grows.
    summary["deltas_vs_paper_match"] = {}
    for (b, a, k, t), md in sorted(agg.items(), key=str):
        if a == "paper-match":
            continue
        ref = agg.get((b, "paper-match", k, t))  # d_sae=20 lane only
        if not ref:
            continue
        for m in ("gauc", "eauc", "nmse"):
            if m in md and m in ref:
                summary["deltas_vs_paper_match"][
                    f"{b}|{a}|{m}|k{k}|T{t}"
                ] = round(md[m][0] - ref[m][0], 4)
    (args.out_dir / "btk_rerun_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"[analysis] figures + summary -> {args.out_dir}")
    for bm, s in sorted(summary["slopes"].items()):
        print(f"  slope {bm}: {s if s is None else round(s, 4)}")


if __name__ == "__main__":
    main()
