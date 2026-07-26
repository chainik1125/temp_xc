"""BTK re-run analysis — d(perf)/dT per arm, the re-run gate figure.

Consumes leaderboard rows produced by ``driver.py`` (the uniform
``eval_window_L == 40`` protocol; archs ``txc_base`` = paper-match
composition vs ``txc_base_btk`` = btk-only). Produces:

1. ``btk_rerun_dperf_dT.{png,pdf}`` — per bench x metric: metric vs T
   (log2 x-axis), one line per (arm, k_pos), seeds aggregated mean ± std;
   per-arm slope d(metric)/d log2 T annotated (least squares over the
   pooled non-degenerate k_pos cells).
2. ``btk_rerun_fingerprint.{png,pdf}`` — realized l0_per_window vs T and
   (btk-only) neg_frac provenance: the mixing fingerprint required by
   the ACTMIX card discipline.
3. ``btk_rerun_summary.json`` — slopes, per-cell table, degenerate-cell
   mask (k_win clipped to d_sae), verdict inputs.

Degenerate cells: at synthetic d_sae=20 the paper budget k_win=k_pos*T
clips whenever k_pos*T >= 20 — both arms clip identically but the cell
no longer distinguishes selection rules (dense code). Slopes exclude
clipped cells; the figure greys them out.

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

ARM_LABEL = {"txc_base": "paper-match (TopK→ReLU)",
             "txc_base_btk": "btk-only (BatchTopK, no ReLU)"}
ARM_COLOR = {"txc_base": "#000000", "txc_base_btk": "#1f77b4"}
BENCH_METRICS = {
    "toy_coupled_K10_M20_d256": ["gauc", "eauc", "nmse"],
    "toy_markov_n20_d40_noisy": ["gauc", "eauc", "nmse"],
}
D_SAE_SYNTH = 20
EVAL_L = 40


def load_rows(leaderboard: Path) -> list[dict]:
    rows = []
    for line in leaderboard.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("experiment") != "synthetic":
            continue
        if r.get("arch") not in ARM_LABEL:
            continue
        ec = r.get("eval_cfg") or {}
        if ec.get("smoke") or ec.get("eval_window_L") != EVAL_L:
            continue
        rows.append(r)
    return rows


def cell_T(r: dict) -> int:
    ov = (r.get("training_cfg") or {}).get("arch_hparams_override") or {}
    return int(ov.get("T", 5))


def cell_k(r: dict) -> int:
    ov = (r.get("training_cfg") or {}).get("arch_hparams_override") or {}
    return int(ov.get("k_pos", (r.get("eval_cfg") or {}).get("k_pos")))


def is_clipped(k_pos: int, T: int) -> bool:
    return k_pos * T >= D_SAE_SYNTH


def aggregate(rows: list[dict]):
    """(bench, arch, k_pos, T) -> {metric: (mean, std, n)}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["datasource"], r["arch"], cell_k(r), cell_T(r))
        for m, v in (r.get("metrics") or {}).items():
            acc[key][m].append(float(v))
    out = {}
    for key, md in acc.items():
        out[key] = {m: (float(np.mean(v)), float(np.std(v)), len(v))
                    for m, v in md.items()}
    return out


def slope_d_dlogT(points: list[tuple[int, float]]) -> float | None:
    """Least-squares slope of metric vs log2 T."""
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
    print(f"[analysis] {len(rows)} canonical rows (eval_window_L={EVAL_L})")
    agg = aggregate(rows)
    benches = sorted({k[0] for k in agg})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"n_rows": len(rows), "slopes": {}, "cells": {}}

    # ── Fig 1: metric vs T per (bench x metric), lines per (arm, k_pos) ──
    metrics = ["gauc", "eauc", "nmse"]
    fig, axes = plt.subplots(len(metrics), max(len(benches), 1),
                             figsize=(6.5 * max(len(benches), 1),
                                      4.2 * len(metrics)),
                             squeeze=False)
    for col, bench in enumerate(benches):
        for mrow, metric in enumerate(metrics):
            ax = axes[mrow][col]
            for arch in ARM_LABEL:
                ks = sorted({k[2] for k in agg if k[0] == bench and k[1] == arch})
                pooled = []
                for k_pos in ks:
                    pts = []
                    for (b, a, kk, T), md in agg.items():
                        if (b, a, kk) != (bench, arch, k_pos) or metric not in md:
                            continue
                        pts.append((T, md[metric][0], md[metric][1],
                                    is_clipped(kk, T)))
                    if not pts:
                        continue
                    pts.sort()
                    Ts = [t for t, *_ in pts]
                    mu = [m for _, m, _, _ in pts]
                    sd = [s for _, _, s, _ in pts]
                    clip = [c for *_, c in pts]
                    alpha = 0.25 + 0.75 * (ks.index(k_pos) + 1) / len(ks)
                    ax.errorbar(Ts, mu, yerr=sd, color=ARM_COLOR[arch],
                                alpha=alpha, marker="o", markersize=3.5,
                                linewidth=1.4, capsize=2,
                                label=(f"{ARM_LABEL[arch]}" if k_pos == ks[0]
                                       else None))
                    for t, m, c in zip(Ts, mu, clip):
                        if c:
                            ax.plot([t], [m], marker="x", color="#999",
                                    markersize=7, zorder=5)
                    pooled += [(t, m) for t, m, _, c in
                               [(t, m, s, c) for t, m, s, c in pts] if not c]
                s = slope_d_dlogT(pooled)
                summary["slopes"][f"{bench}|{metric}|{arch}"] = s
                if s is not None:
                    ax.annotate(
                        f"{ARM_LABEL[arch].split(' ')[0]}: "
                        f"d/dlog2T={s:+.3f}",
                        xy=(0.02, 0.94 - 0.08 * list(ARM_LABEL).index(arch)),
                        xycoords="axes fraction", fontsize=8,
                        color=ARM_COLOR[arch])
            ax.set_xscale("log", base=2)
            ax.set_xlabel("window size T")
            ax.set_ylabel(metric)
            ax.set_title(f"{bench.replace('toy_', '')} — {metric}"
                         f"{' (lower better)' if metric == 'nmse' else ''}",
                         fontsize=10)
            ax.grid(alpha=0.25)
            if mrow == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Paper arch, composite vs btk-only — performance vs window size T\n"
        "(x = clipped cells k_pos·T ≥ d_sae, excluded from slopes; "
        "seeds mean±std; arm shade = k_pos)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"btk_rerun_dperf_dT.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: mixing fingerprint — realized l0 per window vs nominal ──
    fig, axes = plt.subplots(1, len(benches), figsize=(6.5 * len(benches), 4),
                             squeeze=False)
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
            if not Ts:
                continue
            ax.plot(Ts, [float(np.mean(pts[t])) for t in Ts],
                    color=ARM_COLOR[arch], marker="o",
                    label=ARM_LABEL[arch])
        ax.axhline(1.0, color="#999", linestyle="--", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("window size T")
        ax.set_ylabel("realized l0 / nominal k_win")
        ax.set_title(f"{bench.replace('toy_', '')} — mixing fingerprint",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(args.out_dir / f"btk_rerun_fingerprint.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close(fig)

    for key, md in sorted(agg.items(), key=str):
        b, a, k, t = key
        summary["cells"][f"{b}|{a}|k{k}|T{t}"] = {
            m: {"mean": v[0], "std": v[1], "n": v[2]} for m, v in md.items()
        }
    (args.out_dir / "btk_rerun_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"[analysis] figures + summary -> {args.out_dir}")

    for bm, s in sorted(summary["slopes"].items()):
        print(f"  slope {bm}: {s if s is None else round(s, 4)}")


if __name__ == "__main__":
    main()
