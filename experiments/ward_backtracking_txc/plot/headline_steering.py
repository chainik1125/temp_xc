"""Headline steering plot for the NeurIPS backtracking case study.

Reads:
  - flip_matrix.parquet (from build_flip_matrix.py)
  - calibration.json (from calibrate_magnitudes.py)
  - per-run summary.json (from b3_variants.py runs, for rescue/regression rates)
  - optional: coherence grades from results/.../coherence_grades/

Renders 4 panels (calibrated x-axis):
  1. Net rescues (n_ic - n_ci) vs calibrated magnitude — judge-free outcome.
  2. Rescue rate (n_ic / n_truly_wrong) vs calibrated magnitude.
  3. Regression rate (n_ci / n_correct_subsample) vs calibrated magnitude.
  4. Coherence vs calibrated magnitude — only if coherence data is available.

Plus a raw-magnitude version (uncalibrated) saved to *_raw.png.

Final 4-line headline legend: TXC, TXC-H8, SAE, TSAE-paper. Drop H13.

Usage:
  python -m experiments.ward_backtracking_txc.plot.headline_steering \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
      --calibration <out_root>/calibration.json \
      --flip-matrix <out_root>/flip_matrix.parquet \
      --out <out_root>
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.plot.headline")


# Standard project palette. Per Dmitry's standardized arch set
# (T=5 TXC, TSAE-paper, TFA, regular SAE, MLC) the headline is 5 lines.
# H8 is appendix-only; its colour is preserved here for the appendix plot.
ARCH_PALETTE = {
    "TXC":        "#1f4e79",  # deep blue
    "SAE":        "#e07b00",  # orange (TopK SAE)
    "TSAE-paper": "#c83e80",  # pink/magenta
    "TFA":        "#7f3f98",  # purple
    "MLC":        "#2ca02c",  # green
    # Appendix-only:
    "TXC-H8":     "#5b9bd5",  # mid blue
}

# Headline shows ONLY these arches; everything else is appendix.
HEADLINE_LABELS = {"TXC", "SAE", "TSAE-paper", "TFA", "MLC"}


def load_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def load_summary(run_dir: Path) -> dict | None:
    sp = run_dir / "summary.json"
    if not sp.exists():
        return None
    return json.loads(sp.read_text())


def calibrated_x(raw_mag: float, p95: float) -> float:
    if not p95 or p95 <= 0:
        return float(raw_mag)
    return float(raw_mag) / float(p95)


def panel_net_rescues(ax, df: pd.DataFrame, runs: list[dict], calib: dict, calibrated: bool):
    for r in runs:
        meta = r["meta"]
        label = meta.get("label", "?")
        cell, fid, mode = meta["cell_id"], meta["feature_id"], meta["feature_mode"]
        sub = df[(df["cell_id"] == cell)
                 & (df["feature_id"] == fid)
                 & (df["feature_mode"] == mode)]
        if sub.empty:
            log.warning("[skip net_rescues] no flip-matrix rows for %s f%s %s", cell, fid, mode)
            continue
        agg = sub.groupby("magnitude").apply(
            lambda g: (g["transition"] == "ic").sum() - (g["transition"] == "ci").sum(),
            include_groups=False,
        ).rename("net").reset_index().sort_values("magnitude")
        key = f"{cell}__f{fid}_{mode}"
        p95 = calib.get(key, {}).get("p95_pooled", 0)
        x = [calibrated_x(m, p95) for m in agg["magnitude"]] if calibrated else agg["magnitude"]
        ax.plot(x, agg["net"], "-o", label=label, color=ARCH_PALETTE.get(label, "#888"),
                markersize=4, linewidth=1.6)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_ylabel("net rescues  (n_ic − n_ci)")
    ax.set_xlabel("calibrated magnitude (raw / p95)" if calibrated else "raw steering magnitude")
    ax.legend(loc="best", fontsize=8)


def panel_rate(ax, runs: list[dict], calib: dict, key: str, calibrated: bool, ylabel: str):
    """key ∈ {'rescue_rate_by_magnitude', 'regression_rate_by_magnitude'}"""
    for r in runs:
        meta = r["meta"]
        s = r["summary"]
        if s is None or key not in s or not s[key]:
            continue
        label = meta.get("label", "?")
        ent = s[key]
        mags = sorted(map(float, ent.keys()))
        rates = [ent[str(m)]["rate"] for m in mags]
        cell, fid, mode = meta["cell_id"], meta["feature_id"], meta["feature_mode"]
        ckey = f"{cell}__f{fid}_{mode}"
        p95 = calib.get(ckey, {}).get("p95_pooled", 0)
        x = [calibrated_x(m, p95) for m in mags] if calibrated else mags
        ax.plot(x, rates, "-o", label=label, color=ARCH_PALETTE.get(label, "#888"),
                markersize=4, linewidth=1.6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("calibrated magnitude (raw / p95)" if calibrated else "raw steering magnitude")
    ax.legend(loc="best", fontsize=8)


def render(runs: list[dict], df: pd.DataFrame, calib: dict, out_path: Path,
           calibrated: bool, label_filter=None):
    runs = (runs if label_filter is None
            else [r for r in runs if r["meta"].get("label") in label_filter])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    panel_net_rescues(axes[0], df, runs, calib, calibrated=calibrated)
    panel_rate(axes[1], runs, calib, "rescue_rate_by_magnitude", calibrated=calibrated,
               ylabel="rescue rate  (n_ic / n_truly_wrong)")
    panel_rate(axes[2], runs, calib, "regression_rate_by_magnitude", calibrated=calibrated,
               ylabel="regression rate  (n_ci / n_correct_subsample)")
    title = "Backtracking steering — calibrated" if calibrated else "Backtracking steering — raw magnitude"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--flip-matrix", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output directory")
    args = p.parse_args(argv)

    runs = []
    for d in args.runs:
        meta = load_meta(d)
        summary = load_summary(d)
        runs.append({"dir": d, "meta": meta, "summary": summary})
    log.info("[runs] %d", len(runs))

    calib = json.loads(args.calibration.read_text())
    df = pd.read_parquet(args.flip_matrix)
    log.info("[calib] %d entries", len(calib))
    log.info("[flip] %d rows", len(df))

    args.out.mkdir(parents=True, exist_ok=True)
    # Headline: only Dmitry's standardized arch set
    render(runs, df, calib, args.out / "headline_calibrated.png", calibrated=True,
           label_filter=HEADLINE_LABELS)
    render(runs, df, calib, args.out / "headline_raw.png", calibrated=False,
           label_filter=HEADLINE_LABELS)
    # Appendix: include TXC-H8 and any other extras
    render(runs, df, calib, args.out / "appendix_calibrated.png", calibrated=True,
           label_filter=None)
    render(runs, df, calib, args.out / "appendix_raw.png", calibrated=False,
           label_filter=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
