"""Repetition-rate vs (calibrated) magnitude — judge-free auxiliary metric.

Tests the hypothesis that SAE/TSAE narrow magnitude peak is bordered by
sharp transitions to looping; TXC-family degrades more gracefully.

For each generation in phase2_rescue.json, computes:
- max_jaccard: max token-Jaccard between any pair of consecutive sentences
- frac_near_dup: fraction of sentence pairs (consecutive) with token-Jaccard >= 0.7
- max_repeat_run: longest run of near-duplicate consecutive sentences

Plots per-arch mean of `frac_near_dup` vs calibrated magnitude.

Usage:
  python -m experiments.ward_backtracking_txc.plot.repetition_rate \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
      --calibration <out_root>/calibration.json \
      --out <out_root>
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ward_backtracking_txc.plot.headline_steering import (
    ARCH_PALETTE, HEADLINE_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.plot.rep")


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def frac_near_dup(text: str, threshold: float = 0.7) -> float:
    sents = split_sentences(text)
    if len(sents) < 2:
        return 0.0
    pairs = [(sents[i], sents[i+1]) for i in range(len(sents) - 1)]
    sims = [token_jaccard(a, b) for a, b in pairs]
    return float(np.mean([s >= threshold for s in sims]))


def calibrated_x(raw_mag: float, p95: float) -> float:
    if not p95 or p95 <= 0:
        return float(raw_mag)
    return float(raw_mag) / float(p95)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.7,
                   help="token-Jaccard threshold for 'near-duplicate'")
    p.add_argument("--label-filter", action="store_true",
                   help="restrict to headline labels only")
    args = p.parse_args(argv)

    calib = json.loads(args.calibration.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for calibrated_mode, ax in zip([True, False], axes):
        for run_dir in args.runs:
            meta = json.loads((run_dir / "meta.json").read_text())
            label = meta.get("label", "?")
            if args.label_filter and label not in HEADLINE_LABELS:
                continue
            rescue_path = run_dir / "phase2_rescue.json"
            if not rescue_path.exists():
                continue
            rows = json.loads(rescue_path.read_text())
            # Compute frac_near_dup per (mag) by averaging across questions
            from collections import defaultdict
            by_mag = defaultdict(list)
            for r in rows:
                by_mag[float(r["magnitude"])].append(
                    frac_near_dup(r.get("continuation_text", ""), args.threshold)
                )
            mags = sorted(by_mag)
            mean_rep = [float(np.mean(by_mag[m])) for m in mags]
            cell, fid, mode = meta["cell_id"], meta["feature_id"], meta["feature_mode"]
            key = f"{cell}__f{fid}_{mode}"
            p95 = calib.get(key, {}).get("p95_pooled", 0)
            x = [calibrated_x(m, p95) for m in mags] if calibrated_mode else mags
            ax.plot(x, mean_rep, "-o", label=label,
                    color=ARCH_PALETTE.get(label, "#888"),
                    markersize=4, linewidth=1.6)
        ax.set_xlabel("calibrated magnitude (raw/p95)" if calibrated_mode else "raw steering magnitude")
        ax.set_ylabel("frac of consecutive sentence pairs with token-Jaccard ≥ 0.7" if calibrated_mode else "")
        ax.set_title("calibrated" if calibrated_mode else "raw")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(f"Repetition rate (judge-free): near-duplicate sentence-pair fraction (threshold={args.threshold})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = "_headline" if args.label_filter else ""
    out_path = args.out / f"repetition_rate{suffix}.png"
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
