"""Stage A Fig 3 layout × per-hookpoint × {DoM, TXC@pos0, TXC@union} panels.

Reads results/ward_backtracking_txc/steering/b1_steering_results.json and
plots mean keyword rate vs magnitude for:
  - DoM_base (Stage A baseline)
  - DoM_reasoning
  - For each hookpoint, the *best* TXC feature (by max-mag mean kw rate),
    in pos0 mode and union mode.
"""

from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ward_backtracking_txc.plot._common import load_cfg, plots_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    out_dir = plots_dir(cfg)
    res_path = Path(cfg["paths"]["steering"])
    if not res_path.exists():
        print(f"[skip] {res_path} missing"); return
    obj = json.loads(res_path.read_text())
    rows = obj["rows"]
    mags = sorted({r["magnitude"] for r in rows})

    # cell mean: (source, target, magnitude) → mean keyword_rate
    cells = defaultdict(list)
    for r in rows:
        cells[(r["source"], r["target"], r["magnitude"])].append(r["keyword_rate"])

    def curve(source: str, target: str = "reasoning"):
        xs, ys, ses = [], [], []
        for m in mags:
            vs = cells.get((source, target, m), [])
            if vs:
                xs.append(m)
                ys.append(float(np.mean(vs)))
                ses.append(float(np.std(vs, ddof=1) / np.sqrt(len(vs))))
        return np.array(xs), np.array(ys), np.array(ses)

    sources = sorted({r["source"] for r in rows})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # DoM baselines
    if "dom_base_union" in sources:
        x, y, se = curve("dom_base_union")
        ax.errorbar(x, y, yerr=se, label="DoM(base)", marker="o", lw=2, color="black")
    if "dom_reasoning_union" in sources:
        x, y, se = curve("dom_reasoning_union")
        ax.errorbar(x, y, yerr=se, label="DoM(reasoning)", marker="s", lw=2, color="dimgray", linestyle="--")

    # Best TXC source per hookpoint × mode by max keyword rate at any positive magnitude.
    txc_by_hp = defaultdict(list)
    for s in sources:
        if s.startswith("txc_"):
            # tag pattern: txc_<hp_key>_f<id>_<mode>
            parts = s.split("_")
            mode = parts[-1]
            hp_key = parts[1] + "_" + parts[2]  # e.g. resid_L10
            txc_by_hp[(hp_key, mode)].append(s)

    palette = plt.cm.tab10
    for i, ((hp_key, mode), srcs) in enumerate(sorted(txc_by_hp.items())):
        # pick best by peak mean kw at any positive mag
        best = None; best_peak = -1.0
        for s in srcs:
            x, y, _ = curve(s)
            if len(y) == 0: continue
            peak = float(y[(x > 0)].max()) if (x > 0).any() else float(y.max())
            if peak > best_peak:
                best_peak = peak; best = s
        if best is None:
            continue
        x, y, se = curve(best)
        ax.errorbar(x, y, yerr=se,
                    label=f"TXC {hp_key} {mode} (best={best.split('_')[-2]})",
                    marker="^", lw=1.5, color=palette(i % 10))

    ax.set_xlabel("Steering magnitude")
    ax.set_ylabel("Mean keyword rate (wait+hmm)/words")
    ax.set_title("B1 — TXC feature steering vs DoM baseline (target=reasoning)")
    ax.axhline(0.007, color="lightgray", lw=0.7, linestyle=":", label="baseline 0.7%")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = out_dir / "steering_comparison_bars.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
