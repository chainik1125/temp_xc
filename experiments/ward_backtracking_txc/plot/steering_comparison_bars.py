"""B1 steering curves — DoM baselines + best feature per (arch, hookpoint, mode).

Reads results/ward_backtracking_txc/steering/b1_steering_results.json and
plots mean keyword rate vs magnitude for:
  - DoM_base, DoM_reasoning (Stage A baselines)
  - For each (arch, hookpoint, mode), the *best* mined feature
    (by max |kw| across magnitudes — captures both positive and negative
    steering since we now sweep symmetric mags).

Source tags follow the convention `<arch>_<hookpoint_key>_f<id>_<mode>`
where hookpoint_key contains exactly one underscore (e.g. `resid_L10`).
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


def _parse_source_tag(tag: str) -> tuple[str, str, str, str]:
    """Parse `<arch>_<hp>_f<id>_<mode>` → (arch, hp, fid, mode).

    `arch` may itself contain underscores (e.g. `topk_sae`, `stacked_sae`).
    Strategy: split on `_f`, take everything before as `<arch>_<hp>`, then
    split that on `_resid_` or `_attn_` or `_ln1_` to find hp boundary.
    """
    if "_f" not in tag:
        return ("dom", "n/a", "n/a", "dom")
    head, _, ftail = tag.partition("_f")
    fid_str, _, mode = ftail.partition("_")
    # head looks like "<arch>_<hp_component>_<hp_layerN>"
    # We know hookpoint keys end in _L<digits>; everything before that prefix is arch.
    parts = head.split("_")
    # Find the index of the part that starts with 'L' followed by digits
    hp_layer_idx = None
    for i, p in enumerate(parts):
        if p.startswith("L") and p[1:].isdigit():
            hp_layer_idx = i; break
    if hp_layer_idx is None or hp_layer_idx < 1:
        return ("?", head, fid_str, mode or "?")
    arch = "_".join(parts[: hp_layer_idx - 1])
    hp = "_".join(parts[hp_layer_idx - 1: hp_layer_idx + 1])
    return (arch, hp, fid_str, mode or "pos0")


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

    cells = defaultdict(list)
    for r in rows:
        cells[(r["source"], r["target"], r["magnitude"])].append(r["keyword_rate"])

    def curve(source: str, target: str = "reasoning"):
        xs, ys, ses = [], [], []
        for m in mags:
            vs = cells.get((source, target, m), [])
            if vs:
                xs.append(m); ys.append(float(np.mean(vs)))
                ses.append(float(np.std(vs, ddof=1) / np.sqrt(len(vs))))
        return np.array(xs), np.array(ys), np.array(ses)

    sources = sorted({r["source"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 6))

    # DoM baselines
    if "dom_base_union" in sources:
        x, y, se = curve("dom_base_union")
        ax.errorbar(x, y, yerr=se, label="DoM(base)", marker="o", lw=2, color="black")
    if "dom_reasoning_union" in sources:
        x, y, se = curve("dom_reasoning_union")
        ax.errorbar(x, y, yerr=se, label="DoM(reasoning)", marker="s", lw=2,
                    color="dimgray", linestyle="--")

    # Group TXC-style sources by (arch, hp, mode); pick best feature per group.
    by_group: dict[tuple, list[str]] = defaultdict(list)
    for s in sources:
        if s.startswith("dom_"):
            continue
        arch, hp, fid, mode = _parse_source_tag(s)
        by_group[(arch, hp, mode)].append(s)

    palette = plt.cm.tab20
    arch_color = {"txc": 0, "topk_sae": 4, "stacked_sae": 8, "tsae": 12, "?": 16}
    mode_style = {"pos0": "-", "union": ":", "?": "-", "dom": "-"}

    for i, (group, srcs) in enumerate(sorted(by_group.items())):
        arch, hp, mode = group
        # Pick best by max |kw| over any magnitude — symmetric around zero.
        best = None; best_score = -1.0
        for s in srcs:
            x, y, _ = curve(s)
            if len(y) == 0: continue
            score = float(np.abs(y).max())
            if score > best_score:
                best_score = score; best = s
        if best is None: continue
        x, y, se = curve(best)
        color_idx = arch_color.get(arch, 16)
        ls = mode_style.get(mode, "-")
        ax.errorbar(x, y, yerr=se,
                    label=f"{arch}/{hp}/{mode} (best=f{best.split('_f')[-1].split('_')[0]})",
                    marker="^", lw=1.5, color=palette(color_idx),
                    linestyle=ls, capsize=2)

    ax.set_xlabel("Steering magnitude (signed; negatives steer in opposite direction)")
    ax.set_ylabel("Mean keyword rate (wait+hmm)/words")
    ax.set_title("B1 — feature steering vs DoM baselines (target = reasoning)")
    ax.axhline(0.007, color="lightgray", lw=0.7, linestyle=":", label="baseline 0.7%")
    ax.axvline(0, color="lightgray", lw=0.7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    out = out_dir / "steering_comparison_bars.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
