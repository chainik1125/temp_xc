"""Hedging-LEVEL Stage 2 — the T-scaling figure (arm B's second money plot).

Thin wrapper over the merged variance-aware renderer
(`lambda_intensity.render_stage2`: ARCH_STYLE, draw, 95% t CI whiskers,
realized-l0 legend tags — review note 3). One adaptation, because this
panel runs TXC-post at nominal k_pos = 8·T (card_stage2.md § 4): the
budget-match rule compares every arch's realized l0 to the uniform
intended PER-TOKEN budget of 8 (min cell-mean l0 ≥ 4), instead of
`max(k_pos)/2` which would misread post's nominal 128.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.confidence.render_stage2 [ds]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.lambda_intensity.render_stage2 import (
    ARCH_STYLE,
    METRIC,
    draw,
)
from experiments.explorations.task_hunt.support_stats.stats_lib import t_ci95

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"
K_PER_TOKEN = 8            # the panel's uniform intended per-token budget


def build_summary(rows):
    ok = [r for r in rows if r.get("ok")]
    trained = defaultdict(list)
    untrained = defaultdict(list)
    chance = defaultdict(list)
    l0 = defaultdict(list)
    for r in ok:
        key = (r["arch"], r["T"])
        m = r["metrics"]
        bucket = untrained if r.get("kind") == "untrained" else trained
        bucket[key].append(m.get(METRIC, float("nan")))
        chance[key].append(m.get("lambda_chance", float("nan")))
        l0[key].append(m.get("l0_per_token", float("nan")))

    def agg(d):
        return {f"{a}/T{t}": {"mean": float(np.nanmean(v)),
                              "std": float(np.nanstd(v)), "n": len(v)}
                for (a, t), v in sorted(d.items())}

    ci = {}
    for (a, t), v in sorted(trained.items()):
        mean, lo, hi = t_ci95(v)
        ci[f"{a}/T{t}"] = {"mean": mean, "lo": lo, "hi": hi, "n": len(v)}

    l0_range, matched = {}, {}
    for a in {a for (a, _) in trained}:
        cell_means = [float(np.nanmean(l0[(a, t)]))
                      for (aa, t) in sorted(l0) if aa == a]
        if cell_means:
            l0_range[a] = {"min": min(cell_means), "max": max(cell_means)}
            matched[a] = min(cell_means) >= K_PER_TOKEN / 2

    summary = {
        "datasource": None, "metric": METRIC,
        "n_cells_ok": len(ok), "n_cells_total": len(rows),
        "trained": agg(trained), "untrained": agg(untrained),
        "chance": agg(chance), "l0_per_token": agg(l0),
        "ci95_trained": ci,
        "l0_range": l0_range,
        "budget_matched": matched,
        "match_rule": ("min cell-mean realized l0 >= intended per-token "
                       f"budget/2 = {K_PER_TOKEN / 2} (post nominal "
                       "k_pos = 8*T, card_stage2.md §4)"),
        "margin_trained_minus_untrained": {
            f"{a}/T{t}": float(np.nanmean(v)
                               - np.nanmean(untrained.get((a, t), [np.nan])))
            for (a, t), v in sorted(trained.items())},
    }
    return summary, trained, untrained, l0, matched, l0_range


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "ward_real_slope8_distill_l14"
    rows = json.loads((RES / f"stage2_{ds}.json").read_text())
    summary, trained, untrained, l0, matched, l0_range = build_summary(rows)
    summary["datasource"] = ds
    (RES / "stage2_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    variants = [("stage2_tscaling", False,
                 "Task hunt Stage 2 — hedging-trend (slope8) recovery vs T\n"
                 "REAL Ward activations (R1-Distill, resid_post L14); "
                 "whiskers = 95% t CI over seeds;\nhollow/dotted = untrained "
                 "control; TXC-post at nominal k = 8·T (budget-matched)"),
                ("stage2_tscaling_matched", True,
                 "Hedging-trend (slope8) recovery vs T — budget-matched "
                 "archs only\n(realized l0 ≥ 4/token; whiskers = 95% t CI "
                 "over seeds)")]
    raw_path = RES / "stage2_raw_reference.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    for stem, matched_only, title in variants:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        draw(ax, trained, untrained, l0, matched, l0_range, matched_only)
        # Raw-activation anchors (card § 6): reference lines, never cells.
        if raw:
            rt = raw.get("raw_tok", {}).get(METRIC)
            if rt is not None:
                ax.axhline(rt, color="k", ls="-.", lw=1.4, alpha=0.8,
                           label=f"RAW per-token (r = {rt:.3f})")
            Ts = sorted(int(k.split("_T")[1]) for k in raw
                        if k.startswith("raw_mean_T"))
            ys = [raw[f"raw_mean_T{t}"][METRIC] for t in Ts]
            if Ts:
                ax.plot(Ts, ys, "x--", color="k", lw=1.2, alpha=0.6,
                        label="RAW window-mean")
        ax.set_title(title, fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{stem}.{ext}",
                        dpi=140 if ext == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({"n_ok": summary["n_cells_ok"],
                      "n_total": summary["n_cells_total"],
                      "budget_matched": summary["budget_matched"],
                      "l0_range": summary["l0_range"],
                      "ci95_trained": summary["ci95_trained"]}, indent=2))
    print(f"-> {FIGS}/stage2_tscaling.* ; {RES}/stage2_summary.json")


if __name__ == "__main__":
    main()
