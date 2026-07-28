"""Genuine coherent instruction-switches under steering, on StruQ's strong attack.

The paper's backtracking headline is a judge-scored count of GENUINE events under steering,
baseline-corrected. This is that figure for prompt injection: the event is the model obeying an
injected instruction, and because the unsteered rate is already 0.91 the informative direction
is SUPPRESSION -- steering that makes the model stop obeying.

Reads results/txc_wins/struq_gen_creal_ds{0,1,2}_summary.json (whichever exist) and plots each
arm's best suppression against the null. The two random arms are the load-bearing comparison:
an arm that suppresses no more than a random write of the same norm has not been shown to carry
anything about the task.

Writes plots/2026-07-28_struq_switch/struq_switch.png.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "txc_wins"
OUT = ROOT / "plots" / "2026-07-28_struq_switch" / "struq_switch.png"

# (key, label, colour, kind) -- kind drives hatching so supervised/null arms are unmistakable.
ARMS = [
    ("broadcast_optimal", "best constant write\n(supervised)", "#999999", "sup"),
    ("tsae_broadcast", "attention tSAE", "#009E73", "learned"),
    ("tsaep_broadcast", "published T-SAE\n(arXiv:2511.05541)", "#56B4E9", "learned"),
    ("random_broadcast", "random broadcast\n(NULL)", "#D55E00", "null"),
    ("random_slab", "random slab\n(NULL)", "#E8A87C", "null"),
    ("txc_slab", "temporal crosscoder", "#E69F00", "learned"),
    ("txc_flat", "crosscoder, profile removed\n(= paper v7 write)", "#000000", "learned"),
    ("sae_broadcast", "TopK SAE", "#0072B2", "learned"),
    ("dom_slab", "difference-of-means\n(supervised)", "#CCCCCC", "sup"),
]


def load():
    out = []
    for ds in (0, 1, 2):
        p = RES / f"struq_gen_creal_ds{ds}_summary.json"
        if p.exists():
            out.append((ds, json.loads(p.read_text())))
    return out


def main() -> int:
    runs = load()
    if not runs:
        print("[skip] no summaries yet")
        return 1
    print(f"[load] {len(runs)} seed(s): {[ds for ds, _ in runs]}")

    base = float(np.mean([d["switch_headline"]["baseline_rate"] for _, d in runs]))
    per_arm = {}
    for key, *_ in ARMS:
        vals = []
        for _ds, d in runs:
            g = d.get("generations_summary", {}).get(key)
            if not g:
                continue
            # best suppression = lowest judged rate across the dose grid
            rates = [(c["switch_events"] / c["n"]) for c in g.values() if c["n"]]
            if rates:
                vals.append(min(rates))
        if vals:
            per_arm[key] = vals

    order = sorted(per_arm, key=lambda k: np.mean(per_arm[k]))
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    meta = {k: (lab, col, kind) for k, lab, col, kind in ARMS}
    xs = np.arange(len(order))
    for i, k in enumerate(order):
        lab, col, kind = meta[k]
        v = per_arm[k]
        m = float(np.mean(v))
        ax.bar(i, base - m, color=col, edgecolor="black", linewidth=0.9,
               hatch={"sup": "//", "null": "xx", "learned": ""}[kind], zorder=3)
        if len(v) > 1:
            ax.errorbar(i, base - m, yerr=np.std(v, ddof=1) / len(v) ** 0.5,
                        color="black", capsize=4, lw=1.2, zorder=4)
        ax.text(i, base - m + 0.008, f"{base-m:.3f}", ha="center", fontsize=9)

    nulls = [base - float(np.mean(per_arm[k])) for k in per_arm
             if meta[k][2] == "null"]
    if nulls:
        ax.axhline(max(nulls), color="#D55E00", ls="--", lw=1.6, zorder=2)
        ax.annotate("best NULL — an arm below this line has not been shown\n"
                    "to carry anything task-specific",
                    (len(order) - 0.4, max(nulls)), ha="right", va="bottom",
                    fontsize=9.5, color="#D55E00")

    ax.set_xticks(xs)
    ax.set_xticklabels([meta[k][0] for k in order], fontsize=8.5)
    ax.set_ylabel("suppression of genuine instruction-switches\n"
                  f"(baseline rate {base:.3f} minus steered rate; higher = suppresses more)")
    ax.set_title("Steering away a prompt injection: genuine coherent switches suppressed\n"
                 f"StruQ completion_real, judged, {len(runs)} dictionary seed(s), "
                 "dose grid capped at |α| ≤ 0.25", fontsize=12)
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    ax.axhline(0, color="#444444", lw=1.0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=165)
    print("[saved]", OUT)
    for k in order:
        v = per_arm[k]
        print(f"  {k:<22} suppression {base-np.mean(v):+.3f}  "
              f"(seeds {['%.3f' % x for x in v]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
