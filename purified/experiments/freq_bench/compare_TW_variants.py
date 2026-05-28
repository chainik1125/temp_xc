"""Compare arch variants at the strong cell (W=16, raw_k=1, d_sae=1024).

Five archs / configs from the v2 leaderboard at the same cell:
    regular_sae               per-token (chance baseline)
    txc_base_perpos_TW        per-position TopK at T=W
    txc_base_TW               joint TopK at T=W (Dmitry's variant)
    txcdr_t2                  joint TopK at T=2, slid across W
    txcdr_t5                  joint TopK at T=5, slid across W
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "freq_bench" / "v2_sweep"
PROTO = "1.2.0"

ORDER = [
    ("regular_sae", "per-token\n(baseline)", "#888888"),
    ("txc_base_perpos_TW", "txc_base perpos\nT=W (per-pos TopK)", "#ff7f0e"),
    ("txc_base_TW", "txc_base\nT=W (joint TopK)", "#1f77b4"),
    ("txcdr_t2", "txcdr_t2\n(slid T=2 across W)", "#9467bd"),
    ("txcdr_t5", "txcdr_t5\n(slid T=5 across W)", "#2ca02c"),
]
W_TARGET, K_TARGET, DSAE_TARGET = 16, 1, 1024


def pick(label):
    for line in open(LEADERBOARD):
        r = json.loads(line)
        if (r.get("experiment") == "freq_bench"
                and r.get("evaluator_protocol_version") == PROTO):
            ec = r.get("eval_cfg", {})
            if (ec.get("label") == label and ec.get("W") == W_TARGET
                    and ec.get("k_pos") == K_TARGET
                    and ec.get("d_sae") == DSAE_TARGET):
                return r["metrics"]
    return None


def main():
    rows = []
    for label, _, _ in ORDER:
        m = pick(label)
        if m is None:
            print(f"missing: {label} @ W={W_TARGET},k={K_TARGET},dsae={DSAE_TARGET}")
        rows.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    # left: NTPS
    ax = axes[0]
    xs = list(range(len(ORDER)))
    for i, ((label, name, color), m) in enumerate(zip(ORDER, rows)):
        ax.bar(i, m["NTPS"] if m else 0, color=color)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n, _ in ORDER], fontsize=8)
    ax.set_ylabel("NTPS (linear probe)")
    ax.set_ylim(-0.05, 0.85)
    ax.set_title(f"NTPS @ W={W_TARGET}, raw_k={K_TARGET}, d_sae={DSAE_TARGET}")
    ax.grid(alpha=.3, axis="y")
    for i, m in enumerate(rows):
        if m: ax.text(i, m["NTPS"] + 0.02, f"{m['NTPS']:.2f}",
                      ha="center", fontsize=8)

    # right: order controls (A, A_shuffle, A_reverse)
    ax = axes[1]
    w = 0.25
    for i, ((_, name, color), m) in enumerate(zip(ORDER, rows)):
        if m is None: continue
        ax.bar(i - w, m["A"], w, color=color, alpha=1.0,
               label="A (ordered)" if i == 0 else None, edgecolor="k")
        ax.bar(i, m["A_shuffle"], w, color=color, alpha=0.55,
               label="A (shuffled)" if i == 0 else None, edgecolor="k")
        ax.bar(i + w, m["A_reverse"], w, color=color, alpha=0.25,
               label="A (reversed)" if i == 0 else None, edgecolor="k")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n, _ in ORDER], fontsize=8)
    ax.set_ylabel("linear-probe accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Order controls @ same cell")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=.3, axis="y")

    plt.suptitle("Architecture-comparison @ the strong AC cell: "
                 "sliding-T dominates; per-position TopK fails "
                 "(mean-pooled per-pos code is direction-invariant).",
                 fontsize=11)
    plt.tight_layout()
    p = OUT / "TW_variants_comparison.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)

    # also dump
    summary = {label: rows[i] for i, (label, _, _) in enumerate(ORDER)}
    json.dump(summary, open(OUT / "TW_variants_comparison.json", "w"),
              indent=2, default=float)
    print("saved", OUT / "TW_variants_comparison.json")


if __name__ == "__main__":
    main()
