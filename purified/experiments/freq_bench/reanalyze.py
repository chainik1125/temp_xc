"""Re-analysis of Dmitry's FreqBench (2026-05-06), corrected.

Dmitry's headline ("AC: every arch fails; they aggregate, they do not
filter") was read off a single slice — NTPS at ``raw_k=10`` — and from
the raw NTPS number alone. Two things in his own committed JSON tell a
different story:

  1. The AC signal is strongly sparsity-dependent. It peaks at the
     SPARSEST code (raw_k=1), where the windowed/attention archs reach
     NTPS ~0.42; at raw_k=10 (the plotted slice) it has washed out.

  2. The shuffle / reverse order-controls (present in every row, never
     analysed) are unambiguous. At the strong cell (W=16, raw_k=1,
     sigma=0.1): shuffling tokens collapses accuracy to chance, and
     REVERSING the sequence drives it BELOW chance (the probe predicts
     the flipped sign). That is the textbook signature of a
     representation that encodes signed direction, not mere aggregation.

This script recomputes the corrected views from the vendored raw JSON
and writes plots + a summary JSON. No new training.

Run:
    .venv/bin/python experiments/freq_bench/reanalyze.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]            # purified/
RAW = ROOT / "results" / "freq_bench" / "dmitry_raw"
OUT = ROOT / "results" / "freq_bench" / "reanalysis"
OUT.mkdir(parents=True, exist_ok=True)

ARCH_ORDER = ["regular_sae", "txcdr_t2", "txcdr_t5", "txc_base", "tfa",
              "tsae_attn", "tsae_bhalla"]
ARCH_COLORS = {
    "regular_sae": "#888888", "txcdr_t2": "#9467bd", "txcdr_t5": "#2ca02c",
    "txc_base": "#1f77b4", "tfa": "#d62728", "tsae_attn": "#8c564b",
    "tsae_bhalla": "#ff7f0e",
}
ARCH_LABEL = {
    "regular_sae": "regular_sae", "txcdr_t2": "txcdr (T=2)",
    "txcdr_t5": "txcdr (T=5)", "txc_base": "txc_base (T=W)", "tfa": "TFA",
    "tsae_attn": "T-SAE attn (mislabel)", "tsae_bhalla": "T-SAE Bhalla",
}
# Archs that are per-token at inference (the predicted null group).
PER_TOKEN = {"regular_sae", "tsae_bhalla"}


def _clean(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def load(name: str):
    p = RAW / name
    if not p.exists():
        return []
    return json.load(open(p))


def merge_unsigned():
    rows = load("pod3_mixed_results.json") + load("synth1_mixed_unsigned.json")
    by = {}
    for r in rows:
        by[(r["model"], r["W"], round(r["sigma"], 4),
            r.get("variant", "unsigned"), r["raw_k"])] = r
    return list(by.values())


def merge_signed():
    rows = load("pod4_mixed_signed_results.json") + load("synth1_mixed_signed_results.json")
    by = {}
    for r in rows:
        by[(r["model"], r["W"], round(r["sigma"], 4),
            r.get("variant", "signed"), r["raw_k"])] = r
    return list(by.values())


# ── Figure 1: AC NTPS faceted by raw_k (the slice Dmitry never plotted) ──

def fig_ac_ntps_by_rawk(ac):
    raw_ks = sorted({r["raw_k"] for r in ac})
    fig, axes = plt.subplots(1, len(raw_ks), figsize=(3.6 * len(raw_ks), 3.6),
                             sharey=True)
    for ax, k in zip(axes, raw_ks):
        for arch in ARCH_ORDER:
            xs, ys = [], []
            for W in sorted({r["W"] for r in ac}):
                vs = [r["NTPS"] for r in ac
                      if r["model"] == arch and r["raw_k"] == k and r["W"] == W
                      and r["sigma"] == 0.1 and _clean(r["NTPS"])]
                if vs:
                    xs.append(W); ys.append(vs[0])
            if xs:
                ax.plot(xs, ys, "o-", color=ARCH_COLORS[arch],
                        label=ARCH_LABEL[arch], linewidth=1.8, markersize=4)
        ax.axhline(0, color="k", ls="--", alpha=.4, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(f"raw_k={k}")
        ax.set_xlabel("W")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("NTPS (signed velocity)")
    axes[-1].legend(fontsize=7, loc="upper left")
    plt.suptitle("AC bench — NTPS vs W, faceted by raw_k (σ=0.1)\n"
                 "Signal peaks at the SPARSEST code (raw_k=1); Dmitry plotted only raw_k=10",
                 fontsize=11)
    plt.tight_layout()
    out = OUT / "ac_ntps_by_rawk.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ── Figure 2: AC order-controls (the headline correction) ────────────────

def fig_ac_order_controls(ac):
    """A vs A_shuffle vs A_reverse at raw_k=1, σ=0.1, per arch."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    Ws = sorted({r["W"] for r in ac})
    width = 0.0
    x = np.arange(len(ARCH_ORDER))
    # Use W=16 (the strongest window).
    Wsel = 16
    A, Ash, Arev = [], [], []
    for arch in ARCH_ORDER:
        r = next((r for r in ac if r["model"] == arch and r["raw_k"] == 1
                  and r["sigma"] == 0.1 and r["W"] == Wsel), None)
        A.append(r["A"] if r and _clean(r["A"]) else np.nan)
        Ash.append(r["A_shuffle"] if r and _clean(r.get("A_shuffle")) else np.nan)
        Arev.append(r["A_reverse"] if r and _clean(r.get("A_reverse")) else np.nan)
    w = 0.27
    ax.bar(x - w, A, w, label="A (ordered)", color="#1f77b4")
    ax.bar(x, Ash, w, label="A (shuffled)", color="#aaaaaa")
    ax.bar(x + w, Arev, w, label="A (reversed)", color="#d62728")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance = 0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in ARCH_ORDER], rotation=30, ha="right",
                       fontsize=8)
    ax.set_ylabel("linear-probe accuracy")
    ax.set_ylim(0, 0.85)
    ax.set_title(f"AC order-controls @ W={Wsel}, raw_k=1, σ=0.1\n"
                 "Ordered ≫ chance, shuffle → chance, reverse < chance "
                 "⇒ signed-direction encoding (windowed/attn archs)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3, axis="y")
    plt.tight_layout()
    out = OUT / "ac_order_controls.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ── Figure 3: order-sensitivity gap (A - A_shuffle) across benches ───────

def fig_order_gap_summary(ac, mu, ms):
    benches = [("AC", ac), ("Mixed-unsigned", mu), ("Mixed-signed", ms)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (name, rows) in zip(axes, benches):
        for arch in ARCH_ORDER:
            cand = [r for r in rows if r["model"] == arch and _clean(r.get("A"))
                    and _clean(r.get("A_shuffle"))]
            if not cand:
                continue
            best = max(cand, key=lambda r: r["A"] - r["A_shuffle"])
            gap = best["A"] - best["A_shuffle"]
            ax.barh(ARCH_LABEL[arch], gap, color=ARCH_COLORS[arch])
        ax.axvline(0, color="k", lw=.8)
        ax.set_title(name)
        ax.set_xlabel("max order gap  (A − A_shuffle)")
        ax.grid(alpha=.3, axis="x")
    plt.suptitle("Order-sensitivity: best-cell (A − A_shuffle) per arch.  "
                 ">0 ⇒ the arch uses token order. txcdr/tfa lead; per-token archs ~0.",
                 fontsize=11)
    plt.tight_layout()
    out = OUT / "order_gap_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", out)


# ── Summary stats JSON ───────────────────────────────────────────────────

def summary(ac, dc, mu, ms):
    s = {"note": "Re-analysis of Dmitry FreqBench 2026-05-06. NTPS faceted by "
                 "raw_k + shuffle/reverse order-controls.", "benches": {}}

    # DC: confirm the per-token vs windowed split (raw_k=10).
    dc_peak = {}
    for a in ARCH_ORDER:
        vs = [r["NTPS"] for r in dc if r["model"] == a and r["raw_k"] == 10
              and _clean(r["NTPS"])]
        if vs:
            dc_peak[a] = round(max(vs), 3)
    s["benches"]["DC"] = {"peak_NTPS_rawk10": dc_peak}

    # AC: peak NTPS per arch per raw_k + the order-control table at the peak.
    ac_by_k = {}
    for a in ARCH_ORDER:
        ac_by_k[a] = {}
        for k in sorted({r["raw_k"] for r in ac}):
            vs = [r["NTPS"] for r in ac if r["model"] == a and r["raw_k"] == k
                  and _clean(r["NTPS"])]
            if vs:
                ac_by_k[a][f"raw_k={k}"] = round(max(vs), 3)
    ac_ctrl = {}
    for a in ARCH_ORDER:
        cand = [r for r in ac if r["model"] == a and _clean(r["NTPS"])]
        if not cand:
            continue
        r = max(cand, key=lambda r: r["NTPS"])
        ac_ctrl[a] = {
            "cell": f"W={r['W']},raw_k={r['raw_k']},σ={r['sigma']}",
            "NTPS": round(r["NTPS"], 3), "A": round(r["A"], 3),
            "A_shuffle": round(r["A_shuffle"], 3) if _clean(r.get("A_shuffle")) else None,
            "A_reverse": round(r["A_reverse"], 3) if _clean(r.get("A_reverse")) else None,
        }
    s["benches"]["AC"] = {"peak_NTPS_by_rawk": ac_by_k,
                          "order_controls_at_peak": ac_ctrl}

    # Mixed: best order gap per arch.
    for name, rows in [("Mixed_unsigned", mu), ("Mixed_signed", ms)]:
        d = {}
        for a in ARCH_ORDER:
            cand = [r for r in rows if r["model"] == a and _clean(r.get("A"))
                    and _clean(r.get("A_shuffle"))]
            if not cand:
                continue
            r = max(cand, key=lambda r: r["A"] - r["A_shuffle"])
            d[a] = {"cell": f"W={r['W']},raw_k={r['raw_k']},σ={r['sigma']}",
                    "order_gap": round(r["A"] - r["A_shuffle"], 3),
                    "A": round(r["A"], 3), "A_shuffle": round(r["A_shuffle"], 3)}
        s["benches"][name] = {"best_order_gap": d}

    out = OUT / "summary.json"
    json.dump(s, open(out, "w"), indent=2)
    print("saved", out)
    return s


if __name__ == "__main__":
    dc = load("synth1_dc_results.json")
    ac = load("synth2_ac_results.json")
    mu = merge_unsigned()
    ms = merge_signed()
    fig_ac_ntps_by_rawk(ac)
    fig_ac_order_controls(ac)
    fig_order_gap_summary(ac, mu, ms)
    s = summary(ac, dc, mu, ms)
    print("\n=== AC order-controls at each arch's peak NTPS cell ===")
    for a, d in s["benches"]["AC"]["order_controls_at_peak"].items():
        print(f"  {a:13s} {d['cell']:24s} NTPS={d['NTPS']:+.3f} "
              f"A={d['A']:.3f} shuf={d['A_shuffle']} rev={d['A_reverse']}")
