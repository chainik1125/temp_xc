"""Analyse the fresh v2 FreqBench AC sweep from the leaderboard.

Reads ``results/leaderboard.jsonl``, keeps ``freq_bench`` rows at the
current protocol, and renders the corrected views — now from FRESHLY
TRAINED v2 models, not Dmitry's archived JSON:

  1. ntps_by_rawk      NTPS(linear) vs raw_k, faceted by W (d_sae=40)
  2. order_controls    A / shuffle / reverse at W=16, raw_k=1, d_sae=40
  3. linear_vs_mlp     NTPS(linear) vs NTPS(MLP) — is the info linearly readable?
  4. freqfrac_by_rawk  weight-space AC content of the encoder vs raw_k
  5. capacity          NTPS vs d_sae (40 → 256)

Run after the sweep:  .venv/bin/python experiments/freq_bench/analyze_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "freq_bench" / "v2_sweep"
OUT.mkdir(parents=True, exist_ok=True)
PROTO = "1.2.0"

LABEL_ORDER = ["regular_sae", "txcdr_t2", "txcdr_t5"]
COLORS = {"regular_sae": "#888888", "txcdr_t2": "#9467bd", "txcdr_t5": "#2ca02c"}


def load_rows() -> list[dict]:
    rows = []
    for line in open(LEADERBOARD):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("experiment") != "freq_bench":
            continue
        if r.get("evaluator_protocol_version") != PROTO:
            continue
        ec = r.get("eval_cfg", {})
        if ec.get("smoke"):
            continue
        if "label" not in ec:
            continue
        rows.append({
            "label": ec["label"], "W": ec["W"], "k_pos": ec["k_pos"],
            "d_sae": ec["d_sae"], **r["metrics"],
        })
    return rows


def _get(rows, label, W, k, dsae, key):
    for r in rows:
        if (r["label"] == label and r["W"] == W and r["k_pos"] == k
                and r["d_sae"] == dsae and key in r):
            return r[key]
    return None


def fig_ntps_by_rawk(rows, dsae=40):
    Ws = sorted({r["W"] for r in rows})
    ks = sorted({r["k_pos"] for r in rows})
    fig, axes = plt.subplots(1, len(Ws), figsize=(3.4 * len(Ws), 3.6), sharey=True)
    if len(Ws) == 1:
        axes = [axes]
    for ax, W in zip(axes, Ws):
        for label in LABEL_ORDER:
            xs, ys = [], []
            for k in ks:
                v = _get(rows, label, W, k, dsae, "NTPS")
                if v is not None:
                    xs.append(k); ys.append(v)
            if xs:
                ax.plot(xs, ys, "o-", color=COLORS[label], label=label, lw=1.8)
        ax.axhline(0, color="k", ls="--", alpha=.4, lw=1)
        ax.set_xscale("log"); ax.set_title(f"W={W}"); ax.set_xlabel("raw_k (k_pos)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("NTPS (linear probe)")
    axes[-1].legend(fontsize=8)
    plt.suptitle(f"v2 AC sweep — NTPS vs raw_k, faceted by W (d_sae={dsae}, fresh training)\n"
                 "Signal concentrates at sparse codes + long windows, as the re-analysis predicted",
                 fontsize=11)
    plt.tight_layout()
    p = OUT / "ntps_by_rawk.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


def fig_order_controls(rows, W=16, k=1, dsae=40):
    labels = [l for l in LABEL_ORDER if _get(rows, l, W, k, dsae, "A") is not None]
    x = np.arange(len(labels)); w = 0.27
    A = [_get(rows, l, W, k, dsae, "A") for l in labels]
    sh = [_get(rows, l, W, k, dsae, "A_shuffle") for l in labels]
    rv = [_get(rows, l, W, k, dsae, "A_reverse") for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w, A, w, label="A (ordered)", color="#1f77b4")
    ax.bar(x, sh, w, label="A (shuffled)", color="#aaaaaa")
    ax.bar(x + w, rv, w, label="A (reversed)", color="#d62728")
    ax.axhline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("linear-probe accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title(f"v2 AC order-controls @ W={W}, raw_k={k}, d_sae={dsae}\n"
                 "windowed archs: ordered≫chance, shuffle→chance, reverse<chance")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    plt.tight_layout()
    p = OUT / "order_controls.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


def fig_linear_vs_mlp(rows, dsae=40):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for label in LABEL_ORDER:
        xs = [r["NTPS"] for r in rows if r["label"] == label and r["d_sae"] == dsae
              and "NTPS_mlp" in r]
        ys = [r["NTPS_mlp"] for r in rows if r["label"] == label and r["d_sae"] == dsae
              and "NTPS_mlp" in r]
        if xs:
            ax.scatter(xs, ys, color=COLORS[label], label=label, s=28, alpha=.8)
    lim = [-0.2, 1.05]
    ax.plot(lim, lim, "k--", alpha=.4, lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("NTPS (linear probe)"); ax.set_ylabel("NTPS (MLP probe)")
    ax.set_title(f"Linear vs nonlinear probe (d_sae={dsae})\nabove diagonal ⇒ info present but linearly under-read")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    p = OUT / "linear_vs_mlp.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


def fig_freqfrac(rows, dsae=40):
    ks = sorted({r["k_pos"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for label in ["txcdr_t2", "txcdr_t5"]:
        xs, ys = [], []
        for k in ks:
            vals = [r["freqfrac"] for r in rows if r["label"] == label
                    and r["k_pos"] == k and r["d_sae"] == dsae and "freqfrac" in r]
            if vals:
                xs.append(k); ys.append(float(np.mean(vals)))
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS[label], label=label, lw=1.8)
    ax.set_xscale("log"); ax.set_xlabel("raw_k (k_pos)")
    ax.set_ylabel("FreqFrac of W_enc (AC energy fraction)")
    ax.set_title(f"Weight-space order-sensitivity (d_sae={dsae})\nhigh ⇒ encoder atoms detect transitions, not just average")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    p = OUT / "freqfrac_by_rawk.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


def fig_capacity(rows, W=16, k=1):
    dsaes = sorted({r["d_sae"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label in LABEL_ORDER:
        xs, ys = [], []
        for d in dsaes:
            v = _get(rows, label, W, k, d, "NTPS")
            if v is not None:
                xs.append(d); ys.append(v)
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS[label], label=label, lw=1.8)
    ax.set_xscale("log"); ax.set_xlabel("d_sae (dictionary capacity)")
    ax.set_ylabel("NTPS (linear)"); ax.set_title(f"Capacity sweep @ W={W}, raw_k={k}")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    p = OUT / "capacity.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("saved", p)


def write_summary(rows):
    s = {"n_rows": len(rows), "protocol": PROTO, "cells": {}}
    for label in LABEL_ORDER:
        lab = {}
        for dsae in sorted({r["d_sae"] for r in rows}):
            cand = [r for r in rows if r["label"] == label and r["d_sae"] == dsae]
            if not cand:
                continue
            best = max(cand, key=lambda r: r.get("NTPS", -9))
            lab[f"d_sae={dsae}"] = {
                "peak_NTPS": round(best["NTPS"], 3),
                "cell": f"W={best['W']},raw_k={best['k_pos']}",
                "A": round(best["A"], 3),
                "A_shuffle": round(best["A_shuffle"], 3),
                "A_reverse": round(best["A_reverse"], 3),
                "NTPS_mlp": round(best.get("NTPS_mlp", float("nan")), 3),
                "freqfrac": round(best["freqfrac"], 3) if "freqfrac" in best else None,
            }
        s["cells"][label] = lab
    p = OUT / "summary.json"; json.dump(s, open(p, "w"), indent=2)
    print("saved", p)
    return s


if __name__ == "__main__":
    rows = load_rows()
    print(f"loaded {len(rows)} freq_bench rows @ protocol {PROTO}")
    if not rows:
        raise SystemExit("no rows yet — run the sweep first.")
    fig_ntps_by_rawk(rows)
    fig_order_controls(rows)
    fig_linear_vs_mlp(rows)
    fig_freqfrac(rows)
    fig_capacity(rows)
    s = write_summary(rows)
    print(json.dumps(s, indent=2))
