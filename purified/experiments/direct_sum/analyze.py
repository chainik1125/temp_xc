"""Analyse the direct-sum which-process sweep from the leaderboard.

Reads ``results/leaderboard.jsonl``, keeps ``freq_bench`` rows for the
direct-sum datasources (ds_which_*), and renders:

  1. ntps_by_arch       NTPS by arch × J (main headline figure)
  2. order_controls     A / A_shuffle / A_reverse at J=4 (order diagnostic)
  3. per_process_acc    Per-process / per-class accuracy breakdown

Run after the sweep:
    .venv/bin/python experiments/direct_sum/analyze.py

Author: Aniket, 2026-06-01.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "direct_sum"
OUT.mkdir(parents=True, exist_ok=True)
PROTO = "1.2.0"

DIRECT_SUM_DATASOURCES = {"ds_which_J4_W16_s10", "ds_which_J8_W16_s10"}

LABEL_ORDER = ["topk_sae", "txcdr_t5", "txc_base_TW", "tfa"]
COLORS = {
    "topk_sae":    "#888888",
    "txcdr_t5":    "#2ca02c",
    "txc_base_TW": "#9467bd",
    "tfa":         "#1f77b4",
}
LABEL_DISPLAY = {
    "topk_sae":    "topk_sae (per-token)",
    "txcdr_t5":    "txcdr_t5 (sliding T=5)",
    "txc_base_TW": "txc_base T=W (joint)",
    "tfa":         "tfa (attention)",
}


def load_rows() -> list[dict]:
    """Load direct-sum rows from the leaderboard."""
    rows = []
    if not LEADERBOARD.exists():
        return rows
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
        # Only direct-sum datasources (leaderboard field is "datasource")
        ds = r.get("datasource", r.get("datasource_name", ""))
        if ds not in DIRECT_SUM_DATASOURCES:
            continue
        label = ec.get("label")
        if label is None:
            continue
        J = ec.get("J", None)
        if J is None:
            # Try to infer from datasource name
            if "J4" in ds:
                J = 4
            elif "J8" in ds:
                J = 8
        rows.append({
            "label": label,
            "J": J,
            "datasource": ds,
            **r["metrics"],
        })
    return rows


def _get(rows, label, J, key):
    for r in rows:
        if r["label"] == label and r["J"] == J and key in r:
            return r[key]
    return None


def fig_ntps_by_arch(rows):
    """NTPS by arch × J (2-bar-group figure)."""
    Js = sorted({r["J"] for r in rows if r["J"] is not None})
    labels = [l for l in LABEL_ORDER if any(r["label"] == l for r in rows)]

    x = np.arange(len(labels))
    width = 0.35
    offsets = np.linspace(-width * (len(Js) - 1) / 2, width * (len(Js) - 1) / 2, len(Js))
    j_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, J in enumerate(Js):
        vals = [_get(rows, l, J, "NTPS") or 0.0 for l in labels]
        bars = ax.bar(x + offsets[idx], vals, width * 0.9,
                      label=f"J={J}", color=j_colors[idx % len(j_colors)],
                      alpha=0.8, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="k", ls="--", alpha=0.5, lw=1, label="A_loc ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_DISPLAY.get(l, l) for l in labels], rotation=12, ha="right")
    ax.set_ylabel("NTPS (linear probe)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title(
        "Direct-sum which-process: NTPS by arch × J\n"
        "Per-token arch stays at chance (NTPS≈0); temporal archs rise toward oracle",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = OUT / "ntps_by_arch.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def fig_order_controls(rows, J=4):
    """A / A_shuffle / A_reverse at J=4 (order diagnostic)."""
    labels = [l for l in LABEL_ORDER
              if _get(rows, l, J, "A") is not None]
    if not labels:
        print(f"fig_order_controls: no rows for J={J}, skipping")
        return

    x = np.arange(len(labels))
    w = 0.27
    A_loc = 1.0 / J

    A = [_get(rows, l, J, "A") or 0.0 for l in labels]
    sh = [_get(rows, l, J, "A_shuffle") or 0.0 for l in labels]
    rv = [_get(rows, l, J, "A_reverse") or 0.0 for l in labels]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, A, w, label="A (ordered)", color="#1f77b4", alpha=0.85)
    ax.bar(x,      sh, w, label="A (shuffled)", color="#aaaaaa", alpha=0.85)
    ax.bar(x + w,  rv, w, label="A (reversed)", color="#d62728", alpha=0.85)
    ax.axhline(A_loc, color="k", ls=":", lw=1.5, label=f"chance = 1/J = {A_loc:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_DISPLAY.get(l, l) for l in labels], rotation=12, ha="right")
    ax.set_ylabel("linear-probe accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Order controls @ J={J}: A / A_shuffle / A_reverse\n"
        "Temporal archs: ordered≫chance, shuffle→chance; per-token: flat at chance",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = OUT / "order_controls.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def fig_a_vs_a_loc(rows):
    """Scatter: A vs A_loc for all cells (confirms per-token at chance)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in LABEL_ORDER:
        xs = [1.0 / r["J"] for r in rows if r["label"] == label and "A" in r]
        ys = [r["A"] for r in rows if r["label"] == label and "A" in r]
        if xs:
            ax.scatter(xs, ys, color=COLORS.get(label, "#333"), s=60,
                       label=LABEL_DISPLAY.get(label, label), zorder=3)

    lims = [0, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1, label="A = A_loc (chance)")
    ax.set_xlabel("A_loc (chance = 1/J)")
    ax.set_ylabel("A (linear probe, ordered)")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Probe accuracy vs A_loc across all J\n"
        "Above diagonal ⇒ arch uses temporal order",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p = OUT / "a_vs_a_loc.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def write_summary(rows) -> dict:
    """Write a summary JSON and print the headline table."""
    s: dict = {"n_rows": len(rows), "protocol": PROTO, "cells": {}}
    Js = sorted({r["J"] for r in rows if r["J"] is not None})
    print(f"\n{'arch':20s}  " + "  ".join(f"J={J} NTPS" for J in Js))
    print("-" * (24 + 14 * len(Js)))
    for label in LABEL_ORDER:
        row_vals = {}
        line = f"{label:20s}"
        for J in Js:
            v = _get(rows, label, J, "NTPS")
            ntps_str = f"{v:+.3f}" if v is not None else "     -"
            line += f"  {ntps_str:9s}"
            if v is not None:
                row_vals[f"J={J}"] = {
                    "NTPS": round(v, 4),
                    "A": round(_get(rows, label, J, "A") or 0, 4),
                    "A_shuffle": round(_get(rows, label, J, "A_shuffle") or 0, 4),
                    "A_reverse": round(_get(rows, label, J, "A_reverse") or 0, 4),
                    "NTPS_mlp": round(_get(rows, label, J, "NTPS_mlp") or 0, 4),
                }
        print(line)
        s["cells"][label] = row_vals

    p = OUT / "summary.json"
    json.dump(s, open(p, "w"), indent=2)
    print(f"\nsaved {p}")
    return s


if __name__ == "__main__":
    rows = load_rows()
    print(f"Loaded {len(rows)} direct-sum freq_bench rows @ protocol {PROTO}")
    if not rows:
        raise SystemExit("No rows yet — run the sweep first.")

    fig_ntps_by_arch(rows)
    fig_order_controls(rows, J=4)
    fig_a_vs_a_loc(rows)
    s = write_summary(rows)
    print(json.dumps(s, indent=2))
