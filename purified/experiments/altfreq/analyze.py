"""Analyse the altfreq sweep results from the leaderboard.

Reads results/leaderboard.jsonl, keeps rows for the four altfreq datasources
at the readout-optimal protocol (k_pos=1, d_sae=1024, protocol 1.2.0), and
renders three plot families into results/altfreq/:

  (a) ntps_by_bench.png     — NTPS by bench × arch (bar chart, main headline)
  (b) order_controls.png    — A / A_shuffle / A_reverse for chirp + relphase
                              (the signed-direction benches)
  (c) rj_curves.png         — per-class R_j frequency-response curves for
                              multitone + am (the multi-class benches)

Run after the sweep:
    .venv/bin/python experiments/altfreq/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
OUT = ROOT / "results" / "altfreq"
OUT.mkdir(parents=True, exist_ok=True)
PROTO = "1.2.0"

# The four W=16 altfreq datasources
AF_DATASOURCES = [
    "af_chirp_W16_s10",
    "af_multitone_W16_s10",
    "af_am_W16_s10",
    "af_relphase_W16_s10",
]
BENCH_LABELS = {
    "af_chirp_W16_s10":     "chirp",
    "af_multitone_W16_s10": "multitone",
    "af_am_W16_s10":        "am",
    "af_relphase_W16_s10":  "relphase",
}

ARCH_ORDER = ["topk_sae", "txcdr_t5", "txc_base_TW", "tfa"]
ARCH_COLORS = {
    "topk_sae":    "#888888",
    "txcdr_t5":    "#2ca02c",
    "txc_base_TW": "#9467bd",
    "tfa":         "#1f77b4",
}
ARCH_LABELS = {
    "topk_sae":    "topk_sae (per-token)",
    "txcdr_t5":    "txcdr_t5 (sliding T=5)",
    "txc_base_TW": "txc_base_TW (joint T=W)",
    "tfa":         "tfa",
}


def load_rows() -> list[dict]:
    """Load altfreq rows from leaderboard.jsonl."""
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
        ds = r.get("datasource", "")
        if ds not in AF_DATASOURCES:
            continue
        label = ec.get("label")
        if label is None:
            continue
        rows.append({
            "label": label,
            "datasource": ds,
            "bench": BENCH_LABELS.get(ds, ds),
            **r["metrics"],
        })
    return rows


def _get(rows, label, ds, key):
    for r in rows:
        if r["label"] == label and r["datasource"] == ds and key in r:
            return r[key]
    return None


def fig_ntps_by_bench(rows):
    """Bar chart: NTPS by bench × arch."""
    benches = ["chirp", "multitone", "am", "relphase"]
    ds_map = {v: k for k, v in BENCH_LABELS.items()}

    n_benches = len(benches)
    n_archs = len(ARCH_ORDER)
    x = np.arange(n_benches)
    width = 0.18
    offsets = np.linspace(-(n_archs - 1) / 2, (n_archs - 1) / 2, n_archs) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, arch in enumerate(ARCH_ORDER):
        vals = []
        for bench in benches:
            ds = ds_map[bench]
            v = _get(rows, arch, ds, "NTPS")
            vals.append(v if v is not None else float("nan"))
        ax.bar(x + offsets[idx], vals, width, label=ARCH_LABELS[arch],
               color=ARCH_COLORS[arch], alpha=0.85)

    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(benches, fontsize=12)
    ax.set_ylabel("NTPS (linear probe, k_pos=1, d_sae=1024)")
    ax.set_ylim(-0.15, 1.05)
    ax.set_title("Alternate-frequency benches — NTPS by bench × arch\n"
                 "NTPS=0: no better than one-token baseline; NTPS=1: matches oracle",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = OUT / "ntps_by_bench.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def fig_order_controls(rows):
    """A / A_shuffle / A_reverse for chirp + relphase (signed-direction benches)."""
    signed_benches = [
        ("chirp",    "af_chirp_W16_s10"),
        ("relphase", "af_relphase_W16_s10"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (bench, ds) in zip(axes, signed_benches):
        archs_present = [a for a in ARCH_ORDER
                         if _get(rows, a, ds, "A") is not None]
        x = np.arange(len(archs_present))
        w = 0.25
        A_vals  = [_get(rows, a, ds, "A") for a in archs_present]
        sh_vals = [_get(rows, a, ds, "A_shuffle") for a in archs_present]
        rv_vals = [_get(rows, a, ds, "A_reverse") for a in archs_present]
        ax.bar(x - w, A_vals,  w, label="A (ordered)",  color="#1f77b4", alpha=0.85)
        ax.bar(x,     sh_vals, w, label="A (shuffled)", color="#aaaaaa", alpha=0.85)
        ax.bar(x + w, rv_vals, w, label="A (reversed)", color="#d62728", alpha=0.85)
        ax.axhline(0.5, color="k", ls=":", lw=1.2, label="chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABELS.get(a, a) for a in archs_present],
                           fontsize=8, rotation=10)
        ax.set_ylabel("linear-probe accuracy")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{bench} bench — order controls\n"
                     "reverse < chance ⇒ signed direction encoding",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    plt.suptitle("Order controls for signed-direction benches (k_pos=1, d_sae=1024)",
                 fontsize=12)
    plt.tight_layout()
    p = OUT / "order_controls.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def fig_rj_curves(rows):
    """Per-class R_j frequency-response for multitone + am."""
    multi_benches = [
        ("multitone", "af_multitone_W16_s10", "Target tone velocity (ω)"),
        ("am",        "af_am_W16_s10",        "Modulation frequency (f_m)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (bench, ds, xlabel) in zip(axes, multi_benches):
        for arch in ARCH_ORDER:
            row = next((r for r in rows
                        if r["label"] == arch and r["datasource"] == ds), None)
            if row is None:
                continue
            # R_v keys: "R_v+1", "R_v+2", ...
            rv_keys = sorted(
                [k for k in row if k.startswith("R_v")],
                key=lambda k: int(k[3:]),
            )
            if not rv_keys:
                continue
            xs = [int(k[3:]) for k in rv_keys]
            ys = [row[k] for k in rv_keys]
            ax.plot(xs, ys, "o-", color=ARCH_COLORS[arch],
                    label=ARCH_LABELS[arch], lw=1.8)
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5, label="chance")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("R_j = (A_j − 1/n) / (1 − 1/n)")
        ax.set_title(f"{bench} bench — per-class frequency response R_j\n"
                     "above 0 ⇒ above-chance; shape = architecture's spectral profile",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Per-class frequency response (k_pos=1, d_sae=1024)", fontsize=12)
    plt.tight_layout()
    p = OUT / "rj_curves.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def write_summary_json(rows) -> dict:
    """Write a JSON summary of the sweep."""
    s = {"n_rows": len(rows), "protocol": PROTO, "cells": {}}
    for arch in ARCH_ORDER:
        arch_d = {}
        for ds in AF_DATASOURCES:
            bench = BENCH_LABELS[ds]
            r = next((x for x in rows
                      if x["label"] == arch and x["datasource"] == ds), None)
            if r is None:
                arch_d[bench] = None
                continue
            arch_d[bench] = {
                "NTPS": round(r.get("NTPS", float("nan")), 3),
                "A": round(r.get("A", float("nan")), 3),
                "A_shuffle": round(r.get("A_shuffle", float("nan")), 3),
                "A_reverse": round(r.get("A_reverse", float("nan")), 3),
                "NTPS_mlp": round(r.get("NTPS_mlp", float("nan")), 3),
                "order_gap": round(r.get("order_gap", float("nan")), 3),
                "reverse_drop": round(r.get("reverse_drop", float("nan")), 3),
            }
        s["cells"][arch] = arch_d
    p = OUT / "summary.json"
    with open(p, "w") as f:
        json.dump(s, f, indent=2)
    print("saved", p)
    return s


def print_ntps_table(s: dict):
    """Print a Markdown NTPS table."""
    benches = ["chirp", "multitone", "am", "relphase"]
    header = "| arch | " + " | ".join(benches) + " |"
    sep = "|---|" + "---|" * len(benches)
    print("\n## NTPS table (linear probe, k_pos=1, d_sae=1024)\n")
    print(header)
    print(sep)
    for arch in ARCH_ORDER:
        cells = s["cells"].get(arch, {})
        row = f"| {arch} |"
        for b in benches:
            v = cells.get(b)
            if v is None:
                row += " — |"
            else:
                ntps = v.get("NTPS", float("nan"))
                row += f" {ntps:+.3f} |"
        print(row)

    print("\n## A_reverse table\n")
    header2 = "| arch | chirp A_rev | relphase A_rev | chirp<chance? | relphase<chance? |"
    print(header2)
    print("|---|---|---|---|---|")
    for arch in ARCH_ORDER:
        cells = s["cells"].get(arch, {})
        ch = cells.get("chirp") or {}
        rp = cells.get("relphase") or {}
        ch_rev = ch.get("A_reverse", float("nan"))
        rp_rev = rp.get("A_reverse", float("nan"))
        ch_below = "YES" if ch_rev < 0.5 else "no"
        rp_below = "YES" if rp_rev < 0.5 else "no"
        print(f"| {arch} | {ch_rev:.3f} | {rp_rev:.3f} | {ch_below} | {rp_below} |")


if __name__ == "__main__":
    rows = load_rows()
    print(f"loaded {len(rows)} altfreq rows @ protocol {PROTO}")
    if not rows:
        raise SystemExit("no altfreq rows yet — run the sweep first.")
    s = write_summary_json(rows)
    fig_ntps_by_bench(rows)
    fig_order_controls(rows)
    fig_rj_curves(rows)
    print_ntps_table(s)
