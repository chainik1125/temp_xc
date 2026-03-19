"""Regenerate Experiment 1 and 2 plots with all models including TXCDR T=2 and T=5.

Reads data from:
  - results/auc_and_crosscoder/results.json   (SAE, TFA, TFA-shuf, TXCDR T=2 — TopK + L1)
  - results/txcdr_T5/results.json             (TXCDR T=5 — TopK + L1)

Saves to:
  - results/auc_and_crosscoder/exp1_topk_auc.{png,pdf}
  - results/auc_and_crosscoder/exp1_topk_auc.thumb.png
  - results/auc_and_crosscoder/exp2_pareto_auc.{png,pdf}
  - results/auc_and_crosscoder/exp2_pareto_auc.thumb.png

Usage:
  TQDM_DISABLE=1 python src/v2_temporal_schemeC/plot_exp1_exp2.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot import save_figure

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "auc_and_crosscoder")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def pareto_min(xs, ys):
    """Lower-envelope Pareto frontier (minimize y)."""
    pts = sorted(zip(xs, ys))
    fx, fy = [], []
    best = float("inf")
    for x, y in pts:
        if y < best:
            fx.append(x); fy.append(y); best = y
    return fx, fy


def pareto_max(xs, ys):
    """Upper-envelope Pareto frontier (maximize y)."""
    pts = sorted(zip(xs, ys))
    fx, fy = [], []
    best = -float("inf")
    for x, y in pts:
        if y > best:
            fx.append(x); fy.append(y); best = y
    return fx, fy


def main():
    base = os.path.dirname(__file__)

    # ── Load data ─────────────────────────────────────────────────────
    main_data = load_json(os.path.join(base, "results", "auc_and_crosscoder", "results.json"))
    # v2 has TXCDR T=2 L1 sweep data that the original doesn't
    v2_path = os.path.join(base, "results", "auc_and_crosscoder_v2", "results.json")
    v2_data = load_json(v2_path) if os.path.exists(v2_path) else {}
    t5_path = os.path.join(base, "results", "txcdr_T5", "results.json")
    t5_data = load_json(t5_path) if os.path.exists(t5_path) else None

    exp1 = main_data["exp1"]  # keys: sae, tfa, tfa_shuf, txcdr
    exp2_sae = main_data.get("exp2_sae", [])
    exp2_tfa = main_data.get("exp2_tfa", [])
    exp2_txcdr_t2 = v2_data.get("exp2_txcdr", main_data.get("exp2_txcdr", []))

    # T=5 TopK and L1
    t5_topk = t5_data["topk"] if t5_data else []
    t5_l1 = t5_data["l1"] if t5_data else []

    # TFA-pos data
    pos_path = os.path.join(base, "results", "tfa_pos", "results.json")
    pos_data = load_json(pos_path) if os.path.exists(pos_path) else {}
    exp1_pos = pos_data.get("exp1_tfa_pos", [])
    exp1_pos_shuf = pos_data.get("exp1_tfa_pos_shuf", [])
    exp2_pos = pos_data.get("exp2_tfa_pos", [])

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: TopK sweep — NMSE, AUC, NMSE-vs-AUC
    # ══════════════════════════════════════════════════════════════════

    series = [
        ("SAE",         exp1["sae"],      "tab:blue",   "o", "-"),
        ("TFA",         exp1["tfa"],      "tab:orange", "s", "-"),
        ("TFA-shuf",    exp1["tfa_shuf"], "tab:red",    "^", "--"),
        ("TXCDR T=2",   exp1["txcdr"],    "tab:purple", "D", "-."),
    ]
    if t5_topk:
        series.append(("TXCDR T=5", t5_topk, "tab:green", "P", ":"))
    if exp1_pos:
        series.append(("TFA-pos", exp1_pos, "tab:brown", "X", "-"))
    if exp1_pos_shuf:
        series.append(("TFA-pos-shuf", exp1_pos_shuf, "tab:pink", "v", "--"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # NMSE vs k
    ax = axes[0]
    for label, data, c, m, ls in series:
        ks = [r.get("k", r.get("novel_l0", 0)) for r in data]
        nmse = [r["nmse"] for r in data]
        ax.plot(ks, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=label)
    ax.set_xlabel("k")
    ax.set_ylabel("NMSE")
    ax.set_title("NMSE vs k")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # AUC vs k
    ax = axes[1]
    for label, data, c, m, ls in series:
        ks = [r.get("k", r.get("novel_l0", 0)) for r in data]
        auc = [r["auc"] for r in data]
        ax.plot(ks, auc, marker=m, linestyle=ls, color=c, lw=2, ms=7, label=label)
    ax.set_xlabel("k")
    ax.set_ylabel("AUC")
    ax.set_title("Feature Recovery AUC vs k")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # NMSE vs AUC scatter
    ax = axes[2]
    for label, data, c, m, _ in series:
        nmse = [r["nmse"] for r in data]
        auc = [r["auc"] for r in data]
        ax.scatter(nmse, auc, color=c, marker=m, s=60, label=label, zorder=3)
        ax.plot(nmse, auc, color=c, alpha=0.3, lw=1)
    ax.set_xlabel("NMSE")
    ax.set_ylabel("AUC")
    ax.set_title("NMSE vs AUC")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.suptitle("Experiment 1: TopK sweep — NMSE and Feature Recovery AUC", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(RESULTS_DIR, f"exp1_topk_auc.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Experiment 1 plots saved.")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: ReLU+L1 Pareto — L0 vs NMSE, L0 vs AUC, NMSE vs AUC
    # ══════════════════════════════════════════════════════════════════

    l1_series = [
        ("ReLU SAE", exp2_sae, "l0", "tab:blue", "o"),
        ("TFA (novel L0)", exp2_tfa, "novel_l0", "tab:orange", "s"),
    ]
    if exp2_txcdr_t2:
        l1_series.append(("TXCDR T=2", exp2_txcdr_t2, "l0", "tab:purple", "D"))
    if t5_l1:
        l1_series.append(("TXCDR T=5", t5_l1, "l0", "tab:green", "P"))
    if exp2_pos:
        l1_series.append(("TFA-pos (novel L0)", exp2_pos, "novel_l0", "tab:brown", "X"))

    # Total L0 curves (dashed) — TFA and TFA-pos from dedicated run
    total_l0_series = []
    tl0_path = os.path.join(base, "results", "tfa_l1_total_l0", "results.json")
    if os.path.exists(tl0_path):
        tl0_data = load_json(tl0_path)
        if tl0_data.get("tfa"):
            total_l0_series.append(("TFA (total L0)", tl0_data["tfa"], "total_l0", "tab:red", "s"))
        if tl0_data.get("tfa_pos"):
            total_l0_series.append(("TFA-pos (total L0)", tl0_data["tfa_pos"], "total_l0", "tab:pink", "X"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # L0 vs NMSE
    ax = axes[0]
    for label, data, l0_key, c, m in l1_series:
        l0s = [r[l0_key] for r in data]
        nmse = [r["nmse"] for r in data]
        ax.scatter(l0s, nmse, color=c, alpha=0.3, s=30)
        fx, fy = pareto_min(l0s, nmse)
        ax.plot(fx, fy, f"{m}-", color=c, lw=2, ms=7, label=label)
    for label, data, l0_key, c, m in total_l0_series:
        l0s = [r[l0_key] for r in data]
        nmse = [r["nmse"] for r in data]
        fx, fy = pareto_min(l0s, nmse)
        ax.plot(fx, fy, f"{m}--", color=c, lw=1.5, ms=5, alpha=0.7, label=label)
    ax.set_xlabel("L0")
    ax.set_ylabel("NMSE")
    ax.set_title("L0 vs NMSE Pareto")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # L0 vs AUC
    ax = axes[1]
    for label, data, l0_key, c, m in l1_series:
        l0s = [r[l0_key] for r in data]
        auc = [r["auc"] for r in data]
        ax.scatter(l0s, auc, color=c, alpha=0.3, s=30)
        fx, fy = pareto_max(l0s, auc)
        ax.plot(fx, fy, f"{m}-", color=c, lw=2, ms=7, label=label)
    for label, data, l0_key, c, m in total_l0_series:
        if "auc" not in data[0]:
            continue
        l0s = [r[l0_key] for r in data]
        auc = [r["auc"] for r in data]
        fx, fy = pareto_max(l0s, auc)
        ax.plot(fx, fy, f"{m}--", color=c, lw=1.5, ms=5, alpha=0.7, label=label)
    ax.set_xlabel("L0")
    ax.set_ylabel("AUC")
    ax.set_title("L0 vs AUC Pareto")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # NMSE vs AUC — connect points along L1 sweep (sorted by NMSE)
    ax = axes[2]
    for label, data, _, c, m in l1_series:
        pts = sorted(zip([r["nmse"] for r in data], [r["auc"] for r in data]))
        nmse_s, auc_s = zip(*pts) if pts else ([], [])
        ax.plot(nmse_s, auc_s, marker=m, linestyle="-", color=c, lw=1.5, ms=6,
                alpha=0.8, label=label)
    for label, data, _, c, m in total_l0_series:
        if "auc" not in data[0]:
            continue
        pts = sorted(zip([r["nmse"] for r in data], [r["auc"] for r in data]))
        nmse_s, auc_s = zip(*pts) if pts else ([], [])
        ax.plot(nmse_s, auc_s, marker=m, linestyle="--", color=c, lw=1.5, ms=5,
                alpha=0.7, label=label)
    ax.set_xlabel("NMSE")
    ax.set_ylabel("AUC")
    ax.set_title("NMSE vs AUC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.suptitle("Experiment 2: ReLU+L1 Pareto with AUC", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(RESULTS_DIR, f"exp2_pareto_auc.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Experiment 2 plots saved.")


if __name__ == "__main__":
    main()
