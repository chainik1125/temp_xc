"""em-redo Phase A — analysis: tables, overlay figure, verdict quantities.

Reads (a) the em-redo leaderboard rows (evaluator "em", protocol 3.0.0,
the frozen datasources), (b) the probe-currency json, (c) the raw g(ℓ)
map (phase4_em_depth.json + posthoc_mean_decomp.json), and emits:

- results/tracking_summary.json — per (cell_id, layer): seed-mean /
  spread for pr_auc_S16, shuffle_gap_S16, realized l0_per_token, probe
  token/window AUC + advantage; plus δ(arch, layer) vs batchtopk_sae in
  both currencies (the frozen analysis rule).
- figs/tracking_overlay.{png,pdf} — left: paper currency by layer per
  arch; middle: probe currency (code token/window AUC vs the raw
  ceilings); right: temporal advantage δ per layer overlaid on raw g(ℓ)
  and g_order (the tracking test itself).

Analysis-only tooling (no protocol content — the frozen rules live in
TRACKING.md § 1–2).  Run after (or during) the panel:
  .venv/bin/python -m experiments.explorations.conversion_depth.render_tracking
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"
ROOT = HERE.parents[2]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"

DS2LAYER = {
    "qwen_2_5_7b_organism_medical_l9": 9,
    "qwen_2_5_7b_organism_medical_l13": 13,
    "qwen_2_5_7b_organism_medical_l15": 15,
}
LAYERS = [9, 13, 15]
# raw map (RECORD § 4): hs = layer+1
RAW = {
    9:  {"tok": 0.645, "win": 0.765, "g": 0.120, "g_order": 0.034},
    13: {"tok": 0.673, "win": 0.807, "g": 0.134, "g_order": 0.108},
    15: {"tok": 0.748, "win": 0.845, "g": 0.097, "g_order": 0.054},
}
REFERENCE = "batchtopk_sae"
CELL_ORDER = ["batchtopk_sae", "txc_post_k80", "txc_pre_k20", "txc_pre_k40",
              "tsae", "txc_base_anchor", "sae_arditi_anchor"]

# cell_id resolution from leaderboard rows: (arch, k_pos override) → cell_id
def _cell_id(row) -> str | None:
    arch = row["arch"]
    ovr = (row.get("training_cfg") or {}).get("arch_hparams_override") or {}
    if arch == "batchtopk_sae":
        return "batchtopk_sae"
    if arch == "txc_batchtopk_post":
        return "txc_post_k80"
    if arch == "txc_batchtopk_pre":
        return "txc_pre_k40" if ovr.get("k_pos") == 40 else "txc_pre_k20"
    if arch == "tsae":
        return "tsae"
    if arch == "txc_base":
        return "txc_base_anchor"
    if arch == "sae_arditi":
        return "sae_arditi_anchor"
    return None


def load_paper_currency():
    cells = defaultdict(dict)   # (cell_id, layer) -> seed -> metrics
    if not LEADERBOARD.exists():
        return cells
    for line in LEADERBOARD.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("evaluator_name") != "em":
            continue
        if r.get("evaluator_protocol_version") != "3.0.0":
            continue
        ds = r.get("datasource")
        if ds not in DS2LAYER:
            continue
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        cid = _cell_id(r)
        if cid is None:
            continue
        cells[(cid, DS2LAYER[ds])][int(r["seed"])] = r["metrics"]
    return cells


def load_probe_currency():
    p = RES / "em_redo_probe_codes.json"
    out = defaultdict(dict)     # (cell_id, layer) -> seed -> cell dict
    if not p.exists():
        return out
    data = json.loads(p.read_text())
    for key, cell in data.get("cells", {}).items():
        cid, lpart, spart = key.split("/")
        out[(cid, int(lpart[1:]))][int(spart[1:])] = cell
    return out


def seed_stats(d: dict[int, float]):
    v = [x for x in d.values() if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "min": float(np.min(v)),
            "max": float(np.max(v)), "n": len(v),
            "spread": float(np.max(v) - np.min(v))}


def main():
    paper = load_paper_currency()
    probe = load_probe_currency()

    summary = {}
    for (cid, layer), by_seed in sorted(paper.items()):
        e = {}
        e["pr_auc_S16"] = seed_stats(
            {s: m.get("pr_auc_S16") for s, m in by_seed.items()})
        e["pr_auc_S32"] = seed_stats(
            {s: m.get("pr_auc_S32") for s, m in by_seed.items()})
        e["shuffle_gap_S16"] = seed_stats(
            {s: m.get("shuffle_gap_S16") for s, m in by_seed.items()
             if m.get("shuffle_gap_S16") is not None})
        e["l0_per_token"] = seed_stats(
            {s: m.get("l0_per_token") for s, m in by_seed.items()})
        e["seeds"] = sorted(by_seed)
        summary[f"{cid}/L{layer}"] = e
    for (cid, layer), by_seed in sorted(probe.items()):
        e = summary.setdefault(f"{cid}/L{layer}", {})
        e["probe_token_auc"] = seed_stats(
            {s: c["token"]["auc"] for s, c in by_seed.items()})
        e["probe_window_auc"] = seed_stats(
            {s: c["window"]["auc"] for s, c in by_seed.items()})
        e["probe_advantage"] = seed_stats(
            {s: c["advantage"] for s, c in by_seed.items()})

    # δ(arch, layer) vs reference, per currency (frozen analysis rule)
    deltas = {}
    for metric, src in [("pr_auc_S16", "pr_auc_S16"),
                        ("probe_window_auc", "probe_window_auc")]:
        for cid in CELL_ORDER:
            if cid == REFERENCE:
                continue
            for layer in LAYERS:
                a = summary.get(f"{cid}/L{layer}", {}).get(src)
                b = summary.get(f"{REFERENCE}/L{layer}", {}).get(src)
                if a and b:
                    deltas[f"delta_{metric}/{cid}/L{layer}"] = round(
                        a["mean"] - b["mean"], 4)
    out = {"raw_map": RAW, "cells": summary, "deltas": deltas}
    RES.mkdir(exist_ok=True)
    (RES / "tracking_summary.json").write_text(json.dumps(out, indent=1))
    print(f"-> {RES / 'tracking_summary.json'}")

    # ── console table ──
    for cid in CELL_ORDER:
        for layer in LAYERS:
            e = summary.get(f"{cid}/L{layer}")
            if not e:
                continue
            pa = e.get("pr_auc_S16")
            sg = e.get("shuffle_gap_S16")
            l0 = e.get("l0_per_token")
            pt = e.get("probe_token_auc")
            pw = e.get("probe_window_auc")
            print(f"{cid:>18}/L{layer:<2} "
                  f"S16={pa['mean']:.3f}±{pa['spread']/2:.3f}" if pa else
                  f"{cid:>18}/L{layer:<2} S16=—", end="")
            print(f" gap16={sg['mean']:+.3f}" if sg else " gap16=  — ",
                  end="")
            print(f" l0/tok={l0['mean']:.0f}" if l0 else "", end="")
            if pt and pw:
                print(f" | probe tok={pt['mean']:.3f} win={pw['mean']:.3f}",
                      end="")
            print()

    # ── overlay figure ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"batchtopk_sae": "k", "txc_post_k80": "tab:red",
              "txc_pre_k20": "tab:orange", "txc_pre_k40": "tab:brown",
              "tsae": "tab:green", "txc_base_anchor": "tab:purple",
              "sae_arditi_anchor": "tab:gray"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for cid in CELL_ORDER:
        xs, ys, sp = [], [], []
        for layer in LAYERS:
            pa = summary.get(f"{cid}/L{layer}", {}).get("pr_auc_S16")
            if pa:
                xs.append(layer); ys.append(pa["mean"]); sp.append(
                    pa["spread"] / 2)
        if xs:
            ax.errorbar(xs, ys, yerr=sp, marker="o", label=cid,
                        color=colors.get(cid), lw=1.5, capsize=3)
    ax.set_title("paper currency: sparse-probe PR-AUC (S=16)")
    ax.set_xlabel("resid_post layer"); ax.set_xticks(LAYERS)
    ax.axhline(0.323, color="gray", ls=":", lw=1, label="base rate")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(LAYERS, [RAW[l]["tok"] for l in LAYERS], "s--", color="gray",
            label="raw per-token (RECORD §4)")
    ax.plot(LAYERS, [RAW[l]["win"] for l in LAYERS], "s--", color="black",
            label="raw window (RECORD §4)")
    for cid in CELL_ORDER:
        xs, yt, yw = [], [], []
        for layer in LAYERS:
            pt = summary.get(f"{cid}/L{layer}", {}).get("probe_token_auc")
            pw = summary.get(f"{cid}/L{layer}", {}).get("probe_window_auc")
            if pt and pw:
                xs.append(layer); yt.append(pt["mean"]); yw.append(pw["mean"])
        if xs:
            ax.plot(xs, yw, "o-", color=colors.get(cid), lw=1.5,
                    label=f"{cid} win-code")
            ax.plot(xs, yt, "o:", color=colors.get(cid), lw=1, alpha=0.6)
    ax.set_title("probe currency: code AUC (dotted=token, solid=window)")
    ax.set_xlabel("resid_post layer"); ax.set_xticks(LAYERS)
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(LAYERS, [RAW[l]["g"] for l in LAYERS], "s--", color="black",
            label="raw g(ℓ)")
    ax.plot(LAYERS, [RAW[l]["g_order"] for l in LAYERS], "s--",
            color="gray", label="raw g_order(ℓ)")
    for cid in CELL_ORDER:
        if cid == REFERENCE:
            continue
        xs, ys = [], []
        for layer in LAYERS:
            d = deltas.get(f"delta_pr_auc_S16/{cid}/L{layer}")
            if d is not None:
                xs.append(layer); ys.append(d)
        if xs:
            ax.plot(xs, ys, "o-", color=colors.get(cid), lw=1.5,
                    label=f"δ S16 {cid}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("tracking test: temporal advantage δ vs raw headroom")
    ax.set_xlabel("resid_post layer"); ax.set_xticks(LAYERS)
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGS / f"tracking_overlay.{ext}", dpi=160)
    print(f"-> {FIGS / 'tracking_overlay.png'}")


if __name__ == "__main__":
    main()
