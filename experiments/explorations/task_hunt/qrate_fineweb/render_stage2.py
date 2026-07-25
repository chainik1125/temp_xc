"""Stage-2 fineweb punctint-q — the T-scaling money plot (card § 11 item 4).

The λ̂ panel's `lambda_intensity/render_stage2.py` conventions (b's
variance-aware upgrade: 95% t CI whiskers via `stats_lib.t_ci95`,
realized-l0 ranges in the legend, budget-matched flags), applied to the
fineweb panel, plus the card's fineweb-specific overlays:

- **doc-identity floor** (§ 6a, `stage2_support` receipt) drawn as a
  horizontal line — the identity route's ceiling; every window number
  reads against it;
- **evidence line** (§ 7): the in-tile visible-q-count regression r per
  T, drawn as a dotted ladder — a window cell below it is not reading
  more than visible question marks;
- corpus size in the caption (binding 6);
- a **paired-v2 variant** figure (same layout, `lambda_recovery_v2`,
  titled as paired/non-canonical per the 2026-07-25 METHODS DECISION).

TXC-post here is matched BY CONSTRUCTION (nominal k = 8·T from the
start, card § 4) — there is no unmatched post arm, so there is no
"_matched" split; the § 5 falsifier + band bookkeeping live in the
summary JSON (`residual_mismatches`).

Colors: the λ̂ figure's arch identities, with two accessibility
substitutions checked pairwise under normal/protan/deutan vision in
OKLab (session receipt): stacked green → light blue (green↔orange
collapsed to ΔE 4.2 under protanopia against the headline TXC-pre
orange) and token gray darkened. The one remaining sub-target pair
(token gray ↔ tsae brown, protan ΔE 7.8) is carried by secondary
encoding: distinct markers (o/v) and solid-vs-dashed reference lines.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.qrate_fineweb.render_stage2 [ds]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.support_stats.stats_lib import t_ci95

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"

DS_DEFAULT = "fineweb_punctint_q_gemma2_l14"
WINDOW_ARCHS = {
    "stacked_batchtopk": ("#9ecae1", "^-", "Stacked (wide probe, disclosed)"),
    "txc_batchtopk_pre": ("#ff7f0e", "s-", "TXC-pre"),
    "txc_batchtopk_post": ("#1f77b4", "D-", "TXC-post (matched k=8·T)"),
}
TOKEN_ARCHS = {
    "batchtopk_sae": ("#4d4d4d", "o", "-", "per-token BatchTopK SAE"),
    "tsae": ("#8c564b", "v", "--", "T-SAE"),
}
L0_BAND = (5.0, 8.0)


def _load(ds: str, metric: str):
    rows = json.loads((RES / f"stage2_{ds}.json").read_text())
    cells = defaultdict(list)     # (arch, T, kind) -> [(seed, r, l0)]
    for c in rows:
        if not c.get("ok"):
            continue
        m = c["metrics"]
        cells[(c["arch"], c["T"], c["kind"])].append(
            (c["seed"], m.get(metric, float("nan")),
             m.get("l0_per_token", float("nan"))))
    return cells


def _support(ds: str):
    p = RES / f"stage2_support_{ds}.json"
    return json.loads(p.read_text()) if p.exists() else None


def render(ds: str, metric: str, tag: str, title_note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = _load(ds, metric)
    sup = _support(ds)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    Ts = [2, 4, 8, 16]

    summary = {"ds": ds, "metric": metric, "cells": {},
               "residual_mismatches": [], "untrained_post_falsifier": []}

    for arch, (col, mk, ls, label) in TOKEN_ARCHS.items():
        vals = [r for _, r, _ in cells[(arch, 1, "trained")]]
        l0s = [l for _, _, l in cells[(arch, 1, "trained")]]
        m, lo, hi = t_ci95(vals)
        ax.axhline(m, color=col, linestyle=ls, linewidth=1.6, zorder=1)
        ax.fill_between([1.8, 17.8], lo, hi, color=col, alpha=0.10, zorder=0)
        ax.plot([1.9], [m], marker=mk, color=col, markersize=7, zorder=3)
        ax.annotate(f"{label}  {m:.3f}", xy=(1.92, m), xytext=(0, 5),
                    textcoords="offset points", fontsize=8, color=col)
        summary["cells"][f"{arch}/T1"] = {
            "mean": m, "ci95": [lo, hi],
            "l0_range": [min(l0s), max(l0s)]}

    for arch, (col, style, label) in WINDOW_ARCHS.items():
        means, los, his, l0_all = [], [], [], []
        for T in Ts:
            vals = [r for _, r, _ in cells[(arch, T, "trained")]]
            l0s = [l for _, _, l in cells[(arch, T, "trained")]]
            m, lo, hi = t_ci95(vals)
            means.append(m); los.append(lo); his.append(hi)
            l0_all += l0s
            summary["cells"][f"{arch}/T{T}"] = {
                "mean": m, "ci95": [lo, hi],
                "l0_range": [min(l0s), max(l0s)]}
            for s, _, l in cells[(arch, T, "trained")]:
                if not (L0_BAND[0] <= l <= L0_BAND[1]):
                    summary["residual_mismatches"].append(
                        {"arch": arch, "T": T, "seed": s, "l0": l})
        mk = style[0]
        ax.errorbar(Ts, means, yerr=[np.array(means) - np.array(los),
                                     np.array(his) - np.array(means)],
                    fmt=style, color=col, capsize=3, linewidth=2,
                    markersize=7, label=f"{label}  "
                    f"[l0 {min(l0_all):.2f}–{max(l0_all):.2f}]", zorder=4)
        # untrained control, faint
        um = [t_ci95([r for _, r, _ in cells[(arch, T, "untrained")]])[0]
              for T in Ts]
        ax.plot(Ts, um, mk + ":", color=col, alpha=0.35, markersize=4,
                linewidth=1, zorder=2)

    for arch in ("txc_batchtopk_post",):
        for T in Ts:
            for s, _, l in cells[(arch, T, "untrained")]:
                summary["untrained_post_falsifier"].append(
                    {"T": T, "seed": s, "l0": l,
                     "pass": bool(abs(l - 8.0) <= 0.02)})

    if sup and metric == "lambda_recovery":
        floor = [sup["per_T"][str(T)]["doc_floor_r"] for T in Ts]
        ax.axhline(np.mean(floor), color="#333333", linestyle="-.",
                   linewidth=1.2, zorder=1)
        ax.annotate(f"doc-mean identity floor  r≈{np.mean(floor):.2f}",
                    xy=(1.92, np.mean(floor)), xytext=(0, 4),
                    textcoords="offset points", fontsize=8, color="#333333")
        ev = [sup["per_T"][str(T)]["evidence_count_r"] for T in Ts]
        ax.plot(Ts, ev, ".", linestyle=":", color="#666666", linewidth=1.2,
                markersize=5, zorder=2)
        ax.annotate("visible q-count regression (§7)", xy=(Ts[-1], ev[-1]),
                    xytext=(-4, 6), textcoords="offset points", fontsize=8,
                    color="#666666", ha="right")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts); ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel("tile width T (tokens)")
    ax.set_ylabel(f"λ probe held-out r ({metric})")
    ax.set_title(f"punctint-q fineweb Stage-2 panel — gemma-2-2b hs14 "
                 f"{title_note}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.15)
    fig.text(0.01, 0.005,
             "corpus: pinned 400-doc fineweb sample, 766,080 tokens "
             "(ODC-By); 3 seeds; whiskers = 95% t CI; dotted faint = "
             "untrained controls; d_sae 1152, k_pos 8 (post 8·T)",
             fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGS.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"stage2_tscaling{tag}.{ext}", dpi=200)
    plt.close(fig)
    return summary


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else DS_DEFAULT
    s1 = render(ds, "lambda_recovery", "", "(v1, canonical)")
    s2 = render(ds, "lambda_recovery_v2", "_v2",
                "(paired v2 — NOT canonical, reported per methods decision)")
    out = {"v1": s1, "v2": s2}
    (RES / f"stage2_summary_{ds}.json").write_text(json.dumps(out, indent=2))
    n_mm = len(s1["residual_mismatches"])
    n_f = sum(1 for r in s1["untrained_post_falsifier"] if not r["pass"])
    print(f"[render] mismatches {n_mm}/42 trained cells; "
          f"untrained-post falsifier fails: {n_f}/12")
    print(f"-> {FIGS}/stage2_tscaling{{,_v2}}.png|pdf ; "
          f"{RES}/stage2_summary_{ds}.json")


if __name__ == "__main__":
    main()
