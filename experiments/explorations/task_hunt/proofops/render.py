"""Candidate 2 — score the proof-op screen against the frozen card.

Reads results/proofops_screen.json and emits:
  results/proofops_verdict.json — sigma_null, the per-cell ladders, the
      g_tir − g_op CONTRAST (the card's actual claim), and every frozen
      prediction / kill rule scored TRUE/FALSE;
  figs/proofops_tscaling.{png,pdf} — g(T) for tir vs the op ambient
      anchor, plus the within-window shuffle gap vs T.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.proofops.render
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"
TS = [8, 16, 32, 64]
AUC = "auc_macro_ovr"


def ladder(cells, model, hs, tgt):
    tok = cells.get(f"{model}/hs{hs}/{tgt}/tok")
    if tok is None:
        return None
    out = {"tok": tok["linear"][AUC], "T": {}}
    for T in TS:
        c = cells.get(f"{model}/hs{hs}/{tgt}/T{T}")
        if c is None:
            continue
        out["T"][T] = {
            "flat": c["flat"][AUC], "mean": c["mean"][AUC],
            "shuf": c["shuf"][AUC], "g": c["g"], "g_agg": c["g_agg"],
            "g_order": c["g_order"], "shuffle_gap": c["shuffle_gap"],
        }
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    d = json.loads((RES / "proofops_screen.json").read_text())
    cells = d["cells"]

    nulls = []
    for v in cells.values():
        for k in ("null", "null_flat"):
            if k in v:
                nulls.append(abs(v[k][AUC] - 0.5))
    sn = float(np.std(nulls)) if nulls else float("nan")
    three = 3 * sn

    lads = {}
    for model in ("base", "distill"):
        for hs in (13, 11):
            for tgt in ("tir", "boundary", "op"):
                L = ladder(cells, model, hs, tgt)
                if L:
                    lads[f"{model}/hs{hs}/{tgt}"] = L

    # The card's claim: the CONTRAST g_tir(T) - g_op(T) rises with T.
    contrasts = {}
    for model in ("base", "distill"):
        for hs in (13, 11):
            t = lads.get(f"{model}/hs{hs}/tir")
            o = lads.get(f"{model}/hs{hs}/op")
            if not (t and o):
                continue
            c = {T: t["T"][T]["g"] - o["T"][T]["g"]
                 for T in TS if T in t["T"] and T in o["T"]}
            if c:
                contrasts[f"{model}/hs{hs}"] = c

    complete = {k: v for k, v in contrasts.items() if len(v) == len(TS)}
    prim = {k: lads[f"{k}/tir"] for k in complete if f"{k}/tir" in lads}

    def gser(L):
        return [L["T"][T]["g"] for T in TS if T in L["T"]]

    def sgser(L):
        return [L["T"][T]["shuffle_gap"] for T in TS if T in L["T"]]

    p1 = all(
        all(L["T"][T]["g"] <= three for T in (8, 16) if T in L["T"])
        and all(L["T"][T]["g"] > three for T in (32, 64) if T in L["T"])
        for L in prim.values()) if prim else None
    p2 = (all(max(c.values()) > three and
              c[TS[-1]] > c[TS[0]] for c in complete.values())
          if complete else None)
    p3 = all(all(L["T"][T]["g_order"] > 0 for T in (32, 64) if T in L["T"])
             for L in prim.values()) if prim else None
    p5 = None
    b, ds = contrasts.get("base/hs13"), contrasts.get("distill/hs13")
    if b and ds:
        common = set(b) & set(ds)
        if common:
            p5 = max(abs(b[T] - ds[T]) for T in common) <= 0.02

    k = {}
    k["K1_no_window_access"] = (
        all(all(g <= three for g in gser(L)) for L in prim.values())
        if prim else None)
    # K2 as WRITTEN: kill iff g_tir never exceeds g_op by more than the
    # null floor at ANY T.
    k["K2_contrast_never_clears_null"] = (
        all(max(c.values()) <= three for c in complete.values())
        if complete else None)
    k["K3_flat_or_nonmonotone_above_clock"] = (
        all(L["T"].get(64, {}).get("g", -1) <= L["T"].get(32, {}).get("g", 0)
            for L in prim.values()) if prim else None)
    k["K4_run_depth_ambient"] = (
        all(all(L["T"][T]["flat"] - L["tok"] <= 0.02 for T in L["T"])
            for L in prim.values()) if prim else None)

    verdict = {
        "sigma_null": sn, "3sigma_null": three, "n_null_cells": len(nulls),
        "ladders": lads, "contrast_g_tir_minus_g_op": contrasts,
        "predictions": {
            "P1_threshold_at_clock (nothing<=16, clears at 32/64)": p1,
            "P2_contrast_rises_and_clears_null": p2,
            "P3_positive_order_at_T>=32": p3,
            "P5_base_approx_distill_contrast": p5,
        },
        "kill_rules": k,
        "KILLED": any(v for v in k.values() if v is not None),
        "shuffle_gap_ladders": {
            key: {T: L["T"][T]["shuffle_gap"] for T in TS if T in L["T"]}
            for key, L in lads.items()},
        "shuffle_gap_monotone_in_T": {
            key: bool(np.all(np.diff(sgser(L)) > -1e-9))
            for key, L in lads.items() if len(sgser(L)) == len(TS)},
    }
    (RES / "proofops_verdict.json").write_text(json.dumps(verdict, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for key, style, col in (("tir", "o-", "#1f77b4"),
                            ("boundary", "^--", "#2ca02c"),
                            ("op", "s:", "#7f7f7f")):
        L = lads.get(f"base/hs13/{key}")
        if not L:
            continue
        Ts = [T for T in TS if T in L["T"]]
        lab = f"{key}" + (" (AMBIENT anchor)" if key == "op" else "")
        ax[0].plot(Ts, [L["T"][T]["g"] for T in Ts], style, color=col,
                   lw=2, label=lab)
        ax[1].plot(Ts, [L["T"][T]["shuffle_gap"] for T in Ts], style,
                   color=col, lw=2, label=lab)
    for a, ttl, yl in (
            (ax[0], "window − per-token gap g(T)", "ΔAUC (macro OvR)"),
            (ax[1], "ordered − SHUFFLED window", "ΔAUC (macro OvR)")):
        if np.isfinite(three):
            a.axhspan(-three, three, color="grey", alpha=0.18,
                      label="±3σ_null")
        a.axhline(0, color="k", lw=0.6)
        a.set_xscale("log", base=2)
        a.set_xticks(TS); a.set_xticklabels(TS)
        a.set_xlabel("window size T (tokens)")
        a.set_ylabel(yl); a.set_title(ttl, fontsize=11)
        a.grid(True, alpha=0.25); a.legend(fontsize=8)
    fig.suptitle("Candidate 2 — proof-operation run structure, base L12 "
                 "(Stage-1 screen, frozen card)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"proofops_tscaling.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "sigma_null": sn, "3sigma": three,
        "contrast": contrasts, "predictions": verdict["predictions"],
        "kill_rules": k, "KILLED": verdict["KILLED"],
        "shuffle_gap_monotone": verdict["shuffle_gap_monotone_in_T"],
    }, indent=2))
    print(f"-> {RES}/proofops_verdict.json ; {FIGS}/proofops_tscaling.*")


if __name__ == "__main__":
    main()
