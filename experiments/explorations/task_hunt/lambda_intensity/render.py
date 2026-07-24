"""Candidate 1 — render the T-scaling figure + verdict quantities.

Reads results/lambda_screen.json (produced by the frozen `screen.py`)
and emits:
  figs/lambda_tscaling.{png,pdf}  — g(T) per (model, layer) for the
      primary lam_hist and the secondary lam_hat, with the 3 sigma_null
      band and the position-only floor annotated;
  figs/lambda_decomp.{png,pdf}    — g_agg vs g_order vs shuffle gap;
  results/lambda_verdict.json     — sigma_null, the per-cell g ladders,
      and each frozen prediction/kill-rule scored TRUE/FALSE.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.lambda_intensity.render
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"
TS = [2, 4, 8, 16, 32]


def load():
    return json.loads((RES / "lambda_screen.json").read_text())


def sigma_null(cells):
    vals = []
    for v in cells.values():
        for k in ("null", "null_flat"):
            if k in v:
                vals.append(abs(v[k]["auc"] - 0.5))
    return float(np.std(vals)), len(vals)


def ladder(cells, model, hs, tgt):
    tok = cells.get(f"{model}/hs{hs}/{tgt}/tok")
    if tok is None:
        return None
    out = {"tok": tok["linear"]["auc"], "T": {}}
    for T in TS:
        c = cells.get(f"{model}/hs{hs}/{tgt}/T{T}")
        if c is None:
            continue
        # window ceiling = best of flatten/mean (the flatten probe pays a
        # T-fold dimensionality cost at fixed frozen epochs; RECORD § 3
        # notes the same instability). Reported alongside raw flatten.
        out["T"][T] = {
            "flat": c["flat"]["auc"], "mean": c["mean"]["auc"],
            "shuf": c["shuf"]["auc"], "ceil": max(c["flat"]["auc"],
                                                  c["mean"]["auc"]),
            "g_flat": c["g"], "g_agg": c["g_agg"], "g_order": c["g_order"],
            "shuffle_gap": c["shuffle_gap"],
        }
        out["T"][T]["g_ceil"] = out["T"][T]["ceil"] - out["tok"]
    return out


def main():
    FIGS.mkdir(exist_ok=True)
    d = load()
    cells, floors = d["cells"], d["floors"]
    sn, n_null = sigma_null(cells)
    three = 3 * sn

    lads = {}
    for model in ["base", "distill"]:
        for hs in [13, 11]:
            for tgt in ["lam_hist", "lam_hat"]:
                L = ladder(cells, model, hs, tgt)
                if L:
                    lads[f"{model}/hs{hs}/{tgt}"] = L

    # ── verdict scoring against the frozen card ──────────────────────
    prim = {k: v for k, v in lads.items() if k.endswith("lam_hist")}
    def gser(L, key="g_ceil"):
        return [L["T"][T][key] for T in TS if T in L["T"]]

    verdict = {"sigma_null": sn, "3sigma_null": three, "n_null_cells": n_null,
               "floors": floors, "ladders": lads, "predictions": {}}

    p1 = all(all(g > three for T, g in zip(TS, gser(L)) if T >= 8)
             for L in prim.values())
    p1_flat = all(all(L["T"][T]["g_flat"] > three
                      for T in TS if T >= 8 and T in L["T"])
                  for L in prim.values())
    p2 = all(np.all(np.diff(gser(L)) > -1e-9) and gser(L)[-1] > gser(L)[-2]
             for L in prim.values())
    p3_agg = all(all(L["T"][T]["g_agg"] >= 0.5 * L["T"][T]["g_ceil"]
                     for T in L["T"]) for L in prim.values())
    p3_ord = all(any(L["T"][T]["g_order"] > 0 for T in L["T"] if T >= 16)
                 for L in prim.values())
    b = prim.get("base/hs13/lam_hist"); ds = prim.get("distill/hs13/lam_hist")
    p4 = (abs(b["T"][32]["ceil"] - ds["T"][32]["ceil"]) <= 0.02
          if b and ds and 32 in b["T"] and 32 in ds["T"] else None)
    p5 = all(L["T"][16]["ceil"] - floors["lam_hist"]["auc"] >= 0.10
             and L["tok"] > floors["lam_hist"]["auc"]
             for L in prim.values() if 16 in L["T"])
    verdict["predictions"] = {
        "P1_gap_beyond_3sigma_at_T>=8 (window ceiling)": bool(p1),
        "P1_same_on_raw_flatten": bool(p1_flat),
        "P2_monotone_rising_no_saturation": bool(p2),
        "P3_aggregation_dominant": bool(p3_agg),
        "P3_positive_order_component_at_T>=16": bool(p3_ord),
        "P4_base_approx_distill": p4,
        "P5_floor_clearance": bool(p5),
    }
    k = {}
    k["K1_no_window_access"] = all(all(g <= three for g in gser(L))
                                   for L in prim.values())
    k["K2_no_T_growth"] = not p2
    k["K3_floor_not_cleared"] = any(
        L["T"][16]["ceil"] - floors["lam_hist"]["auc"] < 0.05
        for L in prim.values() if 16 in L["T"])
    k["K4_pure_static_aggregation_no_T_response"] = (
        (not p3_ord) and (not p2))
    verdict["kill_rules"] = k
    verdict["KILLED"] = any(k.values())
    (RES / "lambda_verdict.json").write_text(json.dumps(verdict, indent=2))

    # ── figures ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    styles = {"base": ("#1f77b4", "o-"), "distill": ("#d62728", "s--")}
    for ax, tgt, title in [
            (axes[0], "lam_hist", "PRIMARY  λ̂_hist (kernel-only)"),
            (axes[1], "lam_hat", "secondary  λ̂ (with position ramp)")]:
        for model in ["base", "distill"]:
            for hs, alpha in [(13, 1.0), (11, 0.45)]:
                L = lads.get(f"{model}/hs{hs}/{tgt}")
                if not L:
                    continue
                col, mk = styles[model]
                Ts = [T for T in TS if T in L["T"]]
                ax.plot(Ts, [L["T"][T]["g_ceil"] for T in Ts], mk, color=col,
                        alpha=alpha, lw=2 if hs == 13 else 1.2,
                        label=f"{model} L{hs-1}")
        ax.axhspan(-three, three, color="grey", alpha=0.18,
                   label="±3σ_null")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(TS); ax.set_xticklabels(TS)
        ax.set_xlabel("window size T (tokens)")
        ax.set_ylabel("g(T) = AUC(window ceiling) − AUC(per-token)")
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("Candidate 1 — backtracking intensity λ̂: raw window "
                 "access vs T (Stage-1 screen, frozen card)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"lambda_tscaling.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    L = lads.get("base/hs13/lam_hist")
    if L:
        Ts = [T for T in TS if T in L["T"]]
        ax.plot(Ts, [L["T"][T]["g_agg"] for T in Ts], "o-", color="#2ca02c",
                lw=2, label="g_agg  (window MEAN − token)")
        ax.plot(Ts, [L["T"][T]["g_order"] for T in Ts], "s--", color="#9467bd",
                lw=2, label="g_order (flatten − MEAN)")
        ax.plot(Ts, [L["T"][T]["shuffle_gap"] for T in Ts], "^:",
                color="#ff7f0e", lw=2, label="flatten − SHUFFLED flatten")
        ax.axhspan(-three, three, color="grey", alpha=0.18, label="±3σ_null")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log", base=2); ax.set_xticks(Ts)
        ax.set_xticklabels(Ts)
        ax.set_xlabel("window size T (tokens)"); ax.set_ylabel("ΔAUC")
        ax.set_title("λ̂_hist, base L12 — the gap is order-free aggregation",
                     fontsize=11)
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"lambda_decomp.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"sigma_null": sn, "3sigma": three,
                      "predictions": verdict["predictions"],
                      "kill_rules": k, "KILLED": verdict["KILLED"]}, indent=2))
    print(f"-> {FIGS}/lambda_tscaling.* , lambda_decomp.* ; "
          f"{RES}/lambda_verdict.json")


if __name__ == "__main__":
    main()
