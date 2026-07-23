"""Colored sources (FB-3) — grid stats + figures for the bench record.

Reads ``results/colored_grid_results.json``, aggregates over seeds, writes
``results/colored_bench_stats.json`` + figures. The BLIND verdict against the
frozen card § 6 predictions is written by hand in ``bench_record.md`` from
these numbers — this script computes, never interprets.

    .venv/bin/python -m experiments.explorations.synthetic.colored_sources.render_figs
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRID = HERE / "results" / "colored_grid_results.json"
OUT = HERE / "results" / "colored_bench_stats.json"
FIGS = HERE / "figs"

F = 32
LAG_D = 2
PANEL = ["batchtopk_sae", "tsae", "stacked_batchtopk",
         "txc_batchtopk_pre", "txc_batchtopk_post", "spectral_txc"]
METRIC = "colored_rec_adj"
FLOOR_EPS = 0.05          # the gating floor bar (falsifier band)


def _agg(rs, key=METRIC):
    vals = [r["metrics"][key] for r in rs if key in r["metrics"]]
    if not vals:
        return None
    return {"mean": round(float(np.mean(vals)), 4),
            "vals": [round(float(v), 4) for v in sorted(vals)],
            "n": len(vals)}


def main() -> None:
    rows = json.loads(GRID.read_text())
    ok = [r for r in rows if r.get("ok")]
    fails = [r for r in rows if not r.get("ok")]
    g = defaultdict(list)
    for r in ok:
        g[(r["arch"], r["T"], r["d_sae"], r["k_pos"], r["kind"])].append(r)

    out: dict = {"n_ok": len(ok), "n_fail": len(fails),
                 "fail_labels": [f"{r['arch']}/T{r['T']}/d{r['d_sae']}/k{r['k_pos']}/s{r['seed']}"
                                 for r in fails]}

    # ── falsifier 1 FIRST: any trained cell at T ≤ D above the floor band? ──
    # (token archs T=1 and window archs T=2 are provably floored — CS-1; the
    # trained-token empirical bound adds the measured stream leakage ~0.02.)
    viol = []
    for r in ok:
        if r["kind"] == "trained" and r["T"] <= LAG_D:
            v = r["metrics"].get(METRIC)
            if v is not None and v > FLOOR_EPS + 0.02:
                viol.append((f"{r['arch']}/T{r['T']}/d{r['d_sae']}"
                             f"/k{r['k_pos']}/s{r['seed']}", round(v, 4)))
    t_le_D = [r["metrics"][METRIC] for r in ok
              if r["kind"] == "trained" and r["T"] <= LAG_D
              and METRIC in r["metrics"]]
    out["falsifier_T_le_D"] = {
        "n_cells": len(t_le_D),
        "max": round(float(np.max(t_le_D)), 4) if t_le_D else None,
        "mean": round(float(np.mean(t_le_D)), 4) if t_le_D else None,
        "violations_above_band": viol,
    }

    # stacked at ALL T (provably floored per the card)
    stacked = [r["metrics"][METRIC] for r in ok
               if r["arch"] == "stacked_batchtopk" and r["kind"] == "trained"
               and METRIC in r["metrics"]]
    out["stacked_all_T"] = {"max": round(float(np.max(stacked)), 4) if stacked else None,
                            "mean": round(float(np.mean(stacked)), 4) if stacked else None}

    # ── the T-profile per arch (the W = D+1 transition read) ──
    profile = {}
    for arch in PANEL:
        for T in (1, 2, 4, 8):
            for d in (16, 32, 64):
                for k in (1, 2, 4, 8, 16):
                    a = _agg(g.get((arch, T, d, k, "trained"), []))
                    if a:
                        profile[f"{arch}|T{T}|d{d}|k{k}"] = a
    out["profile"] = profile

    out["untrained"] = {
        f"{arch}|T{T}": _agg(g.get((arch, T, F, 1, "untrained"), []))
        for arch in PANEL for T in (1, 2, 4, 8)
        if _agg(g.get((arch, T, F, 1, "untrained"), []))}

    # ρ-quartile curves of the best trained T ∈ {4,8} cell per arch
    quart = {}
    for arch in PANEL:
        cand = [r for r in ok if r["arch"] == arch and r["T"] > LAG_D
                and r["kind"] == "trained"]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["metrics"].get(METRIC, -9))
        quart[arch] = {
            "cell": f"T{best['T']}/d{best['d_sae']}/k{best['k_pos']}/s{best['seed']}",
            "rec_adj": round(best["metrics"][METRIC], 4),
            "q1_q4": [round(best["metrics"].get(f"colored_rec_q{q}", np.nan), 4)
                      for q in (1, 2, 3, 4)]}
    out["rho_quartiles_best"] = quart

    # companions at the best cells (capability gate)
    comp = {}
    for arch in PANEL:
        for T in (4, 8):
            a_n = _agg(g.get((arch, T, F, 2, "trained"), []), key="nmse")
            a_e = _agg(g.get((arch, T if arch not in ("batchtopk_sae", "tsae") else 1,
                              F, 2, "trained"), []), key="eauc")
            if a_n or a_e:
                comp[f"{arch}|T{T}"] = {"nmse": a_n, "eauc": a_e}
    out["companions"] = comp

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} (ok={len(ok)} fail={len(fails)}) "
          f"falsifier max@T<=D = {out['falsifier_T_le_D']['max']} "
          f"({len(viol)} violations)", flush=True)

    # ── figures ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for arch in PANEL:
        Ts, ms = [], []
        for T in (1, 2, 4, 8):
            a = profile.get(f"{arch}|T{T}|d{F}|k2")
            if a:
                Ts.append(T); ms.append(a["mean"])
        if Ts:
            (ln,) = axes[0].plot(Ts, ms, "o-", label=arch)
            uT, um = [], []
            for T in (1, 2, 4, 8):
                u = out["untrained"].get(f"{arch}|T{T}")
                if u:
                    uT.append(T); um.append(u["mean"])
            if uT:
                axes[0].plot(uT, um, ":", color=ln.get_color(), alpha=0.6)
    axes[0].axvline(LAG_D + 0.5, color="tab:red", ls=":", lw=1,
                    label=f"W = D+1 = {LAG_D+1}")
    axes[0].axhspan(-FLOOR_EPS, FLOOR_EPS, color="gray", alpha=0.15)
    axes[0].set(xlabel="T", ylabel="colored_rec_adj",
                title="F-recovery vs T (d_sae=F=32, k_pos=2; dotted=untrained)")
    axes[0].legend(fontsize=6)

    for arch, blob in quart.items():
        axes[1].plot([1, 2, 3, 4], blob["q1_q4"], "o-",
                     label=f"{arch} ({blob['rec_adj']:+.2f})")
    axes[1].set(xlabel="ρ quartile (1=low, 4=high)", ylabel="mean max cos²",
                title="per-source recovery by ρ quartile (best T>D cells)")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "colored_bench.png", dpi=160)
    print(f"wrote {FIGS / 'colored_bench.png'}", flush=True)


if __name__ == "__main__":
    main()
