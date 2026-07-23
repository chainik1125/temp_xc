"""Phasepair (FB-1) — grid stats + figures for the bench record.

Reads ``results/phasepair_grid_results.json``, aggregates over seeds, writes
``results/phasepair_bench_stats.json`` + figures. The BLIND verdict against
the frozen card § 6 is written by hand in ``bench_record.md`` from these
numbers — this script computes, never interprets.

    .venv/bin/python -m experiments.explorations.synthetic.phasepair.render_figs
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRID = HERE / "results" / "phasepair_grid_results.json"
OUT = HERE / "results" / "phasepair_bench_stats.json"
FIGS = HERE / "figs"

F = 101
PANEL = ["batchtopk_sae", "tsae", "stacked_batchtopk",
         "txc_batchtopk_pre", "txc_batchtopk_post", "spectral_txc"]


def _agg(rs, key):
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

    # falsifier 1: any T=1 cell with sign_recovery > 0.1?
    t1 = [r["metrics"].get("sign_recovery") for r in ok
          if r["T"] == 1 and r["kind"] == "trained"
          and "sign_recovery" in r["metrics"]]
    out["falsifier_T1_sign_max"] = round(float(np.max(t1)), 4) if t1 else None

    # frontier: sign + pair per arch × T × k at d = F
    frontier = {}
    for arch in PANEL:
        for T in (1, 2, 4, 8):
            for k in (1, 2, 4, 8, 16):
                cell = g.get((arch, T, F, k, "trained"), [])
                s = _agg(cell, "sign_recovery")
                p = _agg(cell, "pair_recovery")
                v = _agg(cell, "velocity_recovery")
                if s or p:
                    frontier[f"{arch}|T{T}|k{k}"] = {"sign": s, "pair": p,
                                                     "vel6": v}
    out["frontier_dF"] = frontier

    out["untrained_dF"] = {}
    for arch in PANEL:
        for T in (1, 2, 4, 8):
            cell = g.get((arch, T, F, 1, "untrained"), [])
            s = _agg(cell, "sign_recovery")
            p = _agg(cell, "pair_recovery")
            if s or p:
                out["untrained_dF"][f"{arch}|T{T}"] = {"sign": s, "pair": p}

    # capacity slice at k=2
    out["capacity_k2"] = {}
    for arch in PANEL:
        for T in (4, 8):
            for d in (50, 101, 202):
                cell = g.get((arch, T, d, 2, "trained"), [])
                s = _agg(cell, "sign_recovery")
                if s:
                    out["capacity_k2"][f"{arch}|T{T}|d{d}"] = {
                        "sign": s, "pair": _agg(cell, "pair_recovery")}

    # oracle refs
    out["sign_oracle_by_T"] = {}
    for T in (2, 4, 8):
        rs = [r for r in ok if r["T"] == T and "sign_oracle" in r["metrics"]]
        if rs:
            out["sign_oracle_by_T"][T] = round(
                float(np.mean([r["metrics"]["sign_oracle"] for r in rs])), 4)

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} (ok={len(ok)} fail={len(fails)}) "
          f"falsifier_T1_sign_max={out['falsifier_T1_sign_max']}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in ((axes[0], "sign", "SIGN (phase-only) recovery"),
                           (axes[1], "pair", "PAIR (power) recovery")):
        for arch in PANEL:
            Ts, ms = [], []
            for T in (1, 2, 4, 8):
                a = frontier.get(f"{arch}|T{T}|k2", {}).get(key)
                if a:
                    Ts.append(T); ms.append(a["mean"])
            if Ts:
                (ln,) = ax.plot(Ts, ms, "o-", label=arch)
                uT, um = [], []
                for T in (1, 2, 4, 8):
                    u = out["untrained_dF"].get(f"{arch}|T{T}", {}).get(key)
                    if u:
                        uT.append(T); um.append(u["mean"])
                if uT:
                    ax.plot(uT, um, ":", color=ln.get_color(), alpha=0.6)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set(xlabel="T", ylabel=f"{key}_recovery",
               title=f"{title} (d_sae=F, k_pos=2; dotted=untrained)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "phasepair_bench.png", dpi=160)
    print(f"wrote {FIGS / 'phasepair_bench.png'}", flush=True)


if __name__ == "__main__":
    main()
