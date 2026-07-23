"""Multilane (FB-2) — grid stats + figures for the bench record.

Reads ``results/multilane_grid_results.json`` (the run_grid pool dump),
aggregates over seeds, writes ``results/multilane_bench_stats.json`` and the
record figures. The BLIND verdict against the frozen card § 6 predictions is
written by hand in ``bench_record.md`` FROM these numbers — this script only
computes; it never interprets.

    .venv/bin/python -m experiments.explorations.synthetic.multilane.render_figs
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRID = HERE / "results" / "multilane_grid_results.json"
OUT = HERE / "results" / "multilane_bench_stats.json"
FIGS = HERE / "figs"

F = 101
OMEGA = [0, 1, 2, 4, 8, 16, 24, 32, 40, 50]
PANEL = ["batchtopk_sae", "tsae", "stacked_batchtopk",
         "txc_batchtopk_pre", "txc_batchtopk_post", "spectral_txc"]
BAND = ["spectral_txc", "spectral_txc_dcac", "spectral_txc_full"]
METRIC = "multilane_recovery"


def _group(rows):
    g = defaultdict(list)
    for r in rows:
        if not r.get("ok"):
            continue
        g[(r["arch"], r["T"], r["d_sae"], r["k_pos"], r["kind"])].append(r)
    return g


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
    g = _group(rows)
    out: dict = {"n_ok": len(ok), "n_fail": len(fails),
                 "fail_labels": [f"{r['arch']}/T{r['T']}/d{r['d_sae']}/k{r['k_pos']}/s{r['seed']}"
                                 for r in fails]}

    # ── falsifier 1 first: any T=1 cell above 0.1 recovery? ──
    t1 = [r for r in ok if r["T"] == 1 and r["kind"] == "trained"]
    worst_t1 = max((r["metrics"][METRIC] for r in t1), default=float("nan"))
    out["falsifier_T1_worst"] = round(float(worst_t1), 4)

    # ── canonical frontier: recovery vs T per arch at d_sae=F, per k_pos ──
    frontier = {}
    for arch in PANEL:
        for T in (1, 2, 4, 8):
            for k_pos in (1, 2, 4, 8, 16):
                a = _agg(g.get((arch, T, F, k_pos, "trained"), []))
                if a:
                    frontier[f"{arch}|T{T}|k{k_pos}"] = a
    out["frontier_dF"] = frontier

    # untrained access controls (k_pos=1, d=F)
    out["untrained_dF"] = {
        f"{arch}|T{T}": _agg(g.get((arch, T, F, 1, "untrained"), []))
        for arch in PANEL + ["spectral_txc_dcac", "spectral_txc_full"]
        for T in (1, 2, 4, 8)
        if _agg(g.get((arch, T, F, 1, "untrained"), []))}

    # capacity sweep at the per-token matched-ish k_pos=2, T=4/8
    out["capacity"] = {
        f"{arch}|T{T}|d{d}": _agg(g.get((arch, T, d, 2, "trained"), []))
        for arch in PANEL for T in (4, 8) for d in (50, 101, 202)
        if _agg(g.get((arch, T, d, 2, "trained"), []))}

    # ── the band-partition addendum (matched budget k_pos=1) ──
    band = {}
    for arch in BAND:
        for T in (2, 4, 8):
            for d in (50, 101, 202):
                a = _agg(g.get((arch, T, d, 1, "trained"), []))
                if a:
                    band[f"{arch}|T{T}|d{d}"] = a
    out["band_addendum"] = band

    # oracle reference per T (same for all cells of a T)
    out["oracle_by_T"] = {}
    for T in (2, 4, 8):
        rs = [r for r in ok if r["T"] == T and "multilane_oracle" in r["metrics"]]
        if rs:
            out["oracle_by_T"][T] = round(
                float(np.mean([r["metrics"]["multilane_oracle"] for r in rs])), 4)

    # per-lane S(f) of the best trained cell per arch family at T=8, d=F
    sf = {}
    for arch in PANEL + ["spectral_txc_dcac", "spectral_txc_full"]:
        cand = [r for r in ok
                if r["arch"] == arch and r["T"] == 8 and r["d_sae"] == F
                and r["kind"] == "trained"]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["metrics"].get(METRIC, -9))
        recalls = []
        for c in range(len(OMEGA)):
            v = [best["metrics"].get(f"lane{k}_recall_c{c}") for k in range(3)]
            v = [x for x in v if x is not None and not np.isnan(x)]
            recalls.append(round(float(np.mean(v)), 3) if v else None)
        sf[arch] = {"k_pos": best["k_pos"], "seed": best["seed"],
                    "recovery": round(best["metrics"][METRIC], 4),
                    "recall_by_class": recalls}
    out["sf_best_T8_dF"] = sf

    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} (ok={len(ok)}, fail={len(fails)}, "
          f"falsifier_T1_worst={out['falsifier_T1_worst']})", flush=True)

    # ── figures ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.8))
    # (a) recovery vs T at d=F, k_pos=2 (canonical-ish), + untrained dotted
    for arch in PANEL:
        Ts, ms = [], []
        for T in (1, 2, 4, 8):
            a = frontier.get(f"{arch}|T{T}|k2")
            if a:
                Ts.append(T); ms.append(a["mean"])
        if Ts:
            (ln,) = axes[0].plot(Ts, ms, "o-", label=arch)
            uT, um = [], []
            for T in (1, 2, 4, 8):
                u = out["untrained_dF"].get(f"{arch}|T{T}")
                if u:
                    uT.append(T); um.append(u["mean"])
            if uT:
                axes[0].plot(uT, um, ":", color=ln.get_color(), alpha=0.6)
    axes[0].set(xlabel="T", ylabel="multilane_recovery",
                title="recovery vs T (d_sae=F=101, k_pos=2; dotted=untrained)")
    axes[0].legend(fontsize=6)
    # (b) band addendum: 4-band vs 2-band vs 1-band at k_pos=1, d=F
    x = np.arange(3)
    width = 0.25
    for i, T in enumerate((2, 4, 8)):
        means = [band.get(f"{a}|T{T}|d{F}", {}).get("mean", np.nan) for a in BAND]
        axes[1].bar(x + (i - 1) * width, means, width, label=f"T={T}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["4-band", "2-band (dcac)", "1-band (full)"],
                            fontsize=8)
    axes[1].set(ylabel="multilane_recovery",
                title="band partition @ matched budget (k_pos=1, d=F)")
    axes[1].legend(fontsize=7)
    # (c) per-lane S(f) at T=8
    freqs = [y / F for y in OMEGA]
    for arch, blob in sf.items():
        if arch in ("batchtopk_sae", "tsae"):
            continue
        rc = [r if r is not None else np.nan for r in blob["recall_by_class"]]
        axes[2].plot(freqs, rc, "o-", ms=3, label=f"{arch} ({blob['recovery']:+.2f})")
    axes[2].axhline(0.1, color="gray", ls=":", lw=1)
    axes[2].set(xlabel="f = Y/M", ylabel="mean per-lane recall",
                title="per-lane S(f), best T=8 d=F cells")
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "multilane_bench.png", dpi=160)
    print(f"wrote {FIGS / 'multilane_bench.png'}", flush=True)


if __name__ == "__main__":
    main()
