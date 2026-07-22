"""Merge the widened FreqFrac pass (FB-C1 Phase 1) into one table + figure.

Globs ``results/freqfrac_stats_<bench>_s<seed>_T<T>.json`` (the parallel
``--tag`` invocations: seeds {1,2,42} at T_can=4 + seed 1 at T=8), merges into
``results/freqfrac_full_pass.json``, renders ``figs/freqfrac_full_pass.png``,
and prints the PORT.md § G "full-pass results" table:

- per (bench, arch): dc_frac and concentration per seed (trained / init) at
  T=4 — the **seed-stability** read on the axis-1 coordinates;
- the T=8 rows for the window archs — where the **frequency high-pass**
  check has resolution (PORT § G caveat: at T=4, 5 of the 10 Ω tones sit
  below the first DCT bin).

    .venv/bin/python -m experiments.explorations.synthetic.freqbench.freqfrac_merge
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

BENCHES = ["backtracking", "signed_motion", "changepoint",
           "assumption_consequence", "hedging_drift", "frequency"]
ARCH_ORDER = ["batchtopk_sae", "tsae", "stacked_batchtopk",
              "txc_batchtopk_pre", "txc_batchtopk_post", "spectral_txc"]


def main() -> None:
    cells = []
    for p in sorted(RES.glob("freqfrac_stats_*_s*_T*.json")):
        m = re.match(r"freqfrac_stats_(.+)_s(\d+)_T(\d+)\.json", p.name)
        blob = json.loads(p.read_text())
        seed = int(m.group(2))
        T_inv = int(m.group(3))
        for c in blob["cells"]:
            c = dict(c)
            c["seed"] = seed
            c["T_invocation"] = T_inv
            cells.append(c)

    # de-dup (token cells appear in both the T4 and T8 invocations of seed 1)
    seen, uniq = set(), []
    for c in cells:
        key = (c["bench"], c["arch"], c["seed"], c["T"], c["train_key"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    merged = {"n_cells": len(uniq), "cells": uniq}
    out_path = RES / "freqfrac_full_pass.json"
    out_path.write_text(json.dumps(merged, indent=1))
    print(f"merged {len(uniq)} unique cells -> {out_path}\n", flush=True)

    # ── the § G table ──
    by = defaultdict(dict)
    for c in uniq:
        by[(c["bench"], c["arch"], c["T"])][c["seed"]] = c

    def fmt_group(bench, arch, T):
        g = by.get((bench, arch, T))
        if not g:
            return None
        dc_t = [g[s]["trained"]["dc_frac"] for s in sorted(g)]
        dc_u = [g[s]["untrained"]["dc_frac"] for s in sorted(g)]
        co_t = [g[s]["trained"]["concentration"] for s in sorted(g)]
        co_u = [g[s]["untrained"]["concentration"] for s in sorted(g)]
        return {
            "seeds": sorted(g),
            "dc_frac_trained": dc_t, "dc_frac_init": dc_u,
            "conc_trained": co_t, "conc_init": co_u,
            "dc_mean": float(np.mean(dc_t)), "dc_spread": float(np.ptp(dc_t)),
            "conc_mean": float(np.mean(co_t)), "conc_spread": float(np.ptp(co_t)),
            "dc_init_mean": float(np.mean(dc_u)),
            "conc_init_mean": float(np.mean(co_u)),
        }

    summary: dict = {}
    print(f"{'bench':<24s}{'arch':<20s}{'T':>3s} "
          f"{'dc_frac (3 seeds)':<26s}{'conc (3 seeds)':<26s}{'init dc/conc'}")
    for bench in BENCHES:
        for arch in ARCH_ORDER:
            for T in (1, 4, 8):
                s = fmt_group(bench, arch, T)
                if s is None:
                    continue
                summary[f"{bench}/{arch}/T{T}"] = s
                dcs = ",".join(f"{v:.3f}" for v in s["dc_frac_trained"])
                cos = ",".join(f"{v:.3f}" for v in s["conc_trained"])
                print(f"{bench:<24s}{arch:<20s}{T:>3d} "
                      f"{dcs:<26s}{cos:<26s}"
                      f"{s['dc_init_mean']:.3f}/{s['conc_init_mean']:.3f}")

    (RES / "freqfrac_full_pass_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"\nwrote {RES / 'freqfrac_full_pass_summary.json'}", flush=True)

    # ── figure: per-bench T=4 curves (seed-mean) + frequency T=8 panel ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(19, 7), squeeze=False)
    panels = [(b, 4) for b in BENCHES] + [("frequency", 8), ("backtracking", 8)]
    for ax, (bench, T) in zip(axes.flat, panels):
        for arch in ARCH_ORDER:
            g = by.get((bench, arch, T))
            if not g or T == 1:
                continue
            curves = np.array([g[s]["trained"]["curve"] for s in sorted(g)])
            inits = np.array([g[s]["untrained"]["curve"] for s in sorted(g)])
            w = np.arange(curves.shape[1])
            (ln,) = ax.plot(w, curves.mean(0), marker="o", ms=3, label=arch)
            if curves.shape[0] > 1:
                ax.fill_between(w, curves.min(0), curves.max(0),
                                color=ln.get_color(), alpha=0.15)
            ax.plot(w, inits.mean(0), ls=":", lw=1, color=ln.get_color(),
                    alpha=0.6)
        ax.set_title(f"{bench} (T={T})", fontsize=9)
        ax.set_xlabel("DCT index w")
        ax.set_ylabel("firing-weighted energy frac")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
    fig.suptitle("FreqFrac full pass — seed-mean curves (band = seed range; "
                 "dotted = untrained init)", fontsize=11)
    fig.tight_layout()
    figs = HERE / "figs"
    figs.mkdir(exist_ok=True)
    fig.savefig(figs / "freqfrac_full_pass.png", dpi=160)
    print(f"wrote {figs / 'freqfrac_full_pass.png'}", flush=True)


if __name__ == "__main__":
    main()
