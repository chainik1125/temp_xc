"""Figures for both receipts (CARD deliverables). Committed pre-run.

  figs/dilution_tscaling.{png,pdf} — λ̂ recovery vs T: fixed-budget lines A1
      (d=20) and A2 (d=40) solid, budget-scaled B (d=5T) dashed; untrained
      dotted/hollow; realized l0_per_window annotated at each trained point;
      per-token DPI floor for reference.
  figs/tsae_fair.{png,pdf} — T-SAE λ̂ recovery vs pair distance Δ, per-seed
      points + mean line, aux α=0 point, DPI-floor band, untrained reference.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.render_figs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.explorations.task_hunt.support_synthetic.run_dilution import POINTS

HERE = Path(__file__).resolve().parent
RES, FIGS = HERE / "results", HERE / "figs"
METRIC = "lambda_recovery"
DPI_FLOOR = 0.41

LINE_STYLE = {           # name -> (color, marker, ls, label)
    "A1": ("#ff7f0e", "o", "-", "A1 fixed budget (d_sae=20=F)"),
    "A2": ("#8c2d04", "s", "-", "A2 fixed budget (d_sae=40=2F)"),
    "B": ("#2ca02c", "D", "--", "B budget-scaled (d_sae=5·T)"),
}


def _agg(rows, kind):
    out = {}
    for r in rows:
        if r.get("ok") and r["kind"] == kind:
            out.setdefault((r["T"], r["d_sae"]), []).append(r["metrics"])
    return out


def _dilution():
    rows = json.loads((RES / "dilution_grid_results.json").read_text())
    tr, un = _agg(rows, "trained"), _agg(rows, "untrained")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, pts in POINTS.items():
        c, mk, ls, label = LINE_STYLE[name]
        Ts = [T for T, d in pts]
        mean = [np.mean([m[METRIC] for m in tr[(T, d)]]) for T, d in pts]
        sd = [np.std([m[METRIC] for m in tr[(T, d)]], ddof=1) for T, d in pts]
        ax.errorbar(Ts, mean, yerr=sd, color=c, marker=mk, ls=ls, lw=1.8,
                    ms=5, capsize=2, label=label)
        u_mean = [np.mean([m[METRIC] for m in un[(T, d)]]) if (T, d) in un
                  else np.nan for T, d in pts]
        ax.plot(Ts, u_mean, color=c, marker=mk, ls=":", lw=1.0, ms=4,
                mfc="none", alpha=0.6)
        for (T, d), y in zip(pts, mean):
            l0 = np.mean([m.get("l0_per_window", np.nan) for m in tr[(T, d)]])
            ax.annotate(f"l0w={l0:.1f}", (T, y), textcoords="offset points",
                        xytext=(4, -11), fontsize=6.5, color=c, alpha=0.85)
    ax.axhline(DPI_FLOOR, color="#7f7f7f", lw=0.9, ls="-.")
    ax.text(2.02, DPI_FLOOR + 0.008, "per-token DPI floor ≈ 0.41",
            fontsize=7, color="#7f7f7f")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], labels=["2", "4", "8", "16", "32"])
    ax.set_xlabel("window length T")
    ax.set_ylabel("λ̂ recovery (held-out r; chance ≈ 0)")
    ax.set_title("Budget-dilution receipt — TXC-pre on the λ̂ mirror "
                 "(k_pos=1, L=32; dotted = untrained)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"dilution_tscaling.{ext}", dpi=180)
    plt.close(fig)


def _tsae():
    rows = json.loads((RES / "tsae_fair_grid_results.json").read_text())
    tr = {}
    un_vals = []
    for r in rows:
        if not r.get("ok"):
            continue
        if r["kind"] == "untrained":
            un_vals.append(r["metrics"][METRIC])
        else:
            tr.setdefault(r["arch"], []).append(r["metrics"][METRIC])
    deltas = {"tsae_d1": 1, "tsae_d2": 2, "tsae_d4": 4, "tsae_d8": 8}
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.axhspan(0.38, 0.44, color="#7f7f7f", alpha=0.15,
               label="bench T-SAE band (d_sae×k grid)")
    ax.axhline(DPI_FLOOR, color="#7f7f7f", lw=0.9, ls="-.")
    xs = [deltas[a] for a in deltas]
    means = [np.mean(tr[a]) for a in deltas]
    ax.plot(xs, means, color="#8c564b", marker="v", lw=1.8, ms=6,
            label="T-SAE (pair distance Δ)")
    for a, x in deltas.items():
        ax.scatter([x] * len(tr[a]), tr[a], color="#8c564b", s=12, alpha=0.5)
    if "tsae_a0" in tr:
        ax.scatter([1.15] * len(tr["tsae_a0"]), tr["tsae_a0"], color="#1f77b4",
                   s=14, alpha=0.6)
        ax.scatter([1.15], [np.mean(tr["tsae_a0"])], color="#1f77b4", marker="x",
                   s=45, label="aux: α=0 (no contrastive)")
    if un_vals:
        ax.axhline(float(np.mean(un_vals)), color="#8c564b", lw=1.0, ls=":",
                   alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], labels=["1", "2", "4", "8"])
    ax.set_xlabel("contrastive pair distance Δ  (registered T-SAE: Δ=1)")
    ax.set_ylabel("λ̂ recovery (held-out r)")
    ax.set_title("T-SAE fairness receipt — its own temporal knob "
                 "(d_sae=20, k_pos=1; dotted = untrained)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"tsae_fair.{ext}", dpi=180)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    _dilution()
    _tsae()
    print(f"[render] wrote {FIGS}/dilution_tscaling.* and {FIGS}/tsae_fair.*")


if __name__ == "__main__":
    main()
