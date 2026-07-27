"""ttrend two-instrument fallback figure (TT_SHUFFLE_OVERLAY_CARD § 3
fallback branch, invoked by the ~17:25 gate-FAIL entry; pre-approved
in eeb4ee3c4 / 1d2e3de28).

Two panels, instruments labeled as DIFFERENT: (left) the QUOTED v2
trained panel — TXC-post recovery r vs T, 3-seed mean ± sd, per-token
anchors as bands (numbers unchanged from the exhibit); (right) the
committed tt SCREEN's ordered vs within-window-shuffled linear probe
(3-class accuracy) — the order evidence, screen instrument. No number
from the gate-failed retrain appears anywhere.

  .venv/bin/python -m experiments.explorations.task_hunt.diafaces.render_tt_fallback
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT / "figs_writeup"
TXC = "#D55E00"
SAE_C, TSAE_C = "#555555", "#888888"
SEEDS = (1, 2, 42)


def main():
    rows = json.loads((HERE / "results/stage2_dial_real_ttrend_gpt2_l7.json").read_text())
    tr = [r for r in rows if r["kind"] == "trained" and r["ok"]]
    screen = json.loads((HERE / "results/screen_gpt2.json").read_text())["cells"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # left: quoted trained panel (recovery r)
    Ts = sorted({r["T"] for r in tr if r["arch"] == "txc_batchtopk_post"})
    for seed in SEEDS:
        ys = [next(r["metrics"]["lambda_recovery"] for r in tr
                   if r["arch"] == "txc_batchtopk_post" and r["T"] == T
                   and r["seed"] == seed) for T in Ts]
        ax1.plot(Ts, ys, "-", color=TXC, alpha=0.25, lw=1, zorder=1)
    mu = [np.mean([r["metrics"]["lambda_recovery"] for r in tr
                   if r["arch"] == "txc_batchtopk_post" and r["T"] == T])
          for T in Ts]
    sd = [np.std([r["metrics"]["lambda_recovery"] for r in tr
                  if r["arch"] == "txc_batchtopk_post" and r["T"] == T],
                 ddof=1) for T in Ts]
    ax1.plot(Ts, mu, "-", color=TXC, lw=2, marker="o", ms=6,
             label="TXC-post (trained, quoted panel)", zorder=3)
    ax1.errorbar(Ts, mu, yerr=sd, color=TXC, capsize=3, lw=1.2,
                 fmt="none", zorder=2)
    for arm, c, label in (("batchtopk_sae", SAE_C, "per-token SAE (T=1)"),
                          ("tsae", TSAE_C, "T-SAE (T=1)")):
        vals = [r["metrics"]["lambda_recovery"] for r in tr if r["arch"] == arm]
        m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
        ax1.axhspan(m - s, m + s, color=c, alpha=0.12, zorder=0)
        ax1.axhline(m, color=c, lw=1, ls=":", zorder=0)
        ax1.annotate(label, xy=(Ts[-1], m), fontsize=7, color=c,
                     ha="right", va="bottom")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(Ts)
    ax1.set_xticklabels(map(str, Ts))
    ax1.minorticks_off()
    ax1.set_xlabel("T (window length)")
    ax1.set_ylabel("recovery r (turn-length trend)")
    ax1.set_title("trained panel (quoted; no shuffle twin exists)",
                  fontsize=9)
    ax1.grid(True, alpha=0.25, lw=0.5)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")

    # right: screen instrument, ordered vs shuffled
    sTs = sorted({int(k.split("/")[1][1:]) for k in screen
                  if k.startswith("tt/T") and k.endswith("/win_shuf_linear")})
    w = [screen[f"tt/T{T}/win_linear"]["acc_test"] for T in sTs]
    ws = [screen[f"tt/T{T}/win_shuf_linear"]["acc_test"] for T in sTs]
    ax2.plot(sTs, w, "-", color=TXC, lw=2, marker="o", ms=6,
             label="ordered window (screen probe)")
    ax2.plot(sTs, ws, "--", color=TXC, lw=2, marker="s", ms=6, mfc="white",
             mec=TXC, label="within-window shuffled")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sTs)
    ax2.set_xticklabels(map(str, sTs))
    ax2.minorticks_off()
    ax2.set_xlabel("T (window length)")
    ax2.set_ylabel("screen accuracy (3-class)")
    ax2.set_title("screen instrument (committed; carries the shuffle)",
                  fontsize=9)
    ax2.grid(True, alpha=0.25, lw=0.5)
    ax2.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.suptitle("turn-length trend — two instruments (trained panel has no "
                 "shuffle twin; anchor-gated retrain failed its gate — LOG "
                 "~17:25 entry)", fontsize=8.5, y=1.00)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_ttrend_shuffle_tsweep.{ext}", dpi=200,
                    bbox_inches="tight")
    print(f"[render] -> {OUT / 'fig_ttrend_shuffle_tsweep'}.{{png,pdf}}")


if __name__ == "__main__":
    main()
