"""Shuffle-overlay figures — λ̂ + ttrend (SHUFFLE_OVERLAY_CARD § 5 /
TT_SHUFFLE_OVERLAY_CARD § 5; directive eeb4ee3c4 (d)).

Template knob-for-knob with the frozen RLHF/probing pair
(`actmix_rlhf/render_writeup_fig.py`): x = T log2, ordered solid +
shuffled dashed, faint per-seed lines, seed-mean ± sd, coverage note,
--pair-style {mono,blueorange}. Task-specific: y = recovery r; the
claiming arm starts at T = 2 (per-token anchors drawn as horizontal
bands, shuffle ≡ identity at T = 1 — annotated); the QUOTED panel
3-seed means are drawn as grey × ticks (the anchor-gate receipt made
visible; the quoted numbers remain the exhibit numbers).

  .venv/bin/python -m experiments.explorations.task_hunt.render_overlay_figs --task lambda
  .venv/bin/python -m experiments.explorations.task_hunt.render_overlay_figs --task ttrend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_DIR = ROOT / "figs_writeup"
TXC = "#D55E00"
SAE_C, TSAE_C = "#555555", "#888888"

TASKS = {
    "lambda": {
        "overlay": HERE / "lambda_intensity/results/shuffle_overlay.json",
        "arm": "txc_batchtopk_post",
        "out": "fig_lambda_shuffle_tsweep",
        "ylabel": "recovery r (backtracking intensity λ̂_hist)",
        "title_arm": "TXC-post (v2 panel arm, retrained w/ anchor gate)",
    },
    "ttrend": {
        "overlay": HERE / "diafaces/results/tt_shuffle_overlay.json",
        "arm": "txc_batchtopk_post",
        "out": "fig_ttrend_shuffle_tsweep",
        "ylabel": "recovery r (turn-length trend)",
        "title_arm": "TXC-post (v2 panel arm, retrained w/ anchor gate)",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=tuple(TASKS), required=True)
    ap.add_argument("--pair-style", choices=("mono", "blueorange"),
                    default="mono")
    args = ap.parse_args()
    cfg = TASKS[args.task]
    shuf_c = TXC if args.pair_style == "mono" else "#0072B2"
    colors = {"ordered": TXC, "shuffled": shuf_c}

    payload = json.loads(cfg["overlay"].read_text())
    assert payload["anchor_gate_all_pass"], (
        "anchor gate not ALL PASS — the two-instrument fallback applies; "
        "this renderer refuses (card § 3)")
    cells = [c for c in payload["cells"] if c["arch"] == cfg["arm"]]
    seeds = sorted({c["seed"] for c in cells})
    Ts = sorted({c["T"] for c in cells})
    at = {(c["T"], c["seed"]): c for c in cells}

    fig, ax = plt.subplots(figsize=(5.4, 3.7))

    for seed in seeds:
        for field, ls in (("recomputed_r", "-"), ("r_shuf", "--")):
            ys = [at[(T, seed)][field] for T in Ts]
            ax.plot(Ts, ys, ls, color=colors["ordered" if field == "recomputed_r"
                                             else "shuffled"],
                    alpha=0.25, lw=1, zorder=1)

    for field, ls, mk, mfc, label in (
            ("recomputed_r", "-", "o", None, "ordered"),
            ("r_shuf", "--", "s", "white", "within-window shuffled")):
        c = colors["ordered" if field == "recomputed_r" else "shuffled"]
        mu = [np.mean([at[(T, s)][field] for s in seeds]) for T in Ts]
        sd = [np.std([at[(T, s)][field] for s in seeds], ddof=1) for T in Ts]
        ax.plot(Ts, mu, ls, color=c, lw=2, marker=mk, ms=6,
                mfc=mfc or c, mec=c, label=label, zorder=3)
        ax.errorbar(Ts, mu, yerr=sd, color=c, capsize=3, lw=1.2,
                    fmt="none", zorder=2)

    # quoted-panel means (anchor-gate receipt, visible)
    gate = payload["anchor_gate"]
    qT = [T for T in Ts if f"{cfg['arm']}/T{T}" in gate]
    ax.plot(qT, [gate[f"{cfg['arm']}/T{T}"]["quoted_mean"] for T in qT],
            linestyle="none", marker="x", ms=7, color="#333333",
            label="quoted panel (3-seed mean)", zorder=4)

    # per-token anchor bands (quoted + retrained agree under the gate)
    for arm, c, label in (("batchtopk_sae", SAE_C, "per-token SAE (T=1)"),
                          ("tsae", TSAE_C, "T-SAE (T=1)")):
        vals = [c2["recomputed_r"] for c2 in payload["cells"]
                if c2["arch"] == arm]
        if vals:
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            ax.axhspan(m - s, m + s, color=c, alpha=0.12, zorder=0)
            ax.axhline(m, color=c, lw=1, ls=":", zorder=0)
            ax.annotate(label, xy=(Ts[-1], m), fontsize=7, color=c,
                        ha="right", va="bottom")

    ax.annotate("anchors at T=1: shuffle ≡ identity (by construction)",
                xy=(0.03, 0.95), xycoords="axes fraction", ha="left",
                va="top", fontsize=8, color="#555555")
    cov = " ".join(
        f"T{T}:n={len([1 for s in seeds if (T, s) in at])}" for T in Ts)
    ax.annotate(f"anchor gate ALL PASS · {cov} · shuffle seed "
                f"{payload['shuf_eval_seed']}", xy=(0.99, 0.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=6.5, color="#777777")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.minorticks_off()
    ax.set_xlabel("T (window length)")
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(0.02, 0.90))
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{cfg['out']}.{ext}", dpi=200)
    print(f"[render] -> {OUT_DIR / cfg['out']}.{{png,pdf}}")


if __name__ == "__main__":
    main()
