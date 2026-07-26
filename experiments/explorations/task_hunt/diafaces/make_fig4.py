"""diafaces/make_fig4.py — fig4_ttrend_post_confirmation for the writeup
(salvage W1 + n=6 top-up; KEEP-gated deliverable, unlocked by the
top-up verdict).

Source: the CANONICAL leaderboard only (results/leaderboard.jsonl),
budget-convention-enforced selection (primary arm k_pos = 8; fresh
seeds only; freeze stamps 50af78f12 ∪ 85c87fd76). Conventions match
Han's figs 1–2: Okabe-Ito, paired t 95% CI whiskers, visible-cue
evidence line drawn, claiming zone shaded. n per point annotated
(n = 3 at T ≤ 8, n = 6 at claiming Ts).

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.make_fig4
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DS = "dial_real_ttrend_gpt2_l7"
FREEZES = {"50af78f121d4c4cbe5024c93aeaa5a4753daed11",
           "85c87fd7602fb36dd2e63488b8d33ad3311789e5"}
SEEDS = {3, 4, 5, 6, 7, 8}
TCRIT = {3: 4.302652729911275, 6: 2.5705818366147395}
# Okabe-Ito
C_POST, C_SAE, C_TSAE, C_UNTR = "#D55E00", "#0072B2", "#009E73", "#7f7f7f"
OUT = ROOT / "experiments/explorations/task_hunt/figs_writeup"


def _rows():
    sel = {}
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if (r.get("datasource") != DS or r.get("seed") not in SEEDS
                    or r["code_version"]["commit_sha"] not in FREEZES):
                continue
            hp = r["training_cfg"]["arch_hparams_override"]
            if r["arch"] == "txc_batchtopk_post" and hp["k_pos"] != 8:
                continue                       # secondary arm excluded
            kind = "trained" if r["training_cfg"]["n_steps"] else "untrained"
            sel[(r["arch"], kind, hp["T"], r["seed"])] = \
                r["metrics"]["lambda_recovery"]
    return sel


def _stat(vals):
    v = np.array(vals, float)
    n = len(v)
    m = float(v.mean())
    half = (TCRIT[n] * float(v.std(ddof=1)) / math.sqrt(n)) if n > 1 else 0.0
    return m, half, n


def main():
    sel = _rows()
    ev = json.loads((HERE / "results" / "panel_evidence_line_tt.json")
                    .read_text())["per_T"]
    Ts = [2, 4, 8, 16, 32]
    x = np.log2(Ts)

    def series(arch, kind, T):
        vals = [v for (a, k, t, s), v in sel.items()
                if a == arch and k == kind and t == T]
        assert vals, (arch, kind, T)
        return _stat(vals)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axvspan(np.log2(16) - 0.18, np.log2(32) + 0.35, color="#F0E442",
               alpha=0.25, lw=0, zorder=0)
    ax.text(np.log2(23), 0.315, "claiming zone\n(KEEP, n = 6)",
            ha="center", fontsize=8, color="#7a6a00")

    for kind, color, lbl, ls in (
            ("trained", C_POST, "TXC-post trained (8 act/window)", "-"),
            ("untrained", C_UNTR, "TXC-post untrained", "--")):
        m, h, ns = zip(*(series("txc_batchtopk_post", kind, T) for T in Ts))
        ax.errorbar(x, m, yerr=h, color=color, ls=ls, marker="o", ms=4,
                    capsize=3, lw=1.6, label=lbl, zorder=4)
        if kind == "trained":
            for xi, mi, ni in zip(x, m, ns):
                ax.annotate(f"n={ni}", (xi, mi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7,
                            color=color)

    for arch, color, lbl in ((f"batchtopk_sae", C_SAE,
                              "BatchTopK SAE (8 act/token, T=1)"),
                             ("tsae", C_TSAE, "T-SAE (8 act/token, T=1)")):
        m, h, _ = series(arch, "trained", 1)
        ax.axhline(m, color=color, lw=1.4, label=lbl)
        ax.axhspan(m - h, m + h, color=color, alpha=0.14, lw=0)

    ax.plot(x, [ev[str(T)]["pearson_r"] for T in Ts], color="k", ls=":",
            marker="x", ms=5, lw=1.2,
            label="visible-cue evidence line (label-side |r|)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.set_xlabel("window length T (tokens)")
    ax.set_ylabel("λ recovery (v1, canonical)")
    ax.set_title("ttrend (gpt2/hs7): TXC-post fresh-seed confirmation "
                 "(seeds {3–8})", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_ylim(-0.06, 0.36)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"fig4_ttrend_post_confirmation.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None)
        print(f"[fig4] wrote {p}")


if __name__ == "__main__":
    main()
