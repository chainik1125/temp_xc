"""Candidate 3 — render the onset-anticipation screen + score the kill rule.

Reads results/forbidden_word_screen.json and emits:
  results/forbidden_word_verdict.json — sigma_null, per-horizon
      per-token vs best-window, and the card's kill rules scored;
  figs/forbidden_word_tscaling.{png,pdf} — per horizon D, per-token
      (flat reference) vs window ceiling vs T, with the ±3σ_null band.

Run: .venv/bin/python -m experiments.explorations.task_hunt.forbidden_word.render
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"
TS = [2, 4, 8, 16, 32]
HORIZONS = [4, 8, 16]


def main():
    FIGS.mkdir(exist_ok=True)
    c = json.loads((RES / "forbidden_word_screen.json").read_text())["cells"]
    nulls = [abs(v[nk]["auc"] - 0.5)
             for v in c.values() for nk in ("null", "null_flat") if nk in v]
    sn = float(np.std(nulls))
    three = 3 * sn

    per_h = {}
    for D in HORIZONS:
        tok = c[f"D{D}/tok"]["linear"]["auc"]
        ladder = {}
        for T in TS:
            cell = c[f"D{D}/T{T}"]
            ladder[T] = {"flat": cell["flat"]["auc"], "mean": cell["mean"]["auc"],
                         "shuf": cell["shuf"]["auc"], "g": cell["g"],
                         "g_order": cell["g_order"],
                         "ceil": max(cell["flat"]["auc"], cell["mean"]["auc"])}
        best = max(l["ceil"] for l in ladder.values())
        per_h[D] = {"tok": tok, "best_window": best,
                    "within_0.02": abs(tok - best) <= 0.02,
                    "max_g": max(l["g"] for l in ladder.values()),
                    "max_g_order": max(l["g_order"] for l in ladder.values()),
                    "ladder": ladder}

    kill_p4 = all(per_h[D]["within_0.02"] for D in HORIZONS)
    kill_g = max(per_h[D]["max_g"] for D in HORIZONS) <= three
    verdict = {"sigma_null": sn, "3sigma_null": three, "n_null": len(nulls),
               "per_horizon": {str(D): {k: per_h[D][k] for k in
                                        ("tok", "best_window", "within_0.02",
                                         "max_g", "max_g_order")}
                               for D in HORIZONS},
               "KILL_rule_1_ambient_within_0.02_all_horizons": bool(kill_p4),
               "KILL_rule_2_window_never_beats_pertoken_3sigma": bool(kill_g),
               "KILLED": bool(kill_p4 or kill_g)}
    (RES / "forbidden_word_verdict.json").write_text(json.dumps(verdict, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, D in zip(axes, HORIZONS):
        L = per_h[D]["ladder"]
        ax.axhline(per_h[D]["tok"], color="#7f7f7f", ls="--", lw=2,
                   label="per-token")
        ax.fill_between([2, 32], per_h[D]["tok"] - three, per_h[D]["tok"] + three,
                        color="#7f7f7f", alpha=0.15, label="per-token ±3σ_null")
        ax.plot(TS, [L[T]["ceil"] for T in TS], "D-", color="#1f77b4", lw=2,
                label="window ceiling")
        ax.plot(TS, [L[T]["shuf"] for T in TS], "x:", color="#ff7f0e", lw=1.5,
                label="window shuffled")
        ax.set_xscale("log", base=2); ax.set_xticks(TS); ax.set_xticklabels(TS)
        ax.set_xlabel("window size T"); ax.set_title(f"horizon D = {D}",
                                                     fontsize=11)
        ax.grid(True, alpha=0.25)
        if D == HORIZONS[0]:
            ax.set_ylabel("AUC"); ax.legend(fontsize=8)
    fig.suptitle("Candidate 3 — forbidden-word onset anticipation (R1-Distill "
                 "L12): window never beats per-token ⇒ AMBIENT ⇒ KILL",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"forbidden_word_tscaling.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(verdict, indent=2))
    print(f"-> {RES}/forbidden_word_verdict.json ; {FIGS}/forbidden_word_tscaling.*")


if __name__ == "__main__":
    main()
