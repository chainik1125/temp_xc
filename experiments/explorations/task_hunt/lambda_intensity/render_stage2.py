"""Candidate 1 Stage 2 — the T-scaling figure (the hunt's money plot).

Reads results/stage2_<ds>.json (the panel produced by run_stage2.py) and
emits:
  figs/stage2_tscaling.{png,pdf} — lambda_recovery vs T, one line per
      architecture, trained (solid) with the untrained control (hollow)
      and the per-arch empirical chance floor. The hunt's claim would be
      TXC rising while the per-token SAE and T-SAE stay flat.
  results/stage2_summary.json — per (arch, T) mean/std over seeds, the
      untrained control, realized l0_per_token (the fairness check), and
      the trained−untrained margin.

Only rows from the FINAL config count (buffer_tokens 524288, the
n_steps of run_stage2); anything else is a superseded or plumbing cell
and is excluded here, matching RECORD § 4.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.lambda_intensity.render_stage2 [ds]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"

ARCH_STYLE = {
    "batchtopk_sae": ("#7f7f7f", "o-", "per-token BatchTopK SAE"),
    "tsae": ("#8c564b", "v-", "T-SAE"),
    "stacked_batchtopk": ("#2ca02c", "^-", "Stacked"),
    "txc_batchtopk_pre": ("#ff7f0e", "s-", "TXC-pre"),
    "txc_batchtopk_post": ("#1f77b4", "D-", "TXC-post"),
}
METRIC = "lambda_recovery"


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "ward_real_lambda_base_l12"
    rows = json.loads((RES / f"stage2_{ds}.json").read_text())
    ok = [r for r in rows if r.get("ok")]

    trained = defaultdict(list)      # (arch, T) -> [metric]
    untrained = defaultdict(list)
    chance = defaultdict(list)
    l0 = defaultdict(list)
    for r in ok:
        key = (r["arch"], r["T"])
        m = r["metrics"]
        bucket = untrained if r.get("kind") == "untrained" else trained
        bucket[key].append(m.get(METRIC, float("nan")))
        chance[key].append(m.get("lambda_chance", float("nan")))
        l0[key].append(m.get("l0_per_token", float("nan")))

    def agg(d):
        return {f"{a}/T{t}": {"mean": float(np.nanmean(v)),
                              "std": float(np.nanstd(v)), "n": len(v)}
                for (a, t), v in sorted(d.items())}

    summary = {
        "datasource": ds, "metric": METRIC,
        "n_cells_ok": len(ok), "n_cells_total": len(rows),
        "trained": agg(trained), "untrained": agg(untrained),
        "chance": agg(chance), "l0_per_token": agg(l0),
        "margin_trained_minus_untrained": {
            f"{a}/T{t}": float(np.nanmean(v)
                               - np.nanmean(untrained.get((a, t), [np.nan])))
            for (a, t), v in sorted(trained.items())},
    }
    RES.mkdir(exist_ok=True)
    (RES / "stage2_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for arch, (col, mk, lab) in ARCH_STYLE.items():
        Ts = sorted({t for (a, t) in trained if a == arch})
        if not Ts:
            continue
        mu = [np.nanmean(trained[(arch, t)]) for t in Ts]
        sd = [np.nanstd(trained[(arch, t)]) for t in Ts]
        if len(Ts) == 1:            # token archs: draw as a flat reference
            ax.axhline(mu[0], color=col, ls="--", lw=1.6, alpha=0.9,
                       label=f"{lab} (T=1)")
            ax.fill_between([2, 16], mu[0] - sd[0], mu[0] + sd[0],
                            color=col, alpha=0.10)
        else:
            ax.errorbar(Ts, mu, yerr=sd, fmt=mk, color=col, lw=2,
                        capsize=3, label=lab)
            un = [np.nanmean(untrained.get((arch, t), [np.nan])) for t in Ts]
            ax.plot(Ts, un, mk[0], color=col, mfc="none", ls=":", lw=1,
                    alpha=0.7)
    ax.set_xscale("log", base=2)
    Tall = sorted({t for (_, t) in trained if t > 1})
    if Tall:
        ax.set_xticks(Tall); ax.set_xticklabels(Tall)
    ax.set_xlabel("window size T (tokens)")
    ax.set_ylabel(f"{METRIC}  (held-out Pearson r of a linear probe)")
    ax.set_title("Task hunt Stage 2 — λ̂ intensity recovery vs T\n"
                 "REAL Ward activations (base, resid_post L12); "
                 "hollow/dotted = untrained control", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"stage2_tscaling.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"n_ok": len(ok), "n_total": len(rows),
                      "trained": summary["trained"],
                      "untrained": summary["untrained"],
                      "l0_per_token": summary["l0_per_token"]}, indent=2))
    print(f"-> {FIGS}/stage2_tscaling.* ; {RES}/stage2_summary.json")


if __name__ == "__main__":
    main()
