"""Candidate 1 Stage 2 — the T-scaling figure (the hunt's money plot).

Variance-aware renderer (upgraded by runpod-b per
`briefings/hunt-support-stats.md` item 2; provenance RECORD § 3b):
  (a) every arch line carries its realized-l0 range in the legend; an
      arch whose realized l0 falls below half the nominal k in any
      trained cell is flagged NOT budget-matched, with its minimum
      annotated on the plot (review note 3 — mandatory before any
      external use; TXC-post collapses to 0.49 at T = 16 in round 1);
  (b) whiskers are two-sided 95% t confidence intervals over seeds
      (support_stats.stats_lib.t_ci95), not ±std;
  (c) a budget-matched-only variant figure omits non-matched lines.

Reads results/stage2_<ds>.json (the panel produced by run_stage2.py),
PLUS — if present — results/stage2_postmatched_<ds>.json (the
budget-matched TXC-post cells from run_stage2_postmatched.py, frozen in
`card_stage2_postmatched.md`; runpod-d graft). The matched cells enter
as a SEPARATE arch `txc_batchtopk_post_matched` so they can never
silently merge with the round-1 nominal-k=8 post cells; b's CI /
l0-range / budget_matched machinery then applies to them uniformly (they
realize l0 ≈ 8, so they read as budget-matched and appear in BOTH
figures, while round-1 post stays flagged NOT matched). Emits:
  figs/stage2_tscaling.{png,pdf}          — full panel, annotated
  figs/stage2_tscaling_matched.{png,pdf}  — budget-matched archs only
  results/stage2_summary.json — per (arch, T) mean/std/CI over seeds,
      untrained control, realized l0 (fairness check), per-arch l0 range
      + budget_matched flag, trained−untrained margin.

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

from experiments.explorations.task_hunt.support_stats.stats_lib import t_ci95

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"

ARCH_STYLE = {
    "batchtopk_sae": ("#7f7f7f", "o-", "per-token BatchTopK SAE"),
    "tsae": ("#8c564b", "v-", "T-SAE"),
    "stacked_batchtopk": ("#2ca02c", "^-", "Stacked"),
    "txc_batchtopk_pre": ("#ff7f0e", "s-", "TXC-pre"),
    "txc_batchtopk_post": ("#1f77b4", "D--", "TXC-post (k=8/window, NOT matched)"),
    "txc_batchtopk_post_matched": ("#17becf", "D-", "TXC-post (matched, k=8·T)"),
}
METRIC = "lambda_recovery"


def build_summary(rows):
    ok = [r for r in rows if r.get("ok")]
    trained = defaultdict(list)      # (arch, T) -> [metric]
    untrained = defaultdict(list)
    chance = defaultdict(list)
    l0 = defaultdict(list)
    # The budget-matched reference is the round-1 panel's per-TOKEN nominal
    # (k_pos = 8). The grafted matched-post cells carry a per-WINDOW nominal
    # k = 8·T (up to 128); excluding them here keeps b's `>= k_pos/2` flag
    # anchored on the panel's per-token budget instead of being inflated to
    # k_pos/2 = 64 (which would falsely flag every arch as NOT matched).
    k_pos = max((r.get("k_pos", 8) for r in ok
                 if not str(r.get("arch", "")).endswith("_matched")), default=8)
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

    ci = {}
    for (a, t), v in sorted(trained.items()):
        mean, lo, hi = t_ci95(v)
        ci[f"{a}/T{t}"] = {"mean": mean, "lo": lo, "hi": hi, "n": len(v)}

    # realized-l0 range per arch over its cell means (same trained+
    # untrained aggregation as the committed l0_per_token field);
    # budget-matched iff every cell's mean realized l0 >= nominal k / 2
    # (TXC-post's post-squash k_win // T correction collapses it far
    # below this).
    l0_range, matched = {}, {}
    for a in {a for (a, _) in trained}:
        cell_means = [float(np.nanmean(l0[(a, t)]))
                      for (aa, t) in sorted(l0) if aa == a]
        if cell_means:
            l0_range[a] = {"min": min(cell_means), "max": max(cell_means)}
            matched[a] = min(cell_means) >= k_pos / 2

    summary = {
        "datasource": None, "metric": METRIC,
        "n_cells_ok": len(ok), "n_cells_total": len(rows),
        "trained": agg(trained), "untrained": agg(untrained),
        "chance": agg(chance), "l0_per_token": agg(l0),
        "ci95_trained": ci,
        "l0_range": l0_range,
        "budget_matched": matched,
        "match_rule": f"min cell-mean realized l0 >= k_pos/2 = {k_pos / 2}",
        "margin_trained_minus_untrained": {
            f"{a}/T{t}": float(np.nanmean(v)
                               - np.nanmean(untrained.get((a, t), [np.nan])))
            for (a, t), v in sorted(trained.items())},
    }
    return summary, trained, untrained, l0, matched, l0_range


def draw(ax, trained, untrained, l0, matched, l0_range, matched_only):
    def l0_tag(a):
        r = l0_range.get(a)
        if r is None:
            return ""
        tag = (f"l0 {r['min']:.2g}" if r["min"] == r["max"]
               else f"l0 {r['min']:.2g}–{r['max']:.2g}")
        if not matched.get(a, True):
            tag += " — NOT budget-matched"
        return f" ({tag})"

    for arch, (col, mk, lab) in ARCH_STYLE.items():
        if matched_only and not matched.get(arch, False):
            continue
        Ts = sorted({t for (a, t) in trained if a == arch})
        if not Ts:
            continue
        mu, lo, hi = zip(*(t_ci95(trained[(arch, t)]) for t in Ts))
        if len(Ts) == 1:            # token archs: draw as a flat reference
            ax.axhline(mu[0], color=col, ls="--", lw=1.6, alpha=0.9,
                       label=f"{lab} (T=1){l0_tag(arch)}")
            if np.isfinite(lo[0]):
                ax.fill_between([2, 16], lo[0], hi[0], color=col, alpha=0.10)
        else:
            yerr = [np.array(mu) - np.array(lo), np.array(hi) - np.array(mu)]
            ax.errorbar(Ts, mu, yerr=yerr, fmt=mk, color=col, lw=2,
                        capsize=3, label=f"{lab}{l0_tag(arch)}")
            un = [np.nanmean(untrained.get((arch, t), [np.nan])) for t in Ts]
            ax.plot(Ts, un, mk[0], color=col, mfc="none", ls=":", lw=1,
                    alpha=0.7)
            if not matched.get(arch, True):
                t_min = min(Ts, key=lambda t: np.nanmean(l0[(arch, t)]))
                v_min = float(np.nanmean(l0[(arch, t_min)]))
                y_min = float(np.nanmean(trained[(arch, t_min)]))
                ax.annotate(f"realized l0 = {v_min:.2f}\n(nominal k = 8)",
                            xy=(t_min, y_min),
                            xytext=(0.62 * t_min, y_min + 0.03),
                            fontsize=8, color=col, ha="right",
                            arrowprops=dict(arrowstyle="->", color=col,
                                            lw=1))
    ax.set_xscale("log", base=2)
    Tall = sorted({t for (_, t) in trained if t > 1})
    if Tall:
        ax.set_xticks(Tall); ax.set_xticklabels(Tall)
    ax.set_xlabel("window size T (tokens)")
    ax.set_ylabel(f"{METRIC}  (held-out Pearson r of a linear probe)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "ward_real_lambda_base_l12"
    rows = json.loads((RES / f"stage2_{ds}.json").read_text())
    # Graft the budget-matched TXC-post cells (separate file) as a distinct
    # arch, so b's CI / l0-range / budget_matched machinery applies to them
    # uniformly (card_stage2_postmatched.md § 3/§ 5).
    n_matched = 0
    matched_path = RES / f"stage2_postmatched_{ds}.json"
    if matched_path.exists():
        mrows = [{**r, "arch": f"{r['arch']}_matched"}
                 for r in json.loads(matched_path.read_text())]
        n_matched = sum(1 for r in mrows if r.get("ok"))
        rows = rows + mrows
    summary, trained, untrained, l0, matched, l0_range = build_summary(rows)
    summary["n_cells_postmatched"] = n_matched
    summary["datasource"] = ds
    RES.mkdir(exist_ok=True)
    (RES / "stage2_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    variants = [("stage2_tscaling", False,
                 "Task hunt Stage 2 — λ̂ intensity recovery vs T\n"
                 "REAL Ward activations (base, resid_post L12); "
                 "whiskers = 95% t CI over seeds; hollow/dotted = "
                 "untrained control"),
                ("stage2_tscaling_matched", True,
                 "Task hunt Stage 2 — λ̂ recovery vs T, budget-matched "
                 "archs only\n(realized l0 within 2× of nominal k; "
                 "whiskers = 95% t CI over seeds)")]
    for stem, matched_only, title in variants:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        draw(ax, trained, untrained, l0, matched, l0_range, matched_only)
        ax.set_title(title, fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{stem}.{ext}",
                        dpi=140 if ext == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({"n_ok": summary["n_cells_ok"],
                      "n_total": summary["n_cells_total"],
                      "budget_matched": summary["budget_matched"],
                      "l0_range": summary["l0_range"],
                      "ci95_trained": summary["ci95_trained"]}, indent=2))
    print(f"-> {FIGS}/stage2_tscaling.* ; stage2_tscaling_matched.* ; "
          f"{RES}/stage2_summary.json")


if __name__ == "__main__":
    main()
