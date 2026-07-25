"""oprate Stage 2 — the T-scaling figure + summary (CARD_STAGE2.md § 5).

Adapted from `lambda_intensity/render_stage2.py` (runpod-b's
variance-aware machinery: 95% t CIs over seeds, realized-l0 legend
ranges, budget-matched flags) with the oprate panel's own contract:

- **Post is natively matched** (nominal k = 8·T from cell one), so the
  budget-match reference is the PER-TOKEN code rate 8 for every arch
  (min cell-mean realized `l0_per_token` >= 4), not the max nominal
  k_pos — anchoring on nominal k_pos would read post's k = 128 as the
  panel budget and falsely flag every token arch.
- **Paired figures**: `stage2_tscaling.*` (v1, the canonical claim
  column) and `stage2_tscaling_v2.*` (paired v2, reported beside v1,
  NEVER quoted as canonical — the 2026-07-25 methods decision).
- **The evidence-line regression analog** (`evidence_line.py`, card
  § 3) is drawn per T on both figures: a window cell below that line is
  counting visible event sentences and earns no latent-state language.
- **Binding-2 bookkeeping** emitted machine-readable in the summary:
  every trained cell whose realized l0 falls outside the card band
  [5.0, 8.25] is listed as a residual mismatch; untrained post cells
  are checked against exact 8.00 ± 0.02 (§ 2.2 — VOID rule).

Reads `results/stage2_<ds>.json`; if absent, merges the per-pool files
`stage2_<ds>__only-tsae.json` + `stage2_<ds>__skip-tsae.json` (dedup on
cell identity — the standalone smoke cell appears in both) and writes
the merged file. Receipts recompute from the leaderboard; these JSONs
are the pool transcripts.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.oprate.render_stage2 [ds]
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
    "txc_batchtopk_post": ("#17becf", "D-", "TXC-post (matched, k=8·T)"),
}
K_PER_TOKEN = 8                    # the panel's per-token code rate
BAND = (5.0, 8.25)                 # card § 2.2 predicted realized-l0 band
POST_EXACT_TOL = 0.02


def merge_pools(ds: str) -> list[dict]:
    merged_path = RES / f"stage2_{ds}.json"
    if merged_path.exists():
        return json.loads(merged_path.read_text())
    rows, seen = [], set()
    for sel in ("only-tsae", "skip-tsae"):
        p = RES / f"stage2_{ds}__{sel}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            key = (r["arch"], r["T"], r["k_pos"], r["seed"], r["kind"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    merged_path.write_text(json.dumps(rows, indent=2))
    return rows


def build_summary(rows, metric: str):
    ok = [r for r in rows if r.get("ok")]
    trained, untrained, chance, l0 = (defaultdict(list) for _ in range(4))
    mismatches, post_untrained = [], []
    for r in ok:
        key = (r["arch"], r["T"])
        m = r["metrics"]
        bucket = untrained if r.get("kind") == "untrained" else trained
        bucket[key].append(m.get(metric, float("nan")))
        chance[key].append(m.get("lambda_chance", float("nan")))
        l0[key].append(m.get("l0_per_token", float("nan")))
        l0t = m.get("l0_per_token", float("nan"))
        cell_id = f"{r['arch']}/T{r['T']}/k{r['k_pos']}/s{r['seed']}"
        if r.get("kind") != "untrained" and np.isfinite(l0t) and \
                not (BAND[0] <= l0t <= BAND[1]):
            mismatches.append({"cell": cell_id, "l0_per_token": float(l0t),
                               "band": list(BAND)})
        if r.get("kind") == "untrained" and r["arch"] == "txc_batchtopk_post":
            post_untrained.append({"cell": cell_id,
                                   "l0_per_token": float(l0t),
                                   "exact_8": bool(abs(l0t - 8.0)
                                                   <= POST_EXACT_TOL)})

    def agg(d):
        return {f"{a}/T{t}": {"mean": float(np.nanmean(v)),
                              "std": float(np.nanstd(v)), "n": len(v)}
                for (a, t), v in sorted(d.items())}

    ci = {}
    for (a, t), v in sorted(trained.items()):
        mean, lo, hi = t_ci95(v)
        ci[f"{a}/T{t}"] = {"mean": mean, "lo": lo, "hi": hi, "n": len(v)}

    l0_range, matched = {}, {}
    for a in {a for (a, _) in trained}:
        cell_means = [float(np.nanmean(l0[(aa, t)]))
                      for (aa, t) in sorted(l0) if aa == a]
        if cell_means:
            l0_range[a] = {"min": min(cell_means), "max": max(cell_means)}
            matched[a] = min(cell_means) >= K_PER_TOKEN / 2

    summary = {
        "datasource": None, "metric": metric,
        "n_cells_ok": len(ok), "n_cells_total": len(rows),
        "trained": agg(trained), "untrained": agg(untrained),
        "chance": agg(chance), "l0_per_token": agg(l0),
        "ci95_trained": ci,
        "l0_range": l0_range,
        "budget_matched": matched,
        "match_rule": ("min cell-mean realized l0_per_token >= "
                       f"{K_PER_TOKEN}/2 = {K_PER_TOKEN / 2} (per-token code "
                       "rate; post is natively matched at nominal k=8·T)"),
        "band_mismatches_card_2_2": mismatches,
        "post_untrained_exactness_card_2_2": post_untrained,
        "margin_trained_minus_untrained": {
            f"{a}/T{t}": float(np.nanmean(v)
                               - np.nanmean(untrained.get((a, t), [np.nan])))
            for (a, t), v in sorted(trained.items())},
    }
    return summary, trained, untrained, l0, matched, l0_range


def load_evidence_line(target: str):
    p = RES / f"evidence_line_{target}.json"
    if not p.exists():
        return {}
    return {row["T"]: row["r"] for row in json.loads(p.read_text())["per_T"]}


def draw(ax, trained, untrained, l0, matched, l0_range, metric, ev_line):
    def l0_tag(a):
        r = l0_range.get(a)
        if r is None:
            return ""
        tag = (f"l0 {r['min']:.3g}" if r["min"] == r["max"]
               else f"l0 {r['min']:.3g}–{r['max']:.3g}")
        if not matched.get(a, True):
            tag += " — NOT budget-matched"
        return f" ({tag})"

    for arch, (col, mk, lab) in ARCH_STYLE.items():
        Ts = sorted({t for (a, t) in trained if a == arch})
        if not Ts:
            continue
        mu, lo, hi = zip(*(t_ci95(trained[(arch, t)]) for t in Ts))
        if len(Ts) == 1:            # token archs: flat reference line
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
    if ev_line:
        Ts = sorted(t for t in ev_line if t > 1)
        ax.plot(Ts, [ev_line[t] for t in Ts], "k*--", lw=1.4, ms=9,
                alpha=0.85,
                label="visible-evidence regression analog (label-side)")
    ax.set_xscale("log", base=2)
    Tall = sorted({t for (_, t) in trained if t > 1})
    if Tall:
        ax.set_xticks(Tall); ax.set_xticklabels(Tall)
    ax.set_xlabel("window size T (tokens)")
    ax.set_ylabel(f"{metric}  (held-out Pearson r of a linear probe)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "ward_real_oprate_case_base_l12"
    target = "case" if "_case_" in ds else "ver"
    rows = merge_pools(ds)
    ev_line = load_evidence_line(target)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)

    variants = [
        ("lambda_recovery", "stage2_oprate_tscaling", "stage2_summary",
         f"oprate Stage 2 — rate_{target} recovery vs T (v1, CANONICAL)\n"
         "REAL Ward activations (base, resid_post L12); whiskers = 95% t CI "
         "over seeds;\nhollow/dotted = untrained control; ★ = label-side "
         "evidence-line analog"),
        ("lambda_recovery_v2", "stage2_oprate_tscaling_v2", "stage2_summary_v2",
         f"oprate Stage 2 — rate_{target} recovery vs T (paired v2 — "
         "reported beside v1, NOT canonical)\nridge + trace split per "
         "PROBE_V2_SPEC § 2; same cells, same evidence line"),
    ]
    for metric, stem, sumstem, title in variants:
        summary, trained, untrained, l0, matched, l0_range = \
            build_summary(rows, metric)
        summary["datasource"] = ds
        summary["evidence_line"] = ev_line
        (RES / f"{sumstem}_{ds}.json").write_text(json.dumps(summary,
                                                             indent=2))
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        draw(ax, trained, untrained, l0, matched, l0_range, metric, ev_line)
        ax.set_title(title, fontsize=10.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIGS / f"{stem}.{ext}",
                        dpi=140 if ext == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)
        if metric == "lambda_recovery":
            print(json.dumps({
                "n_ok": summary["n_cells_ok"],
                "n_total": summary["n_cells_total"],
                "budget_matched": summary["budget_matched"],
                "l0_range": summary["l0_range"],
                "band_mismatches": summary["band_mismatches_card_2_2"],
                "post_untrained": summary["post_untrained_exactness_card_2_2"],
            }, indent=2))
    print(f"-> {FIGS}/stage2_oprate_tscaling[. _v2.]* ; "
          f"{RES}/stage2_summary[_v2]_{ds}.json")


if __name__ == "__main__":
    main()
