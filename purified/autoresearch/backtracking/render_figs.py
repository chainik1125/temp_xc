"""Render the backtracking (self-exciting) benchmark figures + stats summary.

Reads autoresearch/backtracking/results/backtracking_grid_results.json (the grid driver's dump)
and produces paper-ready frontier figures + a stats JSON for the record:

  fig 1 (headline): lambda_recovery vs d_sae, one line per (arch,T); F=20 marked,
                    scarce region shaded; per-token DPI floor + window ceilings.
  fig 2: lambda_recovery vs window size T (at d_sae=20) — rise + saturation.
  fig 3: trained vs untrained at d_sae=20 — the access/learning decomposition.
  fig 4: eAUC + NMSE vs d_sae (local feature recovery / reconstruction).

    .venv/bin/python -m autoresearch.backtracking.render_figs
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "autoresearch" / "backtracking" / "results" / "backtracking_grid_results.json"
FIG_DIR = ROOT / "autoresearch" / "backtracking" / "figs"
STATS_OUT = ROOT / "autoresearch" / "backtracking" / "results" / "backtracking_bench_stats.json"

F = 20                       # ground-truth feature count
PT_CEIL = 0.408              # per-token DPI ceiling (gating)
WIN_CEIL = {2: 0.908, 4: 0.986, 8: 0.986}   # window linear ceilings (gating)
ARCH_T = [("topk_sae", 1), ("tsae", 1),
          ("txc_base", 2), ("txc_base", 4), ("txc_base", 8),
          ("stacked_sae", 2), ("stacked_sae", 4), ("stacked_sae", 8)]
PER_TOKEN = {("topk_sae", 1), ("tsae", 1)}
D_SAES = [8, 16, 20, 40]
COLORS = {
    ("topk_sae", 1): "#d62728", ("tsae", 1): "#ff7f0e",
    ("txc_base", 2): "#aec7e8", ("txc_base", 4): "#1f77b4", ("txc_base", 8): "#08306b",
    ("stacked_sae", 2): "#98df8a", ("stacked_sae", 4): "#2ca02c", ("stacked_sae", 8): "#00441b",
}


def _label(arch, T):
    return f"{arch} (T={T})" if (arch, T) not in PER_TOKEN else f"{arch} (per-token)"


def _agg(results):
    """(kind, arch, T, d_sae) -> {metric: (mean, std, n)} over seeds."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in results:
        if not r.get("ok"):
            continue
        key = (r["kind"], r["arch"], r["T"], r["d_sae"])
        for m, v in r["metrics"].items():
            if v is not None and np.isfinite(v):
                buckets[key][m].append(float(v))
    agg = {}
    for key, ms in buckets.items():
        agg[key] = {m: (float(np.mean(vs)), float(np.std(vs)), len(vs)) for m, vs in ms.items()}
    return agg


def _series(agg, kind, arch, T, metric):
    xs, ys, es = [], [], []
    for d in D_SAES:
        cell = agg.get((kind, arch, T, d))
        if cell and metric in cell:
            xs.append(d); ys.append(cell[metric][0]); es.append(cell[metric][1])
    return np.array(xs), np.array(ys), np.array(es)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(RESULTS.read_text())
    agg = _agg(results)
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[render] {n_ok}/{len(results)} ok cells")

    # ---- fig 1: lambda_recovery vs d_sae (headline frontier) ----
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.axvspan(min(D_SAES) - 1, F, color="0.92", zorder=0)            # scarce region d_sae<=F
    ax.axvline(F, color="0.4", ls=":", lw=1.2)
    ax.text(F, 0.02, "  F=20", color="0.4", fontsize=9, va="bottom")
    ax.axhline(PT_CEIL, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax.text(D_SAES[-1], PT_CEIL + 0.01, "per-token DPI ceiling 0.41", color="#d62728",
            fontsize=8, ha="right", va="bottom")
    ax.axhline(WIN_CEIL[4], color="#1f77b4", ls="--", lw=1, alpha=0.5)
    ax.text(D_SAES[-1], WIN_CEIL[4] + 0.005, "window ceiling (T≥4) 0.99", color="#1f77b4",
            fontsize=8, ha="right", va="bottom")
    for arch, T in ARCH_T:
        xs, ys, es = _series(agg, "trained", arch, T, "lambda_recovery")
        if len(xs):
            ls = "--" if (arch, T) in PER_TOKEN else "-"
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8, ls=ls,
                        color=COLORS[(arch, T)], label=_label(arch, T), capsize=2)
    ax.set_xlabel("dictionary size d_sae"); ax.set_ylabel("λ recovery (held-out corr)")
    ax.set_title("Backtracking bench: hidden-intensity (λ) recovery vs capacity")
    ax.set_xticks(D_SAES); ax.set_ylim(0, 1.0); ax.legend(fontsize=7.5, ncol=2, loc="center right")
    ax.grid(True, alpha=0.25)
    _save(fig, plt, "backtracking_lambda_frontier")

    # ---- fig 2: lambda_recovery vs T at d_sae=20 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    # per-token archs plotted at T=1
    for arch, T in ARCH_T:
        if (arch, T) in PER_TOKEN:
            cell = agg.get(("trained", arch, T, 20))
            if cell and "lambda_recovery" in cell:
                m, s, _ = cell["lambda_recovery"]
                ax.errorbar([1], [m], yerr=[s], marker="s", ms=8, color=COLORS[(arch, T)],
                            label=_label(arch, T), capsize=3)
    for fam, col in [("txc_base", "#1f77b4"), ("stacked_sae", "#2ca02c")]:
        ts, ys, es = [], [], []
        for T in (2, 4, 8):
            cell = agg.get(("trained", fam, T, 20))
            if cell and "lambda_recovery" in cell:
                ts.append(T); ys.append(cell["lambda_recovery"][0]); es.append(cell["lambda_recovery"][1])
        if ts:
            ax.errorbar(ts, ys, yerr=es, marker="o", ms=6, lw=2, color=col, label=f"{fam} (window)", capsize=3)
    ax.plot([1, 2, 4, 8], [PT_CEIL, WIN_CEIL[2], WIN_CEIL[4], WIN_CEIL[8]], "k:", lw=1, alpha=0.6,
            label="linear ceiling (gating)")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
    ax.set_xlabel("window size T (T=1 = per-token)"); ax.set_ylabel("λ recovery (held-out corr)")
    ax.set_title("λ recovery rises with T, saturates by T=4 (d_sae=20)")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    _save(fig, plt, "backtracking_lambda_vs_T")

    # ---- fig 3: trained vs untrained (access vs learning) at d_sae=20 ----
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels, tr, trs, un, uns = [], [], [], [], []
    for arch, T in ARCH_T:
        ct = agg.get(("trained", arch, T, 20)); cu = agg.get(("untrained", arch, T, 20))
        labels.append(_label(arch, T).replace(" (", "\n("))
        tr.append(ct["lambda_recovery"][0] if ct and "lambda_recovery" in ct else 0)
        trs.append(ct["lambda_recovery"][1] if ct and "lambda_recovery" in ct else 0)
        un.append(cu["lambda_recovery"][0] if cu and "lambda_recovery" in cu else 0)
        uns.append(cu["lambda_recovery"][1] if cu and "lambda_recovery" in cu else 0)
    xpos = np.arange(len(labels)); wbar = 0.38
    ax.bar(xpos - wbar / 2, un, wbar, yerr=uns, label="untrained (access)", color="0.7", capsize=2)
    ax.bar(xpos + wbar / 2, tr, wbar, yerr=trs, label="trained (access+learning)", color="#1f77b4", capsize=2)
    ax.axhline(PT_CEIL, color="#d62728", ls="--", lw=1, alpha=0.7, label="per-token DPI ceiling")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("λ recovery (held-out corr)"); ax.set_ylim(0, 1.0)
    ax.set_title("Access vs learning: trained vs random-init window encoders (d_sae=20)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis="y")
    _save(fig, plt, "backtracking_untrained_control")

    # ---- fig 4: eAUC + NMSE vs d_sae ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, ttl in [(a1, "eauc", "Local feature recovery (eAUC)"),
                            (a2, "nmse", "Reconstruction NMSE")]:
        ax.axvspan(min(D_SAES) - 1, F, color="0.92", zorder=0); ax.axvline(F, color="0.4", ls=":", lw=1.2)
        for arch, T in ARCH_T:
            xs, ys, es = _series(agg, "trained", arch, T, metric)
            if len(xs):
                ls = "--" if (arch, T) in PER_TOKEN else "-"
                ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, lw=1.5, ls=ls,
                            color=COLORS[(arch, T)], label=_label(arch, T), capsize=2)
        ax.set_xlabel("d_sae"); ax.set_title(ttl); ax.set_xticks(D_SAES); ax.grid(True, alpha=0.25)
    a1.set_ylabel("eAUC"); a2.set_ylabel("NMSE"); a1.legend(fontsize=7, ncol=2)
    _save(fig, plt, "backtracking_eauc_nmse")

    # ---- stats summary for the record ----
    summary = {"n_ok": n_ok, "n_total": len(results), "F": F,
               "ceilings": {"per_token": PT_CEIL, "window": WIN_CEIL},
               "agg": {f"{k[0]}|{k[1]}|T{k[2]}|d{k[3]}": v for k, v in agg.items()}}
    STATS_OUT.write_text(json.dumps(summary, indent=2))
    print(f"[render] stats -> {STATS_OUT}")
    _print_headline(agg)


def _print_headline(agg):
    def g(kind, arch, T, d, m="lambda_recovery"):
        c = agg.get((kind, arch, T, d)); return c[m][0] if c and m in c else float("nan")
    print("\n=== HEADLINE (λ recovery, d_sae=20) ===")
    for arch, T in ARCH_T:
        print(f"  {_label(arch,T):<26} trained={g('trained',arch,T,20):.3f}  untrained={g('untrained',arch,T,20):.3f}")


def _save(fig, plt, name):
    fig.tight_layout()
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[render] -> {FIG_DIR}/{name}.*")


if __name__ == "__main__":
    main()
