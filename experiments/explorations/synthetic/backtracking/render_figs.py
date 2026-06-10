"""Backtracking architecture results — SINGLE SOURCE renderer + record populator.

Reads the **canonical leaderboard** (`results/leaderboard.jsonl`) — the one
code-version-stamped source — filters the `toy_backtracking_selfexcite` cells
(protocol 1.2.0, non-smoke), aggregates over seeds, then in one pass:

  1. renders paper-quality figures into `figs/`
     (`backtracking_main`, `backtracking_untrained_control`,
      `backtracking_local_tradeoff`),
  2. writes the machine-readable aggregate `results/backtracking_bench_stats.json`,
  3. fills every `<!-- BEGIN AUTO:<tag> --> … <!-- END AUTO:<tag> -->` block in
     `bench_record.md` (headline numbers + all result tables).

Re-running rebuilds the record's numbers, figures, and stats from the canonical
leaderboard — there is no hand-typing and nothing can drift. The per-token DPI
floor is computed directly from the generator (also canonical).

    .venv/bin/python -m experiments.explorations.synthetic.backtracking.render_figs
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # 
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
FIG_DIR = HERE / "figs"
RES_DIR = HERE / "results"
STATS_OUT = RES_DIR / "backtracking_bench_stats.json"
RECORD = HERE / "bench_record.md"
DS = "toy_backtracking_selfexcite_d64"
PROTOCOL = "1.2.0"

F = 20
D_SAES = [8, 16, 20, 40]
# BatchTopK fair-backbone family: every arch shares the BatchTopK→JumpReLU
# backbone, so the only variable is decode structure.
ARCH_T = [("batchtopk_sae", 1), ("tsae", 1),
          ("txc_batchtopk_pre", 2), ("txc_batchtopk_pre", 4), ("txc_batchtopk_pre", 8),
          ("txc_batchtopk_post", 2), ("txc_batchtopk_post", 4), ("txc_batchtopk_post", 8),
          ("stacked_batchtopk", 2), ("stacked_batchtopk", 4), ("stacked_batchtopk", 8)]
PER_TOKEN = {("batchtopk_sae", 1), ("tsae", 1)}
LABEL = {"batchtopk_sae": "BatchTopK-SAE", "tsae": "T-SAE",
         "txc_batchtopk_pre": "TXC-pre", "txc_batchtopk_post": "TXC-post",
         "stacked_batchtopk": "Stacked-SAE"}
# Window families: TXC-pre = blues, TXC-post = purples, Stacked = greens.
WINDOW_FAMILIES = [("txc_batchtopk_pre", "#3182bd"),
                   ("txc_batchtopk_post", "#807dba"),
                   ("stacked_batchtopk", "#31a354")]
COLORS = {
    ("batchtopk_sae", 1): "#D55E00", ("tsae", 1): "#E69F00",            # per-token: vermillion/orange
    ("txc_batchtopk_pre", 2): "#9ecae1", ("txc_batchtopk_pre", 4): "#3182bd", ("txc_batchtopk_pre", 8): "#08519c",
    ("txc_batchtopk_post", 2): "#bcbddc", ("txc_batchtopk_post", 4): "#807dba", ("txc_batchtopk_post", 8): "#54278f",
    ("stacked_batchtopk", 2): "#a1d99b", ("stacked_batchtopk", 4): "#31a354", ("stacked_batchtopk", 8): "#006d2c",
}
MARK = {1: "o", 2: "s", 4: "^", 8: "D"}

PAPER_STYLE = {
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.7,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
    "figure.facecolor": "white", "mathtext.default": "regular",
}


def label(arch, T):
    return f"{LABEL[arch]} (per-token)" if (arch, T) in PER_TOKEN else f"{LABEL[arch]} (T={T})"


# ── canonical source: the leaderboard ─────────────────────────────────

def load_rows():
    rows = []
    for line in LEADERBOARD.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("datasource") != DS or r.get("evaluator_protocol_version") != PROTOCOL:
            continue
        ec = r.get("eval_cfg") or {}
        if ec.get("smoke"):
            continue
        ov = r.get("training_cfg", {}).get("arch_hparams_override") or {}
        n_steps = int(r.get("training_cfg", {}).get("n_steps", 0))
        rows.append({"arch": r["arch"], "T": int(ov.get("T", 1)), "d_sae": int(ov.get("d_sae")),
                     "k_pos": int(ec.get("k_pos", ov.get("k_pos", 1))), "seed": int(r["seed"]),
                     "kind": "trained" if n_steps > 0 else "untrained", "m": r["metrics"]})
    return rows


def aggregate(rows):
    """(kind, k_pos, arch, T, d_sae) -> {metric: (mean, std, n)} over seeds."""
    buck = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["kind"], r["k_pos"], r["arch"], r["T"], r["d_sae"])
        for m, v in r["m"].items():
            if v is not None and np.isfinite(v):
                buck[key][m].append(float(v))
    return {k: {m: (float(np.mean(vs)), float(np.std(vs)), len(vs)) for m, vs in d.items()}
            for k, d in buck.items()}


def per_token_dpi_floor():
    """Authoritative per-token ceiling sqrt(Var λ / Var b) from the generator."""
    from temp_bench.core.config import load_datasource
    from temp_bench.data.synthetic import materialise
    data = materialise(load_datasource(DS), seed=1)
    lam = data.extra["lambda_labels"].numpy(); b = data.extra["b_labels"].numpy()
    return float(np.sqrt(lam.var() / b.var()))


def g(agg, kind, kpos, arch, T, d, metric="lambda_recovery"):
    c = agg.get((kind, kpos, arch, T, d))
    return c[metric] if c and metric in c else (float("nan"), float("nan"), 0)


# ── paper-quality figures ─────────────────────────────────────────────

def fig_main(agg, pt, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    # (a) frontier: lambda vs d_sae (trained, k_pos=1)
    ax1.axvspan(7, F, color="0.93", zorder=0, lw=0)
    ax1.axvline(F, color="0.45", ls=":", lw=1.1, zorder=1)
    ax1.text(F - 0.5, 0.04, "F", color="0.4", fontsize=10, ha="right", style="italic")
    ax1.axhline(pt, color="#999999", ls="--", lw=1.1, zorder=1)
    ax1.text(40, pt - 0.045, f"per-token DPI floor  $\\sqrt{{Var\\,\\lambda/Var\\,b}}$ = {pt:.2f}",
             color="#555", fontsize=8.3, ha="right", va="top")
    for arch, T in ARCH_T:
        xs, ys, es = [], [], []
        for d in D_SAES:
            m, s, n = g(agg, "trained", 1, arch, T, d)
            if n:
                xs.append(d); ys.append(m); es.append(s)
        if xs:
            ls = "--" if (arch, T) in PER_TOKEN else "-"
            ax1.errorbar(xs, ys, yerr=es, marker=MARK[T], ms=5.5, lw=1.9, ls=ls,
                         color=COLORS[(arch, T)], capsize=2, elinewidth=1, label=label(arch, T))
    ax1.set_xticks(D_SAES); ax1.set_xlim(7, 42); ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("dictionary size  $d_{sae}$"); ax1.set_ylabel("$\\lambda$-recovery (held-out corr.)")
    ax1.set_title("(a) Hidden-intensity recovery vs capacity", loc="left", fontsize=11)

    # (b) lambda vs T at d_sae=20 (trained, k_pos=1)
    ax2.axhline(pt, color="#999999", ls="--", lw=1.1)
    ax2.axhline(1.0, color="#cccccc", ls=":", lw=1.0)
    ax2.text(8, 0.97, "window info ceiling = 1", color="#999", fontsize=8, ha="right", va="top")
    for arch, T in ARCH_T:           # per-token points at T=1
        if (arch, T) in PER_TOKEN:
            m, s, n = g(agg, "trained", 1, arch, T, 20)
            if n:
                ax2.errorbar([1], [m], yerr=[s], marker=MARK[1], ms=8, color=COLORS[(arch, T)],
                             capsize=3, label=label(arch, T))
    for fam, col in WINDOW_FAMILIES:
        ts, ys, es = [], [], []
        for T in (2, 4, 8):
            m, s, n = g(agg, "trained", 1, fam, T, 20)
            if n:
                ts.append(T); ys.append(m); es.append(s)
        if ts:
            ax2.errorbar(ts, ys, yerr=es, marker="o", ms=6, lw=2, color=col, capsize=3,
                         label=f"{LABEL[fam]} (window)")
    ax2.set_xscale("log", base=2); ax2.set_xticks([1, 2, 4, 8]); ax2.set_xticklabels([1, 2, 4, 8])
    ax2.set_xlim(0.85, 9.5); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("window size  $T$   ($T{=}1$: per-token)"); ax2.set_ylabel("$\\lambda$-recovery (held-out corr.)")
    ax2.set_title("(b) Recovery rises with $T$, saturates by $T{=}4$", loc="left", fontsize=11)
    ax2.legend(loc="lower right", fontsize=8.5)

    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=8.3,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, plt, "backtracking_main")


def fig_untrained(agg, pt, plt):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels, un, uns, tr, trs, cols = [], [], [], [], [], []
    for arch, T in ARCH_T:
        u = g(agg, "untrained", 1, arch, T, 20); t = g(agg, "trained", 1, arch, T, 20)
        labels.append(label(arch, T).replace(" (", "\n(")); cols.append(COLORS[(arch, T)])
        un.append(u[0]); uns.append(u[1]); tr.append(t[0]); trs.append(t[1])
    x = np.arange(len(labels)); w = 0.4
    ax.bar(x - w / 2, un, w, yerr=uns, capsize=2, color="#cfcfcf", edgecolor="0.4", lw=0.5,
           label="untrained (architectural access)")
    ax.bar(x + w / 2, tr, w, yerr=trs, capsize=2, color=cols, edgecolor="0.25", lw=0.5,
           label="trained (access + learning)")
    ax.axhline(pt, color="#999999", ls="--", lw=1.2)
    ax.text(len(labels) - 0.5, pt + 0.012, f"per-token DPI floor = {pt:.2f}", color="#555",
            fontsize=8.5, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.2); ax.set_ylim(0, 1.02)
    ax.set_ylabel("$\\lambda$-recovery (held-out corr.)")
    ax.set_title("Access vs learning: random-init vs trained encoders  ($d_{sae}{=}20$, $k_{pos}{=}1$)",
                 fontsize=11.5)
    ax.legend(loc="upper center", ncol=2); ax.grid(axis="x", alpha=0)
    fig.tight_layout(); _save(fig, plt, "backtracking_untrained_control")


def fig_local_tradeoff(agg, plt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, metric, ttl, yl in [(a1, "eauc", "(a) Local feature recovery", "eAUC"),
                                (a2, "nmse", "(b) Reconstruction error", "NMSE")]:
        ax.axvspan(7, F, color="0.93", zorder=0, lw=0); ax.axvline(F, color="0.45", ls=":", lw=1.1)
        for arch, T in ARCH_T:
            xs, ys, es = [], [], []
            for d in D_SAES:
                m, s, n = g(agg, "trained", 1, arch, T, d, metric)
                if n:
                    xs.append(d); ys.append(m); es.append(s)
            if xs:
                ls = "--" if (arch, T) in PER_TOKEN else "-"
                ax.errorbar(xs, ys, yerr=es, marker=MARK[T], ms=5, lw=1.7, ls=ls,
                            color=COLORS[(arch, T)], capsize=2, elinewidth=0.9, label=label(arch, T))
        ax.set_xticks(D_SAES); ax.set_xlim(7, 42); ax.set_xlabel("dictionary size  $d_{sae}$")
        ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=11)
    a1.set_ylim(0, 1.02); a1.legend(ncol=2, fontsize=7.8, loc="upper left")
    fig.tight_layout(); _save(fig, plt, "backtracking_local_tradeoff")


def fig_specialization(agg, plt):
    """The local-vs-temporal plane: eAUC (local) × lambda-recovery (order-sensitive).

    One marker per (arch, T) at the F anchor (d_sae=20); a faint trajectory traces
    d_sae ∈ {8,16,20,40}. Per-token archs sit low on the λ axis (local specialists);
    window archs sit high (temporal specialists) — the architectural specialization.
    """
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.axhspan(0.70, 1.02, color="#eef4fb", zorder=0, lw=0)
    ax.axhline(0.70, color="#9ecae1", lw=0.8, ls=":", zorder=1)
    ax.text(0.015, 1.005, "temporal (order-sensitive) specialist", color="#2b6cb0",
            fontsize=9.5, va="top", style="italic")
    ax.text(0.985, 0.02, "local-feature specialist", color="#a84300",
            fontsize=9.5, ha="right", va="bottom", style="italic")
    for arch, T in ARCH_T:
        col = COLORS[(arch, T)]; mk = "X" if (arch, T) in PER_TOKEN else MARK[T]
        pts = []
        for d in D_SAES:
            ex = g(agg, "trained", 1, arch, T, d, "eauc"); ly = g(agg, "trained", 1, arch, T, d)
            if ex[2] and ly[2]:
                pts.append((d, ex[0], ly[0], ex[1], ly[1]))
        if not pts:
            continue
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=col, lw=1.0, alpha=0.3,
                ls="--" if (arch, T) in PER_TOKEN else "-", zorder=2)
        for d, ex, ly, exs, lys in pts:
            if d == 20:
                ax.scatter([ex], [ly], s=150, color=col, edgecolor="0.2", linewidth=0.8,
                           marker=mk, zorder=4, label=label(arch, T))
                ax.errorbar([ex], [ly], xerr=[exs], yerr=[lys], color=col, lw=1, capsize=2, zorder=4)
            else:
                ax.scatter([ex], [ly], s=26, color=col, marker=mk, alpha=0.4, zorder=3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.04)
    ax.set_xlabel("local feature recovery   (eAUC)")
    ax.set_ylabel("order-sensitive latent recovery   ($\\lambda$-recovery)")
    ax.set_title("Architectural specialization: local vs temporal recovery", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.6,
              title="(arch, T) @ $d_{sae}{=}20$\nfaint trail: $d_{sae}\\in\\{8,16,20,40\\}$",
              title_fontsize=8.2)
    fig.tight_layout()
    _save(fig, plt, "backtracking_specialization")


def _save(fig, plt, name):
    for ext, dpi in [("pdf", None), ("png", 200), ("thumb.png", 70)]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"[fig] {FIG_DIR.name}/{name}.{{pdf,png}}")


# ── tables + headline for bench_record.md ─────────────────────────────

def _f(t, dec=3):
    m, s, n = t
    return "—" if not n else f"{m:.{dec}f}"


def _fs(t):
    m, s, n = t
    return "—" if not n else f"{m:.3f} ±{s:.3f}"


def table_lambda_frontier(agg):
    h = ("| arch / T | " + " | ".join(f"d={d}" for d in D_SAES) + " |\n"
         "|---|" + "|".join("---" for _ in D_SAES) + "|\n")
    for arch, T in ARCH_T:
        cells = " | ".join(_f(g(agg, "trained", 1, arch, T, d)) for d in D_SAES)
        name = f"**{label(arch,T)}**" if (arch, T) not in PER_TOKEN else label(arch, T)
        h += f"| {name} | {cells} |\n"
    return h.rstrip()


def table_eauc(agg):
    h = ("| arch / T | " + " | ".join(f"d={d}" for d in D_SAES) + " |\n"
         "|---|" + "|".join("---" for _ in D_SAES) + "|\n")
    for arch, T in ARCH_T:
        cells = " | ".join(_f(g(agg, "trained", 1, arch, T, d, "eauc")) for d in D_SAES)
        h += f"| {label(arch,T)} | {cells} |\n"
    return h.rstrip()


def table_untrained(agg):
    h = "| arch / T | untrained (access) | trained (access+learning) |\n|---|---|---|\n"
    for arch, T in ARCH_T:
        h += f"| {label(arch,T)} | {_fs(g(agg,'untrained',1,arch,T,20))} | {_fs(g(agg,'trained',1,arch,T,20))} |\n"
    return h.rstrip()


def table_kpos(agg):
    h = ("| arch / T | $\\lambda$ @ $k_{pos}{=}1$ | $\\lambda$ @ $k_{pos}{=}2$ | eAUC @1 | eAUC @2 |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {_f(g(agg,'trained',1,arch,T,20))} | {_f(g(agg,'trained',2,arch,T,20))} "
              f"| {_f(g(agg,'trained',1,arch,T,20,'eauc'))} | {_f(g(agg,'trained',2,arch,T,20,'eauc'))} |\n")
    return h.rstrip()


def headline_block(agg, pt):
    pt_tok = np.nanmean([g(agg, "trained", 1, a, 1, 20)[0] for a in ("batchtopk_sae", "tsae")])
    pre_t2 = g(agg, "trained", 1, "txc_batchtopk_pre", 2, 20)[0]
    pre_t4 = g(agg, "trained", 1, "txc_batchtopk_pre", 4, 20)[0]
    post_t4 = g(agg, "trained", 1, "txc_batchtopk_post", 4, 20)[0]
    stk_t4 = g(agg, "trained", 1, "stacked_batchtopk", 4, 20)[0]
    win_scarce = g(agg, "trained", 1, "txc_batchtopk_pre", 4, 8)[0]
    best_win_t4 = np.nanmax([pre_t4, post_t4, stk_t4])
    un_win = g(agg, "untrained", 1, "txc_batchtopk_pre", 4, 20)[0]
    return (
        f"- **Fair backbone:** every arch shares the BatchTopK→JumpReLU backbone "
        f"(Bussmann et al.) + AuxK + decoder unit-norm, on equal tokens/step — so the "
        f"only variable is decode structure.\n"
        f"- **Per-token DPI floor** (provable, from the generator): "
        f"$\\sqrt{{Var\\,\\lambda/Var\\,b}}$ = **{pt:.2f}**. Trained per-token (BatchTopK) SAEs land at "
        f"**{pt_tok:.2f}** at d_sae=20, flat across all capacities.\n"
        f"- **Window recovery** at d_sae=20: TXC-pre $\\lambda$ = **{pre_t2:.2f}** (T=2) → "
        f"**{pre_t4:.2f}** (T≥4); TXC-post **{post_t4:.2f}**; Stacked **{stk_t4:.2f}** (T=4). "
        f"Holds at d_sae=8 < F=20 (TXC-pre = **{win_scarce:.2f}**, scarce regime).\n"
        f"- **Gap** (best window T4 − per-token): **{best_win_t4 - pt_tok:.2f}**. "
        f"Untrained window already reaches {un_win:.2f} (architectural access); training lifts it to {best_win_t4:.2f}."
    )


def populate(blocks):
    if not RECORD.exists():
        print(f"[warn] {RECORD} missing — populate skipped"); return
    txt = RECORD.read_text(); filled = 0
    for tag, content in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN AUTO:{tag} -->).*?(<!-- END AUTO:{tag} -->)", re.DOTALL)
        if not pat.search(txt):
            print(f"[warn] AUTO:{tag} marker not found in bench_record.md"); continue
        txt = pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", txt); filled += 1
    RECORD.write_text(txt)
    print(f"[record] populated {filled}/{len(blocks)} AUTO block(s) in {RECORD.name}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)
    FIG_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

    rows = load_rows()
    agg = aggregate(rows)
    pt = per_token_dpi_floor()
    n_trained = sum(1 for r in rows if r["kind"] == "trained")
    print(f"[render] {len(rows)} leaderboard cells ({n_trained} trained); per-token DPI floor = {pt:.3f}")

    fig_main(agg, pt, plt)
    fig_specialization(agg, plt)
    fig_untrained(agg, pt, plt)
    fig_local_tradeoff(agg, plt)

    blocks = {
        "headline": headline_block(agg, pt),
        "lambda_frontier": table_lambda_frontier(agg),
        "eauc": table_eauc(agg),
        "untrained": table_untrained(agg),
        "kpos": table_kpos(agg),
    }
    populate(blocks)

    STATS_OUT.write_text(json.dumps({
        "source": "results/leaderboard.jsonl", "n_cells": len(rows), "F": F,
        "per_token_dpi_floor": pt,
        "agg": {f"{k[0]}|kpos{k[1]}|{k[2]}|T{k[3]}|d{k[4]}": v for k, v in agg.items()},
    }, indent=2))
    print(f"[stats] -> {STATS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
