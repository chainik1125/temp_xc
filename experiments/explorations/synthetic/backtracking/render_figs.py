"""Backtracking architecture results — thin driver over the shared record pipeline.

Config + bench-specific figures/tables only; leaderboard→aggregate→AUTO-block→
stats plumbing lives in :mod:`explorations.synthetic`. Reads the canonical
leaderboard, filters the `toy_backtracking_selfexcite` cells (protocol 1.3.0,
non-smoke), aggregates over seeds, renders figures, fills every `<!-- AUTO:* -->`
block in `bench_record.md`, and writes `results/backtracking_bench_stats.json`.
The per-token DPI floor is computed directly from the generator (also canonical).

    .venv/bin/python -m experiments.explorations.synthetic.backtracking.render_figs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from explorations.synthetic import figs, record
from explorations.synthetic.figs import MARK, frontier_series, save_fig
from explorations.synthetic.record import aggregate, fmt, fmt_pm, load_rows, populate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
FIG_DIR = HERE / "figs"
RES_DIR = HERE / "results"
STATS_OUT = RES_DIR / "backtracking_bench_stats.json"
RECORD = HERE / "bench_record.md"
DS = "toy_backtracking_selfexcite_d64"
PROTOCOL = "1.3.0"
KEY_FIELDS = ("kind", "k_pos", "arch", "T", "d_sae")

F = 20
D_SAES = [10, 20, 40]   # {F//2, F, 2F} — uniform clean-room design
# BatchTopK fair-backbone family: every arch shares the BatchTopK→JumpReLU
# backbone, so the only variable is decode structure. spectral_txc joins the
# window family under the uniform design (here it is a DCT-band window arch).
ARCH_T = [("batchtopk_sae", 1), ("tsae", 1),
          ("txc_batchtopk_pre", 2), ("txc_batchtopk_pre", 4), ("txc_batchtopk_pre", 8),
          ("txc_batchtopk_post", 2), ("txc_batchtopk_post", 4), ("txc_batchtopk_post", 8),
          ("stacked_batchtopk", 2), ("stacked_batchtopk", 4), ("stacked_batchtopk", 8),
          ("spectral_txc", 2), ("spectral_txc", 4), ("spectral_txc", 8)]
PER_TOKEN = {("batchtopk_sae", 1), ("tsae", 1)}
LABEL = {"batchtopk_sae": "BatchTopK-SAE", "tsae": "T-SAE",
         "txc_batchtopk_pre": "TXC-pre", "txc_batchtopk_post": "TXC-post",
         "stacked_batchtopk": "Stacked-SAE", "spectral_txc": "Spectral-TXC"}
# Window families: TXC-pre = blues, TXC-post = purples, Stacked = greens,
# Spectral = oranges.
WINDOW_FAMILIES = [("txc_batchtopk_pre", "#3182bd"),
                   ("txc_batchtopk_post", "#807dba"),
                   ("stacked_batchtopk", "#31a354"),
                   ("spectral_txc", "#f16913")]
COLORS = {
    ("batchtopk_sae", 1): "#D55E00", ("tsae", 1): "#E69F00",            # per-token: vermillion/orange
    ("txc_batchtopk_pre", 2): "#9ecae1", ("txc_batchtopk_pre", 4): "#3182bd", ("txc_batchtopk_pre", 8): "#08519c",
    ("txc_batchtopk_post", 2): "#bcbddc", ("txc_batchtopk_post", 4): "#807dba", ("txc_batchtopk_post", 8): "#54278f",
    ("stacked_batchtopk", 2): "#a1d99b", ("stacked_batchtopk", 4): "#31a354", ("stacked_batchtopk", 8): "#006d2c",
    ("spectral_txc", 2): "#fdae6b", ("spectral_txc", 4): "#f16913", ("spectral_txc", 8): "#a63603",
}


def label(arch, T):
    return f"{LABEL[arch]} (per-token)" if (arch, T) in PER_TOKEN else f"{LABEL[arch]} (T={T})"


def per_token_dpi_floor():
    """Authoritative per-token ceiling sqrt(Var λ / Var b) from the generator."""
    from temp_bench.core.config import load_datasource
    from temp_bench.data.synthetic import materialise
    data = materialise(load_datasource(DS), seed=1)
    lam = data.extra["lambda_labels"].numpy(); b = data.extra["b_labels"].numpy()
    return float(np.sqrt(lam.var() / b.var()))


def g(agg, kind, kpos, arch, T, d, metric="lambda_recovery"):
    return record.get(agg, (kind, kpos, arch, T, d), metric)


# ── paper-quality figures ─────────────────────────────────────────────

def fig_main(agg, pt, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    # (a) frontier: lambda vs d_sae (trained, k_pos=1)
    ax1.axvspan(9, F, color="0.93", zorder=0, lw=0)
    ax1.axvline(F, color="0.45", ls=":", lw=1.1, zorder=1)
    ax1.text(F - 0.5, 0.04, "F", color="0.4", fontsize=10, ha="right", style="italic")
    ax1.axhline(pt, color="#999999", ls="--", lw=1.1, zorder=1)
    ax1.text(40, pt - 0.045, f"per-token DPI floor  $\\sqrt{{Var\\,\\lambda/Var\\,b}}$ = {pt:.2f}",
             color="#555", fontsize=8.3, ha="right", va="top")
    frontier_series(ax1, ARCH_T, D_SAES, lambda a, T, d: g(agg, "trained", 1, a, T, d),
                    COLORS, PER_TOKEN, label, ms=5.5, lw=1.9, capsize=2, elinewidth=1)
    ax1.set_xticks(D_SAES); ax1.set_xlim(9, 42); ax1.set_ylim(0, 1.02)
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
    save_fig(fig, FIG_DIR, "backtracking_main", plt)


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
    fig.tight_layout(); save_fig(fig, FIG_DIR, "backtracking_untrained_control", plt)


def fig_local_tradeoff(agg, plt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, metric, ttl, yl in [(a1, "eauc", "(a) Local feature recovery", "eAUC"),
                                (a2, "nmse", "(b) Reconstruction error", "NMSE")]:
        ax.axvspan(9, F, color="0.93", zorder=0, lw=0); ax.axvline(F, color="0.45", ls=":", lw=1.1)
        frontier_series(ax, ARCH_T, D_SAES,
                        lambda a, T, d, _m=metric: g(agg, "trained", 1, a, T, d, _m),
                        COLORS, PER_TOKEN, label, ms=5, lw=1.7, capsize=2, elinewidth=0.9)
        ax.set_xticks(D_SAES); ax.set_xlim(9, 42); ax.set_xlabel("dictionary size  $d_{sae}$")
        ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=11)
    a1.set_ylim(0, 1.02); a1.legend(ncol=2, fontsize=7.8, loc="upper left")
    fig.tight_layout(); save_fig(fig, FIG_DIR, "backtracking_local_tradeoff", plt)


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
              title="(arch, T) @ $d_{sae}{=}20$\nfaint trail: $d_{sae}\\in\\{10,20,40\\}$",
              title_fontsize=8.2)
    fig.tight_layout()
    save_fig(fig, FIG_DIR, "backtracking_specialization", plt)


# ── tables + headline for bench_record.md ─────────────────────────────

def table_lambda_frontier(agg):
    return record.frontier_table(
        ARCH_T, D_SAES, lambda a, T, d: g(agg, "trained", 1, a, T, d),
        label, bold_pred=lambda a, T: (a, T) not in PER_TOKEN)


def table_eauc(agg):
    return record.frontier_table(
        ARCH_T, D_SAES, lambda a, T, d: g(agg, "trained", 1, a, T, d, "eauc"), label)


def table_untrained(agg):
    h = "| arch / T | untrained (access) | trained (access+learning) |\n|---|---|---|\n"
    for arch, T in ARCH_T:
        h += f"| {label(arch,T)} | {fmt_pm(g(agg,'untrained',1,arch,T,20))} | {fmt_pm(g(agg,'trained',1,arch,T,20))} |\n"
    return h.rstrip()


def table_kpos(agg):
    h = ("| arch / T | $\\lambda$ @ $k_{pos}{=}1$ | $\\lambda$ @ $k_{pos}{=}2$ | eAUC @1 | eAUC @2 |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt(g(agg,'trained',1,arch,T,20))} | {fmt(g(agg,'trained',2,arch,T,20))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'eauc'))} | {fmt(g(agg,'trained',2,arch,T,20,'eauc'))} |\n")
    return h.rstrip()


def headline_block(agg, pt):
    pt_tok = np.nanmean([g(agg, "trained", 1, a, 1, 20)[0] for a in ("batchtopk_sae", "tsae")])
    pre_t2 = g(agg, "trained", 1, "txc_batchtopk_pre", 2, 20)[0]
    pre_t4 = g(agg, "trained", 1, "txc_batchtopk_pre", 4, 20)[0]
    post_t4 = g(agg, "trained", 1, "txc_batchtopk_post", 4, 20)[0]
    stk_t4 = g(agg, "trained", 1, "stacked_batchtopk", 4, 20)[0]
    win_scarce = g(agg, "trained", 1, "txc_batchtopk_pre", 4, 10)[0]
    spec_t4 = g(agg, "trained", 1, "spectral_txc", 4, 20)[0]
    best_win_t4 = np.nanmax([pre_t4, post_t4, stk_t4, spec_t4])
    un_win = g(agg, "untrained", 1, "txc_batchtopk_pre", 4, 20)[0]
    return (
        f"- **Fair backbone:** every arch shares the BatchTopK→JumpReLU backbone "
        f"(Bussmann et al.) + AuxK + decoder unit-norm, on equal tokens/step — so the "
        f"only variable is decode structure.\n"
        f"- **Per-token DPI floor** (provable, from the generator): "
        f"$\\sqrt{{Var\\,\\lambda/Var\\,b}}$ = **{pt:.2f}**. Trained per-token (BatchTopK) SAEs land at "
        f"**{pt_tok:.2f}** at d_sae=20, flat across all capacities.\n"
        f"- **Window recovery** at d_sae=20: TXC-pre $\\lambda$ = **{pre_t2:.2f}** (T=2) → "
        f"**{pre_t4:.2f}** (T≥4); TXC-post **{post_t4:.2f}**; Stacked **{stk_t4:.2f}**; "
        f"Spectral **{spec_t4:.2f}** (T=4). "
        f"Holds at d_sae=10 < F=20 (TXC-pre = **{win_scarce:.2f}**, scarce regime).\n"
        f"- **Gap** (best window T4 − per-token): **{best_win_t4 - pt_tok:.2f}**. "
        f"Untrained window already reaches {un_win:.2f} (architectural access); training lifts it to {best_win_t4:.2f}."
    )


def main():
    plt = figs.use_agg_style()
    FIG_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

    rows = load_rows(LEADERBOARD, DS, PROTOCOL)
    agg = aggregate(rows, KEY_FIELDS)
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
    populate(RECORD, blocks)

    base = {"source": "results/leaderboard.jsonl", "n_cells": len(rows), "F": F,
            "per_token_dpi_floor": pt}
    record.write_stats(STATS_OUT, base, agg,
                       lambda k: f"{k[0]}|kpos{k[1]}|{k[2]}|T{k[3]}|d{k[4]}", ROOT)


if __name__ == "__main__":
    main()
