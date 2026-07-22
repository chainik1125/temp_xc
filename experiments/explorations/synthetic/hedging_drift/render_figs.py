"""Hedging-drift architecture results — thin driver over the shared record
pipeline.

Config + bench-specific figures/tables only; the leaderboard→aggregate→AUTO-
block→stats plumbing lives in :mod:`explorations.synthetic` (record/figs).
Reads the canonical leaderboard (`results/leaderboard.jsonl`), filters the
`toy_hedging_drift_d64` cells (protocol 1.3.0, non-smoke, n_steps ∈
{0, 30000}), aggregates over seeds, then renders figures + fills every
`<!-- AUTO:* -->` block in `bench_record.md` + writes
`results/hedging_bench_stats.json`. Ceilings come from the committed gating
stats.

    .venv/bin/python -m experiments.explorations.synthetic.hedging_drift.render_figs
"""

from __future__ import annotations

import json
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
GATING = RES_DIR / "hedging_gating_stats.json"
STATS_OUT = RES_DIR / "hedging_bench_stats.json"
RECORD = HERE / "bench_record.md"
DS = "toy_hedging_drift_d64"
PROTOCOL = "1.3.0"
N_STEPS_GRID = 30_000
KEY_FIELDS = ("kind", "k_pos", "arch", "T", "d_sae")

F = 20
D_SAES = [10, 20, 40]   # {F//2, F, 2F} — uniform clean-room design
ARCH_T = [("batchtopk_sae", 1), ("tsae", 1),
          ("txc_batchtopk_pre", 2), ("txc_batchtopk_pre", 4), ("txc_batchtopk_pre", 8),
          ("txc_batchtopk_post", 2), ("txc_batchtopk_post", 4), ("txc_batchtopk_post", 8),
          ("stacked_batchtopk", 2), ("stacked_batchtopk", 4), ("stacked_batchtopk", 8),
          ("spectral_txc", 2), ("spectral_txc", 4), ("spectral_txc", 8)]
PER_TOKEN = {("batchtopk_sae", 1), ("tsae", 1)}
LABEL = {"batchtopk_sae": "BatchTopK-SAE", "tsae": "T-SAE",
         "txc_batchtopk_pre": "TXC-pre", "txc_batchtopk_post": "TXC-post",
         "stacked_batchtopk": "Stacked-SAE", "spectral_txc": "Spectral-TXC"}
WINDOW_FAMILIES = [("txc_batchtopk_pre", "#3182bd"),
                   ("txc_batchtopk_post", "#807dba"),
                   ("stacked_batchtopk", "#31a354"),
                   ("spectral_txc", "#f16913")]
COLORS = {
    ("batchtopk_sae", 1): "#D55E00", ("tsae", 1): "#E69F00",
    ("txc_batchtopk_pre", 2): "#9ecae1", ("txc_batchtopk_pre", 4): "#3182bd", ("txc_batchtopk_pre", 8): "#08519c",
    ("txc_batchtopk_post", 2): "#bcbddc", ("txc_batchtopk_post", 4): "#807dba", ("txc_batchtopk_post", 8): "#54278f",
    ("stacked_batchtopk", 2): "#a1d99b", ("stacked_batchtopk", 4): "#31a354", ("stacked_batchtopk", 8): "#006d2c",
    ("spectral_txc", 2): "#fdae6b", ("spectral_txc", 4): "#f16913", ("spectral_txc", 8): "#a63603",
}


def label(arch, T):
    return f"{LABEL[arch]} (per-token)" if (arch, T) in PER_TOKEN else f"{LABEL[arch]} (T={T})"


def gating_ceilings():
    """Raw-linear access ceilings (committed gating stats)."""
    gs = json.loads(GATING.read_text())
    return {
        "r2_pt_raw": gs["per_token"]["r2_on_x"],
        "r2_win_raw": {int(T): w["r2_raw_linear"] for T, w in gs["window"].items()},
        "acf1": gs["mirror"]["pooled_acf"]["1"],
        "acf4": gs["mirror"]["pooled_acf"]["4"],
    }


def g(agg, kind, kpos, arch, T, d, metric="conf_recovery"):
    return record.get(agg, (kind, kpos, arch, T, d), metric)


# ── paper-quality figures ─────────────────────────────────────────────

def fig_main(agg, ceil, plt):
    """Headline: the confidence-state frontier against the access ceilings."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.axvspan(9, F, color="0.93", zorder=0, lw=0)
    ax.axvline(F, color="0.45", ls=":", lw=1.1, zorder=1)
    ax.text(F - 0.5, -0.01, "F", color="0.4", fontsize=10, ha="right", style="italic")
    ax.axhline(ceil["r2_pt_raw"], color="#d62728", ls="--", lw=1.0, zorder=1)
    ax.text(41.5, ceil["r2_pt_raw"] + 0.008, "raw per-token ceiling", color="#d62728",
            fontsize=7.2, ha="right", va="bottom")
    ax.axhline(ceil["r2_win_raw"][8], color="#3182bd", ls="--", lw=1.0, zorder=1)
    ax.text(41.5, ceil["r2_win_raw"][8] - 0.032, "raw window ceiling (T=8)",
            color="#3182bd", fontsize=7.2, ha="right", va="bottom")
    ax.axhline(1.0, color="#cccccc", ls=":", lw=1.0)
    ax.text(41.5, 0.975, "spec oracle (unreachable)", color="#999",
            fontsize=7.5, ha="right", va="top")
    frontier_series(ax, ARCH_T, D_SAES,
                    lambda a, T, d: g(agg, "trained", 1, a, T, d, "conf_recovery"),
                    COLORS, PER_TOKEN, label, ms=5.5, lw=1.9, capsize=2, elinewidth=1)
    ax.set_xticks(D_SAES); ax.set_xlim(9, 43); ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("dictionary size  $d_{sae}$")
    ax.set_ylabel("confidence recovery (held-out $R^2$)")
    ax.set_title("DC latent: confidence state $c_i$ — recovery vs capacity",
                 loc="left", fontsize=11.5)
    ax.legend(ncol=2, fontsize=7.4, loc="lower right")
    fig.tight_layout()
    save_fig(fig, FIG_DIR, "hedging_main", plt)


def fig_T(agg, ceil, plt):
    """Confidence recovery vs window size T (the frozen § 5 axis)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    raw = [(1, ceil["r2_pt_raw"])] + [(T, ceil["r2_win_raw"][T]) for T in (2, 4, 8)]
    ax.plot([c[0] for c in raw], [c[1] for c in raw], ls=":", color="0.55",
            lw=1.4, marker="_", ms=11, zorder=1, label="raw-linear access ceiling")
    for arch, T in ARCH_T:
        if (arch, T) in PER_TOKEN:
            m, s, n = g(agg, "trained", 1, arch, T, 20, "conf_recovery")
            if n:
                ax.errorbar([1], [m], yerr=[s], marker=MARK[1], ms=8,
                            color=COLORS[(arch, T)], capsize=3, label=label(arch, T))
    for fam, col in WINDOW_FAMILIES:
        ts, ys, es = [], [], []
        for T in (2, 4, 8):
            m, s, n = g(agg, "trained", 1, fam, T, 20, "conf_recovery")
            if n:
                ts.append(T); ys.append(m); es.append(s)
        if ts:
            ax.errorbar(ts, ys, yerr=es, marker="o", ms=6, lw=2, color=col,
                        capsize=3, label=f"{LABEL[fam]} (window)")
    ax.axhline(0.0, color="#999999", ls="--", lw=1.1)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
    ax.set_xlim(0.85, 9.5); ax.set_ylim(-0.08, 1.0)
    ax.set_xlabel("window size  $T$   ($T{=}1$: per-token)")
    ax.set_ylabel("confidence recovery (held-out $R^2$)")
    ax.set_title("Confidence recovery vs window size ($d_{sae}{=}20$, $k_{pos}{=}1$)",
                 loc="left", fontsize=11)
    ax.legend(fontsize=7.6, loc="lower left")
    fig.tight_layout()
    save_fig(fig, FIG_DIR, "hedging_T", plt)


def fig_untrained(agg, plt):
    """Access vs learning (d_sae=20, k_pos=1)."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels, un, uns, tr, trs, cols = [], [], [], [], [], []
    for arch, T in ARCH_T:
        u = g(agg, "untrained", 1, arch, T, 20, "conf_recovery")
        t = g(agg, "trained", 1, arch, T, 20, "conf_recovery")
        labels.append(label(arch, T).replace(" (", "\n(")); cols.append(COLORS[(arch, T)])
        un.append(u[0]); uns.append(u[1]); tr.append(t[0]); trs.append(t[1])
    x = np.arange(len(labels)); w = 0.4
    ax.bar(x - w / 2, un, w, yerr=uns, capsize=2, color="#cfcfcf", edgecolor="0.4",
           lw=0.5, label="untrained (architectural access)")
    ax.bar(x + w / 2, tr, w, yerr=trs, capsize=2, color=cols, edgecolor="0.25",
           lw=0.5, label="trained (access + learning)")
    ax.axhline(0.0, color="#999999", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.0); ax.set_ylim(-0.1, 1.0)
    ax.set_ylabel("confidence recovery ($R^2$)")
    ax.set_title("Access vs learning: random-init vs trained encoders  "
                 "($d_{sae}{=}20$, $k_{pos}{=}1$)", fontsize=11)
    ax.grid(axis="x", alpha=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, FIG_DIR, "hedging_untrained_control", plt)


def fig_local_tradeoff(agg, plt):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, metric, ttl, yl in [
        (axes[0], "gauc", "(a) Confidence-direction recovery", "gAUC (1 conf dir)"),
        (axes[1], "eauc", "(b) Content recovery", "eAUC (19 content dirs)"),
        (axes[2], "nmse", "(c) Reconstruction error", "NMSE"),
    ]:
        ax.axvspan(9, F, color="0.93", zorder=0, lw=0); ax.axvline(F, color="0.45", ls=":", lw=1.1)
        frontier_series(ax, ARCH_T, D_SAES,
                        lambda a, T, d, _m=metric: g(agg, "trained", 1, a, T, d, _m),
                        COLORS, PER_TOKEN, label, ms=5, lw=1.7, capsize=2, elinewidth=0.9)
        ax.set_xticks(D_SAES); ax.set_xlim(9, 42); ax.set_xlabel("dictionary size  $d_{sae}$")
        ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=11)
    axes[0].set_ylim(0, 1.02); axes[1].set_ylim(0, 1.02)
    axes[0].legend(ncol=2, fontsize=7.2, loc="lower right")
    fig.tight_layout()
    save_fig(fig, FIG_DIR, "hedging_local_tradeoff", plt)


# ── tables + headline for bench_record.md ─────────────────────────────

def _frontier_table(agg, metric):
    return record.frontier_table(
        ARCH_T, D_SAES, lambda a, T, d: g(agg, "trained", 1, a, T, d, metric),
        label, bold_pred=lambda a, T: (a, T) not in PER_TOKEN)


def table_untrained(agg):
    h = ("| arch / T | $c_i$ untrained | $c_i$ trained | corr untrained | corr trained |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt_pm(g(agg,'untrained',1,arch,T,20,'conf_recovery'))} "
              f"| {fmt_pm(g(agg,'trained',1,arch,T,20,'conf_recovery'))} "
              f"| {fmt_pm(g(agg,'untrained',1,arch,T,20,'conf_corr'))} "
              f"| {fmt_pm(g(agg,'trained',1,arch,T,20,'conf_corr'))} |\n")
    return h.rstrip()


def table_kpos(agg):
    h = ("| arch / T | $c_i$ @ $k_{pos}{=}1$ | $c_i$ @ $k_{pos}{=}2$ | $c_i$ @ $k_{pos}{=}4$ |\n"
         "|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt(g(agg,'trained',1,arch,T,20,'conf_recovery'))} "
              f"| {fmt(g(agg,'trained',2,arch,T,20,'conf_recovery'))} "
              f"| {fmt(g(agg,'trained',4,arch,T,20,'conf_recovery'))} |\n")
    return h.rstrip()


def table_feature_recovery(agg):
    h = ("| arch / T | gAUC (conf dir) | eAUC (content dirs) | NMSE |\n"
         "|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt(g(agg,'trained',1,arch,T,20,'gauc'))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'eauc'))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'nmse'))} |\n")
    return h.rstrip()


def headline_block(agg, ceil):
    pt = np.nanmean([g(agg, "trained", 1, a, 1, 20, "conf_recovery")[0]
                     for a in ("batchtopk_sae", "tsae")])
    win = {(fam, T): g(agg, "trained", 1, fam, T, 20, "conf_recovery")[0]
           for fam, _ in WINDOW_FAMILIES for T in (2, 4, 8)}
    best = max(win, key=lambda k: np.nan_to_num(win[k], nan=-9))
    t8_by_fam = {fam: win[(fam, 8)] for fam, _ in WINDOW_FAMILIES}
    un_pt = np.nanmean([g(agg, "untrained", 1, a, 1, 20, "conf_recovery")[0]
                        for a in ("batchtopk_sae", "tsae")])
    return (
        f"- **Per-token holds the DC latent:** confidence recovery R² = "
        f"**{pt:.2f}** at d_sae=20, k_pos=1, against a raw per-token access "
        f"ceiling of {ceil['r2_pt_raw']:.2f} (the c·m multiplicative-noise "
        f"bound; the spec oracle R²=1 is unreachable, gating).\n"
        f"- **Windows vs the frozen § 5 prediction (T=8 best / short windows "
        f"lose the drift):** best window cell = {LABEL[best[0]]} T={best[1]} at "
        f"**{np.nan_to_num(win[best], nan=float('nan')):.2f}**; T=8 by family: "
        + ", ".join(f"{LABEL[f]} {np.nan_to_num(v, nan=float('nan')):.2f}"
                    for f, v in t8_by_fam.items())
        + f". The raw temporal-denoising headroom is only "
        f"+{ceil['r2_win_raw'][8] - ceil['r2_pt_raw']:.3f} R² at T=8 (gating) — "
        f"the substrate's persistence (ACF(1) {ceil['acf1']:.2f}, plateau "
        f"{ceil['acf4']:.2f} at lag 4) shares little extra linear information "
        f"across sentences.\n"
        f"- **Access vs learning:** untrained per-token already reads R² = "
        f"{un_pt:.2f} (the dominant continuous loading passes through a random "
        f"encoder); training closes the rest.\n"
        f"- **Substrate:** the C3 hierarchical-AR(1) mirror (per-trace level + "
        f"trend + AR(1); gate-8 PASS on the ACF plateau), F=20 dirs, "
        f"fair-backbone uniform grid, seeds {{1,2,42}}."
    )


def main():
    plt = figs.use_agg_style()
    FIG_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

    rows = load_rows(LEADERBOARD, DS, PROTOCOL, n_steps_keep={0, N_STEPS_GRID})
    agg = aggregate(rows, KEY_FIELDS)
    ceil = gating_ceilings()
    n_trained = sum(1 for r in rows if r["kind"] == "trained")
    print(f"[render] {len(rows)} leaderboard cells ({n_trained} trained); "
          f"raw ceilings pt={ceil['r2_pt_raw']:.3f} T8={ceil['r2_win_raw'][8]:.3f}")

    fig_main(agg, ceil, plt)
    fig_T(agg, ceil, plt)
    fig_untrained(agg, plt)
    fig_local_tradeoff(agg, plt)

    blocks = {
        "headline": headline_block(agg, ceil),
        "conf_frontier": _frontier_table(agg, "conf_recovery"),
        "untrained": table_untrained(agg),
        "kpos": table_kpos(agg),
        "feature_recovery": table_feature_recovery(agg),
    }
    populate(RECORD, blocks)

    base = {"source": "results/leaderboard.jsonl", "n_cells": len(rows), "F": F,
            "gating_ceilings": ceil}
    record.write_stats(STATS_OUT, base, agg,
                       lambda k: f"{k[0]}|kpos{k[1]}|{k[2]}|T{k[3]}|d{k[4]}", ROOT)


if __name__ == "__main__":
    main()
