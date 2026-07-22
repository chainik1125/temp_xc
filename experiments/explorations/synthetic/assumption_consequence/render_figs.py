"""Assumption→consequence architecture results — thin driver over the shared
record pipeline.

Config + bench-specific figures/tables only; the leaderboard→aggregate→AUTO-
block→stats plumbing lives in :mod:`explorations.synthetic` (record/figs).
Reads the canonical leaderboard (`results/leaderboard.jsonl`), filters the
`toy_assumption_consequence_d64` cells (protocol 1.3.0, non-smoke, n_steps ∈
{0, 30000}), aggregates over seeds, then renders figures + fills every
`<!-- AUTO:* -->` block in `bench_record.md` + writes
`results/assumption_bench_stats.json`. Ceilings come from the committed gating
stats.

    .venv/bin/python -m experiments.explorations.synthetic.assumption_consequence.render_figs
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
GATING = RES_DIR / "assumption_gating_stats.json"
STATS_OUT = RES_DIR / "assumption_bench_stats.json"
RECORD = HERE / "bench_record.md"
DS = "toy_assumption_consequence_d64"
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
    """Raw-access readouts + the analytic oracle (committed gating stats).

    ``next_raw_norm``: the raw-linear readouts converted to the evaluator's
    normalization ``(balacc − 1/3)/(oracle_balacc − 1/3)`` — the shared
    "state-revealing readout" line every arch family sits under.
    """
    gs = json.loads(GATING.read_text())
    an, pt = gs["analytic"], gs["per_token"]
    denom = an["oracle_balacc"] - 1 / 3
    norm = lambda b: (b - 1 / 3) / denom
    return {
        "oracle_balacc": an["oracle_balacc"],
        "next_pt_raw_norm": norm(pt["next_balacc_on_x"]),
        "next_win_raw_norm": {int(T): norm(w["next_balacc_raw_linear"])
                              for T, w in gs["window"].items()},
        "fwd_rate": gs["mirror"]["fwd_rate"],
        "asym": gs["mirror"]["asym"],
    }


def g(agg, kind, kpos, arch, T, d, metric="nextstate_recovery"):
    return record.get(agg, (kind, kpos, arch, T, d), metric)


# ── paper-quality figures ─────────────────────────────────────────────

def _frontier_panel(ax, agg, metric, ylabel, title, *, ceil_lines=None,
                    ylim=(-0.05, 1.02)):
    ax.axvspan(9, F, color="0.93", zorder=0, lw=0)
    ax.axvline(F, color="0.45", ls=":", lw=1.1, zorder=1)
    ax.text(F - 0.5, ylim[0] + 0.04 * (ylim[1] - ylim[0]), "F", color="0.4",
            fontsize=10, ha="right", style="italic")
    if ceil_lines:
        for y, lbl, col in ceil_lines:
            ax.axhline(y, color=col, ls="--", lw=0.9, zorder=1)
            ax.text(41.5, y + 0.008, lbl, color=col, fontsize=7.2, ha="right", va="bottom")
    frontier_series(ax, ARCH_T, D_SAES,
                    lambda a, T, d: g(agg, "trained", 1, a, T, d, metric),
                    COLORS, PER_TOKEN, label, ms=5.5, lw=1.9, capsize=2, elinewidth=1)
    ax.set_xticks(D_SAES); ax.set_xlim(9, 43); ax.set_ylim(*ylim)
    ax.set_xlabel("dictionary size  $d_{sae}$"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)


def fig_main(agg, ceil, plt):
    """Headline: the DC state frontier + the AC-directed next-state frontier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    _frontier_panel(ax1, agg, "state_recovery",
                    "state recovery (norm. balanced acc.)",
                    "(a) DC latent: discourse state $s_i$")
    ax1.axhline(1.0, color="#cccccc", ls=":", lw=1.0)
    ax1.text(41.5, 0.975, "oracle", color="#999", fontsize=7.5, ha="right", va="top")
    raw_lines = [(ceil["next_pt_raw_norm"], "raw per-token readout", "#d62728")]
    _frontier_panel(ax2, agg, "nextstate_recovery",
                    "next-state recovery (norm. balanced acc.)",
                    "(b) AC-directed latent: next state $s_{i+1}$",
                    ceil_lines=raw_lines)
    ax2.axhline(0.0, color="#999999", ls="--", lw=1.1)
    ax2.text(41.5, 0.012, "chance (marginal)", color="#777",
             fontsize=7.5, ha="right", va="bottom")
    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=8.3,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, FIG_DIR, "assumption_main", plt)


def fig_T(agg, ceil, plt):
    """Both latents vs window size T, against the shared raw-access line."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, metric, yl, ttl, raw in [
        (ax1, "state_recovery", "state recovery (norm. balanced acc.)",
         "(a) DC: state $s_i$ vs window size", None),
        (ax2, "nextstate_recovery", "next-state recovery (norm. balanced acc.)",
         "(b) AC-directed: next state $s_{i+1}$",
         [(1, ceil["next_pt_raw_norm"])]
         + [(T, ceil["next_win_raw_norm"][T]) for T in (2, 4, 8)]),
    ]:
        ax.axhline(0.0, color="#999999", ls="--", lw=1.1)
        if raw:
            ax.plot([c[0] for c in raw], [c[1] for c in raw], ls=":",
                    color="0.55", lw=1.4, marker="_", ms=11, zorder=1,
                    label="raw-linear readout (≈ order-1 sufficiency)")
        for arch, T in ARCH_T:
            if (arch, T) in PER_TOKEN:
                m, s, n = g(agg, "trained", 1, arch, T, 20, metric)
                if n:
                    ax.errorbar([1], [m], yerr=[s], marker=MARK[1], ms=8,
                                color=COLORS[(arch, T)], capsize=3, label=label(arch, T))
        for fam, col in WINDOW_FAMILIES:
            ts, ys, es = [], [], []
            for T in (2, 4, 8):
                m, s, n = g(agg, "trained", 1, fam, T, 20, metric)
                if n:
                    ts.append(T); ys.append(m); es.append(s)
            if ts:
                ax.errorbar(ts, ys, yerr=es, marker="o", ms=6, lw=2, color=col,
                            capsize=3, label=f"{LABEL[fam]} (window)")
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
        ax.set_xlim(0.85, 9.5); ax.set_ylim(-0.08, 1.04)
        ax.set_xlabel("window size  $T$   ($T{=}1$: per-token)"); ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left", fontsize=11)
    ax2.legend(loc="lower left", fontsize=7.4)
    fig.tight_layout()
    save_fig(fig, FIG_DIR, "assumption_T", plt)


def fig_untrained(agg, plt):
    """Access vs learning, on BOTH latents (d_sae=20, k_pos=1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, metric, yl, ttl in [
        (axes[0], "state_recovery", "state recovery (norm.)", "(a) DC: state $s_i$"),
        (axes[1], "nextstate_recovery", "next-state recovery (norm.)",
         "(b) AC-directed: next state $s_{i+1}$"),
    ]:
        labels, un, uns, tr, trs, cols = [], [], [], [], [], []
        for arch, T in ARCH_T:
            u = g(agg, "untrained", 1, arch, T, 20, metric)
            t = g(agg, "trained", 1, arch, T, 20, metric)
            labels.append(label(arch, T).replace(" (", "\n(")); cols.append(COLORS[(arch, T)])
            un.append(u[0]); uns.append(u[1]); tr.append(t[0]); trs.append(t[1])
        x = np.arange(len(labels)); w = 0.4
        ax.bar(x - w / 2, un, w, yerr=uns, capsize=2, color="#cfcfcf", edgecolor="0.4",
               lw=0.5, label="untrained (architectural access)")
        ax.bar(x + w / 2, tr, w, yerr=trs, capsize=2, color=cols, edgecolor="0.25",
               lw=0.5, label="trained (access + learning)")
        ax.axhline(0.0, color="#999999", ls="--", lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.0); ax.set_ylim(-0.1, 1.04)
        ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Access vs learning: random-init vs trained encoders  ($d_{sae}{=}20$, $k_{pos}{=}1$)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, FIG_DIR, "assumption_untrained_control", plt)


def fig_local_tradeoff(agg, plt):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, metric, ttl, yl in [
        (axes[0], "gauc", "(a) State-direction recovery", "gAUC (3 state dirs)"),
        (axes[1], "eauc", "(b) Content recovery", "eAUC (17 content dirs)"),
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
    save_fig(fig, FIG_DIR, "assumption_local_tradeoff", plt)


# ── tables + headline for bench_record.md ─────────────────────────────

def _frontier_table(agg, metric):
    return record.frontier_table(
        ARCH_T, D_SAES, lambda a, T, d: g(agg, "trained", 1, a, T, d, metric),
        label, bold_pred=lambda a, T: (a, T) not in PER_TOKEN)


def table_untrained(agg):
    h = ("| arch / T | state untrained | state trained | next untrained | next trained |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt_pm(g(agg,'untrained',1,arch,T,20,'state_recovery'))} "
              f"| {fmt_pm(g(agg,'trained',1,arch,T,20,'state_recovery'))} "
              f"| {fmt_pm(g(agg,'untrained',1,arch,T,20,'nextstate_recovery'))} "
              f"| {fmt_pm(g(agg,'trained',1,arch,T,20,'nextstate_recovery'))} |\n")
    return h.rstrip()


def table_kpos(agg):
    h = ("| arch / T | state @ $k_{pos}{=}1$ | state @ $k_{pos}{=}2$ | next @ $k_{pos}{=}1$ | next @ $k_{pos}{=}2$ |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt(g(agg,'trained',1,arch,T,20,'state_recovery'))} "
              f"| {fmt(g(agg,'trained',2,arch,T,20,'state_recovery'))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'nextstate_recovery'))} "
              f"| {fmt(g(agg,'trained',2,arch,T,20,'nextstate_recovery'))} |\n")
    return h.rstrip()


def table_feature_recovery(agg):
    h = ("| arch / T | gAUC (state dirs) | eAUC (content dirs) | NMSE |\n"
         "|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt(g(agg,'trained',1,arch,T,20,'gauc'))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'eauc'))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'nmse'))} |\n")
    return h.rstrip()


def headline_block(agg, ceil):
    pt_state = np.nanmean([g(agg, "trained", 1, a, 1, 20, "state_recovery")[0]
                           for a in ("batchtopk_sae", "tsae")])
    pt_next = np.nanmean([g(agg, "trained", 1, a, 1, 20, "nextstate_recovery")[0]
                          for a in ("batchtopk_sae", "tsae")])
    win_next = {(fam, T): g(agg, "trained", 1, fam, T, 20, "nextstate_recovery")[0]
                for fam, _ in WINDOW_FAMILIES for T in (2, 4, 8)}
    best_win = max(win_next, key=lambda k: np.nan_to_num(win_next[k], nan=-9))
    win_state_min = min(np.nan_to_num(
        g(agg, "trained", 1, fam, T, 20, "state_recovery")[0], nan=9)
        for fam, _ in WINDOW_FAMILIES for T in (2, 4, 8))
    un_pt_next = np.nanmean([g(agg, "untrained", 1, a, 1, 20, "nextstate_recovery")[0]
                             for a in ("batchtopk_sae", "tsae")])
    return (
        f"- **The frozen § 5 prediction FAILS, informatively — per-token is NOT "
        f"blind to the directed dependency:** per-token next-state recovery is "
        f"**{pt_next:.2f}** (normalized; d_sae=20, k_pos=1) vs the raw-readout line "
        f"{ceil['next_pt_raw_norm']:.2f} — the order-1 mirror makes $s_i$ "
        f"sufficient (gating), so any state-revealing code supports the one-step "
        f"conditional. The best window cell ({LABEL[best_win[0]]} T={best_win[1]}) "
        f"reaches **{win_next[best_win]:.2f}**: no window family beats per-token "
        f"beyond noise anywhere on the frontier.\n"
        f"- **DC state:** per-token **{pt_state:.2f}** at d_sae=20; window families "
        f"pay the usual shared-code price (min {win_state_min:.2f} at d=20 across "
        f"(family, T)).\n"
        f"- **Access vs learning:** untrained per-token already reads "
        f"{un_pt_next:.2f} of the directed latent (the dominant state direction "
        f"passes through a random encoder); training closes the rest.\n"
        f"- **Substrate:** the g7 strict-labeler Markov mirror (fwd P(C|A) = "
        f"{ceil['fwd_rate']:.3f}, directed asym {ceil['asym']:.3f}), F=20 dirs, "
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
          f"raw next-state line {ceil['next_pt_raw_norm']:.3f}")

    fig_main(agg, ceil, plt)
    fig_T(agg, ceil, plt)
    fig_untrained(agg, plt)
    fig_local_tradeoff(agg, plt)

    blocks = {
        "headline": headline_block(agg, ceil),
        "state_frontier": _frontier_table(agg, "state_recovery"),
        "nextstate_frontier": _frontier_table(agg, "nextstate_recovery"),
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
