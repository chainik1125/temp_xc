"""Recipe-instruction phase-runs architecture results — thin driver over the
shared record pipeline (stage-6 #3b, the re-scoped residual axis).

Config + bench-specific figures/tables only; the leaderboard→aggregate→AUTO-
block→stats plumbing lives in :mod:`explorations.synthetic` (record/figs).
Reads the canonical leaderboard (`results/leaderboard.jsonl`), filters the
`toy_recipe_instruction_d64` cells (protocol 1.3.0, non-smoke, n_steps ∈
{0, 30000}), aggregates over seeds, then renders figures + fills every
`<!-- AUTO:* -->` block in `bench_record.md` + writes
`results/recipe_bench_stats.json`. Reference lines come from the committed
§ 8 gating stats (`b463c4a0`) converted to the residual normalization
``resid(b) = (b − 0.771) / 0.229`` — the additive ceiling maps to 0, exact
to 1, and every raw-linear access line lands NEGATIVE (that is the point of
the re-scope).

    .venv/bin/python -m experiments.explorations.synthetic.recipe_instruction_phase_runs.render_figs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explorations.synthetic import figs, record
from explorations.synthetic.figs import MARK, frontier_series, save_fig
from explorations.synthetic.record import aggregate, fmt, fmt_pm, load_rows, populate
from temp_bench.evals.recipe_recovery import EQ_ADDITIVE_CEILING

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
FIG_DIR = HERE / "figs"
RES_DIR = HERE / "results"
GATING = RES_DIR / "recipe_gating_stats.json"
STATS_OUT = RES_DIR / "recipe_bench_stats.json"
RECORD = HERE / "bench_record.md"
DS = "toy_recipe_instruction_d64"
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
ADDITIVE_FAMILIES = ("batchtopk_sae", "tsae", "stacked_batchtopk", "txc_batchtopk_pre")
MIXING_FAMILIES = ("txc_batchtopk_post", "spectral_txc")
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
RESID_YLIM = (-1.35, 1.05)


def label(arch, T):
    return f"{LABEL[arch]} (per-token)" if (arch, T) in PER_TOKEN else f"{LABEL[arch]} (T={T})"


def resid(balacc):
    return (balacc - EQ_ADDITIVE_CEILING) / (1.0 - EQ_ADDITIVE_CEILING)


def gating_ceilings():
    """§ 8 access lines (threshold-optimized, noiseless x) in residual units."""
    gs = json.loads(GATING.read_text())
    nl = gs["noiseless"]
    return {
        "pt_raw_resid": resid(nl["per_token"]["e_balacc_ceiling"]),
        "win_raw_resid": {int(T): resid(w["e_balacc_ceiling"])
                          for T, w in nl["window"].items()},
        "from_ct_resid": resid(gs["analytic"]["e_balacc_from_c_t"]),
        "pair_additive_balacc": gs["analytic"]["e_balacc_pair_additive"],
        "mlp_balacc": nl["window"]["2"]["e_balacc_mlp_ceiling"],
    }


def g(agg, kind, kpos, arch, T, d, metric="equality_residual_recovery"):
    return record.get(agg, (kind, kpos, arch, T, d), metric)


# ── paper-quality figures ─────────────────────────────────────────────

def _frontier_panel(ax, agg, metric, ylabel, title, *, ylim=(-0.05, 1.02)):
    ax.axvspan(9, F, color="0.93", zorder=0, lw=0)
    ax.axvline(F, color="0.45", ls=":", lw=1.1, zorder=1)
    ax.text(F - 0.5, ylim[0] + 0.04 * (ylim[1] - ylim[0]), "F", color="0.4",
            fontsize=10, ha="right", style="italic")
    frontier_series(ax, ARCH_T, D_SAES,
                    lambda a, T, d: g(agg, "trained", 1, a, T, d, metric),
                    COLORS, PER_TOKEN, label, ms=5.5, lw=1.9, capsize=2, elinewidth=1)
    ax.set_xticks(D_SAES); ax.set_xlim(9, 43); ax.set_ylim(*ylim)
    ax.set_xlabel("dictionary size  $d_{sae}$"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)


def _resid_reference_lines(ax, ceil):
    ax.axhline(0.0, color="#2c7bb6", ls="--", lw=1.2, zorder=1)
    ax.text(41.5, 0.015, "additive ceiling (0.771 balacc)", color="#2c7bb6",
            fontsize=7.2, ha="right", va="bottom")
    ax.axhline(1.0, color="#cccccc", ls=":", lw=1.0, zorder=1)
    ax.text(41.5, 0.975, "exact pair rule", color="#999", fontsize=7.2,
            ha="right", va="top")
    ax.axhline(ceil["pt_raw_resid"], color="#d62728", ls="--", lw=0.9, zorder=1)
    ax.text(41.5, ceil["pt_raw_resid"] + 0.015, "raw per-token access (§ 8)",
            color="#d62728", fontsize=7.2, ha="right", va="bottom")
    y_w = ceil["win_raw_resid"][2]
    ax.axhline(y_w, color="#7b3294", ls="--", lw=0.9, zorder=1)
    ax.text(41.5, y_w + 0.015, "raw window access, T=2 (§ 8)",
            color="#7b3294", fontsize=7.2, ha="right", va="bottom")


def fig_main(agg, ceil, plt):
    """Headline: the DC phase control + the regime-3 residual frontier."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    _frontier_panel(ax1, agg, "phase_recovery",
                    "phase recovery (norm. balanced acc.)",
                    "(a) DC control: phase class $c_t$")
    ax1.axhline(1.0, color="#cccccc", ls=":", lw=1.0)
    ax1.text(41.5, 0.975, "oracle", color="#999", fontsize=7.5, ha="right", va="top")
    _frontier_panel(ax2, agg, "equality_residual_recovery",
                    "equality residual (over [0.771, 1] balacc)",
                    "(b) PRIMARY: regime-3 residual of $e_t=[c_t{=}c_{t-1}]$",
                    ylim=RESID_YLIM)
    _resid_reference_lines(ax2, ceil)
    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=8.3,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_fig(fig, FIG_DIR, "recipe_main", plt)


def fig_T(agg, ceil, plt):
    """Both axes vs window size T, with the § 8 raw-access lines per T."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    panels = [
        (ax1, "phase_recovery", "phase recovery (norm. balanced acc.)",
         "(a) DC control: phase $c_t$ vs window size", None, (-0.08, 1.04)),
        (ax2, "equality_residual_recovery",
         "equality residual (over [0.771, 1] balacc)",
         "(b) PRIMARY: residual vs window size",
         [(1, ceil["pt_raw_resid"])]
         + [(T, ceil["win_raw_resid"][T]) for T in (2, 4, 8)], RESID_YLIM),
    ]
    for ax, metric, yl, ttl, raw, ylim in panels:
        ax.axhline(0.0, color="#2c7bb6", ls="--", lw=1.1)
        if raw:
            ax.plot([c[0] for c in raw], [c[1] for c in raw], ls=":",
                    color="0.55", lw=1.4, marker="_", ms=11, zorder=1,
                    label="raw-linear access (§ 8 ceiling)")
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
        ax.set_xlim(0.85, 9.5); ax.set_ylim(*ylim)
        ax.set_xlabel("window size  $T$   ($T{=}1$: per-token)"); ax.set_ylabel(yl)
        ax.set_title(ttl, loc="left", fontsize=11)
    ax2.legend(loc="lower right", fontsize=7.4)
    fig.tight_layout()
    save_fig(fig, FIG_DIR, "recipe_T", plt)


def fig_untrained(agg, plt):
    """Access vs learning, on BOTH axes (d_sae=20, k_pos=1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, metric, yl, ttl, ylim in [
        (axes[0], "phase_recovery", "phase recovery (norm.)",
         "(a) DC control: phase $c_t$", (-0.1, 1.04)),
        (axes[1], "equality_residual_recovery", "equality residual",
         "(b) PRIMARY: regime-3 residual", RESID_YLIM),
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
        ax.axhline(0.0, color="#2c7bb6" if "residual" in metric else "#999999",
                   ls="--", lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.0); ax.set_ylim(*ylim)
        ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Access vs learning: random-init vs trained encoders  ($d_{sae}{=}20$, $k_{pos}{=}1$)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, FIG_DIR, "recipe_untrained_control", plt)


def fig_local_tradeoff(agg, plt):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, metric, ttl, yl in [
        (axes[0], "gauc", "(a) Phase-direction recovery", "gAUC (5 phase dirs)"),
        (axes[1], "eauc", "(b) Content recovery", "eAUC (15 content dirs)"),
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
    save_fig(fig, FIG_DIR, "recipe_local_tradeoff", plt)


# ── tables + headline for bench_record.md ─────────────────────────────

def _frontier_table(agg, metric):
    return record.frontier_table(
        ARCH_T, D_SAES, lambda a, T, d: g(agg, "trained", 1, a, T, d, metric),
        label, bold_pred=lambda a, T: a in MIXING_FAMILIES)


def table_untrained(agg):
    h = ("| arch / T | phase untrained | phase trained | residual untrained | residual trained |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt_pm(g(agg,'untrained',1,arch,T,20,'phase_recovery'))} "
              f"| {fmt_pm(g(agg,'trained',1,arch,T,20,'phase_recovery'))} "
              f"| {fmt_pm(g(agg,'untrained',1,arch,T,20,'equality_residual_recovery'))} "
              f"| {fmt_pm(g(agg,'trained',1,arch,T,20,'equality_residual_recovery'))} |\n")
    return h.rstrip()


def table_kpos(agg):
    h = ("| arch / T | resid @ $k_{pos}{=}1$ | @ 2 | @ 4 | @ 8 | @ 16 |\n"
         "|---|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        cells = " | ".join(fmt(g(agg, "trained", kp, arch, T, 20)) for kp in (1, 2, 4, 8, 16))
        h += f"| {label(arch,T)} | {cells} |\n"
    return h.rstrip()


def table_feature_recovery(agg):
    h = ("| arch / T | gAUC (phase dirs) | eAUC (content dirs) | NMSE |\n"
         "|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {fmt(g(agg,'trained',1,arch,T,20,'gauc'))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'eauc'))} "
              f"| {fmt(g(agg,'trained',1,arch,T,20,'nmse'))} |\n")
    return h.rstrip()


def _best_over_kpos(agg, arch, T, d, metric="equality_residual_recovery"):
    vals = [(g(agg, "trained", kp, arch, T, d, metric), kp) for kp in (1, 2, 4, 8, 16)]
    vals = [((m, s, n), kp) for (m, s, n), kp in vals if n]
    if not vals:
        return (float("nan"), float("nan"), 0), None
    (m, s, n), kp = max(vals, key=lambda v: v[0][0])
    return (m, s, n), kp


def headline_block(agg, ceil):
    add_best = {}
    for fam in ADDITIVE_FAMILIES:
        for T in ((1,) if fam in ("batchtopk_sae", "tsae") else (2, 4, 8)):
            for d in D_SAES:
                (m, s, n), kp = _best_over_kpos(agg, fam, T, d)
                if n and (not add_best or m > add_best["m"]):
                    add_best = {"m": m, "s": s, "fam": fam, "T": T, "d": d, "kp": kp}
    mix_best = {}
    for fam in MIXING_FAMILIES:
        for T in (2, 4, 8):
            for d in D_SAES:
                (m, s, n), kp = _best_over_kpos(agg, fam, T, d)
                if n and (not mix_best or m > mix_best["m"]):
                    mix_best = {"m": m, "s": s, "fam": fam, "T": T, "d": d, "kp": kp}
    pt_resid = np.nanmean([g(agg, "trained", 1, a, 1, 20)[0]
                           for a in ("batchtopk_sae", "tsae")])
    pt_phase = np.nanmean([g(agg, "trained", 1, a, 1, 20, "phase_recovery")[0]
                           for a in ("batchtopk_sae", "tsae")])
    un_mix = (g(agg, "untrained", 1, mix_best["fam"], mix_best["T"], 20)[0]
              if mix_best else float("nan"))
    return (
        f"- **Additive families** (per-token, Stacked, TXC-pre): best residual "
        f"anywhere on the frontier **{add_best['m']:+.2f}** "
        f"({LABEL[add_best['fam']]} T={add_best['T']}, d={add_best['d']}, "
        f"k={add_best['kp']}); per-token mean at d=20,k=1 **{pt_resid:+.2f}** vs "
        f"the § 8 raw per-token access line {ceil['pt_raw_resid']:+.2f}.\n"
        f"- **Position-mixing families** (TXC-post, Spectral): best residual "
        f"**{mix_best['m']:+.2f} ± {mix_best['s']:.2f}** "
        f"({LABEL[mix_best['fam']]} T={mix_best['T']}, d={mix_best['d']}, "
        f"k={mix_best['kp']}); its untrained control at d=20,k=1 sits at "
        f"{un_mix:+.2f}.\n"
        f"- **DC control:** per-token phase recovery {pt_phase:.2f} (≈ oracle, "
        f"as frozen — the control behaves).\n"
        f"- **Reference (§ 8, residual units):** additive ceiling = 0 by "
        f"construction; raw window access T=2 {ceil['win_raw_resid'][2]:+.2f}; "
        f"from-$c_t$ leak {ceil['from_ct_resid']:+.2f}; exact rule +1."
    )


def main():
    plt = figs.use_agg_style()
    FIG_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

    rows = load_rows(LEADERBOARD, DS, PROTOCOL, n_steps_keep={0, N_STEPS_GRID})
    agg = aggregate(rows, KEY_FIELDS)
    ceil = gating_ceilings()
    n_trained = sum(1 for r in rows if r["kind"] == "trained")
    print(f"[render] {len(rows)} leaderboard cells ({n_trained} trained); "
          f"raw access lines (resid units): pt {ceil['pt_raw_resid']:+.3f}, "
          f"T=2 {ceil['win_raw_resid'][2]:+.3f}")

    fig_main(agg, ceil, plt)
    fig_T(agg, ceil, plt)
    fig_untrained(agg, plt)
    fig_local_tradeoff(agg, plt)

    blocks = {
        "headline": headline_block(agg, ceil),
        "phase_frontier": _frontier_table(agg, "phase_recovery"),
        "residual_frontier": _frontier_table(agg, "equality_residual_recovery"),
        "untrained": table_untrained(agg),
        "kpos": table_kpos(agg),
        "feature_recovery": table_feature_recovery(agg),
    }
    populate(RECORD, blocks)

    base = {"source": "results/leaderboard.jsonl", "n_cells": len(rows), "F": F,
            "eq_additive_ceiling": EQ_ADDITIVE_CEILING, "gating_lines": ceil}
    record.write_stats(STATS_OUT, base, agg,
                       lambda k: f"{k[0]}|kpos{k[1]}|{k[2]}|T{k[3]}|d{k[4]}", ROOT)


if __name__ == "__main__":
    main()
