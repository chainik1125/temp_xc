"""Change-point architecture results — SINGLE SOURCE renderer + record populator.

Reads the **canonical leaderboard** (`results/leaderboard.jsonl`) — the one
code-version-stamped source — filters the `toy_changepoint_modes_d64` cells
(protocol 1.2.0, non-smoke, n_steps ∈ {0, 30000}), aggregates over seeds, then
in one pass:

  1. renders paper-quality figures into `figs/`
     (`changepoint_main`, `changepoint_split`, `changepoint_T`,
      `changepoint_untrained_control`, `changepoint_local_tradeoff`),
  2. writes the machine-readable aggregate `results/changepoint_bench_stats.json`,
  3. fills every `<!-- BEGIN AUTO:<tag> --> … <!-- END AUTO:<tag> -->` block in
     `bench_record.md` (headline numbers + all result tables).

Re-running rebuilds the record's numbers, figures, and stats from the canonical
leaderboard — there is no hand-typing and nothing can drift. The τ in-tile info
ceilings come from the committed gating stats (also canonical).

    .venv/bin/python -m experiments.explorations.synthetic.changepoint.render_figs
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
FIG_DIR = HERE / "figs"
RES_DIR = HERE / "results"
GATING = RES_DIR / "changepoint_gating_stats.json"
STATS_OUT = RES_DIR / "changepoint_bench_stats.json"
RECORD = HERE / "bench_record.md"
DS = "toy_changepoint_modes_d64"
PROTOCOL = "1.2.0"
N_STEPS_GRID = 30_000

F = 20
D_SAES = [8, 16, 20, 40]
ARCH_T = [("batchtopk_sae", 1), ("tsae", 1),
          ("txc_batchtopk_pre", 2), ("txc_batchtopk_pre", 4), ("txc_batchtopk_pre", 8),
          ("txc_batchtopk_post", 2), ("txc_batchtopk_post", 4), ("txc_batchtopk_post", 8),
          ("stacked_batchtopk", 2), ("stacked_batchtopk", 4), ("stacked_batchtopk", 8)]
PER_TOKEN = {("batchtopk_sae", 1), ("tsae", 1)}
LABEL = {"batchtopk_sae": "BatchTopK-SAE", "tsae": "T-SAE",
         "txc_batchtopk_pre": "TXC-pre", "txc_batchtopk_post": "TXC-post",
         "stacked_batchtopk": "Stacked-SAE"}
WINDOW_FAMILIES = [("txc_batchtopk_pre", "#3182bd"),
                   ("txc_batchtopk_post", "#807dba"),
                   ("stacked_batchtopk", "#31a354")]
COLORS = {
    ("batchtopk_sae", 1): "#D55E00", ("tsae", 1): "#E69F00",
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


# ── canonical sources: the leaderboard + the gating stats ──────────────

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
        n_steps = int(r.get("training_cfg", {}).get("n_steps", 0))
        if n_steps not in (0, N_STEPS_GRID):       # drop smoke-length runs
            continue
        ov = r.get("training_cfg", {}).get("arch_hparams_override") or {}
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


def gating_ceilings():
    """τ in-tile info ceilings by T + the per-token chance facts (committed gating)."""
    gs = json.loads(GATING.read_text())
    a = gs["anchor"]
    return {
        "tau_info_by_T": {int(T): w["tau_info_ceiling"] for T, w in a["window"].items()},
        "tau_per_token": a["per_token_probes_on_x"]["tau_corr_on_x"],
        "c_per_token_balacc": a["per_token_probes_on_x"]["c_balacc_on_x"],
        "mode_per_token_oracle": a["per_token_probes_on_x"]["mode_balacc_on_x"],
        "base_switch_rate": a["per_token_from_m"]["base_switch_rate"],
    }


def g(agg, kind, kpos, arch, T, d, metric="tss_recovery"):
    c = agg.get((kind, kpos, arch, T, d))
    return c[metric] if c and metric in c else (float("nan"), float("nan"), 0)


# ── paper-quality figures ─────────────────────────────────────────────

def _frontier_panel(ax, agg, metric, ylabel, title, *, ceil_lines=None, ylim=(0, 1.02)):
    ax.axvspan(7, F, color="0.93", zorder=0, lw=0)
    ax.axvline(F, color="0.45", ls=":", lw=1.1, zorder=1)
    ax.text(F - 0.5, ylim[0] + 0.04 * (ylim[1] - ylim[0]), "F", color="0.4",
            fontsize=10, ha="right", style="italic")
    if ceil_lines:
        for y, lbl, col in ceil_lines:
            ax.axhline(y, color=col, ls="--", lw=0.9, zorder=1)
            ax.text(41.5, y + 0.008, lbl, color=col, fontsize=7.2, ha="right", va="bottom")
    for arch, T in ARCH_T:
        xs, ys, es = [], [], []
        for d in D_SAES:
            m, s, n = g(agg, "trained", 1, arch, T, d, metric)
            if n:
                xs.append(d); ys.append(m); es.append(s)
        if xs:
            ls = "--" if (arch, T) in PER_TOKEN else "-"
            ax.errorbar(xs, ys, yerr=es, marker=MARK[T], ms=5.5, lw=1.9, ls=ls,
                        color=COLORS[(arch, T)], capsize=2, elinewidth=1, label=label(arch, T))
    ax.set_xticks(D_SAES); ax.set_xlim(7, 43); ax.set_ylim(*ylim)
    ax.set_xlabel("dictionary size  $d_{sae}$"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)


def fig_main(agg, ceil, plt):
    """THE headline: the DC/AC split on one substrate, two frontiers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    _frontier_panel(ax1, agg, "mode_recovery",
                    "mode recovery (norm. balanced acc.)",
                    "(a) DC latent: mode $m_t$ — per-token home turf")
    ax1.axhline(1.0, color="#cccccc", ls=":", lw=1.0)
    ax1.text(41.5, 0.975, "oracle", color="#999", fontsize=7.5, ha="right", va="top")
    tau_lines = [(ceil["tau_info_by_T"][T], f"info ceiling T={T}",
                  {2: "#9ecae1", 4: "#6baed6", 8: "#3182bd"}[T]) for T in (2, 4, 8)]
    _frontier_panel(ax2, agg, "tss_recovery",
                    "time-since-switch recovery (held-out corr.)",
                    "(b) AC latent: time-since-switch $\\tau_t$",
                    ceil_lines=tau_lines, ylim=(-0.05, 1.02))
    ax2.axhline(0.0, color="#999999", ls="--", lw=1.1)
    ax2.text(41.5, 0.012, "per-token ceiling = chance (DPI)", color="#777",
             fontsize=7.5, ha="right", va="bottom")
    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=8.3,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, plt, "changepoint_main")


def fig_split(agg, plt):
    """The two-axis specialization plane: DC (mode) × AC (τ) per (arch, T)."""
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.axhspan(0.5, 1.04, color="#eef4fb", zorder=0, lw=0)
    ax.text(0.015, 1.02, "boundary (AC) specialist", color="#2b6cb0",
            fontsize=9.5, va="top", style="italic")
    ax.text(0.985, -0.028, "mode (DC) specialist", color="#a84300",
            fontsize=9.5, ha="right", va="bottom", style="italic")
    for arch, T in ARCH_T:
        col = COLORS[(arch, T)]; mk = "X" if (arch, T) in PER_TOKEN else MARK[T]
        pts = []
        for d in D_SAES:
            mx = g(agg, "trained", 1, arch, T, d, "mode_recovery")
            ty = g(agg, "trained", 1, arch, T, d, "tss_recovery")
            if mx[2] and ty[2]:
                pts.append((d, mx[0], ty[0], mx[1], ty[1]))
        if not pts:
            continue
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=col, lw=1.0, alpha=0.3,
                ls="--" if (arch, T) in PER_TOKEN else "-", zorder=2)
        for d, mx, ty, mxs, tys in pts:
            if d == 20:
                ax.scatter([mx], [ty], s=150, color=col, edgecolor="0.2", linewidth=0.8,
                           marker=mk, zorder=4, label=label(arch, T))
                ax.errorbar([mx], [ty], xerr=[mxs], yerr=[tys], color=col, lw=1, capsize=2, zorder=4)
            else:
                ax.scatter([mx], [ty], s=26, color=col, marker=mk, alpha=0.4, zorder=3)
    ax.set_xlim(0, 1.04); ax.set_ylim(-0.05, 1.04)
    ax.set_xlabel("DC latent recovery   (mode, norm. balanced acc.)")
    ax.set_ylabel("AC latent recovery   (time-since-switch, corr.)")
    ax.set_title("The dual-latent split: DC vs AC recovery on one substrate", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.6,
              title="(arch, T) @ $d_{sae}{=}20$\nfaint trail: $d_{sae}\\in\\{8,16,20,40\\}$",
              title_fontsize=8.2)
    fig.tight_layout()
    _save(fig, plt, "changepoint_split")


def fig_T(agg, ceil, plt):
    """AC recovery vs window size T, against the per-T info ceilings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for ax, metric, yl, ttl, ceil_pts in [
        (ax1, "tss_recovery", "time-since-switch recovery (corr.)",
         "(a) $\\tau_t$ recovery vs window size",
         [(T, ceil["tau_info_by_T"][T]) for T in (2, 4, 8)]),
        (ax2, "cp_recovery", "change-point recovery (norm. balanced acc.)",
         "(b) $c_t$ (simple-floor companion)",
         [(T, 1.0) for T in (2, 4, 8)]),
    ]:
        ax.axhline(0.0, color="#999999", ls="--", lw=1.1)
        ax.plot([c[0] for c in ceil_pts], [c[1] for c in ceil_pts], ls=":",
                color="0.55", lw=1.4, marker="_", ms=11, zorder=1, label="in-tile info ceiling")
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
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, plt, "changepoint_T")


def fig_untrained(agg, plt):
    """Access vs learning, on BOTH latents (d_sae=20, k_pos=1)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, metric, yl, ttl in [
        (axes[0], "mode_recovery", "mode recovery (norm.)", "(a) DC: mode $m_t$"),
        (axes[1], "tss_recovery", "$\\tau$ recovery (corr.)", "(b) AC: time-since-switch $\\tau_t$"),
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
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Access vs learning: random-init vs trained encoders  ($d_{sae}{=}20$, $k_{pos}{=}1$)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, plt, "changepoint_untrained_control")


def fig_local_tradeoff(agg, plt):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for ax, metric, ttl, yl in [
        (axes[0], "gauc", "(a) Mode-signature recovery", "gAUC (8 mode dirs)"),
        (axes[1], "eauc", "(b) Content recovery", "eAUC (12 content dirs)"),
        (axes[2], "nmse", "(c) Reconstruction error", "NMSE"),
    ]:
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
    axes[0].set_ylim(0, 1.02); axes[1].set_ylim(0, 1.02)
    axes[0].legend(ncol=2, fontsize=7.2, loc="lower right")
    fig.tight_layout()
    _save(fig, plt, "changepoint_local_tradeoff")


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


def _frontier_table(agg, metric):
    h = ("| arch / T | " + " | ".join(f"d={d}" for d in D_SAES) + " |\n"
         "|---|" + "|".join("---" for _ in D_SAES) + "|\n")
    for arch, T in ARCH_T:
        cells = " | ".join(_f(g(agg, "trained", 1, arch, T, d, metric)) for d in D_SAES)
        name = f"**{label(arch,T)}**" if (arch, T) not in PER_TOKEN else label(arch, T)
        h += f"| {name} | {cells} |\n"
    return h.rstrip()


def table_untrained(agg):
    h = ("| arch / T | mode untrained | mode trained | τ untrained | τ trained |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {_fs(g(agg,'untrained',1,arch,T,20,'mode_recovery'))} "
              f"| {_fs(g(agg,'trained',1,arch,T,20,'mode_recovery'))} "
              f"| {_fs(g(agg,'untrained',1,arch,T,20,'tss_recovery'))} "
              f"| {_fs(g(agg,'trained',1,arch,T,20,'tss_recovery'))} |\n")
    return h.rstrip()


def table_kpos(agg):
    h = ("| arch / T | mode @ $k_{pos}{=}1$ | mode @ $k_{pos}{=}2$ | τ @ $k_{pos}{=}1$ | τ @ $k_{pos}{=}2$ |\n"
         "|---|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {_f(g(agg,'trained',1,arch,T,20,'mode_recovery'))} "
              f"| {_f(g(agg,'trained',2,arch,T,20,'mode_recovery'))} "
              f"| {_f(g(agg,'trained',1,arch,T,20,'tss_recovery'))} "
              f"| {_f(g(agg,'trained',2,arch,T,20,'tss_recovery'))} |\n")
    return h.rstrip()


def table_feature_recovery(agg):
    h = ("| arch / T | gAUC (mode dirs) | eAUC (content dirs) | NMSE |\n"
         "|---|---|---|---|\n")
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} | {_f(g(agg,'trained',1,arch,T,20,'gauc'))} "
              f"| {_f(g(agg,'trained',1,arch,T,20,'eauc'))} "
              f"| {_f(g(agg,'trained',1,arch,T,20,'nmse'))} |\n")
    return h.rstrip()


def headline_block(agg, ceil):
    pt_mode = np.nanmean([g(agg, "trained", 1, a, 1, 20, "mode_recovery")[0]
                          for a in ("batchtopk_sae", "tsae")])
    pt_tss = np.nanmean([g(agg, "trained", 1, a, 1, 20, "tss_recovery")[0]
                         for a in ("batchtopk_sae", "tsae")])
    win_mode = {(a, T): g(agg, "trained", 1, a, T, 20, "mode_recovery")[0]
                for a, T in ARCH_T if (a, T) not in PER_TOKEN}
    win_tss = {(a, T): g(agg, "trained", 1, a, T, 20, "tss_recovery")[0]
               for a, T in ARCH_T if (a, T) not in PER_TOKEN}
    best_w_tss = max(win_tss, key=lambda k: np.nan_to_num(win_tss[k], nan=-9))
    best_w_mode = max(win_mode, key=lambda k: np.nan_to_num(win_mode[k], nan=-9))
    un_best = g(agg, "untrained", 1, best_w_tss[0], best_w_tss[1], 20, "tss_recovery")[0]
    return (
        f"- **DC half (mode `m_t`):** per-token = **{pt_mode:.2f}** normalized balanced "
        f"acc at d_sae=20 vs best window {LABEL[best_w_mode[0]]} T={best_w_mode[1]} = "
        f"**{win_mode[best_w_mode]:.2f}** — the persistent mode is {'NOT a window win' if pt_mode >= win_mode[best_w_mode] - 0.02 else 'see narrative'}.\n"
        f"- **AC half (time-since-switch `τ_t`):** per-token = **{pt_tss:.2f}** "
        f"(provable chance floor ≈ 0) vs best window {LABEL[best_w_tss[0]]} "
        f"T={best_w_tss[1]} = **{win_tss[best_w_tss]:.2f}** "
        f"(in-tile info ceilings {ceil['tau_info_by_T'][2]:.2f}/"
        f"{ceil['tau_info_by_T'][4]:.2f}/{ceil['tau_info_by_T'][8]:.2f} at T=2/4/8).\n"
        f"- **Access control:** untrained {LABEL[best_w_tss[0]]} T={best_w_tss[1]} reaches "
        f"τ = {un_best:.2f}; raw-linear window access is provably ≈ chance "
        f"(gating A4), so trained window recovery above that is learned structure.\n"
        f"- **Substrate:** geometric dwell anchored on the measured topic dwell "
        f"(mean run 1.73 → base switch rate {ceil['base_switch_rate']:.2f}), K_m=8, "
        f"uniform Π, F=20 directions, all archs on the BatchTopK fair backbone."
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
    ceil = gating_ceilings()
    n_trained = sum(1 for r in rows if r["kind"] == "trained")
    print(f"[render] {len(rows)} leaderboard cells ({n_trained} trained); "
          f"τ info ceilings {ceil['tau_info_by_T']}")

    fig_main(agg, ceil, plt)
    fig_split(agg, plt)
    fig_T(agg, ceil, plt)
    fig_untrained(agg, plt)
    fig_local_tradeoff(agg, plt)

    blocks = {
        "headline": headline_block(agg, ceil),
        "mode_frontier": _frontier_table(agg, "mode_recovery"),
        "tss_frontier": _frontier_table(agg, "tss_recovery"),
        "cp_frontier": _frontier_table(agg, "cp_recovery"),
        "untrained": table_untrained(agg),
        "kpos": table_kpos(agg),
        "feature_recovery": table_feature_recovery(agg),
    }
    populate(blocks)

    STATS_OUT.write_text(json.dumps({
        "source": "results/leaderboard.jsonl", "n_cells": len(rows), "F": F,
        "gating_ceilings": ceil,
        "agg": {f"{k[0]}|kpos{k[1]}|{k[2]}|T{k[3]}|d{k[4]}": v for k, v in agg.items()},
    }, indent=2))
    print(f"[stats] -> {STATS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
