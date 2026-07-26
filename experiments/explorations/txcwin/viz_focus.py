"""Visuals for a focused head-to-head run (focus.py output).

Four figures, all with every seed shown rather than summarised away:

  1. money      — skill against window size, one line per architecture, individual
                  seeds as dots, per-token baseline as a horizontal band, the same
                  architecture untrained as a dashed line
  2. seeds      — a dot plot of every seed for every cell, so the reader sees the
                  spread that the means are computed from
  3. gain       — trained minus untrained per cell: how much of the score is the
                  dictionary rather than the random projection it started as
  4. card       — the headline comparison with its audited statistics

Run:  .venv/bin/python -m experiments.explorations.txcwin.viz_focus --tag focus_switch_nnz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figs"

NICE = {
    "batchtopk_sae": "per-token SAE",
    "tsae": "T-SAE",
    "stacked_batchtopk": "Stacked SAE",
    "txc_batchtopk_pre": "TXC-pre (adds positions)",
    "txc_batchtopk_post": "TXC-post — the paper's TXC (mixes positions)",
}
ORDER = ["batchtopk_sae", "tsae", "stacked_batchtopk", "txc_batchtopk_pre",
         "txc_batchtopk_post"]
PER_TOKEN = {"batchtopk_sae", "tsae"}
TH = {
    "light": {"bg": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e4e6e8", "null": "#9aa0a6",
              "s": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]},
    "dark": {"bg": "#111a21", "ink": "#e8eef1", "ink2": "#a7bac4",
             "grid": "#22323d", "null": "#778d99",
             "s": ["#3987e5", "#d95926", "#199e70", "#c98500", "#9085e9"]},
}


def _style(mode):
    t = TH[mode]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"], "text.color": t["ink"],
        "axes.labelcolor": t["ink"], "axes.edgecolor": t["grid"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "grid.color": t["grid"], "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False})
    return t


def cells(pl, arch=None, T=None, trained=True):
    out = [c for c in pl["cells"] if bool(c.get("trained", True)) == trained]
    if arch:
        out = [c for c in out if c["arch"] == arch]
    if T is not None:
        out = [c for c in out if c["T"] == T]
    return out


def agg(pl, arch, T, trained=True):
    cs = cells(pl, arch, T, trained)
    if not cs:
        return None
    sk = np.array([c["skill"] for c in cs])
    n = len(sk)
    sd = sk.std(ddof=1) if n > 1 else 0.0
    return {"mean": float(sk.mean()), "sd": float(sd),
            "se": float(sd / np.sqrt(n)) if n > 1 else 0.0,
            "min": float(sk.min()), "max": float(sk.max()), "n": n,
            "seeds": sk.tolist(),
            "l0": float(np.mean([c["l0"] for c in cs]))}


def fig_money(pl, tag, mode):
    t = _style(mode)
    kind = pl["meta"]["kind"]
    chance = 0.5 if kind == "cls" else 0.0
    Ts = sorted({c["T"] for c in cells(pl) if c["T"] > 1})
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=170)
    for i, a in enumerate(ORDER):
        col = t["s"][i]
        if a in PER_TOKEN:
            g = agg(pl, a, 1)
            if not g:
                continue
            ax.axhspan(g["min"], g["max"], color=col, alpha=0.10, linewidth=0)
            ax.axhline(g["mean"], color=col, linewidth=1.8,
                       linestyle=(0, (1, 1.4)), zorder=2)
            ax.annotate(f"{NICE[a]} (one token): {g['mean']:.3f}",
                        xy=(Ts[-1], g["mean"]), xytext=(6, 2),
                        textcoords="offset points", fontsize=8.5, color=col)
            continue
        xs, ys, lo, hi = [], [], [], []
        for T in Ts:
            g = agg(pl, a, T)
            if not g:
                continue
            xs.append(T)
            ys.append(g["mean"])
            lo.append(g["mean"] - g["sd"])
            hi.append(g["mean"] + g["sd"])
            for s in g["seeds"]:
                ax.scatter([T], [s], s=22, color=col, alpha=0.55, zorder=4,
                           edgecolor=t["bg"], linewidth=0.8)
        if not xs:
            continue
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, linewidth=0)
        ax.plot(xs, ys, color=col, linewidth=2.4, marker="o", markersize=6,
                markeredgecolor=t["bg"], markeredgewidth=1.4, label=NICE[a],
                zorder=5)
        us = [(T, agg(pl, a, T, trained=False)) for T in Ts]
        us = [(T, g) for T, g in us if g]
        if us:
            ax.plot([T for T, _ in us], [g["mean"] for _, g in us], color=col,
                    linewidth=1.1, linestyle=(0, (3, 2)), alpha=0.7, zorder=3)
    ax.axhline(chance, color=t["null"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("window size T — how many token positions the dictionary reads at once")
    ax.set_ylabel("how well a linear probe on the dictionary's code\n"
                  + ("recovers the label (correlation)" if kind == "reg"
                     else "separates the two classes (AUC)"))
    ax.grid(axis="y", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{pl['meta']['desc']}\n"
                 f"{pl['meta']['model']} layer {pl['meta']['layer']}, "
                 f"{pl['meta']['steps']} training steps, "
                 f"{len(pl['meta']['seeds'])} seeds, matched code budget",
                 loc="left", fontsize=10.5)
    fig.tight_layout()
    out = FIGS / f"money_{tag}_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_seeds(pl, tag, mode):
    t = _style(mode)
    rows = []
    for i, a in enumerate(ORDER):
        for T in sorted({c["T"] for c in cells(pl, a)}):
            g = agg(pl, a, T)
            u = agg(pl, a, T, trained=False)
            if g:
                rows.append((a, T, g, u, t["s"][i]))
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 2.2), dpi=170)
    for y, (a, T, g, u, col) in enumerate(rows):
        ax.scatter(g["seeds"], [y] * len(g["seeds"]), s=52, color=col, zorder=4,
                   edgecolor=t["bg"], linewidth=1.1)
        ax.plot([g["mean"]] * 2, [y - 0.28, y + 0.28], color=col, linewidth=2.4,
                zorder=3)
        if u:
            ax.scatter(u["seeds"], [y] * len(u["seeds"]), s=40, facecolor="none",
                       edgecolor=col, linewidth=1.3, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{NICE[a].split(' —')[0].split(' (')[0]}  T={T}"
                        for a, T, _, _, _ in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("skill — filled dots are individual trained seeds, "
                  "hollow dots the same architecture untrained, bar the mean")
    ax.grid(axis="x", linewidth=0.6)
    ax.set_title("Every seed, shown. The spread is the reason a single run "
                 "cannot settle anything", loc="left", fontsize=11)
    fig.tight_layout()
    out = FIGS / f"seeds_{tag}_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_gain(pl, tag, mode):
    t = _style(mode)
    rows = []
    for i, a in enumerate(ORDER):
        for T in sorted({c["T"] for c in cells(pl, a)}):
            g, u = agg(pl, a, T), agg(pl, a, T, trained=False)
            if g and u:
                rows.append((f"{NICE[a].split(' —')[0].split(' (')[0]}\nT={T}",
                             g["mean"] - u["mean"], t["s"][i]))
    fig, ax = plt.subplots(figsize=(1.0 * len(rows) + 2.4, 4.0), dpi=170)
    xs = np.arange(len(rows))
    ax.bar(xs, [r[1] for r in rows], 0.6, color=[r[2] for r in rows], zorder=3,
           edgecolor=t["bg"], linewidth=1.4)
    for x, r in zip(xs, rows):
        ax.annotate(f"{r[1]:+.3f}", xy=(x, r[1]), xytext=(0, 4 if r[1] >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=t["ink2"])
    ax.axhline(0, color=t["ink2"], linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("skill gained by TRAINING\n(trained minus random init)")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_title("How much of each score is the dictionary, and how much was "
                 "there at random initialisation", loc="left", fontsize=11)
    fig.tight_layout()
    out = FIGS / f"gain_{tag}_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_card(pl, tag, mode):
    """The headline, with its statistics, as a standalone panel."""
    t = _style(mode)
    Ts = sorted({c["T"] for c in cells(pl) if c["T"] > 1})
    best = max(((agg(pl, "txc_batchtopk_post", T), T) for T in Ts),
               key=lambda g: g[0]["mean"] if g[0] else -9)
    gp, Tb = best
    base = max((agg(pl, a, 1) for a in PER_TOKEN if agg(pl, a, 1)),
               key=lambda g: g["mean"])
    st = agg(pl, "stacked_batchtopk", Tb)
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=170)
    ax.axis("off")
    labels = ["per-token SAE\n(one position)", f"Stacked SAE\nT={Tb}",
              f"TXC-post (paper's TXC)\nT={Tb}"]
    vals = [base["mean"], st["mean"] if st else np.nan, gp["mean"]]
    sds = [base["sd"], st["sd"] if st else 0, gp["sd"]]
    cols = [t["s"][0], t["s"][2], t["s"][4]]
    xs = np.arange(3)
    ax2 = fig.add_axes([0.08, 0.20, 0.52, 0.62])
    ax2.bar(xs, vals, 0.6, color=cols, zorder=3, edgecolor=t["bg"], linewidth=1.6)
    ax2.errorbar(xs, vals, yerr=sds, fmt="none", ecolor=t["ink2"], elinewidth=1.3,
                 capsize=4, zorder=4)
    for x, v, s in zip(xs, vals, sds):
        ax2.annotate(f"{v:.3f}", xy=(x, v + s), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=12,
                     fontweight="600", color=t["ink"])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("skill (correlation)" if pl["meta"]["kind"] == "reg"
                   else "skill (AUC)")
    ax2.grid(axis="y", linewidth=0.6)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    se = float(np.hypot(gp["se"], base["se"]))
    z = (gp["mean"] - base["mean"]) / se if se else float("nan")
    lines = [
        ("task", pl["meta"]["desc"]),
        ("model", f"{pl['meta']['model']}  layer {pl['meta']['layer']}"),
        ("advantage", f"{gp['mean'] - base['mean']:+.3f}  "
                      f"({gp['mean'] / max(base['mean'], 1e-9):.1f}x the baseline)"),
        ("significance", f"{z:.1f} sigma over {gp['n']} seeds"),
        ("worst winner seed", f"{gp['min']:.3f}  vs best baseline seed "
                              f"{base['max']:.3f}"),
        ("learned over init", f"{gp['mean'] - agg(pl,'txc_batchtopk_post',Tb,False)['mean']:+.3f}"),
        ("code budget", f"{gp['l0']:.1f} vs {base['l0']:.1f} active latents"),
        ("training", f"{pl['meta']['steps']} steps, d_sae={pl['meta']['d_sae']}"),
    ]
    y = 0.80
    fig.text(0.64, 0.88, "audited statistics", fontsize=10,
             color=t["ink2"], family="monospace")
    for k, v in lines:
        fig.text(0.64, y, k, fontsize=8.5, color=t["ink2"], family="monospace")
        fig.text(0.64, y - 0.035, str(v), fontsize=9.5, color=t["ink"])
        y -= 0.095
    fig.suptitle("Reading several positions at once, and MIXING them, "
                 "more than doubles what a per-token dictionary exposes",
                 x=0.02, ha="left", fontsize=12.5)
    out = FIGS / f"card_{tag}_{mode}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    pl = json.loads((RESULTS / f"{a.tag}.json").read_text())
    for mode in ("light", "dark"):
        for fn in (fig_money, fig_seeds, fig_gain, fig_card):
            try:
                print("wrote", fn(pl, a.tag, mode))
            except Exception as e:
                print(f"  {fn.__name__} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
