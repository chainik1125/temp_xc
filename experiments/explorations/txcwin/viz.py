"""Visuals for the TXC-win sweep. Plain labels, CIs everywhere, honest hatching.

Six figures:
  1. label_atlas   — what each candidate label actually looks like over a real
                     document (the intuition figure: sawtooth clocks, drifting
                     rates), so "state tracking" is visible rather than asserted
  2. heatmap       — task x architecture skill, with the per-token baseline column
                     marked and undertrained cells hatched
  3. curves        — skill vs window size per task, one line per architecture,
                     bootstrap CI bands, untrained control dashed in the same hue
  4. advantage     — per task: best window architecture minus best per-token
                     baseline, sorted, with CIs; coloured by which family won
  5. learning      — trained skill vs untrained skill, one point per cell. The
                     diagonal is "training changed nothing". This is the figure
                     that decides whether any of this is about dictionaries at all
  6. fairness      — realized code rate per architecture and window size, so the
                     matched-sparsity claim is visible instead of trusted

Run:  .venv/bin/python -m experiments.explorations.txcwin.viz --tag gpt2_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figs"
LABELS = HERE.parent / "task_hunt" / "labels"

# validated categorical slots (dataviz reference instance), fixed order
SLOTS = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e87ba4"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#9085e9", "#d55181"],
}
TH = {
    "light": {"bg": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e4e6e8", "null": "#9aa0a6", "hatch": "#b9bcc0"},
    "dark": {"bg": "#111a21", "ink": "#e8eef1", "ink2": "#a7bac4",
             "grid": "#22323d", "null": "#778d99", "hatch": "#4a5a66"},
}
NICE = {
    "batchtopk_sae": "per-token SAE",
    "tsae": "T-SAE",
    "stacked_batchtopk": "Stacked SAE",
    "txc_batchtopk_pre": "TXC-pre (additive)",
    "txc_batchtopk_post": "TXC-post — the paper's TXC",
}
ORDER = ["batchtopk_sae", "tsae", "stacked_batchtopk", "txc_batchtopk_pre",
         "txc_batchtopk_post"]
PER_TOKEN = {"batchtopk_sae", "tsae"}
TASK_NICE = {
    "switch_clock": "tokens since source switch",
    "source_id": "which source document",
    "turn_clock": "tokens since speaker change",
    "turn_level": "trailing turn length",
    "novelty_rate": "trailing novelty rate",
    "novelty_resid": "novelty (trend removed)",
    "list_density": "list-marker density",
    "question_rate": "question rate",
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
        "legend.frameon": False,
    })
    return t, SLOTS[mode]


def load(tag: str) -> dict:
    return json.loads((RESULTS / f"sweep_{tag}.json").read_text())


def tr_cells(pl):
    return [c for c in pl["cells"] if c.get("trained", True)]


def un_cells(pl):
    return [c for c in pl["cells"] if not c.get("trained", True)]


def get(cells, task=None, arch=None, T=None):
    out = cells
    if task is not None:
        out = [c for c in out if c["task"] == task]
    if arch is not None:
        out = [c for c in out if c["arch"] == arch]
    if T is not None:
        out = [c for c in out if c["T"] == T]
    return out


# ── 1. what the labels look like ────────────────────────────────────────
def fig_label_atlas(mode: str) -> Path:
    """The intuition figure: these labels are running states, not word properties."""
    t, cols = _style(mode)
    panels = [
        ("interleave_fineweb", "tss", "tokens since the text switched source",
         "a sawtooth: it climbs while one document continues, and resets on every switch"),
        ("novelty_fineweb", "nov_rate", "trailing rate of new word types",
         "a drifting level: it rises where the document enters new material"),
        ("punctint_fineweb", "lam_list", "trailing density of list markers",
         "bursty: near zero in prose, high inside enumerations"),
        ("dialevel_dailydialog", "tlevel", "trailing mean turn length",
         "a slow regime signal: rapid-fire exchange versus long-form turns"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.4, 8.2), dpi=170)
    for ax, (stem, field, title, note), col in zip(axes, panels, cols):
        f = LABELS / f"{stem}_gpt2.npz"
        if not f.exists():
            continue
        npz = np.load(f)
        y = npz[field].astype(np.float32)
        off = npz["doc_off"].astype(np.int64)
        # pick a document with real variation
        best, bi = -1, 1
        for i in range(1, min(40, len(off) - 1)):
            seg = y[off[i]:off[i + 1]]
            if len(seg) > 300 and np.nanstd(seg) > best:
                best, bi = np.nanstd(seg), i
        seg = y[off[bi]:off[bi + 1]][:600]
        ax.plot(seg, color=col, linewidth=1.5)
        ax.fill_between(np.arange(len(seg)), seg, seg.min(), color=col, alpha=0.13,
                        linewidth=0)
        ax.set_title(f"{title}", loc="left", fontsize=10.5)
        ax.annotate(note, xy=(0.0, -0.34), xycoords="axes fraction",
                    fontsize=9, color=t["ink2"])
        ax.set_xlim(0, len(seg))
        ax.grid(axis="y", linewidth=0.6)
        ax.set_ylabel("label value", fontsize=9)
    axes[-1].set_xlabel("token position within one real document")
    fig.suptitle("What these tasks ask the dictionary to represent — running "
                 "states, not properties of the current word",
                 x=0.012, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=3.4)
    out = FIGS / f"labels_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── 2. heatmap ──────────────────────────────────────────────────────────
def fig_heatmap(pl, tag: str, mode: str) -> Path:
    t, cols = _style(mode)
    cells = tr_cells(pl)
    un = un_cells(pl)
    tasks = [k for k in TASK_NICE if get(cells, task=k)]
    colsx = []
    for a in ORDER:
        for T in sorted({c["T"] for c in get(cells, arch=a)}):
            colsx.append((a, T))
    M = np.full((len(tasks), len(colsx)), np.nan)
    U = np.zeros_like(M, dtype=bool)
    for i, task in enumerate(tasks):
        for j, (a, T) in enumerate(colsx):
            c = get(cells, task=task, arch=a, T=T)
            if not c:
                continue
            M[i, j] = c[0]["skill"]
            u = get(un, task=task, arch=a, T=T)
            if u and c[0]["skill"] - u[0]["skill"] < 0.02:
                U[i, j] = True          # training added nothing
    fig, ax = plt.subplots(figsize=(1.05 * len(colsx) + 3.6,
                                    0.52 * len(tasks) + 2.6), dpi=170)
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    for i in range(len(tasks)):
        for j in range(len(colsx)):
            if np.isnan(M[i, j]):
                continue
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] < 0.62 else "#111111")
            if U[i, j]:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           hatch="////", edgecolor=t["hatch"],
                                           linewidth=0))
    ax.set_xticks(range(len(colsx)))
    ax.set_xticklabels([f"{NICE[a].split(' —')[0]}\nT={T}" for a, T in colsx],
                       fontsize=8, rotation=0)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([TASK_NICE[k] for k in tasks], fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="held-out skill (r or AUC)")
    ax.set_title("Every task against every architecture — hatched cells are ones "
                 "where training beat random init by less than 0.02", loc="left",
                 fontsize=10.5)
    fig.tight_layout()
    out = FIGS / f"heatmap_{tag}_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── 3. curves ───────────────────────────────────────────────────────────
def fig_curves(pl, tag: str, mode: str) -> Path:
    t, cols = _style(mode)
    cells, un = tr_cells(pl), un_cells(pl)
    tasks = [k for k in TASK_NICE if get(cells, task=k)]
    n = len(tasks)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.3 * nrow),
                             dpi=170, squeeze=False)
    for idx, task in enumerate(tasks):
        ax = axes[idx // ncol][idx % ncol]
        for ai, a in enumerate(ORDER):
            cs = sorted(get(cells, task=task, arch=a), key=lambda c: c["T"])
            if not cs:
                continue
            col = cols[ai]
            xs = [c["T"] for c in cs]
            ys = [c["skill"] for c in cs]
            lo = [c["ci_lo"] for c in cs]
            hi = [c["ci_hi"] for c in cs]
            if len(xs) == 1:                     # per-token: a horizontal band
                ax.axhline(ys[0], color=col, linewidth=1.8,
                           linestyle=(0, (1, 1.2)), zorder=2)
                ax.axhspan(lo[0], hi[0], color=col, alpha=0.10, linewidth=0)
            else:
                ax.fill_between(xs, lo, hi, color=col, alpha=0.15, linewidth=0)
                ax.plot(xs, ys, color=col, linewidth=2, marker="o", markersize=4.5,
                        markeredgecolor=t["bg"], markeredgewidth=1.2, zorder=3)
            us = sorted(get(un, task=task, arch=a), key=lambda c: c["T"])
            if len(us) > 1:
                ax.plot([c["T"] for c in us], [c["skill"] for c in us],
                        color=col, linewidth=1.1, linestyle=(0, (3, 2)),
                        alpha=0.75, zorder=2)
        ax.set_title(TASK_NICE[task], loc="left", fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({c["T"] for c in cells if c["T"] > 1}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(axis="y", linewidth=0.6)
        if idx % ncol == 0:
            ax.set_ylabel("held-out skill")
        if idx // ncol == nrow - 1:
            ax.set_xlabel("window size T")
    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].axis("off")
    handles = [plt.Line2D([], [], color=cols[i], lw=2, label=NICE[a])
               for i, a in enumerate(ORDER)]
    handles += [plt.Line2D([], [], color=t["null"], lw=1.1, ls=(0, (3, 2)),
                           label="same architecture, untrained (random init)")]
    fig.legend(handles=handles, loc="lower center", ncols=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Does a wider window help? Solid = trained, dotted band = "
                 "per-token baseline, dashed = random init",
                 x=0.012, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.955))
    out = FIGS / f"curves_{tag}_{mode}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 4. advantage ────────────────────────────────────────────────────────
def fig_advantage(pl, tag: str, mode: str) -> Path:
    t, cols = _style(mode)
    cells, un = tr_cells(pl), un_cells(pl)
    rows = []
    for task in TASK_NICE:
        cs = get(cells, task=task)
        if not cs:
            continue
        base = [c for c in cs if c["arch"] in PER_TOKEN]
        wins = [c for c in cs if c["arch"] not in PER_TOKEN and c["T"] >= 2
                and not c.get("degenerate")]
        if not base or not wins:
            continue
        b = max(base, key=lambda c: c["skill"])
        w = max(wins, key=lambda c: c["skill"])
        u = get(un, task=task, arch=w["arch"], T=w["T"])
        rows.append((w["skill"] - b["skill"], task, w, b,
                     u[0]["skill"] if u else np.nan))
    rows.sort()
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(rows) + 2.6), dpi=170)
    ys = np.arange(len(rows))
    for y, (adv, task, w, b, ui) in zip(ys, rows):
        is_post = w["arch"] == "txc_batchtopk_post"
        col = cols[4] if is_post else cols[2]
        ax.barh(y, adv, height=0.6, color=col, zorder=3,
                edgecolor=t["bg"], linewidth=1.4)
        # CI on the difference, conservative: winner CI vs baseline point
        ax.plot([w["ci_lo"] - b["skill"], w["ci_hi"] - b["skill"]], [y, y],
                color=t["ink2"], linewidth=1.2, zorder=4)
        learned = (w["skill"] - ui) if np.isfinite(ui) else np.nan
        tagtxt = (f"{NICE[w['arch']].split(' —')[0]} @T{w['T']}  "
                  f"{w['skill']:.3f} vs {b['skill']:.3f}"
                  f"   over-init {learned:+.3f}")
        ax.annotate(tagtxt, xy=(max(adv, 0) + 0.006, y), va="center", fontsize=8.5,
                    color=t["ink2"])
    ax.axvline(0, color=t["ink2"], linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([TASK_NICE[r[1]] for r in rows], fontsize=9.5)
    ax.set_xlabel("advantage of the best window architecture over the best "
                  "per-token baseline (skill difference)")
    ax.grid(axis="x", linewidth=0.6)
    ax.legend(handles=[Patch(color=cols[4], label="won by TXC-post (the paper's TXC)"),
                       Patch(color=cols[2], label="won by an additive window arch")],
              loc="lower right", fontsize=9)
    ax.set_title("Which tasks reward a window at all — and which family collects "
                 "the reward", loc="left", fontsize=11)
    fig.tight_layout()
    out = FIGS / f"advantage_{tag}_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── 5. learning check ───────────────────────────────────────────────────
def fig_learning(pl, tag: str, mode: str) -> Path:
    """Trained vs untrained. Anything on the diagonal means the dictionary is
    irrelevant and the probe is reading the raw activations through it."""
    t, cols = _style(mode)
    cells, un = tr_cells(pl), un_cells(pl)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=170)
    for ai, a in enumerate(ORDER):
        xs, ys = [], []
        for c in get(cells, arch=a):
            u = get(un, task=c["task"], arch=a, T=c["T"])
            if not u:
                continue
            xs.append(u[0]["skill"])
            ys.append(c["skill"])
        if not xs:
            continue
        ax.scatter(xs, ys, s=46, color=cols[ai], alpha=0.85, zorder=3,
                   edgecolor=t["bg"], linewidth=1.1, label=NICE[a])
    lim = [-0.05, 1.02]
    ax.plot(lim, lim, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.3)
    ax.annotate("training changed nothing", xy=(0.62, 0.62), xytext=(0.30, 0.86),
                fontsize=9, color=t["ink2"],
                arrowprops=dict(arrowstyle="->", color=t["ink2"], lw=1))
    ax.annotate("above the line:\nthe dictionary learned something", xy=(0.05, 0.55),
                fontsize=9, color=t["ink2"])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("skill with a RANDOM, untrained dictionary")
    ax.set_ylabel("skill with the TRAINED dictionary")
    ax.grid(linewidth=0.6)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Is any of this about dictionaries? One point per "
                 "(task, architecture, window)", loc="left", fontsize=11)
    fig.tight_layout()
    out = FIGS / f"learning_{tag}_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ── 6. fairness ─────────────────────────────────────────────────────────
def fig_fairness(pl, tag: str, mode: str) -> Path:
    t, cols = _style(mode)
    cells = tr_cells(pl)
    combos = []
    for ai, a in enumerate(ORDER):
        for T in sorted({c["T"] for c in get(cells, arch=a)}):
            cs = get(cells, arch=a, T=T)
            combos.append((a, T, float(np.mean([c["l0"] for c in cs])), cols[ai]))
    fig, ax = plt.subplots(figsize=(9.0, 3.8), dpi=170)
    xs = np.arange(len(combos))
    ax.bar(xs, [c[2] for c in combos], 0.62, color=[c[3] for c in combos],
           zorder=3, edgecolor=t["bg"], linewidth=1.4)
    for x, c in zip(xs, combos):
        ax.annotate(f"{c[2]:.1f}", xy=(x, c[2]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=t["ink2"])
    target = float(np.median([c[2] for c in combos]))
    ax.axhline(target, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.2)
    ax.annotate(f"median {target:.1f} active latents per read",
                xy=(len(combos) - 0.5, target), xytext=(0, 6),
                textcoords="offset points", ha="right", fontsize=8.5,
                color=t["ink2"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{NICE[a].split(' —')[0]}\nT={T}" for a, T, _, _ in combos],
                       fontsize=8)
    ax.set_ylabel("active latents per read")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_title("Fairness check: every architecture is read at a comparable code "
                 "rate, so no one wins by spending more", loc="left", fontsize=11)
    fig.tight_layout()
    out = FIGS / f"fairness_{tag}_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt2_pilot")
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    pl = load(a.tag)
    for mode in ("light", "dark"):
        for fn in (fig_label_atlas,):
            print("wrote", fn(mode))
        for fn in (fig_heatmap, fig_curves, fig_advantage, fig_learning,
                   fig_fairness):
            try:
                print("wrote", fn(pl, a.tag, mode))
            except Exception as e:
                print(f"  {fn.__name__} failed: {e}")


if __name__ == "__main__":
    main()
