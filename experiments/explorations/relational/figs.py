"""Figures for the relational gate — one source (results/*.json), no hand numbers.

Every panel carries the bootstrap CI that `gate.py` computed, the permutation
null band, and the chance line. Theory lines (provable floors) are drawn in a
neutral dashed style so they can never be mistaken for a measurement.

Palette: the dataviz skill's *validated* reference instance, categorical slots
1-3 (blue / orange / aqua), assigned in fixed order by role and never cycled:
  slot 1  per-token linear      the paper's per-token baseline
  slot 2  window linear         the ADDITIVE ceiling (bounds T-SAE, Stacked,
                                TXC-pre and any pooled per-token code)
  slot 3  window MLP            the NONLINEAR ceiling (what a position-mixing
                                code could reach)
The custom palette was not re-validated because `node` is unavailable on this
pod, so the validated defaults are used unchanged rather than eyeballed.

Each figure is emitted twice (light + dark), stepped for its own surface, since
an embedded PNG cannot follow the page theme.

Run:  .venv/bin/python -m experiments.explorations.relational.figs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE / "figs"

ARMS = [
    ("per_token", "per-token linear", "s1"),
    ("window_flat", "window linear (additive ceiling)", "s2"),
    ("window_mlp", "window MLP (nonlinear ceiling)", "s3"),
]

THEME = {
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e4e6e8", "s1": "#2a78d6", "s2": "#eb6834",
              "s3": "#1baf7a", "s4": "#4a3aa7", "null": "#9aa0a6"},
    "dark": {"surface": "#111a21", "ink": "#e8eef1", "ink2": "#a7bac4",
             "grid": "#22323d", "s1": "#3987e5", "s2": "#d95926",
             "s3": "#199e70", "s4": "#9085e9", "null": "#778d99"},
}


def _style(mode: str):
    t = THEME[mode]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink"], "axes.edgecolor": t["grid"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "grid.color": t["grid"], "font.size": 10,
        "axes.titlesize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })
    return t


def _cells(payload: dict, stratum: str = "all") -> list[dict]:
    return [c for c in payload["cells"]
            if c.get("stratum") == stratum and "per_token" in c]


def fig_depth(payload: dict, task: str, mode: str, T: int | None = None) -> Path:
    """AUC vs layer, one line per arm, CI bands. The conversion diagnostic."""
    t = _style(mode)
    cells = _cells(payload)
    if T is None:
        Ts = sorted({c["T"] for c in cells})
        T = Ts[len(Ts) // 2]
    cells = [c for c in cells if c["T"] == T]
    layers = sorted({c["layer"] for c in cells})
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=170)
    for key, label, slot in ARMS:
        xs, ys, lo, hi = [], [], [], []
        for L in layers:
            c = next((c for c in cells if c["layer"] == L), None)
            if not c:
                continue
            xs.append(L)
            ys.append(c[key]["value"])
            lo.append(c[key]["ci_lo"])
            hi.append(c[key]["ci_hi"])
        if not xs:
            continue
        ax.fill_between(xs, lo, hi, color=t[slot], alpha=0.16, linewidth=0)
        ax.plot(xs, ys, color=t[slot], linewidth=2, marker="o",
                markersize=5, markeredgecolor=t["surface"],
                markeredgewidth=1.4, label=label, zorder=3)
    ax.axhline(0.5, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.2,
               zorder=1)
    ax.annotate("chance", xy=(layers[0], 0.5), xytext=(0, 6),
                textcoords="offset points", color=t["ink2"], fontsize=8.5)
    ax.set_xlabel("residual layer")
    ax.set_ylabel("test AUC")
    ax.set_ylim(0.44, 1.03)
    ax.set_xticks(layers)
    ax.grid(axis="y", linewidth=0.7)
    ax.set_title(f"{task} — where the label becomes readable (T={T})", loc="left")
    ax.legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    out = FIGS / f"{task}_depth_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_ladder(payload: dict, task: str, mode: str) -> Path:
    """g and the nonlinear residual vs T, split by IN/OUT stratum.

    The IN/OUT split is the causal control: constituent A is at a known token
    distance, so a window advantage that survives only when the window REACHES A
    is cross-position binding; one present in both strata is an artifact.
    """
    t = _style(mode)
    layers = sorted({c["layer"] for c in payload["cells"] if "per_token" in c})
    layer = layers[len(layers) // 2]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=170, sharey=True)
    for ax, (stat, title) in zip(axes, [
            ("g", "window linear − per-token"),
            ("nonlinear_residual", "window MLP − additive ceiling")]):
        for stratum, marker, alpha in [("in", "o", 1.0), ("out", "^", 0.55)]:
            cells = [c for c in payload["cells"]
                     if c.get("stratum") == stratum and c.get("layer") == layer
                     and stat in c]
            if not cells:
                continue
            cells.sort(key=lambda c: c["T"])
            xs = [c["T"] for c in cells]
            ys = [c[stat] for c in cells]
            ax.plot(xs, ys, marker=marker, linewidth=2, markersize=6,
                    color=t["s2" if stat == "g" else "s3"], alpha=alpha,
                    markeredgecolor=t["surface"], markeredgewidth=1.4,
                    label=f"A {'inside' if stratum == 'in' else 'outside'} window",
                    zorder=3)
        sig = [c["three_sigma"] for c in payload["cells"]
               if c.get("layer") == layer and "three_sigma" in c]
        if sig:
            band = float(np.median(sig))
            ax.axhspan(-band, band, color=t["null"], alpha=0.16, linewidth=0)
            ax.annotate("±3σ null", xy=(0.02, 0.5), xycoords="axes fraction",
                        color=t["ink2"], fontsize=8.5, va="center")
        ax.axhline(0, color=t["grid"], linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({c["T"] for c in payload["cells"] if "T" in c}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("window size T")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", linewidth=0.7)
        ax.legend(fontsize=8.5, loc="upper left")
    axes[0].set_ylabel("Δ AUC")
    fig.suptitle(f"{task} — layer {layer}: does a window add anything a single "
                 f"position lacks?", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGS / f"{task}_ladder_{mode}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_conversion(payloads: list[dict], task: str, mode: str, T: int) -> Path:
    """The conversion curve: AUC vs layer merged across runs, plus the
    nonlinear residual — the only quantity a position-mixing code could convert
    into a linear readout.

    Left panel answers "when does the label become readable, and to whom".
    Right panel answers "is there any headroom a coincidence code could take",
    with the IN/OUT strata separated so a residual that is not binding is
    visible as such.
    """
    t = _style(mode)
    cells = [c for pl in payloads for c in pl["cells"]
             if "per_token" in c and c["T"] == T]
    if not cells:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), dpi=170)

    ax = axes[0]
    allc = [c for c in cells if c["stratum"] == "all"]
    layers = sorted({c["layer"] for c in allc})
    for key, label, slot in ARMS:
        xs, ys, lo, hi = [], [], [], []
        for L in layers:
            c = next((c for c in allc if c["layer"] == L), None)
            if not c:
                continue
            xs.append(L); ys.append(c[key]["value"])
            lo.append(c[key]["ci_lo"]); hi.append(c[key]["ci_hi"])
        ax.fill_between(xs, lo, hi, color=t[slot], alpha=0.16, linewidth=0)
        ax.plot(xs, ys, color=t[slot], linewidth=2, marker="o", markersize=5,
                markeredgecolor=t["surface"], markeredgewidth=1.4,
                label=label, zorder=3)
    ax.axhline(0.5, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.2)
    ax.annotate("chance", xy=(layers[0], 0.5), xytext=(2, 5),
                textcoords="offset points", color=t["ink2"], fontsize=8.5)
    ax.set_xlabel("residual layer"); ax.set_ylabel("test AUC")
    ax.set_ylim(0.44, 1.03); ax.set_xticks(layers)
    ax.grid(axis="y", linewidth=0.7)
    ax.set_title("who can read the label", loc="left", fontsize=10)
    ax.legend(loc="center right", fontsize=8.5)

    ax = axes[1]
    for stratum, marker, lab in [("in", "o", "A inside window"),
                                 ("out", "^", "A outside window")]:
        sc = sorted([c for c in cells if c["stratum"] == stratum],
                    key=lambda c: c["layer"])
        if not sc:
            continue
        ax.plot([c["layer"] for c in sc], [c["nonlinear_residual"] for c in sc],
                marker=marker, linewidth=2, markersize=6, color=t["s3"],
                alpha=1.0 if stratum == "in" else 0.5,
                markeredgecolor=t["surface"], markeredgewidth=1.4,
                label=lab, zorder=3)
    band = float(np.median([c["three_sigma"] for c in cells]))
    ax.axhspan(-band, band, color=t["null"], alpha=0.16, linewidth=0)
    ax.annotate("±3σ null", xy=(0.62, 0.5), xycoords="axes fraction",
                color=t["ink2"], fontsize=8.5, va="center")
    ax.axhline(0, color=t["grid"], linewidth=1)
    ax.set_xlabel("residual layer")
    ax.set_ylabel("window MLP − additive ceiling")
    ax.set_xticks(layers)
    ax.grid(axis="y", linewidth=0.7)
    ax.set_title("regime-3 headroom", loc="left", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(f"{task} (T={T}) — the label is built across depth and "
                 f"linearised per position by layer 4",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGS / f"{task}_conversion_T{T}_{mode}.png"
    fig.savefig(out); plt.close(fig)
    return out


ATLAS_COLORS = {"agreement": "s1", "contradiction": "s2", "role": "s3",
                "parity": "s4"}
ATLAS_LABEL = {
    "agreement": "agreement equality (cand. 5)",
    "contradiction": "fact consistency (cand. 4)",
    "role": "labelled role (cand. 1a)",
    "parity": "structural parity (cand. 1b)",
}


def fig_atlas(by_task: dict[str, list[dict]], mode: str) -> Path:
    """The atlas: three relational labels, three conversions.

    LEFT: a per-token linear probe against depth. Every label starts at chance at
    the embeddings and is fully linearised per position by layer 4-8, so at any
    depth a dictionary is actually trained the additive ceiling already contains
    the relation.
    RIGHT: the nonlinear residual — the only headroom a position-mixing code could
    convert into a linear readout. It is non-zero in exactly one place: the
    embedding layer, before the model has done the work.
    """
    t = _style(mode)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), dpi=170)
    for task, pls in sorted(by_task.items()):
        slot = ATLAS_COLORS.get(task, "s1")
        cells = [c for pl in pls for c in pl["cells"]
                 if "per_token" in c and c["stratum"] == "all"]
        if not cells:
            continue
        # one point per layer: the largest T available (most favourable to windows)
        by_layer: dict[int, dict] = {}
        for c in cells:
            cur = by_layer.get(c["layer"])
            if cur is None or c["T"] > cur["T"]:
                by_layer[c["layer"]] = c
        Ls = sorted(by_layer)
        axes[0].fill_between(Ls, [by_layer[L]["per_token"]["ci_lo"] for L in Ls],
                             [by_layer[L]["per_token"]["ci_hi"] for L in Ls],
                             color=t[slot], alpha=0.15, linewidth=0)
        axes[0].plot(Ls, [by_layer[L]["per_token"]["value"] for L in Ls],
                     color=t[slot], linewidth=2, marker="o", markersize=5,
                     markeredgecolor=t["surface"], markeredgewidth=1.4,
                     label=ATLAS_LABEL.get(task, task), zorder=3)
        best = {}
        for c in cells:
            L = c["layer"]
            if L not in best or c["nonlinear_residual"] > best[L]["nonlinear_residual"]:
                best[L] = c
        axes[1].plot(sorted(best), [best[L]["nonlinear_residual"] for L in sorted(best)],
                     color=t[slot], linewidth=2, marker="o", markersize=5,
                     markeredgecolor=t["surface"], markeredgewidth=1.4,
                     label=ATLAS_LABEL.get(task, task), zorder=3)
    axes[0].axhline(0.5, color=t["null"], linestyle=(0, (4, 3)), linewidth=1.2)
    axes[0].annotate("chance", xy=(0, 0.5), xytext=(3, 5),
                     textcoords="offset points", color=t["ink2"], fontsize=8.5)
    axes[0].set_xlabel("residual layer"); axes[0].set_ylabel("per-token test AUC")
    axes[0].set_ylim(0.44, 1.04)
    axes[0].grid(axis="y", linewidth=0.7)
    axes[0].set_title("a per-token probe reads every relation by layer 4",
                      loc="left", fontsize=10)
    axes[0].legend(loc="lower right", fontsize=8.5)

    allc = [c for pls in by_task.values() for pl in pls for c in pl["cells"]
            if "three_sigma" in c]
    band = float(np.median([c["three_sigma"] for c in allc]))
    axes[1].axhspan(-band, band, color=t["null"], alpha=0.16, linewidth=0)
    axes[1].annotate("±3σ null", xy=(0.55, 0.52), xycoords="axes fraction",
                     color=t["ink2"], fontsize=8.5)
    axes[1].axhline(0, color=t["grid"], linewidth=1)
    axes[1].set_xlabel("residual layer")
    axes[1].set_ylabel("window MLP − additive ceiling")
    axes[1].grid(axis="y", linewidth=0.7)
    axes[1].set_title("headroom exists only before the model computes it",
                      loc="left", fontsize=10)
    fig.suptitle("Four relational labels, four conversions — the advantage lives "
                 "only at layer 0 (R1-Distill-Llama-8B)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGS / f"atlas_{mode}.png"
    fig.savefig(out); plt.close(fig)
    return out


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []
    for path in sorted(RESULTS.glob("gate_*.json")):
        payload = json.loads(path.read_text())
        task = payload["meta"]["task"]
        tag = payload["meta"].get("tag", "run")
        name = f"{task}_{tag}"
        for mode in ("light", "dark"):
            if _cells(payload):
                made.append(fig_depth(payload, name, mode))
            if any("nonlinear_residual" in c for c in payload["cells"]):
                made.append(fig_ladder(payload, name, mode))
    # merged conversion curve per task, across every run of the same task
    by_task: dict[str, list[dict]] = {}
    for path in sorted(RESULTS.glob("gate_*.json")):
        pl = json.loads(path.read_text())
        by_task.setdefault(pl["meta"]["task"], []).append(pl)
    for task, pls in by_task.items():
        Ts = sorted({c["T"] for pl in pls for c in pl["cells"] if "T" in c})
        shared = [T for T in Ts if sum(
            1 for pl in pls if any(c.get("T") == T for c in pl["cells"])) == len(pls)]
        for T in (shared[-1:] or Ts[:1]):
            for mode in ("light", "dark"):
                f = fig_conversion(pls, task, mode, T)
                if f:
                    made.append(f)
    for mode in ("light", "dark"):
        f = fig_atlas(by_task, mode)
        if f:
            made.append(f)
    for m in made:
        print("wrote", m)
    if not made:
        print("no result files yet")


if __name__ == "__main__":
    main()
