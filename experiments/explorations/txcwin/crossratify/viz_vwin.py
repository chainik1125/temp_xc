"""V-win visible-cue decomposition figure (CROSSRATIFY § G-2 / R-X2).

One figure, two panels (gpt2, llama31-8B): every visible-cue arm at the
claims' T=8 on nov_resid, against the two dictionary references. Encodes
the ruling-licensed reading (LOG 56654864d item 3 + 46e0021a7): the
window-computable floor V-win is the operative comparator; V-pos / V-all
carry the oracle-position caveat and are marked; the T=16 nuance rides in
the footer, quoted in the licensed dictionary-vs-V-all-at-that-T form.

Cue-arm skills + CIs are read from the committed artifacts
(`results/visible_cue_{gpt2,llama31}.json`); the two dictionary rows are
the CROSSRATIFY.md § G-2 table constants (their provenance spans the
focus/panel artifacts and is ratified in the doc).

Run:  .venv/bin/python -m experiments.explorations.txcwin.crossratify.viz_vwin
Writes figs/vwin_decomposition_{light,dark}.png (house viz_focus theme).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGS = HERE.parent / "figs"

# House themes (viz_focus.TH) + this figure's roles: one accent
# (TXC-post keeps its house series color), everything else labeled
# neutrals — per-token baselines are neutral in the house money-fig
# grammar, and the single-accent design is what passes CVD checks on
# the dark surface (blue+purple adjacent fails; validated 2026-07-27).
TH = {
    "light": {"bg": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e4e6e8", "null": "#9aa0a6", "accent": "#4a3aa7",
              "floor_hi": "#52514e", "floor": "#9aa0a6", "oracle": "#c3c7cb",
              "dict": "#6b7075"},
    "dark": {"bg": "#111a21", "ink": "#e8eef1", "ink2": "#a7bac4",
             "grid": "#22323d", "null": "#778d99", "accent": "#9085e9",
             "floor_hi": "#a7bac4", "floor": "#778d99", "oracle": "#4a5b66",
             "dict": "#8fa3ad"},
}

T_CLAIM = 8
FIELD = "nov_resid"

# CROSSRATIFY.md § G-2 table (ratified constants; see module docstring).
DICT_ROWS = {
    "gpt2": {"best per-token dictionary": 0.215, "TXC-post @ T=8": 0.463},
    "llama31": {"best per-token dictionary": 0.129, "TXC-post @ T=8": 0.393},
}
PANELS = (("gpt2", "gpt2 (L6)"), ("llama31", "Llama-3.1-8B (L12)"))

ROW_ORDER = [  # bottom -> top in the horizontal chart
    "V-rep (window repetition)",
    "V-uni (token-identity prior)",
    "V-win — window-computable floor",
    "V-pos (document position) †",
    "V-all (incl. position) †",
    "best per-token dictionary",
    "TXC-post @ T=8",
]
ARM_KEY = {"V-rep": "V-rep (window repetition)",
           "V-uni (token-identity prior)": "V-uni (token-identity prior)",
           "V-uni": "V-uni (token-identity prior)",
           "V-win": "V-win — window-computable floor",
           "V-pos": "V-pos (document position) †",
           "V-all": "V-all (incl. position) †"}

FOOTER = (
    "Quoting guard (rulings 56654864d / 46e0021a7): gpt2 — floor < per-token dictionary with CI separation"
    " (+0.054 [−0.002, +0.104] vs +0.215); 8B — floor-vs-per-token NOT CI-separated, quotable form is"
    " TXC-post ≈ 4× the window-computable floor (+0.393 vs +0.097).\n"
    "† not computable from window tokens (oracle position); drives the 8B band-2 call by the card's letter."
    "  T=16 nuance (licensed as dictionary-vs-V-all-at-that-T): 8B post +0.507 ≈ 2× V-all +0.247;"
    " gpt2 post +0.417 ≈ 2× V-all +0.212."
)


def _style(t):
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"], "text.color": t["ink"],
        "axes.labelcolor": t["ink"], "axes.edgecolor": t["grid"],
        "xtick.color": t["ink2"], "ytick.color": t["ink"],
        "grid.color": t["grid"], "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False})


def _cells(model):
    d = json.loads((RESULTS / f"visible_cue_{model}.json").read_text())
    out = {}
    for c in d["cells"]:
        if c["field"] == FIELD and c["T"] == T_CLAIM and c["arm"] in ARM_KEY:
            out[ARM_KEY[c["arm"]]] = c
    return out


def render(mode: str) -> Path:
    t = TH[mode]
    _style(t)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), dpi=170, sharex=True)

    for ax, (model, title) in zip(axes, PANELS):
        cue = _cells(model)
        ys, vals, colors, cis = [], [], [], []
        for i, row in enumerate(ROW_ORDER):
            ys.append(i)
            if row in DICT_ROWS[model]:
                vals.append(DICT_ROWS[model][row])
                colors.append(t["accent"] if row.startswith("TXC-post") else t["dict"])
                cis.append(None)
            else:
                c = cue[row]
                vals.append(c["skill"])
                colors.append(t["floor_hi"] if row.startswith("V-win")
                              else t["oracle"] if row.endswith("†") else t["floor"])
                cis.append((c["ci_lo"], c["ci_hi"]))
        ax.barh(ys, vals, height=0.62, color=colors, edgecolor=t["bg"], linewidth=1.2,
                zorder=3)
        for y, v, ci in zip(ys, vals, cis):
            if ci is not None:
                ax.plot(list(ci), [y, y], color=t["ink2"], linewidth=1.1, zorder=4)
        for y, v, ci in zip(ys, vals, cis):
            anchor = max(v, 0.0, (ci[1] if ci is not None else v))
            ax.annotate(f"{v:+.3f}", xy=(anchor, y), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8.5,
                        color=t["ink"], zorder=5)
        ax.axvline(0, color=t["grid"], linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(ROW_ORDER)))
        ax.set_yticklabels(ROW_ORDER if ax is axes[0] else [""] * len(ROW_ORDER),
                           fontsize=9)
        if ax is not axes[0]:
            ax.tick_params(axis="y", length=0)
        for lbl in (ax.get_yticklabels() if ax is axes[0] else []):
            if lbl.get_text().startswith("V-win"):
                lbl.set_fontweight("bold")
            if lbl.get_text().startswith(("TXC-post",)):
                lbl.set_color(t["accent"])
                lbl.set_fontweight("bold")
        ax.set_title(title)
        ax.grid(axis="x", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("probe skill on nov_resid at T = 8")
        ax.set_xlim(-0.08, 0.55)

    fig.suptitle("Visible-cue decomposition at the claims' T=8 — the window-computable "
                 "floor vs the dictionaries", y=1.005, fontsize=12)
    fig.text(0.01, -0.075, FOOTER, fontsize=7.4, color=t["ink2"], va="top")
    fig.tight_layout()
    out = FIGS / f"vwin_decomposition_{mode}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    for mode in ("light", "dark"):
        print(render(mode))
