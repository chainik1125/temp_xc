"""Early-layer addendum figures — g_order(ℓ) for lag4, g_agg(ℓ) for slope8.

Reads results/depth.json (produced by run_depth.py under the frozen
PREDICTIONS.md) and draws the two depth curves the addendum asks for:

  figs/depth_lag4_gorder.{png,pdf}  — per replag model: g_order = win −
      mean at T ∈ {4, 8} vs capture depth (fraction of model layers),
      plus per-token acc (the conversion axis) on a twin scale.
  figs/depth_slope8_gagg.{png,pdf}  — per Ward reader: tok(ℓ) and
      mean64(ℓ) across the 17 capture points; g_agg = mean − tok shaded.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.depth_addendum.render
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "depth.json"
FIGS = HERE / "figs"

N_LAYERS = {"gpt2": 12, "gemma2_2b": 26, "llama31_8b": 32}
MODEL_COL = {"gpt2": "#d62728", "gemma2_2b": "#2ca02c",
             "llama31_8b": "#1f77b4"}
READER_COL = {"distill": "#1f77b4", "base": "#7f7f7f"}


def main() -> None:
    cells = json.loads(RES.read_text())["cells"]
    FIGS.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def acc(key):
        c = cells.get(key)
        return c["acc_test"] if c else None

    # ---- lag4 g_order(ℓ) ----
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for T, ax in zip((4, 8), axes):
        for model, nl in N_LAYERS.items():
            hss, gords, toks = [], [], []
            for hs in range(0, 40):
                w = acc(f"replag/{model}/hs{hs}/T{T}/win_linear")
                m = acc(f"replag/{model}/hs{hs}/T{T}/win_mean_linear")
                if w is None or m is None:
                    continue
                hss.append(hs / nl)
                gords.append(w - m)
                toks.append(acc(f"replag/{model}/hs{hs}/tok_linear"))
            ax.plot(hss, gords, "o-", color=MODEL_COL[model], lw=2,
                    label=f"{model} g_order")
            ax.plot(hss, [t - 0.25 for t in toks], "s:", color=MODEL_COL[model],
                    lw=1.2, alpha=0.6, mfc="none",
                    label=f"{model} tok − chance")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"lag4, T = {T}")
        ax.set_xlabel("capture depth (hs / n_layers)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("acc difference (4-class, chance 0.25)")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Early-layer addendum — lag4 order signal vs depth "
                 "(solid: g_order = win − mean; dotted: per-token − chance)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"depth_lag4_gorder.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    # ---- slope8 g_agg(ℓ) ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for reader, col in READER_COL.items():
        hss, toks, means = [], [], []
        for hs in [0] + list(range(1, 32, 2)):
            t = acc(f"slope8/{reader}/hs{hs}/tok_linear")
            m = acc(f"slope8/{reader}/hs{hs}/T64/win_mean_linear")
            if t is None or m is None:
                continue
            hss.append(hs)
            toks.append(t)
            means.append(m)
        ax.plot(hss, means, "o-", color=col, lw=2, label=f"{reader} mean64")
        ax.plot(hss, toks, "s:", color=col, lw=1.2, mfc="none",
                label=f"{reader} tok")
        ax.fill_between(hss, toks, means, color=col, alpha=0.12)
    ax.axhline(1 / 3, color="k", lw=0.8, ls="--", label="chance")
    ax.set_xlabel("capture point hs (0 = embeddings, odd = resid_post L(hs−1))")
    ax.set_ylabel("slope8 acc (3-class)")
    ax.set_title("Early-layer addendum — slope8 aggregation gap vs depth\n"
                 "(shaded band = g_agg = mean64 − tok; matched rows)",
                 fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"depth_slope8_gagg.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {FIGS}/depth_lag4_gorder.* ; depth_slope8_gagg.*")


if __name__ == "__main__":
    main()
