"""Architectural block diagrams for T-SAE vs TXC.

Reads the same conventions as build_figures.py — emits PNG block
diagrams of the two forward passes with tensor shapes annotated.
Run with: uv run safety_research/scripts/tempbench/build_arch_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "tempbench"
OUT.mkdir(parents=True, exist_ok=True)

# Palette
ENC_COLOR = "#a3c8e0"
DEC_COLOR = "#f4a3a3"
LATENT_COLOR_TSAE = "#b6e3b6"
LATENT_COLOR_TXC = "#d6c0e3"
TOPK_COLOR = "#ffd97a"
INPUT_COLOR = "#e8e8e8"


def _box(ax, x, y, w, h, text, color, fontsize=9, edgecolor="black", lw=1.2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=lw, edgecolor=edgecolor, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize)


def _arrow(ax, x1, y1, x2, y2, label=None, color="black", lw=1.4, dotted=False):
    style = "->,head_length=8,head_width=6"
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color, linewidth=lw,
        linestyle=":" if dotted else "-",
    )
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label,
                ha="center", va="bottom", fontsize=8, color=color,
                fontstyle="italic")


def fig_tsae_diagram() -> None:
    """T-SAE — T independent TopK SAEs, one per position."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    T = 5
    x_positions = list(range(T))

    # Title
    ax.text(2, 6.2,
            "T-SAE (Stacked SAE) — $T$ independent TopK SAEs, one per position\n"
            r"per-position L0 = $k$;  window-level L0 = $kT$",
            fontsize=12, fontweight="bold", ha="left")

    for t in x_positions:
        x0 = 0.5 + t * 2.4

        # Input slice x[:, t, :]
        _box(ax, x0, 4.7, 1.6, 0.55, fr"$x_{{:,t={t},:}}$  ($B,d$)",
             INPUT_COLOR, fontsize=9)

        # Encoder
        _box(ax, x0, 3.85, 1.6, 0.55,
             fr"$W_{{\mathrm{{enc}}}}^{{({t})}}$  ($h, d$)",
             ENC_COLOR, fontsize=9)

        # Pre-act
        _box(ax, x0, 3.0, 1.6, 0.5, fr"pre$^{{({t})}}$  ($B,h$)",
             "#fafafa", fontsize=8)

        # TopK
        _box(ax, x0, 2.15, 1.6, 0.55, r"TopK$_k$ ∘ ReLU",
             TOPK_COLOR, fontsize=8)

        # Latent u^(t)
        _box(ax, x0, 1.30, 1.6, 0.55, fr"$u^{{({t})}}$  ($B,h$); $k$ nonzeros",
             LATENT_COLOR_TSAE, fontsize=8)

        # Decoder
        _box(ax, x0, 0.45, 1.6, 0.55,
             fr"$W_{{\mathrm{{dec}}}}^{{({t})}}$  ($d, h$)",
             DEC_COLOR, fontsize=9)

        # Output
        _box(ax, x0, -0.40, 1.6, 0.55, fr"$\hat{{x}}_{{:,t={t},:}}$",
             "#e0e0e0", fontsize=9)

        # arrows
        for y_top, y_bot in [(4.7, 4.40), (3.85, 3.50), (3.00, 2.70), (2.15, 1.85), (1.30, 1.00), (0.45, 0.15)]:
            _arrow(ax, x0 + 0.8, y_top, x0 + 0.8, y_bot)

    # Annotation: independence
    ax.annotate("", xy=(0.5 + 4 * 2.4 + 0.8, 1.55), xytext=(0.5 + 0.8, 1.55),
                arrowprops=dict(arrowstyle="<|-|>", color="grey", lw=1,
                                connectionstyle="arc3,rad=0.06"))
    ax.text(0.5 + 2 * 2.4 + 0.8, 1.85,
            "no cross-position weight sharing — $T$ independent dictionaries",
            ha="center", fontsize=9, color="grey", fontstyle="italic")

    # Loss
    ax.text(6.5, -1.25,
            r"$\mathcal{L}_{\mathrm{T-SAE}}(x) = \dfrac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}_B\left[\,\sum_{i=1}^{d}(\hat{x}_{:,t,i} - x_{:,t,i})^2\,\right]$",
            ha="center", fontsize=12, color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e0", ec="#888888"))

    ax.set_xlim(-0.2, 13)
    ax.set_ylim(-1.7, 6.7)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "arch_tsae.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_txc_diagram() -> None:
    """TXC — shared latent z fed by all T positions, decoded back to all T."""
    fig, ax = plt.subplots(figsize=(13, 7))
    T = 5

    # Title
    ax.text(2, 6.6,
            "TXC (Temporal Crosscoder) — single shared latent $z$ across the length-$T$ window\n"
            r"window-level L0 = $kT$ (matches T-SAE)",
            fontsize=12, fontweight="bold", ha="left")

    # Input row of T positions
    for t in range(T):
        x0 = 0.5 + t * 2.4
        _box(ax, x0, 5.6, 1.6, 0.55, fr"$x_{{:,t={t},:}}$",
             INPUT_COLOR, fontsize=9)

    # Per-position encoder slabs feeding into a single sum
    for t in range(T):
        x0 = 0.5 + t * 2.4
        _box(ax, x0, 4.65, 1.6, 0.55,
             fr"$W_{{\mathrm{{enc}}}}^{{({t})}}$",
             ENC_COLOR, fontsize=9)
        _arrow(ax, x0 + 0.8, 5.6, x0 + 0.8, 5.20)

    # Sum-and-bias node
    sum_x, sum_y = 6.3, 3.85
    _box(ax, sum_x - 1.5, sum_y, 3.0, 0.55,
         r"$\Sigma_t\, x_{:,t,:} W_{\mathrm{enc}}^{(t)} \; + \; b_{\mathrm{enc}}$"
         r"  →  pre  ($B,h$)",
         "#fafafa", fontsize=10)
    for t in range(T):
        x0 = 0.5 + t * 2.4
        _arrow(ax, x0 + 0.8, 4.65, sum_x, sum_y + 0.55, color="#444444")

    # Single TopK
    _box(ax, sum_x - 1.0, 3.05, 2.0, 0.55, r"TopK$_{kT}$ ∘ ReLU",
         TOPK_COLOR, fontsize=10)
    _arrow(ax, sum_x + 0.5, 3.85, sum_x + 0.5, 3.60)

    # Latent z
    _box(ax, sum_x - 1.0, 2.20, 2.0, 0.55,
         r"$z$  ($B,h$); $kT$ nonzeros",
         LATENT_COLOR_TXC, fontsize=10)
    _arrow(ax, sum_x + 0.5, 3.05, sum_x + 0.5, 2.75)

    # z fans out to all T decoders
    for t in range(T):
        x0 = 0.5 + t * 2.4
        _box(ax, x0, 1.20, 1.6, 0.55,
             fr"$W_{{\mathrm{{dec}}}}^{{({t})}}$",
             DEC_COLOR, fontsize=9)
        _arrow(ax, sum_x + 0.5, 2.20, x0 + 0.8, 1.75, color="#444444")

        _box(ax, x0, 0.30, 1.6, 0.55, fr"$\hat{{x}}_{{:,t={t},:}}$",
             "#e0e0e0", fontsize=9)
        _arrow(ax, x0 + 0.8, 1.20, x0 + 0.8, 0.85)

    # Annotation
    ax.text(6.5, 1.85,
            "one $z$ writes to every position via a different "
            "$W_{\\mathrm{dec}}^{(t)}$ slice",
            ha="center", fontsize=9, color="grey", fontstyle="italic")

    # Loss
    ax.text(6.5, -0.55,
            r"$\mathcal{L}_{\mathrm{TXC}}(x) = \mathbb{E}_B\left[\,\dfrac{1}{T}\sum_{t=0}^{T-1}\sum_{i=1}^{d}(\hat{x}_{:,t,i} - x_{:,t,i})^2\,\right]$",
            ha="center", fontsize=12, color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e0", ec="#888888"))

    ax.set_xlim(-0.2, 13)
    ax.set_ylim(-1.0, 7.2)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "arch_txc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_side_by_side() -> None:
    """Compact two-panel comparison highlighting the structural difference."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))

    # ── Panel 1: T-SAE ──
    ax = axs[0]
    ax.set_title("T-SAE — $T$ independent TopK SAEs",
                 fontsize=12, fontweight="bold")
    T = 5
    for t in range(T):
        x0 = 0.1 + t * 1.5
        _box(ax, x0, 4.4, 1.2, 0.5, fr"$x_{t}$", INPUT_COLOR, fontsize=8)
        _box(ax, x0, 3.55, 1.2, 0.5, fr"$W^{{({t})}}_{{\mathrm{{enc}}}}$",
             ENC_COLOR, fontsize=8)
        _box(ax, x0, 2.60, 1.2, 0.5, r"TopK$_k$", TOPK_COLOR, fontsize=8)
        _box(ax, x0, 1.65, 1.2, 0.55, fr"$u^{{({t})}}$",
             LATENT_COLOR_TSAE, fontsize=8)
        _box(ax, x0, 0.70, 1.2, 0.5, fr"$W^{{({t})}}_{{\mathrm{{dec}}}}$",
             DEC_COLOR, fontsize=8)
        _box(ax, x0, -0.20, 1.2, 0.5, fr"$\hat{{x}}_{t}$",
             "#e0e0e0", fontsize=8)
        for y_top, y_bot in [(4.4, 4.05), (3.55, 3.10), (2.60, 2.20), (1.65, 1.20), (0.70, 0.30)]:
            _arrow(ax, x0 + 0.6, y_top, x0 + 0.6, y_bot, lw=1)

    ax.text(0.5 + (T-1) / 2 * 1.5 + 0.6, 1.95,
            f"{T} parallel TopK pickers; one $u^{{(t)}}$ per position",
            ha="center", fontsize=9, color="grey", fontstyle="italic")

    ax.text(0.1 + (T-1) / 2 * 1.5 + 0.6, -1.10,
            r"$\mathcal{L} = \dfrac{1}{T}\sum_t\,\|\hat{x}_t-x_t\|_2^2$",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff8e0", ec="#888888"))
    ax.set_xlim(-0.2, 7.7)
    ax.set_ylim(-1.5, 5.4)
    ax.axis("off")

    # ── Panel 2: TXC ──
    ax = axs[1]
    ax.set_title("TXC — single shared latent $z$",
                 fontsize=12, fontweight="bold")
    for t in range(T):
        x0 = 0.1 + t * 1.5
        _box(ax, x0, 4.4, 1.2, 0.5, fr"$x_{t}$", INPUT_COLOR, fontsize=8)
        _box(ax, x0, 3.55, 1.2, 0.5, fr"$W^{{({t})}}_{{\mathrm{{enc}}}}$",
             ENC_COLOR, fontsize=8)
        _arrow(ax, x0 + 0.6, 4.4, x0 + 0.6, 4.05, lw=1)

    # Single TopK + z
    sx = 0.1 + 2 * 1.5 + 0.6  # centre of middle column
    _box(ax, sx - 1.4, 2.55, 2.8, 0.5,
         r"$\Sigma$  ➜  TopK$_{kT}$  ∘ ReLU",
         TOPK_COLOR, fontsize=10)
    _box(ax, sx - 1.4, 1.65, 2.8, 0.55,
         r"$z$ ($B,h$);  $kT$ nonzeros",
         LATENT_COLOR_TXC, fontsize=10)

    # encoder-to-sum fan-in
    for t in range(T):
        x0 = 0.1 + t * 1.5
        _arrow(ax, x0 + 0.6, 3.55, sx, 3.05, color="#444444", lw=1)
    _arrow(ax, sx, 2.55, sx, 2.20, lw=1)

    # decoder fan-out
    for t in range(T):
        x0 = 0.1 + t * 1.5
        _box(ax, x0, 0.70, 1.2, 0.5,
             fr"$W^{{({t})}}_{{\mathrm{{dec}}}}$",
             DEC_COLOR, fontsize=8)
        _arrow(ax, sx, 1.65, x0 + 0.6, 1.20, color="#444444", lw=1)
        _box(ax, x0, -0.20, 1.2, 0.5, fr"$\hat{{x}}_{t}$",
             "#e0e0e0", fontsize=8)
        _arrow(ax, x0 + 0.6, 0.70, x0 + 0.6, 0.30, lw=1)

    ax.text(sx, 1.30,
            "single TopK; same $z$ writes to every position",
            ha="center", fontsize=9, color="grey", fontstyle="italic")

    ax.text(0.1 + (T-1) / 2 * 1.5 + 0.6, -1.10,
            r"$\mathcal{L} = \dfrac{1}{T}\sum_t\,\|\hat{x}_t(z)-x_t\|_2^2$",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff8e0", ec="#888888"))
    ax.set_xlim(-0.2, 7.7)
    ax.set_ylim(-1.5, 5.4)
    ax.axis("off")

    # Shared legend at top
    legend_handles = [
        mpatches.Patch(color=INPUT_COLOR, label="input / output"),
        mpatches.Patch(color=ENC_COLOR, label="encoder weights"),
        mpatches.Patch(color=TOPK_COLOR, label="TopK ∘ ReLU"),
        mpatches.Patch(color=LATENT_COLOR_TSAE, label="T-SAE latent $u^{(t)}$"),
        mpatches.Patch(color=LATENT_COLOR_TXC, label="TXC latent $z$"),
        mpatches.Patch(color=DEC_COLOR, label="decoder weights"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=6, bbox_to_anchor=(0.5, 0.05), fontsize=10, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT / "arch_side_by_side.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_decoder_atom_geometry() -> None:
    """Single decoder atom: T-SAE = position-localised; TXC = window-spanning."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    T = 5
    h = 8

    # T-SAE: one chosen feature j fires only at one position
    ax = axs[0]
    ax.set_title("T-SAE decoder atom — feature $j$ at position $t^\\star$",
                 fontsize=12, fontweight="bold")
    for t in range(T):
        for j in range(h):
            colour = DEC_COLOR if (t == 2 and j == 3) else "#f0f0f0"
            ax.add_patch(plt.Rectangle((t, j), 0.95, 0.95,
                                        facecolor=colour, edgecolor="#888"))
    ax.text(2.5, 3.4, r"$W^{(t^\star)}_{\mathrm{dec},:,j}$",
            ha="center", fontsize=11, color="#990000")
    ax.set_xlim(-0.5, T + 0.5); ax.set_ylim(-0.5, h + 0.5)
    ax.set_xlabel("position $t$")
    ax.set_ylabel("feature index $j$")
    ax.set_xticks(range(T)); ax.set_yticks(range(h))
    ax.text(T / 2, -1.2,
            "ablating one (j, t*) entry only affects position $t^\\star$",
            ha="center", fontsize=9, color="grey", fontstyle="italic")

    # TXC: one chosen feature j fires across all positions
    ax = axs[1]
    ax.set_title("TXC decoder atom — feature $j$ writes at all $T$ positions",
                 fontsize=12, fontweight="bold")
    for t in range(T):
        for j in range(h):
            colour = "#f0f0f0"
            if j == 3:
                colour = "#f4a3a3"
            ax.add_patch(plt.Rectangle((t, j), 0.95, 0.95,
                                        facecolor=colour, edgecolor="#888"))
    ax.text(T / 2, 3.4, r"$W_{\mathrm{dec},j,:,:}$  spans all $T$ positions",
            ha="center", fontsize=11, color="#990000")
    ax.set_xlim(-0.5, T + 0.5); ax.set_ylim(-0.5, h + 0.5)
    ax.set_xlabel("position $t$")
    ax.set_ylabel("feature index $j$")
    ax.set_xticks(range(T)); ax.set_yticks(range(h))
    ax.text(T / 2, -1.2,
            "ablating one $j$ removes mass from every position simultaneously",
            ha="center", fontsize=9, color="grey", fontstyle="italic")

    plt.tight_layout()
    plt.savefig(OUT / "arch_decoder_atoms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_tsae_diagram()
    fig_txc_diagram()
    fig_side_by_side()
    fig_decoder_atom_geometry()
    print(f"Wrote architecture diagrams to {OUT}")
    for p in sorted(OUT.glob("arch_*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
