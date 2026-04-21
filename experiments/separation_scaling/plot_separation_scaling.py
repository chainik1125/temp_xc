"""Plot separation_scaling results, split into single-latent vs all-latent panels.

Two panels sharing a common δ axis and R² scale:

  LEFT  — single-latent metric: for each arch, the best single SAE latent's R²
          at predicting each component, averaged across components. This is the
          "can I point to one feature?" interpretability metric.
  RIGHT — all-latents metric: a linear probe trained on *all* SAE latents
          simultaneously, per component, averaged. Plus dense-residual baselines
          (single-position, windowed, MLP) which have no sparse decomposition.

Both panels include the Bayes-optimal R² ceiling (mean-over-positions) as a
red dash-dot line.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ARCH_ORDER = [
    "TopK SAE",
    "TXC",
    "MatryoshkaTXC",
    "MultiLayerCrosscoder",
    "TFA",
    "TFA-pos",
    "Temporal BatchTopK SAE",
]
ARCH_DISPLAY = [
    "TopK SAE",
    "TXC",
    "MatTXC",
    "MLxC",
    "TFA",
    "TFA-pos",
    "T-SAE",
]
ARCH_COLORS = [
    "#4c78a8",  # TopK
    "#f58518",  # TXC
    "#54a24b",  # MatTXC
    "#b279a2",  # MLxC
    "#72b7b2",  # TFA
    "#eeca3b",  # TFA-pos
    "#e45756",  # T-SAE
]

TAU_BY_DELTA = {0.0: 0.0, 0.05: 0.116, 0.1: 0.374, 0.15: 0.560, 0.2: 0.601}
# Bayes-optimal R² ceiling, mean over positions t ∈ [0, T-1] (T=128).
R2_CEILING_MEAN_BY_DELTA = {0.0: 0.0, 0.05: 0.066, 0.1: 0.215, 0.15: 0.360, 0.2: 0.446}


def _load_cells(results_root: Path) -> list[dict]:
    cells = []
    for p in sorted(results_root.glob("cell_delta_*/results.json")):
        cells.append(json.loads(p.read_text()))
    cells.sort(key=lambda c: float(c["sweep_value"]))
    return cells


def main() -> None:
    cells = _load_cells(ROOT / "results")
    if not cells:
        raise SystemExit(f"no cell results under {ROOT / 'results'}")
    deltas = np.array([float(c["sweep_value"]) for c in cells])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Bayes R² ceiling (mean over positions) — same on both panels.
    r2_max_arr = np.array([R2_CEILING_MEAN_BY_DELTA.get(float(d), np.nan) for d in deltas])
    for ax in (ax_l, ax_r):
        ax.plot(deltas, r2_max_arr, marker="^", color="#d62728", linewidth=2.2,
                linestyle="-.", alpha=0.95, label=r"Bayes $R^2_{\max}$ (mean over t)")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.set_xlabel(r"component separation δ   (x = 0.25 ± δ,  a = 0.6 ∓ δ)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # ---------------- LEFT: single-latent per arch ----------------
    for arch, disp, color in zip(ARCH_ORDER, ARCH_DISPLAY, ARCH_COLORS):
        if not all(arch in c["architectures"] for c in cells):
            continue
        pc = np.array([c["architectures"][arch]["per_component_best_r2"] for c in cells])
        mean = pc.mean(axis=1)
        ax_l.plot(deltas, mean, marker="o", color=color, linewidth=2.0, label=disp)

    ax_l.set_ylabel(r"mean per-component $R^2$")
    ax_l.legend(fontsize=9, loc="upper left", ncol=2)
    ax_l.set_title("Best single latent per arch\n"
                   r"(for each component $c$: max over features of $R^2(z_f \to \mathrm{onehot}_c)$)")

    # ---------------- RIGHT: all-latents / dense probes ----------------
    # Dense single-position linear
    if all(c.get("dense_linear") for c in cells):
        lin_mean = np.array([np.array(c["dense_linear"]["per_component_r2"]).mean() for c in cells])
        ax_r.plot(deltas, lin_mean, marker="s", color="black", linewidth=1.6,
                  linestyle="--", label="dense linear (single-pos)")

    # Dense windowed linear (W=5 by default)
    win_means = []
    for c in cells:
        d = float(c["sweep_value"])
        wp = ROOT / "results" / f"cell_delta_{d:g}" / "window_probes_w5.json"
        if wp.exists():
            payload = json.loads(wp.read_text())
            win_means.append(float(payload["window"]["mean_r2"]))
        else:
            win_means.append(np.nan)
    if not np.isnan(win_means).all():
        ax_r.plot(deltas, np.array(win_means), marker="X", color="#444444", linewidth=1.6,
                  linestyle="-", label="dense linear (window=5)")

    # Dense MLP probe
    if all(c.get("dense_mlp") for c in cells):
        mlp_mean = np.array([np.array(c["dense_mlp"]["per_component_r2"]).mean() for c in cells])
        ax_r.plot(deltas, mlp_mean, marker="D", color="#888888", linewidth=1.6,
                  linestyle=":", label="dense MLP (early-stopped)")

    # Linear probe on each arch's full latent vector
    for arch, disp, color in zip(ARCH_ORDER, ARCH_DISPLAY, ARCH_COLORS):
        if not all(arch in c["architectures"] and
                   c["architectures"][arch].get("linear_per_component_r2")
                   for c in cells):
            continue
        per_c = np.array([c["architectures"][arch]["linear_per_component_r2"] for c in cells])
        lin_mean = per_c.mean(axis=1)
        ax_r.plot(deltas, lin_mean, marker="o", color=color, linewidth=2.0, label=disp)

    ax_r.legend(fontsize=9, loc="upper left", ncol=2)
    ax_r.set_title("Linear probe on all latents\n"
                   r"(ridge regression $W \in \mathbb{R}^{d_{\mathrm{sae}} \times 3}$ fit to onehot)")

    fig.suptitle(
        "Component-identity recovery vs δ — single-latent vs all-latents readouts "
        "(r=0, σ=1e-3, d_model=64, ctx=128, 20k steps)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "plots" / "separation_scaling.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Markdown table — show both metrics per arch per δ
    out_md = ROOT / "tables" / "separation_scaling.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    hdr = ("| δ | τ | R²_max | tf_loss | dense-lin | dense-MLP | "
           + " | ".join(f"{d} best/all" for d in ARCH_DISPLAY) + " |")
    sep = "| " + " | ".join("---:" for _ in range(6 + len(ARCH_DISPLAY))) + " |"
    lines = [hdr, sep]
    for c in cells:
        d = float(c["sweep_value"])
        tau = TAU_BY_DELTA.get(d, float("nan"))
        r2max = R2_CEILING_MEAN_BY_DELTA.get(d, float("nan"))
        tf_loss = c["transformer"]["final_loss"]
        lin = c.get("dense_linear", {}).get("mean_r2", float("nan"))
        mlp = c.get("dense_mlp", {}).get("mean_r2", float("nan"))
        arch_cells = []
        for arch in ARCH_ORDER:
            a = c["architectures"].get(arch)
            if a is None:
                arch_cells.append("-")
                continue
            best = np.array(a["per_component_best_r2"]).mean()
            all_ = a.get("linear_mean_r2", float("nan"))
            arch_cells.append(f"{best:.3f}/{all_:.3f}")
        lines.append(
            f"| {d:.2f} | {tau:.2f} | {r2max:.2f} | {tf_loss:.3f} | "
            f"{lin:+.3f} | {mlp:+.3f} | " + " | ".join(arch_cells) + " |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
