"""Plot gap-recovery: each method's fraction of the Bayes-optimal R² ceiling.

Ceiling: R²_max averaged over t ∈ [59, 127] — the window where MatTXC's codes
are defined (W_win = 60). This restricts the comparison to positions where
every method can in principle operate, so we're not penalising crosscoders for
early positions they don't cover.

Y axis: share of gap = (arch mean-per-component R²) / R²_max[59..127].
X axis: δ.

For the dense probes (single-pos linear / window linear / MLP) we use the
all-positions R² already computed by the driver. This is a small approximation
— strictly the dense probes' R² on [59, 127] would be ~0.02–0.05 higher at
informative cells (early positions have less signal). We flag this in the
caveat footnote. The SAE arches' `linear_mean_r2` use each arch's native valid
window range (MatTXC [59,127], TXC [29,127], others per-position), which
introduces the same mild bias in opposite directions.
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
    "#4c78a8", "#f58518", "#54a24b", "#b279a2",
    "#72b7b2", "#eeca3b", "#e45756",
]

MIN_T = 59  # MatTXC T_win = 60, so valid positions start at t=59


def _load_cells(results_root: Path) -> list[dict]:
    cells = []
    for p in sorted(results_root.glob("cell_delta_*/results.json")):
        cells.append(json.loads(p.read_text()))
    cells.sort(key=lambda c: float(c["sweep_value"]))
    return cells


def _r2_ceiling_restricted() -> dict[float, float]:
    """Load per-δ Bayes R² ceiling averaged over t ∈ [MIN_T, T-1]."""
    d = json.loads((ROOT / "r2_ceiling.json").read_text())
    out = {}
    for row in d["rows"]:
        r2_by_t = np.array(row["r2_by_t"])
        out[float(row["delta"])] = float(r2_by_t[MIN_T:].mean())
    return out


def _build_series(cells: list[dict], include_ridge: bool):
    """Build method_series list. If include_ridge=True, dense-window probes use
    the CV-tuned ridge from ridge_sweep.json; otherwise they use the λ≈0 fits
    from window_probes_wN.json."""
    series: list[tuple[str, np.ndarray, str, str]] = []

    # SAE arches — linear probe on all latents (λ≈0 ridge; same in both plots)
    for arch, disp, color in zip(ARCH_ORDER, ARCH_DISPLAY, ARCH_COLORS):
        if not all(arch in c["architectures"] and
                   c["architectures"][arch].get("linear_per_component_r2")
                   for c in cells):
            continue
        per_c = np.array([c["architectures"][arch]["linear_per_component_r2"] for c in cells])
        series.append((disp, per_c.mean(axis=1), color, "-"))

    # Dense single-position linear (λ≈0 by default, not affected by ridge flag)
    if all(c.get("dense_linear") for c in cells):
        lin = np.array([np.array(c["dense_linear"]["per_component_r2"]).mean() for c in cells])
        series.append(("dense linear (single-pos)", lin, "black", "--"))

    # Dense linear windowed probes
    window_colors = {5: "#555555", 20: "#222222", 60: "#000000"}
    window_styles_no_ridge = {5: "-", 20: "-", 60: ":"}
    window_colors_ridge = {5: "#555555", 20: "#116611", 60: "#006699"}
    if include_ridge:
        # Replace W=20 and W=60 dense-linear probes with CV-tuned versions.
        # Keep W=5 at λ≈0 since it's effectively unaffected (d=320 << N).
        for W in (5,):
            vals = []
            for c in cells:
                d = float(c["sweep_value"])
                wp = ROOT / "results" / f"cell_delta_{d:g}" / f"window_probes_w{W}.json"
                if wp.exists():
                    vals.append(float(json.loads(wp.read_text())["window"]["mean_r2"]))
                else:
                    vals.append(np.nan)
            if not np.isnan(vals).all():
                series.append((f"dense linear (W={W})", np.array(vals),
                               window_colors[W], "-"))
        for W in (20, 60):
            vals = []
            for c in cells:
                d = float(c["sweep_value"])
                rs = ROOT / "results" / f"cell_delta_{d:g}" / "ridge_sweep.json"
                if not rs.exists():
                    vals.append(np.nan); continue
                payload = json.loads(rs.read_text())
                per_ridge = payload["results"].get(str(W), {})
                if not per_ridge:
                    vals.append(np.nan); continue
                best = max(per_ridge.values(), key=lambda r: r["mean_r2"])
                vals.append(float(best["mean_r2"]))
            if not np.isnan(vals).all():
                series.append((f"dense linear (W={W}, ridge-CV)",
                               np.array(vals), window_colors_ridge[W], "-"))
    else:
        for W in (5, 20, 60):
            vals = []
            for c in cells:
                d = float(c["sweep_value"])
                wp = ROOT / "results" / f"cell_delta_{d:g}" / f"window_probes_w{W}.json"
                if wp.exists():
                    vals.append(float(json.loads(wp.read_text())["window"]["mean_r2"]))
                else:
                    vals.append(np.nan)
            if not np.isnan(vals).all():
                series.append((f"dense linear (W={W}, λ≈0)", np.array(vals),
                               window_colors[W], window_styles_no_ridge[W]))

    # Dense MLP (same in both plots)
    if all(c.get("dense_mlp") for c in cells):
        mlp = np.array([np.array(c["dense_mlp"]["per_component_r2"]).mean() for c in cells])
        series.append(("dense MLP (early-stopped)", mlp, "#888888", ":"))

    return series


def _make_plot(cells: list[dict], out_path: Path, *, include_ridge: bool, title_suffix: str):
    deltas = np.array([float(c["sweep_value"]) for c in cells])
    ceilings = _r2_ceiling_restricted()
    ceiling_arr = np.array([ceilings[float(d)] for d in deltas])
    EPS = 1e-3
    method_series = _build_series(cells, include_ridge=include_ridge)

    fig, (ax_r2, ax_gap) = plt.subplots(1, 2, figsize=(16, 6))
    for disp, y, color, style in method_series:
        ax_r2.plot(deltas, y, marker="o", color=color, linestyle=style, linewidth=2.0, label=disp)
    ax_r2.plot(deltas, ceiling_arr, marker="^", color="#d62728", linewidth=2.2,
               linestyle="-.", label=r"Bayes $R^2_{\max}$ (t∈[59,127])")
    ax_r2.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax_r2.set_xlabel(r"component separation δ")
    ax_r2.set_ylabel(r"mean per-component $R^2$  (all-latents readout)")
    ax_r2.set_ylim(-0.05, 0.65)
    ax_r2.grid(True, alpha=0.3)
    ax_r2.legend(fontsize=8, loc="upper left", ncol=2)
    ax_r2.set_title("Raw R² (all-latents readout)")

    for disp, y, color, style in method_series:
        share = np.where(ceiling_arr > EPS, np.clip(y, 0, None) / ceiling_arr, np.nan)
        ax_gap.plot(deltas, share, marker="o", color=color, linestyle=style,
                    linewidth=2.0, label=disp)
    ax_gap.axhline(1.0, color="#d62728", linewidth=1.6, linestyle="-.", alpha=0.9,
                   label="Bayes ceiling (= 1.0)")
    ax_gap.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax_gap.set_xlabel(r"component separation δ")
    ax_gap.set_ylabel(r"share of gap recovered  $R^2 / R^2_{\max}(t\in[59,127])$")
    ax_gap.set_ylim(-0.05, 1.15)
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(fontsize=8, loc="lower right", ncol=2)
    ax_gap.set_title(r"Gap recovery: share of Bayes $R^2_{\max}$")

    fig.suptitle(
        f"Gap recovery vs δ — all-latents linear probe {title_suffix}  "
        "(ceiling/positions fixed to t∈[59,127] where MatTXC is defined)",
        fontsize=11,
    )
    fig.text(
        0.5, 0.01,
        "Note: non-crosscoder SAE probes (TopK, TFA, TFA-pos, T-SAE, MLxC) use their native position ranges "
        "(typically t∈[0,127]); the R²_max ceiling uses t∈[59,127]. "
        "Early positions carry less component info, so those numbers are mildly pessimistic (≲0.02 R²).",
        ha="center", fontsize=8, style="italic", alpha=0.7,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Print table of shares
    print(f"\nShare of Bayes R²_max (t∈[59,127]) recovered [{title_suffix}]:")
    hdr = "| method                                  | " + " | ".join(f"δ={d:g}" for d in deltas) + " |"
    sep = "| ---                                      | " + " | ".join("---:" for _ in deltas) + " |"
    print(hdr); print(sep)
    for disp, y, _, _ in method_series:
        share = np.where(ceiling_arr > EPS, np.clip(y, 0, None) / ceiling_arr, np.nan)
        print(f"| {disp:<40s} | " + " | ".join(
            f"{'-' if np.isnan(s) else f'{s:.2f}'}" for s in share) + " |")


def main() -> None:
    cells = _load_cells(ROOT / "results")
    if not cells:
        raise SystemExit("no cells")

    # Original (no ridge on dense probes)
    _make_plot(
        cells,
        out_path=ROOT / "plots" / "gap_recovery_no_ridge.png",
        include_ridge=False,
        title_suffix="(dense probes use λ≈0 ridge — apparent crosscoder advantage)",
    )

    # With ridge-CV for W=20 and W=60 dense probes
    _make_plot(
        cells,
        out_path=ROOT / "plots" / "gap_recovery.png",
        include_ridge=True,
        title_suffix="(dense W=20/60 use CV-tuned ridge — crosscoder advantage collapses)",
    )


if __name__ == "__main__":
    main()
