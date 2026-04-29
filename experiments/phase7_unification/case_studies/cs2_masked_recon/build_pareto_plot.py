"""CS2 final figure — Pareto plot of held-out reconstruction quality.

Combines:
  - MSE-FVE from controls_masked_recon.json (higher = better
    reconstruction-MSE)
  - downstream LM ΔCE from behavioral_masked_recon.json (lower = more
    on-manifold reconstruction; LM behaviour preserved better)

Each method becomes a point in (FVE, ΔCE) space. The Pareto frontier
is the upper-left envelope (high FVE, low ΔCE).

Output:
  results/case_studies/cs2_masked_recon/cs2_pareto.{png,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SUBDIR = Path(
    "experiments/phase7_unification/results/case_studies/cs2_masked_recon"
)
PALETTE = {
    "topk_sae": "#1f77b4",
    "tsae_paper_k20": "#d62728",
    "agentic_txc_02": "#2ca02c",
    "phase5b_subseq_h8": "#17becf",
    "zero": "#7f7f7f",
    "mean_residual": "#bcbd22",
    "x_t_minus_1": "#e377c2",
    "mean_context_T5": "#16a085",
}
LABEL = {
    "topk_sae": "TopKSAE k=500",
    "tsae_paper_k20": "T-SAE k=20",
    "agentic_txc_02": "TXC matryoshka T=5",
    "phase5b_subseq_h8": "SubseqH8 T=10",
    "zero": "predict-zero",
    "mean_residual": "predict-mean-residual",
    "x_t_minus_1": "predict-x[t-1]",
    "mean_context_T5": "predict-mean-context T=5",
}


def _load() -> tuple[dict, dict]:
    with (SUBDIR / "controls_masked_recon.json").open() as f:
        ctrl = json.load(f)
    with (SUBDIR / "behavioral_masked_recon.json").open() as f:
        beh = json.load(f)
    return ctrl, beh


def _gather_points(ctrl: dict, beh: dict) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    # SAE archs.
    for arch_id, arch_d in ctrl["archs"].items():
        if "skipped" in arch_d:
            continue
        # MSE side: width=1 holdout.
        ws = arch_d.get("width_sweep", {})
        ws1 = ws.get("1", ws.get(1, {}))
        if "fve" not in ws1:
            continue
        fve = ws1["fve"]
        # CE side: arch_id key in behavioural file.
        if arch_id not in beh:
            continue
        delta_ce = beh[arch_id]["delta_ce_mean"]
        points[arch_id] = (fve, delta_ce)
    # Naive baselines.
    naive = ctrl.get("naive_baselines", {})
    naive_to_ce = {
        "predict_zero": "zero",
        "predict_mean_residual": "mean_residual",
        "predict_x_t_minus_1": "x_t_minus_1",
        "predict_mean_context_T5": "mean_context_T5",
    }
    for naive_key, beh_key in naive_to_ce.items():
        if naive_key not in naive or beh_key not in beh:
            continue
        fve = naive[naive_key]["fve"]
        delta_ce = beh[beh_key]["delta_ce_mean"]
        points[beh_key] = (fve, delta_ce)
    return points


def _pareto_frontier(points: dict[str, tuple[float, float]]) -> list[str]:
    """Return method ids on the (high FVE, low ΔCE) Pareto frontier."""
    keys = list(points.keys())
    is_pareto = []
    for k in keys:
        f1, c1 = points[k]
        dominated = False
        for other in keys:
            if other == k:
                continue
            f2, c2 = points[other]
            if f2 >= f1 and c2 <= c1 and (f2 > f1 or c2 < c1):
                dominated = True
                break
        if not dominated:
            is_pareto.append(k)
    return is_pareto


def main() -> None:
    ctrl, beh = _load()
    points = _gather_points(ctrl, beh)
    pareto = _pareto_frontier(points)

    print("(FVE, ΔCE) per method — Pareto*: lower-right is better (high FVE, low ΔCE)")
    print(f"  {'method':<28}  {'FVE':>8}  {'ΔCE':>8}  Pareto?")
    for k, (f, c) in sorted(points.items(), key=lambda kv: -kv[1][0]):
        flag = "*" if k in pareto else ""
        print(f"  {LABEL.get(k, k):<28}  {f:>8.3f}  {c:>8.3f}  {flag}")
    print(f"\nPareto frontier: {[LABEL.get(k, k) for k in pareto]}")

    payload = {
        "points": {k: {"fve": f, "delta_ce": c} for k, (f, c) in points.items()},
        "pareto_frontier": pareto,
    }
    json_path = SUBDIR / "cs2_pareto.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {json_path}")

    _plot(points, pareto, SUBDIR / "cs2_pareto.png")


def _plot(points: dict, pareto: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    # Pareto envelope: connect Pareto points sorted by FVE.
    pareto_sorted = sorted(pareto, key=lambda k: -points[k][0])
    fs = [points[k][0] for k in pareto_sorted]
    cs = [points[k][1] for k in pareto_sorted]
    ax.plot(fs, cs, "-", color="#888", lw=1.5, alpha=0.7,
            label="Pareto frontier", zorder=1)

    # Per-method scatter.
    for k, (f, c) in points.items():
        marker = "*" if k in pareto else "o"
        size = 380 if k in pareto else 180
        ax.scatter([f], [c], s=size, color=PALETTE.get(k, "#888"),
                   marker=marker, edgecolor="black", lw=1.2, zorder=3)
        ax.annotate(LABEL.get(k, k), (f, c),
                    xytext=(8, -3), textcoords="offset points", fontsize=9)

    ax.set_xlabel("MSE-FVE on held-out reconstruction (higher = better)")
    ax.set_ylabel("downstream LM ΔCE at next token (lower = more on-manifold)")
    ax.set_title(
        "CS2 final — held-out-position imputation Pareto\n"
        "(30 FineWeb-edu × 16 random held-out positions = 480 datapoints)"
    )
    ax.grid(True, ls=":", alpha=0.4)
    # Annotate Pareto direction.
    ax.text(0.02, 0.02, "↓ ← better corner",
            transform=ax.transAxes, fontsize=10, color="#666",
            ha="left", va="bottom")
    ax.legend(loc="upper left")
    plt.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"wrote {out_path}  (+ thumb)")


if __name__ == "__main__":
    main()
