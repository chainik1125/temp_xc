"""Plot probe performance vs window size W, per δ cell.

Reads `results/cell_delta_*/window_probes_w{W}.json` (both linear-regression R²
and logistic-regression log-loss + accuracy). Produces two figures:
  - probe_window_sweep.png      — linear R² curves + τ(δ) dotted ceilings + MatTXC linear-probe dash-dot
  - probe_window_sweep_logistic.png — logistic log-loss curves + H(C|X_T) dotted ceilings,
    accompanied by an accuracy panel.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).parent
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


TAU_BY_DELTA = {0.0: 0.0, 0.05: 0.116, 0.1: 0.374, 0.15: 0.560, 0.2: 0.601}
# Bayes-optimal R² ceiling (mean over positions 0..T-1) from compute_r2_ceiling.py.
# This is the correct upper bound for R² probes that average over all positions.
R2_CEILING_MEAN_BY_DELTA = {0.0: 0.0, 0.05: 0.066, 0.1: 0.215, 0.15: 0.360, 0.2: 0.446}
# R² ceiling at the final position (t=T-1) — what a position-aware probe could reach.
R2_CEILING_FINAL_BY_DELTA = {0.0: 0.0, 0.05: 0.122, 0.1: 0.357, 0.15: 0.508, 0.2: 0.539}
# H(C | X_T) in bits and nats (for the logistic-probe log-loss floor).
H_BITS_BY_DELTA = {0.0: 1.585, 0.05: 1.401, 0.1: 0.993, 0.15: 0.698, 0.2: 0.633}
H_NATS_BY_DELTA = {d: v * np.log(2) for d, v in H_BITS_BY_DELTA.items()}
DELTA_COLORS = {
    0.0: "#9e9e9e",
    0.05: "#4c78a8",
    0.1: "#f58518",
    0.15: "#54a24b",
    0.2: "#e45756",
}


def main() -> None:
    results_root = ROOT / "results"
    cells = {}
    for cell_dir in sorted(results_root.glob("cell_delta_*")):
        results_json = cell_dir / "results.json"
        if not results_json.exists():
            continue
        r = json.loads(results_json.read_text())
        delta = float(r["sweep_value"])
        # single-position baseline = W=1
        pts = {1: float(r["dense_linear"]["mean_r2"])}
        for wp in cell_dir.glob("window_probes_w*.json"):
            m = re.match(r"window_probes_w(\d+)\.json", wp.name)
            if not m:
                continue
            W = int(m.group(1))
            payload = json.loads(wp.read_text())
            pts[W] = float(payload["window"]["mean_r2"])
        cells[delta] = pts

    # Also fetch MatTXC's linear probe on latents from each cell's results.json
    mattxc_lin_mean = {}
    for cell_dir in sorted(results_root.glob("cell_delta_*")):
        results_json = cell_dir / "results.json"
        if not results_json.exists():
            continue
        r = json.loads(results_json.read_text())
        d = float(r["sweep_value"])
        arches = r.get("architectures", {})
        if "MatryoshkaTXC" in arches and arches["MatryoshkaTXC"].get("linear_per_component_r2"):
            per = arches["MatryoshkaTXC"]["linear_per_component_r2"]
            mattxc_lin_mean[d] = float(np.mean(per))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    for delta in sorted(cells):
        color = DELTA_COLORS.get(delta, "#000000")
        pts = cells[delta]
        Ws = sorted(pts.keys())
        R2s = [pts[W] for W in Ws]
        r2_max_mean = R2_CEILING_MEAN_BY_DELTA.get(delta, float("nan"))
        ax.plot(Ws, R2s, marker="o", color=color, linewidth=2.0,
                label=f"δ={delta:g}  (R²_max={r2_max_mean:.2f})")
        # Correct Bayes R² ceiling (mean over positions) as dotted horizontal
        ax.axhline(r2_max_mean, color=color, linewidth=1.0, linestyle=":", alpha=0.7)
        # MatTXC linear probe as a horizontal dash-dot at each δ
        if delta in mattxc_lin_mean:
            ax.axhline(mattxc_lin_mean[delta], color=color, linewidth=1.1,
                       linestyle="-.", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 30, 60])
    ax.set_xticklabels([1, 2, 5, 10, 20, 30, 60])
    ax.set_xlabel("probe window size W (log)")
    ax.set_ylabel("mean per-component $R^2$  (linear probe on stacked window)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Linear probe R² vs window size  "
                 "(dotted = Bayes R²_max mean-over-positions; dash-dot = MatTXC linear probe on latents)")
    fig.tight_layout()
    out = ROOT / "plots" / "probe_window_sweep.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Table (linear-R²)
    deltas = sorted(cells.keys())
    Ws = sorted({w for d in deltas for w in cells[d].keys()})
    hdr = "| δ | τ | " + " | ".join(f"W={w}" for w in Ws) + " |"
    sep = "| ---: | ---: | " + " | ".join("---:" for _ in Ws) + " |"
    lines = [hdr, sep]
    for d in deltas:
        tau = TAU_BY_DELTA.get(d, float("nan"))
        row = f"| {d:.2f} | {tau:.2f} | " + " | ".join(
            f"{cells[d].get(w, float('nan')):+.3f}" for w in Ws
        ) + " |"
        lines.append(row)
    out_md = ROOT / "tables" / "probe_window_sweep.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_md}")
    print("\n" + "\n".join(lines))

    # --------------------------------------------------------------------
    # Logistic-regression variant
    # --------------------------------------------------------------------
    logit_cells: dict[float, dict[int, dict]] = {}
    for cell_dir in sorted(results_root.glob("cell_delta_*")):
        delta = None
        rj = cell_dir / "results.json"
        if rj.exists():
            delta = float(json.loads(rj.read_text())["sweep_value"])
        for wp in cell_dir.glob("window_probes_w*.json"):
            m = re.match(r"window_probes_w(\d+)\.json", wp.name)
            if not m:
                continue
            W = int(m.group(1))
            payload = json.loads(wp.read_text())
            if delta is None:
                delta = float(payload["sweep_value"])
            lg = payload.get("window", {}).get("logistic")
            # W=1 single-pos is saved as `single_position.logistic` too
            if W == 1:
                lg = payload.get("single_position", {}).get("logistic", lg)
            if lg is None:
                continue
            logit_cells.setdefault(delta, {})[W] = lg

    if logit_cells:
        fig, (ax_ll, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.5))
        for delta in sorted(logit_cells):
            color = DELTA_COLORS.get(delta, "#000000")
            tau = TAU_BY_DELTA.get(delta, float("nan"))
            Ws_d = sorted(logit_cells[delta].keys())
            lls = [logit_cells[delta][W]["log_loss"] for W in Ws_d]
            accs = [logit_cells[delta][W]["accuracy"] for W in Ws_d]

            ax_ll.plot(Ws_d, lls, marker="o", color=color, linewidth=2.0,
                       label=f"δ={delta:g}  (τ={tau:.2f})")
            # Bayes floor = H(C | X_T) in nats
            h_nats = H_NATS_BY_DELTA.get(delta)
            if h_nats is not None:
                ax_ll.axhline(h_nats, color=color, linewidth=0.9, linestyle=":", alpha=0.6)

            ax_acc.plot(Ws_d, accs, marker="o", color=color, linewidth=2.0,
                        label=f"δ={delta:g}  (τ={tau:.2f})")
            # Chance = 1/3 on a 3-class balanced problem
        ax_acc.axhline(1 / 3, color="k", linewidth=0.8, linestyle="--", alpha=0.5, label="chance")

        for ax in (ax_ll, ax_acc):
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20, 30, 60])
            ax.set_xticklabels([1, 2, 5, 10, 20, 30, 60])
            ax.set_xlabel("probe window size W (log)")
            ax.grid(True, alpha=0.3, which="both")
        ax_ll.set_ylabel("test log-loss (nats)  — lower is better")
        ax_ll.set_title("Multinomial logistic probe log-loss  "
                        "(dotted = Bayes floor H(C|X$_T$) in nats)")
        ax_ll.legend(fontsize=9, loc="upper right")
        ax_ll.set_ylim(0, np.log(3) + 0.05)
        ax_acc.set_ylabel("test accuracy — higher is better")
        ax_acc.set_title("Logistic probe test accuracy (3-class)")
        ax_acc.set_ylim(0.25, 1.02)
        ax_acc.legend(fontsize=9, loc="upper left")

        fig.tight_layout()
        out = ROOT / "plots" / "probe_window_sweep_logistic.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

        # Logistic table
        Ws_all = sorted({W for d in logit_cells for W in logit_cells[d].keys()})
        hdr = "| δ | τ | H(C\\|X_T) nats | " + " | ".join(f"W={w} ll/acc" for w in Ws_all) + " |"
        sep = "| ---: | ---: | ---: | " + " | ".join("---:" for _ in Ws_all) + " |"
        lines = [hdr, sep]
        for d in sorted(logit_cells):
            h = H_NATS_BY_DELTA.get(d, float("nan"))
            row = f"| {d:.2f} | {TAU_BY_DELTA.get(d, float('nan')):.2f} | {h:.3f} | " + " | ".join(
                f"{logit_cells[d][W]['log_loss']:.3f}/{logit_cells[d][W]['accuracy']:.2f}"
                if W in logit_cells[d] else "-"
                for W in Ws_all
            ) + " |"
            lines.append(row)
        out_md = ROOT / "tables" / "probe_window_sweep_logistic.md"
        out_md.write_text("\n".join(lines) + "\n")
        print(f"Saved {out_md}")
        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
