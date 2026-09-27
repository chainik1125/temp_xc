"""Plotting for colored-source experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


_TXC_COLOR = "#1f77b4"
_H8_COLOR = "#9467bd"
_SAE_COLOR = "#d62728"
_ORACLE_COLOR = "#2ca02c"
_RANDOM_COLOR = "#888888"


def plot_phase_transition(stage1_path: Path, out_path: Path) -> None:
    """Stage 1 figure: S_adj vs W at D=1.

    Layout: TXC curve over W (solid), SAE flat baseline (dashed horizontal),
    oracle ceiling (dotted), random floor (dotted gray).
    """
    with open(stage1_path) as f:
        data = json.load(f)

    txc_cells = data["txc_cells"]
    W_grid = sorted(c["W"] for c in txc_cells)
    by_W = {c["W"]: c for c in txc_cells}
    txc_y = [by_W[W]["txc"]["s_adj"] for W in W_grid]
    h8_y = [by_W[W].get("txc_h8", {}).get("s_adj", float("nan")) for W in W_grid]
    has_h8 = any(not (y != y) for y in h8_y)  # any non-NaN

    sae_s = data["sae"]["s_adj"]
    oracle_s = data["oracle"]["s_adj"]
    random_s = data["random"]["s_adj"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(W_grid, txc_y, "o-", color=_TXC_COLOR, label="TXC (TopK recon)")
    if has_h8:
        ax.plot(W_grid, h8_y, "s-", color=_H8_COLOR, label="TXC H8 (recon + InfoNCE)")
    ax.axhline(sae_s, linestyle="--", color=_SAE_COLOR, label=f"Regular SAE (S_adj={sae_s:.2f})")
    ax.axhline(oracle_s, linestyle=":", color=_ORACLE_COLOR, label=f"Spectral oracle (S_adj={oracle_s:.2f})")
    ax.axhline(random_s, linestyle=":", color=_RANDOM_COLOR, label=f"Random (S_adj={random_s:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(W_grid)
    ax.set_xticklabels([str(W) for W in W_grid])
    ax.set_xlabel("Window length W")
    ax.set_ylabel("Chance-adjusted recovery S_adj")
    ax.set_title("Colored sources, D=1: TXC vs H8 (with InfoNCE) vs SAE")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_phase_transition_by_delay(stage2_path: Path, out_path: Path) -> None:
    """Stage 2 headline: TXC S_adj vs W, one curve per D; SAE at horizontal
    baseline per D (dashed); vertical lines at W=D+1; oracle ceiling per D."""
    with open(stage2_path) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cmap = plt.get_cmap("viridis")
    D_entries = sorted(data["by_D"], key=lambda e: e["D"])
    n_D = len(D_entries)
    all_W: set[int] = set()

    for i, entry in enumerate(D_entries):
        D = entry["D"]
        color = cmap(i / max(n_D - 1, 1))
        txc_W = sorted(entry["txc_cells"], key=lambda c: c["W"])
        W_grid = [c["W"] for c in txc_W]
        all_W.update(W_grid)
        txc_y = [c["txc"]["s_adj"] for c in txc_W]
        sae_s = entry["sae"]["s_adj"]
        oracle_s = entry["oracle"]["s_adj"]

        ax.plot(W_grid, txc_y, "o-", color=color, label=f"TXC D={D}")
        ax.axhline(sae_s, color=color, linestyle="--", alpha=0.6, label=f"SAE D={D} (S_adj={sae_s:.2f})")
        ax.axvline(D + 1, color=color, linestyle=":", alpha=0.4)
        ax.axhline(oracle_s, color=color, linestyle=":", alpha=0.3)

    all_W_sorted = sorted(all_W)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_W_sorted)
    ax.set_xticklabels([str(W) for W in all_W_sorted])
    ax.set_xlabel("Window length W")
    ax.set_ylabel("Chance-adjusted recovery S_adj")
    ax.set_title("Colored sources: phase transition at W = D + 1")
    ax.set_ylim(-0.05, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_polynomial_clock_phase_transition(results_path: Path, out_path: Path) -> None:
    """Two-panel figure for one polynomial-clock stage:

    Panel A: Acc(Y) vs W. Confirms the impossibility bound at W <= h
        (everyone at 1/q) and shows that *every* windowed probe (raw,
        stacked SAE, TXC-global) reaches 1.0 at W >= h+1 — i.e., the
        latent-prediction metric does not architecturally differentiate.

    Panel B: Rec_temp vs W. Architectural finding. TXC-global learns
        polynomial templates, stacked SAE doesn't.

    Vertical dotted line at W = h+1 marks the proposal's predicted phase
    transition.
    """
    with open(results_path) as f:
        data = json.load(f)
    cfg = data["config"]
    h = cfg["h"]
    q = cfg["q"]
    cells = sorted(data["cells"], key=lambda c: c["W"])
    W_grid = [c["W"] for c in cells]
    raw_y = [c["raw_probe"]["val_accuracy"] for c in cells]
    sae_local_y = [c["sae_local_probe"]["val_accuracy"] for c in cells]
    sae_window_y = [c["sae_window_probe"]["val_accuracy"] for c in cells]
    tsae_y = [
        c["tsae_probe"]["val_accuracy"] if c.get("tsae_probe") else float("nan")
        for c in cells
    ]
    tfa_y = [
        c["tfa_probe"]["val_accuracy"] if c.get("tfa_probe") else float("nan")
        for c in cells
    ]
    txc_y = [c["txc_probe"]["val_accuracy"] for c in cells]
    txc_rec = [c["txc_rec_temp"] for c in cells]
    chance = 1.0 / q
    has_tsae = any(not (y != y) for y in tsae_y)
    has_tfa = any(not (y != y) for y in tfa_y)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    axA.plot(W_grid, raw_y, "o-", color=_RANDOM_COLOR, label="raw window probe", linewidth=1.5)
    axA.plot(W_grid, sae_local_y, "s--", color=_SAE_COLOR, label="regular SAE, single-position latent", linewidth=1.5)
    axA.plot(W_grid, sae_window_y, "^-.", color="#ff7f0e", label="regular SAE, window-concat latent", linewidth=1.5)
    if has_tsae:
        axA.plot(W_grid, tsae_y, "v-.", color="#9467bd", label="Bhalla TSAE (k=20, α=0.1)", linewidth=1.5)
    if has_tfa:
        axA.plot(W_grid, tfa_y, "P-.", color="#17becf", label="TFA (k=20, AdamW+cosine, pos enc)", linewidth=1.5)
    # TSAE k-sweep at low k (per-token TopK budgets).
    tsae_palette = ["#9467bd", "#b591d4", "#7c4ca6"]
    for i, k_extra in enumerate([1, 2, 5]):
        key = f"tsae_k{k_extra}_probe"
        if any(c.get(key) for c in cells):
            ys = [c[key]["val_accuracy"] if c.get(key) else float("nan") for c in cells]
            axA.plot(W_grid, ys, "v:", color=tsae_palette[i],
                     label=f"Bhalla TSAE k={k_extra}", linewidth=1.0, alpha=0.6)
    # TFA k-sweep at low k.
    tfa_palette = ["#17becf", "#5dd5e3", "#0e9eb0"]
    for i, k_extra in enumerate([1, 2, 5]):
        key = f"tfa_k{k_extra}_probe"
        if any(c.get(key) for c in cells):
            ys = [c[key]["val_accuracy"] if c.get(key) else float("nan") for c in cells]
            axA.plot(W_grid, ys, "P:", color=tfa_palette[i],
                     label=f"TFA k={k_extra}", linewidth=1.0, alpha=0.6)
    # TXC k-sweep (k=1 from original run, k=2/5/10 added later if present).
    txc_palette = ["#1f77b4", "#7fbcd4", "#5499c7", "#2e86c1"]
    axA.plot(W_grid, txc_y, "D-", color=_TXC_COLOR, label="TXC-global k=1", linewidth=2.0)
    for i, k_extra in enumerate([2, 5, 10]):
        key = f"txc_k{k_extra}_probe"
        if any(c.get(key) for c in cells):
            ys = [c[key]["val_accuracy"] if c.get(key) else float("nan") for c in cells]
            axA.plot(W_grid, ys, "D--", color=txc_palette[i + 1],
                     label=f"TXC-global k={k_extra}", linewidth=1.5, alpha=0.9)
    # SAE k-sweep (k=1 from existing sae_window_probe; k=2/5/10 added).
    sae_palette = ["#d62728", "#e36b6b", "#a83232", "#7a1f1f"]
    for i, k_extra in enumerate([2, 5, 10]):
        key = f"sae_k{k_extra}_window_probe"
        if any(c.get(key) for c in cells):
            ys = [c[key]["val_accuracy"] if c.get(key) else float("nan") for c in cells]
            axA.plot(W_grid, ys, "s:", color=sae_palette[i + 1],
                     label=f"regular SAE k={k_extra} (W·H concat)", linewidth=1.0, alpha=0.7)
    axA.axhline(chance, color="black", linestyle=":", alpha=0.5, label=f"chance = 1/q = {chance:.3f}")
    axA.axvline(h + 1, color="grey", linestyle="--", alpha=0.5, label=f"W = h+1 = {h+1}")
    axA.set_xticks(W_grid)
    axA.set_xlabel("Window length W")
    axA.set_ylabel("Probe val accuracy on Y")
    axA.set_title(f"Panel A: Acc(Y) vs W   (h={h}, q={q})")
    axA.set_ylim(-0.02, 1.05)
    axA.grid(True, alpha=0.3)
    axA.legend(loc="best", fontsize=9)

    axB.plot(W_grid, txc_rec, "D-", color=_TXC_COLOR, label="TXC-global k=1 Rec_temp", linewidth=2.0)
    for i, k_extra in enumerate([2, 5, 10]):
        key = f"txc_k{k_extra}_rec_temp"
        if any(c.get(key) is not None for c in cells):
            ys = [c[key] if c.get(key) is not None else float("nan") for c in cells]
            axB.plot(W_grid, ys, "D--", color=txc_palette[i + 1],
                     label=f"TXC-global k={k_extra} Rec_temp", linewidth=1.5, alpha=0.9)
    axB.axvline(h + 1, color="grey", linestyle="--", alpha=0.5)
    n_atoms = cells[0]["n_atoms"]
    axB.set_xticks(W_grid)
    axB.set_xlabel("Window length W")
    axB.set_ylabel(f"Rec_temp (avg max cos² vs G_β,  M={n_atoms})")
    axB.set_title("Panel B: TXC-global temporal-atom recovery vs W")
    axB.set_ylim(-0.02, 1.05)
    axB.grid(True, alpha=0.3)
    axB.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"Polynomial clock h={h}, q={q}: latent prediction vs temporal-atom recovery",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_multilane_rs(results_path: Path, out_path: Path) -> None:
    """Two-panel figure for the multilane Reed-Solomon experiment.

    Panel A: per-token reconstruction MSE vs W. Plots TXC k=m, TXC k=mW,
        SAE k_pos=1 (k_total=W), SAE k_pos=m (k_total=mW), and the noise
        floor `sigma^2 * d`. Adds the local-alphabet predicted error
        `noise_floor + (1 - k/(m*W))` for k=m as a dashed grey reference.

    Panel B: probe-Y val accuracy vs W for the same architectures, with
        chance line `1/q`.
    """
    with open(results_path) as f:
        data = json.load(f)
    cfg = data["config"]
    m, q, h = cfg["m"], cfg["q"], cfg["h"]
    cells = sorted(data["cells"], key=lambda c: c["W"])
    W_grid = [c["W"] for c in cells]
    chance = 1.0 / q

    sae_k1_mse = [c["sae_k1"]["recon_mse_per_token"] for c in cells]
    sae_km_mse = [c[f"sae_k{m}"]["recon_mse_per_token"] for c in cells]
    txc_km_mse = [c[f"txc_k{m}"]["recon_mse_per_token"] for c in cells]
    txc_kmW_mse = [c[f"txc_k{m * W}"]["recon_mse_per_token"] for c, W in zip(cells, W_grid)]

    sae_k1_probe = [c["sae_k1"]["probe_val_acc"] for c in cells]
    sae_km_probe = [c[f"sae_k{m}"]["probe_val_acc"] for c in cells]
    txc_km_probe = [c[f"txc_k{m}"]["probe_val_acc"] for c in cells]
    txc_kmW_probe = [c[f"txc_k{m * W}"]["probe_val_acc"] for c, W in zip(cells, W_grid)]

    noise_floor = cells[0]["noise_floor_mse_per_token"]
    # Predicted local-alphabet bound at k_total = m: per-token MSE >=
    # noise_floor + (1 - m / (m * W)) = noise_floor + (1 - 1/W).
    local_lb_at_km = [noise_floor + (1.0 - min(m, m * W) / (m * W)) for W in W_grid]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    axA.plot(W_grid, sae_k1_mse, "o:", color="#d62728", label=f"SAE k_pos=1 (k_total=W)", linewidth=1.5)
    axA.plot(W_grid, sae_km_mse, "s--", color="#ff7f0e", label=f"SAE k_pos=m={m} (k_total=mW)", linewidth=1.5)
    axA.plot(W_grid, txc_km_mse, "D-", color="#1f77b4", label=f"TXC k_total=m={m}", linewidth=2.0)
    axA.plot(W_grid, txc_kmW_mse, "D--", color="#2e86c1", label="TXC k_total=mW", linewidth=1.5)
    axA.axhline(noise_floor, color="black", linestyle=":", alpha=0.6, label=f"noise floor σ²d = {noise_floor:.2f}")
    axA.plot(W_grid, local_lb_at_km, "x-.", color="grey", alpha=0.6,
             label=f"local-alphabet LB at k_total=m")
    axA.axvline(h + 1, color="grey", linestyle="--", alpha=0.5, label=f"W = h+1 = {h+1}")
    axA.set_xticks(W_grid)
    axA.set_xlabel("Window length W")
    axA.set_ylabel("Per-token reconstruction MSE")
    axA.set_title(f"Panel A: Reconstruction MSE   (h={h}, q={q}, m={m})")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="best", fontsize=8)

    axB.plot(W_grid, sae_k1_probe, "o:", color="#d62728", label=f"SAE k_pos=1", linewidth=1.5)
    axB.plot(W_grid, sae_km_probe, "s--", color="#ff7f0e", label=f"SAE k_pos=m", linewidth=1.5)
    axB.plot(W_grid, txc_km_probe, "D-", color="#1f77b4", label=f"TXC k_total=m", linewidth=2.0)
    axB.plot(W_grid, txc_kmW_probe, "D--", color="#2e86c1", label="TXC k_total=mW", linewidth=1.5)
    axB.axhline(chance, color="black", linestyle=":", alpha=0.6, label=f"chance = 1/q = {chance:.3f}")
    axB.axvline(h + 1, color="grey", linestyle="--", alpha=0.5)
    axB.set_xticks(W_grid)
    axB.set_xlabel("Window length W")
    axB.set_ylabel("Probe val accuracy on Y")
    axB.set_title("Panel B: Probe accuracy (caveat: train data used)")
    axB.set_ylim(-0.02, 1.05)
    axB.grid(True, alpha=0.3)
    axB.legend(loc="best", fontsize=8)

    fig.suptitle(
        f"Multilane Reed-Solomon  h={h}, q={q}, m={m}: reconstruction-loss separation"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ambiguous_pair_probes(results_path: Path, out_path: Path) -> None:
    """Bar chart of pair-classification probe accuracy on the ambiguous-pair
    HMM: regular SAE (chance), TXC at each W (perfect), raw-x sanity baseline,
    1/R chance line."""
    with open(results_path) as f:
        data = json.load(f)

    R = data["config"]["R"]
    chance = 1.0 / R
    sae_acc = data["sae_probe"]["val_accuracy"]
    raw_acc = data["raw_token_probe"]["val_accuracy"]
    txc_entries = sorted(data["txc"], key=lambda e: e["W"])

    labels = ["raw x\n(sanity)", "SAE\nlatent"] + [f"TXC W={e['W']}\nlatent" for e in txc_entries]
    accs = [raw_acc, sae_acc] + [e["txc_probe"]["val_accuracy"] for e in txc_entries]
    colors = [_RANDOM_COLOR, _SAE_COLOR] + [_TXC_COLOR] * len(txc_entries)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                f"{acc:.2f}", ha="center", fontsize=10)
    ax.axhline(chance, color="black", linestyle=":", label=f"chance = 1/R = {chance:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pair-classification val accuracy at middle position")
    ax.set_title(
        f"Ambiguous-pair HMM (R={R}, d={data['config']['d']}, σ={data['config']['sigma']}): "
        "local probe bounded at 1/R, temporal probes hit 1.0"
    )
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot colored-source results.")
    parser.add_argument(
        "--stage", type=str,
        choices=[
            "1", "2", "ambiguous_pair",
            "poly_h1_q31", "poly_h2_q11", "poly_h3_q7",
            "multilane_smoke", "multilane_main",
        ],
        required=True,
    )
    parser.add_argument(
        "--results_dir", type=str, default="results/v6_colored_sources"
    )
    parser.add_argument(
        "--out_dir", type=str, default="plots/v6_colored_sources"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    if args.stage == "1":
        plot_phase_transition(
            results_dir / "stage1.json", out_dir / "phase_transition_stage1.png"
        )
    elif args.stage == "2":
        plot_phase_transition_by_delay(
            results_dir / "stage2.json", out_dir / "phase_transition_stage2.png"
        )
    elif args.stage == "ambiguous_pair":
        plot_ambiguous_pair_probes(
            results_dir / "ambiguous_pair.json",
            out_dir / "ambiguous_pair_probes.png",
        )
    elif args.stage.startswith("multilane_"):
        label = args.stage.replace("multilane_", "")
        plot_multilane_rs(
            results_dir / f"multilane_rs_{label}.json",
            out_dir / f"multilane_rs_{label}.png",
        )
    else:
        # poly_h{H}_q{Q}
        suffix = args.stage.replace("poly_", "polynomial_clock_")
        plot_polynomial_clock_phase_transition(
            results_dir / f"{suffix}.json",
            out_dir / f"{suffix}.png",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
