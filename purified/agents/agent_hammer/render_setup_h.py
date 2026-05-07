"""Setup H ρ-sweep renderer.

Generates the headline ρ-sweep panel + the per-ρ scatter plots for
Setup H (ρ-sweep on D-np10 max-overlap regime). The panel shows gAUC
and eAUC vs ρ at fixed k_pos for each canonical arch, mirroring
Setup C's ρ-sweep figure but applied to the noisy + max-overlap
regime where TXC wins are largest.

Ρ values: {0.0, 0.3, 0.6, 0.9}. Datasources:
- ρ=0.0/0.3/0.6: ``toy_coupled_noisy_K10_M20_d256_pB05_np10_rho{00,03,06}``
  (added by agent_hammer 2026-05-07).
- ρ=0.9: ``toy_coupled_noisy_K10_M20_d256_pB05_np10`` (existing D-np10).

Run:
    TQDM_DISABLE=1 .venv/bin/python -m agents.agent_hammer.render_setup_h
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.c2_synthetic_coupled.plot_headline import (
    ARCH_COLORS,
    ARCH_ORDER,
    LEADERBOARD,
    _arch_label,
)


PLOT_DIR = Path("experiments/c2_synthetic_coupled/plots")
RHO_DATASOURCES = {
    0.0: "toy_coupled_noisy_K10_M20_d256_pB05_np10_rho00",
    0.3: "toy_coupled_noisy_K10_M20_d256_pB05_np10_rho03",
    0.6: "toy_coupled_noisy_K10_M20_d256_pB05_np10_rho06",
    0.9: "toy_coupled_noisy_K10_M20_d256_pB05_np10",
}
DS_TO_RHO = {v: k for k, v in RHO_DATASOURCES.items()}


def _load_setup_h_cells():
    """Return dict: (arch_label, rho, k_pos) → list of metrics dicts."""
    grouped = defaultdict(lambda: {"gauc": [], "eauc": []})
    latest: dict[str, dict] = {}
    with LEADERBOARD.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ec = d.get("eval_cfg") or {}
            if ec.get("smoke") is True:
                continue
            ds = d.get("datasource")
            if ds not in DS_TO_RHO:
                continue
            latest[d["eval_key"]] = d
    for d in latest.values():
        ec = d.get("eval_cfg") or {}
        rho = DS_TO_RHO[d["datasource"]]
        t_label = ec.get("t_label", "default")
        arch_label = _arch_label(d["arch"], t_label)
        if arch_label is None:
            continue
        k_pos = ec.get("k_pos")
        if k_pos is None:
            continue
        gauc = d["metrics"].get("gauc")
        eauc = d["metrics"].get("eauc")
        if gauc is None or eauc is None:
            continue
        grouped[(arch_label, rho, int(k_pos))]["gauc"].append(float(gauc))
        grouped[(arch_label, rho, int(k_pos))]["eauc"].append(float(eauc))
    return grouped


def render_rho_sweep_panel(out_path: Path, *, k_pos: int = 1) -> None:
    """gAUC + eAUC vs ρ at fixed k_pos. One line per arch."""
    grouped = _load_setup_h_cells()
    rhos = sorted(DS_TO_RHO.values())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric in zip(axes, ["gauc", "eauc"]):
        for arch_label in ARCH_ORDER:
            ys = []
            ystds = []
            xs_valid = []
            for rho in rhos:
                vals = grouped.get((arch_label, rho, k_pos), {}).get(metric, [])
                if not vals:
                    continue
                ys.append(np.mean(vals))
                ystds.append(np.std(vals) if len(vals) > 1 else 0.0)
                xs_valid.append(rho)
            if not xs_valid:
                continue
            color, lab, marker = ARCH_COLORS.get(
                arch_label, ("#000", arch_label, "o"))
            ax.errorbar(xs_valid, ys, yerr=ystds, marker=marker, color=color,
                        linewidth=1.6, capsize=4, markersize=8, label=lab)
        ax.set_xlabel(r"$\rho$  (temporal autocorrelation)", fontsize=12)
        ax.set_ylabel("gAUC" if metric == "gauc" else "eAUC", fontsize=12)
        ax.set_xticks(rhos)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title(
            f"Setup H (D-np10 ρ-sweep) — {metric.upper()} vs ρ at "
            f"k_pos={k_pos}", fontsize=11)
        ax.legend(fontsize=8, loc="lower right" if metric == "gauc" else "upper right",
                  framealpha=0.92, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".thumb.png"), dpi=64, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def render_gauc_vs_k(out_path: Path, *, rho: float) -> None:
    """gAUC vs k_pos at fixed ρ. One line per arch."""
    grouped = _load_setup_h_cells()
    ks_all = sorted({k for (a, r, k) in grouped.keys() if r == rho})
    if not ks_all:
        print(f"[plot] no Setup H data at ρ={rho}")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arch_label in ARCH_ORDER:
        ys = []
        ystds = []
        ks_v = []
        for k in ks_all:
            vals = grouped.get((arch_label, rho, k), {}).get("gauc", [])
            if not vals:
                continue
            ys.append(np.mean(vals))
            ystds.append(np.std(vals) if len(vals) > 1 else 0.0)
            ks_v.append(k)
        if not ks_v:
            continue
        color, lab, marker = ARCH_COLORS.get(arch_label, ("#000", arch_label, "o"))
        ax.errorbar(ks_v, ys, yerr=ystds, marker=marker, color=color,
                    linewidth=1.6, capsize=3, markersize=7, label=lab)
    ax.set_xlabel(r"$k_{\rm pos}$", fontsize=12)
    ax.set_ylabel("gAUC", fontsize=12)
    ax.set_xticks(ks_all)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(f"Setup H (D-np10) — gAUC vs $k_{{\\rm pos}}$ at ρ={rho}",
                 fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.92, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".thumb.png"), dpi=64, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def render_eauc_vs_k(out_path: Path, *, rho: float) -> None:
    """eAUC vs k_pos at fixed ρ. One line per arch."""
    grouped = _load_setup_h_cells()
    ks_all = sorted({k for (a, r, k) in grouped.keys() if r == rho})
    if not ks_all:
        print(f"[plot] no Setup H data at ρ={rho}")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arch_label in ARCH_ORDER:
        ys = []
        ystds = []
        ks_v = []
        for k in ks_all:
            vals = grouped.get((arch_label, rho, k), {}).get("eauc", [])
            if not vals:
                continue
            ys.append(np.mean(vals))
            ystds.append(np.std(vals) if len(vals) > 1 else 0.0)
            ks_v.append(k)
        if not ks_v:
            continue
        color, lab, marker = ARCH_COLORS.get(arch_label, ("#000", arch_label, "o"))
        ax.errorbar(ks_v, ys, yerr=ystds, marker=marker, color=color,
                    linewidth=1.6, capsize=3, markersize=7, label=lab)
    ax.set_xlabel(r"$k_{\rm pos}$", fontsize=12)
    ax.set_ylabel("eAUC", fontsize=12)
    ax.set_xticks(ks_all)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_title(f"Setup H (D-np10) — eAUC vs $k_{{\\rm pos}}$ at ρ={rho}",
                 fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.92, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".thumb.png"), dpi=64, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    # ρ-sweep panel at k_pos=1 + k_pos=3
    render_rho_sweep_panel(PLOT_DIR / "c2_setup_h_rho_sweep_k1.png", k_pos=1)
    render_rho_sweep_panel(PLOT_DIR / "c2_setup_h_rho_sweep_k3.png", k_pos=3)
    # gAUC + eAUC vs k at the highest ρ (where the Effect 1+2 story is clearest)
    render_gauc_vs_k(PLOT_DIR / "c2_setup_h_gauc_vs_k.png", rho=0.9)
    render_eauc_vs_k(PLOT_DIR / "c2_setup_h_eauc_vs_k.png", rho=0.9)


if __name__ == "__main__":
    main()
