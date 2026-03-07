"""Generate improved Pareto frontier plots from B1+B2 results.

Shows Pareto-optimal frontier (lower envelope) rather than connecting all points.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "b1_b2_pareto")
EXPECTED_L0 = 10.0


def pareto_frontier(l0_list, nmse_list):
    """Compute the Pareto frontier (lower-left envelope) in (L0, NMSE) space.

    A point is Pareto-optimal if no other point has both lower L0 AND lower NMSE.
    Returns sorted points on the frontier.
    """
    points = sorted(zip(l0_list, nmse_list))
    frontier = []
    min_nmse = float("inf")
    for l0, nmse in points:
        if nmse < min_nmse:
            frontier.append((l0, nmse))
            min_nmse = nmse
    return frontier


def main():
    with open(os.path.join(RESULTS_DIR, "results.json")) as f:
        data = json.load(f)

    sae_results = data["sae_results"]
    tfa_results = data["tfa_results"]

    # Filter out dead SAE runs (L0 = 0)
    sae_alive = [r for r in sae_results if r["l0"] > 0.5]

    # Compute Pareto frontiers
    sae_frontier = pareto_frontier(
        [r["l0"] for r in sae_alive],
        [r["nmse"] for r in sae_alive],
    )
    tfa_novel_frontier = pareto_frontier(
        [r["novel_l0"] for r in tfa_results],
        [r["nmse"] for r in tfa_results],
    )
    tfa_total_frontier = pareto_frontier(
        [r["total_l0"] for r in tfa_results],
        [r["nmse"] for r in tfa_results],
    )

    # ── Plot 1: All points + Pareto frontiers ──
    fig, ax = plt.subplots(figsize=(10, 7))

    # All SAE points (faded)
    ax.scatter([r["l0"] for r in sae_alive],
               [r["nmse"] for r in sae_alive],
               color="tab:blue", alpha=0.3, s=40, zorder=2)
    # SAE frontier
    sae_f_l0, sae_f_nmse = zip(*sae_frontier)
    ax.plot(sae_f_l0, sae_f_nmse, "o-", color="tab:blue", linewidth=2.5,
            markersize=8, label="Standard SAE (frontier)", zorder=4)

    # All TFA novel points (faded)
    ax.scatter([r["novel_l0"] for r in tfa_results],
               [r["nmse"] for r in tfa_results],
               color="tab:orange", alpha=0.3, s=40, marker="s", zorder=2)
    # TFA novel frontier
    tfa_nf_l0, tfa_nf_nmse = zip(*tfa_novel_frontier)
    ax.plot(tfa_nf_l0, tfa_nf_nmse, "s-", color="tab:orange", linewidth=2.5,
            markersize=8, label="TFA novel L0 (frontier)", zorder=4)

    # TFA total frontier (dashed)
    tfa_tf_l0, tfa_tf_nmse = zip(*tfa_total_frontier)
    ax.plot(tfa_tf_l0, tfa_tf_nmse, "^--", color="tab:red", linewidth=1.5,
            markersize=7, alpha=0.7, label="TFA total L0 (frontier)", zorder=3)

    ax.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5,
               label=f"$E[L_0]$ = {EXPECTED_L0:.0f}")

    ax.set_xlabel("L0 (avg active features)", fontsize=13)
    ax.set_ylabel("NMSE", fontsize=13)
    ax.set_title("Pareto Frontier: NMSE vs L0 (ReLU + L1 sweep)\n"
                 f"n={20}, d={40}, $\\pi$=0.5, $E[L_0]$=10",
                 fontsize=13)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"pareto_frontier_improved.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Zoomed to binding regime (L0 < 15) ──
    fig, ax = plt.subplots(figsize=(9, 6))

    # Filter to L0 < 15
    sae_bind = [(l, n) for l, n in zip(
        [r["l0"] for r in sae_alive], [r["nmse"] for r in sae_alive]
    ) if l < 15]
    tfa_bind = [(r["novel_l0"], r["nmse"]) for r in tfa_results if r["novel_l0"] < 15]

    if sae_bind:
        sae_bl, sae_bn = zip(*sae_bind)
        ax.scatter(sae_bl, sae_bn, color="tab:blue", s=60, zorder=3, label="Standard SAE")
    if tfa_bind:
        tfa_bl, tfa_bn = zip(*tfa_bind)
        ax.scatter(tfa_bl, tfa_bn, color="tab:orange", s=60, marker="s",
                   zorder=3, label="TFA (novel L0)")

    # Frontiers in this range
    sae_f_bind = [(l, n) for l, n in sae_frontier if l < 15]
    tfa_f_bind = [(l, n) for l, n in tfa_novel_frontier if l < 15]
    if sae_f_bind:
        sl, sn = zip(*sae_f_bind)
        ax.plot(sl, sn, "o-", color="tab:blue", linewidth=2, markersize=8)
    if tfa_f_bind:
        tl, tn = zip(*tfa_f_bind)
        ax.plot(tl, tn, "s-", color="tab:orange", linewidth=2, markersize=8)

    ax.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5,
               label=f"$E[L_0]$ = {EXPECTED_L0:.0f}")
    ax.set_xlabel("L0 (avg active features)", fontsize=13)
    ax.set_ylabel("NMSE", fontsize=13)
    ax.set_title("Binding Regime (L0 < 15): TFA vs SAE", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(left=0, right=15)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"pareto_binding_regime.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Improved plots saved.")

    # Print frontier points
    print("\nSAE Pareto frontier:")
    for l0, nmse in sae_frontier:
        print(f"  L0={l0:.2f}, NMSE={nmse:.6f}")
    print("\nTFA novel-L0 Pareto frontier:")
    for l0, nmse in tfa_novel_frontier:
        print(f"  novel_L0={l0:.2f}, NMSE={nmse:.6f}")


if __name__ == "__main__":
    main()
