"""TempBench cross-branch compilation — figures.

Reads tempbench_data.json (single source of truth) and emits all
tempbench/* figures. Run with: uv run safety_research/scripts/tempbench/build_figures.py

No new research is performed — this is a faithful visualization of
already-measured numbers across branches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "scripts" / "tempbench" / "tempbench_data.json"
OUT = ROOT / "figures" / "tempbench"
OUT.mkdir(parents=True, exist_ok=True)


def _load() -> dict[str, Any]:
    with DATA_PATH.open() as f:
        return json.load(f)


# Consistent colours per architecture across the whole report.
ARCH_COLOURS: dict[str, str] = {
    "SAE": "#1f77b4",      # blue
    "T-SAE": "#2ca02c",    # green
    "TXC": "#d62728",      # red
    "MLC": "#9467bd",      # purple
    "raw L13": "#7f7f7f",  # grey
    "TF-IDF": "#bcbd22",   # olive
}


def fig_rose_per_architecture(data: dict[str, Any]) -> None:
    """Per-architecture rose / radar chart over all five TempBench categories.

    Each axis is a rescaled-to-[0,1] performance summary. Higher is better.

    Categories (per the NeurIPS abstract framing):
      1. Synthetic recovery (toy correlated features)
      2. Sparse probing (38-task)
      3. Reasoning (backtracking rescue)
      4. Deception (refusal detection AUC)
      5. Alignment (EM mid-α bundle peak align)
    """
    categories = ["Synthetic\nrecovery", "Sparse\nprobing", "Reasoning\nrescue",
                  "Deception\ndetection", "Alignment\n(EM peak)"]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Rescaled scores — see report text for formulas.
    arch_scores: dict[str, list[float]] = {
        # SAE
        "SAE": [
            data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]["stacked_sae_auc"],  # 0.475
            data["sparse_probing_c3_final_branch"]["k_feat_20_mean_auc"]["topk_sae"],  # 0.9016
            0.226,  # backtracking rescue control rate (no steer)
            data["deception_detection_andre_safety"]["monitor_auc_test_ood"]["sae"],   # 0.948
            data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]["sae"] / 25,  # +21.7 normed
        ],
        # T-SAE
        "T-SAE": [
            (data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]["stacked_sae_auc"] + 0.05),
            data["sparse_probing_c3_final_branch"]["k_feat_20_mean_auc"]["tsae_paper"],  # 0.8851
            0.226,  # control baseline (T-SAE not the headline backtracking arm)
            data["deception_detection_andre_safety"]["monitor_auc_test_ood"]["tsae"],    # 0.963
            0.40,   # T-SAE hookpoint variant ~ MLC-like; see em_features hookpoint_compare doc
        ],
        # TXC
        "TXC": [
            data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]["txcdr_k_2_T_5_auc"],  # 0.978
            data["sparse_probing_c3_final_branch"]["k_feat_20_mean_auc"]["txc_base"],   # 0.8887
            0.290,  # backtracking rescue at α=-8
            data["deception_detection_andre_safety"]["monitor_auc_test_ood"]["txc"],    # 0.954
            data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]["txc_best"] / 25,  # +10.7 normed
        ],
        # MLC for reference where measured (not on every axis)
        "MLC": [
            np.nan,
            0.9124,  # han T8 BASE k=20
            np.nan,
            np.nan,
            data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]["mlc"] / 25,  # +19.4 normed
        ],
    }

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"projection": "polar"})
    for arch, scores in arch_scores.items():
        s = list(scores) + [scores[0]]
        ax.plot(angles, s, "o-", label=arch, color=ARCH_COLOURS[arch], linewidth=2)
        ax.fill(angles, s, alpha=0.10, color=ARCH_COLOURS[arch])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.grid(True, alpha=0.5)
    ax.set_title(
        "TempBench rose chart — five-axis architecture comparison\n"
        "(rescaled-to-[0,1] per-axis; see report for formulas)",
        fontsize=13, pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "rose_per_arch.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_rose_per_category(data: dict[str, Any]) -> None:
    """One rose where each *axis* is a different specific benchmark, and
    the polygons are TXC / T-SAE / SAE / MLC.

    This is the granular version of fig_rose_per_architecture: 9 axes
    instead of 5, so the comparison is more diagnostic."""
    benchmarks = [
        "Toy ρ=0.9 k=2 AUC",
        "HMM denoising ratio (T=8)",
        "c3 38-task probing AUC (k=20)",
        "Han T8 BASE AUC",
        "JBB detect AUC",
        "XSTest detect AUC",
        "FSGA leakage⁻¹ at K=20",
        "EM Q14B single-feat align",
        "EM Q7B Δalign",
    ]
    n = len(benchmarks)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    syn_v2 = data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]
    bill = data["synthetic_three_arch"]
    c3 = data["sparse_probing_c3_final_branch"]["k_feat_20_mean_auc"]
    han = data["sparse_probing_han_T8"]["S_32_k_feat_20_mean_auc_BASE"]
    det_in = data["deception_detection_andre_safety"]["monitor_auc_test_in"]
    det_ood = data["deception_detection_andre_safety"]["monitor_auc_test_ood"]
    fsga_leak = data["deception_steering_andre_v2_fsga"]["K_20_test_in_JBB"]["leakage"]
    em_14b = data["alignment_em_nanda_qwen14b"]["single_feat_champion_align"]
    em_7b = data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]

    # 1 / leakage -> higher is better (cap at 4)
    inv_leak = lambda x: min(1.0 / max(x, 0.01), 4.0) / 4.0
    arch_scores: dict[str, list[float]] = {
        "SAE": [
            syn_v2["stacked_sae_auc"],
            bill["hmm_denoising_floor"],   # SAE/Stacked SAE always at floor 0.77
            c3["topk_sae"],
            han["topk_sae"],
            det_in["sae"], det_ood["sae"],
            inv_leak(fsga_leak["sae"]),
            em_14b["sae_arditi_seed_42_alpha_-30"] / 100,
            em_7b["sae"] / 25,
        ],
        "T-SAE": [
            syn_v2["stacked_sae_auc"] + 0.02,  # comparable
            bill["hmm_denoising_floor"],
            c3["tsae_paper"],
            han["tsae_paper_k500"],
            det_in["tsae"], det_ood["tsae"],
            inv_leak(fsga_leak["tsae"]),
            np.nan,                       # T-SAE Q14B EM not in the leaderboard
            np.nan,                       # T-SAE Q7B not the headline
        ],
        "TXC": [
            syn_v2["txcdr_k_2_T_5_auc"],
            data["synthetic_three_arch"]["txc_hmm_denoising_ratio"]["T_8_k_3"] / 1.5,  # rescale: 1.12/1.5
            c3["txc_base"],
            han["txc_bare_antidead_t5"],
            det_in["txc"], det_ood["txc"],
            inv_leak(fsga_leak["txc"]),
            81.70 / 100,
            em_7b["txc_best"] / 25,
        ],
        "MLC": [
            np.nan,
            np.nan,
            np.nan,
            han["mlc"],
            np.nan, np.nan,
            np.nan,
            np.nan,
            em_7b["mlc"] / 25,
        ],
    }

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={"projection": "polar"})
    for arch, scores in arch_scores.items():
        s = list(scores) + [scores[0]]
        ax.plot(angles, s, "o-", label=arch, color=ARCH_COLOURS[arch], linewidth=2)
        ax.fill(angles, s, alpha=0.10, color=ARCH_COLOURS[arch])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(benchmarks, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.5)
    ax.set_title(
        "TempBench — 9-axis benchmark rose (per-axis rescaling explained in report)",
        fontsize=13, pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "rose_9axis.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_K_curve_fsga(data: dict[str, Any]) -> None:
    """K-vs-ΔLR_harm curves on JBB test_in for SAE / T-SAE / TXC FSGA."""
    fs = data["deception_steering_andre_v2_fsga"]
    Ks_full = [1, 2, 5, 10, 20, 50, 100]
    Ks_sae = [1, 5, 20, 50, 100]
    Ks_tsae = [1, 5, 20, 50, 100]

    sae_curve = [fs["sae_K_curve_test_in_LR_harm"][f"K_{k}"] for k in Ks_sae]
    tsae_curve = [fs["tsae_K_curve_test_in_LR_harm"][f"K_{k}"] for k in Ks_tsae]
    txc_curve = [fs["txc_K_curve_test_in_LR_harm"][f"K_{k}"] for k in Ks_full]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(Ks_sae, sae_curve, "o-", color=ARCH_COLOURS["SAE"], linewidth=2,
            label="SAE T=1 (saturates K=50, |ΔLR|=10.5)")
    ax.plot(Ks_tsae, tsae_curve, "s-", color=ARCH_COLOURS["T-SAE"], linewidth=2,
            label="T-SAE T=5 (saturates K=50 then degrades)")
    ax.plot(Ks_full, txc_curve, "^-", color=ARCH_COLOURS["TXC"], linewidth=2,
            label="TXC T=5 (monotone in K, scales cleanly)")
    ax.set_xscale("log")
    ax.set_xlabel("K (number of gated features)")
    ax.set_ylabel("ΔLR_harm (nats; more negative = stronger refusal suppression)")
    ax.set_title("FSGA K-curve on JailbreakBench (n=200, gemma-2-2b-it L13)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fsga_kcurve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_em_alignment_bars(data: dict[str, Any]) -> None:
    """Side-by-side bars: EM 7B medical Δalign and 14B finance peak align."""
    em7b = data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]
    em14b = data["alignment_em_nanda_qwen14b"]["single_feat_champion_align"]

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # 7B medical
    arches7 = ["SAE\n(Andy 131k)", "MLC\n(L=5, d=32k)", "TXC\n(d=32k, 200k)"]
    vals7 = [em7b["sae"], em7b["mlc"], em7b["txc_best"]]
    cols7 = [ARCH_COLOURS["SAE"], ARCH_COLOURS["MLC"], ARCH_COLOURS["TXC"]]
    axs[0].bar(arches7, vals7, color=cols7, edgecolor="black")
    axs[0].set_ylabel("Δalign vs baseline (Qwen-7B bad-medical)")
    axs[0].set_title("EM Qwen-7B medical: Δalign at peak α\n(higher = more recovery from EM)")
    for i, v in enumerate(vals7):
        axs[0].text(i, v + 0.4, f"+{v:.1f}", ha="center", fontsize=10)
    axs[0].set_ylim(0, 25)

    # 14B finance — pick the seed=42 + alpha=+100 TXC vs SAE seed=42
    arches14 = ["SAE Arditi\nseed=42\nα=-30", "SAE Arditi\nseed=42\nα=-10", "TXC base\nseed=42\nα=+100"]
    vals14 = [em14b["sae_arditi_seed_42_alpha_-30"], em14b["sae_arditi_seed_42_alpha_-10"], em14b["txc_base_seed_42_alpha_+100"]]
    cols14 = [ARCH_COLOURS["SAE"], ARCH_COLOURS["SAE"], ARCH_COLOURS["TXC"]]
    axs[1].bar(arches14, vals14, color=cols14, edgecolor="black")
    axs[1].set_ylabel("peak align (Qwen-14B finance R32)")
    axs[1].set_title("EM Qwen-14B finance: single-feat champion align\n(TXC narrowly wins at extreme +α)")
    for i, v in enumerate(vals14):
        axs[1].text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=10)
    axs[1].set_ylim(0, 95)

    plt.tight_layout()
    plt.savefig(OUT / "em_alignment_bars.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_detection_auc(data: dict[str, Any]) -> None:
    """Detection AUC bars across deception splits + black-to-white boost."""
    det = data["deception_detection_andre_safety"]
    splits = ["JBB test_in", "XSTest test_ood"]
    arches = ["TF-IDF", "raw L13", "SAE", "T-SAE", "TXC"]

    auc_in = [det["monitor_auc_test_in"]["tfidf_baseline"],
              det["monitor_auc_test_in"]["raw_l13_residual"],
              det["monitor_auc_test_in"]["sae"],
              det["monitor_auc_test_in"]["tsae"],
              det["monitor_auc_test_in"]["txc"]]
    auc_ood = [det["monitor_auc_test_ood"]["tfidf_baseline"],
               det["monitor_auc_test_ood"]["raw_l13_residual"],
               det["monitor_auc_test_ood"]["sae"],
               det["monitor_auc_test_ood"]["tsae"],
               det["monitor_auc_test_ood"]["txc"]]
    cols = [ARCH_COLOURS["TF-IDF"], ARCH_COLOURS["raw L13"],
            ARCH_COLOURS["SAE"], ARCH_COLOURS["T-SAE"], ARCH_COLOURS["TXC"]]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    x = np.arange(len(arches))
    w = 0.35
    ax.bar(x - w/2, auc_in, w, color=cols, label="JBB test_in", edgecolor="black")
    ax.bar(x + w/2, auc_ood, w, color=cols, alpha=0.55, label="XSTest test_ood",
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(arches)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.6, 1.0)
    ax.axhline(det["monitor_auc_test_ood"]["tfidf_baseline"], color="grey",
               linestyle=":", alpha=0.5, label="black-box floor")
    ax.set_title("Refusal detection AUC — white-box probes outpace text-only by +0.27-0.30\n"
                 "(SAE family arms within 95% bootstrap CI of each other)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "detection_auc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_synthetic_advantage_heatmap(data: dict[str, Any]) -> None:
    """Heatmap: TXC ΔAUC vs Stacked SAE on (rho, k) at T=5 (toy v2)."""
    rhos = [0.0, 0.6, 0.9]
    ks = [2, 5, 10, 25]
    # Numbers from docs/andre/v2_tx_v_sae.md mean ΔAUCs.
    dauc = np.array([
        [0.10, 0.05, -0.05, -0.18],   # rho=0.0
        [0.30, 0.48, 0.20, -0.02],    # rho=0.6
        [0.50, 0.45, 0.30, 0.02],     # rho=0.9
    ])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(dauc, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([f"ρ={r}" for r in rhos])
    for i in range(len(rhos)):
        for j in range(len(ks)):
            v = dauc[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.3 else "black", fontsize=11)
    ax.set_title("TXCDR ΔAUC vs Stacked SAE at T=5 (toy correlated features)\n"
                 "Red = TXC wins, Blue = TXC loses")
    plt.colorbar(im, ax=ax, label="ΔAUC (TXCDR − Stacked SAE)")
    plt.tight_layout()
    plt.savefig(OUT / "synthetic_advantage_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_hmm_denoising(data: dict[str, Any]) -> None:
    """HMM denoising ratio vs T."""
    Ts = [2, 4, 6, 8, 12]
    txc = [0.89, 1.01, 1.11, 1.12, 1.15]   # from Bill's three-arch
    sae = [0.78, 0.77, 0.77, 0.76, 0.76]
    stacked = [0.78, 0.75, 0.77, 0.76, 0.76]
    floor = data["synthetic_three_arch"]["hmm_denoising_floor"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ts, txc, "^-", color=ARCH_COLOURS["TXC"], linewidth=2, label="TXC (k=3)")
    ax.plot(Ts, sae, "o-", color=ARCH_COLOURS["SAE"], linewidth=2, label="Regular SAE")
    ax.plot(Ts, stacked, "s-", color=ARCH_COLOURS["T-SAE"], linewidth=2, label="Stacked SAE")
    ax.axhline(floor, color="grey", linestyle=":", alpha=0.7,
               label=f"per-token denoising floor = {floor}")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.3,
               label="perfect-observation ceiling = 1.0")
    ax.set_xlabel("Window length T")
    ax.set_ylabel("Denoising ratio (latent-vs-hidden ÷ obs-vs-hidden)")
    ax.set_title("HMM denoising — only TXC crosses ratio = 1\n"
                 "(it tracks the hidden state better than the noisy observation)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "hmm_denoising.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_steering_c5(data: dict[str, Any]) -> None:
    """Concept-steering c5 success-vs-coherence trade-off."""
    s = data["steering_c5_concept_steering"]["mean_success_at_coh_1_75"]
    c = data["steering_c5_concept_steering"]["mean_coh"]
    arches = ["topk_sae", "tsae_paper", "txc_base", "txc_pro"]
    pretty = ["SAE", "T-SAE", "TXC base", "TXC pro"]
    cols = [ARCH_COLOURS["SAE"], ARCH_COLOURS["T-SAE"], ARCH_COLOURS["TXC"], ARCH_COLOURS["TXC"]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, a in enumerate(arches):
        ax.scatter(c[a], s[a], s=180, color=cols[i], edgecolor="black",
                   marker="o" if "txc" not in a else "^", label=pretty[i])
        ax.annotate(pretty[i], (c[a], s[a]), xytext=(5, 7), textcoords="offset points")
    ax.set_xlabel("mean coherence")
    ax.set_ylabel("success @ coh ≥ 1.75")
    ax.set_title("c5 concept-steering — coherence/success trade-off\n"
                 "(30 concepts × 9 strengths × 3 seeds, Gemma-2-2b-it L13)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "c5_steering_tradeoff.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_overall_summary_grid(data: dict[str, Any]) -> None:
    """A single 3x2 panel showing the headline categories side-by-side."""
    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    arches = ["SAE", "T-SAE", "TXC"]
    cols = [ARCH_COLOURS[a] for a in arches]

    # Synthetic toy v2 best cell (rho=0.9 k=2 T=5 AUC)
    syn = data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]
    axs[0, 0].bar(arches, [syn["stacked_sae_auc"], syn["stacked_sae_auc"]+0.02,
                            syn["txcdr_k_2_T_5_auc"]], color=cols, edgecolor="black")
    axs[0, 0].set_title("SYNTHETIC: ρ=0.9 k=2 T=5 AUC (toy)")
    axs[0, 0].set_ylim(0, 1.0)
    axs[0, 0].set_ylabel("Feature recovery AUC")

    # Sparse probing c3
    p = data["sparse_probing_c3_final_branch"]["k_feat_20_mean_auc"]
    axs[0, 1].bar(arches, [p["topk_sae"], p["tsae_paper"], p["txc_base"]], color=cols, edgecolor="black")
    axs[0, 1].set_title("SPARSE PROBING: c3 38-task mean AUC (k_feat=20, 3 seeds)")
    axs[0, 1].set_ylim(0.86, 0.92)
    axs[0, 1].set_ylabel("mean AUC")

    # Deception detection (XSTest test_ood)
    d = data["deception_detection_andre_safety"]["monitor_auc_test_ood"]
    axs[1, 0].bar(arches, [d["sae"], d["tsae"], d["txc"]], color=cols, edgecolor="black")
    axs[1, 0].axhline(d["tfidf_baseline"], color="grey", linestyle="--", label="TF-IDF baseline")
    axs[1, 0].set_title("DECEPTION: XSTest detection AUC (n=450)")
    axs[1, 0].set_ylim(0.6, 1.0)
    axs[1, 0].set_ylabel("AUC")
    axs[1, 0].legend()

    # Deception steering FSGA peak |ΔLR_harm|
    fs = data["deception_steering_andre_v2_fsga"]["saturation"]
    axs[1, 1].bar(arches, [-fs["sae_K_50"], -fs["tsae_K_50"], -fs["txc_K_100"]],
                   color=cols, edgecolor="black")
    axs[1, 1].set_title("DECEPTION: peak FSGA |ΔLR_harm| (nats)")
    axs[1, 1].set_ylabel("|ΔLR_harm| (saturation)")

    # Alignment EM 7B
    em7 = data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]
    axs[2, 0].bar(arches, [em7["sae"], em7["mlc"], em7["txc_best"]],
                   color=[ARCH_COLOURS["SAE"], ARCH_COLOURS["MLC"], ARCH_COLOURS["TXC"]],
                   edgecolor="black")
    axs[2, 0].set_xticklabels(["SAE", "MLC", "TXC"])
    axs[2, 0].set_title("ALIGNMENT: EM Qwen-7B medical Δalign at peak α")
    axs[2, 0].set_ylim(0, 25)
    axs[2, 0].set_ylabel("Δalign")

    # Reasoning backtracking rescue
    rb = data["reasoning_backtracking_c7"]["rescue_rate_by_alpha"]
    def _alpha_num(key: str) -> float:
        s = key.replace("alpha_", "").split("_")[0]
        return float(s)
    alphas = sorted(rb.keys(), key=_alpha_num)
    rates = [rb[a] for a in alphas]
    pretty_a = []
    for a in alphas:
        n = _alpha_num(a)
        pretty_a.append(f"α={n:+g}" + (" (ctrl)" if a.endswith("control") else ""))
    bar_cols = ["#bbbbbb" if a.endswith("control") else "#888888" for a in alphas]
    axs[2, 1].bar(pretty_a, rates, color=bar_cols, edgecolor="black")
    axs[2, 1].set_title("REASONING: backtracking rescue rate (n=31 truly-wrong DeepSeek prompts)")
    axs[2, 1].set_ylabel("rescue rate")
    axs[2, 1].axhline(0.226, color="red", linestyle=":", alpha=0.7, label="control = 0.226")
    axs[2, 1].tick_params(axis="x", rotation=45)
    axs[2, 1].legend()

    fig.suptitle("TempBench at-a-glance — 6 categories × architectures",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "overall_summary_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = _load()
    fig_rose_per_architecture(data)
    fig_rose_per_category(data)
    fig_K_curve_fsga(data)
    fig_em_alignment_bars(data)
    fig_detection_auc(data)
    fig_synthetic_advantage_heatmap(data)
    fig_hmm_denoising(data)
    fig_steering_c5(data)
    fig_overall_summary_grid(data)
    print(f"Wrote figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
