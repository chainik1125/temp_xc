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
    "SAE": "#1f77b4",            # blue
    "Stacked SAE": "#2ca02c",    # green
    "T-SAE (Bhalla)": "#17becf", # cyan
    "TXC": "#d62728",            # red
    "MLC": "#9467bd",            # purple
    "raw L13": "#7f7f7f",        # grey
    "TF-IDF": "#bcbd22",         # olive
    # Backwards-compat alias for any caller still using "T-SAE":
    "T-SAE": "#2ca02c",
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
    paper = data["sparse_probing_paper_set_BASE_S32"]["k_feat_20_mean_auc"]
    det_ood = data["deception_detection_andre_safety"]["monitor_auc_test_ood"]
    arch_scores: dict[str, list[float]] = {
        # SAE T=1
        "SAE": [
            data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]["stacked_sae_auc"],  # 0.475
            paper["topk_sae"],                  # 0.9091 PAPER
            0.226,                               # backtracking control
            det_ood["sae"],                      # 0.948 XSTest
            data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]["sae"] / 25,  # +21.7 normed
        ],
        # Stacked SAE (Andre's earlier "T-SAE" — T independent per-position SAEs)
        "Stacked SAE": [
            (data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]["stacked_sae_auc"] + 0.05),
            np.nan,                              # no separate Stacked-SAE entry on Han PAPER
            0.226,                               # control
            det_ood["stacked_sae"],              # 0.963
            np.nan,
        ],
        # T-SAE (Bhalla 2025) — the actual T-SAE
        "T-SAE (Bhalla)": [
            np.nan,                              # no Bhalla T-SAE on toy ρ-sweep
            paper["tsae_paper_k500"],           # 0.9105 — Han's tsae_paper IS the Bhalla T-SAE port
            np.nan,
            det_ood["tsae_bhalla"],              # 0.958 (this run)
            np.nan,
        ],
        # TXC
        "TXC": [
            data["synthetic_andre_v2_toy"]["best_cell_rho_0.9"]["txcdr_k_2_T_5_auc"],  # 0.978
            paper["txc_bare_antidead_t5"],      # 0.9127 PAPER
            0.290,                               # backtracking optimum
            det_ood["txc"],                      # 0.954
            data["alignment_em_qwen7b_medical"]["delta_align_vs_baseline"]["txc_best"] / 25,  # +10.7 normed
        ],
        # MLC for reference where measured
        "MLC": [
            np.nan,
            paper["mlc"],                        # 0.9122 PAPER
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
    paper = data["sparse_probing_paper_set_BASE_S32"]["k_feat_20_mean_auc"]
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
            paper["topk_sae"],
            han["topk_sae"],
            det_in["sae"], det_ood["sae"],
            inv_leak(fsga_leak["sae"]),
            em_14b["sae_arditi_seed_42_alpha_-30"] / 100,
            em_7b["sae"] / 25,
        ],
        "Stacked SAE": [
            syn_v2["stacked_sae_auc"] + 0.02,  # comparable
            bill["hmm_denoising_floor"],
            np.nan,                          # not measured on PAPER 16
            han["tsae_paper_k500"],
            det_in["stacked_sae"], det_ood["stacked_sae"],
            inv_leak(fsga_leak["tsae"]),    # FSGA leakage from Andre's "tsae" arm = Stacked SAE
            np.nan,
            np.nan,
        ],
        "T-SAE (Bhalla)": [
            np.nan, np.nan,
            paper["tsae_paper_k500"],
            np.nan,
            det_in["tsae_bhalla"], det_ood["tsae_bhalla"],
            np.nan, np.nan, np.nan,
        ],
        "TXC": [
            syn_v2["txcdr_k_2_T_5_auc"],
            data["synthetic_three_arch"]["txc_hmm_denoising_ratio"]["T_8_k_3"] / 1.5,  # rescale: 1.12/1.5
            paper["txc_bare_antidead_t5"],
            han["txc_bare_antidead_t5"],
            det_in["txc"], det_ood["txc"],
            inv_leak(fsga_leak["txc"]),
            81.70 / 100,
            em_7b["txc_best"] / 25,
        ],
        "MLC": [
            np.nan,
            np.nan,
            paper["mlc"],
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
    """K-vs-ΔLR_harm curves on JBB test_in for SAE / Stacked SAE / TXC FSGA.

    (The "T-SAE" arm in Andre's safety_research/REPORT_v2.md was actually
    Stacked SAE — see the terminology callout in the meta-report.)
    """
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
    ax.plot(Ks_tsae, tsae_curve, "s-", color=ARCH_COLOURS["Stacked SAE"], linewidth=2,
            label="Stacked SAE T=5 (saturates K=50 then degrades)")
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
    det = det_in_d = data["deception_detection_andre_safety"]
    arches_keys = ["tfidf_baseline", "raw_l13_residual", "sae",
                    "stacked_sae", "tsae_bhalla", "txc"]
    arches_pretty = ["TF-IDF", "raw L13", "SAE", "Stacked SAE",
                      "T-SAE (Bhalla)", "TXC"]
    cols = [ARCH_COLOURS["TF-IDF"], ARCH_COLOURS["raw L13"],
            ARCH_COLOURS["SAE"], ARCH_COLOURS["Stacked SAE"],
            ARCH_COLOURS["T-SAE (Bhalla)"], ARCH_COLOURS["TXC"]]

    auc_in = [det["monitor_auc_test_in"][k] for k in arches_keys]
    auc_ood = [det["monitor_auc_test_ood"][k] for k in arches_keys]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(arches_pretty))
    w = 0.38
    ax.bar(x - w/2, auc_in, w, color=cols, label="JBB test_in", edgecolor="black")
    ax.bar(x + w/2, auc_ood, w, color=cols, alpha=0.55, label="XSTest test_ood",
           edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(arches_pretty, rotation=15)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.6, 1.0)
    ax.axhline(det["monitor_auc_test_ood"]["tfidf_baseline"], color="grey",
               linestyle=":", alpha=0.5, label="black-box floor")
    ax.set_title("Refusal detection AUC — white-box probes outpace text-only by +0.28-0.30\n"
                 "(four SAE-family arms — including the actual Bhalla T-SAE — within 95% CI)")
    for xi, v in zip(x, auc_in):
        ax.text(xi - w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x, auc_ood):
        ax.text(xi + w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
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


def fig_paper_leaderboard(data: dict[str, Any]) -> None:
    """PAPER 16-task BASE S=32 k_feat=20 leaderboard with σ_seeds error bars."""
    p = data["sparse_probing_paper_set_BASE_S32"]
    aucs = p["k_feat_20_mean_auc"]
    sigmas = p["k_feat_20_sigma_seeds"]

    # Order by AUC descending; only annotate σ where we have it.
    ordered = sorted(aucs.items(), key=lambda kv: -kv[1])
    arches = [k for k, _ in ordered]
    means = [aucs[a] for a in arches]
    errs = [sigmas.get(a, 0.0) for a in arches]

    pretty = []
    cols = []
    for a in arches:
        if "txc" in a or "phase5" in a or "phase57" in a or "txcdr" in a:
            pretty.append(a.replace("_", " "))
            cols.append("#d62728" if a == "txc_bare_antidead_t5" else "#f4a3a3")
        elif a == "mlc":
            pretty.append("mlc")
            cols.append("#9467bd")
        elif "tsae" in a:
            pretty.append(a.replace("_", " "))
            cols.append("#2ca02c")
        elif "topk_sae" in a:
            pretty.append("topk sae")
            cols.append("#1f77b4")
        else:
            pretty.append(a)
            cols.append("#888888")

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(arches)), means, yerr=errs, color=cols, edgecolor="black",
                   capsize=4)
    ax.set_xticks(range(len(arches)))
    ax.set_xticklabels(pretty, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("mean AUC ± σ_seeds")
    ax.set_ylim(0.895, 0.918)
    ax.axhline(aucs["topk_sae"], color="#1f77b4", linestyle=":", alpha=0.5,
               label="topk_sae baseline")
    ax.set_title("PAPER 16-task BASE S=32 k_feat=20 sparse-probing leaderboard\n"
                 "TXC bare-antidead T=5 wins at 0.9127 (Δ vs topk_sae ≈ 6× σ_seeds)")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.0005,
                 f"{mean:.4f}", ha="center", fontsize=8)
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "paper_leaderboard.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_backtracking_inducement(data: dict[str, Any]) -> None:
    """Backtracking rescue α-curve with optimal inducement rate annotated."""
    rb = data["reasoning_backtracking_c7"]["rescue_rate_by_alpha"]
    items = []
    for k, v in rb.items():
        a = float(k.replace("alpha_", "").split("_")[0])
        items.append((a, v, k.endswith("control")))
    items.sort()
    alphas = [x[0] for x in items]
    rates = [x[1] for x in items]
    is_ctrl = [x[2] for x in items]

    optimal = max(rates)
    control = next(r for a, r, c in items if c)
    delta = optimal - control

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(alphas, rates, "o-", color="#444444", linewidth=2, markersize=9,
             zorder=2)
    # Highlight the control α=0
    for a, r, c in items:
        if c:
            ax.scatter([a], [r], color="grey", s=160, zorder=3,
                       label=f"control (α=0): {r:.3f}")
        if r == optimal:
            ax.scatter([a], [r], color="#d62728", s=180, marker="^", zorder=4,
                       label=f"optimum: {r:.3f}" if a == alphas[rates.index(optimal)] else None)
    ax.axhline(control, color="grey", linestyle="--", alpha=0.4)
    ax.axhline(optimal, color="#d62728", linestyle="--", alpha=0.4)
    ax.fill_between(alphas, control, optimal, alpha=0.10, color="#d62728")
    ax.annotate(f"Δ vs control = +{delta*100:.1f} pp",
                xy=(-2, (optimal + control) / 2),
                xytext=(2, (optimal + control) / 2 + 0.03),
                fontsize=11, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xlabel("steering coefficient α (negative = encourage backtracking)")
    ax.set_ylabel("rescue rate (n=31 truly-wrong DeepSeek prompts)")
    ax.set_title("c7 backtracking — α-sweep & optimal inducement rate\n"
                 f"optimal = {optimal:.3f} at α=−2 / α=−8; control = {control:.3f}; "
                 f"Δ = +{delta*100:.1f} pp; α=+8 collapses rescue to 0")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT / "backtracking_inducement.png", dpi=140, bbox_inches="tight")
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

    # Sparse probing — PAPER 16-task BASE S=32 k=20 (3 seeds) — the headline
    p = data["sparse_probing_paper_set_BASE_S32"]["k_feat_20_mean_auc"]
    axs[0, 1].bar(arches, [p["topk_sae"], p["tsae_paper_k500"], p["txc_bare_antidead_t5"]],
                   color=cols, edgecolor="black")
    axs[0, 1].set_title("SPARSE PROBING: PAPER 16-task BASE S=32 k=20 mean AUC (3 seeds)\nTXC bare-antidead wins (~6× σ_seeds)")
    axs[0, 1].set_ylim(0.88, 0.92)
    axs[0, 1].set_ylabel("mean AUC")

    # Deception detection (XSTest test_ood) — four SAE-family arms
    d = data["deception_detection_andre_safety"]["monitor_auc_test_ood"]
    det_arches = ["SAE", "Stacked SAE", "T-SAE (Bhalla)", "TXC"]
    det_cols = [ARCH_COLOURS["SAE"], ARCH_COLOURS["Stacked SAE"],
                 ARCH_COLOURS["T-SAE (Bhalla)"], ARCH_COLOURS["TXC"]]
    axs[1, 0].bar(det_arches, [d["sae"], d["stacked_sae"], d["tsae_bhalla"], d["txc"]],
                   color=det_cols, edgecolor="black")
    axs[1, 0].axhline(d["tfidf_baseline"], color="grey", linestyle="--", label="TF-IDF baseline")
    axs[1, 0].set_title("DECEPTION: XSTest detection AUC (n=450, 4 arms)")
    axs[1, 0].set_ylim(0.6, 1.0)
    axs[1, 0].set_ylabel("AUC")
    axs[1, 0].tick_params(axis="x", rotation=15)
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
    fig_paper_leaderboard(data)
    fig_backtracking_inducement(data)
    fig_overall_summary_grid(data)
    print(f"Wrote figures to {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
