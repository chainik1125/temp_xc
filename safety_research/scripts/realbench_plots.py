"""
Generate all figures for the realbench report.

Figures:
  fig1_detect_auc.png      - bar chart of test_in/test_ood AUC w/ 95% CIs
  fig2_roc_curves.png      - ROC curves per arm on both splits
  fig3_b2w_boost.png       - black-to-white boost (arm AUC - TF-IDF AUC)
  fig4_per_feat_top.png    - top-feature AUC distribution per arm
  fig5_steer_pareto.png    - refusal-LR shift on harmful vs benign per arm
  fig6_steer_doseresponse.png  - LR vs alpha curves
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/cs29824/andre/temp_xc/safety_research")
DET = ROOT / "results" / "realbench" / "detect"
STR_ = ROOT / "results" / "realbench" / "steer"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ARMS = ["sae", "tsae", "txc"]
ARM_LABEL = {"sae": "SAE (T=1)", "tsae": "T-SAE (T=5)", "txc": "TXC (T=5)"}
ARM_COLOR = {"sae": "#888", "tsae": "#1976d2", "txc": "#d32f2f"}


def fig1_detect_auc():
    summary = json.load(open(DET / "summary.json"))
    bb = summary["blackbox"]
    raw = summary["raw_residual"]
    arms = summary["arms"]

    splits = ["test_in", "test_ood"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    width = 0.18
    xs = np.arange(len(splits))

    rows = [
        ("TF-IDF (text)", bb, "#bbb"),
        ("raw L13 resid", raw, "#666"),
    ]
    for arm in ARMS:
        rows.append((ARM_LABEL[arm], arms[arm]["results"], ARM_COLOR[arm]))

    for j, (label, res, color) in enumerate(rows):
        ys = [res[s]["auc"] for s in splits]
        if "ci" in res[splits[0]]:
            errs = [[res[s]["auc"] - res[s]["ci"]["lo"] for s in splits],
                    [res[s]["ci"]["hi"] - res[s]["auc"] for s in splits]]
        else:
            errs = None
        ax.bar(xs + (j - 2) * width, ys, width, label=label, color=color,
               yerr=errs, capsize=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(["test_in (JBB)", "test_ood (XSTest)"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.6)
    ax.set_title("Real-benchmark detection AUC (95% bootstrap CI)")
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_detect_auc.png", dpi=140)
    plt.close(fig)


def fig2_roc_curves():
    summary = json.load(open(DET / "summary.json"))
    splits = ["test_in", "test_ood"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, split in zip(axes, splits):
        for arm in ARMS:
            roc = summary["arms"][arm]["roc"][split]
            auc = summary["arms"][arm]["results"][split]["auc"]
            ax.plot(roc["fpr"], roc["tpr"], color=ARM_COLOR[arm],
                    label=f"{ARM_LABEL[arm]} (AUC={auc:.3f})", linewidth=1.6)
        ax.plot([0, 1], [0, 1], "k:", linewidth=0.7)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR" if split == "test_in" else "")
        ax.set_title(split)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("ROC curves — sparse-feature probes")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_roc_curves.png", dpi=140)
    plt.close(fig)


def fig3_b2w_boost():
    summary = json.load(open(DET / "summary.json"))
    splits = ["test_in", "test_ood"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    width = 0.27
    xs = np.arange(len(splits))
    for j, arm in enumerate(ARMS):
        b = [summary["arms"][arm]["black_to_white_boost"][s] for s in splits]
        ax.bar(xs + (j - 1) * width, b, width, color=ARM_COLOR[arm],
               label=ARM_LABEL[arm])
    ax.set_xticks(xs)
    ax.set_xticklabels(["test_in (JBB)", "test_ood (XSTest)"])
    ax.set_ylabel("Black-to-white boost (Δ AUC vs TF-IDF)")
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_title("White-box AUC lift over text-only baseline")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig3_b2w_boost.png", dpi=140)
    plt.close(fig)


def fig4_per_feat_top():
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for arm in ARMS:
        per_auc = np.load(DET / f"{arm}_per_feat_auc.npy")
        signed = np.abs(per_auc - 0.5)
        signed = np.sort(signed)[::-1]
        ax.plot(np.arange(1, 201), signed[:200] + 0.5,
                color=ARM_COLOR[arm], label=ARM_LABEL[arm], linewidth=1.5)
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("|AUC − 0.5| + 0.5  (refusal alignment)")
    ax.set_title("Top 200 most refusal-aligned features per arm")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_per_feat_top.png", dpi=140)
    plt.close(fig)


def fig5_steer_pareto():
    if not (STR_ / "baseline.json").exists():
        return
    base = json.load(open(STR_ / "baseline.json"))
    base_h = base["test_in"]["lr_harm_mean"]
    base_b = base["test_in"]["lr_ben_mean"]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))

    methods = {
        "DoM (no SAE)": (STR_ / "dom.json", "#000", "o"),
    }
    for arm in ARMS:
        for dn, marker in (("coef_dir", "s"), ("centroid_dir", "^")):
            path = STR_ / f"{arm}_{dn}.json"
            if path.exists():
                methods[f"{ARM_LABEL[arm]} {dn[:-4]}"] = (path, ARM_COLOR[arm], marker)

    for label, (path, color, marker) in methods.items():
        d = json.load(open(path))
        if "test_in" not in d:
            continue
        for alpha_str, res in d["test_in"].items():
            if alpha_str == "ablate":
                continue
            try:
                alpha = float(alpha_str)
            except ValueError:
                continue
            if res.get("lr_harm_mean") is None:
                continue
            dh = res["lr_harm_mean"] - base_h
            db = res["lr_ben_mean"] - base_b
            ax.scatter(dh, db, color=color, marker=marker,
                       s=10 + abs(alpha) * 30, alpha=0.7,
                       edgecolors="k", linewidths=0.4)
        # connect by alpha
        items = sorted(((float(a), r) for a, r in d["test_in"].items()
                        if a not in ("ablate",)),
                       key=lambda kv: kv[0])
        if len(items) >= 2:
            xs = [r["lr_harm_mean"] - base_h for _, r in items]
            ys = [r["lr_ben_mean"] - base_b for _, r in items]
            ax.plot(xs, ys, color=color, alpha=0.4, linewidth=1.0)
        ax.scatter([], [], color=color, marker=marker, label=label, s=40,
                   edgecolors="k", linewidths=0.4)

    ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Δ refusal-LR on harmful prompts (want > 0 if injecting)")
    ax.set_ylabel("Δ refusal-LR on benign prompts (want ≈ 0)")
    ax.set_title("Steering Pareto on JBB test_in — α∈{-2,0,1,2,4}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_steer_pareto.png", dpi=140)
    plt.close(fig)


def fig6_steer_doseresponse():
    if not (STR_ / "baseline.json").exists():
        return
    base = json.load(open(STR_ / "baseline.json"))
    base_h = base["test_in"]["lr_harm_mean"]
    base_b = base["test_in"]["lr_ben_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    ax_h, ax_b = axes
    methods = {"DoM": (STR_ / "dom.json", "#000")}
    for arm in ARMS:
        path = STR_ / f"{arm}_coef_dir.json"
        if path.exists():
            methods[f"{ARM_LABEL[arm]} probe-coef"] = (path, ARM_COLOR[arm])

    for label, (path, color) in methods.items():
        d = json.load(open(path))
        items = sorted(((float(a), r) for a, r in d["test_in"].items()
                        if a not in ("ablate",) and r.get("lr_harm_mean") is not None),
                       key=lambda kv: kv[0])
        alphas = [a for a, _ in items]
        h = [r["lr_harm_mean"] - base_h for _, r in items]
        b = [r["lr_ben_mean"] - base_b for _, r in items]
        ax_h.plot(alphas, h, color=color, label=label, marker="o")
        ax_b.plot(alphas, b, color=color, label=label, marker="o")

    for ax, ttl in [(ax_h, "harmful prompts"), (ax_b, "benign prompts")]:
        ax.axhline(0, color="k", linewidth=0.6, linestyle=":")
        ax.axvline(0, color="k", linewidth=0.6, linestyle=":")
        ax.set_xlabel("inject α")
        ax.set_ylabel("Δ refusal-LR vs baseline")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Dose-response: refusal-LR vs steering magnitude")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_steer_doseresponse.png", dpi=140)
    plt.close(fig)


def main():
    fig1_detect_auc()
    fig2_roc_curves()
    fig3_b2w_boost()
    fig4_per_feat_top()
    fig5_steer_pareto()
    fig6_steer_doseresponse()
    print(f"Saved figures to {FIG}")


if __name__ == "__main__":
    main()
