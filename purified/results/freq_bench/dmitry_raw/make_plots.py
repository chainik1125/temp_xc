"""FrequencyBench plots — DC, AC, Mixed.

Run after pulling each bench's results.json into the per-bench dirs:
- synth1_dc/results.json (DC, 250 cells)
- synth2_ac/results.json (AC, 240 cells)
- pod3_mixed/results.json (Mixed unsigned, 220 cells)
- pod4_mixed_signed/results.json + synth1_mixed_signed/results.json (signed, merged)
"""
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PLOTS = Path("/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/plots/2026-05-06_freq_bench")
PLOTS.mkdir(parents=True, exist_ok=True)

ARCH_ORDER = ["regular_sae", "txcdr_t2", "txcdr_t5", "txc_base", "tfa",
              "tsae_attn", "tsae_bhalla"]
ARCH_COLORS = {
    "regular_sae": "#888888",
    "txcdr_t2":    "#9467bd",
    "txcdr_t5":    "#2ca02c",
    "txc_base":    "#1f77b4",
    "tfa":         "#d62728",
    "tsae_attn":   "#8c564b",
    "tsae_bhalla": "#ff7f0e",
}
ARCH_LABEL = {
    "regular_sae": "regular_sae",
    "txcdr_t2":    "txcdr (T=2)",
    "txcdr_t5":    "txcdr (T=5)",
    "txc_base":    "txc_base (T=W)",
    "tfa":         "TFA",
    "tsae_attn":   "T-SAE attn (mislabeled)",
    "tsae_bhalla": "T-SAE Bhalla",
}


def load(name):
    p = ROOT / name / "results.json"
    if not p.exists():
        return None
    return json.load(open(p))


def merge_signed():
    """Merge synth1's and pod-4's mixed-signed results (deduplicate by key)."""
    a = load("pod4_mixed_signed") or []
    b = load("synth1_mixed_signed") or []
    by_key = {}
    for r in a + b:
        k = (r["model"], r["W"], round(r["sigma"], 4), r.get("variant", "signed"), r["raw_k"])
        by_key[k] = r
    return list(by_key.values())


def merge_unsigned():
    """Merge pod-3's and synth1's mixed-unsigned results."""
    a = load("pod3_mixed") or []
    p2 = ROOT / "synth1_mixed_unsigned.json"
    b = json.load(open(p2)) if p2.exists() else []
    by_key = {}
    for r in a + b:
        k = (r["model"], r["W"], round(r["sigma"], 4),
             r.get("variant", "unsigned"), r["raw_k"])
        by_key[k] = r
    return list(by_key.values())


# ---------------------------------------------------------------------------
# DC: NTPS vs W per arch, faceted by p (raw_k=10)
# ---------------------------------------------------------------------------
def fig_dc_ntps_vs_W(rows):
    if not rows:
        return
    p_values = sorted({r["p"] for r in rows})
    fig, axes = plt.subplots(1, len(p_values), figsize=(6 * len(p_values), 4),
                             sharey=True)
    if len(p_values) == 1:
        axes = [axes]
    for ax, p in zip(axes, p_values):
        for arch in ARCH_ORDER:
            xs, ys = [], []
            for W in sorted({r["W"] for r in rows}):
                vs = [r["NTPS"] for r in rows
                      if r["W"] == W and r["p"] == p and r["model"] == arch
                      and r["raw_k"] == 10 and r["NTPS"] is not None
                      and not (isinstance(r["NTPS"], float) and math.isnan(r["NTPS"]))]
                if vs:
                    xs.append(W); ys.append(vs[0])
            if xs:
                ax.plot(xs, ys, "o-", label=ARCH_LABEL[arch],
                        color=ARCH_COLORS[arch], linewidth=2)
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.5, linewidth=1,
                   label="local ceiling (NTPS=0)")
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.5, linewidth=1,
                   label="oracle (NTPS=1)")
        ax.set_xlabel("W (window length)")
        ax.set_title(f"DC bench, p={p}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("NTPS")
    axes[0].legend(fontsize=8, loc="lower right", ncol=1)
    plt.suptitle("DC bench — NTPS vs W (raw_k=10, σ=0.1)", fontsize=13)
    plt.tight_layout()
    out = PLOTS / "dc_ntps_vs_W.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_dc_A_vs_W(rows):
    """Probe accuracy A vs W per arch (raw_k=10), with A_loc + A_oracle reference."""
    if not rows:
        return
    p_values = sorted({r["p"] for r in rows})
    fig, axes = plt.subplots(1, len(p_values), figsize=(6 * len(p_values), 4),
                             sharey=True)
    if len(p_values) == 1:
        axes = [axes]
    for ax, p in zip(axes, p_values):
        Ws = sorted({r["W"] for r in rows})
        # Reference lines
        loc_ys = []
        oracle_ys = []
        for W in Ws:
            for r in rows:
                if r["W"] == W and r["p"] == p:
                    loc_ys.append(r["A_loc"]); oracle_ys.append(r["A_oracle"])
                    break
        ax.plot(Ws, loc_ys, "k--", linewidth=1.5, alpha=0.5,
                label=f"A_loc⋆ = p")
        ax.plot(Ws, oracle_ys, "k:", linewidth=1.5, alpha=0.5,
                label="A_oracle (majority vote)")
        for arch in ARCH_ORDER:
            xs, ys = [], []
            for W in Ws:
                vs = [r["A"] for r in rows
                      if r["W"] == W and r["p"] == p and r["model"] == arch
                      and r["raw_k"] == 10]
                if vs:
                    xs.append(W); ys.append(vs[0])
            if xs:
                ax.plot(xs, ys, "o-", label=ARCH_LABEL[arch],
                        color=ARCH_COLORS[arch], linewidth=2)
        ax.set_xlabel("W")
        ax.set_title(f"DC bench, p={p}")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("probe accuracy A")
    axes[0].legend(fontsize=8, loc="lower right", ncol=1)
    plt.suptitle("DC bench — probe accuracy vs W (raw_k=10, σ=0.1)", fontsize=13)
    plt.tight_layout()
    out = PLOTS / "dc_A_vs_W.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_sae_stacked_diagnostic(rows, bench_name="dc"):
    """SAE: one-token A vs stacked-window A as a function of W."""
    if not rows:
        return
    sae_rows = [r for r in rows if r["model"] == "regular_sae"]
    if not sae_rows or all(r.get("A_stacked_sae") is None for r in sae_rows):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if "p" in sae_rows[0]:
        # DC: one panel per p
        for p in sorted({r["p"] for r in sae_rows}):
            xs, ys_one, ys_stack = [], [], []
            for W in sorted({r["W"] for r in sae_rows}):
                row = next((r for r in sae_rows
                            if r["W"] == W and r["p"] == p and r["raw_k"] == 10),
                           None)
                if row and row.get("A_stacked_sae") is not None:
                    xs.append(W); ys_one.append(row["A"])
                    ys_stack.append(row["A_stacked_sae"])
            if xs:
                ax.plot(xs, ys_one, "o-", label=f"one-token (p={p})", linewidth=2)
                ax.plot(xs, ys_stack, "s--", label=f"stacked W tokens (p={p})",
                        linewidth=2)
    ax.axhline(0.5, color="k", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("W")
    ax.set_ylabel("probe accuracy A")
    ax.set_title(f"SAE diagnostic — one-token vs stacked-window probe ({bench_name})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = PLOTS / f"{bench_name}_sae_stacked_diagnostic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# AC: NTPS vs W per arch (raw_k=10, σ=0.1), with shuffle/reverse controls
# ---------------------------------------------------------------------------
def fig_ac_ntps_vs_W(rows):
    if not rows:
        return
    sigmas = sorted({r["sigma"] for r in rows})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(6 * len(sigmas), 4),
                             sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    for ax, sigma in zip(axes, sigmas):
        for arch in ARCH_ORDER:
            xs, ys = [], []
            for W in sorted({r["W"] for r in rows}):
                vs = [r["NTPS"] for r in rows
                      if r["W"] == W and r["sigma"] == sigma and r["model"] == arch
                      and r["raw_k"] == 10 and r["NTPS"] is not None
                      and not (isinstance(r["NTPS"], float) and math.isnan(r["NTPS"]))]
                if vs:
                    xs.append(W); ys.append(vs[0])
            if xs:
                ax.plot(xs, ys, "o-", label=ARCH_LABEL[arch],
                        color=ARCH_COLORS[arch], linewidth=2)
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xlabel("W")
        ax.set_xscale("log", base=2)
        ax.set_title(f"AC bench, σ={sigma}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("NTPS")
    axes[0].legend(fontsize=8, loc="upper left", ncol=1)
    plt.suptitle("AC bench — signed-velocity NTPS vs W (raw_k=10)", fontsize=13)
    plt.tight_layout()
    out = PLOTS / "ac_ntps_vs_W.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Mixed: per-frequency response curve R_j
# ---------------------------------------------------------------------------
def fig_mixed_freq_response(rows, name):
    if not rows:
        return
    # Pick W=8 (representative mid-range), σ=0.1, raw_k=10
    target_W = 8
    target_sigma = 0.1
    target_k = 10
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arch in ARCH_ORDER:
        row = next((r for r in rows if r["W"] == target_W
                    and r["sigma"] == target_sigma and r["raw_k"] == target_k
                    and r["model"] == arch), None)
        if row is None or "per_class_acc_by_velocity" not in row:
            continue
        per = row["per_class_acc_by_velocity"]
        # x-axis = velocity, y = R_j = (A_j - 1/n) / (1 - 1/n)
        n_classes = row["n_classes"]
        denom = 1.0 - 1.0 / n_classes
        items = sorted(((float(v), a) for v, a in per.items()), key=lambda t: t[0])
        xs = [v for v, _ in items]
        ys = [(a - 1.0 / n_classes) / denom for _, a in items]
        ax.plot(xs, ys, "o-", label=ARCH_LABEL[arch], color=ARCH_COLORS[arch],
                linewidth=2)
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
    ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("velocity ω_j")
    ax.set_ylabel("R_j")
    ax.set_title(f"Mixed bench — frequency response ({name}, W={target_W}, σ={target_sigma}, raw_k={target_k})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS / f"mixed_{name}_freq_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_mixed_ntps_vs_W(rows, name):
    if not rows:
        return
    sigmas = sorted({r["sigma"] for r in rows})
    fig, axes = plt.subplots(1, len(sigmas), figsize=(6 * len(sigmas), 4),
                             sharey=True)
    if len(sigmas) == 1:
        axes = [axes]
    for ax, sigma in zip(axes, sigmas):
        for arch in ARCH_ORDER:
            xs, ys = [], []
            for W in sorted({r["W"] for r in rows}):
                vs = [r["NTPS"] for r in rows
                      if r["W"] == W and r["sigma"] == sigma
                      and r["model"] == arch and r["raw_k"] == 10
                      and r["NTPS"] is not None
                      and not (isinstance(r["NTPS"], float) and math.isnan(r["NTPS"]))]
                if vs:
                    xs.append(W); ys.append(vs[0])
            if xs:
                ax.plot(xs, ys, "o-", label=ARCH_LABEL[arch],
                        color=ARCH_COLORS[arch], linewidth=2)
        ax.axhline(0.0, color="k", linestyle="--", alpha=0.5)
        ax.axhline(1.0, color="k", linestyle=":", alpha=0.5)
        ax.set_xlabel("W")
        ax.set_xscale("log", base=2)
        ax.set_title(f"Mixed {name}, σ={sigma}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("NTPS")
    axes[0].legend(fontsize=8, loc="upper left")
    plt.suptitle(f"Mixed bench {name} — NTPS vs W (raw_k=10)", fontsize=13)
    plt.tight_layout()
    out = PLOTS / f"mixed_{name}_ntps_vs_W.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    # DC
    dc = load("synth1_dc")
    fig_dc_ntps_vs_W(dc)
    fig_dc_A_vs_W(dc)
    fig_sae_stacked_diagnostic(dc, "dc")
    # AC (when available)
    ac = load("synth2_ac")
    if ac:
        fig_ac_ntps_vs_W(ac)
    # Mixed unsigned (merged)
    mu = merge_unsigned()
    if mu:
        fig_mixed_freq_response(mu, "unsigned")
        fig_mixed_ntps_vs_W(mu, "unsigned")
    # Mixed signed (merged)
    ms = merge_signed()
    if ms:
        fig_mixed_freq_response(ms, "signed")
        fig_mixed_ntps_vs_W(ms, "signed")
