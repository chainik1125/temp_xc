"""Generate summary figures for the 2026-05-06 overnight chain on a40_synth_3gpu/2."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
PLOTS = Path("/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/plots/2026-05-06_overnight")
PLOTS.mkdir(parents=True, exist_ok=True)

ARCH_ORDER = ["regular_sae", "txc_base", "txcdr_t5", "txcdr_t2", "txc_pro"]
ARCH_COLORS = {
    "regular_sae": "#888888",
    "txc_base":    "#1f77b4",
    "txcdr_t5":    "#2ca02c",
    "txcdr_t2":    "#9467bd",
    "txc_pro":     "#d62728",
}


def load(pod, bench, kind="results"):
    p = ROOT / pod / bench / f"{kind}.json"
    if not p.exists():
        return None
    return json.load(open(p))


def best_per_arch(rows, metric, raw_k=10):
    """Pick rows at fixed raw_k and return {arch: best metric over sweep}."""
    out = {}
    for r in rows:
        if r.get("raw_k") != raw_k:
            continue
        a = r["model"]
        v = r.get(metric)
        if v is None:
            continue
        if a not in out or v > out[a]:
            out[a] = v
    return out


BENCHES_OVERVIEW = [
    ("synth2", "global_necessary_a_sparsity",       "GN-A sparsity"),
    ("synth2", "global_necessary_b_magnitude_noise","GN-B magnitude"),
    ("synth2", "global_necessary_c_smoothed",       "GN-C smoothed"),
    ("synth1", "bench_d_separable_smoothed",        "Bench D"),
    ("synth1", "bench_e_denoising_recon",           "Bench E"),
    ("synth1", "e1_pure_smoother",                  "E1 smoother"),
    ("synth2", "e4_zero_mean_edge",                 "E4 zero-mean"),
    ("synth2", "e4_dense_edge",                     "E4 dense"),
    ("synth1", "coupled_rho_sweep",                 "Coupled ρ"),
    ("synth1", "coupled_noisy_overlap_sweep",       "Noisy overlap"),
    ("synth1", "temporal_derivative_v2_sweep",      "Temp deriv v2"),
]


def _overview_panel(ax, metric, raw_k, ylabel, ylim=None):
    arch_keys_seen = set()
    data = {}
    for pod, b, label in BENCHES_OVERVIEW:
        rows = load(pod, b)
        if rows is None:
            continue
        per = best_per_arch(rows, metric, raw_k=raw_k)
        data[label] = per
        arch_keys_seen.update(per.keys())
    arches = [a for a in ARCH_ORDER if a in arch_keys_seen]
    labels = list(data.keys())
    x = np.arange(len(labels))
    width = 0.16
    for i, a in enumerate(arches):
        vals = [data[lab].get(a, np.nan) for lab in labels]
        ax.bar(x + (i - (len(arches)-1)/2) * width, vals,
               width, label=a, color=ARCH_COLORS.get(a, "k"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(f"raw_k={raw_k}")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 1: Bench overview — best metric per arch at each raw_k ∈ {1,2,5,10}
# ---------------------------------------------------------------------------
def fig_overview_per_k():
    """One figure per metric, with a panel per raw_k value."""
    metrics = [
        ("emission_auc",     "best eAUC (local emission)",  "overview_eauc_per_k.png",   (0, 1.05)),
        ("hidden_auc",       "best gAUC (global hidden)",   "overview_gauc_per_k.png",   (0, 1.05)),
        ("hidden_corr_mean", "best hidden corr",            "overview_hcorr_per_k.png",  (0, 1.05)),
    ]
    for metric, ylabel, fname, ylim in metrics:
        fig, axes = plt.subplots(4, 1, figsize=(13, 14))
        for ax, k in zip(axes, [1, 2, 5, 10]):
            _overview_panel(ax, metric, raw_k=k, ylabel=ylabel, ylim=ylim)
        plt.suptitle(f"{ylabel} — per arch and raw_k", fontsize=13)
        plt.tight_layout()
        out = PLOTS / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure (legacy): Bench overview at raw_k=10 — keep emission+hidden corr combined
# ---------------------------------------------------------------------------
def fig_overview():
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    _overview_panel(axes[0], "emission_auc",     raw_k=10, ylabel="best eAUC",        ylim=(0, 1.05))
    _overview_panel(axes[1], "hidden_corr_mean", raw_k=10, ylabel="best hidden corr", ylim=(0, 1.05))
    plt.suptitle("Bench overview at raw_k=10 (eAUC + hidden corr)", fontsize=12)
    plt.tight_layout()
    out = PLOTS / "overview_emission_hidden.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2: E9 DC/AC ablation — recon_clean per arch, original/dc_only/ac_only
# ---------------------------------------------------------------------------
def fig_e9_ablation():
    benches = [
        ("synth2", "global_necessary_a_sparsity",       "GN-A"),
        ("synth2", "global_necessary_b_magnitude_noise","GN-B"),
        ("synth2", "global_necessary_c_smoothed",       "GN-C"),
        ("synth1", "bench_d_separable_smoothed",        "Bench D"),
        ("synth1", "bench_e_denoising_recon",           "Bench E"),
        ("synth1", "e1_pure_smoother",                  "E1"),
        ("synth2", "e4_zero_mean_edge",                 "E4"),
        ("synth1", "coupled_rho_sweep",                 "Coupled ρ"),
        ("synth1", "coupled_noisy_overlap_sweep",       "Noisy overlap"),
        ("synth1", "temporal_derivative_v2_sweep",      "Temp deriv v2"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), sharey=False)
    axes = axes.flatten()
    for k, (pod, b, label) in enumerate(benches):
        ax = axes[k]
        rows = load(pod, b, kind="e9_ablation")
        if rows is None:
            ax.set_title(f"{label} — missing")
            ax.axis("off")
            continue
        # Aggregate per (arch, projection): mean over sweep
        from collections import defaultdict
        bucket = defaultdict(list)
        for r in rows:
            bucket[(r["arch"], r["projection"])].append(r.get("h_corr_mean"))
        arches = sorted({a for a, _ in bucket}, key=lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else 99)
        projs = ["original", "dc_only", "ac_only"]
        proj_colors = {"original": "#444", "dc_only": "#ff7f0e", "ac_only": "#1f77b4"}
        x = np.arange(len(arches))
        width = 0.27
        for i, p in enumerate(projs):
            vals = [np.mean([v for v in bucket[(a, p)] if v is not None]) if bucket[(a, p)] else np.nan for a in arches]
            ax.bar(x + (i - 1) * width, vals, width, label=p, color=proj_colors[p])
        ax.set_xticks(x)
        ax.set_xticklabels(arches, rotation=20, ha="right", fontsize=8)
        ax.set_title(label)
        ax.set_ylabel("h-corr (mean across sweep)")
        ax.grid(axis="y", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    for ax in axes[len(benches):]:
        ax.axis("off")
    plt.suptitle("E9 DC/AC ablation — hidden-state correlation per arch and projection",
                 fontsize=13)
    plt.tight_layout()
    out = PLOTS / "e9_dc_ac_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3: E1 σ sweep — recon_clean & h_corr vs σ per arch (raw_k=10)
# ---------------------------------------------------------------------------
def fig_e1_sweep():
    rows = load("synth1", "e1_pure_smoother")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    metrics = [("recon_nmse_clean", "recon NMSE (clean)", True),
               ("hidden_corr_mean", "hidden corr",        False)]
    for ax, (m, ylabel, log) in zip(axes, metrics):
        per = {}
        for r in rows:
            if r.get("raw_k") != 10:
                continue
            per.setdefault(r["model"], []).append((r["sigma"], r.get(m)))
        for arch, pts in per.items():
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "o-", label=arch, color=ARCH_COLORS.get(arch, "k"))
        ax.set_xlabel("σ_noise")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.suptitle("E1 pure smoother — σ sweep (raw_k=10)")
    plt.tight_layout()
    out = PLOTS / "e1_sigma_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4: GN A/B/C — emission AUC + hidden corr vs ρ per arch
# ---------------------------------------------------------------------------
def fig_gn_sweep():
    benches = [("global_necessary_a_sparsity",        "GN-A sparsity"),
               ("global_necessary_b_magnitude_noise", "GN-B magnitude"),
               ("global_necessary_c_smoothed",        "GN-C smoothed")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True)
    for col, (b, label) in enumerate(benches):
        rows = load("synth2", b)
        for ax_i, (m, ylab) in enumerate([
            ("emission_auc",     "emission AUC"),
            ("hidden_corr_mean", "hidden corr"),
        ]):
            ax = axes[ax_i, col]
            per = {}
            for r in rows:
                if r.get("raw_k") != 10:
                    continue
                per.setdefault(r["model"], []).append((r["rho"], r.get(m)))
            for arch, pts in per.items():
                pts.sort()
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "o-", label=arch, color=ARCH_COLORS.get(arch, "k"))
            if ax_i == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(ylab)
            if ax_i == 1:
                ax.set_xlabel("ρ")
            ax.grid(alpha=0.3)
            if ax_i == 0 and col == 0:
                ax.legend(fontsize=9)
    plt.suptitle("Global-necessary A/B/C — ρ sweep (raw_k=10)")
    plt.tight_layout()
    out = PLOTS / "gn_rho_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5: Bench D and Bench E — original headline benches
# ---------------------------------------------------------------------------
def fig_bench_d_e():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    plots = [
        ("synth1", "bench_d_separable_smoothed", "Bench D — separable smoothed"),
        ("synth1", "bench_e_denoising_recon",    "Bench E — denoising recon"),
    ]
    for col, (pod, b, label) in enumerate(plots):
        rows = load(pod, b)
        for ax_i, (m, ylab) in enumerate([
            ("emission_auc",     "emission AUC"),
            ("hidden_corr_mean", "hidden corr"),
        ]):
            ax = axes[ax_i, col]
            per = {}
            for r in rows:
                if r.get("raw_k") != 10:
                    continue
                per.setdefault(r["model"], []).append((r["rho"], r.get(m)))
            for arch, pts in per.items():
                pts.sort()
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "o-", label=arch, color=ARCH_COLORS.get(arch, "k"))
            if ax_i == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(ylab)
            if ax_i == 1:
                ax.set_xlabel("ρ")
            ax.grid(alpha=0.3)
            if ax_i == 0 and col == 0:
                ax.legend(fontsize=9)
    plt.suptitle("Bench D / Bench E — ρ sweep (raw_k=10)")
    plt.tight_layout()
    out = PLOTS / "bench_d_e.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6: E4 zero-mean vs dense edge — recon and rise_corr_mean vs ρ
# ---------------------------------------------------------------------------
def fig_e4():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    plots = [
        ("synth2", "e4_zero_mean_edge", "E4 zero-mean edge"),
        ("synth2", "e4_dense_edge",     "E4 dense edge"),
    ]
    for col, (pod, b, label) in enumerate(plots):
        rows = load(pod, b)
        for ax_i, (m, ylab) in enumerate([
            ("recon_nmse_clean", "recon NMSE (clean)"),
            ("rise_corr_mean",   "rise corr"),
        ]):
            ax = axes[ax_i, col]
            per = {}
            for r in rows:
                if r.get("raw_k") != 10:
                    continue
                per.setdefault(r["model"], []).append((r["rho"], r.get(m)))
            for arch, pts in per.items():
                pts.sort()
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "o-", label=arch, color=ARCH_COLORS.get(arch, "k"))
            if ax_i == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(ylab)
            if ax_i == 1:
                ax.set_xlabel("ρ")
            ax.grid(alpha=0.3)
            if ax_i == 0 and col == 0:
                ax.legend(fontsize=9)
    plt.suptitle("E4 zero-mean vs dense — ρ sweep (raw_k=10)")
    plt.tight_layout()
    out = PLOTS / "e4_zero_vs_dense.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_overview()
    fig_overview_per_k()
    fig_e9_ablation()
    fig_e1_sweep()
    fig_gn_sweep()
    fig_bench_d_e()
    fig_e4()
