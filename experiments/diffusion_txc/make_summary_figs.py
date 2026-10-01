"""Figures for the diffusion-txc arc summary (one per context).

    uv run --no-sync --with matplotlib python experiments/diffusion_txc/make_summary_figs.py

Reads the committed result JSONs where they exist (synthetic, masked TXC,
Gemma per-token, steering). The detection numbers live on the Modal volume
`diffusion-txc` (`backtracking_eval/*`), so they are transcribed here from the
tables in docs/dmitry/proposals/2026-08-11_backtracking_detection_dsm.md.

Colour convention: reconstruction = blue, denoising (DSM) = vermillion,
bayes_gate = pink, the paper's trace-trained dictionaries = green,
conventional (DoM) steering = black, controls/references = grey.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BIRD = ROOT / "experiments/bird_clock/results"
OUT = ROOT / "docs/dmitry/proposals/figures"

RECON, DSM, BAYES, PAPER, DOM, REF = "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#000000", "#9a9a9a"
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": "#888888",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def grid(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#e6e6e6", lw=0.8)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"2026-09-30_dsm_summary_{name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path.relative_to(ROOT))


def paired_bars(ax, groups, vals, seeds, width=0.36, labels=("reconstruction", "denoising (DSM)"),
                label_pad=0.012):
    """Recon/DSM bar pairs per group, with per-seed dots."""
    for j, (col, lab) in enumerate(zip((RECON, DSM), labels)):
        xs = [i + (j - 0.5) * (width + 0.03) for i in range(len(groups))]
        ax.bar(xs, [v[j] for v in vals], width, color=col, label=lab, zorder=3)
        for x, s in zip(xs, seeds):
            ax.scatter([x] * len(s[j]), s[j], s=9, color="white", edgecolor=INK,
                       linewidth=0.6, zorder=4)
        for x, v, s in zip(xs, vals, seeds):
            top = max([v[j], *s[j]])
            ax.text(x, top + label_pad, f"{v[j]:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


# ---------------------------------------------------------------- synthetic
def fig_synthetic() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw={"wspace": 0.3})

    # (a) B1: objective swap at fixed architecture, clock task, H=1024
    ax = axes[0]
    refs = load(BIRD / "b1_refs.json")
    vals, seeds = [], []
    for arch in ("head", "txc"):
        r = load(BIRD / f"b1_{arch}_recon_H1024.json")
        d = load(BIRD / f"b1_{arch}_dsm_H1024.json")
        vals.append((r["probe_acc_mean"], d["probe_acc_mean"]))
        seeds.append(([s["probe_acc"] for s in r["per_seed"]],
                      [s["probe_acc"] for s in d["per_seed"]]))
    paired_bars(ax, ["posterior head", "TopK TXC"], vals, seeds)
    for y, lab in ((refs["analytic_linear"], "analytic Bayes code"), (refs["chance"], "chance")):
        ax.axhline(y, color=REF, ls="--", lw=1, zorder=2)
        ax.text(1.62, y + 0.012, lab, ha="right", fontsize=7.5, color=MUTED)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("held-out probe accuracy")
    ax.set_title("(a) Clock: swap the loss, keep the model")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=8)
    grid(ax)

    # (b) B2b: per-band accuracy, TXC recon vs DSM
    ax = axes[1]
    groups, vals, seeds = [], [], []
    for setting in ("singlefreq", "crowded"):
        for band, label in (("low", "slow"), ("high", "fast")):
            pair_v, pair_s = [], []
            for arm in ("txc_recon", "txc_dsm"):
                d = load(BIRD / f"b2b_{setting}_{arm}.json")
                classes = [str(c) for c in d["meta"]["bands"][band]]  # label indices (b2_settings.LOW_BAND)
                per_seed = []
                for s in d["per_seed"]:
                    lab_means = [sum(pc[c] for c in classes) / len(classes)
                                 for pc in s["per_class"].values()]
                    per_seed.append(sum(lab_means) / len(lab_means))
                pair_v.append(sum(per_seed) / len(per_seed))
                pair_s.append(per_seed)
            groups.append(f"{setting}\n{label} tones")
            vals.append(tuple(pair_v))
            seeds.append(tuple(pair_s))
    paired_bars(ax, groups, vals, seeds, width=0.34, label_pad=0.003)
    ax.set_ylim(0.8, 1.035)
    ax.set_ylabel("per-class probe accuracy")
    ax.set_title("(b) TopK TXC: gain is all in slow features")
    ax.tick_params(axis="x", labelsize=8)
    grid(ax)

    # (c) B3: atom quality vs ground truth, population mean, TXC
    ax = axes[2]
    specs = [("clock", "txc_recon_k32", "txc_dsm_k32", "purity", "clock\npurity"),
             ("singlefreq", "txc_recon", "txc_dsm", "in_plane", "singlefreq\nin-plane"),
             ("crowded", "txc_recon", "txc_dsm", "in_plane", "crowded\nin-plane"),
             ("coupled", "txc_recon", "txc_dsm", "emission_purity", "coupled\nemission")]
    groups, vals, seeds, rand = [], [], [], []
    for setting, ra, da, m, lab in specs:
        r, d = load(BIRD / f"b3_{setting}_{ra}.json"), load(BIRD / f"b3_{setting}_{da}.json")
        groups.append(lab)
        vals.append((r["metrics_mean"][m], d["metrics_mean"][m]))
        seeds.append(([s[m] for s in r["per_seed"]], [s[m] for s in d["per_seed"]]))
        rand.append(r["random_baseline"][m])
    paired_bars(ax, groups, vals, seeds, width=0.34)
    for i, y in enumerate(rand):
        ax.plot([i - 0.42, i + 0.42], [y, y], color=REF, ls="--", lw=1, zorder=5)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("atom quality vs ground truth\n(dashed = random-init dictionary)")
    ax.set_title("(c) TopK TXC: cleaner atoms (not coupled)")
    ax.tick_params(axis="x", labelsize=8)
    grid(ax)
    save(fig, "synthetic")


# ---------------------------------------------------------------- masked TXC
def fig_masked() -> None:
    cells = [json.loads(l) for l in (ROOT / "results/dtxc_e1/e1b_main_cells.jsonl").open()]
    cells = [c for c in cells if c["W"] == 5 and c["lr"] == 1e-3 and c["arm"] in ("plain", "mask_rand")]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    vals, seeds = [], []
    for mode in ("fixed", "fresh"):
        pv, ps = [], []
        for arm in ("plain", "mask_rand"):
            xs = [c["imp_signal_rec"] for c in cells if c["data_mode"] == mode and c["arm"] == arm]
            pv.append(sum(xs) / len(xs))
            ps.append(xs)
        vals.append(tuple(pv))
        seeds.append(tuple(ps))
    paired_bars(ax, ["fixed episode cache\n(memorisable)", "fresh episodes\nevery batch"], vals, seeds,
                labels=("plain reconstruction", "masked position (inpainting)"))
    for t in ax.texts:  # labels for negative bars sit below them
        if float(t.get_text()) < 0:
            t.set_y(float(t.get_text()) - 0.03)
            t.set_va("top")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylim(-0.17, 0.2)
    ax.set_ylabel("masked-position signal recovery")
    ax.set_title("Reed–Solomon windows (W=5): masking helps only via memorisation")
    ax.legend(loc="upper right", fontsize=8)
    grid(ax)
    save(fig, "masked_txc")


# ---------------------------------------------------------------- Gemma per-token
def fig_gemma() -> None:
    R = ROOT / "experiments/diffusion_txc/topk_vs_topkdiff/results"

    def ev(arm, suffix=""):
        return [load(R / f"evals_{arm}_s{s}{suffix}.json") for s in (0, 1)]

    def mean(xs):
        return sum(xs) / len(xs)

    def probe5(e):
        return mean([v["k=5"] for v in e["sparse_probing"].values()])

    rows = []  # (label, recon, dsm, higher_is_better)
    for tag, suf in (("10M", ""), ("100M", "_logs_100M")):
        r, d = ev("recon", suf), ev("dsm", suf)
        rows.append((f"absorption rate ({tag})", mean([e["absorption"]["absorption_rate_mean"] for e in r]),
                     mean([e["absorption"]["absorption_rate_mean"] for e in d]), False))
        rows.append((f"active-set overlap under noise ({tag})",
                     mean([e["fragility"]["eps=0.5"]["support_jaccard"] for e in r]),
                     mean([e["fragility"]["eps=0.5"]["support_jaccard"] for e in d]), True))
        rows.append((f"sparse probing, k=5 ({tag})", mean([probe5(e) for e in r]),
                     mean([probe5(e) for e in d]), True))
    r, d = ev("recon"), ev("dsm")
    rows.append(("judged explainability (10M)",
                 mean([e["autointerp"]["detection_balanced_acc_mean"] for e in r]),
                 mean([e["autointerp"]["detection_balanced_acc_mean"] for e in d]), True))
    # Training-side numbers: topk_vs_topkdiff/README.md results table (step 6000).
    rows.append(("loss recovered (10M)", 0.908, 0.841, True))
    rows.append(("dead features (10M)", 0.076, 0.269, False))
    order = [0, 3, 1, 4, 2, 5, 6, 7, 8]
    rows = [rows[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (lab, rv, dv, up) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([rv, dv], [y, y], color="#cccccc", lw=2.2, zorder=2)
        ax.scatter(rv, y, s=46, color=RECON, zorder=3, label="reconstruction" if i == 0 else None)
        ax.scatter(dv, y, s=46, color=DSM, zorder=3, label="denoising (DSM)" if i == 0 else None)
        better = (dv > rv) == up
        verdict = "DSM better" if better else "recon better"
        if abs(dv - rv) < 0.03:
            verdict = "≈ tie"
        ax.text(1.02, y, verdict, va="center", fontsize=8,
                color=INK if verdict != "≈ tie" else MUTED,
                fontweight="bold" if better and verdict != "≈ tie" else "normal")
        arrow = "↑" if up else "↓"
        ax.text(-0.02, y, f"{lab} {arrow}", ha="right", va="center", fontsize=8.5, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("metric value (arrow = better direction)")
    ax.set_title("Gemma-2-2B L12, per-token TopK SAE (2 seeds; 10M and 100M tokens)", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=2, fontsize=8.5)
    grid(ax, "x")
    save(fig, "gemma_per_token")


# ---------------------------------------------------------------- detection
def fig_detection() -> None:
    # PR-AUC, sentence negatives, S=8; from the detection doc's tables (several runs;
    # fold spread ~±0.03, so differences of <0.02 are not resolved).
    rows = [
        ("random untrained dictionary", 0.128, REF, None),
        ("raw ln1 activations (no dictionary)", 0.190, REF, None),
        ("per-token SAE, recon (FineWeb)", 0.1905, RECON, "10% dead"),
        ("per-token SAE, DSM (FineWeb)", 0.1835, DSM, "50% dead"),
        ("window TXC, recon (FineWeb)", 0.196, RECON, "8% dead"),
        ("window TXC, DSM (FineWeb)", 0.181, DSM, "96% dead"),
        ("window TXC, recon (mixed corpus)", 0.190, RECON, "7% dead"),
        ("window TXC, DSM (mixed corpus)", 0.208, DSM, "95% dead"),
        ("  recon's top-248 latents by mass", 0.223, RECON, "selected control"),
        ("paper TXC (trained on traces)", 0.215, PAPER, None),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, (lab, v, col, note) in enumerate(rows):
        y = len(rows) - 1 - i
        hatch = "////" if "top-248" in lab else None
        ax.barh(y, v, 0.62, color=col if not hatch else "white", edgecolor=col,
                hatch=hatch, lw=1.2, zorder=3)
        ax.text(v + 0.003, y, f"{v:.3f}" + (f"   {note}" if note else ""), va="center",
                fontsize=8, color=INK)
    ax.axvline(0.126, color=REF, ls="--", lw=1, zorder=2)
    ax.text(0.127, len(rows) - 0.35, "base rate", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5)
    ax.set_xlim(0.1, 0.27)
    ax.set_xlabel("backtracking detection PR-AUC (sentence negatives, S=8)\n"
                  "'% dead' = latents that never fire on reasoning traces")
    ax.set_title("Llama-3.1-8B ln1 L10: backtracking detection", loc="left")
    grid(ax, "x")
    save(fig, "detection")


# ---------------------------------------------------------------- steering
def fig_steering() -> None:
    w1 = load(ROOT / "results/backtracking_steering/wave1/symmetry.json")
    w2 = load(ROOT / "results/backtracking_steering/wave2/symmetry.json")
    rows = [  # (label, source dict, key, colour)
        ("paper TXC, feature A (pos 0)", w1, "stageB_txc_f14621_pos0", PAPER),
        ("paper TXC, feature A (all pos)", w1, "stageB_txc_f14621_union", PAPER),
        ("paper TXC h13, feature B", w1, "stageB_txc_h13_f1183_pos0", PAPER),
        ("paper per-token SAE", w1, "stageB_topk_sae_f9876_pos0", PAPER),
        ("conventional DoM steering", w1, "dom_base_union", DOM),
        ("per-token SAE, recon", w1, "ours_recon_s2_f13776_pos0", RECON),
        ("window TXC, recon", w2, "w6_recon_f10063_pos0", RECON),
        ("per-token SAE, DSM", w1, "ours_dsm_s2_f4366_pos0", DSM),
        ("window TXC, DSM", w2, "w6_dsm_f10440_pos0_FOLD", DSM),
        ("window TXC, bayes_gate", w2, "w6_bayes_f8209_pos0_FOLD", BAYES),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for i, (lab, src, key, col) in enumerate(rows):
        y = len(rows) - 1 - i
        b = src[key]["bootstrap_excess_anti"]
        x, (lo, hi) = b["excess_anti"], b["ci95"]
        ax.barh(y, x, 0.62, color=col, alpha=1.0 if b["excludes_zero"] else 0.45, zorder=3)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.3, zorder=4)
        for e in (lo, hi):
            ax.plot([e, e], [y - 0.13, y + 0.13], color=INK, lw=1.3, zorder=4)
        ax.text(max(hi, 0) + 0.02, y, f"{x:+.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, ls="--", lw=1.1, zorder=2)
    ax.text(0.005, len(rows) - 0.4, "random direction", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5)
    ax.set_xlim(-0.32, 0.68)
    ax.set_xlabel("directional steering effect beyond a norm-matched random vector\n"
                  "(odd part of Δ backtracking-count curve, 95% bootstrap CI; faded = CI includes 0)")
    ax.set_title("DeepSeek-R1-distill-Llama-8B: backtracking steering", loc="left")
    grid(ax, "x")
    save(fig, "steering")


if __name__ == "__main__":
    fig_synthetic()
    fig_masked()
    fig_gemma()
    fig_detection()
    fig_steering()
