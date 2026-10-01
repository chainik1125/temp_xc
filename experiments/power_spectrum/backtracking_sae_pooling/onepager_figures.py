"""Paper-format C7 figures for the pooled-SAE one-pager.

Reproduces the two main-text panels of the paper (fig:c7-headline) for three
arms that share one evaluation protocol and one 20k-step training budget:
final-token TopK SAE, max-pooled TopK SAE (T=5), and TXC-base (T=5).

- Inducement: mean genuine-backtracking count gc(a, m) per question at m=0 and
  at each arm's peak-Delta-gc magnitude, bootstrap 95% CI over the 61 questions.
- Detection: sparse-probe PR-AUC at S=8, chance = positive-class prior.

Run: uv run --no-project --with matplotlib --with numpy python onepager_figures.py
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STEER = HERE / "steering_baselines" / "results" / "fresh_25mag_seed42"
OUT = HERE / "onepager"

# Paper palette (sampled from paper/figs/c7_*_compact.png); pooled SAE is new.
ARMS = {
    "topk_sae": ("TopK SAE\n(final token)", "#5579b4", "last"),
    "pooled_sae_max": ("TopK SAE\n(max-pool, T=5)", "#2a9d8f", "max"),
    "txc_base": ("TXC-base\n(T=5)", "#58447f", None),
}
UNSTEERED = "#c1c1c1"
N_BOOT = 10_000


def load_gc() -> dict[str, dict[tuple[str, float], int]]:
    gc: dict[str, dict[tuple[str, float], int]] = defaultdict(dict)
    with gzip.open(STEER / "judge_outputs.jsonl.gz", "rt") as fh:
        for line in fh:
            row = json.loads(line)
            gc[row["arch"]][(row["transcript_id"], float(row["magnitude"]))] = int(row["label"])
    return gc


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def inducement(ax: plt.Axes, stats: dict) -> None:
    order = sorted(ARMS, key=lambda a: -stats[a]["peak_mean"])
    w = 0.4
    for i, a in enumerate(order):
        s = stats[a]
        for dx, mean, ci, colour in [
            (-w / 2, s["zero_mean"], s["zero_ci"], UNSTEERED),
            (w / 2, s["peak_mean"], s["peak_ci"], ARMS[a][1]),
        ]:
            err = [[mean - ci[0]], [ci[1] - mean]]
            ax.bar(i + dx, mean, w, color=colour, yerr=err, capsize=3,
                   error_kw={"elinewidth": 1, "ecolor": "#222"})
        ax.text(i + w / 2, s["peak_ci"][1] + 0.03, f"{s['peak_mean']:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.bar(0, 0, color=UNSTEERED, label="unsteered ($m=0$)")
    ax.bar(0, 0, color="#555", label="optimal $m$")
    ticks = [f"{ARMS[a][0]}\n$m^*={stats[a]['peak_m']:+g}$" for a in order]
    ax.set_xticks(range(len(order)), ticks, fontsize=8)
    ax.set_ylim(0, 1.32)
    ax.set_ylabel("$gc(a,m)$ per question")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(a) Inducement: genuine backtracking", fontsize=9)
    style(ax)


def detection(ax: plt.Axes, prauc: dict[str, float], chance: float) -> None:
    order = sorted(ARMS, key=lambda a: -prauc[a])
    for i, a in enumerate(order):
        ax.bar(i, prauc[a], 0.6, color=ARMS[a][1])
        ax.text(i, prauc[a] + 0.003, f"{prauc[a]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(chance, ls=":", color="#555", lw=1, label=f"chance $\\approx${chance:.2f}")
    ax.set_xticks(range(len(order)), [ARMS[a][0] for a in order], fontsize=8)
    ax.set_ylabel("PR-AUC at $S=8$")
    ax.set_ylim(0, 0.23)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("(b) Detection: sparse probe", fontsize=9)
    style(ax)


def curves(ax: plt.Axes, gc: dict, summary: dict, rng: np.random.Generator) -> None:
    mags = sorted(summary["magnitudes"])
    for a, (label, colour, _) in ARMS.items():
        qs = sorted({q for q, _ in gc[a]})
        base = np.array([gc[a][(q, 0.0)] for q in qs])
        mean, lo, hi = [], [], []
        for m in mags:
            d = np.array([gc[a][(q, float(m))] for q in qs]) - base
            mean.append(d.mean())
            ci = boot_ci(d, rng)
            lo.append(ci[0])
            hi.append(ci[1])
        ax.plot(mags, mean, "-o", ms=3, color=colour, label=label.replace("\n", " "))
        ax.fill_between(mags, lo, hi, color=colour, alpha=0.15, lw=0)
    ax.axhline(0, color="#555", lw=0.6)
    ax.axvline(0, color="#555", lw=0.6)
    ax.set_xlabel("steering magnitude $m$")
    ax.set_ylabel("$\\Delta gc(a,m)$")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    gc = load_gc()
    summary = json.loads((STEER / "summary.json").read_text())
    detect = json.loads((HERE / "results" / "raw_results.json").read_text())

    stats = {}
    for a in ARMS:
        m = float(summary["arms"][a]["delta_gc_peak_magnitude"])
        qs = sorted({q for q, _ in gc[a]})
        assert len(qs) == 61
        z = np.array([gc[a][(q, 0.0)] for q in qs], dtype=float)
        p = np.array([gc[a][(q, m)] for q in qs], dtype=float)
        assert np.isclose(p.mean() - z.mean(), summary["arms"][a]["delta_gc_peak"])
        stats[a] = {
            "peak_m": m,
            "zero_mean": z.mean(), "zero_ci": boot_ci(z, rng),
            "peak_mean": p.mean(), "peak_ci": boot_ci(p, rng),
            "delta": p.mean() - z.mean(), "delta_ci": boot_ci(p - z, rng),
        }

    txc_ref = detect["references"]["models"]["txc_base"]["pr_auc"]["8"]
    prauc = {a: detect["arms"][k]["pr_auc"]["8"] for a, (_, _, k) in ARMS.items() if k}
    prauc["txc_base"] = txc_ref
    chance = detect["data"]["positive_rate"]

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 200})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    inducement(ax1, stats)
    detection(ax2, prauc, chance)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"headline.{ext}", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    curves(ax, gc, summary, rng)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"delta_gc_vs_magnitude.{ext}", bbox_inches="tight")

    (OUT / "numbers.json").write_text(json.dumps(
        {"inducement": stats, "detection_pr_auc_S8": prauc, "chance": chance}, indent=2))
    print(json.dumps({"inducement": stats, "pr_auc_S8": prauc}, indent=1))


if __name__ == "__main__":
    main()
