"""Render paper-ready C3 sparse-probing plots in c7-paper style.

Outputs (no embedded titles, captions live in the .tex; matches the visual
language of c7_paper_renderer.py and rlhf_paper_renderer.py):

  c3_auc_by_k.png            — mean SAEBench-36 AUC vs k_feat, all archs incl
                                TFA, log-x.
  c3_auc_by_k_no_tfa.png     — same, TFA dropped so the y-axis can zoom on the
                                temporal-family / TopK-SAE separation.
  c3_per_task_heatmap.png    — per-arch × per-task ROC-AUC at k_feat=160,
                                with dataset-group separators.

Aggregation matches purified/experiments/c3_probing/analysis.py:
  - filter component=c3, drop smoke, keep eval_protocol_version=1.1.0
  - drop the 2 cross-token tasks (winogrande/wsc) → 36-task headline
  - txc_base is split into T-variants via config.json arch_hparams_override.T
  - dedup by (arch_label, k_feat, seed) keeping max mean_auc

Usage (defaults resolve to in-repo canonical paths)::

    cd purified
    .venv/bin/python -m scripts.c3_paper_renderer
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── c7-paper style (verbatim from c7_paper_renderer.py header) ─────────
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
})

# ── Canonical palette (extends c7_paper_renderer's PAPER_ARCH_COLOR with
#    txc_base T-sweep variants — same purple hue, distinct linestyles).
PAPER_ARCH_COLOR = {
    "topk_sae":     "#4C72B0",  # blue
    "tsae_paper":   "#55A868",  # green
    "mlc":          "#C44E52",  # red
    "txc_base_T5":  "#8172B2",  # purple
    "txc_base_T10": "#8172B2",
    "txc_base_T20": "#8172B2",
    "txc_pro":      "#CCB974",  # gold
    "tfa":          "#777777",  # grey
}

ARCH_LINESTYLE = {
    "topk_sae":     "-",
    "tsae_paper":   "-",
    "mlc":          "-",
    "txc_base_T5":  "-",
    "txc_base_T10": "--",
    "txc_base_T20": ":",
    "txc_pro":      "-",
    "tfa":          "-",
}

ARCH_MARKER = {
    "topk_sae":     "D",
    "tsae_paper":   "s",
    "mlc":          "o",
    "txc_base_T5":  "P",
    "txc_base_T10": "P",
    "txc_base_T20": "P",
    "txc_pro":      "^",
    "tfa":          "v",
}

ARCH_DISPLAY = {
    "topk_sae":     "TopK-SAE",
    "tsae_paper":   "T-SAE",
    "mlc":          "MLC",
    "txc_base_T5":  "TXC-base ($T{=}5$)",
    "txc_base_T10": "TXC-base ($T{=}10$)",
    "txc_base_T20": "TXC-base ($T{=}20$)",
    "txc_pro":      "TXC-pro",
    "tfa":          "TFA",
}

# Order = best→worst at peak k (matches c3.md). Drives plot z-order
# (later = on top) and legend listing.
ARCH_ORDER = (
    "mlc",
    "tsae_paper",
    "txc_base_T20",
    "txc_base_T10",
    "txc_base_T5",
    "txc_pro",
    "topk_sae",
    "tfa",
)

# 2 cross-token tasks dropped from the 36-task headline.
CT_TASKS = {"winogrande_correct_completion", "wsc_coreference"}
N_HEADLINE_TASKS = 36
CANONICAL_ARCHS = {"mlc", "tsae_paper", "topk_sae", "tfa", "txc_pro", "txc_base"}
CANONICAL_PROTOCOL = "1.1.0"
K_GRID = (5, 10, 20, 40, 80, 160, 320, 640)

# IT-only datasources for the headline (BASE replication deferred per c3.md
# 2026-05-04 deadline override). Anything not matching is dropped.
CANONICAL_DATASOURCES = {
    "gemma_2_2b_it_l13_fineweb_24k128",
    "gemma_2_2b_it_l11to15_fineweb_24k128",  # MLC multi-layer
}

# Per-arch canonical training_cfg, from c3.md "Setup". Missing fields are
# don't-cares. Rows whose train_key's config.json doesn't match are
# dropped so pre-canonical experiment cells (different bs / steps /
# train_window_size) don't pollute the headline.
CANONICAL_TRAINING_CFG = {
    "topk_sae":   {"batch_size": 1024, "n_steps": 20_000, "train_window_size": 1},
    "tsae_paper": {"batch_size": 1024, "n_steps": 20_000, "train_window_size": 2},
    "mlc":        {"batch_size": 1024, "n_steps": 20_000},
    "tfa":        {"batch_size": 32,   "n_steps": 20_000},
    "txc_pro":    {"batch_size": 1024, "n_steps": 20_000},
    "txc_base":   {"batch_size": 1024, "n_steps": 20_000},
}


def is_canonical_config(cfg: dict, arch: str) -> bool:
    """Return True iff the training_cfg matches the canonical setup for arch.
    Returns True if the config file is missing (some MLC checkpoints are
    HF-only and have no committed config; rely on the datasource filter)."""
    if not cfg:
        return True
    tc = cfg.get("training_cfg") or {}
    spec = CANONICAL_TRAINING_CFG.get(arch, {})
    for k, want in spec.items():
        got = tc.get(k)
        # Special-case train_window_size: paper canonical for TXC/MLC/TFA
        # leaves it at the arch's default (None or unset). We only enforce
        # it where c3.md explicitly fixes a value (topk_sae=1, tsae=2).
        if got != want:
            return False
    return True


def load_config(train_key: str, ckpt_dir: Path) -> dict:
    """Read checkpoints/<train_key>/config.json (empty dict if missing)."""
    p = ckpt_dir / train_key / "config.json"
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def resolve_txc_base_label(cfg: dict) -> str:
    """Map a txc_base config to txc_base_T{5|10|20}."""
    override = (cfg.get("training_cfg") or {}).get("arch_hparams_override") or {}
    T = override.get("T")
    if T in (10, 20):
        return f"txc_base_T{T}"
    return "txc_base_T5"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def aggregate(
    rows: list[dict],
    ckpt_dir: Path,
) -> dict[str, dict[int, dict]]:
    """Returns summary[arch_label][k_feat] = {mean_auc, std_seeds, n_seeds}."""
    grouped: dict[tuple[str, int, int], dict] = {}
    for r in rows:
        if r.get("component") != "c3":
            continue
        if r.get("eval_cfg", {}).get("smoke", False):
            continue
        if r.get("eval_protocol_version") != CANONICAL_PROTOCOL:
            continue
        if r.get("arch") not in CANONICAL_ARCHS:
            continue
        if r.get("datasource") not in CANONICAL_DATASOURCES:
            continue
        k_feat = r.get("eval_cfg", {}).get("k_feat")
        if not isinstance(k_feat, int):
            continue
        # Recompute mean_auc over the 36 SAEBench tasks (drop CT_TASKS).
        per_task = {
            k[5:]: float(v)
            for k, v in r.get("metrics", {}).items()
            if k.startswith("auc__")
        }
        per_task_36 = {t: v for t, v in per_task.items() if t not in CT_TASKS}
        if len(per_task_36) < N_HEADLINE_TASKS:
            continue
        mean_auc = mean(per_task_36.values())
        # Pull config.json once for both canonical filter + T-variant label.
        cfg = load_config(r["train_key"], ckpt_dir)
        if not is_canonical_config(cfg, r["arch"]):
            continue
        # Resolve arch label (txc_base → txc_base_T{n}).
        arch_label = (
            resolve_txc_base_label(cfg) if r["arch"] == "txc_base" else r["arch"]
        )
        seed = r.get("seed")
        # Dedup by (arch, k, seed) keeping max mean_auc — covers the
        # rare case where the same cell got re-eval'd.
        key = (arch_label, k_feat, seed)
        if key not in grouped or mean_auc > grouped[key]["mean_auc"]:
            grouped[key] = {"mean_auc": mean_auc, "seed": seed}

    summary: dict[str, dict[int, dict]] = defaultdict(dict)
    by_arch_k: dict[tuple[str, int], list[float]] = defaultdict(list)
    # Per-seed map preserved alongside the aggregated summary so
    # downstream callers (e.g. error-bar plots) can recompute seed-level
    # spreads without re-loading the leaderboard.
    per_seed: dict[str, dict[int, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (arch, k, seed), agg in grouped.items():
        by_arch_k[(arch, k)].append(agg["mean_auc"])
        per_seed[arch][k][int(seed) if seed is not None else -1] = float(
            agg["mean_auc"]
        )
    for (arch, k), aucs in by_arch_k.items():
        summary[arch][k] = {
            "n_seeds": len(aucs),
            "mean_auc": float(mean(aucs)),
            "std_seeds_auc": float(stdev(aucs)) if len(aucs) > 1 else 0.0,
        }
    # Stash the per-seed map on the summary dict so the plot fns can
    # reach it without changing every call signature.
    summary["__per_seed__"] = per_seed  # type: ignore[assignment]
    return summary


def _save_png_pdf(fig, out_stem: Path) -> None:
    """Write {out_stem}.png and {out_stem}.pdf — both at savefig.dpi/bbox."""
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))


def _trapezoidal_mean(xs: list[int], ys: list[float]) -> float:
    """Mean of y over the log-x range, computed by trapezoidal integration.

    Equivalent to ``np.trapz(y, log10(x)) / (log10(x_max) - log10(x_min))``.
    Gives a single $\\overline{\\mathrm{AUC}}$ summary across the k_feat
    sweep that weights all decades equally rather than biasing toward
    the dense end of the (log-spaced) grid.
    """
    if len(xs) < 2:
        return float(ys[0]) if ys else float("nan")
    x = np.log10(np.asarray(xs, dtype=float))
    y = np.asarray(ys, dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _plot_auc_of_auc(
    summary: dict[str, dict[int, dict]],
    out_stem: Path,
    *,
    archs: tuple[str, ...],
) -> None:
    """Horizontal bar chart of per-arch $\\overline{\\mathrm{AUC}}$.

    Bar = mean across seeds of the per-seed trapezoidal mean of mean_auc
    over log(k_feat). Error bars span the seed min/max range at that
    arch's per-seed trapezoidal value (so the bar reflects scientific
    signal and the error bar reflects the seed-level reproducibility).
    Bars sorted best→worst, color-matched to PAPER_ARCH_COLOR.
    """
    per_seed = summary.get("__per_seed__", {})  # type: ignore[arg-type]

    bars: list[tuple[str, float, float, float, int]] = []
    for arch in archs:
        if arch not in summary or arch == "__per_seed__":
            continue
        # Per-seed trapezoidal mean: reorganise
        # per_seed[arch][k_feat][seed] → seed → {k_feat: mean_auc} so we
        # can do one sweep per seed.
        per_seed_sweeps = _swap_levels(per_seed.get(arch, {}))
        per_seed_values: list[float] = []
        for seed, by_k in per_seed_sweeps.items():
            xs = sorted(by_k.keys())
            if len(xs) < 2:
                continue
            ys = [by_k[k] for k in xs]
            per_seed_values.append(_trapezoidal_mean(xs, ys))
        if not per_seed_values:
            # Fall back to seed-mean trapezoidal (single bar, no error).
            kpoints = sorted(summary[arch].items())
            xs = [k for k, _ in kpoints]
            ys = [v["mean_auc"] for _, v in kpoints]
            if not xs:
                continue
            v = _trapezoidal_mean(xs, ys)
            bars.append((arch, v, v, v, 0))
            continue
        bar_h = float(np.mean(per_seed_values))
        bars.append(
            (arch, bar_h, float(min(per_seed_values)),
             float(max(per_seed_values)), len(per_seed_values))
        )
    if not bars:
        return
    bars.sort(key=lambda ab: ab[1], reverse=True)

    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(bars) + 1.0))
    y_pos = np.arange(len(bars))[::-1]  # best at the top

    for (arch, score, lo, hi, n_seeds), y in zip(bars, y_pos):
        ax.barh(y, score,
                color=PAPER_ARCH_COLOR.get(arch, "#333"),
                edgecolor="black", linewidth=0.6, alpha=0.92,
                label=ARCH_DISPLAY.get(arch, arch))
        # Asymmetric whisker from seed min/max — only when ≥2 seeds.
        if n_seeds >= 2 and (hi - lo) > 0:
            ax.errorbar(
                score, y,
                xerr=[[score - lo], [hi - score]],
                fmt="none", ecolor="#222", capsize=3, elinewidth=0.9,
                zorder=5,
            )
        # Numeric label sits past the upper whisker so it stays
        # readable even when the error bar is wide.
        text_x = max(score, hi) + 0.004
        ax.text(text_x, y, f"{score:.3f}",
                va="center", ha="left", fontsize=9, color="#222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([ARCH_DISPLAY.get(a, a) for a, *_ in bars])
    ax.set_xlabel(r"$\overline{\mathrm{AUC}}$ "
                  r"(trapezoidal mean over $\log k_{\mathrm{feat}}$, "
                  r"seeds 1/2/42)")
    # x-range chosen so the smallest bar is still readable but the
    # spread between archs (the actual scientific signal) dominates.
    score_min = min(s for _, s, *_ in bars)
    score_max = max(s for _, s, *_ in bars)
    # Account for whisker tips when sizing the x-range.
    whisker_max = max((hi for *_, hi, _ in bars), default=score_max)
    pad = max(0.02, 0.05 * (score_max - score_min))
    ax.set_xlim(
        max(0.0, score_min - 4 * pad),
        min(1.0, max(score_max, whisker_max) + 5 * pad),
    )
    ax.tick_params(axis="y", which="both", left=False, length=0)
    ax.grid(axis="x", linewidth=0.6, alpha=0.25)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    _save_png_pdf(fig, out_stem)
    plt.close(fig)


def _swap_levels(by_k_seed: dict[int, dict[int, float]]
                 ) -> dict[int, dict[int, float]]:
    """Reorganise ``{k_feat: {seed: mean_auc}}`` into
    ``{seed: {k_feat: mean_auc}}`` so we can compute per-seed sweeps."""
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for k, seed_map in by_k_seed.items():
        for seed, val in seed_map.items():
            out[seed][k] = val
    return out


def _plot_curves(
    summary: dict[str, dict[int, dict]],
    out_path: Path,
    *,
    archs: tuple[str, ...],
) -> None:
    """Single-panel AUC vs k_feat line plot, c7-paper style.

    Legend lives outside the axes on the right so it never overlaps a
    curve. No embedded title — caption goes in the .tex.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    for arch in archs:
        if arch not in summary:
            continue
        kpoints = sorted(summary[arch].items())
        if not kpoints:
            continue
        xs = [k for k, _ in kpoints]
        ys = [v["mean_auc"] for _, v in kpoints]
        es = [v["std_seeds_auc"] for _, v in kpoints]
        ax.errorbar(
            xs, ys, yerr=es,
            color=PAPER_ARCH_COLOR.get(arch, "#333"),
            linestyle=ARCH_LINESTYLE.get(arch, "-"),
            marker=ARCH_MARKER.get(arch, "o"),
            markersize=5,
            linewidth=1.7,
            capsize=2.5,
            elinewidth=1.0,
            label=ARCH_DISPLAY.get(arch, arch),
        )

    ax.set_xscale("log")
    ax.set_xticks(list(K_GRID))
    ax.set_xticklabels([str(k) for k in K_GRID])
    ax.set_xlabel(r"$k_{\mathrm{feat}}$ (top-$k$ features by class-mean diff)")
    ax.set_ylabel("Mean AUC across SAEBench ($n{=}36$)")
    ax.minorticks_off()
    # Legend outside the axes, on the right. No overlap with any curve.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
        handlelength=2.4,
        labelspacing=0.5,
    )
    fig.tight_layout()
    _save_png_pdf(fig, out_path.with_suffix(""))
    plt.close(fig)


# 38-task panel grouped by source dataset. Order matches c3.md / appendix.
DATASET_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("BiB-1",       "Bias in Bios (set 1)",      ("bias_in_bios_set1_prof0", "bias_in_bios_set1_prof1", "bias_in_bios_set1_prof2", "bias_in_bios_set1_prof6", "bias_in_bios_set1_prof9")),
    ("BiB-2",       "Bias in Bios (set 2)",      ("bias_in_bios_set2_prof11", "bias_in_bios_set2_prof13", "bias_in_bios_set2_prof14", "bias_in_bios_set2_prof18", "bias_in_bios_set2_prof19")),
    ("BiB-3",       "Bias in Bios (set 3)",      ("bias_in_bios_set3_prof20", "bias_in_bios_set3_prof21", "bias_in_bios_set3_prof22", "bias_in_bios_set3_prof25", "bias_in_bios_set3_prof26")),
    ("Amazon-cat",  "Amazon (category)",         ("amazon_reviews_cat1", "amazon_reviews_cat2", "amazon_reviews_cat3", "amazon_reviews_cat5", "amazon_reviews_cat6")),
    ("Amazon-sent", "Amazon (sentiment)",        ("amazon_reviews_sentiment_1star", "amazon_reviews_sentiment_5star")),
    ("GitHub",      "GitHub Code",               ("github_code_C", "github_code_HTML", "github_code_Java", "github_code_PHP", "github_code_Python")),
    ("AG News",     "AG News",                   ("ag_news_business", "ag_news_scitech", "ag_news_sports", "ag_news_world")),
    ("Europarl",    "Europarl",                  ("europarl_de", "europarl_en", "europarl_es", "europarl_fr", "europarl_nl")),
    ("CT",          "Cross-token",               ("winogrande_correct_completion", "wsc_coreference")),
)


def _short_task_label(task: str) -> str:
    """Compact x-axis label per task — strips the dataset prefix."""
    if task.startswith("bias_in_bios_set"):
        # e.g. bias_in_bios_set1_prof9 → "p9"
        return "p" + task.rsplit("prof", 1)[-1]
    if task.startswith("amazon_reviews_cat"):
        return "c" + task[len("amazon_reviews_cat"):]
    if task.startswith("amazon_reviews_sentiment_"):
        # "1star" → "1*", "5star" → "5*"
        return task[len("amazon_reviews_sentiment_"):].replace("star", r"$\star$")
    if task.startswith("github_code_"):
        return task[len("github_code_"):]
    if task.startswith("ag_news_"):
        return task[len("ag_news_"):]
    if task.startswith("europarl_"):
        return task[len("europarl_"):].upper()
    if task == "winogrande_correct_completion":
        return "winog"
    if task == "wsc_coreference":
        return "wsc"
    return task


def _plot_per_task_heatmap(
    rows: list[dict],
    ckpt_dir: Path,
    *,
    out_path: Path,
    archs: tuple[str, ...],
    k_feat: int,
) -> None:
    """Per-arch × per-task ROC-AUC heatmap at fixed k_feat. Cells = mean over
    seeds. Dataset groups separated by black vertical lines."""
    # 1) Build a flat task list + dataset-boundary indices.
    tasks: list[str] = []
    boundaries: list[int] = [0]
    short_names: list[str] = []
    long_names: list[str] = []
    for short, long_, ds_tasks in DATASET_GROUPS:
        tasks.extend(ds_tasks)
        boundaries.append(len(tasks))
        short_names.append(short)
        long_names.append(long_)

    # 2) Filter rows to canonical / IT-only / k_feat / per-arch and bucket.
    by_arch_task: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r.get("component") != "c3":
            continue
        if r.get("eval_cfg", {}).get("smoke", False):
            continue
        if r.get("eval_protocol_version") != CANONICAL_PROTOCOL:
            continue
        if r.get("arch") not in CANONICAL_ARCHS:
            continue
        if r.get("datasource") not in CANONICAL_DATASOURCES:
            continue
        if r.get("eval_cfg", {}).get("k_feat") != k_feat:
            continue
        cfg = load_config(r["train_key"], ckpt_dir)
        if not is_canonical_config(cfg, r["arch"]):
            continue
        arch_label = (
            resolve_txc_base_label(cfg) if r["arch"] == "txc_base" else r["arch"]
        )
        for k, v in r.get("metrics", {}).items():
            if not k.startswith("auc__"):
                continue
            tname = k[len("auc__"):]
            by_arch_task.setdefault((arch_label, tname), []).append(float(v))

    # 3) Build the matrix (n_archs × n_tasks). Mean over seeds; NaN if missing.
    matrix = np.full((len(archs), len(tasks)), np.nan)
    for i, arch in enumerate(archs):
        for j, task in enumerate(tasks):
            vals = by_arch_task.get((arch, task), [])
            if vals:
                matrix[i, j] = float(np.mean(vals))

    # 4) Render. Cell-value labels inside each cell; dataset separators.
    fig, ax = plt.subplots(figsize=(14.0, 0.55 * len(archs) + 1.6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.40, vmax=1.00)

    # Cell-value labels — drop the "0." for compactness, white on dark cells.
    n_archs = len(archs)
    n_tasks = len(tasks)
    for i in range(n_archs):
        for j in range(n_tasks):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v < 0.72 else "black"
            ax.text(j, i, f"{v:.2f}".lstrip("0"),
                    ha="center", va="center", fontsize=6.5, color=txt_color)

    # Dataset-group vertical separators (between, not at edges).
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color="black", linewidth=1.0)

    # Axis tick labels.
    ax.set_yticks(range(n_archs))
    ax.set_yticklabels([ARCH_DISPLAY[a] for a in archs])
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([_short_task_label(t) for t in tasks],
                       rotation=90, fontsize=7)

    # Group labels above the heatmap.
    y_top = -0.62
    for k, short in enumerate(short_names):
        start = boundaries[k]
        end = boundaries[k + 1]
        mid = (start + end - 1) / 2.0
        ax.text(mid, y_top, short,
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="#222")

    # Hide grid / spines for cleanliness — heatmap stands alone.
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, length=2)
    ax.tick_params(axis="y", which="both", left=False, length=0)

    # Colorbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    cbar.set_label(rf"ROC-AUC at $k_{{\mathrm{{feats}}}}\!=\!{k_feat}$",
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _save_png_pdf(fig, out_path.with_suffix(""))
    plt.close(fig)


def main(*, leaderboard: Path, checkpoints_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(leaderboard)
    summary = aggregate(rows, checkpoints_dir)
    archs_present = [a for a in ARCH_ORDER if a in summary]
    if not archs_present:
        print(f"[c3_paper] no canonical c3 rows in {leaderboard}")
        return

    # Filenames match the Overleaf snippet (paired subfigure layout).
    # Plot 1: full sweep with TFA on the same axes.
    _plot_curves(
        summary, output_dir / "c3_sparse_probing_curves_with_tfa_gemma_it.png",
        archs=tuple(archs_present),
    )
    # Plot 2: TFA excluded → y-axis can zoom on the cluster. This is the
    # subfigure (b) in the main-paper SP figure.
    archs_no_tfa = tuple(a for a in archs_present if a != "tfa")
    _plot_curves(
        summary, output_dir / "c3_sparse_probing_curves_gemma_it.png",
        archs=archs_no_tfa,
    )
    # Plot 3: $\overline{\mathrm{AUC}}$ across the feature-budget sweep —
    # subfigure (a) in the main-paper SP figure.
    _plot_auc_of_auc(
        summary,
        output_dir / "c3_sparse_probing_auc_of_auc_gemma_it",
        archs=tuple(archs_present),
    )
    # Plot 4: per-task heatmap at k_feat=160 (appendix).
    _plot_per_task_heatmap(
        rows, checkpoints_dir,
        out_path=output_dir / "c3_per_task_heatmap.png",
        archs=tuple(archs_present),
        k_feat=160,
    )
    # Print the headline table for spot-checking.
    print(f"[c3_paper] {len(rows)} rows → {len(archs_present)} archs")
    cols = list(K_GRID)
    head = "arch".ljust(20) + "  " + "  ".join(f"k={k:<5}" for k in cols)
    print(head)
    for a in archs_present:
        cells = []
        for k in cols:
            v = summary[a].get(k)
            cells.append(f"{v['mean_auc']:.3f}" if v else "  -  ")
        print(ARCH_DISPLAY[a].ljust(20) + "  " + "  ".join(c.ljust(7) for c in cells))


def _purified_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cli() -> None:
    root = _purified_root()
    ap = argparse.ArgumentParser(description=(
        "C3 (sparse probing) paper figure renderer. "
        "Defaults to in-repo canonical paths."
    ))
    ap.add_argument(
        "--leaderboard", type=Path,
        default=root / "results" / "leaderboard.jsonl",
        help="Leaderboard jsonl (default: purified/results/leaderboard.jsonl).",
    )
    ap.add_argument(
        "--checkpoints-dir", type=Path,
        default=root / "checkpoints",
        help="Directory containing <train_key>/config.json files (default: purified/checkpoints/).",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=root / "figs" / "c3",
        help="Output directory (default: purified/figs/c3/).",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(
        leaderboard=args.leaderboard,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    cli()
