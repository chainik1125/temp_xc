"""Make c6 (Emergent Misalignment) paper figures for Qwen-7B-medical.

Outputs three PDFs into ``temp_xc_tex/figs/``:

- ``c6_em_alignment_delta_7bmed.pdf`` — bar chart: mean Δalign at coh≥70
  per arch (SAE, TXC), averaged over seeds, error bars = min/max over seeds.
- ``c6_em_detection_prauc_7bmed.pdf`` — bar chart: PR-AUC@S=16 per arch,
  averaged over seeds, error bars = min/max over seeds.
- ``c6_em_steering_grid_7bmed.pdf`` — 2×2 appendix grid: cols = arches,
  rows = seeds, each panel showing the full α frontier (mean_align +
  mean_coh, with the coh≥70 region shaded).

Inputs:

- Canonical Wang stage-4 frontier (3 finalists × 27-α grid), from
  ``temp_xc:final:purified/results/runs/c6_<train_key>/stage4_frontier.json``.
- Extended dense α-sweep (top finalist × 30-α grid extension), from
  ``temp_xc/local_data/c6_redteam/h100_em_4/sweep_outputs/c6_<train_key>/wang_full_extended.json``.
- Detection PR-AUC@S=16, from
  ``temp_xc:final:purified/results/leaderboard.jsonl``
  (filter ``component=c6, eval_protocol_version=3.0.0``).

Usage::

    python scripts/make_c6_em_figures.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_DATA = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc"
)
REPO_TEX = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc_tex"
)
# Two source of dense α-extension data:
#   - LOCAL_REDTEAM: original sweep_outputs from h100_4_em (sae_arditi + txc_base)
#   - LOCAL_OVERNIGHT: newer overnight 5-arch sweep on h100_3 / h100_em_*
LOCAL_REDTEAM   = REPO_DATA / "local_data/c6_redteam/h100_em_4/sweep_outputs"
LOCAL_OVERNIGHT = REPO_DATA / "dmitry/pre_purified/c6_em_overnight/sweep_outputs"
LOCAL_OVERNIGHT_RUNS = REPO_DATA / "dmitry/pre_purified/c6_em_overnight/runs"
FIGS_OUT = REPO_TEX / "figs"

# Qwen-7B-medical cells. The first 4 are the canonical sae_arditi / txc_base
# pair (Han's c6 sweep on origin/final). The last 4 are the overnight 5-arch
# extension run on h100_3 + h100_em_2gpu_1 + h100_em_4/5 — same datasource,
# same Wang protocol, n=2 paired seeds {1, 42}. Only steering data is present
# for the new archs (txc_pro, tsae_paper); detection eval is still in flight.
CELLS = [
    {"arch": "sae_arditi",  "seed": 1,  "train_key": "9b011dfeea88f8af"},
    {"arch": "sae_arditi",  "seed": 42, "train_key": "c0da3ed8794554a1"},
    {"arch": "txc_base",    "seed": 1,  "train_key": "2016074933c41e7f"},
    {"arch": "txc_base",    "seed": 42, "train_key": "88a4ddf6819d8057"},
    {"arch": "txc_pro",     "seed": 1,  "train_key": "e561456612fe29ff"},
    {"arch": "txc_pro",     "seed": 42, "train_key": "0689047ce9bce927"},
    {"arch": "tsae_paper",  "seed": 1,  "train_key": "6f6d047132771676"},
    {"arch": "tsae_paper",  "seed": 42, "train_key": "819604d52a131b54"},
    # TFA: only seed=1 available; bar / line / detection panels handle n=1
    # gracefully (error bar collapses to a point). Steering grid skips the
    # missing (TFA, seed=42) panel.
    {"arch": "tfa",         "seed": 1,  "train_key": "e3b029548824f240"},
]

# Architectures whose Wang stage-4 outputs are committed on origin/final
# (we read them via `git show`). Other archs are read from LOCAL_OVERNIGHT_RUNS.
CANONICAL_FINAL_ARCHS = {"sae_arditi", "txc_base"}

COH_FLOOR = 70.0
ARCH_LABEL = {
    "sae_arditi": "SAE-arditi",
    "txc_base":   "TXC-base",
    "txc_pro":    "TXC-pro",
    "tsae_paper": "T-SAE",
    "tfa":        "TFA",
}
ARCH_COLOR = {
    "sae_arditi": "#4477AA",   # blue
    "txc_base":   "#EE6677",   # red
    "txc_pro":    "#CC8800",   # orange
    "tsae_paper": "#229922",   # green
    "tfa":        "#888888",   # gray
}
ARCH_ORDER = ["sae_arditi", "txc_base", "txc_pro", "tsae_paper", "tfa"]


def git_show(branch_path: str) -> str:
    """Read a file from origin/<branch> using ``git show``. Path includes branch."""
    return subprocess.check_output(
        ["git", "-C", str(REPO_DATA), "show", branch_path], text=True
    )


def load_canonical(train_key: str, arch: str) -> dict:
    """Load Wang stage4_frontier.json for one cell.

    For sae_arditi / txc_base cells the canonical sweep is committed on
    origin/final under purified/results/runs/c6_<tk>/stage4_frontier.json.
    For txc_pro / tsae_paper cells the data lives in the overnight 5-arch
    sweep at temp_xc/dmitry/pre_purified/c6_em_overnight/runs/c6_<tk>/.
    """
    if arch in CANONICAL_FINAL_ARCHS:
        txt = git_show(f"origin/final:purified/results/runs/c6_{train_key}/stage4_frontier.json")
        return json.loads(txt)
    p = LOCAL_OVERNIGHT_RUNS / f"c6_{train_key}" / "stage4_frontier.json"
    if not p.exists():
        raise FileNotFoundError(f"stage4_frontier.json not found for {arch}/{train_key}: {p}")
    return json.loads(p.read_text())


def load_extended(train_key: str, arch: str) -> dict | None:
    """Load the dense α-sweep extension for one cell (local-only).

    Try the redteam sweep first (h100_4_em), then the overnight sweep."""
    for root in (LOCAL_REDTEAM, LOCAL_OVERNIGHT):
        p = root / f"c6_{train_key}" / "wang_full_extended.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def load_detection_prauc_s16() -> dict[str, float]:
    """Map train_key → PR-AUC@S=16 for c6 cells from leaderboard.jsonl."""
    return {tk: m["pr_auc_S16"] for tk, m in load_detection_metrics().items()
            if "pr_auc_S16" in m}


S_GRID = (1, 2, 4, 8, 16, 32)
S_ALL = 32768  # extension point: probe over the full d_sae


DETECT_LOCAL = REPO_DATA / "local_data/c6_redteam/h100_em_4/extended_detection_S_all"


def load_detection_metrics() -> dict[str, dict]:
    """Map train_key → metrics dict for c6 detection. Prefer the local
    extended sweep at h100_em_4:.../extended_detection_S_all/c6_<tk>/pr_auc.json
    (covers all 4 7B-medical archs, includes S=32768); fall back to the
    git-committed leaderboard rows for any train_key not in the local mirror.
    Output schema is normalised to {pr_auc_S<k>: float, positive_rate, n_sent}."""
    out: dict[str, dict] = {}

    # Local extended sweep (paper-canonical for the figure).
    if DETECT_LOCAL.exists():
        for cell_dir in sorted(DETECT_LOCAL.iterdir()):
            if not cell_dir.is_dir():
                continue
            f = cell_dir / "pr_auc.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            tk = d.get("cell", {}).get("train_key")
            if not tk:
                tk = cell_dir.name.removeprefix("c6_")
            metrics = {f"pr_auc_S{S}": float(v) for S, v in d.get("pr_auc", {}).items()}
            metrics["positive_rate"] = float(d.get("positive_rate", float("nan")))
            metrics["n_sent"] = (d.get("encode_shape") or [None])[0]
            metrics["_source"] = "extended_detection_S_all"
            out[tk] = metrics

    # Leaderboard fallback (for cells without a local extended pr_auc.json).
    lb = git_show("origin/final:purified/results/leaderboard.jsonl")
    for line in lb.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            d.get("component") == "c6"
            and d.get("eval_protocol_version", "").startswith("3.")
        ):
            tk = d.get("train_key")
            if tk and tk not in out:
                m = dict(d.get("metrics", {}))
                m["_source"] = "leaderboard"
                out[tk] = m
    return out


def per_cell_rows(canonical: dict, extended: dict | None) -> list[dict]:
    """Combine all (feature_id, alpha, mean_align, mean_coh) rows across canonical + extended.
    Skip rows where the judge returned no align/coh (rare, on some extended-α cells)."""
    def _row(r):
        a = r.get("mean_align")
        c = r.get("mean_coh")
        if a is None or c is None:
            return None
        return {
            "feature_id": r["feature_id"],
            "alpha": float(r["alpha"]),
            "mean_align": float(a),
            "mean_coh": float(c),
        }
    rows = []
    for f in canonical["finalists"]:
        for r in f["rows"]:
            x = _row(r)
            if x is not None:
                rows.append(x)
    if extended:
        for r in extended.get("rows", []):
            x = _row(r)
            if x is not None:
                rows.append(x)
    return rows


def alpha0_baseline(canonical: dict) -> float:
    """Mean align at α=0 across the 3 finalists (same prompt, no steering — should be ~equal).
    Skip rows where the judge returned None."""
    vals = []
    for f in canonical["finalists"]:
        for r in f["rows"]:
            if abs(r["alpha"]) < 1e-9 and r.get("mean_align") is not None:
                vals.append(float(r["mean_align"]))
    return float(np.mean(vals)) if vals else float("nan")


def delta_at_coh_floor(rows: list[dict], coh_floor: float) -> float:
    """Alignment dynamic range within the coh ≥ coh_floor region:
    (max align across eligible (finalist, α) cells) − (min align across the same cells).
    Measures how far steering can push alignment in either direction while keeping the
    model coherent. NaN if no rows clear the floor."""
    eligible = [r["mean_align"] for r in rows if r["mean_coh"] >= coh_floor]
    if not eligible:
        return float("nan")
    return max(eligible) - min(eligible)


# ── Build per-cell metrics ────────────────────────────────────────────

def build_table() -> list[dict]:
    detection_metrics = load_detection_metrics()
    table = []
    for cell in CELLS:
        canon = load_canonical(cell["train_key"], cell["arch"])
        ext = load_extended(cell["train_key"], cell["arch"])
        rows = per_cell_rows(canon, ext)
        baseline = alpha0_baseline(canon)
        delta70 = delta_at_coh_floor(rows, COH_FLOOR)
        m = detection_metrics.get(cell["train_key"], {})
        prauc = float(m.get("pr_auc_S16", float("nan")))
        prauc_by_S = {S: float(m.get(f"pr_auc_S{S}", float("nan"))) for S in S_GRID}
        prauc_by_S[S_ALL] = float(m.get(f"pr_auc_S{S_ALL}", float("nan")))
        positive_rate = float(m.get("positive_rate", float("nan")))
        # Surface max-align and min-align cells at coh≥70 for the appendix grid.
        eligible = [r for r in rows if r["mean_coh"] >= COH_FLOOR]
        max_at_70 = max(eligible, key=lambda r: r["mean_align"]) if eligible else None
        min_at_70 = min(eligible, key=lambda r: r["mean_align"]) if eligible else None
        table.append({
            **cell,
            "baseline_align": baseline,
            "delta70": delta70,
            "max_at_coh70": max_at_70,
            "min_at_coh70": min_at_70,
            "pr_auc_S16": prauc,
            "pr_auc_by_S": prauc_by_S,
            "positive_rate": positive_rate,
            "rows": rows,
            "canonical": canon,
            "extended": ext,
        })
        print(
            f"  {ARCH_LABEL[cell['arch']]:12} seed={cell['seed']:2} | "
            f"α=0 align={baseline:5.2f} | "
            f"max@coh≥{COH_FLOOR:.0f}={max_at_70['mean_align']:5.2f} | "
            f"min@coh≥{COH_FLOOR:.0f}={min_at_70['mean_align']:5.2f} | "
            f"Δalign={delta70:5.2f} | "
            f"PR-AUC@S=16={prauc:.3f}"
        )
    return table


# ── Plots ─────────────────────────────────────────────────────────────

def _seed_stats(rows_for_arch: list[dict], key: str) -> tuple[float, float, float]:
    vals = np.array([r[key] for r in rows_for_arch], dtype=float)
    return float(vals.mean()), float(vals.min()), float(vals.max())


def plot_alignment_delta(table: list[dict], out_path: Path) -> None:
    archs = [a for a in ARCH_ORDER if any(r["arch"] == a for r in table)]
    means, lows, highs = [], [], []
    for a in archs:
        m, lo, hi = _seed_stats([r for r in table if r["arch"] == a], "delta70")
        means.append(m)
        lows.append(m - lo)
        highs.append(hi - m)

    # Wider figure to fit 4 bars cleanly.
    fig, ax = plt.subplots(figsize=(4.3, 3.2), dpi=200)
    x = np.arange(len(archs))
    bars = ax.bar(
        x, means, yerr=[lows, highs], capsize=8, width=0.6,
        color=[ARCH_COLOR[a] for a in archs], edgecolor="black", linewidth=0.7,
        error_kw=dict(ecolor="black", lw=1.0, capthick=1.0),
    )
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width()/2, m + max(highs) * 0.05,
            f"{m:+.1f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs], fontsize=9.5,
                       rotation=15, ha="right")
    ax.set_ylabel(rf"$\Delta\,$align at coh$\geq${COH_FLOOR:.0f}  (max $-$ min)",
                  fontsize=10)
    ax.set_title("Alignment dynamic range", fontsize=11)
    ax.axhline(0, color="black", lw=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    print(f"  → {out_path.name} (+ .png)")


def plot_detection_prauc(table: list[dict], out_path: Path) -> None:
    """Saturation curve: sparse-probe PR-AUC across S ∈ {1,2,4,8,16,32}, mean over
    seeds with min/max band. Per-arch chance level (= mean positive rate across
    seeds) drawn as a horizontal dashed line. S=1 ≈ best single feature; S=2 ≈
    top-2 features in joint LR; etc.

    Only archs with detection data committed in `purified/results/leaderboard.jsonl`
    are plotted; archs whose detection eval is still in flight (TXC-pro, T-SAE
    in this version) are listed in a panel-level note instead."""
    fig, ax = plt.subplots(figsize=(3.8, 3.2), dpi=200)
    Ss_topS = list(S_GRID)              # canonical sparse-probe sweep
    Ss_all  = Ss_topS + [S_ALL]         # extended with S = d_sae

    archs_with_data = []
    archs_without_data = []
    for a in ARCH_ORDER:
        cells = [c for c in table if c["arch"] == a]
        if not cells:
            continue
        if all(np.isnan(c["pr_auc_by_S"][16]) for c in cells):
            archs_without_data.append(a)
            continue
        archs_with_data.append(a)
        # Plot through whatever S values have non-NaN data for ALL cells of this arch.
        Ss_arch = [S for S in Ss_all
                   if all(not np.isnan(c["pr_auc_by_S"].get(S, float("nan"))) for c in cells)]
        ys_mean = [float(np.mean([c["pr_auc_by_S"][S] for c in cells])) for S in Ss_arch]
        ys_min  = [float(np.min ([c["pr_auc_by_S"][S] for c in cells])) for S in Ss_arch]
        ys_max  = [float(np.max ([c["pr_auc_by_S"][S] for c in cells])) for S in Ss_arch]
        chance  = float(np.mean([c["positive_rate"] for c in cells]))

        ax.fill_between(Ss_arch, ys_min, ys_max, color=ARCH_COLOR[a], alpha=0.18, lw=0)
        ax.plot(Ss_arch, ys_mean, "-o", color=ARCH_COLOR[a], lw=1.4, ms=4.0,
                label=ARCH_LABEL[a])

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ss_all)
    ax.set_xticklabels([str(S) if S != S_ALL else r"all" for S in Ss_all],
                       fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("$S$  (top features used by probe)", fontsize=10)
    ax.set_ylabel("Sparse-probe PR-AUC", fontsize=10)
    ax.set_title("Misalignment detection", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    if archs_without_data:
        missing = ", ".join(ARCH_LABEL[a] for a in archs_without_data)
        ax.text(0.02, 0.97, f"{missing}: detection in flight",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="#555555", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    print(f"  → {out_path.name} (+ .png)")


def plot_steering_grid(table: list[dict], out_path: Path) -> None:
    """2x2 grid: cols = arch, rows = seed. Per panel: per-finalist α-frontier of
    mean_align (solid lines) + top-finalist mean_coh (dashed). Coh≥70 region shaded
    in green tint for visual reference; peak-at-coh≥70 marked with a star."""
    archs = [a for a in ARCH_ORDER if any(c["arch"] == a for c in table)]
    seeds = [1, 42]
    fig, axes = plt.subplots(
        nrows=len(seeds), ncols=len(archs), figsize=(3.0 * len(archs), 5.8),
        dpi=200, sharex=True, sharey=True, squeeze=False,
    )

    align_colors = ["#222222", "#7777bb", "#aaaaaa"]  # top → bottom finalist (align)
    coh_color = "#cc8800"

    handles_for_legend: list = []
    for r, seed in enumerate(seeds):
        for c, arch in enumerate(archs):
            cell = next((t for t in table
                         if t["arch"] == arch and t["seed"] == seed), None)
            ax = axes[r, c]
            if cell is None:
                ax.set_title(f"{ARCH_LABEL[arch]}, seed={seed}", fontsize=10)
                ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="#888888",
                        style="italic")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            # Shade the coh≥70 reference band (this is just on the y-axis, since coh
            # and align share the 0-100 scale here).
            ax.axhspan(COH_FLOOR, 100, color="#88cc88", alpha=0.10, zorder=0)
            # group rows by feature_id
            by_feat: dict[int, list[dict]] = {}
            for row in cell["rows"]:
                by_feat.setdefault(row["feature_id"], []).append(row)
            ranked_finalists = [f["feature_id"] for f in cell["canonical"]["finalists"]]
            for i, feat in enumerate(ranked_finalists):
                rows = sorted(by_feat.get(feat, []), key=lambda x: x["alpha"])
                if not rows:
                    continue
                xs = [x["alpha"] for x in rows]
                ys = [x["mean_align"] for x in rows]
                cs = [x["mean_coh"] for x in rows]
                col = align_colors[min(i, 2)]
                lw_align = 1.4 if i == 0 else 0.9
                ms_align = 3.5 if i == 0 else 2.0
                lh, = ax.plot(
                    xs, ys, "-o", color=col, lw=lw_align, ms=ms_align,
                    label=f"feat {feat} align (rank {i+1})", zorder=3,
                )
                if r == 0 and c == 0:
                    handles_for_legend.append(lh)
                if i == 0:  # show coh line only for the top finalist
                    ch, = ax.plot(
                        xs, cs, "--", color=coh_color, lw=1.0, alpha=0.85,
                        label=f"feat {feat} coh", zorder=2,
                    )
                    if r == 0 and c == 0:
                        handles_for_legend.append(ch)
            # α=0 baseline (mean across finalists)
            bh = ax.axhline(
                cell["baseline_align"], color="#229922", lw=1.0, ls=":",
                alpha=0.9, label=r"$\alpha\!=\!0$ align baseline",
            )
            if r == 0 and c == 0:
                handles_for_legend.append(bh)
            # coh=70 reference
            ch_ref = ax.axhline(
                COH_FLOOR, color=coh_color, lw=0.8, ls="-.", alpha=0.6,
                label=f"coh={COH_FLOOR:.0f} threshold",
            )
            if r == 0 and c == 0:
                handles_for_legend.append(ch_ref)
            # Mark max- and min-align cells within coh≥70 (the two endpoints whose
            # difference defines Δalign).
            mx = cell.get("max_at_coh70")
            mn = cell.get("min_at_coh70")
            if mx:
                ax.plot(
                    mx["alpha"], mx["mean_align"], marker="^", color="red",
                    markersize=11, markeredgecolor="black", markeredgewidth=0.7,
                    zorder=5, label="max align at coh≥70",
                )
                ax.annotate(
                    f"  α={mx['alpha']:.0f}\n  align={mx['mean_align']:.1f}",
                    xy=(mx["alpha"], mx["mean_align"]),
                    xytext=(8, -2), textcoords="offset points",
                    fontsize=7.5, color="black",
                )
            if mn:
                ax.plot(
                    mn["alpha"], mn["mean_align"], marker="v", color="blue",
                    markersize=11, markeredgecolor="black", markeredgewidth=0.7,
                    zorder=5, label="min align at coh≥70",
                )
                ax.annotate(
                    f"  α={mn['alpha']:.0f}\n  align={mn['mean_align']:.1f}",
                    xy=(mn["alpha"], mn["mean_align"]),
                    xytext=(8, 4), textcoords="offset points",
                    fontsize=7.5, color="black",
                )
            # Δalign annotation in panel corner
            if mx and mn:
                ax.text(
                    0.02, 0.97,
                    rf"$\Delta\,$align = {mx['mean_align'] - mn['mean_align']:.1f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="gray", alpha=0.85),
                )
            # Cosmetics
            ax.set_title(f"{ARCH_LABEL[arch]}, seed={seed}", fontsize=10)
            ax.grid(linestyle=":", alpha=0.4)
            ax.set_xlim(-220, 220)
            ax.set_ylim(0, 105)
            if r == len(seeds) - 1:
                ax.set_xlabel(r"steering coefficient $\alpha$", fontsize=10)
            if c == 0:
                ax.set_ylabel("mean align / mean coh (%)", fontsize=10)

    # Single shared legend below the figure.
    legend_labels = [h.get_label() for h in handles_for_legend]
    # Drop the per-feat labels (they vary per panel) and add generic ones.
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=align_colors[0], lw=1.4, marker="o", ms=3.5,
                       label="top finalist  align"),
            plt.Line2D([0], [0], color=align_colors[1], lw=0.9, marker="o", ms=2.0,
                       label="2nd finalist  align"),
            plt.Line2D([0], [0], color=align_colors[2], lw=0.9, marker="o", ms=2.0,
                       label="3rd finalist  align"),
            plt.Line2D([0], [0], color=coh_color, lw=1.0, ls="--",
                       label="top finalist  coh"),
            plt.Line2D([0], [0], color="#229922", lw=1.0, ls=":",
                       label=r"$\alpha\!=\!0$ align baseline"),
            plt.Line2D([0], [0], color=coh_color, lw=0.8, ls="-.",
                       label=f"coh={COH_FLOOR:.0f} threshold"),
            plt.Line2D([0], [0], marker="^", color="red", lw=0,
                       markersize=11, markeredgecolor="black",
                       label=r"max align where coh$\geq$70"),
            plt.Line2D([0], [0], marker="v", color="blue", lw=0,
                       markersize=11, markeredgecolor="black",
                       label=r"min align where coh$\geq$70"),
        ],
        loc="lower center", ncol=4, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Qwen-7B-medical — full Wang stage-4 $\\alpha$ frontier "
        "(canonical 27-$\\alpha$ + extended $|\\alpha|\\!\\in\\!\\{110\\ldots200\\}$); "
        "all three stage-4 finalists shown.",
        fontsize=10, y=0.99,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path.name} (+ .png)")


def plot_pareto_frontier(table: list[dict], out_path: Path) -> None:
    """Coherence/alignment Pareto frontier (matches the format of em-nanda's
    ``frontier_sae_vs_txc.png``): x = mean coh, y = mean align, color = α
    (diverging colormap), marker shape = arch, edge style = seed. One curve per
    (arch, seed) drawn through the top stage-4 finalist's full α frontier
    (canonical 27-α grid + dense ±110…±200 extension)."""
    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=200)
    arch_marker = {"sae_arditi": "o", "txc_base": "s", "txc_pro": "D",
                   "tsae_paper": "^", "tfa": "v"}
    seed_edge = {1: "black", 42: "#444444"}
    seed_alpha = {1: 1.0, 42: 0.55}

    # global α range for shared colormap
    all_alphas = []
    for cell in table:
        ranked = [f["feature_id"] for f in cell["canonical"]["finalists"]]
        top_feat = ranked[0]
        all_alphas.extend(r["alpha"] for r in cell["rows"] if r["feature_id"] == top_feat)
    norm = mpl.colors.TwoSlopeNorm(vmin=min(all_alphas), vcenter=0.0, vmax=max(all_alphas))
    cmap = mpl.cm.coolwarm_r

    sc = None
    for cell in table:
        ranked = [f["feature_id"] for f in cell["canonical"]["finalists"]]
        top_feat = ranked[0]
        rows = sorted(
            (r for r in cell["rows"] if r["feature_id"] == top_feat),
            key=lambda r: r["alpha"],
        )
        if not rows:
            continue
        xs = [r["mean_coh"] for r in rows]
        ys = [r["mean_align"] for r in rows]
        cs = [r["alpha"] for r in rows]
        # Faded line connecting α-ordered points within each (arch, seed).
        ax.plot(
            xs, ys, "-", color=ARCH_COLOR[cell["arch"]],
            alpha=0.25 * seed_alpha[cell["seed"]], lw=0.9, zorder=1,
        )
        sc = ax.scatter(
            xs, ys, c=cs, cmap=cmap, norm=norm,
            marker=arch_marker[cell["arch"]], s=55,
            edgecolor=seed_edge[cell["seed"]],
            linewidth=0.9 if cell["seed"] == 1 else 0.45,
            alpha=seed_alpha[cell["seed"]], zorder=3,
            label=f"{ARCH_LABEL[cell['arch']]}, seed={cell['seed']} (feat {top_feat})",
        )
        # Mark α=0 with a black star, same convention as em-nanda's frontier_sae_vs_txc.
        zero = next((r for r in rows if abs(r["alpha"]) < 1e-9), None)
        if zero:
            ax.scatter(
                [zero["mean_coh"]], [zero["mean_align"]],
                marker="*", s=130, c="black", zorder=4,
            )

    # coh=70 vertical reference line
    ax.axvline(COH_FLOOR, color="#cc8800", lw=0.9, ls="-.", alpha=0.7,
               label=f"coh = {COH_FLOOR:.0f}")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r"steering coefficient $\alpha$")
    ax.set_xlabel("mean coherence (Anthropic Claude Haiku 4.5 judge)")
    ax.set_ylabel("mean alignment (Anthropic Claude Haiku 4.5 judge)")
    ax.set_title("Coherence / alignment frontier on Qwen-2.5-7B bad-medical (L15)\n"
                 "top stage-4 finalist per (arch, seed); ★ marks $\\alpha\\!=\\!0$",
                 fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=200)
    plt.close(fig)
    print(f"  → {out_path.name} (+ .png)")


def main():
    print("== loading per-cell metrics ==")
    table = build_table()
    print()
    print("== writing figures ==")
    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    plot_alignment_delta(table, FIGS_OUT / "c6_em_alignment_delta_7bmed.pdf")
    plot_detection_prauc(table, FIGS_OUT / "c6_em_detection_prauc_7bmed.pdf")
    plot_steering_grid(table, FIGS_OUT / "c6_em_steering_grid_7bmed.pdf")
    plot_pareto_frontier(table, FIGS_OUT / "c6_em_pareto_frontier_7bmed.pdf")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
