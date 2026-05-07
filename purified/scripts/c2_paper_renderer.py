"""Render paper-ready C2 synthetic-bench plots in c7-paper style.

Outputs (no embedded titles, captions live in the .tex; matches the visual
language of c7_paper_renderer.py / c3_paper_renderer.py):

  c2_synth_global_headline.{png,pdf}    — Per-arch best global-recovery
                                           bars across both benches on a
                                           shared [0,1] scale. Denoising
                                           bar = R^2_global, Coupling bar
                                           = gAUC. Each bar is the
                                           seed-mean of the arch's best
                                           (T, k_pos) cell; error bars
                                           min/max over seeds.
  c2_setup_b_singlelatent.{png,pdf}     — Setup B local-vs-global single-
                                           latent correlation scatter
                                           (gamma = 0.25). Reads
                                           denoising_probe_results.json.
  c2_setup_d_scatter_clean.{png,pdf}    — Setup D pB05_np10 eauc-vs-gauc
                                           dictionary scatter, "clean"
                                           variant: scatter points only,
                                           no per-arch trail lines.

Backward-compat aliases also written under the older filenames
``c2_noisy_singlelatent_scatter.{png,pdf}`` and
``c2_setup_d_np10_scatter.{png,pdf}`` (with-trail variant for the
latter) so existing main.tex includes don't break.

Usage::

    .venv/bin/python -m scripts.c2_paper_renderer \\
        --setup-b-json /tmp/c2_render/setup_b_results.json \\
        --c2-leaderboard /tmp/c2_render/c2_leaderboard.jsonl \\
        --output-dir /workspace/aniket/temp_xc_paper/purified/docs/aniket/figs/c2
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

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

# ── Palette (c7-consistent for shared archs; new shades for stacked_sae/
#    tfa_pos which only appear in c2). txc_base T-sweep variants share the
#    purple base hue, differentiated by colormap shade + marker.
PURPLES = plt.get_cmap("Purples")
TXC_BASE_T_VALUES = (2, 4, 5, 6, 8, 10, 12)
TXC_BASE_T_SHADE = {
    2:  PURPLES(0.40),
    4:  PURPLES(0.55),
    5:  PURPLES(0.68),  # canonical T = 5 — c7 paper purple #8172B2 is approx this
    6:  PURPLES(0.75),
    8:  PURPLES(0.82),
    10: PURPLES(0.90),
    12: PURPLES(0.97),
}
TXC_BASE_T_MARKER = {
    2: "o", 4: "^", 5: "*", 6: "v", 8: "<", 10: ">", 12: "p",
}

ARCH_STYLE = {
    "topk_sae":          {"color": "#4C72B0", "marker": "D", "label": "TopK-SAE"},
    "tsae_paper":        {"color": "#55A868", "marker": "s", "label": "T-SAE"},
    "tfa_pos":           {"color": "#777777", "marker": "X", "label": "TFA-pos"},
    "stacked_sae_T2":    {"color": "#64B5CD", "marker": "h", "label": "Stacked-SAE $T{=}2$"},
    "stacked_sae_T5":    {"color": "#3A7C8C", "marker": "H", "label": "Stacked-SAE $T{=}5$"},
    "txc_pro":           {"color": "#CCB974", "marker": "*", "label": "TXC-pro"},
}


def _txc_base_style(T: int) -> dict:
    return {
        "color": TXC_BASE_T_SHADE.get(T, "#8172B2"),
        "marker": TXC_BASE_T_MARKER.get(T, "^"),
        "label": rf"TXC-base $T{{=}}{T}$",
    }


# Fixed legend order: temporal family first by T (small → large), per-token archs
# at the bottom. Matches the ARCH_ORDER of plot_headline.py.
def _legend_order_b(present_keys: set[str]) -> list[str]:
    order: list[str] = []
    for T in TXC_BASE_T_VALUES:
        k = f"txc_base_T{T}"
        if k in present_keys:
            order.append(k)
    for k in ("tsae_paper", "stacked_sae_T5", "stacked_sae_T2",
              "txc_pro", "tfa_pos", "topk_sae"):
        if k in present_keys:
            order.append(k)
    return order


def _arch_key_b(arch: str, t_label: str) -> str | None:
    """Map (arch, t_label) → ARCH_STYLE key for Setup B / D."""
    if arch == "topk_sae":
        return "topk_sae"
    if arch == "tsae_paper":
        return "tsae_paper"
    if arch == "tfa_pos":
        return "tfa_pos"
    if arch == "txc_pro":
        return "txc_pro"
    if arch == "stacked_sae":
        if t_label == "T=2":
            return "stacked_sae_T2"
        if t_label in ("default", "T=5"):
            return "stacked_sae_T5"
        return None
    if arch == "txc_base":
        if t_label == "default":
            T_val = 5
        else:
            try:
                T_val = int(t_label.split("=")[1])
            except (ValueError, IndexError):
                T_val = 5
        return f"txc_base_T{T_val}"
    return None  # exclude


def _style_for(key: str) -> dict | None:
    if key.startswith("txc_base_T"):
        T = int(key[len("txc_base_T"):])
        return _txc_base_style(T)
    return ARCH_STYLE.get(key)


def _save_png_pdf(fig, out_stem: Path) -> None:
    """Write {out_stem}.png and {out_stem}.pdf — both at savefig.dpi/bbox."""
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))


# ── Setup B — single-latent local-vs-global scatter ─────────────────────


def _aggregate_setup_b(records: list[dict]) -> dict:
    """Mean over seeds, keyed by (arch_key, k_pos)."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        ak = _arch_key_b(r["arch_name"], r["t_label"])
        if ak is None:
            continue
        grouped[(ak, int(r["k_pos"]))].append(r)
    return {
        key: {
            "local":  float(np.mean([r["sl_mean_local"]  for r in rs])),
            "global": float(np.mean([r["sl_mean_global"] for r in rs])),
            "n":      len(rs),
        }
        for key, rs in grouped.items()
    }


def render_setup_b_scatter(records: list[dict], out_path: Path) -> None:
    agg = _aggregate_setup_b(records)
    if not agg:
        print("[c2_paper] no Setup B records — skipping")
        return
    keys_present = {k for (k, _) in agg.keys()}
    legend_order = _legend_order_b(keys_present)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    # y = x reference line.
    lim_hi = max(
        max(v["local"]  for v in agg.values()),
        max(v["global"] for v in agg.values()),
    ) + 0.015
    ax.plot([0, lim_hi], [0, lim_hi], color="#666666",
            linestyle="--", linewidth=1.0, alpha=0.7,
            label=r"$y = x$ (no denoising)")

    for ak in legend_order:
        s = _style_for(ak)
        if s is None:
            continue
        xs, ys = [], []
        for (k_ak, _kpos), v in agg.items():
            if k_ak == ak:
                xs.append(v["local"])
                ys.append(v["global"])
        if not xs:
            continue
        ax.scatter(xs, ys, color=s["color"], marker=s["marker"],
                   s=72, alpha=0.85, edgecolors="black", linewidths=0.4,
                   label=s["label"], zorder=5)

    ax.set_xlim(-0.01, lim_hi)
    ax.set_ylim(-0.01, lim_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Local correlation $\bar r_{\mathrm{local}}$ "
                  r"($z_j \to s_i$, noisy obs)")
    ax.set_ylabel(r"Global correlation $\bar r_{\mathrm{global}}$ "
                  r"($z_j \to h_i$, hidden state)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, ncol=1, handlelength=1.6,
              labelspacing=0.45)
    fig.tight_layout()
    _save_png_pdf(fig, out_path.with_suffix(""))
    plt.close(fig)


# ── Setup D — eauc-vs-gauc scatter, np10 datasource ─────────────────────

ZOOM_CUTOFF_TS = "2026-05-06T22:54:30Z"
SETUP_D_DATASOURCE = "toy_coupled_noisy_K10_M20_d256_pB05_np10"


def _load_setup_d_per_cell(rows: list[dict]) -> dict[str, list[tuple[int, float, float]]]:
    """Filter c2 rows to Setup D np10 zoom+tsweep, dedup by eval_key (latest),
    group by arch_key → list of (k_pos, eauc, gauc)."""
    latest: dict[str, dict] = {}
    for d in rows:
        if d.get("component") != "c2":
            continue
        if d.get("datasource") != SETUP_D_DATASOURCE:
            continue
        ec = d.get("eval_cfg") or {}
        if ec.get("hunt_phase") not in ("zoom", "tsweep"):
            continue
        if ec.get("smoke") is True:
            continue
        if ec.get("hunt_phase") == "zoom" and d["ts"] < ZOOM_CUTOFF_TS:
            continue
        ek = d.get("eval_key")
        if ek and (ek not in latest or d["ts"] > latest[ek]["ts"]):
            latest[ek] = d

    by_arch: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for d in latest.values():
        ec = d.get("eval_cfg") or {}
        ak = _arch_key_b(d["arch"], ec.get("t_label", "default"))
        if ak is None:
            continue
        if ak == "txc_pro":
            # plot_headline excludes txc_pro overrides from Setup D —
            # only canonical T_max=10 exists, and we follow the same convention.
            continue
        kp = ec.get("k_pos")
        eauc = d["metrics"].get("eauc")
        gauc = d["metrics"].get("gauc")
        if kp is None or eauc is None or gauc is None:
            continue
        by_arch[ak][int(kp)].append((float(eauc), float(gauc)))

    # Mean over seeds per (arch, k_pos).
    out: dict[str, list[tuple[int, float, float]]] = {}
    for ak, by_k in by_arch.items():
        out[ak] = sorted(
            [
                (kp, float(np.mean([e for e, _ in vs])),
                     float(np.mean([g for _, g in vs])))
                for kp, vs in by_k.items()
            ],
            key=lambda x: x[0],
        )
    return out


def render_setup_d_np10_scatter(rows: list[dict], out_path: Path,
                                *, draw_trail: bool = True) -> None:
    data = _load_setup_d_per_cell(rows)
    if not data:
        print("[c2_paper] no Setup D records — skipping")
        return
    keys_present = set(data.keys())
    legend_order = _legend_order_b(keys_present)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--",
            linewidth=1.0, alpha=0.7,
            label=r"$y = x$ (equal local/global alignment)")

    for ak in legend_order:
        s = _style_for(ak)
        if s is None:
            continue
        cells = data.get(ak, [])
        if not cells:
            continue
        ks  = [c[0] for c in cells]
        xs  = [c[1] for c in cells]
        ys  = [c[2] for c in cells]
        if draw_trail:
            # Trail connecting cells in k-order so reader can read trajectory.
            ax.plot(xs, ys, color=s["color"], alpha=0.40, linewidth=1.0, zorder=3)
        ax.scatter(xs, ys, color=s["color"], marker=s["marker"],
                   s=72, alpha=0.90, edgecolors="black", linewidths=0.4,
                   label=s["label"], zorder=5)

    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"eAUC (local emission recovery, vs $f_l$)")
    ax.set_ylabel(r"gAUC (global hidden recovery, vs $f_g$)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, ncol=1, handlelength=1.6,
              labelspacing=0.45)
    fig.tight_layout()
    _save_png_pdf(fig, out_path.with_suffix(""))
    plt.close(fig)


# ── Global headline — best per-arch recovery on both benches ───────────


def _best_per_arch_b(records: list[dict],
                     metric: str = "lp_mean_global_r2"
                     ) -> dict[str, dict]:
    """For each arch_key, return the best (T, k_pos) cell's seed-mean +
    seed min/max for the Denoising bench. Records come from
    denoising_probe_results.json (one row per (arch, T, k_pos, seed))."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        ak = _arch_key_b(r["arch_name"], r["t_label"])
        if ak is None:
            continue
        v = r.get(metric)
        if v is None:
            continue
        grouped[(ak, int(r["k_pos"]))].append(float(v))
    # Per cell: seed-mean + seed min/max.
    cell_stats: dict[tuple[str, int], dict] = {}
    for key, vs in grouped.items():
        if not vs:
            continue
        cell_stats[key] = {
            "seed_mean": float(np.mean(vs)),
            "seed_min":  float(min(vs)),
            "seed_max":  float(max(vs)),
            "n_seeds":   len(vs),
        }
    # Per arch_key: best cell by seed_mean.
    best: dict[str, dict] = {}
    for (ak, kp), st in cell_stats.items():
        if ak not in best or st["seed_mean"] > best[ak]["seed_mean"]:
            best[ak] = {**st, "best_k_pos": kp}
    return best


def _best_per_arch_d(rows: list[dict], metric: str = "gauc"
                     ) -> dict[str, dict]:
    """Coupling bench (Setup D pB05_np10) analogue of _best_per_arch_b."""
    latest: dict[str, dict] = {}
    for d in rows:
        if d.get("component") != "c2":
            continue
        if d.get("datasource") != SETUP_D_DATASOURCE:
            continue
        ec = d.get("eval_cfg") or {}
        if ec.get("smoke") is True:
            continue
        if ec.get("hunt_phase") not in ("zoom", "tsweep"):
            continue
        if ec.get("hunt_phase") == "zoom" and d["ts"] < ZOOM_CUTOFF_TS:
            continue
        ek = d.get("eval_key")
        if ek and (ek not in latest or d["ts"] > latest[ek]["ts"]):
            latest[ek] = d

    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for d in latest.values():
        ec = d.get("eval_cfg") or {}
        ak = _arch_key_b(d["arch"], ec.get("t_label", "default"))
        if ak is None:
            continue
        if ak == "txc_pro":
            # plot_headline excludes txc_pro overrides from Setup D.
            continue
        kp = ec.get("k_pos")
        v = d["metrics"].get(metric)
        if kp is None or v is None:
            continue
        grouped[(ak, int(kp))].append(float(v))

    cell_stats: dict[tuple[str, int], dict] = {}
    for key, vs in grouped.items():
        cell_stats[key] = {
            "seed_mean": float(np.mean(vs)),
            "seed_min":  float(min(vs)),
            "seed_max":  float(max(vs)),
            "n_seeds":   len(vs),
        }
    best: dict[str, dict] = {}
    for (ak, kp), st in cell_stats.items():
        if ak not in best or st["seed_mean"] > best[ak]["seed_mean"]:
            best[ak] = {**st, "best_k_pos": kp}
    return best


# Visual style for the two benches (color, hatch optional). Keep simple
# and rely on the per-arch bar color so the bench distinction shows
# through brightness/alpha.
_BENCH_DENOISING = {"label": r"Denoising ($R^2_{\mathrm{global}}$)",
                    "alpha": 0.95, "edge": "black", "hatch": ""}
_BENCH_COUPLING  = {"label": r"Coupling ($g\mathrm{AUC}$)",
                    "alpha": 0.55, "edge": "black", "hatch": "//"}

# Headline plot collapses the T-sweep arch variants into a single
# "architecture" — caption is "best (T, k_pos) cell", so T is a
# free knob. Maps detailed arch_key → display label + canonical color.
HEADLINE_ARCH_BASE: dict[str, tuple[str, str]] = {
    # txc_base T-sweep all collapse to one TXC-base entry
    "txc_base_T2":     ("txc_base", "#8172B2"),
    "txc_base_T4":     ("txc_base", "#8172B2"),
    "txc_base_T5":     ("txc_base", "#8172B2"),
    "txc_base_T6":     ("txc_base", "#8172B2"),
    "txc_base_T8":     ("txc_base", "#8172B2"),
    "txc_base_T10":    ("txc_base", "#8172B2"),
    "txc_base_T12":    ("txc_base", "#8172B2"),
    # stacked_sae T variants collapse
    "stacked_sae_T2":  ("stacked_sae", "#64B5CD"),
    "stacked_sae_T5":  ("stacked_sae", "#64B5CD"),
    # one-arch keys pass through with a label remap
    "txc_pro":         ("txc_pro",    "#CCB974"),
    "tsae_paper":      ("tsae_paper", "#55A868"),
    "tfa_pos":         ("tfa_pos",    "#777777"),
    "topk_sae":        ("topk_sae",   "#4C72B0"),
}
HEADLINE_ARCH_DISPLAY = {
    "txc_base":    "TXC-base",
    "txc_pro":     "TXC-pro",
    "tsae_paper":  "T-SAE",
    "stacked_sae": "Stacked-SAE",
    "tfa_pos":     "TFA-pos",
    "topk_sae":    "TopK-SAE",
}
# Order on the x-axis (best→worst at peak, c7 convention).
HEADLINE_ARCH_ORDER = (
    "txc_base", "txc_pro", "tsae_paper", "stacked_sae", "topk_sae", "tfa_pos",
)


def _collapse_to_base(per_detailed: dict[str, dict]) -> dict[str, dict]:
    """Collapse detailed arch_keys (txc_base_T5 etc.) to base archs by
    picking the best-cell across T-variants for each base. Each value
    is ``{seed_mean, seed_min, seed_max, n_seeds, best_k_pos, best_T}``.
    """
    out: dict[str, dict] = {}
    for ak, st in per_detailed.items():
        if ak not in HEADLINE_ARCH_BASE:
            continue
        base, _color = HEADLINE_ARCH_BASE[ak]
        if base not in out or st["seed_mean"] > out[base]["seed_mean"]:
            # Carry the T-variant info so tooltips / future analysis
            # can recover which T the arch's best cell came from.
            out[base] = {**st, "best_arch_variant": ak}
    return out


def render_global_headline(setup_b_records: list[dict],
                           c2_rows: list[dict],
                           out_path: Path) -> None:
    """Per-architecture best-cell bars on a shared [0,1] axis. T-sweep
    variants of an arch (e.g. TXC-base T=2..12, Stacked-SAE T=2/5)
    collapse into ONE bar per architecture — the best (T, k_pos) cell.
    Left bar within each arch group = Denoising R^2_global, right bar
    = Coupling gAUC. Error bars span seed min/max at that best cell."""
    best_d_detailed = _best_per_arch_b(setup_b_records,
                                        metric="lp_mean_global_r2")
    best_c_detailed = _best_per_arch_d(c2_rows, metric="gauc")
    best_d = _collapse_to_base(best_d_detailed)
    best_c = _collapse_to_base(best_c_detailed)

    # X-axis: keep canonical order, drop archs absent from BOTH benches.
    arch_order = [a for a in HEADLINE_ARCH_ORDER
                  if a in best_d or a in best_c]
    if not arch_order:
        print("[c2_paper] no archs present — skipping global headline")
        return

    n = len(arch_order)
    x = np.arange(n)
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    for i, base in enumerate(arch_order):
        # Pick a representative color for the base arch from the
        # collapsed mapping (any T-variant has the same color).
        color = next((c for ak, (b, c) in HEADLINE_ARCH_BASE.items()
                      if b == base), "#444")
        # Denoising bar (left).
        if base in best_d:
            st = best_d[base]
            ax.bar(x[i] - width/2, st["seed_mean"], width=width,
                   color=color, edgecolor=_BENCH_DENOISING["edge"],
                   linewidth=0.7, alpha=_BENCH_DENOISING["alpha"],
                   hatch=_BENCH_DENOISING["hatch"])
            ax.errorbar(x[i] - width/2, st["seed_mean"],
                        yerr=[[st["seed_mean"] - st["seed_min"]],
                              [st["seed_max"] - st["seed_mean"]]],
                        fmt="none", ecolor="#222", capsize=3,
                        elinewidth=0.9)
        # Coupling bar (right).
        if base in best_c:
            st = best_c[base]
            ax.bar(x[i] + width/2, st["seed_mean"], width=width,
                   color=color, edgecolor=_BENCH_COUPLING["edge"],
                   linewidth=0.7, alpha=_BENCH_COUPLING["alpha"],
                   hatch=_BENCH_COUPLING["hatch"])
            ax.errorbar(x[i] + width/2, st["seed_mean"],
                        yerr=[[st["seed_mean"] - st["seed_min"]],
                              [st["seed_max"] - st["seed_mean"]]],
                        fmt="none", ecolor="#222", capsize=3,
                        elinewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([HEADLINE_ARCH_DISPLAY.get(a, a) for a in arch_order],
                       rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Best global recovery (shared $[0,1]$)")
    ax.tick_params(axis="x", which="both", bottom=False, length=0)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linewidth=0.6, alpha=0.25)

    # Custom 2-element bench legend (bar color encodes arch separately).
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#888", edgecolor="black",
              alpha=_BENCH_DENOISING["alpha"], hatch=_BENCH_DENOISING["hatch"],
              label=_BENCH_DENOISING["label"]),
        Patch(facecolor="#888", edgecolor="black",
              alpha=_BENCH_COUPLING["alpha"], hatch=_BENCH_COUPLING["hatch"],
              label=_BENCH_COUPLING["label"]),
    ]
    # Legend below the axes — keeps the data area clear of the tall
    # Coupling bars (TXC-base hits ~0.99, TopK-SAE ~0.92).
    ax.legend(handles=handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, ncol=2,
              handlelength=2.0, columnspacing=2.0)

    fig.tight_layout()
    _save_png_pdf(fig, out_path.with_suffix(""))
    plt.close(fig)


# ── CLI ─────────────────────────────────────────────────────────────────


def main(*, setup_b_json: Path, c2_leaderboard: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if setup_b_json.exists():
        records = json.loads(setup_b_json.read_text())
        # Setup B scatter — paper-snippet name.
        render_setup_b_scatter(
            records, output_dir / "c2_setup_b_singlelatent.png"
        )
        # Backward-compat alias under the older filename.
        render_setup_b_scatter(
            records, output_dir / "c2_noisy_singlelatent_scatter.png"
        )
        print(f"[c2_paper] wrote c2_setup_b_singlelatent.{{png,pdf}} "
              f"+ alias ({len(records)} setup B records)")
    rows: list[dict] = []
    if c2_leaderboard.exists():
        for ln in c2_leaderboard.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        # "Clean" Setup D variant (no per-arch trail lines) — paper
        # snippet name.
        render_setup_d_np10_scatter(
            rows, output_dir / "c2_setup_d_scatter_clean.png",
            draw_trail=False,
        )
        # Backward-compat: with-trail under the older filename.
        render_setup_d_np10_scatter(
            rows, output_dir / "c2_setup_d_np10_scatter.png",
            draw_trail=True,
        )
        print(f"[c2_paper] wrote c2_setup_d_scatter_clean.{{png,pdf}} "
              f"+ trail alias ({len(rows)} c2 leaderboard rows scanned)")
    # Headline bar across both benches — needs both data sources.
    if records and rows:
        render_global_headline(
            records, rows, output_dir / "c2_synth_global_headline.png"
        )
        print("[c2_paper] wrote c2_synth_global_headline.{png,pdf}")


def _purified_root() -> Path:
    """Resolve the in-repo ``purified/`` root from this script's location."""
    return Path(__file__).resolve().parent.parent


def cli() -> None:
    root = _purified_root()
    ap = argparse.ArgumentParser(description=(
        "C2 (noisy-filler) paper figure renderer. "
        "Defaults to in-repo canonical paths."
    ))
    ap.add_argument(
        "--setup-b-json", type=Path,
        default=root / "experiments" / "c1_noisy_filler" / "denoising_probe_results.json",
        help="Setup-B denoising probe results (default: in-repo).",
    )
    ap.add_argument(
        "--c2-leaderboard", type=Path,
        default=root / "results" / "leaderboard.jsonl",
        help="Leaderboard jsonl (default: purified/results/leaderboard.jsonl).",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=root / "figs" / "c2",
        help="Output directory for PNG/PDF figures (default: purified/figs/c2/).",
    )
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(setup_b_json=args.setup_b_json,
         c2_leaderboard=args.c2_leaderboard,
         output_dir=args.output_dir)


if __name__ == "__main__":
    cli()
