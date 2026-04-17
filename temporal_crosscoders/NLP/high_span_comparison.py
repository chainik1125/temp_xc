#!/usr/bin/env python3
"""
high_span_comparison.py — Direct answer to: do TXCDR and TFA extract
*substantially different* temporal features, specifically for features
that span multiple positions?

Metric: per-feature temporal concentration in [1/T, 1] where 1/T = uniform
spread, 1 = localized at one position.

  - Crosscoder: decoder-based. For feature f, energy at position t is
    ||W_dec[f, t, :]||. Concentration = max_t e_t / sum_t e_t. This is
    model-inherent — independent of data.
  - TFA novel / TFA pred: activation-based. Per feature, averaged across
    its top-10 exemplar windows: max_pos / sum_pos of the per-position
    novel (or pred) activations. Data-dependent.
  - Stacked SAE: trivially 1.0 by construction (independent per-position
    SAEs); not plotted.

"High span" threshold: concentration < 0.35 (i.e., the feature spreads
its contribution over more than 1/0.35 ≈ 3 positions on average).

For each arch with high-span features:
  - Count
  - Top-N high-span features by activation mass (for TXCDR, by in-window
    decoder energy norm; for TFA, by scan mass)
  - Their top-5 text exemplars
  - Their LLM labels (from labels__*.json if present)

Output:
  - JSON: results/nlp_sweep/gemma/scans/high_span__resid_L25__k50.json
  - Figure: results/nlp_sweep/gemma/figures/high_span_comparison.png
    (full-res + doc-size + thumb)

The figure has 3 rows × 3 cols:
  Row 1: concentration histograms for crosscoder / TFA novel / TFA pred
  Row 2: top-5 high-span feature exemplar texts (selected) side-by-side
  Row 3: fraction of features that are high-span + median concentration
         summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.plot import save_figure
from temporal_crosscoders.NLP.autointerp import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("high_span")

SCANS = "results/nlp_sweep/gemma/scans"
CKPTS = "results/nlp_sweep/gemma/ckpts"
FIGS  = "results/nlp_sweep/gemma/figures"
HIGH_SPAN_THRESHOLD = 0.35

ARCH_COLORS = {
    "crosscoder":   "#2ca02c",
    "tfa_pos":      "#d62728",
    "tfa_pos_pred": "#ff7f0e",
}
ARCH_LABELS = {
    "crosscoder":   "Crosscoder",
    "tfa_pos":      "TFA novel",
    "tfa_pos_pred": "TFA pred",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────────────
# 1. Crosscoder decoder-based concentration (new — fills the analysis gap)
# ────────────────────────────────────────────────────────────────────
def crosscoder_temporal_concentration(layer_key: str = "resid_L25", k: int = 50) -> dict[int, dict]:
    """For each crosscoder feature, compute decoder-norm concentration
    across T=5 positions. Model-inherent; no data needed."""
    ckpt = f"{CKPTS}/crosscoder__gemma-2-2b-it__fineweb__{layer_key}__k{k}__seed42.pt"
    model = load_model(ckpt_path=ckpt, model_type="crosscoder",
                       subject_model="gemma-2-2b-it", k=k, T=5).cpu()
    with torch.no_grad():
        # W_dec: (d_sae, T, d_in)
        W = model.W_dec.data
        e = W.norm(dim=-1)  # (d_sae, T)
        total = e.sum(dim=-1).clamp(min=1e-12)
        conc = e.max(dim=-1).values / total
        peak_pos = e.argmax(dim=-1)
    out = {}
    for fi in range(W.shape[0]):
        out[int(fi)] = {
            "concentration": float(conc[fi].item()),
            "peak_position": int(peak_pos[fi].item()),
            "per_position_energy": e[fi].tolist(),
        }
    return out


# ────────────────────────────────────────────────────────────────────
# 2. TFA concentrations loaded from existing tspread JSONs
# ────────────────────────────────────────────────────────────────────
def tfa_concentrations(arch: str, layer_key: str = "resid_L25", k: int = 50) -> dict[int, dict]:
    d = _load(f"{SCANS}/tspread__{arch}__{layer_key}__k{k}.json")
    return {int(fi): rec for fi, rec in d["features"].items()}


# ────────────────────────────────────────────────────────────────────
# 3. Pick high-span features per arch; bring in their top exemplar texts.
# ────────────────────────────────────────────────────────────────────
def _is_empty_window(text: str) -> bool:
    """True if the >>>…<<< payload has no meaningful visible content.
    TFA novel's 'high-span' activations often peak on padding/end-of-seq
    tokens that decode to empty strings; those aren't real features."""
    try:
        s = text.split(">>>", 1)[1].split("<<<", 1)[0].strip()
        return len(s) < 2
    except Exception:
        return True


def _content_fraction(examples: list[dict]) -> float:
    if not examples:
        return 0.0
    non_empty = sum(1 for ex in examples if not _is_empty_window(ex["text"]))
    return non_empty / len(examples)


def pick_high_span(
    concs: dict[int, dict],
    scan: dict,
    top_n: int,
    min_content_frac: float = 0.6,
) -> tuple[list[dict], list[dict]]:
    """Return (raw_candidates, content_bearing_candidates). Raw pick is
    by scan mass among features with concentration < threshold. The
    content-filtered subset additionally requires ≥min_content_frac of
    top-5 exemplars to have non-empty >>>…<<< payload (filters TFA's
    padding-fire artifacts)."""
    feat_mass = {int(fi): rec["mass"] for fi, rec in scan["features"].items()}
    candidates = []
    for fi in feat_mass:
        if fi not in concs or concs[fi]["concentration"] >= HIGH_SPAN_THRESHOLD:
            continue
        rec = scan["features"][str(fi)]
        top5 = rec["examples"][:5]
        candidates.append({
            "feat_idx": fi,
            "concentration": concs[fi]["concentration"],
            "mass": feat_mass[fi],
            "content_fraction": _content_fraction(top5),
            "top_texts": [ex["text"] for ex in top5],
            "top_activations": [ex["activation"] for ex in top5],
        })
    candidates.sort(key=lambda r: r["mass"], reverse=True)
    raw = candidates[:top_n]
    # Key metric A: of the raw top-N (highest mass), how many pass the
    # content filter? Shows the "do headline high-span features detect
    # real content?" story — this is where TFA's padding-fire artifact
    # becomes visible.
    content_within_raw = [r for r in raw
                          if r["content_fraction"] >= min_content_frac]
    # Key metric B: top-N content-bearing features regardless of rank.
    # We use this for the exemplar-text panel so the comparison table is
    # populated even when content_within_raw is nearly empty.
    top_content_bearing = [r for r in candidates
                           if r["content_fraction"] >= min_content_frac][:top_n]
    return raw, content_within_raw, top_content_bearing


# ────────────────────────────────────────────────────────────────────
# Plotting helper that ALSO saves a doc-sized PNG (~1200px wide)
# ────────────────────────────────────────────────────────────────────
def save_three_sizes(fig, path_base: str):
    """Save .png (full-res, 200 DPI), .doc.png (1200px doc-embed), and
    .thumb.png (288px agent) variants."""
    from PIL import Image
    full_path = path_base + ".png"
    fig.savefig(full_path, dpi=200, bbox_inches="tight")
    # Doc-embed at 1200px
    im = Image.open(full_path)
    if im.width > 1400:
        ratio = 1400 / im.width
        im_doc = im.resize((1400, int(im.height * ratio)), Image.LANCZOS)
    else:
        im_doc = im
    im_doc.save(path_base + ".doc.png", optimize=True)
    # Thumb at 480px (larger than default 288 so it's still useful at-a-glance)
    if im.width > 480:
        ratio = 480 / im.width
        im_thumb = im.resize((480, int(im.height * ratio)), Image.LANCZOS)
    else:
        im_thumb = im
    im_thumb.save(path_base + ".thumb.png", optimize=True)
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Main figure: concentration distributions + high-span subset counts +
# selected exemplar excerpts
# ────────────────────────────────────────────────────────────────────
def plot_high_span_figure(concentrations: dict, high_span: dict, fig_suffix: str = ""):
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.28,
                          height_ratios=[1.0, 1.2, 1.0])

    archs = ["crosscoder", "tfa_pos", "tfa_pos_pred"]

    # Row 1: concentration histograms
    for col, arch in enumerate(archs):
        ax = fig.add_subplot(gs[0, col])
        concs = [rec["concentration"] for rec in concentrations[arch].values()
                 if not np.isnan(rec.get("concentration", float("nan")))]
        bins = np.linspace(0.18, 1.02, 30)
        ax.hist(concs, bins=bins, color=ARCH_COLORS[arch], alpha=0.85,
                edgecolor="white", label=f"all {len(concs):,} feats")
        ax.axvline(HIGH_SPAN_THRESHOLD, color="black", linestyle="--", lw=1.5,
                   label=f"high-span threshold = {HIGH_SPAN_THRESHOLD}")
        ax.axvline(0.2, color="grey", linestyle=":", lw=1,
                   label="1/T = 0.2 (uniform)")
        ax.set_xlabel("temporal concentration  (low = spread across T=5)", fontsize=11)
        ax.set_ylabel("feature count", fontsize=11)
        metric = "decoder-based" if arch == "crosscoder" else "activation-based"
        ax.set_title(f"{ARCH_LABELS[arch]}\n({metric} concentration)",
                     fontsize=12)
        ax.legend(fontsize=9, loc="upper center")
        ax.grid(axis="y", alpha=0.3)

    # Row 2: top-3 CONTENT-BEARING high-span feature exemplar texts per arch.
    # "Content-bearing" = ≥60% of top-5 exemplars have non-empty >>>…<<<
    # payload. Filters TFA novel's padding/empty-window artifacts so that
    # what the reader sees is a fair cross-arch comparison of the kinds
    # of temporal content each arch actually detects.
    for col, arch in enumerate(archs):
        ax = fig.add_subplot(gs[1, col])
        ax.axis("off")
        # high_span[arch] is (raw, content_within_raw, top_content) tuple
        raw_list, content_within_raw_list, top_content_list = high_span[arch]
        records = top_content_list[:3]
        n_raw = len(raw_list)
        n_content = len(content_within_raw_list)
        y = 0.99
        ax.text(0.02, y, f"Top-3 *content-bearing* high-span "
                f"{ARCH_LABELS[arch]} features",
                fontsize=12, fontweight="bold", transform=ax.transAxes,
                va="top")
        y -= 0.055
        ax.text(0.02, y,
                f"({n_content}/{n_raw} raw high-span features had ≥60% "
                f"non-empty exemplars)",
                fontsize=9, transform=ax.transAxes, va="top", style="italic")
        y -= 0.08
        if not records:
            ax.text(0.02, y,
                    "(no content-bearing high-span features —\n"
                    "this arch's 'high-span' behavior is an\n"
                    "artifact of padding / empty-window fires)",
                    fontsize=11, transform=ax.transAxes, va="top",
                    color="#aa3333")
            continue
        for r in records:
            ax.text(0.02, y,
                    f"feat {r['feat_idx']}  |  conc={r['concentration']:.2f}  "
                    f"|  mass={r['mass']:.2e}",
                    fontsize=10, fontweight="bold", family="monospace",
                    transform=ax.transAxes, va="top")
            y -= 0.06
            for txt in r["top_texts"][:3]:
                t = txt.replace("\n", " ").strip()
                if len(t) > 120:
                    t = t[:117] + "..."
                ax.text(0.04, y, "• " + t, fontsize=8, family="monospace",
                        transform=ax.transAxes, va="top")
                y -= 0.055
            y -= 0.02

    # Row 3: two-bar comparison per arch — raw high-span (by concentration
    # only) vs content-bearing high-span (after filtering padding-fire
    # artifacts). The gap between the two is the TFA artifact.
    ax = fig.add_subplot(gs[2, :])
    x = np.arange(len(archs))
    width = 0.35
    raw_counts = []
    content_counts = []
    for arch in archs:
        raw_list, content_within_raw, _ = high_span[arch]
        raw_counts.append(len(raw_list))
        content_counts.append(len(content_within_raw))
    bars_raw = ax.bar(x - width/2, raw_counts, width,
                      color=[ARCH_COLORS[a] for a in archs],
                      alpha=0.55, edgecolor="black", linewidth=1.2,
                      label="raw high-span (by concentration only)")
    bars_ct = ax.bar(x + width/2, content_counts, width,
                     color=[ARCH_COLORS[a] for a in archs],
                     edgecolor="black", linewidth=1.5,
                     hatch="//",
                     label=("content-bearing "
                            "(≥60% non-empty >>>…<<< payloads)"))
    for bar, v in zip(bars_raw, raw_counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v}", ha="center", fontsize=11, fontweight="bold")
    for bar, v in zip(bars_ct, content_counts):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                f"{v}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in archs], fontsize=13)
    ax.set_ylabel("# high-span features among top-15 by mass",
                  fontsize=12)
    ax.set_title(
        "Among the top-15 high-span features: how many detect real content?\n"
        "(the gap = padding / empty-window artifacts)",
        fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, max(raw_counts) * 1.25 + 1)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "High-span feature comparison — TXCDR vs TFA\n"
        "(features whose reconstruction contribution spans multiple positions of the T=5 window)",
        fontsize=14, y=0.995,
    )

    os.makedirs(FIGS, exist_ok=True)
    base = f"{FIGS}/high_span_comparison{fig_suffix}"
    save_three_sizes(fig, base)
    log.info(f"wrote {base}.png (+.doc.png, +.thumb.png)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-key", default="resid_L25")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--fig-suffix", default="",
                        help="appended to figure basename (e.g. '_L13')")
    args = parser.parse_args()
    layer_key = args.layer_key
    k = args.k

    log.info(f"computing Crosscoder decoder-based concentration ({layer_key} k={k})")
    xcdr_concs = crosscoder_temporal_concentration(layer_key, k)

    log.info("loading TFA novel + pred concentrations from tspread JSONs")
    tfa_novel_concs = tfa_concentrations("tfa_pos", layer_key, k)
    tfa_pred_concs = tfa_concentrations("tfa_pos_pred", layer_key, k)

    concentrations = {
        "crosscoder":   xcdr_concs,
        "tfa_pos":      tfa_novel_concs,
        "tfa_pos_pred": tfa_pred_concs,
    }

    # Load scans (top-300 features w/ texts + masses)
    scans = {
        a: _load(f"{SCANS}/scan__{a}__{layer_key}__k{k}.json")
        for a in ["crosscoder", "tfa_pos", "tfa_pos_pred"]
    }
    # For crosscoder, restrict to features that actually appeared in the
    # scan (crosscoder's decoder-conc is defined for all 18432 features,
    # but we only have texts for features that ranked in top-300).
    high_span = {
        a: pick_high_span(concentrations[a], scans[a], top_n=15)
        for a in ["crosscoder", "tfa_pos", "tfa_pos_pred"]
    }

    # Persist the high-span subset (raw + content-bearing) for each arch
    os.makedirs(SCANS, exist_ok=True)
    out_path = f"{SCANS}/high_span__{layer_key}__k{k}.json"
    with open(out_path, "w") as f:
        json.dump({
            "threshold": HIGH_SPAN_THRESHOLD,
            "min_content_fraction": 0.6,
            "by_arch": {
                a: {
                    "raw_top15_by_mass":             high_span[a][0],
                    "content_within_raw_top15":      high_span[a][1],
                    "top15_content_bearing_anyrank": high_span[a][2],
                }
                for a in high_span
            },
            "counts": {
                a: {
                    "total_features_with_conc": len(concentrations[a]),
                    "high_span_count": sum(
                        1 for r in concentrations[a].values()
                        if r.get("concentration", 1) < HIGH_SPAN_THRESHOLD
                    ),
                    "raw_top15": len(high_span[a][0]),
                    "content_within_raw_top15": len(high_span[a][1]),
                }
                for a in concentrations
            },
        }, f, indent=2)
    log.info(f"wrote {out_path}")

    # Summary
    for arch in concentrations:
        concs = [rec["concentration"] for rec in concentrations[arch].values()
                 if not np.isnan(rec.get("concentration", float("nan")))]
        hs = sum(1 for c in concs if c < HIGH_SPAN_THRESHOLD)
        log.info(f"  {ARCH_LABELS[arch]:<14s}  "
                 f"total={len(concs):,}  high-span={hs:,} "
                 f"({100*hs/len(concs):.1f}%)  "
                 f"median conc={np.median(concs):.3f}")

    plot_high_span_figure(concentrations, high_span, fig_suffix=args.fig_suffix)


if __name__ == "__main__":
    main()
