#!/usr/bin/env python3
"""
plot_autointerp_summary.py — Generate summary figures for the cross-arch
autointerp analysis on Gemma-2-2B-IT resid_L25 k=50.

Five figures, saved via src.utils.plot.save_figure (high-res + thumbnail):

  1. passage_locality.png          — analysis 1 (novel vs pred is the killer)
  2. temporal_concentration.png    — analysis 2 (TFA novel spread signature)
  3. tfa_novel_vs_pred_mass.png    — analysis 1 (disjoint two-library)
  4. cross_arch_decoder_sim.png    — analysis 3 (TFA is orthogonal)
  5. semantic_categories.png       — analysis 4 (TFA pred ≈ stacked, novel unique)
  6. txcdr_vs_tfa_hero.png         — the two-arch focused comparison

Reads existing JSON outputs in results/nlp_sweep/gemma/scans/ plus the
model checkpoints to recompute the decoder-similarity distribution.
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
log = logging.getLogger("plot_summary")

SCANS = "results/nlp_sweep/gemma/scans"
CKPTS = "results/nlp_sweep/gemma/ckpts"
OUT = "results/nlp_sweep/gemma/figures"

ARCHS = ["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"]
ARCH_COLORS = {
    "stacked_sae":  "#1f77b4",  # blue
    "crosscoder":   "#2ca02c",  # green
    "tfa_pos":      "#d62728",  # red — TFA novel
    "tfa_pos_pred": "#ff7f0e",  # orange — TFA pred
}
ARCH_LABELS = {
    "stacked_sae":  "Stacked SAE",
    "crosscoder":   "Crosscoder",
    "tfa_pos":      "TFA novel",
    "tfa_pos_pred": "TFA pred",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def load_scans():
    return {a: _load(f"{SCANS}/scan__{a}__resid_L25__k50.json") for a in ARCHS}


def load_tspread():
    return {
        a: _load(f"{SCANS}/tspread__{a}__resid_L25__k50.json")
        for a in ["stacked_sae", "tfa_pos", "tfa_pos_pred"]
    }


def load_tfa_pred_novel():
    return _load(f"{SCANS}/tfa_pred_novel__resid_L25__k50.json")


def load_labels():
    return {
        a: _load(f"{SCANS}/labels__{a}__resid_L25__k50.json")["labels"]
        for a in ARCHS
    }


# ────────────────────────────────────────────────────────────────────
# 1. Passage-locality: for each arch, fraction of top-100 features
#    where all 10 exemplars come from a single chain.
# ────────────────────────────────────────────────────────────────────
def plot_passage_locality(scans):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: stacked bar of chain-diversity distribution (1..10 unique chains)
    ax = axes[0]
    width = 0.2
    x = np.arange(1, 11)
    for i, arch in enumerate(ARCHS):
        top_fis = list(scans[arch]["features"].keys())[:100]
        divs = []
        for fi in top_fis:
            chains = {ex["chain_idx"] for ex in scans[arch]["features"][fi]["examples"]}
            divs.append(len(chains))
        counts = np.bincount(divs, minlength=11)[1:11]
        ax.bar(x + (i - 1.5) * width, counts, width,
               color=ARCH_COLORS[arch], label=ARCH_LABELS[arch])
    ax.set_xlabel("# unique chains among feature's top-10 exemplars")
    ax.set_ylabel("count (of top-100 features)")
    ax.set_title("Chain diversity of top features")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: headline bar — "all 10 exemplars from same chain"
    ax = axes[1]
    same_chain = []
    for arch in ARCHS:
        top_fis = list(scans[arch]["features"].keys())[:100]
        c = sum(1 for fi in top_fis
                if len({ex["chain_idx"] for ex in scans[arch]["features"][fi]["examples"]}) == 1)
        same_chain.append(c)
    bars = ax.bar([ARCH_LABELS[a] for a in ARCHS], same_chain,
                  color=[ARCH_COLORS[a] for a in ARCHS])
    for bar, v in zip(bars, same_chain):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v}/100",
                ha="center", fontsize=10)
    ax.set_ylabel("features with all 10 exemplars from one chain")
    ax.set_title("Passage-local features (of top 100)")
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Analysis 1 — TFA novel codes are passage-local; pred codes generalize",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, f"{OUT}/passage_locality.png")
    plt.close(fig)
    log.info("  → passage_locality.png")


# ────────────────────────────────────────────────────────────────────
# 2. Temporal concentration: histogram per arch of the per-feature
#    max_position / sum_position metric across top-300 features.
# ────────────────────────────────────────────────────────────────────
def plot_temporal_concentration(tspreads):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    bins = np.linspace(0.18, 1.02, 30)
    archs = ["stacked_sae", "tfa_pos", "tfa_pos_pred"]
    for ax, arch in zip(axes, archs):
        data = tspreads[arch]
        concs = [r["concentration"] for r in data["features"].values()
                 if not np.isnan(r["concentration"])]
        ax.hist(concs, bins=bins, color=ARCH_COLORS[arch], alpha=0.85,
                edgecolor="white")
        ax.axvline(1.0 / 5, color="black", linestyle="--", linewidth=1,
                   label="1/T (uniform)")
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1,
                   label="1 (fully localized)")
        ax.set_xlabel("concentration = max_pos / sum_pos")
        ax.set_title(ARCH_LABELS[arch])
        ax.set_xlim(0.18, 1.02)
        ax.legend(fontsize=8, loc="upper center")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("count (of top-300 features)")
    fig.suptitle(
        "Analysis 2 — TFA novel is the ONLY arch with features that spread across positions",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, f"{OUT}/temporal_concentration.png")
    plt.close(fig)
    log.info("  → temporal_concentration.png")


# ────────────────────────────────────────────────────────────────────
# 3. TFA novel vs pred mass per feature — scatter.
# ────────────────────────────────────────────────────────────────────
def plot_tfa_novel_vs_pred(pn):
    feats = pn["features"]
    novel = np.array([r["novel_mass"] for r in feats.values()])
    pred  = np.array([r["pred_mass"]  for r in feats.values()])

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(novel, pred, s=18, alpha=0.6, color=ARCH_COLORS["tfa_pos"])
    ax.set_xlabel("novel-codes mass at top-10 exemplars")
    ax.set_ylabel("pred-codes mass at same exemplars")
    ax.set_title(
        "TFA-pos: novel-dominant features have ~0 pred mass (and vice-versa)\n"
        f"{len(feats)} top-by-novel-mass features"
    )
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.axhline(0, color="black", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="black", alpha=0.3, linewidth=0.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"{OUT}/tfa_novel_vs_pred_mass.png")
    plt.close(fig)
    log.info("  → tfa_novel_vs_pred_mass.png")


# ────────────────────────────────────────────────────────────────────
# 4. Cross-arch decoder similarity CDF. Recomputes per-position-0
#    best-match distribution since the summary JSON only has stats.
# ────────────────────────────────────────────────────────────────────
def _load_decoder_pos0(arch):
    model = load_model(
        ckpt_path=f"{CKPTS}/{arch.replace('_pred', '')}__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt",
        model_type=arch if not arch.endswith("_pred") else arch,
        subject_model="gemma-2-2b-it", k=50, T=5,
    )
    with torch.no_grad():
        if arch == "stacked_sae":
            D = model.saes[0].W_dec.data.T  # (d_sae, d_in)
        elif arch == "crosscoder":
            D = model.W_dec.data[:, 0, :]
        elif arch.startswith("tfa"):
            D = model._inner.D.data
        else:
            raise ValueError(arch)
    return F.normalize(D.float(), dim=-1)


def plot_cross_arch_sim():
    log.info("  loading decoders at position 0...")
    decoders = {
        "stacked_sae": _load_decoder_pos0("stacked_sae"),
        "crosscoder":  _load_decoder_pos0("crosscoder"),
        "tfa_pos":     _load_decoder_pos0("tfa_pos"),
    }

    pairs = [
        ("stacked_sae", "crosscoder"),
        ("stacked_sae", "tfa_pos"),
        ("crosscoder", "tfa_pos"),
    ]
    # Within-arch control
    xcdr_p1 = F.normalize(
        load_model(
            ckpt_path=f"{CKPTS}/crosscoder__gemma-2-2b-it__fineweb__resid_L25__k50__seed42.pt",
            model_type="crosscoder", subject_model="gemma-2-2b-it", k=50, T=5,
        ).W_dec.data[:, 1, :].float(),
        dim=-1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: CDF of best-match cosine for each A->B pair
    ax = axes[0]
    for a, b in pairs:
        sim = (decoders[a] @ decoders[b].T)
        best = sim.max(dim=1).values.cpu().numpy()
        xs = np.sort(best)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, lw=2, label=f"{ARCH_LABELS[a]} → {ARCH_LABELS[b]}")
    # Control
    sim_ctrl = decoders["crosscoder"] @ xcdr_p1.T
    best_c = sim_ctrl.max(dim=1).values.cpu().numpy()
    xs = np.sort(best_c); ys = np.arange(1, len(xs)+1)/len(xs)
    ax.plot(xs, ys, lw=2, linestyle="--", color="grey",
            label="control: Crosscoder pos0 → Crosscoder pos1")
    ax.axvline(0.5, color="black", alpha=0.3, lw=0.8)
    ax.axvline(0.7, color="black", alpha=0.3, lw=0.8)
    ax.text(0.52, 0.02, "0.5", fontsize=8)
    ax.text(0.72, 0.02, "0.7", fontsize=8)
    ax.set_xlabel("best-match cosine similarity")
    ax.set_ylabel("cumulative fraction of features")
    ax.set_xlim(-0.05, 1.0)
    ax.set_title("Cross-arch decoder matching (pos 0)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    # Right: count with sim > threshold per pair
    ax = axes[1]
    thresholds = [0.3, 0.5, 0.7, 0.9]
    x = np.arange(len(thresholds))
    width = 0.22
    bars_data = []
    for i, (a, b) in enumerate(pairs):
        sim = (decoders[a] @ decoders[b].T)
        best = sim.max(dim=1).values.cpu().numpy()
        counts = [(best > t).sum() for t in thresholds]
        ax.bar(x + (i - 1) * width, counts, width,
               label=f"{ARCH_LABELS[a]} ↔ {ARCH_LABELS[b]}")
        bars_data.append(counts)
    ax.set_xticks(x)
    ax.set_xticklabels([f"> {t}" for t in thresholds])
    ax.set_xlabel("best-match cosine threshold")
    ax.set_ylabel("# features (of 18432) above threshold")
    ax.set_yscale("symlog")
    ax.set_title("Features with similarity above threshold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Analysis 3 — Stacked & Crosscoder share a core; TFA is orthogonal",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, f"{OUT}/cross_arch_decoder_sim.png")
    plt.close(fig)
    log.info("  → cross_arch_decoder_sim.png")


# ────────────────────────────────────────────────────────────────────
# 5. Semantic category stacked bar.
# ────────────────────────────────────────────────────────────────────
CATEGORIES = {
    "document_start": ["beginning", "start of", "first token", "sequence-initial",
                       "document-initial", "opening", "introduce", "introducing",
                       "headline", "title"],
    "proper_noun_or_name": ["proper noun", "name", "organization", "author", "capitalized"],
    "date_or_time": ["date", "time stamp", "timestamp", "weekday", "month", "year", "day of"],
    "url_or_code":  ["url", "hyperlink", "code snippet", "fragment", "alphanumeric",
                     "identifier", "function call"],
    "punctuation_delim": ["punctuation", "comma", "bracket", "list marker", "delimiter",
                          "separator", "bullet"],
    "section_transition": ["transition", "topic", "section", "boundaries of", "break"],
    "domain_specific": ["botan", "biolog", "medical", "legal", "intel processor",
                        "hadoop", "windows"],
    "promotional": ["promot", "advertis", "commercial", "offer"],
    "tokenization_boundary": ["camelcase", "tokenization", "morpheme", "split across",
                              "classification code", "catalog number", "path separator",
                              "hierarchical identifier"],
    "unclear_or_noisy": ["unclear", "diverse contexts", "no coherent", "heterogeneous"],
}

def _cat(text):
    t = text.lower()
    hits = [c for c, keys in CATEGORIES.items() if any(k in t for k in keys)]
    return hits or ["other"]


def plot_semantic_categories(labels):
    import collections
    cats_order = list(CATEGORIES.keys()) + ["other"]
    per_arch = {a: collections.Counter() for a in ARCHS}
    for a, L in labels.items():
        for fi, lbl in L.items():
            if lbl.startswith("ERROR"): continue
            for c in _cat(lbl):
                per_arch[a][c] += 1

    fig, ax = plt.subplots(figsize=(11, 5.5))
    y_pos = np.arange(len(ARCHS))
    left = np.zeros(len(ARCHS))
    palette = plt.cm.tab20(np.linspace(0, 1, len(cats_order)))
    for i, c in enumerate(cats_order):
        vals = np.array([per_arch[a].get(c, 0) for a in ARCHS])
        if vals.sum() == 0: continue
        ax.barh(y_pos, vals, left=left, label=c,
                color=palette[i], edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            if v >= 3:
                ax.text(left[j] + v/2, y_pos[j], str(v),
                        ha="center", va="center", fontsize=9,
                        color="white" if i % 4 in (1, 2) else "black")
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels([ARCH_LABELS[a] for a in ARCHS])
    ax.set_xlabel("# feature tags (features can have multiple)")
    ax.set_title(
        "Analysis 4 — Semantic category distribution\n"
        "TFA novel is dominated by 'unclear' (passage-local URLs/codes); "
        "TFA pred mirrors Stacked."
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, f"{OUT}/semantic_categories.png")
    plt.close(fig)
    log.info("  → semantic_categories.png")


# ────────────────────────────────────────────────────────────────────
# 6. HERO: TXCDR vs TFA focused comparison.
# ────────────────────────────────────────────────────────────────────
def plot_txcdr_vs_tfa_hero(scans, tspreads, labels):
    import collections
    archs3 = ["crosscoder", "tfa_pos", "tfa_pos_pred"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (0,0) — Chain diversity of top-100 features
    ax = axes[0, 0]
    x = np.arange(1, 11)
    width = 0.28
    for i, arch in enumerate(archs3):
        top_fis = list(scans[arch]["features"].keys())[:100]
        divs = [len({e["chain_idx"] for e in scans[arch]["features"][fi]["examples"]})
                for fi in top_fis]
        counts = np.bincount(divs, minlength=11)[1:11]
        ax.bar(x + (i - 1) * width, counts, width,
               color=ARCH_COLORS[arch], label=ARCH_LABELS[arch])
    ax.set_xlabel("# unique chains in top-10 exemplars")
    ax.set_ylabel("count (of top-100 features)")
    ax.set_title("A. Chain diversity — TFA novel is passage-local, pred is not")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # (0,1) — Temporal concentration (TFA only has per-pos; crosscoder has no per-pos)
    ax = axes[0, 1]
    bins = np.linspace(0.18, 1.02, 25)
    for arch in ["tfa_pos", "tfa_pos_pred"]:
        concs = [r["concentration"] for r in tspreads[arch]["features"].values()
                 if not np.isnan(r["concentration"])]
        ax.hist(concs, bins=bins, alpha=0.6, color=ARCH_COLORS[arch],
                label=ARCH_LABELS[arch], edgecolor="white")
    ax.axvline(0.2, color="black", linestyle="--", lw=1, label="1/T=0.2 (spread)")
    ax.axvline(1.0, color="black", linestyle=":", lw=1, label="1.0 (localized)")
    ax.set_xlabel("temporal concentration per feature")
    ax.set_ylabel("count")
    ax.set_title("B. Temporal spread — TFA novel uniquely spreads across positions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.95,
            "Crosscoder omitted:\nhas single z per window\n(no per-position codes)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    # (1,0) — Semantic category comparison, same-archs subset
    ax = axes[1, 0]
    cats_order = list(CATEGORIES.keys()) + ["other"]
    per_arch = {a: collections.Counter() for a in archs3}
    for a in archs3:
        for fi, lbl in labels[a].items():
            if lbl.startswith("ERROR"): continue
            for c in _cat(lbl):
                per_arch[a][c] += 1
    y_pos = np.arange(len(archs3))
    left = np.zeros(len(archs3))
    palette = plt.cm.tab20(np.linspace(0, 1, len(cats_order)))
    for i, c in enumerate(cats_order):
        vals = np.array([per_arch[a].get(c, 0) for a in archs3])
        if vals.sum() == 0: continue
        ax.barh(y_pos, vals, left=left, label=c, color=palette[i],
                edgecolor="white", linewidth=0.5)
        for j, v in enumerate(vals):
            if v >= 4:
                ax.text(left[j] + v/2, y_pos[j], str(v),
                        ha="center", va="center", fontsize=8,
                        color="white" if i % 4 in (1, 2) else "black")
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels([ARCH_LABELS[a] for a in archs3])
    ax.set_xlabel("# feature tags (features can have multiple)")
    ax.set_title("C. Semantic categories — TFA novel dominated by 'unclear' + 'tokenization_boundary'")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="x", alpha=0.3)

    # (1,1) — active-feature counts + label success
    ax = axes[1, 1]
    metrics = [
        ("active feats / 18432", [scans[a]["num_active_features"] for a in archs3]),
        ("'unclear' labels / 50", [
            sum(1 for v in labels[a].values() if "unclear" in v.lower() and not v.startswith("ERROR"))
            for a in archs3
        ]),
    ]
    x = np.arange(len(archs3))
    width = 0.35
    active = np.array(metrics[0][1])
    unclear = np.array(metrics[1][1])
    # Two y-axes so active (thousands) and unclear (50-max) don't wash each other out
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, active, width,
                   color=[ARCH_COLORS[a] for a in archs3], alpha=0.85,
                   label="active features", edgecolor="white")
    bars2 = ax2.bar(x + width/2, unclear, width,
                    color=[ARCH_COLORS[a] for a in archs3], alpha=0.4,
                    label="'unclear' labels (of 50)", edgecolor="black",
                    linewidth=1.2)
    for bar, v in zip(bars1, active):
        ax.text(bar.get_x() + bar.get_width()/2, v + 200,
                f"{v:,}", ha="center", fontsize=9)
    for bar, v in zip(bars2, unclear):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.7,
                 str(v), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in archs3])
    ax.set_ylabel("active features / 18432 (solid)")
    ax2.set_ylabel("'unclear' labels of top 50 (hatched)")
    ax.set_title("D. Capacity used vs label clarity")
    ax.set_ylim(0, 20000)
    ax2.set_ylim(0, 50)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "TXCDR (Crosscoder) vs TFA — focused comparison",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, f"{OUT}/txcdr_vs_tfa_hero.png")
    plt.close(fig)
    log.info("  → txcdr_vs_tfa_hero.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-decoder-sim", action="store_true",
                        help="Skip plot 4 which needs the GPU + model loads")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)

    scans = load_scans()
    tspreads = load_tspread()
    pn = load_tfa_pred_novel()
    labels = load_labels()

    log.info("plotting...")
    plot_passage_locality(scans)
    plot_temporal_concentration(tspreads)
    plot_tfa_novel_vs_pred(pn)
    if not args.skip_decoder_sim:
        plot_cross_arch_sim()
    plot_semantic_categories(labels)
    plot_txcdr_vs_tfa_hero(scans, tspreads, labels)
    log.info(f"all figures under {OUT}/")


if __name__ == "__main__":
    main()
