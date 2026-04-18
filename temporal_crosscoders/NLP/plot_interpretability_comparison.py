"""Plot the summary figure for "is TFA pred or TXCDR better for
interpretability?"

Three panels per layer (L25 on top, L13 on bottom). Columns:
  A. Filter-pass bar chart — of span-weighted top-25 features per
     arch, how many pass progressively stricter quality filters.
     (content-bearing → cross-chain ≥ 3 → position-diverse ≥ 3 →
     Haiku-labeled-not-unclear). TFA pred drops to 0 at the strict
     end; TXCDR / Stacked / TFA novel stay high.
  B. Chain × position diversity scatter — each point is a
     span-weighted top-25 feature. Shows TFA pred L25 pile at
     (chain=1) and TFA pred L13 pile at (w-start=1).
  C. Label-family count — how many distinct patterns (5-word-prefix
     signatures) do the 25 labels represent? TFA pred collapses to
     ~1 family; others have 15+ distinct.

Crosscoder is included using its decoder-based high-span results
(top-15 from `high_span__*__k50.json`) because activation-based
concentration is undefined for Crosscoder's (B, d_sae) code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ARCHS_ORDER = ["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"]
ARCH_LABELS = {
    "stacked_sae": "Stacked SAE",
    "crosscoder": "Crosscoder\n(TXCDR)",
    "tfa_pos": "TFA novel",
    "tfa_pos_pred": "TFA pred",
}
ARCH_COLORS = {
    "stacked_sae": "#1f77b4",
    "crosscoder": "#2ca02c",
    "tfa_pos": "#d62728",
    "tfa_pos_pred": "#ff7f0e",
}


def _content_frac(exs: list[dict]) -> float:
    if not exs:
        return 0.0
    n_good = 0
    for ex in exs:
        m = re.search(r">>>(.*?)<<<", ex["text"], re.DOTALL)
        if m and len(m.group(1).strip()) >= 3:
            n_good += 1
    return n_good / len(exs)


def _chain_div(exs: list[dict]) -> int:
    return len({ex["chain_idx"] for ex in exs})


def _wstart_div(exs: list[dict]) -> int:
    return len({ex["window_start"] for ex in exs})


def _label_class(label: str) -> str:
    l = label.lower().strip().rstrip(".")
    if l.startswith("error"):
        return "error"
    if l == "unclear" or "unclear" in l[:30] or "weak or" in l:
        return "unclear"
    return "labeled"


def _label_family(label: str) -> str:
    """Pattern signature: first 5 content words after stripping the
    common "the feature fires on" preamble, which varies across
    labels of the same pattern."""
    l = label.lower()
    for pat in (
        r"^the\s+feature\s+(fires on|activates on|marks|represents|is about|identifies|indicates|captures)\s+",
        r"^this\s+feature\s+(fires on|activates on|marks|represents|is about|identifies|indicates|captures)\s+",
        r"^feature\s+(fires on|activates on|marks|represents)\s+",
    ):
        l = re.sub(pat, "", l)
    words = re.findall(r"\w+", l)
    return " ".join(words[:5])


def _load_feats(scan_dir: Path, arch: str, layer: str, k: int) -> dict:
    """Load scan + labels for span-weighted top-25 of arch/layer.

    For Crosscoder, span-weighted activation-based is undefined; use
    the decoder-based high-span top-15 instead (from the existing
    high_span_comparison output).
    """
    if arch == "crosscoder":
        hs = json.load(open(scan_dir / f"high_span__{layer}__k{k}.json"))
        scan = json.load(open(scan_dir / f"scan__crosscoder__{layer}__k{k}.json"))
        labels_all = json.load(
            open(scan_dir / f"labels__crosscoder__{layer}__k{k}.json")
        )["labels"]
        # Use the raw top-15 by decoder concentration
        entries = hs["by_arch"]["crosscoder"]["raw_top15_by_mass"]
        feats: dict[str, dict] = {}
        lab_map: dict[str, str] = {}
        for e in entries:
            fid = str(e["feat_idx"])
            if fid in scan["features"]:
                feats[fid] = scan["features"][fid]
            if fid in labels_all:
                lab_map[fid] = labels_all[fid]
        return {"features": feats, "labels": lab_map}
    else:
        scan = json.load(open(
            scan_dir / f"span_weighted_scan__{arch}__{layer}__k{k}.json"
        ))
        labels = json.load(open(
            scan_dir / f"span_weighted_labels__{arch}__{layer}__k{k}.json"
        ))
        return {"features": scan["features"], "labels": labels.get("labels", {})}


def filter_pass_counts(feats: dict, labels: dict) -> dict:
    """Progressive filters: count features passing each."""
    # content-bearing
    cb = [fid for fid, f in feats.items() if _content_frac(f["examples"]) >= 0.6]
    # AND cross-chain
    cc = [fid for fid in cb if _chain_div(feats[fid]["examples"]) >= 3]
    # AND w-start diverse
    wd = [fid for fid in cc if _wstart_div(feats[fid]["examples"]) >= 3]
    # AND Haiku labeled-not-unclear
    ln = [fid for fid in wd
          if _label_class(labels.get(fid, "unclear")) == "labeled"]
    return {
        "scanned": len(feats),
        "content-bearing": len(cb),
        "+cross-chain≥3": len(cc),
        "+w-start-diverse≥3": len(wd),
        "+clear Haiku label": len(ln),
    }


def label_family_count(labels: dict) -> tuple[int, int, Counter]:
    if not labels:
        return 0, 0, Counter()
    fams = Counter(_label_family(v) for v in labels.values()
                   if _label_class(v) != "error")
    max_cluster = max(fams.values()) if fams else 0
    return len(fams), max_cluster, fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default="results/nlp_sweep/gemma/scans")
    ap.add_argument("--out-dir", default="results/nlp_sweep/gemma/figures")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    data = {}  # layer -> arch -> {features, labels, filter_counts, fams}
    for layer in ["resid_L25", "resid_L13"]:
        data[layer] = {}
        for arch in ARCHS_ORDER:
            try:
                f = _load_feats(scan_dir, arch, layer, args.k)
            except FileNotFoundError as e:
                print(f"skip {arch} @ {layer}: {e}")
                continue
            fc = filter_pass_counts(f["features"], f["labels"])
            nfams, maxc, fams = label_family_count(f["labels"])
            data[layer][arch] = dict(**f, filter_counts=fc, n_families=nfams,
                                     max_cluster=maxc, families=fams)
            print(f"  {layer} {arch}: {fc}  n_fams={nfams} max_cluster={maxc}")

    # ────────────────────────── Figure ──────────────────────────
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32,
                          width_ratios=[1.3, 1.0, 0.9])

    filter_keys = ["scanned", "content-bearing", "+cross-chain≥3",
                   "+w-start-diverse≥3", "+clear Haiku label"]

    for row, layer in enumerate(["resid_L25", "resid_L13"]):
        # Panel A — filter-pass bar chart
        ax = fig.add_subplot(gs[row, 0])
        x = np.arange(len(filter_keys))
        bar_w = 0.2
        archs_here = [a for a in ARCHS_ORDER if a in data[layer]]
        for i, arch in enumerate(archs_here):
            vals = [data[layer][arch]["filter_counts"][k] for k in filter_keys]
            offset = (i - (len(archs_here) - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=ARCH_LABELS[arch],
                          color=ARCH_COLORS[arch], alpha=0.9)
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                            f"{v}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(filter_keys, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(f"# features (of top-25 by span-weighted mass)\n"
                      f"Crosscoder: top-15 by decoder conc.", fontsize=10)
        ax.set_title(f"A. Quality-filter pass count — {layer}",
                     fontsize=12)
        if row == 0:
            ax.legend(fontsize=9, loc="upper right", ncol=2)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 30)

        # Panel B — chain × position diversity scatter
        ax = fig.add_subplot(gs[row, 1])
        for arch in archs_here:
            feats = data[layer][arch]["features"]
            xs, ys = [], []
            for fid, f in feats.items():
                xs.append(_chain_div(f["examples"]))
                ys.append(_wstart_div(f["examples"]))
            # jitter for visual clarity
            rng = np.random.default_rng(hash((arch, layer)) % 2**32)
            xs = np.array(xs) + rng.normal(0, 0.15, size=len(xs))
            ys = np.array(ys) + rng.normal(0, 0.15, size=len(ys))
            ax.scatter(xs, ys, s=40, alpha=0.6, color=ARCH_COLORS[arch],
                       label=ARCH_LABELS[arch].replace("\n", " "),
                       edgecolors="black", linewidths=0.3)
        ax.set_xlabel("# unique chains in top-10 exemplars", fontsize=10)
        ax.set_ylabel("# unique window_start in top-10", fontsize=10)
        ax.set_title(f"B. Per-feature exemplar diversity — {layer}",
                     fontsize=12)
        ax.set_xlim(-0.5, 11)
        ax.set_ylim(-0.5, 11)
        ax.axvline(3, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(3, color="gray", linestyle=":", alpha=0.5)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(fontsize=8, loc="lower left")
        # annotate degeneracy regions
        ax.annotate("single-chain\n(TFA pred L25)", xy=(1, 10), xytext=(1.5, 7),
                    fontsize=8, color="darkred", alpha=0.8,
                    arrowprops=dict(arrowstyle="->", color="darkred", alpha=0.5)
                    ) if layer == "resid_L25" else None
        ax.annotate("BOS-detector\n(TFA pred L13)", xy=(10, 1), xytext=(5.5, 3.2),
                    fontsize=8, color="darkred", alpha=0.8,
                    arrowprops=dict(arrowstyle="->", color="darkred", alpha=0.5)
                    ) if layer == "resid_L13" else None

        # Panel C — size of largest label cluster (= how many features
        # Haiku assigns to the same pattern). Big = collapse.
        ax = fig.add_subplot(gs[row, 2])
        names = [ARCH_LABELS[a].replace("\n", " ") for a in archs_here]
        max_clusters = [data[layer][a]["max_cluster"] for a in archs_here]
        n_scanned = [data[layer][a]["filter_counts"]["scanned"] for a in archs_here]
        colors = [ARCH_COLORS[a] for a in archs_here]
        bars = ax.barh(names, max_clusters, color=colors, alpha=0.9,
                       edgecolor="black")
        for b, v, n in zip(bars, max_clusters, n_scanned):
            pct = v / n * 100 if n else 0
            ax.text(v + 0.3, b.get_y() + b.get_height() / 2,
                    f"{v}/{n}  ({pct:.0f}%)", va="center",
                    fontsize=10, fontweight="bold")
        ax.set_xlabel(
            "# features in largest label cluster\n"
            "(features Haiku describes with the same pattern)",
            fontsize=10,
        )
        ax.set_title(f"C. Label collapse — {layer}", fontsize=12)
        ax.set_xlim(0, max(max(max_clusters) * 1.4, 6))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "TFA pred vs TXCDR: interpretability of temporal-span features\n"
        "(Under span-weighted ranking, TFA pred collapses to architectural artifacts; "
        "TXCDR and Stacked/TFA-novel produce diverse content features)",
        fontsize=14, y=0.995,
    )

    # Save at three sizes
    base = os.path.join(args.out_dir, "interpretability_comparison_hero")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    im = Image.open(base + ".png")
    if im.width > 1400:
        ratio = 1400 / im.width
        im.resize((1400, int(im.height * ratio)), Image.LANCZOS).save(
            base + ".doc.png", optimize=True)
    else:
        im.save(base + ".doc.png", optimize=True)
    if im.width > 480:
        ratio = 480 / im.width
        im.resize((480, int(im.height * ratio)), Image.LANCZOS).save(
            base + ".thumb.png", optimize=True)
    else:
        im.save(base + ".thumb.png", optimize=True)
    plt.close(fig)
    print(f"wrote {base}.png (+.doc.png +.thumb.png)")


if __name__ == "__main__":
    main()
