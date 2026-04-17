"""Phase 3 extension #5: content-based cross-arch feature matching.

For each pair of archs (A, B), compute per-feature set of
(chain_idx, window_start) tuples from the top-K exemplars, then match
each A-feature to its best-Jaccard B-feature. This is the alternative
to decoder-cosine similarity; it's architecture-agnostic.

Outputs:
  - JSON: content_match__<layer>__k<k>.json with per-pair match distributions
  - Figure: content_based_match.{png,doc.png,thumb.png}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("content_match")

ARCHS = ["stacked_sae", "crosscoder", "tfa_pos", "tfa_pos_pred"]
ARCH_LABELS = {
    "stacked_sae": "Stacked SAE",
    "crosscoder": "Crosscoder",
    "tfa_pos": "TFA novel",
    "tfa_pos_pred": "TFA pred",
}
ARCH_COLORS = {
    "stacked_sae": "#1f77b4",
    "crosscoder": "#2ca02c",
    "tfa_pos": "#d62728",
    "tfa_pos_pred": "#ff7f0e",
}


def exemplar_set(feat: dict, top_k: int) -> frozenset:
    """Set of (chain_idx, window_start) tuples from the top-K exemplars.
    (window_start is the position in the chain; combined with chain_idx
    it uniquely identifies the exact window the feature fired on.)"""
    ex = feat["examples"][:top_k]
    return frozenset((e["chain_idx"], e["window_start"]) for e in ex)


def best_jaccard(src_sets: dict[int, frozenset],
                 tgt_sets: dict[int, frozenset]) -> dict[int, tuple[int, float]]:
    """For each src feature, find the tgt feature with max Jaccard overlap.

    Optimization: index tgt features by the (chain, start) tuples they
    contain, so only tgt features that share at least one exemplar are
    scored — sparse matrices would do the same job.
    """
    # invert: (chain, start) -> list of tgt feat ids that include it
    inv = defaultdict(list)
    for tid, tset in tgt_sets.items():
        for key in tset:
            inv[key].append(tid)

    result = {}
    for sid, sset in src_sets.items():
        if not sset:
            result[sid] = (-1, 0.0)
            continue
        # candidates: any tgt feature sharing at least one exemplar
        cand_ids = set()
        for key in sset:
            cand_ids.update(inv.get(key, ()))
        if not cand_ids:
            result[sid] = (-1, 0.0)
            continue
        best_tid = -1
        best_j = 0.0
        for tid in cand_ids:
            tset = tgt_sets[tid]
            inter = len(sset & tset)
            union = len(sset | tset)
            j = inter / union if union else 0.0
            if j > best_j:
                best_j = j
                best_tid = tid
        result[sid] = (best_tid, best_j)
    return result


def summarize(pair_label: str, matches: dict[int, tuple[int, float]]) -> dict:
    jaccs = np.array([j for (_, j) in matches.values()])
    return {
        "pair": pair_label,
        "n_src": len(matches),
        "n_with_any_overlap": int((jaccs > 0).sum()),
        "n_jaccard_ge_0.1": int((jaccs >= 0.1).sum()),
        "n_jaccard_ge_0.3": int((jaccs >= 0.3).sum()),
        "n_jaccard_ge_0.5": int((jaccs >= 0.5).sum()),
        "mean_max_jaccard": float(jaccs.mean()),
        "median_max_jaccard": float(np.median(jaccs)),
        "p90_max_jaccard": float(np.quantile(jaccs, 0.9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default="results/nlp_sweep/gemma/scans")
    ap.add_argument("--out-dir", default="results/nlp_sweep/gemma/figures")
    ap.add_argument("--layer-key", default="resid_L25")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--top-k-exemplars", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    scans = {}
    for arch in ARCHS:
        path = os.path.join(
            args.scan_dir,
            f"scan__{arch}__{args.layer_key}__k{args.k}.json",
        )
        if not os.path.exists(path):
            log.warning(f"missing {path}")
            continue
        scans[arch] = json.load(open(path))

    exemplar_sets = {
        arch: {
            int(fid): exemplar_set(feat, args.top_k_exemplars)
            for fid, feat in scans[arch]["features"].items()
        }
        for arch in scans
    }
    for arch, s in exemplar_sets.items():
        nonempty = sum(1 for v in s.values() if v)
        log.info(f"  {arch}: {len(s)} feats, {nonempty} with at least 1 exemplar")

    # all directed pairs
    summaries = []
    matches_full = {}
    for a, b in [(x, y) for x in ARCHS if x in scans for y in ARCHS if y in scans if x != y]:
        label = f"{a}->{b}"
        m = best_jaccard(exemplar_sets[a], exemplar_sets[b])
        matches_full[label] = m
        summaries.append(summarize(label, m))

    # save JSON
    out_json = os.path.join(
        args.scan_dir,
        f"content_match__{args.layer_key}__k{args.k}.json",
    )
    json.dump(
        {
            "layer_key": args.layer_key,
            "k": args.k,
            "top_k_exemplars": args.top_k_exemplars,
            "n_scan_features_per_arch": {a: len(exemplar_sets[a]) for a in scans},
            "summaries": summaries,
            "matches": {
                label: {str(s): {"best_tgt": int(t), "jaccard": float(j)}
                        for s, (t, j) in m.items()}
                for label, m in matches_full.items()
            },
        },
        open(out_json, "w"),
        indent=2,
    )
    log.info(f"wrote {out_json}")

    # print table
    print()
    print(
        f"{'pair':<30} {'n':>4} {'mean':>6} {'med':>5} {'p90':>5}"
        f" {'≥0.1':>6} {'≥0.3':>6} {'≥0.5':>6}"
    )
    print("-" * 80)
    for s in summaries:
        print(
            f"{s['pair']:<30} {s['n_src']:>4d} {s['mean_max_jaccard']:>6.3f}"
            f" {s['median_max_jaccard']:>5.2f} {s['p90_max_jaccard']:>5.2f}"
            f" {s['n_jaccard_ge_0.1']:>6d} {s['n_jaccard_ge_0.3']:>6d}"
            f" {s['n_jaccard_ge_0.5']:>6d}"
        )

    # Figure: max-Jaccard distribution for each src arch against "best of
    # remaining archs" (i.e. for each src feat, pool candidates across all
    # other archs combined). This says "does the src arch have features
    # that no other arch captures?" — the reviewer's question in concrete
    # form.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: CDF of max Jaccard for each src arch against one specific
    # other arch (crosscoder vs tfa_pos is the headline pair)
    ax = axes[0]
    pairs_to_plot = [
        ("crosscoder", "tfa_pos"),
        ("tfa_pos", "crosscoder"),
        ("crosscoder", "stacked_sae"),
        ("stacked_sae", "crosscoder"),
        ("tfa_pos", "tfa_pos_pred"),
    ]
    for src, tgt in pairs_to_plot:
        label = f"{src}->{tgt}"
        if label not in matches_full:
            continue
        jaccs = np.sort([j for (_, j) in matches_full[label].values()])
        xs = np.concatenate([[0], jaccs])
        ys = np.linspace(0, 1, len(xs))
        ax.plot(
            xs, ys,
            label=f"{ARCH_LABELS[src]} → {ARCH_LABELS[tgt]}",
            color=ARCH_COLORS[src],
            linestyle="-" if tgt in ("crosscoder", "tfa_pos_pred") else "--",
            alpha=0.9,
        )
    ax.set_xlabel("Jaccard over top-10 (chain, start) exemplars")
    ax.set_ylabel("Cumulative fraction of src features")
    ax.set_title("Content-based feature matching: CDF of best Jaccard")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(0.51, 0.02, "J≥0.5", color="gray", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: bar chart of mean max Jaccard for each (src, tgt) pair
    ax = axes[1]
    bars_src = list(scans)
    bars_tgt = [t for t in scans]
    bar_w = 0.18
    xs = np.arange(len(bars_src))
    for i, tgt in enumerate(bars_tgt):
        vals = []
        for src in bars_src:
            label = f"{src}->{tgt}"
            if src == tgt or label not in matches_full:
                vals.append(0)
                continue
            jaccs = [j for (_, j) in matches_full[label].values()]
            vals.append(float(np.mean(jaccs)))
        ax.bar(xs + (i - 1.5) * bar_w, vals, bar_w,
               label=ARCH_LABELS[tgt], color=ARCH_COLORS[tgt])
    ax.set_xticks(xs)
    ax.set_xticklabels([ARCH_LABELS[a] for a in bars_src], rotation=15, ha="right")
    ax.set_xlabel("Source arch")
    ax.set_ylabel("Mean max Jaccard vs target")
    ax.set_title("Cross-arch exemplar-set overlap (mean best Jaccard)")
    ax.axhline(8 / 124000, color="gray", linestyle=":", alpha=0.5,
               label="~chance (rand 10 of 124k)")
    ax.legend(title="Target arch", loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Content-based (exemplar-set) feature matching — {args.layer_key} k={args.k}",
        fontsize=13,
    )
    fig.tight_layout()

    # save as three sizes (matches existing convention)
    base = os.path.join(args.out_dir, "content_based_match")
    from PIL import Image
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
    log.info(f"wrote {base}.png (+.doc.png +.thumb.png)")


if __name__ == "__main__":
    main()
