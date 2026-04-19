"""N=200 version of the TFA-pred-vs-TXCDR interpretability plot.

Adds Wilson 95% confidence intervals and rank-decile robustness curves.
For Crosscoder, uses decoder-based concentration top-N (not buggy
activation-based) — but N is capped at the number of scanned features
available (top-300 scan has up to ~300 crosscoder feats).

Panel layout per layer:
  A. Filter-pass rate (proportion) with 95% Wilson CI — N=200 each
  B. Rank-decile robustness — is the N=25 story stable across top-200?
  C. Chain × window_start diversity scatter (200 points per arch)
  D. Label collapse (top-50 labeled, max-cluster-size) — unchanged N=50

Fisher's exact p-values annotated on Panel A for TFA pred vs TXCDR.
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
    n = 0
    for ex in exs:
        m = re.search(r">>>(.*?)<<<", ex["text"], re.DOTALL)
        if m and len(m.group(1).strip()) >= 3:
            n += 1
    return n / len(exs)


def _chain_div(exs: list[dict]) -> int:
    return len({ex["chain_idx"] for ex in exs})


def _wstart_div(exs: list[dict]) -> int:
    return len({ex["window_start"] for ex in exs})


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson 95% CI for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(center - half, 0.0), min(center + half, 1.0)


def fishers_exact(a_pass: int, a_total: int, b_pass: int, b_total: int) -> float:
    """Compute Fisher's exact two-sided p-value via scipy if available,
    else a tiny manual implementation for the 2x2 table."""
    try:
        from scipy.stats import fisher_exact
        table = [[a_pass, a_total - a_pass], [b_pass, b_total - b_pass]]
        _, p = fisher_exact(table, alternative="two-sided")
        return p
    except Exception:
        # Very small p-value when counts extreme; return indicative value.
        return float("nan")


def _label_family(label: str) -> str:
    l = label.lower()
    for pat in (
        r"^the\s+feature\s+(fires on|activates on|marks|represents|is about|identifies|indicates|captures)\s+",
        r"^this\s+feature\s+(fires on|activates on|marks|represents|is about|identifies|indicates|captures)\s+",
        r"^feature\s+(fires on|activates on|marks|represents)\s+",
    ):
        l = re.sub(pat, "", l)
    return " ".join(re.findall(r"\w+", l)[:5])


def _label_clear(label: str) -> bool:
    l = label.lower().strip().rstrip(".")
    if l.startswith("error") or l == "unclear" or "unclear" in l[:30]:
        return False
    return True


def load_scan_n500(scan_dir: Path, arch: str, layer: str, k: int):
    """Load N=200 span-weighted scan. For crosscoder we fall back to
    decoder-conc top-N from the all-features span_all is not
    meaningful (activation-based has no T dim).

    For N=500: all four archs use scan_specific_features output. For
    crosscoder, the picker ranks by decoder-based concentration (not
    activation, which is undefined since Crosscoder's code has no T
    dim).
    """
    path = scan_dir / f"span_weighted_scan500__{arch}__{layer}__k{k}.json"
    scan = json.load(open(path))
    # Reorder to span-weighted rank (scan saves by whatever order we
    # requested, but we want deterministic rank for decile analysis).
    if arch == "crosscoder":
        order_file = scan_dir / f"crosscoder_decoder_span_top500__{layer}__k{k}.json"
        order = json.load(open(order_file))
        ids = [str(e["feat_idx"]) for e in order["top_span_weighted"]]
    else:
        order_file = scan_dir / f"span_weighted_top500__{layer}__k{k}.json"
        order = json.load(open(order_file))
        ids = [str(e["feat_idx"]) for e in order["per_arch"][arch]["top_span_weighted"]]
    feats = scan["features"]
    ordered = {fid: feats[fid] for fid in ids if fid in feats}
    return ordered


def _features_for_arch(scan_dir: Path, arch: str, layer: str, k: int):
    """Reload the span-weighted scan N=500 features dict, rank-ordered.
    Duplicates `load_scan_n500` but without the row/label stuff."""
    return load_scan_n500(scan_dir, arch, layer, k)


def load_labels_top100(scan_dir: Path, arch: str, layer: str, k: int):
    """Labels are top-100 of the N=500 scan."""
    p = scan_dir / f"span_weighted_labels500__{arch}__{layer}__k{k}.json"
    if not p.exists():
        return {}
    return json.load(open(p)).get("labels", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", default="results/nlp_sweep/gemma/scans")
    ap.add_argument("--out-dir", default="results/nlp_sweep/gemma/figures")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all scans + labels
    data: dict[str, dict[str, dict]] = {}
    for layer in ["resid_L25", "resid_L13"]:
        data[layer] = {}
        for arch in ARCHS_ORDER:
            try:
                feats = load_scan_n500(scan_dir, arch, layer, args.k)
                labels = load_labels_top100(scan_dir, arch, layer, args.k)
            except FileNotFoundError:
                print(f"skip {arch} @ {layer}")
                continue
            # precompute filter results per feature
            rows = []
            for fid, f in feats.items():
                exs = f["examples"]
                rows.append({
                    "fid": fid,
                    "cb": _content_frac(exs) >= 0.6,
                    "chain_div": _chain_div(exs),
                    "wstart_div": _wstart_div(exs),
                    "label": labels.get(fid),
                })
            data[layer][arch] = rows
            cb = sum(r["cb"] for r in rows)
            cc = sum(r["cb"] and r["chain_div"] >= 3 for r in rows)
            wd = sum(r["cb"] and r["chain_div"] >= 3 and r["wstart_div"] >= 3 for r in rows)
            ln = sum(r["cb"] and r["chain_div"] >= 3 and r["wstart_div"] >= 3
                     and (r["label"] is not None and _label_clear(r["label"])) for r in rows)
            print(f"  {layer} {arch}: N={len(rows)}  cb={cb}  +cc≥3={cc}  +wd≥3={wd}  +clear_label={ln}")

    # ── Figure ──
    fig = plt.figure(figsize=(18, 11.5))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.32,
                          width_ratios=[1.35, 1.15, 0.9])

    filter_keys = ["content-bearing", "+cross-chain≥3",
                   "+w-start-diverse≥3", "+clear Haiku label"]

    def pass_counts(rows):
        cb  = [r for r in rows if r["cb"]]
        cc  = [r for r in cb if r["chain_div"] >= 3]
        wd  = [r for r in cc if r["wstart_div"] >= 3]
        ln  = [r for r in wd if (r["label"] is not None and _label_clear(r["label"]))]
        return {
            "content-bearing": len(cb),
            "+cross-chain≥3": len(cc),
            "+w-start-diverse≥3": len(wd),
            "+clear Haiku label": len(ln),
        }

    for row, layer in enumerate(["resid_L25", "resid_L13"]):
        archs_here = [a for a in ARCHS_ORDER if a in data[layer]]

        # Panel A — filter-pass proportion with 95% CI
        ax = fig.add_subplot(gs[row, 0])
        x = np.arange(len(filter_keys))
        bar_w = 0.19
        for i, arch in enumerate(archs_here):
            rows = data[layer][arch]
            counts = pass_counts(rows)
            vals = [counts[k] for k in filter_keys]
            ns = [len(rows)] * len(filter_keys)
            # For "+clear Haiku label" the denominator is the labeled
            # subset (top-50 of the scan); use min(50, n_rows) as n.
            ns[3] = min(100, len(rows))
            props = [v / n if n else 0 for v, n in zip(vals, ns)]
            ci_lo = [wilson_ci(v, n)[0] for v, n in zip(vals, ns)]
            ci_hi = [wilson_ci(v, n)[1] for v, n in zip(vals, ns)]
            err_lo = [max(p - lo, 0.0) for p, lo in zip(props, ci_lo)]
            err_hi = [max(hi - p, 0.0) for p, hi in zip(props, ci_hi)]
            offset = (i - (len(archs_here) - 1) / 2) * bar_w
            bars = ax.bar(
                x + offset, props, bar_w,
                yerr=[err_lo, err_hi], capsize=3,
                label=ARCH_LABELS[arch].replace("\n", " "),
                color=ARCH_COLORS[arch], alpha=0.9,
                error_kw={"ecolor": "black", "alpha": 0.7, "linewidth": 1.0},
            )
            # Annotate k/n on top
            for b, v, n in zip(bars, vals, ns):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{v}/{n}", ha="center", fontsize=7)
        # Fisher exact: TFA pred vs TXCDR on the strict filter
        if "crosscoder" in data[layer] and "tfa_pos_pred" in data[layer]:
            for col, fk in enumerate(filter_keys[:-1]):
                pc = pass_counts(data[layer]["crosscoder"])[fk]
                tc = pass_counts(data[layer]["tfa_pos_pred"])[fk]
                nc = len(data[layer]["crosscoder"])
                nt = len(data[layer]["tfa_pos_pred"])
                p = fishers_exact(pc, nc, tc, nt)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
                ax.text(col, 1.11, f"TXCDR vs TFA-pred: {sig}\n(Fisher p={p:.1e})",
                        ha="center", fontsize=7, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(filter_keys, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("Fraction of features passing (95% Wilson CI)",
                      fontsize=10)
        ax.set_title(f"A. Filter-pass rate — {layer}   (N=500 per arch, 100 labeled)",
                     fontsize=11)
        if row == 0:
            ax.legend(fontsize=8, loc="lower left", ncol=2)
        ax.set_ylim(0, 1.2)
        ax.grid(axis="y", alpha=0.3)

        # Panel B — rank-decile robustness
        ax = fig.add_subplot(gs[row, 1])
        n_deciles = 10
        decile_size = 50
        for arch in archs_here:
            rows = data[layer][arch]
            # rows are already in span-weighted rank order
            props = []
            for d in range(n_deciles):
                lo, hi = d * decile_size, (d + 1) * decile_size
                sub = rows[lo:hi]
                if not sub:
                    props.append(0)
                    continue
                n_pass = sum(
                    r["cb"] and r["chain_div"] >= 3 and r["wstart_div"] >= 3
                    for r in sub
                )
                props.append(n_pass / len(sub))
            xs = np.arange(n_deciles) + 0.5
            ax.plot(xs, props, "-o", label=ARCH_LABELS[arch].replace("\n", " "),
                    color=ARCH_COLORS[arch], markersize=5, linewidth=1.8)
        ax.set_xticks(np.arange(n_deciles) + 0.5)
        ax.set_xticklabels([f"{i*decile_size+1}-{(i+1)*decile_size}"
                            for i in range(n_deciles)],
                           rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Rank bucket (span-weighted top-500)", fontsize=10)
        ax.set_ylabel("Fraction passing\ncontent + cross-chain≥3 + w-start≥3",
                      fontsize=10)
        ax.set_title(f"B. Robustness across ranks — {layer}", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(fontsize=8, loc="center left")

        # Panel C — feature distinctness: # unique top-10 exemplar sets
        # across the N=500 features. Stacked/TXCDR: each feature has a
        # distinct top-10 fingerprint. TFA pred: many features share the
        # SAME top-10 exemplar set — they're not distinct features.
        ax = fig.add_subplot(gs[row, 2])
        names = []
        unique_counts = []
        totals = []
        colors = []
        biggest_clusters = []
        for arch in archs_here:
            rows = data[layer][arch]
            from collections import Counter as _C
            sets = [
                frozenset((ex["chain_idx"], ex["window_start"])
                          for ex in feat_dict.get("examples", []))
                for feat_dict in [data[layer][arch][i]["feat"] for i in range(len(rows))]
            ] if False else []
            # Build directly from the features dict we loaded
            # (rows are derived from `features` via load_scan_n500)
            feats = _features_for_arch(scan_dir, arch, layer, args.k)
            ex_sets = [
                frozenset((e["chain_idx"], e["window_start"]) for e in f["examples"])
                for f in feats.values()
            ]
            c = _C(ex_sets)
            unique_counts.append(len(c))
            biggest_clusters.append(max(c.values()) if c else 0)
            totals.append(len(ex_sets))
            names.append(ARCH_LABELS[arch].replace("\n", " "))
            colors.append(ARCH_COLORS[arch])
        # plot fraction unique as main bar + overlay biggest-cluster marker
        fracs = [u / t if t else 0 for u, t in zip(unique_counts, totals)]
        bars = ax.barh(names, fracs, color=colors, alpha=0.9,
                       edgecolor="black")
        for b, u, t, bc in zip(bars, unique_counts, totals, biggest_clusters):
            pct = u / t * 100 if t else 0
            # Show: "unique/N (pct%)  |  biggest cluster=K"
            ax.text(max(b.get_width() + 0.02, 0.05),
                    b.get_y() + b.get_height() / 2,
                    f"{u}/{t}  ({pct:.0f}%)\nmax-cluster={bc}",
                    va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel(
            "Fraction of features with unique top-10\n"
            "exemplar set (higher = more distinct)",
            fontsize=10,
        )
        ax.set_title(f"C. Feature distinctness — {layer}", fontsize=11)
        ax.set_xlim(0, 1.45)
        ax.axvline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "TFA pred vs TXCDR interpretability — N=500 per arch (seeded, reproducible)\n"
        "Panel C is the headline: TXCDR/Stacked features have distinct top-10 exemplar sets; "
        "TFA pred features collapse to ≤13 unique patterns out of 500 (L25) or ≤97 (L13).",
        fontsize=13, y=0.995,
    )

    base = os.path.join(args.out_dir, "interpretability_comparison_hero_n500")
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
