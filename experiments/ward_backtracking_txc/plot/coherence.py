"""Coherence diagnostics for B1 steering: distinct-2, repetition rate,
unique-token ratio, and length, vs steering magnitude per source.

Reads results/ward_backtracking_txc/steering/b1_steering_results.json (which
already contains the full generated text per cell — no new generation needed)
and computes per-text coherence proxies, then averages across the 20 prompts
per (source, magnitude) cell.

Output: a 2x2 grid PNG, one panel per metric, plotted vs steering magnitude
with one line per source (DoM baselines + TXC headline features).
"""

from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.ward_backtracking_txc.plot._common import load_cfg, plots_dir

WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _distinct_n(toks: list[str], n: int) -> float:
    """Fraction of unique n-grams over all n-grams."""
    if len(toks) < n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def _max_repeat_run(toks: list[str]) -> int:
    """Longest run of the *same* word repeated consecutively. A 'wait wait wait'
    collapse goes to ~30+. Healthy text stays under ~3.
    """
    if not toks:
        return 0
    best = run = 1
    for a, b in zip(toks, toks[1:]):
        if a == b:
            run += 1; best = max(best, run)
        else:
            run = 1
    return best


def _ttr(toks: list[str]) -> float:
    return len(set(toks)) / max(1, len(toks))


def _coherence_proxies(text: str) -> dict[str, float]:
    toks = _tokens(text)
    return {
        "n_words": float(len(toks)),
        "distinct_2": _distinct_n(toks, 2),
        "ttr": _ttr(toks),
        "max_repeat_run": float(_max_repeat_run(toks)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    out_dir = plots_dir(cfg)
    res_path = Path(cfg["paths"]["steering"])
    if not res_path.exists():
        print(f"[skip] {res_path} missing"); return
    rows = json.loads(res_path.read_text())["rows"]

    # Compute coherence per row
    for r in rows:
        c = _coherence_proxies(r.get("text", ""))
        r.update({"_" + k: v for k, v in c.items()})

    # cells: (source, mag) -> list of per-prompt coherence dicts
    cells = defaultdict(list)
    for r in rows:
        cells[(r["source"], r["magnitude"])].append(r)

    mags = sorted({m for _, m in cells})

    # Curated source list: 2 DoM + best TXC pos0 + best TXC union per hookpoint.
    sources = sorted({s for s, _ in cells})
    keep = [
        "dom_base_union",
        "dom_reasoning_union",
        "txc_resid_L10_f1444_pos0",
        "txc_resid_L10_f1444_union",
        "txc_resid_L10_f4944_pos0",
        "txc_attn_L10_f3151_pos0",
    ]
    keep = [s for s in keep if s in sources]

    metric_specs = [
        ("_distinct_2", "Distinct-2 (unique bigrams / total bigrams)", "higher = more diverse"),
        ("_max_repeat_run", "Max consecutive same-word run", "higher = collapse"),
        ("_ttr", "Type-token ratio (unique words / total)", "higher = more diverse"),
        ("_n_words", "Output length (n_words)", "drops when model collapses early"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    palette = {
        "dom_base_union": "black",
        "dom_reasoning_union": "dimgray",
        "txc_resid_L10_f1444_pos0": "tab:green",
        "txc_resid_L10_f1444_union": "tab:red",
        "txc_resid_L10_f4944_pos0": "tab:orange",
        "txc_attn_L10_f3151_pos0": "tab:blue",
    }
    style = {
        "dom_base_union": "-",
        "dom_reasoning_union": "--",
        "txc_resid_L10_f1444_pos0": "-",
        "txc_resid_L10_f1444_union": "-",
        "txc_resid_L10_f4944_pos0": ":",
        "txc_attn_L10_f3151_pos0": ":",
    }

    for ax, (key, title, hint) in zip(axes.flat, metric_specs):
        for src in keep:
            xs, ys, ses = [], [], []
            for m in mags:
                vs = [r[key] for r in cells.get((src, m), [])]
                if vs:
                    xs.append(m); ys.append(float(np.mean(vs)))
                    ses.append(float(np.std(vs, ddof=1) / np.sqrt(len(vs))))
            ax.errorbar(xs, ys, yerr=ses, marker="o", lw=1.5,
                        color=palette.get(src, "gray"), linestyle=style.get(src, "-"),
                        label=src, capsize=2)
        ax.set_xlabel("Steering magnitude")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\n({hint})", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")

    fig.suptitle("B1 — coherence diagnostics vs steering magnitude (target = reasoning)",
                 fontsize=12)
    fig.tight_layout()
    out = out_dir / "coherence.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[saved] {out}")

    # Also dump a numerical table for the headline cells.
    print("\n=== Per-cell summary (mean across 20 prompts) ===")
    print(f"{'source':32s} {'mag':>5s} {'kw':>7s} {'n_words':>8s} {'dist2':>7s} {'ttr':>6s} {'max_run':>8s}")
    for src in keep:
        for m in mags:
            cs = cells.get((src, m), [])
            if not cs: continue
            kw = np.mean([r["keyword_rate"] for r in cs])
            nw = np.mean([r["_n_words"] for r in cs])
            d2 = np.mean([r["_distinct_2"] for r in cs])
            tt = np.mean([r["_ttr"] for r in cs])
            mr = np.mean([r["_max_repeat_run"] for r in cs])
            print(f"{src:32s} {m:>+5.0f} {kw:>7.4f} {nw:>8.0f} {d2:>7.3f} {tt:>6.3f} {mr:>8.1f}")


if __name__ == "__main__":
    main()
