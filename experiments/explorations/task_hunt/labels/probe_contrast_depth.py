"""How much within-document contrast can a Stage-2 screen actually buy?
(corpus-scaleup, item 1/2 follow-up — label-side census, no new data.)

    .venv/bin/python -m experiments.explorations.task_hunt.labels.probe_contrast_depth

The scaled bundles ship position-matched manifests, which optimise
BREADTH: equal class counts inside every log2 position stratum, spread
thinly over thousands of documents. A within-document control wants the
opposite — DEPTH: many rows of both the top and the bottom class inside
the same document. The ladder in `SCALEUP.md` §5 answers "how many
documents qualify at a given per-class minimum"; this answers the
question a screen designer actually asks: **"if I take the K best
documents, how many rows per class do I get?"**

Pure census over the shipped manifest rows (test documents only) — it
ships no arrays and changes no manifest. Building a depth-first manifest
variant is a design decision for the screen owner; this is the number
that decision needs.

Writes ``contrast_depth.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KS = (10, 25, 50, 100, 200, 500)
SPECS = [
    ("punctint list (4,000 docs)", "punctint4k_fineweb_gpt2.npz", "man_list"),
    ("punctint q (4,000 docs)", "punctint4k_fineweb_gpt2.npz", "man_q"),
    ("refmark (2,000 convs)", "refmark2k_wildchat_gpt2.npz", "man_rlam"),
]


def census(z, prefix):
    d = z[f"{prefix}_doc"]
    c = z[f"{prefix}_cls"]
    split = z["doc_split"]
    keep = split[d] == 1
    d, c = d[keep], c[keep]
    n = int(split.size)
    n_top = np.bincount(d[c == 2], minlength=n)
    n_bot = np.bincount(d[c == 0], minlength=n)
    depth = np.minimum(n_top, n_bot)          # usable pairs-per-document
    order = np.argsort(-depth)
    out = {"test_docs_with_any_contrast": int((depth > 0).sum()),
           "total_usable_rows_per_class": int(depth.sum()),
           "by_top_k_documents": {}}
    for k in KS:
        take = order[:k]
        dk = depth[take]
        out["by_top_k_documents"][str(k)] = {
            "docs_with_contrast": int((dk > 0).sum()),
            "rows_per_class": int(dk.sum()),
            "min_rows_per_class_in_set": int(dk.min()) if dk.size else 0,
            "median_rows_per_class": float(np.median(dk)) if dk.size else 0.0}
    return out


def main():
    out = {"note": "test-document manifest rows only; 'rows per class' = "
                   "min(top, bottom) summed over the chosen documents — "
                   "the balanced within-document contrast a screen can form",
           "faces": {}}
    for label, fname, prefix in SPECS:
        z = np.load(HERE / fname)
        rec = census(z, prefix)
        out["faces"][label] = rec
        line = " ".join(f"K={k}:{rec['by_top_k_documents'][str(k)]['rows_per_class']:,}"
                        for k in KS)
        print(f"{label:28s} usable docs "
              f"{rec['test_docs_with_any_contrast']:4d}, total "
              f"{rec['total_usable_rows_per_class']:,} rows/class | {line}",
              flush=True)
    p = HERE / "contrast_depth.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
