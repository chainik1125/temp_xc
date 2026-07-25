"""Depth-first within-document row-sets + demeaning sufficient
statistics (candidate-factory-broad-3 ADDENDUM item 3 — a label-side
assist to the two Stage-2 panels; strictly NON-BLOCKING both
directions: the panels compute their own if this is not there in time).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_depth_rowsets

The contrast-depth census (``contrast_depth.json``) ended with "the
depth-first variant is the screen owner's call"; ``stage2-fineweb.md``
made that call — the within-document receipt on punctint q is BINDING
at panel. This ships, WITHOUT touching any shipped bundle:

- ``punctint_q_wdrows_<tok>.npz`` (400-doc grid — the panel's) and
  ``punctint4k_q_wdrows_<tok>.npz`` (scaled, optional): for thresholds
  t in {20, 50} (the census ladder), ALL manifest rows lying in
  documents that hold >= t manifest rows of BOTH the top and the
  bottom class (``wd<t>_doc/pos/cls``; middle-tercile rows in those
  documents are included so the panel can filter, the qualification
  itself uses top/bottom counts only), plus the per-document
  top/bottom manifest-row counts so any other threshold is
  re-derivable, plus per-document SUFFICIENT STATISTICS for demeaning:
  sum / count / sum-of-squares of ``lam_q`` over (a) all
  finite-label rows and (b) screen-eligible rows (event-masked bin
  >= 0 and pos >= MIN_MANIFEST_POS).
- ``oprate_case_tracestats.json``: the same statistics per TRACE for
  the Ward ``rate_case`` target (runpod-d's panel's trace-mean floor),
  over (a) valid finite cells and (b) manifest rows.

DELIBERATELY NOT SHIPPED: any pre-demeaned array. Demeaning must be
split-consistent, and the split discipline belongs to the panel.
Per-document (per-trace) statistics are split-ATOMIC — the split is by
document/trace, so any train-restricted or test-restricted mean can be
formed from these sums with no row-level leakage. That statement is
repeated in the stats JSON where the panel will read it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib

HERE = Path(__file__).resolve().parent
THRESHOLDS = (20, 50)
TOKENIZERS = ("gpt2", "gemma2", "llama31")
GRIDS = {"punctint": "punctint_fineweb_{tok}.npz",
         "punctint4k": "punctint4k_fineweb_{tok}.npz"}


def suff_stats(vals, mask, group_of, n_groups):
    """(sum, count, sumsq) of vals over mask rows, grouped."""
    s = np.zeros(n_groups)
    c = np.zeros(n_groups, dtype=np.int64)
    s2 = np.zeros(n_groups)
    g, v = group_of[mask], vals[mask].astype(np.float64)
    np.add.at(s, g, v)
    np.add.at(c, g, 1)
    np.add.at(s2, g, v * v)
    return s, c, s2


def build_punctint(grid, tok):
    z = np.load(HERE / GRIDS[grid].format(tok=tok))
    doc_off, split = z["doc_off"], z["doc_split"]
    n_docs = len(split)
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    lam, bins, evt = z["lam_q"], z["q_bin"], z["is_q"]
    masked = np.where(evt == 1, -1, bins).astype(np.int8)
    fin = np.isfinite(lam)
    elig = fin & (masked >= 0) & (pos_of >= lib.MIN_MANIFEST_POS)

    d, p, c = z["man_q_doc"], z["man_q_pos"], z["man_q_cls"]
    n_top = np.bincount(d[c == 2], minlength=n_docs)
    n_bot = np.bincount(d[c == 0], minlength=n_docs)

    arrays = {"doc_split": split,
              "doc_q_top_rows": n_top.astype(np.int32),
              "doc_q_bot_rows": n_bot.astype(np.int32)}
    for name, mask in (("finite", fin), ("elig", elig)):
        s, cnt, s2 = suff_stats(lam, mask, doc_of, n_docs)
        arrays[f"doc_lamq_sum_{name}"] = s
        arrays[f"doc_lamq_cnt_{name}"] = cnt
        arrays[f"doc_lamq_sumsq_{name}"] = s2

    st = {}
    for t in THRESHOLDS:
        qual = (n_top >= t) & (n_bot >= t)
        keep = qual[d]
        arrays[f"wd{t}_doc"] = d[keep]
        arrays[f"wd{t}_pos"] = p[keep]
        arrays[f"wd{t}_cls"] = c[keep]
        st[str(t)] = {
            "docs": int(qual.sum()),
            "docs_train": int((qual & (split == 0)).sum()),
            "docs_test": int((qual & (split == 1)).sum()),
            "rows": int(keep.sum()),
            "rows_top": int((c[keep] == 2).sum()),
            "rows_bottom": int((c[keep] == 0).sum()),
            "min_rows_per_class_per_doc": t}
    out = HERE / f"{grid}_q_wdrows_{tok}.npz"
    np.savez_compressed(out, **arrays)
    print(f"[{grid}/{tok}] " + " ".join(
        f">= {t}/class: {st[str(t)]['docs']} docs "
        f"({st[str(t)]['docs_test']} test, {st[str(t)]['rows']:,} rows)"
        for t in THRESHOLDS), flush=True)
    return {"artifact": out.name, "n_docs": int(n_docs),
            "eligible_rows": int(elig.sum()),
            "finite_rows": int(fin.sum()),
            "by_threshold": st}


def build_oprate():
    z = np.load(HERE / "oprate.npz")
    rate, valid = z["rate_case"], z["valid"]
    tr, tsplit = z["trace_idx"], z["trace_split"]
    n_traces = len(tsplit)
    cell_ok = valid & np.isfinite(rate)
    trace_flat = np.repeat(tr, rate.shape[1])
    s, cnt, s2 = suff_stats(rate.ravel(), cell_ok.ravel(),
                            trace_flat, n_traces)
    md, mp = z["man_case_doc"], z["man_case_pos"]
    mvals = rate[md, mp]
    mok = np.isfinite(mvals)
    ms, mcnt, ms2 = suff_stats(mvals, mok, tr[md], n_traces)
    out = {
        "target": "rate_case (oprate.npz, Ward window grid)",
        "note": ("SUFFICIENT STATISTICS ONLY — no pre-demeaned array is "
                 "shipped. Demeaning must be split-consistent; these "
                 "per-trace sums are split-atomic (the split is by "
                 "trace), so form any train-/test-restricted mean from "
                 "them. all_cells = valid & finite cells of the "
                 "(4044, 128) grid; manifest_rows = man_case_* rows "
                 f"({int((~mok).sum())} non-finite manifest rows "
                 "excluded)."),
        "trace_split": tsplit.tolist(),
        "all_cells": {"sum": s.tolist(), "cnt": cnt.tolist(),
                      "sumsq": s2.tolist()},
        "manifest_rows": {"sum": ms.tolist(), "cnt": mcnt.tolist(),
                          "sumsq": ms2.tolist()},
    }
    p = HERE / "oprate_case_tracestats.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[oprate/case] {n_traces} traces, "
          f"{int(cell_ok.sum()):,} valid cells, "
          f"{int(mok.sum()):,} manifest rows -> {p.name}", flush=True)
    return {"artifact": p.name, "n_traces": int(n_traces),
            "valid_cells": int(cell_ok.sum()),
            "manifest_rows": int(mok.sum())}


def main():
    stats = {
        "purpose": ("depth-first within-document row-sets + demeaning "
                    "sufficient statistics; broad-3 addendum item 3; "
                    "non-blocking assist to stage2-fineweb (punctint q) "
                    "and stage2-oprate (rate_case)"),
        "demeaning_contract": (
            "STATISTICS, NOT PRE-DEMEANED ARRAYS: the panel applies its "
            "own split-consistent demeaning. Per-document/per-trace "
            "sums are split-atomic (splits are by document/trace), so "
            "train- or test-restricted means come from these sums with "
            "no row-level leakage."),
        "row_set_rule": (
            "wd<t>_* = ALL manifest rows (all classes) in documents "
            "holding >= t manifest rows of BOTH class 0 and class 2; "
            "qualification uses top/bottom counts only; per-document "
            "top/bottom counts ship so other thresholds are "
            "re-derivable"),
        "eligibility": {
            "finite": "isfinite(lam_q)",
            "elig": "isfinite(lam_q) & event-masked bin >= 0 & "
                    f"pos >= {lib.MIN_MANIFEST_POS}"},
        "thresholds": list(THRESHOLDS),
        "punctint_q": {}, "punctint4k_q": {},
    }
    for grid, key in (("punctint", "punctint_q"),
                      ("punctint4k", "punctint4k_q")):
        for tok in TOKENIZERS:
            stats[key][tok] = build_punctint(grid, tok)
    stats["oprate_case"] = build_oprate()
    p = HERE / "depth_rowsets_stats.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
