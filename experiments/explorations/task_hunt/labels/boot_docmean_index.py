"""The `doc_mean_only_auc` threshold dataset, across every committed
label bundle (corpus-scaleup item 3, generalized).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.boot_docmean_index

Item 3 asked for bootstrap CIs on the novelty family "because its
numbers feed the same threshold dataset". The same argument covers every
bundle that ships a manifest: the review that will decide whether
`doc_mean_only_auc` becomes a frozen bar needs the statistic's
DISTRIBUTION over families, with intervals, not four point estimates.

Deliberately narrow, to stay honest about bundles I did not build:

- the row set is each bundle's OWN shipped manifest (`man_*` arrays),
  restricted to its test documents. A manifest is the author's own
  statement of which rows ship, masks and eligibility already applied,
  so nothing here re-interprets another agent's conventions;
- the only statistic is `doc_mean_only_auc` — document-mean of the
  label as the sole feature, class 2 vs class 0 — plus a 1,000-rep
  document-level bootstrap CI. No unigram or position numbers are
  recomputed here: those need each bundle's token stream and its own
  eligibility rules, and getting them subtly wrong would be worse than
  not reporting them;
- the Ward-stream bundles are (row, pos) windows over reasoning traces,
  so the clustering unit is the TRACE (`trace_idx`), not the stream row.

Writes ``docmean_index.json`` and prints the table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import boot_lib as bo

HERE = Path(__file__).resolve().parent
SEED = 0

# (label, npz, value key, manifest prefix, doc-map key, split key)
# doc-map "doc_off": flat token arrays, manifest doc/pos index into it.
# doc-map "trace_idx": Ward (row, pos) stream; manifest doc = stream row.
FLAT = "doc_off"
WARD = "trace_idx"
SPECS = [
    ("sc_lambda (Ward backtracking λ̂)", "sc_lambda.npz", "lam_sc",
     "man", WARD, "trace_split"),
    ("qrate (Ward question rate)", "qrate.npz", "lam_q",
     "man", WARD, "trace_split"),
    ("oprate ver (Ward verify-op rate)", "oprate.npz", "rate_ver",
     "man_ver", WARD, "trace_split"),
    ("oprate case (Ward case-op rate)", "oprate.npz", "rate_case",
     "man_case", WARD, "trace_split"),
    ("verbosity vslope (Ward)", "verbosity.npz", "vslope",
     "man_vslope", WARD, "trace_split"),
    ("dialevel tlevel (DailyDialog, gpt2)",
     "dialevel_dailydialog_gpt2.npz", "tlevel", "man_tlevel", FLAT,
     "doc_split"),
    ("interleave tss (fineweb pairs, gpt2)",
     "interleave_fineweb_gpt2.npz", "tss", "man_tss", FLAT, "doc_split"),
]


def rows_for(z, value_key, man_prefix, doc_map, split_key):
    """(scores, classes, cluster ids) on the bundle's own manifest rows,
    restricted to test documents."""
    d = z[f"{man_prefix}_doc"]
    p = z[f"{man_prefix}_pos"]
    c = z[f"{man_prefix}_cls"]
    raw = z[value_key]
    v = raw.astype(float)
    # the program's universal sentinel: -1 = undefined in INTEGER label
    # arrays (`lib.py` conventions; floats use NaN). Applying it is
    # following the convention, not reinterpreting it — without this an
    # integer face's document mean is dragged by its guard rows.
    if np.issubdtype(raw.dtype, np.integer):
        v = np.where(raw < 0, np.nan, v)
    split = z[split_key]
    if doc_map == FLAT:
        doc_off = z["doc_off"]
        n_docs = len(doc_off) - 1
        vals = v[doc_off[d] + p]
        cluster = d
        docmean = np.full(n_docs, np.nan)
        for i in range(n_docs):
            seg = v[doc_off[i]: doc_off[i + 1]]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                docmean[i] = seg.mean()
        test = split[d] == 1
    else:
        trace = z[WARD]
        cluster = trace[d]                    # the DOCUMENT is the trace
        vals = v[d, p]
        n_traces = int(trace.max()) + 1
        docmean = np.full(n_traces, np.nan)
        for t in range(n_traces):
            seg = v[trace == t]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                docmean[t] = seg.mean()
        test = split[cluster] == 1
    scores = docmean[cluster]
    keep = test & np.isfinite(scores) & ((c == 0) | (c == 2))
    return scores[keep], (c[keep] == 2).astype(int), cluster[keep], vals


def main():
    out = {"statistic": "doc_mean_only_auc on the bundle's OWN shipped "
                        "manifest rows, test documents only",
           "bootstrap": {"unit": "document / trace (cluster)",
                         "n_reps": bo.N_REPS, "ci_pct": list(bo.CI_PCT),
                         "seed": SEED},
           "bundles": []}
    for label, fname, vkey, mpre, dmap, skey in SPECS:
        path = HERE / fname
        if not path.exists():
            print(f"[skip] {label}: {fname} absent")
            continue
        z = np.load(path)
        s, y, cl, _ = rows_for(z, vkey, mpre, dmap, skey)
        b = bo.bootstrap_auc(s, y, cl, n_reps=bo.N_REPS, seed=SEED)
        rec = {"bundle": label, "artifact": fname,
               "cluster": "trace" if dmap == WARD else "document",
               "doc_mean_only_auc": b}
        out["bundles"].append(rec)
        print(f"{label:38s} {b['point_direction_agnostic']:.4f} "
              f"[{b['ci_lo_direction_agnostic']:.4f}, "
              f"{b['ci_hi_direction_agnostic']:.4f}] "
              f"({b['n_rows']:,} rows / {b['n_docs']} clusters)", flush=True)
    p = HERE / "docmean_index.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
