"""Label-side DESIGN PROBE for the dialevel screen — run BEFORE the card
is frozen, reads NO activations.

`dialevel` ships with the loudest caveat of the factory batch: the
all-eligible-row position AUC is 0.930-0.936 via a **dialogue-length
selection route** (with the turn-count floor fixed at 8, a dialogue is
long substantially BECAUSE its turns are long), so mac-local's binding
qualification 2 forecloses the naive screen: the card must neutralize
that route (within-dialogue contrasts or length matching) AND run
position/doc-length floor probes.

That obligation cannot be discharged by assertion — the within-dialogue
control has to have measurable POWER on this corpus before it is worth
a forward pass. Dialogues are short (median ~150 tokens), `tlevel` is
constant inside a turn, and the 5-turn warm-up leaves only the tail of
each dialogue labeled, so the number of distinct label values available
INSIDE one dialogue is small by construction. This script measures, on
labels alone:

1. **Cache geometry** under the replag chunking rule (non-overlapping
   128-token content rows, document tails dropped) and the row yield
   under the screen's uniform eligibility (`pos % content >= OFF_MIN`,
   `pos >= POS_MIN`, boundary tokens masked, `tlevel` finite).
2. **`doc_mean_only_auc`** — this agent's proposed triage bar (the one
   the frozen unigram/position bars cannot see): score every row by its
   DIALOGUE's mean `tlevel` and read the terciles. High here = the
   confound the qualification names.
3. **Within-dialogue contrast power** — how many dialogues carry >= 2
   distinct `tlevel` values among eligible rows, how many rows that
   yields per class, and how the within-dialogue |delta tlevel| compares
   to the global tercile contrast. If the within-dialogue contrast is a
   small fraction of the global one, the control is under-powered and
   the card must say so BEFORE the screen runs, not after.
4. **What the within-dialogue split does to the confound**: position
   AUC and dialogue-length AUC recomputed on the within-dialogue
   classes (dialogue length must fall to chance by construction;
   position is the number that decides whether a floor probe suffices).

Writes `results/design_probe.json`. No activations, no leaderboard rows.

Run: .venv/bin/python -m experiments.explorations.task_hunt.dialevel.design_probe
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.labels.interleave_lib import rank_auc
from experiments.explorations.task_hunt.novelty.screen import OFF_MIN, POS_MIN
from experiments.explorations.task_hunt.replag.build_labels import MODELS, SEQ_LEN

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
RES = HERE / "results"
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}


def _auc(scores, y, mask):
    m = mask & np.isfinite(scores)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float("nan")
    a = rank_auc(scores[m], y[m].astype(int))
    return float(max(a, 1.0 - a))          # direction-agnostic, as the bars


def geometry(key):
    """Rows per document under the replag chunking rule; flat row/pos
    index for every token that survives into a cached row."""
    z = np.load(LABELS / f"dialevel_dailydialog_{TOK_TAG[key]}.npz")
    off = z["doc_off"]
    n_prefix = 1 if MODELS[key]["bos"] else 0
    content = SEQ_LEN - n_prefix
    lens = np.diff(off)
    n_chunks = lens // content                      # tails dropped
    n = int(off[-1])
    doc_of = np.searchsorted(off, np.arange(n), side="right") - 1
    pos_of = np.arange(n) - off[doc_of]
    # a token survives iff its chunk index is inside its document's yield
    chunk = pos_of // content
    in_row = chunk < n_chunks[doc_of]
    cache_pos = n_prefix + pos_of % content
    return z, dict(n_prefix=n_prefix, content=content, lens=lens,
                   n_chunks=n_chunks, doc_of=doc_of, pos_of=pos_of,
                   chunk=chunk, in_row=in_row, cache_pos=cache_pos)


def probe(key):
    z, g = geometry(key)
    tlevel, boundary = z["tlevel"], z["is_boundary"]
    bins, split = z["tlevel_bin"], z["doc_split"]
    doc_of, pos_of, lens = g["doc_of"], g["pos_of"], g["lens"]
    fin = np.isfinite(tlevel)

    elig = (g["in_row"] & fin & (boundary == 0) & (pos_of >= POS_MIN)
            & (pos_of % g["content"] >= OFF_MIN))
    test = split[doc_of] == 1
    out = {
        "n_dialogues": int(len(lens)),
        "n_tokens": int(len(tlevel)),
        "n_prefix": g["n_prefix"], "content": int(g["content"]),
        "rows_cached": int(g["n_chunks"].sum()),
        "doc_len_median": float(np.median(lens)),
        "chunks_per_doc": {str(v): int(c) for v, c in
                           zip(*np.unique(g["n_chunks"], return_counts=True))},
        "tokens_kept_frac": float(g["in_row"].mean()),
        "labeled_frac": float(fin.mean()),
        "eligible_rows": int(elig.sum()),
        "eligible_docs": int(len(np.unique(doc_of[elig]))),
        "elig_pos_q": [float(q) for q in
                       np.quantile(pos_of[elig], [0.05, .5, .95])] if elig.any()
        else [],
    }
    for name, flag in (("train", 0), ("test", 1)):
        m = elig & (split[doc_of] == flag)
        out[f"class_counts_{name}"] = [int((m & (bins == v)).sum())
                                       for v in range(3)]

    # --- 2. doc-identity route (the proposed triage bar) ----------------
    dmean = np.full(len(lens), np.nan)
    np.add.at(dmean, doc_of[elig], 0)                    # touch = 0.0
    sums = np.bincount(doc_of[elig], weights=tlevel[elig], minlength=len(lens))
    cnts = np.bincount(doc_of[elig], minlength=len(lens))
    dmean = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    row_dmean = dmean[doc_of]
    tri = elig & test & (bins != 1) & (bins >= 0)
    out["doc_mean_only_auc"] = _auc(row_dmean, (bins == 2), tri)
    out["position_auc_eligible"] = _auc(pos_of.astype(float), (bins == 2), tri)
    out["doclen_auc_eligible"] = _auc(lens[doc_of].astype(float),
                                      (bins == 2), tri)
    # variance decomposition over the eligible pool
    v_tot = float(np.var(tlevel[elig]))
    v_bet = float(np.var(row_dmean[elig]))
    out["between_doc_var_frac"] = v_bet / v_tot if v_tot > 0 else float("nan")
    out["doclen_tlevel_corr"] = float(np.corrcoef(
        lens[cnts > 0], dmean[cnts > 0])[0, 1])

    # --- 3/4. within-dialogue contrast power ---------------------------
    # Inside one dialogue tlevel is constant across a turn, so the usable
    # contrast is between DISTINCT trailing-mean values in the pool.
    order = np.lexsort((tlevel[elig], doc_of[elig]))
    idx = np.flatnonzero(elig)[order]
    d_sorted, t_sorted = doc_of[idx], tlevel[idx]
    bounds = np.flatnonzero(np.r_[True, d_sorted[1:] != d_sorted[:-1],
                                  True])
    lo_cls = np.zeros(len(idx), dtype=np.int8) - 1
    n_distinct, deltas = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        vals = t_sorted[a:b]
        uq = np.unique(vals)
        n_distinct.append(len(uq))
        if len(uq) < 2:
            continue
        deltas.append(float(uq[-1] - uq[0]))
        lo_cls[a:b] = np.where(vals == uq[0], 0,
                               np.where(vals == uq[-1], 1, -1))
    n_distinct = np.asarray(n_distinct)
    wd = lo_cls >= 0
    wd_rows, wd_cls, wd_doc = idx[wd], lo_cls[wd], d_sorted[wd]
    gtop = tlevel[elig & test & (bins == 2)].mean() if (
        elig & test & (bins == 2)).any() else np.nan
    gbot = tlevel[elig & test & (bins == 0)].mean() if (
        elig & test & (bins == 0)).any() else np.nan
    out["within_doc"] = {
        "docs_with_eligible_rows": int(len(n_distinct)),
        "distinct_tlevel_per_doc_q": [float(q) for q in np.quantile(
            n_distinct, [0.1, 0.5, 0.9])] if len(n_distinct) else [],
        "docs_usable": int((n_distinct >= 2).sum()),
        "rows_class0": int((wd_cls == 0).sum()),
        "rows_class1": int((wd_cls == 1).sum()),
        "delta_tlevel_median": float(np.median(deltas)) if deltas else None,
        "global_tercile_contrast": float(gtop - gbot),
        "contrast_ratio": (float(np.median(deltas)) / float(gtop - gbot)
                           if deltas and np.isfinite(gtop - gbot) else None),
    }
    for name, flag in (("train", 0), ("test", 1)):
        m = split[wd_doc] == flag
        out["within_doc"][f"rows_{name}"] = [int((m & (wd_cls == 0)).sum()),
                                             int((m & (wd_cls == 1)).sum())]
        out["within_doc"][f"docs_{name}"] = int(len(np.unique(wd_doc[m])))
    te = split[wd_doc] == 1
    full = np.zeros(len(tlevel), dtype=bool)
    full[wd_rows] = True
    y2 = np.zeros(len(tlevel), dtype=np.int8)
    y2[wd_rows] = wd_cls
    mte = full.copy()
    mte[wd_rows] = te
    out["within_doc"]["position_auc"] = _auc(pos_of.astype(float), y2 == 1, mte)
    out["within_doc"]["doclen_auc"] = _auc(lens[doc_of].astype(float),
                                           y2 == 1, mte)
    out["within_doc"]["doc_mean_only_auc"] = _auc(row_dmean, y2 == 1, mte)
    out["within_doc"]["tst_auc"] = _auc(z["tst"].astype(float), y2 == 1, mte)
    return out


def main():
    RES.mkdir(exist_ok=True)
    res = {k: probe(k) for k in TOK_TAG}
    (RES / "design_probe.json").write_text(json.dumps(res, indent=1))
    for k, r in res.items():
        w = r["within_doc"]
        print(f"[{k}] rows={r['rows_cached']} elig={r['eligible_rows']} "
              f"docmean_auc={r['doc_mean_only_auc']:.3f} "
              f"pos_auc={r['position_auc_eligible']:.3f} "
              f"doclen_auc={r['doclen_auc_eligible']:.3f} "
              f"betweenvar={r['between_doc_var_frac']:.3f} || "
              f"WD docs={w['docs_usable']} rows={w['rows_class0']}/"
              f"{w['rows_class1']} dlt={w['delta_tlevel_median']} "
              f"ratio={w['contrast_ratio']} pos={w['position_auc']:.3f} "
              f"len={w['doclen_auc']:.3f} tst={w['tst_auc']:.3f}")
    print(f"-> {RES / 'design_probe.json'}")


if __name__ == "__main__":
    main()
