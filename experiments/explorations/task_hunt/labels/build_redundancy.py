"""Window redundancy rate ρ̂ on the Ward stream (factory candidate 5).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_redundancy

Protocol frozen in `../redundancy/CARD_DRAFT.md` BEFORE this ran:
b_t = bigram-seen-earlier flag, ρ̂ = trailing W = 32 rate (current
token excluded), is_rep rows masked from manifests, token-shuffle
frequency null (seed 105 + trace_idx), standard triage kill authority
(position flagged as the expected failure face).

Arrays in ``redundancy.npz``: red / red_null / red_bin / red_null_bin,
``is_rep`` int8 (ambient control + the mask), standard grid fields,
manifests ``man_*`` / ``man_null_*``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factory_lib as fl
from . import lib, wardmap
from .interleave_lib import rank_auc

HERE = Path(__file__).resolve().parent
NAME = "redundancy"
SEED = 0
NULL_SEED = 105
W = 32
EVIDENCE_TS = (8, 16, 32)
MIN_MANIFEST_CLASS = 2000


def main() -> None:
    tok, traces, by_qid = wardmap.load_inputs()
    trace_len = {}

    def payload(ti, ids, offs):
        trace_len[ti] = len(ids)
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        b = (lib.delta_prev_ngram(ids, 2) > 0).astype(np.int8)
        rng = np.random.default_rng(NULL_SEED + ti)
        ids_null = np.asarray(ids)[rng.permutation(len(ids))]
        b_null = (lib.delta_prev_ngram(ids_null, 2) > 0).astype(np.int8)
        out = {"red": fl.trailing_rate_prev(b, W),
               "red_null": fl.trailing_rate_prev(b_null, W),
               "is_rep": b, "tok_id": np.asarray(ids, dtype=np.int32),
               "sent_idx": sidx.astype(np.int16),
               "in_span": in_span.astype(np.int8)}
        for T in EVIDENCE_TS:
            out[f"ev{T}"] = fl.trailing_count_incl(b, T)
        return out

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    in_span = grids["in_span"] == 1
    valid = valid_rt & in_span
    red, red_null = grids["red"], grids["red_null"]
    red[~valid] = np.nan
    red_null[~valid] = np.nan
    is_rep = grids["is_rep"]

    core = fl.bundle_core(red, red_null, is_rep == 1, valid, trace_idx,
                          win_start, trace_len, grids["tok_id"], seed=SEED)
    man_doc, man_pos, man_cls = core["man"]

    evidence = {}
    for T in EVIDENCE_TS:
        sc = grids[f"ev{T}"][core["ext_rows_d"], core["ext_rows_p"]]
        ok = core["ext_is_test"] & np.isfinite(sc)
        evidence[f"T{T}"] = rank_auc(sc[ok], core["ext_is_top"][ok])

    bins = core["bins"]
    mval = valid & (bins >= 0) & (is_rep >= 0)
    rep_by_bin = [float(np.mean(is_rep[mval & (bins == k)] == 1))
                  for k in (0, 1, 2)]
    fin = valid & np.isfinite(red) & np.isfinite(red_null)
    n_per_class = int(len(man_doc) // 3)
    stats = {
        "frozen": {"W": W, "event": "bigram ending at t seen earlier "
                   "in trace", "null_seed": NULL_SEED, "seed": SEED,
                   "mask": "is_rep == 1 excluded from all manifests"},
        "stream_shape": [int(v) for v in red.shape],
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "is_rep_rate": float((is_rep == 1)[valid].mean()),
        "red": {"mean": float(np.nanmean(red)),
                "bin_scheme": core["scheme"], "edges": core["edges"]},
        "red_null": {"bin_scheme": core["null_scheme"],
                     "edges": core["null_edges"],
                     "corr_with_real": float(np.corrcoef(
                         red[fin], red_null[fin])[0, 1])},
        "is_rep_rate_by_bin": rep_by_bin,
        "manifest_rows_per_class": n_per_class,
        "null_manifest_rows_per_class": int(len(core["man_null"][0]) // 3),
        "trace_split_test_frac": float(core["trace_split"].mean()),
        "triage": core["triage"],
        "visible_evidence_auc": evidence,
    }
    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    assert rep_by_bin[0] < rep_by_bin[2], (
        "no autocorrelation: is_rep rate not monotone in bin")
    assert n_per_class >= MIN_MANIFEST_CLASS, "manifest too small — kill"
    if core["triage"]["verdict"] == "FAIL":
        print(f"[TRIAGE FAIL] {NAME}: npz NOT shipped (free kill)")
        return
    np.savez_compressed(
        HERE / f"{NAME}.npz",
        red=red, red_null=red_null, red_bin=bins,
        red_null_bin=core["null_bins"], is_rep=is_rep,
        sent_idx=grids["sent_idx"], in_span=in_span, valid=valid,
        trace_idx=trace_idx, win_start=win_start, man_doc=man_doc,
        man_pos=man_pos, man_cls=man_cls,
        man_null_doc=core["man_null"][0],
        man_null_pos=core["man_null"][1],
        man_null_cls=core["man_null"][2],
        trace_split=core["trace_split"])
    print(f"-> {HERE / f'{NAME}.npz'}")


if __name__ == "__main__":
    main()
