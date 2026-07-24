"""Question-rate intensity λ̂_q on the Ward stream (factory candidate 2).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_qrate

Protocol frozen in `../qrate/CARD_DRAFT.md` BEFORE this ran. Deltas
from `build_sc_lambda`: events = sentence ends with "?"; mask = tokens
whose char span contains any "?" (`is_q_tok`); null seed 102. The
shared pipeline (binning, masked manifests, split, triage kill
authority) is `factory_lib.bundle_core`.

Arrays in ``qrate.npz`` mirror ``sc_lambda.npz`` with lam_q /
lam_q_null / lam_bin / lam_null_bin / is_q / is_q_tok + the standard
grid/manifest/split fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factory_lib as fl
from . import lib, wardmap
from .interleave_lib import rank_auc

HERE = Path(__file__).resolve().parent
NAME = "qrate"
SEED = 0
NULL_SEED = 102
EVIDENCE_TS = (8, 16, 32)
MIN_MANIFEST_CLASS = 2000


def main() -> None:
    tok, traces, by_qid = wardmap.load_inputs()
    trace_len = {}

    def payload(ti, ids, offs):
        trace_len[ti] = len(ids)
        text = traces[ti]["full_response"]
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        sents = [text[a:b] for a, b in spans]
        ev = fl.sentence_events_question(sents)
        lam = fl.kernel_rate(ev)
        rng = np.random.default_rng(NULL_SEED + ti)
        lam_null = fl.kernel_rate(fl.shuffle_events(ev.astype(float), rng))
        qspans = [(i, i + 1) for i, ch in enumerate(text) if ch == "?"]
        qtok = fl.token_mask_from_spans(offs, qspans)
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        out = {"lam_q": lam[sidx], "lam_q_null": lam_null[sidx],
               "is_q": ev[sidx], "is_q_tok": qtok,
               "tok_id": np.asarray(ids, dtype=np.int32),
               "sent_idx": sidx.astype(np.int16),
               "in_span": in_span.astype(np.int8)}
        for T in EVIDENCE_TS:
            out[f"ev{T}"] = fl.trailing_count_incl(qtok, T)
        return out

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    in_span = grids["in_span"] == 1
    valid = valid_rt & in_span
    lam, lam_null = grids["lam_q"], grids["lam_q_null"]
    lam[~valid] = np.nan
    lam_null[~valid] = np.nan

    core = fl.bundle_core(lam, lam_null, grids["is_q_tok"] == 1, valid,
                          trace_idx, win_start, trace_len,
                          grids["tok_id"], seed=SEED)
    man_doc, man_pos, man_cls = core["man"]

    evidence = {}
    for T in EVIDENCE_TS:
        sc = grids[f"ev{T}"][core["ext_rows_d"], core["ext_rows_p"]]
        ok = core["ext_is_test"] & np.isfinite(sc)
        evidence[f"T{T}"] = rank_auc(sc[ok], core["ext_is_top"][ok])

    is_q, lam_bin = grids["is_q"], core["bins"]
    mval = valid & (lam_bin >= 0)
    rate_by_bin = [float(np.mean(is_q[mval & (lam_bin == k)]))
                   for k in (0, 1, 2)]
    sc_l = np.load(HERE / "sc_lambda.npz")["lam_sc"]
    ward = np.load(HERE / "ward_lambda.npz")["lam_hist"]
    both_sc = valid & np.isfinite(lam) & np.isfinite(sc_l)
    both_w = valid & np.isfinite(lam) & np.isfinite(ward)
    fin_null = valid & np.isfinite(lam) & np.isfinite(lam_null)

    n_per_class = int(len(man_doc) // 3)
    stats = {
        "frozen": {"kernel": {"tau": fl.FROZEN_TAU, "k": fl.FROZEN_K,
                              "min_history": fl.MIN_HISTORY},
                   "event": "sentence ends with '?'",
                   "null_seed": NULL_SEED, "seed": SEED,
                   "mask": "is_q_tok (any '?' char) excluded from manifests"},
        "stream_shape": [int(v) for v in lam.shape],
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "event_rate_sentences": float(np.mean(is_q[valid] == 1)),
        "q_tok_rate": float((grids["is_q_tok"] == 1)[valid].mean()),
        "lam": {"mean": float(np.nanmean(lam)),
                "bin_scheme": core["scheme"], "edges": core["edges"]},
        "lam_null": {"bin_scheme": core["null_scheme"],
                     "edges": core["null_edges"],
                     "corr_with_real": float(np.corrcoef(
                         lam[fin_null], lam_null[fin_null])[0, 1])},
        "is_q_rate_by_bin": rate_by_bin,
        "corr_lam_q_lam_sc": float(np.corrcoef(
            lam[both_sc], sc_l[both_sc])[0, 1]),
        "corr_lam_q_ward_lam_hist": float(np.corrcoef(
            lam[both_w], ward[both_w])[0, 1]),
        "manifest_rows_per_class": n_per_class,
        "null_manifest_rows_per_class": int(len(core["man_null"][0]) // 3),
        "trace_split_test_frac": float(core["trace_split"].mean()),
        "triage": core["triage"],
        "visible_evidence_auc": evidence,
    }
    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    assert rate_by_bin[0] < rate_by_bin[2], (
        "no self-excitation: current-event rate not monotone in bin")
    assert n_per_class >= MIN_MANIFEST_CLASS, "manifest too small — kill"
    if core["triage"]["verdict"] == "FAIL":
        print(f"[TRIAGE FAIL] {NAME}: npz NOT shipped (free kill)")
        return
    np.savez_compressed(
        HERE / f"{NAME}.npz",
        lam_q=lam, lam_q_null=lam_null, lam_bin=lam_bin,
        lam_null_bin=core["null_bins"], is_q=is_q,
        is_q_tok=grids["is_q_tok"], sent_idx=grids["sent_idx"],
        in_span=in_span, valid=valid, trace_idx=trace_idx,
        win_start=win_start, man_doc=man_doc, man_pos=man_pos,
        man_cls=man_cls, man_null_doc=core["man_null"][0],
        man_null_pos=core["man_null"][1], man_null_cls=core["man_null"][2],
        trace_split=core["trace_split"])
    print(f"-> {HERE / f'{NAME}.npz'}")


if __name__ == "__main__":
    main()
