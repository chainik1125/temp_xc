"""Self-correction marker intensity λ̂_sc on the Ward stream (factory 1).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_sc_lambda

Protocol frozen in `../sc_lambda/CARD_DRAFT.md` BEFORE this ran: the
marker list, kernel (exponential τ = 3, K = 8, causal, normalized,
history guard i ≥ 4), marker-token masking, binning fallback, triage
kill thresholds and the event-shuffle null all live in `factory_lib`.

Events are detected on `full_response[char_start:char_end]` (the exact
judged spans), so marker char spans offset back into trace coordinates
for the token mask. Label-side triage is the kill authority: on FAIL
the stats JSON (the kill receipt) is written and the npz is NOT.

Arrays in ``sc_lambda.npz`` (grids (4044, 128); -1/NaN undefined):
- ``lam_sc``      float32 — kernel-only trailing marker rate (primary);
- ``lam_sc_null`` float32 — same kernel over within-trace-shuffled
  events (seed 101 + trace_idx) — the trace-rate-preserving null;
- ``lam_bin`` / ``lam_null_bin`` int8 — 3-class targets (scheme in the
  stats JSON);
- ``is_sc``  int8 — current sentence contains a marker (ambient
  control, the `is_bt` analogue); ``is_marker_tok`` int8 — current
  token overlaps a marker span (masked out of every manifest);
- ``sent_idx`` int16, ``in_span``/``valid``, ``trace_idx``/``win_start``
  (N,), ``trace_split``; manifests ``man_*`` (primary) and
  ``man_null_*`` (null), both valid & unmasked & pos ≥ 32, balanced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factory_lib as fl
from . import lib, wardmap
from .interleave_lib import rank_auc

HERE = Path(__file__).resolve().parent
NAME = "sc_lambda"
SEED = 0
NULL_SEED = 101
EVIDENCE_TS = (8, 16, 32)
MIN_MANIFEST_CLASS = 2000


def build_grids():
    tok, traces, by_qid = wardmap.load_inputs()
    trace_len = {}

    def payload(ti, ids, offs):
        trace_len[ti] = len(ids)
        text = traces[ti]["full_response"]
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        sents = [text[a:b] for a, b in spans]
        ev = fl.sentence_events_markers(sents)
        lam = fl.kernel_rate(ev)
        rng = np.random.default_rng(NULL_SEED + ti)
        lam_null = fl.kernel_rate(fl.shuffle_events(ev.astype(float), rng))
        mspans = [(a + s, a + e) for (a, _), sent in zip(spans, sents)
                  for s, e in fl.marker_spans_in_sentence(sent)]
        mtok = fl.token_mask_from_spans(offs, mspans)
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        out = {"lam_sc": lam[sidx], "lam_sc_null": lam_null[sidx],
               "is_sc": ev[sidx], "is_marker_tok": mtok,
               "tok_id": np.asarray(ids, dtype=np.int32),
               "sent_idx": sidx.astype(np.int16),
               "in_span": in_span.astype(np.int8)}
        for T in EVIDENCE_TS:
            out[f"ev{T}"] = fl.trailing_count_incl(mtok, T)
        return out

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    return grids, valid_rt, trace_idx, win_start, n_mm, trace_len


def main() -> None:
    grids, valid_rt, trace_idx, win_start, n_mm, trace_len = build_grids()
    in_span = grids["in_span"] == 1
    valid = valid_rt & in_span
    lam, lam_null = grids["lam_sc"], grids["lam_sc_null"]
    lam[~valid] = np.nan
    lam_null[~valid] = np.nan
    masked = grids["is_marker_tok"] == 1

    scheme, edges, lam_bin = fl.zero_inflated_bins(lam)
    scheme_n, edges_n, lam_null_bin = fl.zero_inflated_bins(lam_null)

    N, S = lam.shape
    w_of, p_of = np.meshgrid(np.arange(N), np.arange(S), indexing="ij")
    manifests = {}
    for tag, bins in (("", lam_bin), ("null_", lam_null_bin)):
        cls = np.where(masked, -1, bins).astype(np.int8)
        d, p, c = lib.balanced_manifest(cls.ravel(), w_of.ravel(),
                                        p_of.ravel(), seed=SEED)
        manifests[tag] = (d, p, c)
    (man_doc, man_pos, man_cls) = manifests[""]
    tr_split = lib.doc_split(int(trace_idx.max()) + 1, seed=SEED)

    # label-side triage (kill authority) on primary manifest rows
    ext = (man_cls == 0) | (man_cls == 2)
    is_top = (man_cls[ext] == 2).astype(int)
    rows_d, rows_p = man_doc[ext], man_pos[ext]
    is_test = tr_split[trace_idx[rows_d]] == 1
    tok_rows = grids["tok_id"][rows_d, rows_p]
    o_raw = (win_start[rows_d] + rows_p - 1).astype(float)
    o_frac = o_raw / np.array([trace_len[int(t)]
                               for t in trace_idx[rows_d]], dtype=float)
    tok_auc = fl.token_id_triage_auc(tok_rows, is_top, ~is_test, is_test)
    pos_raw_auc = fl.position_triage_auc(o_raw, is_top, is_test)
    pos_frac_auc = fl.position_triage_auc(o_frac, is_top, is_test)
    triage = fl.triage_verdict(tok_auc, [pos_raw_auc, pos_frac_auc])
    triage.update(tok_auc=tok_auc, pos_raw_auc=pos_raw_auc,
                  pos_frac_auc=pos_frac_auc,
                  n_test_rows=int(is_test.sum()))

    # visible-evidence ceiling: in-window marker count alone, test rows
    evidence = {}
    for T in EVIDENCE_TS:
        sc = grids[f"ev{T}"][rows_d, rows_p]
        ok = is_test & np.isfinite(sc)
        evidence[f"T{T}"] = rank_auc(sc[ok], is_top[ok])

    # sanity: self-excitation — current-event rate monotone in the bin
    is_sc, mval = grids["is_sc"], valid & (lam_bin >= 0)
    rate_by_bin = [float(np.mean(is_sc[mval & (lam_bin == k)]))
                   for k in (0, 1, 2)]
    ward = np.load(HERE / "ward_lambda.npz")
    both = valid & np.isfinite(lam) & np.isfinite(ward["lam_hist"])
    corr_ward = float(np.corrcoef(lam[both], ward["lam_hist"][both])[0, 1])

    n_per_class = int(len(man_doc) // 3)
    stats = {
        "frozen": {"kernel": {"tau": fl.FROZEN_TAU, "k": fl.FROZEN_K,
                              "min_history": fl.MIN_HISTORY},
                   "marker_patterns": list(fl.MARKER_PATTERNS),
                   "null_seed": NULL_SEED, "seed": SEED,
                   "mask": "is_marker_tok excluded from all manifests"},
        "stream_shape": [int(N), int(S)],
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "event_rate_sentences": float(np.mean(is_sc[valid] == 1)),
        "marker_tok_rate": float(masked[valid].mean()),
        "lam": {"mean": float(np.nanmean(lam)), "bin_scheme": scheme,
                "edges": edges},
        "lam_null": {"bin_scheme": scheme_n, "edges": edges_n,
                     "corr_with_real": float(np.corrcoef(
                         lam[both], lam_null[both])[0, 1])},
        "is_sc_rate_by_bin": rate_by_bin,
        "corr_lam_sc_ward_lam_hist": corr_ward,
        "manifest_rows_per_class": n_per_class,
        "null_manifest_rows_per_class": int(len(manifests["null_"][0]) // 3),
        "trace_split_test_frac": float(tr_split.mean()),
        "triage": triage,
        "visible_evidence_auc": evidence,
    }
    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    assert rate_by_bin[0] < rate_by_bin[2], (
        "no self-excitation: current-event rate not monotone in bin")
    assert n_per_class >= MIN_MANIFEST_CLASS, "manifest too small — kill"
    if triage["verdict"] == "FAIL":
        print(f"[TRIAGE FAIL] {NAME}: npz NOT shipped (free kill)")
        return
    np.savez_compressed(
        HERE / f"{NAME}.npz",
        lam_sc=lam, lam_sc_null=lam_null, lam_bin=lam_bin,
        lam_null_bin=lam_null_bin, is_sc=is_sc,
        is_marker_tok=grids["is_marker_tok"], sent_idx=grids["sent_idx"],
        in_span=in_span, valid=valid, trace_idx=trace_idx,
        win_start=win_start, man_doc=man_doc, man_pos=man_pos,
        man_cls=man_cls, man_null_doc=manifests["null_"][0],
        man_null_pos=manifests["null_"][1],
        man_null_cls=manifests["null_"][2], trace_split=tr_split)
    print(f"-> {HERE / f'{NAME}.npz'}")


if __name__ == "__main__":
    main()
