"""Verbosity LEVEL (+ slope secondary) on the Ward stream (factory 4).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_verbosity

Protocol frozen in `../verbosity/CARD_DRAFT.md` BEFORE this ran.
`vlevel` = trailing mean sentence length (previous min(i, 8)
sentences, current excluded, >= 4 required); `vslope` = OLS slope over
the same lengths, secondary, independently triaged. Null = within-trace
length permutation (seed 104 + trace_idx). Build-sanity monotone gate
applies to vlevel only (card states why the slope is exempt).

Arrays in ``verbosity.npz``: vlevel / vlevel_null / vlevel_bin /
vlevel_null_bin (+ the vslope family if it passed), ``cur_sent_len``
int16 (ambient control), ``tok_in_sent`` int16, standard grid fields,
per-label manifests ``man_vlevel_*`` / ``man_vslope_*`` (+ ``_null_``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factory_lib as fl
from . import lib, wardmap
from .interleave_lib import rank_auc

HERE = Path(__file__).resolve().parent
NAME = "verbosity"
SEED = 0
NULL_SEED = 104
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
        cnt = np.bincount(sidx[in_span], minlength=len(spans)).astype(float)
        lens = np.where(cnt > 0, cnt, np.nan)
        rng = np.random.default_rng(NULL_SEED + ti)
        lens_null = fl.shuffle_events(lens, rng)
        tok_in_sent = np.zeros(len(sidx), dtype=np.int16)
        for j in range(1, len(sidx)):
            tok_in_sent[j] = tok_in_sent[j - 1] + 1 \
                if sidx[j] == sidx[j - 1] else 0
        starts = (tok_in_sent == 0).astype(np.int8)
        out = {"vlevel": fl.trailing_mean_prev(lens)[sidx],
               "vlevel_null": fl.trailing_mean_prev(lens_null)[sidx],
               "vslope": fl.trailing_slope_prev(lens)[sidx],
               "vslope_null": fl.trailing_slope_prev(lens_null)[sidx],
               "cur_sent_len": np.where(np.isfinite(lens), lens,
                                        -1).astype(np.int16)[sidx],
               "tok_in_sent": tok_in_sent,
               "tok_id": np.asarray(ids, dtype=np.int32),
               "sent_idx": sidx.astype(np.int16),
               "in_span": in_span.astype(np.int8)}
        for T in EVIDENCE_TS:
            out[f"ev{T}"] = fl.trailing_count_incl(starts, T)
        return out

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    in_span = grids["in_span"] == 1
    valid = valid_rt & in_span
    cur_len = grids["cur_sent_len"]

    arrays = {"cur_sent_len": cur_len, "tok_in_sent": grids["tok_in_sent"],
              "sent_idx": grids["sent_idx"], "in_span": in_span,
              "valid": valid, "trace_idx": trace_idx,
              "win_start": win_start}
    stats = {
        "frozen": {"window": {"k": fl.FROZEN_K,
                              "min_prev": fl.MIN_HISTORY,
                              "aggregate": "unweighted mean / OLS slope, "
                                           "current sentence excluded"},
                   "null_seed": NULL_SEED, "seed": SEED, "mask": "none "
                   "(no token-local leak of previous lengths); "
                   "cur_sent_len is the disclosed ambient control"},
        "stream_shape": [int(v) for v in cur_len.shape],
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "labels": {},
    }
    shipped = []
    for key, gate_monotone in (("vlevel", True), ("vslope", False)):
        lam = grids[key]
        lam_null = grids[f"{key}_null"]
        lam[~valid] = np.nan
        lam_null[~valid] = np.nan
        core = fl.bundle_core(lam, lam_null,
                              np.zeros_like(valid, dtype=bool), valid,
                              trace_idx, win_start, trace_len,
                              grids["tok_id"], seed=SEED)
        evidence = {}
        for T in EVIDENCE_TS:
            sc = grids[f"ev{T}"][core["ext_rows_d"], core["ext_rows_p"]]
            ok = core["ext_is_test"] & np.isfinite(sc)
            evidence[f"T{T}"] = rank_auc(sc[ok], core["ext_is_top"][ok])
        bins = core["bins"]
        mval = valid & (bins >= 0) & (cur_len > 0)
        len_by_bin = [float(np.mean(cur_len[mval & (bins == k)]))
                      for k in (0, 1, 2)]
        fin = valid & np.isfinite(lam) & np.isfinite(lam_null)
        n_per_class = int(len(core["man"][0]) // 3)
        monotone = len_by_bin[0] < len_by_bin[2]
        lab = {
            "label_coverage_rate": float(np.isfinite(lam[valid]).mean()),
            "bin_scheme": core["scheme"], "edges": core["edges"],
            "null_corr_with_real": float(np.corrcoef(
                lam[fin], lam_null[fin])[0, 1]),
            "cur_sent_len_by_bin": len_by_bin,
            "manifest_rows_per_class": n_per_class,
            "triage": core["triage"],
            "visible_evidence_auc": evidence,
            "build_sanity": {"manifest_size_ok":
                             n_per_class >= MIN_MANIFEST_CLASS,
                             "autocorr_monotone": monotone,
                             "monotone_gated": gate_monotone},
        }
        stats["labels"][key] = lab
        if core["triage"]["verdict"] == "FAIL" \
                or n_per_class < MIN_MANIFEST_CLASS \
                or (gate_monotone and not monotone):
            lab["shipped"] = False
            continue
        lab["shipped"] = True
        shipped.append(key)
        arrays[key] = lam
        arrays[f"{key}_null"] = lam_null
        arrays[f"{key}_bin"] = bins
        arrays[f"{key}_null_bin"] = core["null_bins"]
        for tag, (d, p, c) in (("", core["man"]),
                               ("null_", core["man_null"])):
            arrays[f"man_{key}_{tag}doc"] = d
            arrays[f"man_{key}_{tag}pos"] = p
            arrays[f"man_{key}_{tag}cls"] = c
        arrays["trace_split"] = core["trace_split"]

    vl, vs = grids["vlevel"], grids["vslope"]
    both = valid & np.isfinite(vl) & np.isfinite(vs)
    stats["corr_vlevel_vslope"] = float(np.corrcoef(
        vl[both], vs[both])[0, 1])
    stats["shipped_labels"] = shipped

    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    if not shipped:
        print(f"[TRIAGE FAIL] {NAME}: all labels killed — npz NOT shipped")
        return
    np.savez_compressed(HERE / f"{NAME}.npz", **arrays)
    print(f"-> {HERE / f'{NAME}.npz'} (labels: {shipped})")


if __name__ == "__main__":
    main()
