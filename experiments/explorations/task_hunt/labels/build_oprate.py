"""Operation-class run-rates on the Ward stream (factory candidate 3, ×2).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_oprate

Protocol frozen in `../oprate/CARD_DRAFT.md` BEFORE this ran. Two
labels from the committed proofops 5-class sentence labels:
``rate_ver`` (class 3) and ``rate_case`` (class 2), each an exponential
kernel rate (τ = 3, K = 8, causal) with NaN wherever any kernel-lag
sentence is unlabeled. Per-label masks (current sentence = event class
OR unlabeled), independent triage; a failing label's manifests are
dropped, the npz ships iff at least one label passes.

Arrays in ``oprate.npz``: rate_ver / rate_ver_null / ver_bin /
ver_null_bin + the case_* family, ``op`` int8 (ambient anchor),
standard grid fields, and per-label manifests ``man_ver_*`` /
``man_ver_null_*`` / ``man_case_*`` / ``man_case_null_*`` (only for
labels that passed triage — the stats JSON records which).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factory_lib as fl
from . import lib, wardmap
from .interleave_lib import rank_auc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABELS = (ROOT / "experiments/explorations/synthetic/expansion/records/"
          "proof-operation-phase-runs/labels.json")
NAME = "oprate"
SEED = 0
NULL_SEED = 103
EVIDENCE_TS = (8, 16, 32)
MIN_MANIFEST_CLASS = 2000
CLASSES = {"ver": 3, "case": 2}


def main() -> None:
    tok, traces, by_qid = wardmap.load_inputs()
    rec = json.loads(LABELS.read_text())
    ops_by_qid = dict(zip(rec["doc_ids"], rec["labels"]))
    trace_len = {}

    def payload(ti, ids, offs):
        trace_len[ti] = len(ids)
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        labels = ops_by_qid.get(traces[ti]["question_id"])
        if labels is None:
            labels = [None] * len(spans)
        op_s = np.array([-1 if l is None else l for l in labels],
                        dtype=np.int8)
        rng = np.random.default_rng(NULL_SEED + ti)
        lab_pos = np.flatnonzero(op_s >= 0)
        op_null = op_s.copy()
        op_null[lab_pos] = op_s[lab_pos][rng.permutation(lab_pos.size)]
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        out = {"op": op_s[sidx], "tok_id": np.asarray(ids, dtype=np.int32),
               "sent_idx": sidx.astype(np.int16),
               "in_span": in_span.astype(np.int8)}
        for key, cls in CLASSES.items():
            ev = np.where(op_s >= 0, (op_s == cls).astype(float), np.nan)
            evn = np.where(op_null >= 0, (op_null == cls).astype(float),
                           np.nan)
            out[f"rate_{key}"] = fl.kernel_rate(ev)
            out[f"rate_{key}_null"] = fl.kernel_rate(evn)
            flags = (op_s[sidx] == cls).astype(np.int8)
            for T in EVIDENCE_TS:
                out[f"ev_{key}{T}"] = fl.trailing_count_incl(flags, T)
        return out

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    in_span = grids["in_span"] == 1
    valid = valid_rt & in_span
    op = grids["op"]

    arrays = {"op": op, "sent_idx": grids["sent_idx"], "in_span": in_span,
              "valid": valid, "trace_idx": trace_idx,
              "win_start": win_start}
    stats = {
        "frozen": {"kernel": {"tau": fl.FROZEN_TAU, "k": fl.FROZEN_K,
                              "min_history": fl.MIN_HISTORY},
                   "labels_source": str(LABELS.relative_to(ROOT)),
                   "classes": CLASSES, "null_seed": NULL_SEED, "seed": SEED,
                   "mask": "current sentence == event class OR unlabeled",
                   "nan_rule": "label NaN if any kernel-lag sentence "
                               "unlabeled"},
        "stream_shape": [int(v) for v in op.shape],
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "labels": {},
    }
    shipped = []
    for key, cls in CLASSES.items():
        lam = grids[f"rate_{key}"]
        lam_null = grids[f"rate_{key}_null"]
        lam[~valid] = np.nan
        lam_null[~valid] = np.nan
        mask_rows = (op == cls) | (op == -1)
        core = fl.bundle_core(lam, lam_null, mask_rows, valid, trace_idx,
                              win_start, trace_len, grids["tok_id"],
                              seed=SEED)
        evidence = {}
        for T in EVIDENCE_TS:
            sc = grids[f"ev_{key}{T}"][core["ext_rows_d"],
                                       core["ext_rows_p"]]
            ok = core["ext_is_test"] & np.isfinite(sc)
            evidence[f"T{T}"] = rank_auc(sc[ok], core["ext_is_top"][ok])
        bins = core["bins"]
        mval = valid & (bins >= 0) & (op >= 0)
        rate_by_bin = [float(np.mean(op[mval & (bins == k)] == cls))
                       for k in (0, 1, 2)]
        fin = valid & np.isfinite(lam) & np.isfinite(lam_null)
        n_per_class = int(len(core["man"][0]) // 3)
        lab = {
            "label_coverage_rate": float(np.isfinite(lam[valid]).mean()),
            "event_rate_sentences": float(np.mean(op[valid & (op >= 0)]
                                                  == cls)),
            "bin_scheme": core["scheme"], "edges": core["edges"],
            "null_corr_with_real": float(np.corrcoef(
                lam[fin], lam_null[fin])[0, 1]),
            "cur_class_rate_by_bin": rate_by_bin,
            "manifest_rows_per_class": n_per_class,
            "triage": core["triage"],
            "visible_evidence_auc": evidence,
        }
        stats["labels"][key] = lab
        ok_size = n_per_class >= MIN_MANIFEST_CLASS
        monotone = rate_by_bin[0] < rate_by_bin[2]
        lab["build_sanity"] = {"manifest_size_ok": ok_size,
                               "self_excitation_monotone": monotone}
        if core["triage"]["verdict"] == "FAIL" or not ok_size \
                or not monotone:
            lab["shipped"] = False
            continue
        lab["shipped"] = True
        shipped.append(key)
        arrays[f"rate_{key}"] = lam
        arrays[f"rate_{key}_null"] = lam_null
        arrays[f"{key}_bin"] = bins
        arrays[f"{key}_null_bin"] = core["null_bins"]
        for tag, (d, p, c) in (("", core["man"]), ("null_",
                                                   core["man_null"])):
            arrays[f"man_{key}_{tag}doc"] = d
            arrays[f"man_{key}_{tag}pos"] = p
            arrays[f"man_{key}_{tag}cls"] = c
        arrays["trace_split"] = core["trace_split"]

    rv, rc = grids["rate_ver"], grids["rate_case"]
    both = valid & np.isfinite(rv) & np.isfinite(rc)
    stats["corr_rate_ver_rate_case"] = float(np.corrcoef(
        rv[both], rc[both])[0, 1])
    sc_l = np.load(HERE / "sc_lambda.npz")["lam_sc"]
    both_sc = valid & np.isfinite(rv) & np.isfinite(sc_l)
    stats["corr_rate_ver_lam_sc"] = float(np.corrcoef(
        rv[both_sc], sc_l[both_sc])[0, 1])
    stats["shipped_labels"] = shipped

    (HERE / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    if not shipped:
        print(f"[TRIAGE FAIL] {NAME}: BOTH labels killed — npz NOT shipped")
        return
    np.savez_compressed(HERE / f"{NAME}.npz", **arrays)
    print(f"-> {HERE / f'{NAME}.npz'} (labels: {shipped})")


if __name__ == "__main__":
    main()
