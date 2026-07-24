"""Proof-operation run features on the Ward stream (candidate 3).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_proofops

Broadcasts the frozen 5-class per-sentence operation labels
(``synthetic/expansion/records/proof-operation-phase-runs/labels.json``,
judged sentence-in-isolation per its prereg card; classes 0 other /
1 algebraic-manipulation / 2 case-enumeration / 3 verification-check /
4 restatement-setup; unlabeled sentences → -1 and BREAK runs) onto the
canonical Ward cache grid, with run features computed at SENTENCE level:

- ``op``           int8  — operation class of the containing sentence
  (per-sentence readable BY CONSTRUCTION — the regime-1 control target);
- ``time_in_run``  int32 — 0-based index of the sentence within its
  constant-label run (the temporal target);
- ``is_run_start`` int8  — 1 iff the sentence starts a run (boundary);
- ``sent_idx`` int16, ``tok_in_sent`` int16, ``valid`` bool,
  ``trace_idx``/``win_start`` (N,), ``trace_split`` per trace;
- manifests: ``man_op_*`` (5-class balanced — the ambient anchor),
  ``man_boundary_*`` (run start vs interior), ``man_tir_*`` (time-in-run
  binned {0, 1, ≥2}); all valid & p ≥ 32.

The sentence→token CLOCK BRIDGE (substrate-audit item 6) is measured and
printed: tokens-per-sentence distribution under the Ward tokenizer and
the implied sentences-per-window at each screen T — the T range for any
screen on these targets must be chosen from these numbers, not assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib, wardmap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABELS = (ROOT / "experiments/explorations/synthetic/expansion/records/"
          "proof-operation-phase-runs/labels.json")
SEED = 0
SCREEN_TS = (8, 16, 32, 64, 128)


def main() -> None:
    tok, traces, by_qid = wardmap.load_inputs()
    rec = json.loads(LABELS.read_text())
    ops_by_qid = dict(zip(rec["doc_ids"], rec["labels"]))

    tokens_per_sentence = []

    def payload(ti, ids, offs):
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        labels = ops_by_qid.get(traces[ti]["question_id"])
        if labels is None:
            labels = [None] * len(spans)
        op_s, tir_s, start_s = lib.run_features(labels)
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        # clock bridge: tokens per sentence (in-span tokens only)
        cnt = np.bincount(sidx[in_span], minlength=len(spans))
        tokens_per_sentence.extend(cnt[cnt > 0].tolist())
        tok_in_sent = np.zeros(len(sidx), dtype=np.int16)
        for j in range(1, len(sidx)):
            tok_in_sent[j] = tok_in_sent[j - 1] + 1 \
                if sidx[j] == sidx[j - 1] else 0
        return {"op": op_s[sidx], "time_in_run": tir_s[sidx],
                "is_run_start": start_s[sidx],
                "sent_idx": sidx.astype(np.int16),
                "tok_in_sent": tok_in_sent,
                "in_span": in_span.astype(np.int8)}

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    valid = valid_rt & (grids["in_span"] == 1)
    op, tir, start = grids["op"], grids["time_in_run"], grids["is_run_start"]
    labeled = valid & (op >= 0)

    N, S = op.shape
    w_of, p_of = np.meshgrid(np.arange(N), np.arange(S), indexing="ij")
    arrays = {
        "op": op, "time_in_run": tir, "is_run_start": start,
        "sent_idx": grids["sent_idx"], "tok_in_sent": grids["tok_in_sent"],
        "valid": valid, "labeled": labeled,
        "trace_idx": trace_idx, "win_start": win_start,
        "trace_split": lib.doc_split(int(trace_idx.max()) + 1, seed=SEED),
    }
    tir_bin = np.where(labeled, np.minimum(tir, 2), -1).astype(np.int8)
    for name, cls in (("op", np.where(labeled, op, -1)),
                      ("boundary", np.where(labeled, start, -1)),
                      ("tir", tir_bin)):
        d, p, c = lib.balanced_manifest(cls.ravel(), w_of.ravel(),
                                        p_of.ravel(), seed=SEED)
        arrays[f"man_{name}_doc"], arrays[f"man_{name}_pos"] = d, p
        arrays[f"man_{name}_cls"] = c
    np.savez_compressed(HERE / "proofops.npz", **arrays)

    tps = np.array(tokens_per_sentence, dtype=float)
    clock = {
        "tokens_per_sentence": {
            "mean": float(tps.mean()), "median": float(np.median(tps)),
            "p10": float(np.percentile(tps, 10)),
            "p90": float(np.percentile(tps, 90)), "n": int(len(tps))},
        "sentences_per_window_at_T": {
            str(T): round(T / float(np.median(tps)), 2) for T in SCREEN_TS},
        "min_T_spanning_2_sentences_median_clock":
            int(np.ceil(2 * float(np.median(tps)))),
    }
    stats = {
        "labels_source": str(LABELS.relative_to(ROOT)),
        "coverage": rec["coverage"], "seed": SEED,
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "labeled_rate_pos1plus": float(labeled[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "op_class_counts": {str(k): int((op[labeled] == k).sum())
                            for k in range(5)},
        "boundary_rate_sentencelevel_proxy":
            float(np.mean(start[labeled & (grids["tok_in_sent"] == 0)])),
        "tir_bin_counts": {str(k): int((tir_bin == k).sum())
                           for k in (0, 1, 2)},
        "manifest_rows_per_class": {
            k: int(len(arrays[f"man_{k}_doc"])
                   // max(1, len(np.unique(arrays[f"man_{k}_cls"]))))
            for k in ("op", "boundary", "tir")},
        "clock_bridge": clock,
    }
    (HERE / "proofops_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"[clock] median {clock['tokens_per_sentence']['median']:.1f} "
          f"tok/sentence -> a window spans 2 sentences only at T >= "
          f"{clock['min_T_spanning_2_sentences_median_clock']}")
    print(f"-> {HERE / 'proofops.npz'}")


if __name__ == "__main__":
    main()
