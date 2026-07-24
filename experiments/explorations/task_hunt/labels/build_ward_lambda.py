"""Backtracking-intensity λ̂ targets on the Ward stream (candidate 1).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_ward_lambda

Applies the COMMITTED backtracking mirror
(``synthetic/backtracking/results/backtracking_mirror_stats.json``:
logistic intensity, position term + K=8 self-excitation kernel over the
Sonnet sentence labels) causally per sentence, then broadcasts sentence
λ̂ to tokens and maps into the canonical Ward cache coordinates (4044 ×
128) via ``wardmap`` (verbatim ``conversion_depth.build_ward_stream``
machinery) — so the arrays align position-for-position with any cache
built from the canonical stream. λ̂_i uses sentences i-1..i-8 ONLY
(never the current sentence's own label): recovering it means reading
recent event history, not detecting the current sentence.

Arrays in ``ward_lambda.npz`` (grids are (4044, 128); -1/NaN undefined):
- ``lam``       float32 — full intensity (position trend + history);
- ``lam_hist``  float32 — history-only variant (position term dropped;
  separates self-excitation recovery from trace-trend recovery);
- ``lam_bin``   int8    — terciles of ``lam`` over valid positions
  (edges in the stats JSON) — the classification variant;
- ``is_bt``     int8    — current sentence's own event label (control:
  detecting the current event ≠ recovering the intensity);
- ``sent_idx``  int16, ``in_span``/``valid`` bool (round-trip identity
  AND sentence-span check), ``trace_idx``/``win_start`` (N,);
- ``man_doc/pos/cls`` — tercile-balanced probe rows (valid, p ≥ 32);
  doc = WINDOW index; group splits by ``trace_idx`` via ``trace_split``
  (per-trace 0 train / 1 test, seed 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib, wardmap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MIRROR = (ROOT / "experiments/explorations/synthetic/backtracking/results/"
          "backtracking_mirror_stats.json")
WARD_STATS = (ROOT / "experiments/explorations/conversion_depth/results/"
              "ward_stream_stats.json")
SEED = 0


def main() -> None:
    tok, traces, by_qid = wardmap.load_inputs()
    mirror = json.loads(MIRROR.read_text())
    icpt, cpos, kw = (mirror["intercept"], mirror["coef_position"],
                      mirror["kernel_w"])

    def payload(ti, ids, offs):
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        b = [1 if s["is_backtracking"] else 0 for s in slab["sentences"]]
        lam_s, lam_hist_s = lib.lambda_for_sentences(b, icpt, cpos, kw)
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        return {"lam": lam_s[sidx], "lam_hist": lam_hist_s[sidx],
                "is_bt": np.asarray(b, dtype=np.int8)[sidx],
                "sent_idx": sidx.astype(np.int16),
                "in_span": in_span.astype(np.int8)}

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    in_span = grids["in_span"] == 1
    valid = valid_rt & in_span
    lam, lam_hist, is_bt = grids["lam"], grids["lam_hist"], grids["is_bt"]
    lam[~valid] = np.nan
    lam_hist[~valid] = np.nan

    edges, lam_bin = lib.tercile_bins(np.where(valid, lam, np.nan))
    lam_bin[~valid] = -1

    N, S = lam.shape
    w_of, p_of = np.meshgrid(np.arange(N), np.arange(S), indexing="ij")
    man_doc, man_pos, man_cls = lib.balanced_manifest(
        lam_bin.ravel(), w_of.ravel(), p_of.ravel(), seed=SEED)
    tr_split = lib.doc_split(int(trace_idx.max()) + 1, seed=SEED)

    np.savez_compressed(
        HERE / "ward_lambda.npz",
        lam=lam, lam_hist=lam_hist, lam_bin=lam_bin, is_bt=is_bt,
        sent_idx=grids["sent_idx"], in_span=in_span, valid=valid,
        trace_idx=trace_idx, win_start=win_start,
        man_doc=man_doc, man_pos=man_pos, man_cls=man_cls,
        trace_split=tr_split)

    # sanity: λ̂ tercile must be monotone in the current event rate
    rate_by_bin = [float(np.mean(is_bt[valid & (lam_bin == k)]))
                   for k in (0, 1, 2)]
    stats = {
        "mirror_params": {"intercept": icpt, "coef_position": cpos,
                          "kernel_w": kw,
                          "source": str(MIRROR.relative_to(ROOT))},
        "stream_shape": [int(N), int(S)], "tokenizer": wardmap.BASE_MODEL,
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "roundtrip_valid_rate_pos1plus": float(valid_rt[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "lam": {"mean": float(np.nanmean(lam)),
                "tercile_edges": [float(e) for e in edges]},
        "is_bt_rate_by_lam_tercile": rate_by_bin,
        "is_bt_rate_valid": float(np.mean(is_bt[valid])),
        "manifest_rows_per_class": int(len(man_doc) // 3),
        "trace_split_test_frac": float(tr_split.mean()),
        "seed": SEED,
    }
    if WARD_STATS.exists():
        ref = json.loads(WARD_STATS.read_text())
        stats["ref_map_ok_rate_committed"] = ref["map_ok_rate_pos1plus"]
    assert rate_by_bin[0] < rate_by_bin[2], (
        "λ̂ terciles not monotone in event rate — mapping broken")
    (HERE / "ward_lambda_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'ward_lambda.npz'}")


if __name__ == "__main__":
    main()
