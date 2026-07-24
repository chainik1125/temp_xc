"""Confidence-trend targets on the Ward stream (candidate 4).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_confidence

Broadcasts the frozen per-sentence confidence labels
(``synthetic/expansion/records/uncertainty-hedging-drift/labels.json``:
0 hedged / 1 neutral / 2 committed, judged per sentence) onto the Ward
cache grid, plus the DRIFT targets the candidate actually needs:

- ``hedge``     int8   — the sentence's confidence state (lexically
  stamped ⇒ the regime-1 control target);
- ``slope4``/``slope8`` float32 — least-squares slope of the state over
  the trailing 4/8 SENTENCES (NaN if the window has any unlabeled
  sentence or doesn't fit) — the hedging→commitment trend;
- ``sent_frac`` float32 — sentence position i/L in the trace (the
  DC-drift covariate);
- ``sent_idx``, ``valid``, ``trace_idx``/``win_start``, ``trace_split``;
- manifests: ``man_state_*`` (3-class balanced anchor) and
  ``man_slope_*`` (terciles of slope8, edges in stats) — valid, p ≥ 32.

Clock caveat (why this is the riskiest candidate): the slope is defined
over trailing SENTENCES; a token window T sees only T/median-tokens-per-
sentence of them (numbers in ``proofops_stats.json``'s clock bridge, same
tokenizer/grid). Screening below T ≈ 2 sentences is meaningless — the
card mandates T ∈ {16, 32, 64} with "timescale unreachable" as a valid
honest kill.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib, wardmap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABELS = (ROOT / "experiments/explorations/synthetic/expansion/records/"
          "uncertainty-hedging-drift/labels.json")
SEED = 0


def main() -> None:
    tok, traces, by_qid = wardmap.load_inputs()
    rec = json.loads(LABELS.read_text())
    hedge_by_qid = dict(zip(rec["doc_ids"], rec["labels"]))

    def payload(ti, ids, offs):
        slab = by_qid[traces[ti]["question_id"]]
        spans = [(s["char_start"], s["char_end"]) for s in slab["sentences"]]
        labels = hedge_by_qid.get(traces[ti]["question_id"]) \
            or [None] * len(spans)
        h = np.array([-1 if v is None else int(v) for v in labels],
                     dtype=np.int8)
        s4 = lib.trailing_slope(labels, 4)
        s8 = lib.trailing_slope(labels, 8)
        L = max(len(spans), 1)
        frac = (np.arange(len(spans)) / L).astype(np.float32)
        sidx, in_span = lib.sentence_index_per_token(offs, spans)
        return {"hedge": h[sidx], "slope4": s4[sidx], "slope8": s8[sidx],
                "sent_frac": frac[sidx], "sent_idx": sidx.astype(np.int16),
                "in_span": in_span.astype(np.int8)}

    grids, valid_rt, trace_idx, win_start, n_mm = wardmap.broadcast(
        tok, traces, payload)
    valid = valid_rt & (grids["in_span"] == 1)
    hedge, s8 = grids["hedge"], grids["slope8"]

    edges, slope_bin = lib.tercile_bins(np.where(valid, s8, np.nan))
    slope_bin[~valid] = -1

    N, S = hedge.shape
    w_of, p_of = np.meshgrid(np.arange(N), np.arange(S), indexing="ij")
    arrays = {
        "hedge": hedge, "slope4": grids["slope4"], "slope8": s8,
        "slope8_bin": slope_bin, "sent_frac": grids["sent_frac"],
        "sent_idx": grids["sent_idx"], "valid": valid,
        "trace_idx": trace_idx, "win_start": win_start,
        "trace_split": lib.doc_split(int(trace_idx.max()) + 1, seed=SEED),
    }
    for name, cls in (("state", np.where(valid, hedge, -1)),
                      ("slope", slope_bin)):
        d, p, c = lib.balanced_manifest(cls.ravel(), w_of.ravel(),
                                        p_of.ravel(), seed=SEED)
        arrays[f"man_{name}_doc"], arrays[f"man_{name}_pos"] = d, p
        arrays[f"man_{name}_cls"] = c
    np.savez_compressed(HERE / "confidence.npz", **arrays)

    stats = {
        "labels_source": str(LABELS.relative_to(ROOT)),
        "coverage": rec["coverage"], "seed": SEED,
        "valid_rate_pos1plus": float(valid[:, 1:].mean()),
        "n_roundtrip_mismatch_tokens": int(n_mm),
        "state_counts": {str(k): int((hedge[valid] == k).sum())
                         for k in (-1, 0, 1, 2)},
        "slope8": {"valid_rate": float(np.isfinite(s8[valid]).mean()),
                   "mean": float(np.nanmean(s8[valid])),
                   "tercile_edges": [float(e) for e in edges]},
        # the drift the candidate posits: state should rise with position
        "state_mean_by_third": [
            float(hedge[valid & (grids["sent_frac"] < 1 / 3)].mean()),
            float(hedge[valid & (grids["sent_frac"] >= 1 / 3)
                        & (grids["sent_frac"] < 2 / 3)].mean()),
            float(hedge[valid & (grids["sent_frac"] >= 2 / 3)].mean())],
        "manifest_rows_per_class": {
            k: int(len(arrays[f"man_{k}_doc"])
                   // max(1, len(np.unique(arrays[f"man_{k}_cls"]))))
            for k in ("state", "slope")},
    }
    (HERE / "confidence_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'confidence.npz'}")


if __name__ == "__main__":
    main()
