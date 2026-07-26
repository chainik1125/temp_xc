"""Third-generation hunt faces (overnight § 1): `cnov` + `nvtrend` +
`tempo` (+ `qres` pre-measure) on the EXISTING dialevel substrate —
same committed token stream, same caches, zero new forward passes.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_hunt3

Everything derives from committed arrays (`dialevel_dailydialog_<tok>
.npz`); face logic in ``hunt3_lib.py``. Per-face label-side triage
(unigram / position / doc-mean AUC + wd viability) AND the per-T
VISIBLE-FLOOR evidence lines (first-in-WINDOW novelty rate/slope vs
the face terciles) are computed HERE, before any screen — the
briefing's "evidence line pre-measured per candidate" requirement.
Overlap pre-screen: |Spearman| of tempo (and nvtrend) vs the ttrend
face decides whether tempo earns GPU at all.

Artifacts: ``hunt3_dailydialog_<tok>.npz`` + ``hunt3_stats.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import hunt3_lib as h3
from .build_diafaces import TOKENIZERS, vocab_q_mask

HERE = Path(__file__).resolve().parent
SEED = 0
MIN_POS = 16           # manifest floor (house default; faces NaN < 64 anyway)
TRIAGE_POS = 64        # triage + evidence lines quoted at screen eligibility
WD_MIN_DOC_ROWS = 30


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def build_for_tokenizer(key: str, model: str):
    z = np.load(HERE / f"dialevel_dailydialog_{key}.npz")
    ids, off = z["token_ids"], z["doc_off"]
    turn_idx, boundary, split = z["turn_idx"], z["is_boundary"], z["doc_split"]
    n_docs = len(off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(off)])
    assert np.array_equal(split, lib.doc_split(n_docs, seed=SEED))
    is_qtok = vocab_q_mask(model)[ids]

    n = len(ids)
    cnov = np.full(n, np.nan, dtype=np.float32)
    nvtrend = np.full(n, np.nan, dtype=np.float32)
    tempo = np.full(n, np.nan, dtype=np.float32)
    qres_tok = np.full(n, np.nan, dtype=np.float32)
    floors_rate = {T: np.full(n, np.nan, dtype=np.float32)
                   for T in h3.FLOOR_TS}
    floors_slope = {T: np.full(n, np.nan, dtype=np.float32)
                    for T in h3.FLOOR_TS}
    q_lat_hist: list[int] = []
    q_turn_rate = []

    for d in range(n_docs):
        s, e = off[d], off[d + 1]
        dsl = ids[s:e]
        t_idx = turn_idx[s:e]
        n_turns = int(t_idx.max()) + 1
        sizes = np.bincount(t_idx, minlength=n_turns)

        last_occ = h3.last_occurrence(dsl)
        novel = (last_occ < 0).astype(np.int8)
        cnov[s:e] = h3.filter_rate(novel, h3.SUPPORT_TOK)
        for T in h3.FLOOR_TS:
            floors_rate[T][s:e] = h3.floor_rate(last_occ, T)
            floors_slope[T][s:e] = h3.floor_slope(last_occ, T)

        tn = h3.turn_novelty_rates(novel, t_idx)
        nvtrend[s:e] = h3.trailing_turn_slope(tn)[t_idx]
        tempo[s:e] = h3.trailing_turn_slope(1.0 / np.maximum(sizes, 1))[t_idx]

        has_q = np.zeros(n_turns, dtype=bool)
        np.logical_or.at(has_q, t_idx, is_qtok[s:e])
        qres_tok[s:e] = h3.qres_latency(has_q)[t_idx]
        q_turn_rate.append(float(has_q.mean()))
        i = 0
        while i < n_turns:                       # per-question latency dist
            if has_q[i]:
                j = i + 1
                while j < n_turns and has_q[j]:
                    j += 1
                if j < n_turns:
                    q_lat_hist.append(j - i)
                i = j
            else:
                i += 1

    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1
    strata = pl.pos_strata(pos_of, min_pos=MIN_POS)

    ttrend = np.load(HERE / f"diafaces_dailydialog_{key}.npz")["ttrend"]

    out_npz = {"cnov": cnov, "nvtrend": nvtrend, "tempo": tempo,
               "qres": qres_tok, "doc_split": split,
               **{f"floor_rate_T{T}": floors_rate[T] for T in h3.FLOOR_TS},
               **{f"floor_slope_T{T}": floors_slope[T] for T in h3.FLOOR_TS}}
    stats = {"tokenizer": model, "n_dialogues": n_docs,
             "n_tokens": int(ids.size),
             "overlap_spearman_vs_ttrend": {
                 "tempo": _spearman(tempo, ttrend),
                 "nvtrend": _spearman(nvtrend, ttrend),
                 "cnov": _spearman(cnov, ttrend)},
             "qres_premeasure": {
                 "n_questions_resolved": len(q_lat_hist),
                 "q_rate_per_turn": float(np.mean(q_turn_rate)),
                 "p_latency_1": float(np.mean(np.array(q_lat_hist) == 1))
                 if q_lat_hist else float("nan"),
                 "p_latency_2": float(np.mean(np.array(q_lat_hist) == 2))
                 if q_lat_hist else float("nan"),
                 "anchor_note": "anchor turn carries a visible '?' token "
                                "(dq's marker one step removed)"},
             "faces": {}}

    for face, vals in (("cnov", cnov), ("nvtrend", nvtrend),
                       ("tempo", tempo)):
        scheme, edges, bins = pl.zero_split_bins(vals, train_rows)
        masked = np.where(boundary == 1, -1, bins).astype(np.int8)
        d_, p_, c_ = pl.stratified_balanced_manifest(
            masked, strata, doc_of, pos_of, seed=SEED)
        out_npz[f"{face}_bin"] = masked
        out_npz[f"man_{face}_doc"] = d_
        out_npz[f"man_{face}_pos"] = p_
        out_npz[f"man_{face}_cls"] = c_

        elig = (masked >= 0) & (pos_of >= TRIAGE_POS)
        unigram = nl.type_mean_scores(ids, vals, train_rows & elig)
        dm = np.zeros(n_docs)
        cnt = np.zeros(n_docs)
        fin = np.isfinite(vals) & elig
        np.add.at(dm, doc_of[fin], vals[fin])
        np.add.at(cnt, doc_of[fin], 1)
        dm_score = np.where(cnt > 0, dm / np.maximum(cnt, 1), np.nan)[doc_of]

        n_wd = 0
        for dd in np.unique(doc_of[elig & test_rows]):
            sel = vals[(doc_of == dd) & elig]
            if len(sel) >= WD_MIN_DOC_ROWS:
                q1, q2 = np.quantile(sel, [1 / 3, 2 / 3])
                n_wd += int(q2 > q1)

        fs = {
            "scheme": scheme,
            "labeled_frac": float(np.isfinite(vals).mean()),
            "manifest_rows_per_class": int(len(d_) // 3),
            "unigram_auc": nl.tercile_auc(unigram, masked, test_rows & elig),
            "position_auc": nl.tercile_auc(pos_of.astype(float), masked,
                                           test_rows & elig),
            "doc_mean_only_auc": nl.tercile_auc(dm_score, masked,
                                                test_rows & elig),
            "wd_viable_test_docs": n_wd,
        }
        floor_of = floors_rate if face == "cnov" else floors_slope
        if face in ("cnov", "nvtrend"):
            fs["visible_floor_auc_by_T"] = {
                str(T): nl.tercile_auc(floor_of[T], masked, test_rows & elig)
                for T in h3.FLOOR_TS}
        stats["faces"][face] = fs

    out = HERE / f"hunt3_dailydialog_{key}.npz"
    np.savez_compressed(out, **out_npz)
    stats["artifact"] = out.name
    print(f"[{key}] " + json.dumps({
        "overlap": {k: round(v, 3) for k, v in
                    stats["overlap_spearman_vs_ttrend"].items()},
        "qres_p1": round(stats["qres_premeasure"]["p_latency_1"], 3),
        **{f: {k: round(v, 3) for k, v in s.items() if k.endswith("auc")}
           for f, s in stats["faces"].items()}}))
    return stats


def main():
    stats = {"substrate": "dialevel_dailydialog (REUSED stream + caches; "
                          "builders labels/build_dialevel.py + "
                          "dialevel/cache_acts.py)",
             "support_tok": h3.SUPPORT_TOK, "cnov_hl": h3.CNOV_HL,
             "support_turns": h3.SUPPORT_TURNS,
             "min_manifest_pos": MIN_POS, "triage_pos": TRIAGE_POS,
             "floor_ts": list(h3.FLOOR_TS),
             "per_tokenizer": {}}
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(key, model)
    (HERE / "hunt3_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'hunt3_stats.json'}")


if __name__ == "__main__":
    main()
