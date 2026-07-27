"""Fourth-generation hunt faces (gen-4 directive 59ad15f38): `xnov`
(cross-speaker adoption) + `tret` (topic-return intensity) + `sdom`
(signed speaker novelty-dominance) on the EXISTING dialevel substrate
— same committed token stream, same caches, zero new forward passes.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_hunt4

Everything derives from committed arrays (`dialevel_dailydialog_<tok>
.npz`); face logic in ``hunt4_lib.py`` (kernels/filters = hunt3's
verbatim, imported). Per-face label-side triage (unigram / position /
doc-mean AUC + wd viability) AND the per-T VISIBLE-FLOOR evidence
lines are computed HERE, before any screen. Overlap pre-screen:
|Spearman| vs every confirmed/kept face (ttrend, cnov, nvtrend) and
pairwise among the new faces decides who earns GPU (tempo precedent:
kill bar |rho| > 0.8).

Speaker attribution: DailyDialog is a strictly alternating two-party
corpus (the tempo kill's premise), so speaker = turn_idx % 2.

Artifacts: ``hunt4_dailydialog_<tok>.npz`` + ``hunt4_stats.json``.
Floors are stored float16 (the screen casts floor features to fp16
anyway) to keep the committed bundles at hunt3 weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import hunt3_lib as h3
from . import hunt4_lib as h4
from .build_diafaces import TOKENIZERS
from .build_hunt3 import _spearman

HERE = Path(__file__).resolve().parent
SEED = 0
MIN_POS = 16           # manifest floor (house default; faces NaN < 64 anyway)
TRIAGE_POS = 64        # triage + evidence lines quoted at screen eligibility
WD_MIN_DOC_ROWS = 30

FACES = ("xnov", "tret", "sdom", "xret", "xtrend", "tretd")


def build_for_tokenizer(key: str, model: str):
    z = np.load(HERE / f"dialevel_dailydialog_{key}.npz")
    ids, off = z["token_ids"], z["doc_off"]
    turn_idx, boundary, split = z["turn_idx"], z["is_boundary"], z["doc_split"]
    n_docs = len(off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(off)])
    assert np.array_equal(split, lib.doc_split(n_docs, seed=SEED))

    n = len(ids)
    faces = {f: np.full(n, np.nan, dtype=np.float32) for f in FACES}
    fl = {}
    for T in h3.FLOOR_TS:
        for name in ("floor_rate", "xfloor_rate", "xfloor_slope",
                     "sdom_floor", "sdom_kcur", "sdom_koth"):
            fl[f"{name}_T{T}"] = np.full(n, np.nan, dtype=np.float32)
    ev_rate = {"adopt": [], "ret64": [], "xret": []}
    sdom_nan = []

    for d in range(n_docs):
        s, e = off[d], off[d + 1]
        dsl = ids[s:e]
        t_idx = turn_idx[s:e]
        spk = (t_idx % 2).astype(np.int8)

        last_occ = h3.last_occurrence(dsl)
        same, oth = h4.last_occurrence_by_speaker(dsl, spk)
        assert np.array_equal(np.maximum(same, oth), last_occ)
        novel = (last_occ < 0).astype(np.int8)
        adopt = h4.adoption_events(same, oth)
        ret = h4.long_return_events(last_occ)
        xret = h4.cross_return_events(same, oth)

        faces["xnov"][s:e] = h3.filter_rate(adopt, h3.SUPPORT_TOK)
        faces["tret"][s:e] = h3.filter_rate(ret, h3.SUPPORT_TOK)
        faces["sdom"][s:e] = h4.sdom_face(novel, spk)
        faces["xret"][s:e] = h3.filter_rate(xret, h3.SUPPORT_TOK)
        ta = h3.turn_novelty_rates(adopt, t_idx)
        faces["xtrend"][s:e] = h3.trailing_turn_slope(ta)[t_idx]
        faces["tretd"][s:e] = h4.return_depth_face(last_occ)

        full = np.arange(e - s) >= h3.SUPPORT_TOK
        if full.any():
            ev_rate["adopt"].append(float(adopt[full].mean()))
            ev_rate["ret64"].append(float(ret[full].mean()))
            ev_rate["xret"].append(float(xret[full].mean()))
            sdom_nan.append(float(np.isnan(faces["sdom"][s:e][full]).mean()))

        for T in h3.FLOOR_TS:
            fl[f"floor_rate_T{T}"][s:e] = h3.floor_rate(last_occ, T)
            xev = h4.xnov_floor_events(same, oth, T)
            fl[f"xfloor_rate_T{T}"][s:e] = h3.filter_rate(
                xev, min(T, h3.SUPPORT_TOK))
            fl[f"xfloor_slope_T{T}"][s:e] = h3.filter_slope(
                xev, min(T, h3.SUPPORT_TOK))
            dfl, kc, ko = h4.sdom_floor(last_occ, spk, T)
            fl[f"sdom_floor_T{T}"][s:e] = dfl
            fl[f"sdom_kcur_T{T}"][s:e] = kc
            fl[f"sdom_koth_T{T}"][s:e] = ko

    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1
    strata = pl.pos_strata(pos_of, min_pos=MIN_POS)

    ttrend = np.load(HERE / f"diafaces_dailydialog_{key}.npz")["ttrend"]
    z3 = np.load(HERE / f"hunt3_dailydialog_{key}.npz")
    cnov, nvtrend = z3["cnov"], z3["nvtrend"]

    out_npz = {**{f: v for f, v in faces.items()}, "doc_split": split,
               **{k: v.astype(np.float16) for k, v in fl.items()}}
    overlaps = {}
    for f in FACES:
        overlaps[f] = {
            "ttrend": _spearman(faces[f], ttrend),
            "cnov": _spearman(faces[f], cnov),
            "nvtrend": _spearman(faces[f], nvtrend)}
    for i, a in enumerate(FACES):
        for b in FACES[i + 1:]:
            overlaps[f"{a}~{b}"] = _spearman(faces[a], faces[b])

    stats = {"tokenizer": model, "n_dialogues": n_docs,
             "n_tokens": int(ids.size),
             "event_rates_full_support": {
                 "adopt_mean": float(np.mean(ev_rate["adopt"])),
                 "ret64_mean": float(np.mean(ev_rate["ret64"])),
                 "xret_mean": float(np.mean(ev_rate["xret"])),
                 "ret64_zero_docs": float(np.mean(
                     np.array(ev_rate["ret64"]) == 0.0)),
                 "sdom_nan_frac": float(np.mean(sdom_nan))},
             "overlap_spearman": overlaps, "faces": {}}

    for face in FACES:
        vals = faces[face]
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

        floor_key = {"xnov": "xfloor_rate", "tret": "floor_rate",
                     "sdom": "sdom_floor", "xret": "floor_rate",
                     "xtrend": "xfloor_slope", "tretd": "floor_rate"}[face]
        stats["faces"][face] = {
            "scheme": scheme,
            "labeled_frac": float(np.isfinite(vals).mean()),
            "manifest_rows_per_class": int(len(d_) // 3),
            "unigram_auc": nl.tercile_auc(unigram, masked, test_rows & elig),
            "position_auc": nl.tercile_auc(pos_of.astype(float), masked,
                                           test_rows & elig),
            "doc_mean_only_auc": nl.tercile_auc(dm_score, masked,
                                                test_rows & elig),
            "wd_viable_test_docs": n_wd,
            "visible_floor_auc_by_T": {
                str(T): nl.tercile_auc(
                    fl[f"{floor_key}_T{T}"].astype(np.float64), masked,
                    test_rows & elig)
                for T in h3.FLOOR_TS},
        }

    out = HERE / f"hunt4_dailydialog_{key}.npz"
    np.savez_compressed(out, **out_npz)
    stats["artifact"] = out.name
    print(f"[{key}] " + json.dumps({
        "events": {k: round(v, 4) for k, v in
                   stats["event_rates_full_support"].items()},
        "overlap": {k: (round(v, 3) if isinstance(v, float)
                        else {kk: round(vv, 3) for kk, vv in v.items()})
                    for k, v in overlaps.items()},
        **{f: {k: round(v, 3) for k, v in s.items() if k.endswith("auc")}
           for f, s in stats["faces"].items()}}))
    return stats


def main():
    stats = {"substrate": "dialevel_dailydialog (REUSED stream + caches; "
                          "builders labels/build_dialevel.py + "
                          "dialevel/cache_acts.py)",
             "directive": "gen-4 (59ad15f38); speaker = turn_idx % 2 "
                          "(strict alternation, tempo-kill premise)",
             "support_tok": h3.SUPPORT_TOK, "cnov_hl": h3.CNOV_HL,
             "ret_gap": h4.RET_GAP, "sdom_min_mass": h4.SDOM_MIN_MASS,
             "min_manifest_pos": MIN_POS, "triage_pos": TRIAGE_POS,
             "floor_ts": list(h3.FLOOR_TS),
             "per_tokenizer": {}}
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(key, model)
    (HERE / "hunt4_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'hunt4_stats.json'}")


if __name__ == "__main__":
    main()
