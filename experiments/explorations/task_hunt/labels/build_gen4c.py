"""Gen-4 CORPUS SCOUT builder (mac-c lane, beat review ~12:15 item
3): label-side pre-measures for the return family transplanted onto
two substrates no hunt face has touched — WikiText-103 encyclopedic
narrative (``tret``/``tretd``/``sage``) and permissively-licensed
Python code (``tret``/``drev``) — CPU-only, feeding mac-a's slate.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_gen4c

Corpora from the committed pullers (``pull_wikitext103`` /
``pull_pycode``, pinned revisions + receipts). Face logic:
hunt3/hunt4 machinery imported verbatim; scout-specific logic in
``gen4c_lib.py`` (tests ``tests/test_gen4c_labels.py``). Triage
(unigram / position / doc-mean AUC + wd viability) and the per-T
VISIBLE-FLOOR evidence lines mirror ``build_hunt4`` exactly.

Artifacts: ``gen4c_<corpus>_<tok>.npz`` for the first-wave pair
(gpt2, gemma2 — hunt4's screen pair; llama31 numbers land in the
stats JSON and its npz is regenerable by re-running this builder,
committed-weight discipline) + ``gen4c_stats.json``. Floors are NOT
stored in the npz (regenerable deterministically from the committed
stream by this builder — stated deviation from hunt4's fp16-floor
bundles, same reason: committed weight).
"""

from __future__ import annotations

import gzip
import io
import json
import keyword
import tokenize as pytokenize
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import hunt3_lib as h3
from . import hunt4_lib as h4
from . import gen4c_lib as g4
from .build_hunt3 import _spearman
from .pull_wikitext103 import RE_H2, is_h1

HERE = Path(__file__).resolve().parent
SEED = 0
MIN_POS = 16
TRIAGE_POS = 64
WD_MIN_DOC_ROWS = 30
MAX_DOC_TOK = 1024     # uniform per-doc token cap (gpt2 context; keeps
                       # every stream row screenable by every model)

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",
}
COMMIT_NPZ = ("gpt2", "gemma2")   # hunt4's first-wave screen pair

CORPORA = {
    "wikitext103": {"artifact": "wikitext103_corpus.json.gz",
                    "faces": ("tret", "tretd", "sage"),
                    "floor_key": {"tret": "floor_rate",
                                  "tretd": "floor_rate",
                                  "sage": "sage_floor"}},
    "pycode": {"artifact": "pycode_corpus.json.gz",
               "faces": ("tret", "drev"),
               "floor_key": {"tret": "floor_rate",
                             "drev": "drev_floor"}},
}


def _header_spans(lines):
    spans, off = [], 0
    for ln in lines:
        if is_h1(ln) or RE_H2.match(ln):
            spans.append((off, off + len(ln)))
        off += len(ln)
    return spans


def _ident_spans(src: str):
    line_off = [0]
    for ln in src.splitlines(keepends=True):
        line_off.append(line_off[-1] + len(ln))
    spans = []
    try:
        for tok in pytokenize.generate_tokens(io.StringIO(src).readline):
            if (tok.type == pytokenize.NAME
                    and not keyword.iskeyword(tok.string)):
                (r1, c1), (r2, c2) = tok.start, tok.end
                if r1 - 1 < len(line_off) and r2 - 1 < len(line_off):
                    spans.append((line_off[r1 - 1] + c1,
                                  line_off[r2 - 1] + c2))
    except (pytokenize.TokenError, IndentationError, SyntaxError):
        pass
    return spans


def _mark(offsets, spans, n):
    out = np.zeros(n, dtype=np.int8)
    if not spans:
        return out
    si = 0
    spans = sorted(spans)
    for i, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        while si < len(spans) and spans[si][1] <= a:
            si += 1
        j = si
        while j < len(spans) and spans[j][0] < b:
            if spans[j][1] > a:
                out[i] = 1
                break
            j += 1
    return out


def _load_docs(corpus: str):
    with gzip.open(HERE / CORPORA[corpus]["artifact"], "rt") as f:
        docs = json.load(f)
    out = []
    for d in docs:
        if corpus == "wikitext103":
            text = "".join(d["lines"])
            out.append((text, _header_spans(d["lines"])))
        else:
            src = d["content"]
            out.append((src, _ident_spans(src)))
    return out


def build_corpus_tokenizer(corpus: str, key: str, model: str):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model)
    docs = _load_docs(corpus)

    ids_l, mark_l, off = [], [], [0]
    for text, spans in docs:
        enc = tk(text, return_offsets_mapping=True,
                 add_special_tokens=False)
        tid = np.asarray(enc["input_ids"], dtype=np.int32)[:MAX_DOC_TOK]
        ids_l.append(tid)
        mark_l.append(_mark(enc["offset_mapping"], spans,
                            len(enc["input_ids"]))[:MAX_DOC_TOK])
        off.append(off[-1] + len(tid))
    ids = np.concatenate(ids_l)
    mark = np.concatenate(mark_l)          # boundary (wt) / ident (py)
    off = np.asarray(off, dtype=np.int64)
    n_docs = len(docs)
    split = lib.doc_split(n_docs, seed=SEED)
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(off)])

    faces_names = CORPORA[corpus]["faces"]
    n = len(ids)
    faces = {f: np.full(n, np.nan, dtype=np.float32) for f in faces_names}
    fl = {}
    floor_names = sorted(set(CORPORA[corpus]["floor_key"].values()))
    for T in h3.FLOOR_TS:
        for name in floor_names:
            fl[f"{name}_T{T}"] = np.full(n, np.nan, dtype=np.float32)
    ev_rate = {k: [] for k in ("ret64", "drev", "boundary", "ident")}

    for d in range(n_docs):
        s, e = off[d], off[d + 1]
        dsl = ids[s:e]
        msl = mark[s:e]
        last_occ = h3.last_occurrence(dsl)
        ret = h4.long_return_events(last_occ)
        faces["tret"][s:e] = h3.filter_rate(ret, h3.SUPPORT_TOK)
        if corpus == "wikitext103":
            faces["tretd"][s:e] = h4.return_depth_face(last_occ)
            faces["sage"][s:e] = g4.sage_face(msl)
        else:
            lom = g4.last_occurrence_masked(dsl, msl)
            dev = g4.masked_return_events(lom, msl)
            faces["drev"][s:e] = h3.filter_rate(dev, h3.SUPPORT_TOK)

        full = np.arange(e - s) >= h3.SUPPORT_TOK
        if full.any():
            ev_rate["ret64"].append(float(ret[full].mean()))
            if corpus == "wikitext103":
                ev_rate["boundary"].append(float(msl[full].mean()))
            else:
                ev_rate["drev"].append(float(dev[full].mean()))
                ev_rate["ident"].append(float(msl[full].mean()))

        for T in h3.FLOOR_TS:
            fl[f"floor_rate_T{T}"][s:e] = h3.floor_rate(last_occ, T)
            if corpus == "wikitext103":
                fl[f"sage_floor_T{T}"][s:e] = g4.sage_floor(msl, T)
            else:
                fl[f"drev_floor_T{T}"][s:e] = g4.drev_floor(lom, msl, T)

    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1
    strata = pl.pos_strata(pos_of, min_pos=MIN_POS)
    boundary = mark if corpus == "wikitext103" else np.zeros(n, np.int8)

    out_npz = {"token_ids": ids, "doc_off": off, "doc_split": split,
               ("is_boundary" if corpus == "wikitext103"
                else "is_ident"): mark,
               **{f: v.astype(np.float16) for f, v in faces.items()}}

    overlaps = {}
    for i, a in enumerate(faces_names):
        for b in faces_names[i + 1:]:
            overlaps[f"{a}~{b}"] = _spearman(faces[a], faces[b])

    stats = {"tokenizer": model, "n_docs": n_docs, "n_tokens": int(n),
             "event_rates_full_support": {
                 k: float(np.mean(v)) for k, v in ev_rate.items() if v},
             "ret64_zero_docs": float(np.mean(
                 np.array(ev_rate["ret64"]) == 0.0)),
             "overlap_spearman": overlaps, "faces": {}}

    for face in faces_names:
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
        dm_score = np.where(cnt > 0, dm / np.maximum(cnt, 1),
                            np.nan)[doc_of]

        n_wd = 0
        for dd in np.unique(doc_of[elig & test_rows]):
            sel = vals[(doc_of == dd) & elig]
            if len(sel) >= WD_MIN_DOC_ROWS:
                q1, q2 = np.quantile(sel, [1 / 3, 2 / 3])
                n_wd += int(q2 > q1)

        fkey = CORPORA[corpus]["floor_key"][face]
        stats["faces"][face] = {
            "scheme": scheme,
            "labeled_frac": float(np.isfinite(vals).mean()),
            "manifest_rows_per_class": int(len(d_) // 3),
            "unigram_auc": nl.tercile_auc(unigram, masked,
                                          test_rows & elig),
            "position_auc": nl.tercile_auc(pos_of.astype(float), masked,
                                           test_rows & elig),
            "doc_mean_only_auc": nl.tercile_auc(dm_score, masked,
                                                test_rows & elig),
            "wd_viable_test_docs": n_wd,
            "visible_floor_auc_by_T": {
                str(T): nl.tercile_auc(
                    fl[f"{fkey}_T{T}"].astype(np.float64), masked,
                    test_rows & elig)
                for T in h3.FLOOR_TS},
        }

    if key in COMMIT_NPZ:
        out = HERE / f"gen4c_{corpus}_{key}.npz"
        np.savez_compressed(out, **out_npz)
        stats["artifact"] = out.name
    else:
        stats["artifact"] = ("NOT COMMITTED (regenerable by this "
                             "builder; committed-weight discipline)")

    print(f"[{corpus}/{key}] " + json.dumps({
        "events": {k: round(v, 4) for k, v in
                   stats["event_rates_full_support"].items()},
        "overlap": {k: round(v, 3) for k, v in overlaps.items()},
        **{f: {k: round(v, 3) for k, v in s.items()
               if k.endswith("auc")}
           for f, s in stats["faces"].items()}}), flush=True)
    return stats


def main():
    stats = {"lane": "gen-4 corpus scout (mac-c, beat review ~12:15 "
                     "item 3; CPU-only, label-side, no Modal spend)",
             "support_tok": h3.SUPPORT_TOK, "cnov_hl": h3.CNOV_HL,
             "ret_gap": h4.RET_GAP, "min_manifest_pos": MIN_POS,
             "triage_pos": TRIAGE_POS, "max_doc_tok": MAX_DOC_TOK,
             "floor_ts": list(h3.FLOOR_TS),
             "committed_npz_tokenizers": list(COMMIT_NPZ),
             "corpora": {}}
    for corpus in CORPORA:
        stats["corpora"][corpus] = {"per_tokenizer": {}}
        for key, model in TOKENIZERS.items():
            stats["corpora"][corpus]["per_tokenizer"][key] = (
                build_corpus_tokenizer(corpus, key, model))
    (HERE / "gen4c_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'gen4c_stats.json'}")


if __name__ == "__main__":
    main()
