"""Quoted-speech intensity labels on the pinned PG19 fiction corpus
(CANDIDATES.md B9 `quotedens`; card `../quotedens/CARD_DRAFT.md`).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_quotedens

Machinery follows ``build_punctint4k.py`` unchanged where it applies
(same sentence→token bridge, position-matched manifests, doc-level
1,000-rep bootstrap CIs on every triage AUC, ``doc_mean_only_auc``
disclosure + within-document-contrast census, event-sentence token
masking). Frozen event grammar + kernel live in ``quotedens_lib.py``
(the kernel IS punctint's, re-exported). New-corpus specifics: no
replag prefix receipt exists — instead the stats carry the pinned
corpus receipt fields and the CACHING COST (every token is new for
all three models), plus the corpus-composition disclosures the
single-quote exclusion obliges: per-book event-rate distribution and
the zero-event book fraction.

Artifacts: ``quotedens_pg19_<tok>.npz`` + ``quotedens_stats.json``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from . import boot_lib as bo
from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import quotedens_lib as ql
from .build_punctint4k import supported_rows_per_class, within_doc_contrast

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "pg19_corpus.json.gz"
SEED = 0
MANIFEST_CAP = 100_000
FACE = "qd"

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",
}


def load_docs():
    sample = json.loads(gzip.decompress(CORPUS.read_bytes()))
    return [d["sentences"] for d in sample["docs"]], sample["meta"]


def build_for_tokenizer(key, model, sent_lists, events, n_reps, out_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    t0 = time.time()

    id_docs, off, sent_idx_all, in_span_all = [], [0], [], []
    lam_docs, evt_docs = [], []
    for d, sents in enumerate(sent_lists):
        text = " ".join(sents)
        spans, pos = [], 0
        for s in sents:
            spans.append((pos, pos + len(s)))
            pos += len(s) + 1
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        ids = np.asarray(enc["input_ids"], dtype=np.int32)
        s_idx, in_sp = lib.sentence_index_per_token(enc["offset_mapping"],
                                                    spans)
        id_docs.append(ids)
        off.append(off[-1] + ids.size)
        sent_idx_all.append(s_idx)
        in_span_all.append(in_sp.astype(np.int8))
        s_lam = ql.sentence_lambda(events[d])
        lam_docs.append(pl.token_labels_from_sentences(
            s_lam, s_idx).astype(np.float32))
        evt_docs.append(events[d][s_idx])

    ids_flat = np.concatenate(id_docs)
    doc_off = np.array(off, dtype=np.int64)
    n_docs = len(doc_off) - 1
    print(f"[{key}] tokenized {n_docs} books -> {ids_flat.size:,} tokens "
          f"in {time.time() - t0:.0f}s (ALL new for caching)", flush=True)

    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = lib.doc_split(n_docs, seed=SEED)
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1

    v = np.concatenate(lam_docs)
    evt = np.concatenate(evt_docs).astype(np.int8)
    scheme, edges, bins = pl.zero_split_bins(v, train_rows)
    masked_bins = np.where(evt == 1, -1, bins).astype(np.int8)
    strata = pl.pos_strata(pos_of, min_pos=lib.MIN_MANIFEST_POS)
    d_, p_, c_ = pl.stratified_balanced_manifest(
        masked_bins, strata, doc_of, pos_of, cap=MANIFEST_CAP, seed=SEED)

    arrays = {"token_ids": ids_flat, "doc_off": doc_off,
              "doc_split": split,
              "sent_idx": np.concatenate(sent_idx_all),
              "in_span": np.concatenate(in_span_all),
              f"lam_{FACE}": v, f"{FACE}_bin": bins, f"is_{FACE}": evt,
              f"man_{FACE}_doc": d_, f"man_{FACE}_pos": p_,
              f"man_{FACE}_cls": c_}

    elig = (masked_bins >= 0) & (pos_of >= lib.MIN_MANIFEST_POS)
    unigram = nl.type_mean_scores(ids_flat, v, train_rows & elig)
    docmean = np.full(n_docs, np.nan)
    for d in range(n_docs):
        seg = v[doc_off[d]: doc_off[d + 1]]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            docmean[d] = seg.mean()
    docmean_row = docmean[doc_of]

    man_rows = np.zeros(len(pos_of), dtype=bool)
    man_rows[doc_off[:-1][d_] + p_] = True
    row_sets = {"triage_all_eligible_rows": test_rows & elig,
                "triage_manifest_rows": man_rows & test_rows}
    scores = {"unigram_auc": unigram,
              "position_auc": pos_of.astype(float),
              "doc_mean_only_auc": docmean_row}
    tri, boot = {}, {}
    for rname, rmask in row_sets.items():
        tri[rname] = {s: nl.tercile_auc(sc, masked_bins, rmask)
                      for s, sc in scores.items()}
        boot[rname] = {}
        for s, sc in scores.items():
            boot[rname][s] = bo.bootstrap_tercile_auc(
                sc, masked_bins, rmask, doc_of, n_reps=n_reps, seed=SEED)
        print(f"[{key}/{FACE}] {rname}: "
              + " ".join(f"{s.split('_auc')[0]}="
                         f"{boot[rname][s]['point']:.4f}"
                         f"[{boot[rname][s]['ci_lo']:.3f},"
                         f"{boot[rname][s]['ci_hi']:.3f}]"
                         for s in scores), flush=True)

    fin = np.isfinite(v)
    out = out_dir / f"quotedens_pg19_{key}.npz"
    np.savez_compressed(out, **arrays)
    print(f"[{key}] wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - t0:.0f}s total", flush=True)
    return {"tokenizer": model, "n_docs": n_docs,
            "n_tokens": int(ids_flat.size),
            "new_tokens_needing_cache": int(ids_flat.size),
            "face": {
                "scheme": scheme, "edges": [float(e) for e in edges],
                "lam_mean": float(v[fin].mean()),
                "lam_std": float(v[fin].std()),
                "lam_zero_frac": float((v[fin] == 0).mean()),
                "masked_token_frac": float(evt.mean()),
                "eligible_frac": float(elig.mean()),
                "unigram_train_docs": int((split == 0).sum()),
                "manifest_cap_per_class": MANIFEST_CAP,
                "manifest_rows_per_class": int(len(d_) // 3),
                "manifest_rows_per_class_supported":
                    supported_rows_per_class(masked_bins, strata),
                "within_document_contrast": within_doc_contrast(
                    d_, c_, split),
                "triage_all_eligible_rows": tri["triage_all_eligible_rows"],
                "triage_manifest_rows": tri["triage_manifest_rows"],
                "bootstrap": boot,
            },
            "artifact": out.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=bo.N_REPS)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)

    sent_lists, meta = load_docs()
    events = [np.array([ql.is_quote_sentence(s) for s in sents],
                       dtype=np.int8) for sents in sent_lists]

    # corpus-composition disclosures obliged by the single-quote
    # exclusion: per-book event-rate distribution + zero-event books
    rates = np.array([e.mean() for e in events])
    stats = {
        "source": CORPUS.name, "source_meta": meta, "seed": SEED,
        "frozen_logic": "quotedens_lib (committed before this run) — "
                        "double-quote family events, punctint kernel "
                        "re-exported, event-sentence masking",
        "quote_chars": ql.QUOTE_CHARS,
        "event_sentence_rate": float(np.concatenate(events).mean()),
        "per_book_event_rate": {
            "mean": float(rates.mean()),
            "quartiles": [float(q) for q in
                          np.quantile(rates, [0.25, 0.5, 0.75])],
            "zero_event_book_frac": float((rates == 0).mean()),
            "below_2pct_book_frac": float((rates < 0.02).mean())},
        "kernel": {"half_life_sentences": pl.HALF_LIFE_S,
                   "support_sentences": pl.SUPPORT_S,
                   "mass_within_sentences": {
                       n: pl.kernel_mass_within_sentences(n)
                       for n in (1, 2, 4, 8)}},
        "min_manifest_pos": lib.MIN_MANIFEST_POS,
        "manifest_cap_per_class": MANIFEST_CAP,
        "bootstrap": {"unit": "book (cluster)", "n_reps": a.reps,
                      "ci_pct": list(bo.CI_PCT), "seed": SEED},
        "per_tokenizer": {},
    }
    print(f"[corpus] event sentence rate {stats['event_sentence_rate']:.3f}; "
          f"per-book median {stats['per_book_event_rate']['quartiles'][1]:.3f}, "
          f"zero-event books {stats['per_book_event_rate']['zero_event_book_frac']:.3f}",
          flush=True)
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, sent_lists, events, a.reps, out_dir)
    p = out_dir / "quotedens_stats.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
