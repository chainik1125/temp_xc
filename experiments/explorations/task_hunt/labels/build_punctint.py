"""Sentence-event intensity labels on fineweb (CANDIDATES.md B3
list/enumeration density + B4 question rate) — exact, zero-API,
CPU-only.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_punctint

Same corpus + tokenization + alignment contract as ``build_replag.py``
(pinned fineweb sample, single-space sentence join,
``add_special_tokens=False``) — ``token_ids`` ASSERTED byte-identical
to the committed ``replag_fineweb_<tok>.npz``, so the existing GPU
caches drop on with zero new caching. Sentence spans use the committed
char-midpoint bridge (``lib.sentence_index_per_token``), the same one
``build_interleave.py`` ships.

Per-token arrays (positions index the no-special-tokens sequence):

- ``lam_list`` / ``lam_q``  float32 — kernel intensity of the event
  stream over the PREVIOUS 8 sentences (half-life 2 — the winner
  family's 8-lag shape; current sentence NEVER in its own label);
  NaN while sentence idx < 8;
- ``list_bin`` / ``q_bin``  int8 — 3-class labels via the frozen
  conditional scheme (zero_split when the train zero fraction
  exceeds 1/3, else terciles; scheme recorded in stats);
- ``is_list`` / ``is_q``    int8 — CURRENT-sentence event flags (the
  ambient/anchor faces; also the masking rule: manifests EXCLUDE
  tokens of event sentences for their own face);
- ``sent_idx`` int32, ``in_span`` int8 — the token→sentence bridge;
- ``doc_split`` int8 per doc (20 % test, seed 0);
- ``man_list_*`` / ``man_q_*`` — balanced (doc,pos,cls) manifests,
  cap 20k/class, pos >= 32, event-sentence tokens masked out.

Label-side triage (kill authority; bars frozen in the two card drafts
BEFORE this ran): current-token type-mean AUC and position AUC
(direction-agnostic), top vs bottom class, test-doc rows, masked.
Artifacts: ``punctint_fineweb_<tok>.npz`` + ``punctint_stats.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FINEWEB = (ROOT / "experiments/explorations/synthetic/expansion/data/"
           "fineweb_sample.json")
SEED = 0

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}

FACES = ("list", "q")


def load_docs():
    sample = json.loads(FINEWEB.read_text())
    return [d["sentences"] for d in sample["docs"]], sample["meta"]


def doc_events(sent_lists):
    ev = {"list": [], "q": []}
    for sents in sent_lists:
        ev["list"].append(np.array([pl.is_list_sentence(s) for s in sents],
                                   dtype=np.int8))
        ev["q"].append(np.array([pl.is_question_sentence(s) for s in sents],
                                dtype=np.int8))
    return ev


def build_for_tokenizer(key, model, sent_lists, events):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    ids_flat, off = [], [0]
    sent_idx_all, in_span_all = [], []
    lam = {f: [] for f in FACES}
    is_evt = {f: [] for f in FACES}
    for d, sents in enumerate(sent_lists):
        text = " ".join(sents)
        spans, pos = [], 0
        for s in sents:
            spans.append((pos, pos + len(s)))
            pos += len(s) + 1
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        ids = enc["input_ids"]
        s_idx, in_sp = lib.sentence_index_per_token(enc["offset_mapping"],
                                                    spans)
        ids_flat.extend(ids)
        off.append(len(ids_flat))
        sent_idx_all.append(s_idx)
        in_span_all.append(in_sp.astype(np.int8))
        for f in FACES:
            s_lam = pl.sentence_lambda(events[f][d])
            lam[f].append(pl.token_labels_from_sentences(
                s_lam, s_idx).astype(np.float32))
            is_evt[f].append(events[f][d][s_idx])

    ids_flat = np.array(ids_flat, dtype=np.int32)
    doc_off = np.array(off, dtype=np.int64)

    ref = np.load(HERE / f"replag_fineweb_{key}.npz")
    assert np.array_equal(ids_flat, ref["token_ids"]), \
        f"{key}: token_ids diverge from replag — cache reuse would break"

    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = lib.doc_split(n_docs, seed=SEED)
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1

    arrays = {
        "token_ids": ids_flat, "doc_off": doc_off, "doc_split": split,
        "sent_idx": np.concatenate(sent_idx_all),
        "in_span": np.concatenate(in_span_all),
    }
    stats_faces = {}
    for f in FACES:
        v = np.concatenate(lam[f])
        evt = np.concatenate(is_evt[f]).astype(np.int8)
        scheme, edges, bins = pl.zero_split_bins(v, train_rows)
        masked_bins = np.where(evt == 1, -1, bins).astype(np.int8)
        d_, p_, c_ = lib.balanced_manifest(masked_bins, doc_of, pos_of,
                                           seed=SEED)
        arrays[f"lam_{f}"] = v
        arrays[f"{f}_bin"] = bins
        arrays[f"is_{f}"] = evt
        arrays[f"man_{f}_doc"], arrays[f"man_{f}_pos"] = d_, p_
        arrays[f"man_{f}_cls"] = c_

        elig = (masked_bins >= 0) & (pos_of >= lib.MIN_MANIFEST_POS)
        unigram = nl.type_mean_scores(ids_flat, v, train_rows & elig)
        tri = {
            "unigram_auc": nl.tercile_auc(unigram, masked_bins,
                                          test_rows & elig),
            "position_auc": nl.tercile_auc(pos_of.astype(float), masked_bins,
                                           test_rows & elig),
        }
        fin = np.isfinite(v)
        stats_faces[f] = {
            "scheme": scheme, "edges": edges,
            "event_sentence_rate": float(np.mean(np.concatenate(
                [events[f][d] for d in range(n_docs)]))),
            "lam_zero_frac": float((v[fin] == 0).mean()),
            "lam_mean": float(v[fin].mean()), "lam_std": float(v[fin].std()),
            "masked_token_frac": float(evt.mean()),
            "manifest_rows_per_class": int(len(d_) // 3),
            "triage": tri,
        }

    out = HERE / f"punctint_fineweb_{key}.npz"
    np.savez_compressed(out, **arrays)
    print(f"[{key}] {ids_flat.size:,} tok; " + "; ".join(
        f"{f}: scheme={stats_faces[f]['scheme']}, "
        f"evrate={stats_faces[f]['event_sentence_rate']:.3f}, "
        f"triage={json.dumps(stats_faces[f]['triage'])}" for f in FACES))
    return {"tokenizer": model, "n_docs": n_docs,
            "n_tokens": int(ids_flat.size), "token_ids_match_replag": True,
            "faces": stats_faces, "artifact": out.name}


def main():
    sent_lists, meta = load_docs()
    events = doc_events(sent_lists)
    stats = {
        "source": str(FINEWEB.relative_to(ROOT)), "source_meta": meta,
        "seed": SEED,
        "kernel": {"half_life_sentences": pl.HALF_LIFE_S,
                   "support_sentences": pl.SUPPORT_S,
                   "mass_within_sentences": {
                       n: pl.kernel_mass_within_sentences(n)
                       for n in (1, 2, 4, 8)}},
        "list_regex": pl.LIST_RE.pattern,
        "min_manifest_pos": lib.MIN_MANIFEST_POS,
        "per_tokenizer": {},
    }
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, sent_lists, events)
    (HERE / "punctint_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'punctint_stats.json'}")


if __name__ == "__main__":
    main()
