"""Interleaved-document (anti-conversion) corpus + labels — exact,
CPU-only (hunt-support-stats item 3; round-3 optionality).

    HF_TOKEN=... .venv/bin/python -m \
        experiments.explorations.task_hunt.labels.build_interleave

Corpus design (pure logic in ``interleave_lib.py``, tested):

- Each pinned fineweb doc (``synthetic/expansion/data/fineweb_sample.json``,
  400 docs) is split in half by sentence: the FIRST half builds the
  corpus (USE), the SECOND half is held out to estimate source unigram
  distributions for the triage (EST) — any in-corpus estimator leaks
  the source through its own count asymmetry (see interleave_lib tests).
- Docs are paired by **lexical-overlap matching** (greedy max Jaccard
  over USE-half content types) — the control that keeps source identity
  from being trivial vocabulary detection. A seeded random pairing is
  evaluated in-memory as the matching null (its corpus is not saved).
- Each pair is interleaved in strictly alternating blocks of 1–4
  sentences (uniform jitter, per-pair seed = pair index), truncating
  when a source cannot serve a full block.
- Per-token labels: ``source`` (0/1 within the pair), ``tss`` (tokens
  since the last switch; -1 on the first block = no prior switch),
  ``block``. The **shuffled-block null** (per-pair seed = 1000 + pair
  index) ships as ``null_perm`` (within-doc token permutation realized
  by shuffling whole blocks; the null corpus is token_ids[perm]) with
  ``source_null``/``tss_null`` recomputed on the permuted order
  (adjacent same-source blocks merge). The null id-sequence is defined
  by the permutation (it need not retokenize from text) — same
  convention as replag's within-doc shuffle null.
- **Per-token-first triage, labels only** (no activations): (a) per-pair
  source AUC from held-out unigram log-odds, matched vs random pairing —
  the kill-risk number this candidate lives or dies by; (b) global
  unigram→tss-tercile AUC (train-doc type means, test-doc AUC top vs
  bottom tercile); (c) the switch-hazard profile h(t) — the honest
  disclosure of how much generative signal tss carries (jittered
  sentence blocks are NOT memoryless).

Artifacts: ``interleave_fineweb_<tok>.npz`` + ``interleave_stats.json``
here. Same alignment contract as replag: consumers feed these exact
``token_ids`` (positions index the no-special-tokens sequence).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import lib
from . import interleave_lib as il

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FINEWEB = (ROOT / "experiments/explorations/synthetic/expansion/data/"
           "fineweb_sample.json")
SEED = 0
BLOCK_LO, BLOCK_HI = 1, 4          # sentences per block, uniform jitter
NULL_SEED_BASE = 1000

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}


def load_docs():
    sample = json.loads(FINEWEB.read_text())
    use, est = [], []
    for d in sample["docs"]:
        sents = d["sentences"]
        h = len(sents) // 2
        use.append(sents[:h])
        est.append(sents[h:])
    return use, est, sample["meta"]


def interleave_pair(use_a, use_b, pair_idx):
    """Block plan + the interleaved sentence list + per-sentence block
    ids for one pair."""
    plan = il.plan_blocks(len(use_a), len(use_b), seed=pair_idx,
                          lo=BLOCK_LO, hi=BLOCK_HI)
    cursors = [0, 0]
    sents, sent_block = [], []
    for b, (src, n) in enumerate(plan):
        pool = use_a if src == 0 else use_b
        for _ in range(n):
            sents.append(pool[cursors[src]])
            cursors[src] += 1
            sent_block.append(b)
    return plan, sents, np.array(sent_block, dtype=np.int32)


def tokenize_with_blocks(tok, sents, sent_block, n_blocks):
    """Tokenize the joined interleaved text; map tokens -> sentence via
    the committed char-midpoint rule -> per-block token counts."""
    text = " ".join(sents)
    spans, pos = [], 0
    for s in sents:
        spans.append((pos, pos + len(s)))
        pos += len(s) + 1
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.array(enc["input_ids"], dtype=np.int32)
    sent_idx, _ = lib.sentence_index_per_token(enc["offset_mapping"], spans)
    tok_block = sent_block[sent_idx]
    counts = np.bincount(tok_block, minlength=n_blocks).tolist()
    return ids, counts


def flat_ids(tok, sents):
    if not sents:
        return []
    return tok(" ".join(sents), add_special_tokens=False)["input_ids"]


def split_blocks(ids, counts):
    out, c = [], 0
    for n in counts:
        out.append(ids[c:c + n].tolist())
        c += n
    return out


def pair_triage_auc(tok, est_a, est_b, plan, ids, counts):
    return il.source_lexical_auc(flat_ids(tok, est_a), flat_ids(tok, est_b),
                                 plan, split_blocks(ids, counts))


def build_for_tokenizer(key, model, use, est, matched, rand_pairs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    per_doc = []          # matched-corpus per-pair dicts
    matched_aucs, rand_aucs = [], []
    counts_all = []
    for pair_idx, (i, j, ov) in enumerate(matched):
        plan, sents, sent_block = interleave_pair(use[i], use[j], pair_idx)
        ids, counts = tokenize_with_blocks(tok, sents, sent_block, len(plan))
        src, tss, blk = il.token_labels(plan, counts)
        perm, src_n, tss_n = il.block_shuffle(
            plan, counts, seed=NULL_SEED_BASE + pair_idx)
        matched_aucs.append(pair_triage_auc(tok, est[i], est[j],
                                            plan, ids, counts))
        counts_all.append(counts)
        per_doc.append(dict(ids=ids, src=src, tss=tss, blk=blk, perm=perm,
                            src_null=src_n, tss_null=tss_n,
                            a=i, b=j, ov=ov))
    # the matching null: same construction on random pairs, stats only
    for pair_idx, (i, j, _) in enumerate(rand_pairs):
        plan, sents, sent_block = interleave_pair(use[i], use[j], pair_idx)
        ids, counts = tokenize_with_blocks(tok, sents, sent_block, len(plan))
        rand_aucs.append(pair_triage_auc(tok, est[i], est[j],
                                         plan, ids, counts))

    n_docs = len(per_doc)
    doc_split = lib.doc_split(n_docs, seed=SEED)
    ids = np.concatenate([d["ids"] for d in per_doc])
    doc_off = np.concatenate(
        [[0], np.cumsum([len(d["ids"]) for d in per_doc])]).astype(np.int64)
    src = np.concatenate([d["src"] for d in per_doc])
    tss = np.concatenate([d["tss"] for d in per_doc])
    blk = np.concatenate([d["blk"] for d in per_doc])
    perm = np.concatenate([d["perm"] for d in per_doc])
    src_null = np.concatenate([d["src_null"] for d in per_doc])
    tss_null = np.concatenate([d["tss_null"] for d in per_doc])

    # global unigram -> tss-tercile triage (train-doc type means)
    doc_of_tok = np.repeat(np.arange(n_docs), np.diff(doc_off))
    pos_of_tok = np.concatenate(
        [np.arange(n) for n in np.diff(doc_off)]).astype(np.int32)
    train_tok = (doc_split[doc_of_tok] == 0) & (tss >= 0)
    test_tok = (doc_split[doc_of_tok] == 1) & (tss >= 0)
    edges, _ = lib.tercile_bins(tss[train_tok].astype(float))
    sums = np.zeros(int(ids.max()) + 1)
    cnts = np.zeros(int(ids.max()) + 1)
    np.add.at(sums, ids[train_tok], tss[train_tok])
    np.add.at(cnts, ids[train_tok], 1)
    gmean = float(tss[train_tok].mean())
    type_mean = np.where(cnts > 0, sums / np.maximum(cnts, 1), gmean)
    tt = tss[test_tok].astype(float)
    scores = type_mean[ids[test_tok]]
    lab = np.where(tt > edges[1], 1, np.where(tt <= edges[0], 0, -1))
    tss_auc = il.rank_auc(scores[lab >= 0], lab[lab >= 0])

    # manifests (pos >= 32 so any screened T <= 32 fits)
    _, tss_bins = lib.tercile_bins(np.where(tss >= 0, tss,
                                            np.nan).astype(float))
    man_t = lib.balanced_manifest(tss_bins, doc_of_tok, pos_of_tok,
                                  seed=SEED)
    man_s = lib.balanced_manifest(src.astype(np.int8), doc_of_tok,
                                  pos_of_tok, seed=SEED)

    np.savez_compressed(
        HERE / f"interleave_fineweb_{key}.npz",
        token_ids=ids, doc_off=doc_off, source=src, tss=tss, block=blk,
        null_perm=perm, source_null=src_null, tss_null=tss_null,
        pair_doc_a=np.array([d["a"] for d in per_doc], dtype=np.int16),
        pair_doc_b=np.array([d["b"] for d in per_doc], dtype=np.int16),
        pair_overlap=np.array([d["ov"] for d in per_doc],
                              dtype=np.float32),
        doc_split=doc_split,
        man_tss_doc=man_t[0], man_tss_pos=man_t[1], man_tss_cls=man_t[2],
        man_src_doc=man_s[0], man_src_pos=man_s[1], man_src_cls=man_s[2])

    m_auc = np.array(matched_aucs, dtype=float)
    r_auc = np.array(rand_aucs, dtype=float)
    return {
        "n_pairs": n_docs, "n_tokens": int(ids.size),
        "tss_valid_rate": float((tss >= 0).mean()),
        "tss_tercile_edges_train": [float(e) for e in edges],
        "manifest_rows_per_class": {"tss": int(len(man_t[0]) // 3),
                                    "src": int(len(man_s[0]) // 2)},
        "triage": {
            "source_auc_matched": {
                "mean": float(np.nanmean(m_auc)),
                "median": float(np.nanmedian(m_auc)),
                "p90": float(np.nanquantile(m_auc, 0.9))},
            "source_auc_random_pairing": {
                "mean": float(np.nanmean(r_auc)),
                "median": float(np.nanmedian(r_auc)),
                "p90": float(np.nanquantile(r_auc, 0.9))},
            "tss_unigram_auc_top_vs_bottom_tercile": float(tss_auc)},
        "switch_hazard": il.switch_hazard(counts_all),
    }


def main():
    if "HF_TOKEN" in os.environ:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN",
                              os.environ["HF_TOKEN"])
    use, est, meta = load_docs()
    vocabs = [il.content_types(u) for u in use]
    matched = il.pair_docs_by_overlap(vocabs)
    rand_pairs = il.random_pairing(len(use), seed=SEED)
    m_ov = np.array([ov for _, _, ov in matched])
    r_ov = np.array([il.jaccard(vocabs[i], vocabs[j])
                     for i, j, _ in rand_pairs])

    stats = {
        "source_sample": meta,
        "design": {
            "use_est_split": "first half of each doc's sentences builds "
                             "the corpus; second half estimates unigram "
                             "distributions for the triage",
            "block_sentences": [BLOCK_LO, BLOCK_HI],
            "pair_seed": "pair index", "null_seed": "1000 + pair index",
            "split_seed": SEED},
        "pairing": {
            "n_matched_pairs": len(matched),
            "overlap_matched": {
                "mean": float(m_ov.mean()), "median": float(np.median(m_ov)),
                "p10": float(np.quantile(m_ov, 0.1)),
                "p90": float(np.quantile(m_ov, 0.9))},
            "overlap_random": {
                "mean": float(r_ov.mean()),
                "median": float(np.median(r_ov)),
                "p90": float(np.quantile(r_ov, 0.9))}},
        "tokenizers": {},
    }
    for key, model in TOKENIZERS.items():
        stats["tokenizers"][key] = build_for_tokenizer(
            key, model, use, est, matched, rand_pairs)
        print(f"[{key}] {json.dumps(stats['tokenizers'][key]['triage'])}")
    (HERE / "interleave_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE}/interleave_fineweb_<tok>.npz ; interleave_stats.json")


if __name__ == "__main__":
    main()
