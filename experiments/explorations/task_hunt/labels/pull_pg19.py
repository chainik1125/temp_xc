"""PG19-class fiction pull for the quotedens bundle (CANDIDATES.md B9;
round-3 factory). New-corpus rules: exact re-pull script, pinned
revision, funnel counters, receipt.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.pull_pg19

Source: ``emozilla/pg19`` (parquet mirror of DeepMind PG19 — the
original ``deepmind/pg19`` is a script-based dataset that modern
``datasets`` refuses to load; the mirror carries the same fields:
short_book_title / publication_date / url / text), revision PINNED
below. Pre-1919 published books, English, Project Gutenberg licence.

Recipe (label-FREE by design — no quote statistic is consulted at
pull time, so the corpus is not selected on the label):

- stream ``train``, shuffled seed 0 / buffer 1,000 (deterministic);
- per book, sentence-split the FIRST 400k characters with the pinned
  program splitter (``expansion.corpus.split_sentences v1`` — the
  same splitter every fineweb artifact uses);
- keep books yielding >= 250 sentences; ship sentences [100, 250) —
  a 150-sentence span with a 100-sentence front-matter guard
  (title pages, tables of contents);
- stop at 1,000 kept books.

Funnel counters (scanned / too-short / kept) and the first-doc
identity receipt go to ``pg19_corpus_receipt.json``; corpus to
``pg19_corpus.json.gz``. Idempotent: an existing artifact
short-circuits the pull.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from explorations.synthetic.expansion.corpus import split_sentences

HERE = Path(__file__).resolve().parent
OUT = HERE / "pg19_corpus.json.gz"
RECEIPT = HERE / "pg19_corpus_receipt.json"

DATASET = "emozilla/pg19"
REVISION = "c021754c8e01c5b1cc83a1f549c1f97fbbb756b8"  # pinned parquet mirror
SEED = 0
SHUFFLE_BUFFER = 1_000
HEAD_CHARS = 400_000
MIN_SENTS = 250
SPAN = (100, 250)
N_BOOKS = 1_000


def main():
    if OUT.exists():
        print(f"{OUT.name} exists — pull short-circuited (idempotent)")
        return
    import datasets as hfds

    ds = hfds.load_dataset(DATASET, split="train", streaming=True,
                           revision=REVISION)
    ds = ds.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
    docs, n_seen, n_short = [], 0, 0
    for row in ds:
        n_seen += 1
        sents = split_sentences(row["text"][:HEAD_CHARS])
        if len(sents) < MIN_SENTS:
            n_short += 1
            continue
        span = sents[SPAN[0]: SPAN[1]]
        docs.append({"id": row.get("url") or row.get("short_book_title"),
                     "title": row.get("short_book_title"),
                     "publication_date": row.get("publication_date"),
                     "sentences": span})
        if len(docs) % 100 == 0:
            print(f"[pg19] {len(docs)}/{N_BOOKS} books "
                  f"({n_seen} scanned, {n_short} too short)", flush=True)
        if len(docs) >= N_BOOKS:
            break

    meta = {"dataset": DATASET, "revision": REVISION, "split": "train",
            "seed": SEED, "shuffle_buffer": SHUFFLE_BUFFER,
            "head_chars": HEAD_CHARS, "min_sents": MIN_SENTS,
            "span": list(SPAN), "n_docs": len(docs),
            "n_scanned": n_seen, "n_too_short": n_short,
            "n_sentences": sum(len(d["sentences"]) for d in docs),
            "splitter": "expansion.corpus.split_sentences v1",
            "label_free_pull": ("no quote statistic consulted at pull "
                                "time — the corpus is not selected on "
                                "the label")}
    out = {"meta": meta, "docs": docs}
    OUT.write_bytes(gzip.compress(json.dumps(out).encode()))
    first = docs[0]
    RECEIPT.write_text(json.dumps({
        **meta,
        "first_doc": {"title": first["title"], "id": first["id"],
                      "sha256_sentences": hashlib.sha256(
                          json.dumps(first["sentences"]).encode()
                      ).hexdigest()},
    }, indent=1))
    print(f"-> {OUT.name} ({OUT.stat().st_size / 1e6:.1f} MB), "
          f"{meta['n_sentences']:,} sentences from {len(docs)} books "
          f"({n_seen} scanned, {n_short} too short)")


if __name__ == "__main__":
    main()
