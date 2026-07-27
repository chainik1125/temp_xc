"""WikiText-103 pull for the gen-4 CORPUS SCOUT (mac-c lane, beat
review ~12:15 item 3): long-form encyclopedic narrative — the first
NON-dialogue, non-fiction substrate for the return-family faces.
New-corpus rules (pg19 precedent): exact re-pull script, pinned
revision, funnel counters, receipt.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.pull_wikitext103

Source: ``Salesforce/wikitext`` config ``wikitext-103-raw-v1`` (parquet
mirror of the original WikiText-103; CC-BY-SA-3.0), revision PINNED
below. The stream is LINES, not documents: articles begin at a
depth-1 header line (`` = Title = ``) and section boundaries are
depth-2+ headers (`` = = Section = = ``) — documents are reassembled
here, deterministically, in stream order (no shuffle: line-streams
cannot be row-shuffled without splitting documents; first-N-kept is
label-free).

Recipe (label-FREE by design — no face statistic is consulted):

- stream ``train`` sequentially; assemble docs between depth-1
  headers (the header line ships with its doc);
- keep docs with body length in [MIN_CHARS, CAP_CHARS] after
  truncating the line list at CAP_CHARS (truncation at a line edge,
  never mid-line);
- stop at N_DOCS kept.

Funnel counters (headers seen / too short / kept) and the first-doc
identity receipt go to ``wikitext103_corpus_receipt.json``; corpus to
``wikitext103_corpus.json.gz``. Idempotent.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "wikitext103_corpus.json.gz"
RECEIPT = HERE / "wikitext103_corpus_receipt.json"

DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-103-raw-v1"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"  # pinned parquet mirror
SPLIT = "train"
MIN_CHARS = 1_200
CAP_CHARS = 5_000
N_DOCS = 400

# depth-1 header = exactly one '=' each side; depth-2+ starts '= ='
RE_H1 = re.compile(r"^\s*=\s*[^=\s].*?\s*=\s*$")
RE_H2 = re.compile(r"^\s*=\s*=")


def is_h1(line: str) -> bool:
    return bool(RE_H1.match(line)) and not RE_H2.match(line)


def main():
    if OUT.exists():
        print(f"{OUT.name} exists — pull short-circuited (idempotent)")
        return
    import datasets as hfds

    ds = hfds.load_dataset(DATASET, CONFIG, split=SPLIT, streaming=True,
                           revision=REVISION)
    docs, cur, n_h1, n_short = [], None, 0, 0

    def close(cur):
        nonlocal n_short
        if cur is None:
            return
        lines, total = [], 0
        for ln in cur["lines"]:
            if total + len(ln) > CAP_CHARS:
                break
            lines.append(ln)
            total += len(ln)
        if total < MIN_CHARS:
            n_short += 1
            return
        docs.append({"title": cur["title"], "lines": lines})

    for row in ds:
        line = row["text"]
        if is_h1(line):
            n_h1 += 1
            close(cur)
            if len(docs) >= N_DOCS:
                cur = None
                break
            cur = {"title": line.strip(), "lines": [line]}
        elif cur is not None:
            cur["lines"].append(line)
    if cur is not None and len(docs) < N_DOCS:
        close(cur)
    docs = docs[:N_DOCS]

    meta = {"dataset": DATASET, "config": CONFIG, "revision": REVISION,
            "split": SPLIT, "order": "stream order, first-N kept (no "
            "shuffle possible on a line stream; label-free)",
            "min_chars": MIN_CHARS, "cap_chars": CAP_CHARS,
            "n_docs": len(docs), "n_h1_seen": n_h1,
            "n_too_short": n_short,
            "n_chars": sum(len(l) for d in docs for l in d["lines"]),
            "label_free_pull": ("no face statistic consulted at pull "
                                "time — the corpus is not selected on "
                                "any label")}
    first = docs[0]
    meta["first_doc_receipt"] = {
        "title": first["title"],
        "sha256": hashlib.sha256(
            "".join(first["lines"]).encode()).hexdigest()}

    with gzip.open(OUT, "wt") as f:
        json.dump(docs, f)
    RECEIPT.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
