"""Fineweb 400 → 4,000-doc scale-up pull (corpus-scaleup campaign, item 1).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.pull_fineweb4k

**Same recipe as the pinned 400-doc sample, only the stop count moves.**
``expansion.corpus.sample_fineweb`` is called with the pinned arguments —
``HuggingFaceFW/fineweb`` / ``sample-10BT`` / ``train``, seed 0, shuffle
buffer 10,000, keep docs with 60–200 sentences under the pinned
``split_sentences v1`` — with ``n_docs`` 400 → 4,000. The stream is
shuffled by seed alone (independent of ``n_docs``) and the sampler simply
breaks later, so the pull **scans the same rows in the same order** and
its first 400 kept documents should be exactly the pinned sample.

That prediction is a receipt, not an assumption: this script writes
``fineweb4k_corpus_receipt.json`` recording, doc by doc, whether the
first 400 documents match the pinned sample's ids and sentence lists.
A PASS means the scaled corpus is a deterministic SUPERSET of the pinned
one (the token-level consequence — existing GPU caches already cover the
first 400 docs — is asserted separately in ``build_punctint4k``). A FAIL
is disclosed, not hidden: the scaled corpus is then a different sample of
the same recipe, the cache-reuse claim is dropped, and the shipped
400-doc bundle stands unaffected on its own artifact.

Artifact: ``fineweb4k_corpus.json.gz`` here (gzip — the plain JSON is
~38 MB). The pinned original
``synthetic/expansion/data/fineweb_sample.json`` is NEVER touched.
Idempotent twice over: an existing artifact short-circuits the pull, and
an interrupted pull leaves the plain-JSON work file, which
``sample_fineweb`` itself reuses on the next run (it is removed once the
gzip artifact is written).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from explorations.synthetic.expansion import corpus as ec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PINNED = (ROOT / "experiments/explorations/synthetic/expansion/data/"
          "fineweb_sample.json")
OUT = HERE / "fineweb4k_corpus.json.gz"
WORK = HERE / "fineweb4k_corpus.json"          # transient; removed on success
RECEIPT = HERE / "fineweb4k_corpus_receipt.json"

N_DOCS = 4_000
SEED = 0


def load() -> dict:
    """Read the committed scaled corpus artifact."""
    with gzip.open(OUT, "rt") as fh:
        return json.load(fh)


def prefix_receipt(sample: dict) -> dict:
    """Is the pinned 400-doc sample the prefix of the scaled pull?"""
    pinned = json.loads(PINNED.read_text())
    p_docs = pinned["docs"]
    got = sample["docs"][: len(p_docs)]
    id_match = sum(a.get("id") == b.get("id") for a, b in zip(p_docs, got))
    sent_match = sum(a["sentences"] == b["sentences"]
                     for a, b in zip(p_docs, got))
    first_bad = next((i for i, (a, b) in enumerate(zip(p_docs, got))
                      if a.get("id") != b.get("id")
                      or a["sentences"] != b["sentences"]), None)
    return {
        "pinned_n_docs": len(p_docs),
        "compared": len(got),
        "id_match": int(id_match),
        "sentences_match": int(sent_match),
        "first_mismatch_index": first_bad,
        "prefix_identity": bool(len(got) == len(p_docs)
                                and id_match == len(p_docs)
                                and sent_match == len(p_docs)),
        "pinned_meta": pinned["meta"],
    }


def main() -> None:
    if OUT.exists():
        print(f"[pull] artifact present: {OUT.name}")
        sample = load()
    else:
        sample = ec.sample_fineweb(
            WORK, n_docs=N_DOCS, seed=SEED,
            dataset="HuggingFaceFW/fineweb", name="sample-10BT",
            split="train", min_sents=60, max_sents=200,
            shuffle_buffer=10_000, log=print)
        with gzip.open(OUT, "wt") as fh:
            json.dump(sample, fh)
        WORK.unlink(missing_ok=True)
        print(f"[pull] wrote {OUT.name} "
              f"({OUT.stat().st_size / 1e6:.1f} MB gz)")

    rec = prefix_receipt(sample)
    out = {"meta": sample["meta"], "prefix_receipt": rec,
           "artifact": OUT.name,
           "n_tokens_hint": "see build_punctint4k stats (per tokenizer)"}
    RECEIPT.write_text(json.dumps(out, indent=1))
    print(json.dumps(sample["meta"], indent=1))
    print(f"[pull] prefix_identity={rec['prefix_identity']} "
          f"(ids {rec['id_match']}/{rec['pinned_n_docs']}, "
          f"sentences {rec['sentences_match']}/{rec['pinned_n_docs']})")
    print(f"-> {RECEIPT}")


if __name__ == "__main__":
    main()
