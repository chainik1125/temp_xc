"""Repetition-lag Δ labels (task-hunt candidate 2) — exact, zero-API.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_replag

For each screen tokenizer (gpt2, gemma-2-2b, Llama-3.1-8B base — the last
is also the Ward stream's tokenizer), tokenize the pinned fineweb sample
(``synthetic/expansion/data/fineweb_sample.json``, 400 docs; text
reconstructed by joining the pinned sentence split with single spaces) at
``add_special_tokens=False`` and emit per-position labels:

- ``delta1``/``delta2``   int32 — distance to the previous occurrence of the
  current 1-/2-gram (end-to-end), -1 = none;
- ``delta1_shuf``/``delta2_shuf`` — the same after a seeded within-doc token
  shuffle: the frequency-only null (a per-token feature can reach at most
  this Δ structure; the real-vs-shuf gap is the order signal);
- ``bucket1``/``bucket2`` int8 — Δ buckets 0:1–4, 1:5–8, 2:9–16,
  3:"none" (no occurrence within 64), -1: guard band 17–64 (excluded);
- ``logfreq``  float32 — log10 in-corpus frequency of the current token
  (for frequency-stratified inspection);
- ``token_ids`` int32 + ``doc_off`` int64 (n_docs+1 prefix offsets) — the
  EXACT token sequences; consumers must feed these ids (not re-tokenize)
  so labels and activations align by construction;
- ``doc_split`` int8 per doc (0 train / 1 test, 20% test, seed 0) — probes
  must split BY DOC;
- ``man{1,2}_doc/pos/cls`` — balanced probe-row manifests over the four
  buckets (cap 20k/bucket, pos ≥ 32 so any screened T ≤ 32 fits).

Artifacts: ``replag_fineweb_<tok>.npz`` + ``replag_stats.json`` here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib

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


def doc_texts() -> list[str]:
    sample = json.loads(FINEWEB.read_text())
    return [" ".join(d["sentences"]) for d in sample["docs"]], sample["meta"]


def build_for_tokenizer(key: str, model: str, texts: list[str]) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    rng = np.random.default_rng(SEED)

    ids_flat, off = [], [0]
    d1, d2, d1s, d2s = [], [], [], []
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids_flat.extend(ids)
        off.append(len(ids_flat))
        d1.append(lib.delta_prev_ngram(ids, 1))
        d2.append(lib.delta_prev_ngram(ids, 2))
        d1s.append(lib.shuffled_doc_null(ids, 1, rng))
        d2s.append(lib.shuffled_doc_null(ids, 2, rng))

    ids_flat = np.array(ids_flat, dtype=np.int32)
    doc_off = np.array(off, dtype=np.int64)
    delta1, delta2 = np.concatenate(d1), np.concatenate(d2)
    delta1_shuf, delta2_shuf = np.concatenate(d1s), np.concatenate(d2s)
    bucket1, bucket2 = lib.bucketize_delta(delta1), lib.bucketize_delta(delta2)

    counts = np.bincount(ids_flat, minlength=int(ids_flat.max()) + 1)
    logfreq = np.log10(counts[ids_flat].astype(np.float64)
                       / len(ids_flat)).astype(np.float32)

    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(doc_off[i + 1] - doc_off[i],
                                       dtype=np.int32)
                             for i in range(n_docs)])
    split = lib.doc_split(n_docs, seed=SEED)

    arrays = {
        "token_ids": ids_flat, "doc_off": doc_off,
        "delta1": delta1, "delta2": delta2,
        "delta1_shuf": delta1_shuf, "delta2_shuf": delta2_shuf,
        "bucket1": bucket1, "bucket2": bucket2,
        "logfreq": logfreq, "doc_split": split,
    }
    for n, bucket in (("1", bucket1), ("2", bucket2)):
        d, p, c = lib.balanced_manifest(bucket, doc_of, pos_of, seed=SEED)
        arrays[f"man{n}_doc"], arrays[f"man{n}_pos"] = d, p
        arrays[f"man{n}_cls"] = c

    out = HERE / f"replag_fineweb_{key}.npz"
    np.savez_compressed(out, **arrays)

    def bucket_hist(b):
        return {str(i): int((b == i).sum()) for i in (-1, 0, 1, 2, 3)}

    import transformers
    stats = {
        "tokenizer": model, "transformers_version": transformers.__version__,
        "vocab": len(tok), "n_docs": n_docs, "n_tokens": int(len(ids_flat)),
        "tokens_per_doc": {
            "mean": float(np.diff(doc_off).mean()),
            "median": float(np.median(np.diff(doc_off))),
            "min": int(np.diff(doc_off).min()),
            "max": int(np.diff(doc_off).max())},
        "bucket1": bucket_hist(bucket1), "bucket2": bucket_hist(bucket2),
        "bucket1_shuf": bucket_hist(lib.bucketize_delta(delta1_shuf)),
        "bucket2_shuf": bucket_hist(lib.bucketize_delta(delta2_shuf)),
        "manifest_rows_per_bucket": {
            "n1": int(len(arrays["man1_doc"]) // 4),
            "n2": int(len(arrays["man2_doc"]) // 4)},
        "artifact": out.name,
    }
    print(f"[{key}] {stats['n_tokens']:,} tokens; bucket1 {stats['bucket1']}; "
          f"manifest/bucket n1={stats['manifest_rows_per_bucket']['n1']:,}")
    return stats


def main() -> None:
    texts, meta = doc_texts()
    stats = {"source": str(FINEWEB.relative_to(ROOT)),
             "source_meta": meta, "seed": SEED,
             "bucket_scheme": {"edges": list(lib.BUCKET_EDGES),
                               "none_min": lib.NONE_MIN,
                               "guard_excluded": [lib.BUCKET_EDGES[-1] + 1,
                                                  lib.NONE_MIN]},
             "min_manifest_pos": lib.MIN_MANIFEST_POS,
             "per_tokenizer": {}}
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(key, model, texts)
    (HERE / "replag_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'replag_stats.json'}")


if __name__ == "__main__":
    main()
