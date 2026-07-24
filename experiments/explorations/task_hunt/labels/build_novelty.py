"""Vocabulary-novelty trailing-rate labels (CANDIDATES.md B2) — exact,
zero-API, CPU-only.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_novelty

Same corpus, tokenization, and alignment contract as
``build_replag.py`` (the pinned fineweb sample, sentences joined by
single spaces, ``add_special_tokens=False``, per-doc sequences) — and
the builder ASSERTS byte-identical ``token_ids`` against the committed
``replag_fineweb_<tok>.npz``, so any GPU cache built for the replag
screen drops onto these labels with ZERO new caching.

Per-token arrays (positions index the no-special-tokens sequence):

- ``nov``          int8   — 1 at the first in-doc occurrence of the type;
- ``nov_rate``     float32 — kernel-smoothed trailing novelty rate over
  PREVIOUS tokens only (lags 1..64, half-life 16; the current token
  never contributes to its own label); NaN for pos < 64;
- ``nov_resid``    float32 — nov_rate minus the train-doc mean of its
  log2 position bin (Heaps-trend removal — the PRIMARY face);
- ``nov_bin``      int8   — terciles of nov_resid (train-row edges);
- ``nov_raw_bin``  int8   — terciles of nov_rate (position-confounded
  face, DISCLOSED secondary only);
- ``null_perm``    int64  — seeded within-doc permutation; the null
  corpus is token_ids[null_perm], with ``nov_null``/``nov_rate_null``
  recomputed ON the permuted order (frequency-only null: the number of
  first occurrences per doc is permutation-invariant, their ORDER is
  not — same convention as replag's shuffle null);
- ``doc_split``    int8 per doc (20 % test, seed 0);
- ``man_nov_*`` / ``man_novraw_*`` — balanced (doc,pos,cls) manifests,
  cap 20k/class, **pos >= 64** (kernel support; a stated deviation from
  the shared pos >= 32 floor).

Label-side triage (kill authority; bars frozen in
``../novelty/CARD_DRAFT.md`` BEFORE this ran): current-token type-mean
AUC and position-only AUC, each on raw and detrended terciles
(top vs bottom, test-doc rows), plus the kernel clock bridge and
real-vs-null rate spread. Artifacts: ``novelty_fineweb_<tok>.npz`` +
``novelty_stats.json`` here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FINEWEB = (ROOT / "experiments/explorations/synthetic/expansion/data/"
           "fineweb_sample.json")
SEED = 0
NULL_SEED = 1000

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}


def doc_texts():
    sample = json.loads(FINEWEB.read_text())
    return [" ".join(d["sentences"]) for d in sample["docs"]], sample["meta"]


def build_for_tokenizer(key: str, model: str, texts) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    ids_flat, off = [], [0]
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids_flat.extend(ids)
        off.append(len(ids_flat))
    ids_flat = np.array(ids_flat, dtype=np.int32)
    doc_off = np.array(off, dtype=np.int64)

    ref = np.load(HERE / f"replag_fineweb_{key}.npz")
    assert np.array_equal(ids_flat, ref["token_ids"]), \
        f"{key}: token_ids diverge from replag — cache reuse would break"
    assert np.array_equal(doc_off, ref["doc_off"])

    n_docs = len(doc_off) - 1
    nov = np.concatenate([nl.novelty_bits(ids_flat[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])
    rate = np.concatenate(
        [nl.trailing_rate(nov[doc_off[d]:doc_off[d + 1]])
         for d in range(n_docs)]).astype(np.float32)

    perm = nl.within_doc_perm(doc_off, seed=NULL_SEED)
    ids_null = ids_flat[perm]
    nov_null = np.concatenate(
        [nl.novelty_bits(ids_null[doc_off[d]:doc_off[d + 1]])
         for d in range(n_docs)])
    rate_null = np.concatenate(
        [nl.trailing_rate(nov_null[doc_off[d]:doc_off[d + 1]])
         for d in range(n_docs)]).astype(np.float32)

    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = lib.doc_split(n_docs, seed=SEED)
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1

    pbin = nl.position_bin(pos_of)
    resid, expected = nl.detrend(rate, pbin, train_rows)
    resid_null, _ = nl.detrend(rate_null, pbin, train_rows)

    def terciles(vals):
        fin = np.isfinite(vals) & train_rows
        edges = np.quantile(vals[fin], [1 / 3, 2 / 3])
        out = np.full(vals.shape, -1, dtype=np.int8)
        m = np.isfinite(vals)
        out[m] = np.digitize(vals[m], edges).astype(np.int8)
        return [float(e) for e in edges], out

    edges_resid, nov_bin = terciles(resid)
    edges_raw, nov_raw_bin = terciles(rate)

    man = {}
    for name, b in (("nov", nov_bin), ("novraw", nov_raw_bin)):
        d, p, c = lib.balanced_manifest(b, doc_of, pos_of,
                                        min_pos=nl.SUPPORT, seed=SEED)
        man[f"man_{name}_doc"], man[f"man_{name}_pos"] = d, p
        man[f"man_{name}_cls"] = c

    out = HERE / f"novelty_fineweb_{key}.npz"
    np.savez_compressed(
        out, token_ids=ids_flat, doc_off=doc_off, nov=nov, nov_rate=rate,
        nov_resid=resid, nov_bin=nov_bin, nov_raw_bin=nov_raw_bin,
        null_perm=perm, nov_null=nov_null, nov_rate_null=rate_null,
        nov_resid_null=resid_null, doc_split=split, **man)

    # ── label-side triage (test-doc rows, pos >= SUPPORT) ──────────────
    elig = pos_of >= nl.SUPPORT
    tri = {}
    for face, vals, terc in (("raw", rate, nov_raw_bin),
                             ("resid", resid, nov_bin)):
        unigram = nl.type_mean_scores(ids_flat, vals, train_rows & elig)
        tri[face] = {
            "unigram_auc": nl.tercile_auc(unigram, terc, test_rows & elig),
            "position_auc": nl.tercile_auc(pos_of.astype(float), terc,
                                           test_rows & elig),
        }
    fin = np.isfinite(rate)
    stats = {
        "tokenizer": model, "n_docs": n_docs, "n_tokens": int(ids_flat.size),
        "token_ids_match_replag": True,
        "novelty_base_rate": float(nov.mean()),
        "rate_mean": float(rate[fin].mean()),
        "rate_std": float(rate[fin].std()),
        "rate_null_std": float(rate_null[np.isfinite(rate_null)].std()),
        "resid_std": float(resid[np.isfinite(resid)].std()),
        "resid_null_std": float(resid_null[np.isfinite(resid_null)].std()),
        # lags > kernel SUPPORT share no input bits — null ~ 0 there by
        # construction, so real-above-null is drift structure, not filter
        "resid_autocorr": {
            str(lag): {"real": nl.pooled_doc_autocorr(resid, doc_off, lag),
                       "null": nl.pooled_doc_autocorr(resid_null, doc_off,
                                                      lag)}
            for lag in (16, 32, 64, 128)},
        "position_bin_expected_rate": expected,
        "tercile_edges": {"resid": edges_resid, "raw": edges_raw},
        "manifest_rows_per_class": {
            "nov": int(len(man["man_nov_doc"]) // 3),
            "novraw": int(len(man["man_novraw_doc"]) // 3)},
        "triage": tri,
        "artifact": out.name,
    }
    print(f"[{key}] {stats['n_tokens']:,} tok; base {nov.mean():.3f}; "
          f"triage {json.dumps(tri)}")
    return stats


def main():
    texts, meta = doc_texts()
    stats = {
        "source": str(FINEWEB.relative_to(ROOT)), "source_meta": meta,
        "seed": SEED, "null_seed": NULL_SEED,
        "kernel": {"half_life": nl.HALF_LIFE, "support": nl.SUPPORT,
                   "mass_within_T": {T: nl.kernel_mass_within(T)
                                     for T in (4, 8, 16, 32, 64)}},
        "min_manifest_pos": nl.SUPPORT,
        "per_tokenizer": {},
    }
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(key, model, texts)
    (HERE / "novelty_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'novelty_stats.json'}")


if __name__ == "__main__":
    main()
