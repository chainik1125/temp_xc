"""Vocabulary-novelty labels at 10x scale (corpus-scaleup, extension):
the SAME frozen label logic as ``build_novelty.py``, on the 4,000-doc
fineweb pull instead of the pinned 400.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_novelty4k

**Why this exists, given the briefing scoped item 3 to a label-side
bootstrap.** The briefing's premise is that surviving faces should not
rest on 400 documents. When it was written, `novelty` had screened
NEGATIVE, so it needed only CIs. runpod-e has since **withdrawn that
verdict** (scoring error in their own best-window rule; the face
re-scores KEEP as the card is written, pending review of the convention
itself) — which moves `novelty` into exactly the population the campaign
exists to serve. The fineweb-4k corpus and the frozen `novelty_lib` were
both already in hand, so the marginal cost is one build.

Frozen logic reused unchanged: `novelty_lib.novelty_bits` /
`trailing_rate` (half-life 16, support 64, current token never in its
own label), the Heaps position-bin detrend, the seeded within-doc
permutation null (`NULL_SEED` 1000), tercile edges from train rows, and
`lib.balanced_manifest` with `min_pos = SUPPORT` — note this bundle
predates the position-matched manifest convention and **keeps its own**:
changing it would be a design decision belonging to the screen owner,
not to a scale-up. What the position-matched alternative WOULD support
is reported as a census (`position_matched_support_per_class`) so that
call can be made on numbers.

Changes, none of them label logic: the corpus (4,000 docs), the manifest
cap (20k -> 100k rows/class), the replag assert becomes a PREFIX assert
(the scaled corpus is a token-for-token superset — receipt in
`SCALEUP.md` §2), and every triage AUC gains a 1,000-rep document-level
bootstrap CI plus the adopted `doc_mean_only_auc`.

**The npz artifacts are written but deliberately NOT committed** —
~145 MB per tokenizer (the null permutation and four float32 label
arrays over 7.9M tokens). They are exactly regenerable: this builder is
committed, the corpus artifact is committed, and every seed is pinned.
Committed instead: ``novelty4k_stats.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import boot_lib as bo
from . import lib
from . import novelty_lib as nl
from .pull_fineweb4k import load as load_corpus

HERE = Path(__file__).resolve().parent
SEED = 0
NULL_SEED = 1000
MANIFEST_CAP = 100_000

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}


def doc_texts(n_docs=None):
    sample = load_corpus()
    docs = sample["docs"] if n_docs is None else sample["docs"][:n_docs]
    return [" ".join(d["sentences"]) for d in docs], sample["meta"]


def position_matched_support(bins, pos_of):
    """What a position-matched manifest WOULD support per class (sum over
    log2 strata of the smallest class count) — reported, not shipped."""
    from . import punctint_lib as pl
    strata = pl.pos_strata(pos_of, min_pos=nl.SUPPORT)
    ok = (bins >= 0) & (strata >= 0)
    total = 0
    for s in np.unique(strata[ok]):
        in_s = ok & (strata == s)
        total += min(int((in_s & (bins == c)).sum())
                     for c in np.unique(bins[ok]))
    return int(total)


def build_for_tokenizer(key, model, texts, n_reps, out_dir, tag):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    t0 = time.time()

    id_docs, off = [], [0]
    for text in texts:
        ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"],
                         dtype=np.int32)
        id_docs.append(ids)
        off.append(off[-1] + ids.size)
    ids_flat = np.concatenate(id_docs)
    doc_off = np.array(off, dtype=np.int64)
    print(f"[{key}] tokenized -> {ids_flat.size:,} tokens "
          f"({time.time() - t0:.0f}s)", flush=True)

    ref = np.load(HERE / f"replag_fineweb_{key}.npz")
    ref_ids, ref_off = ref["token_ids"], ref["doc_off"]
    tok_prefix = bool(ids_flat.size >= ref_ids.size
                      and np.array_equal(ids_flat[:ref_ids.size], ref_ids))
    off_prefix = bool(np.array_equal(doc_off[: len(ref_off)], ref_off))
    print(f"[{key}] replag prefix: tokens={tok_prefix} doc_off={off_prefix}",
          flush=True)

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
    print(f"[{key}] labels + null built ({time.time() - t0:.0f}s)",
          flush=True)

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
        d, p, c = lib.balanced_manifest(b, doc_of, pos_of, cap=MANIFEST_CAP,
                                        min_pos=nl.SUPPORT, seed=SEED)
        man[f"man_{name}_doc"], man[f"man_{name}_pos"] = d, p
        man[f"man_{name}_cls"] = c

    out = out_dir / f"novelty{tag}_fineweb_{key}.npz"
    np.savez_compressed(
        out, token_ids=ids_flat, doc_off=doc_off, nov=nov, nov_rate=rate,
        nov_resid=resid, nov_bin=nov_bin, nov_raw_bin=nov_raw_bin,
        null_perm=perm, nov_null=nov_null, nov_rate_null=rate_null,
        nov_resid_null=resid_null, doc_split=split, **man)

    elig = pos_of >= nl.SUPPORT
    tri, boot, census = {}, {}, {}
    for face, vals, terc, mname in (("raw", rate, nov_raw_bin, "novraw"),
                                    ("resid", resid, nov_bin, "nov")):
        unigram = nl.type_mean_scores(ids_flat, vals, train_rows & elig)
        docmean = np.full(n_docs, np.nan)
        for d in range(n_docs):
            seg = vals[doc_off[d]: doc_off[d + 1]]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                docmean[d] = seg.mean()
        man_rows = np.zeros(len(pos_of), dtype=bool)
        man_rows[doc_off[:-1][man[f"man_{mname}_doc"]]
                 + man[f"man_{mname}_pos"]] = True
        scores = {"unigram_auc": unigram,
                  "position_auc": pos_of.astype(float),
                  "doc_mean_only_auc": docmean[doc_of]}
        row_sets = {"triage_all_eligible_rows": test_rows & elig,
                    "triage_manifest_rows": man_rows & test_rows}
        tri[face], boot[face] = {}, {}
        for rname, rmask in row_sets.items():
            tri[face][rname] = {s: nl.tercile_auc(sc, terc, rmask)
                                for s, sc in scores.items()}
            boot[face][rname] = {}
            for s, sc in scores.items():
                b = bo.bootstrap_tercile_auc(sc, terc, rmask, doc_of,
                                             n_reps=n_reps, seed=SEED)
                boot[face][rname][s] = b
                print(f"[{key}/{face}] {rname}.{s}: {b['point']:.4f} "
                      f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}] "
                      f"({b['n_rows']:,} rows)", flush=True)
        census[face] = {
            "manifest_rows_per_class": int(len(man[f"man_{mname}_doc"]) // 3),
            "manifest_cap_per_class": MANIFEST_CAP,
            "position_matched_support_per_class": position_matched_support(
                terc, pos_of),
        }

    fin = np.isfinite(rate)
    stats = {
        "tokenizer": model, "n_docs": n_docs, "n_tokens": int(ids_flat.size),
        "token_ids_prefix_matches_replag": tok_prefix,
        "doc_off_prefix_matches_replag": off_prefix,
        "novelty_base_rate": float(nov.mean()),
        "rate_mean": float(rate[fin].mean()), "rate_std": float(rate[fin].std()),
        "rate_null_std": float(rate_null[np.isfinite(rate_null)].std()),
        "resid_std": float(resid[np.isfinite(resid)].std()),
        "resid_null_std": float(resid_null[np.isfinite(resid_null)].std()),
        "resid_autocorr": {
            str(lag): {"real": nl.pooled_doc_autocorr(resid, doc_off, lag),
                       "null": nl.pooled_doc_autocorr(resid_null, doc_off,
                                                      lag)}
            for lag in (16, 32, 64, 128)},
        "position_bin_expected_rate": expected,
        "tercile_edges": {"resid": edges_resid, "raw": edges_raw},
        "census": census,
        "triage": tri, "bootstrap": boot,
        "artifact": out.name,
        "artifact_committed": False,
        "artifact_note": "~145 MB/tokenizer; regenerable exactly from this "
                         "committed builder + the committed corpus artifact "
                         "(all seeds pinned)",
    }
    print(f"[{key}] done in {time.time() - t0:.0f}s "
          f"({out.stat().st_size / 1e6:.0f} MB npz, uncommitted)", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--reps", type=int, default=bo.N_REPS)
    ap.add_argument("--tag", default="4k")
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts, meta = doc_texts(a.n_docs)
    stats = {
        "source": "fineweb4k_corpus.json.gz", "source_meta": meta,
        "n_docs_used": len(texts), "seed": SEED, "null_seed": NULL_SEED,
        "frozen_logic": "novelty_lib + lib.balanced_manifest, unchanged "
                        "(this bundle predates the position-matched "
                        "manifest convention and keeps its own)",
        "kernel": {"half_life": nl.HALF_LIFE, "support": nl.SUPPORT,
                   "mass_within_T": {T: nl.kernel_mass_within(T)
                                     for T in (4, 8, 16, 32, 64)}},
        "min_manifest_pos": nl.SUPPORT,
        "bootstrap": {"unit": "document (cluster)", "n_reps": a.reps,
                      "ci_pct": list(bo.CI_PCT), "seed": SEED},
        "per_tokenizer": {},
    }
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, texts, a.reps, out_dir, a.tag)
    p = out_dir / f"novelty{a.tag}_stats.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
