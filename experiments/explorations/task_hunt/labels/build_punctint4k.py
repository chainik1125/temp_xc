"""Punctuation-event intensity labels at 10x scale (corpus-scaleup item
1): the SAME frozen label logic as ``build_punctint.py``, on the 4,000-doc
fineweb pull instead of the pinned 400.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_punctint4k

``build_punctint.py`` and ``punctint_lib.py`` are NOT touched: this
builder imports the frozen lib unchanged (same LIST_RE grammar, same
8-sentence/half-life-2 kernel, same zero_split scheme, same
position-matched manifest routine) and writes NEW versioned artifacts
beside the shipped ones. Three things are new, and none of them is label
logic:

1. **The replag assert becomes a PREFIX assert — and it is a receipt.**
   The shipped builder asserted ``token_ids`` byte-identity against
   ``replag_fineweb_<tok>.npz``; that covers only the pinned 400 docs.
   Here the first-400-doc token slice (and the doc offsets that carve it)
   is compared instead. A PASS proves both that the 4,000-doc pull is a
   deterministic superset of the pinned sample AND that tokenization is
   unchanged — so **the GPU pods' existing caches already cover the first
   400 documents; only the new ~3,600 need a caching pass.** A FAIL is
   recorded and the build continues without the cache-reuse claim (the
   shipped 400-doc bundle is a different artifact and is unaffected).
2. **Manifest cap raised 20k -> 100k rows/class**, and the stats say what
   the data actually supports: ``manifest_rows_per_class_supported`` is
   the uncapped position-matched ceiling (sum over log2 strata of the
   smallest class count in that stratum), so a binding cap is visible as
   such.
3. **Doc-level bootstrap CIs (>= 1,000 reps) on every triage AUC**
   (``boot_lib``), plus the adopted ``doc_mean_only_auc`` disclosure
   statistic on both faces, plus the within-document-contrast census: how
   many documents carry manifest rows of BOTH the top and bottom class
   (runpod-e's punctint-list control rested on 8 documents at 400).

Triage bars are unchanged and stay frozen: direction-agnostic
max(AUC, 1-AUC), current-token type-mean AUC >= 0.65 => KILL, position
AUC >= 0.65 => KILL, manifest rows operative, 0.55-0.65 ships with
disclosure. A bar firing at scale is a FINDING that binds the Stage-2
design; it does not retro-kill the shipped small-corpus bundle.

Artifacts: ``punctint4k_fineweb_<tok>.npz`` + ``punctint4k_stats.json``.
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

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "fineweb4k_corpus.json.gz"
SEED = 0
MANIFEST_CAP = 100_000

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}

FACES = ("list", "q")


def load_docs(n_docs=None):
    sample = json.loads(gzip.decompress(CORPUS.read_bytes()))
    docs = sample["docs"] if n_docs is None else sample["docs"][:n_docs]
    return [d["sentences"] for d in docs], sample["meta"]


def doc_events(sent_lists):
    ev = {"list": [], "q": []}
    for sents in sent_lists:
        ev["list"].append(np.array([pl.is_list_sentence(s) for s in sents],
                                   dtype=np.int8))
        ev["q"].append(np.array([pl.is_question_sentence(s) for s in sents],
                                dtype=np.int8))
    return ev


def supported_rows_per_class(masked_bins, strata):
    """The uncapped position-matched ceiling: summed over log2 strata,
    the smallest class count within the stratum (what the manifest could
    ship per class if `cap` did not bind)."""
    ok = (masked_bins >= 0) & (strata >= 0)
    classes = np.unique(masked_bins[ok])
    total = 0
    for s in np.unique(strata[ok]):
        in_s = ok & (strata == s)
        counts = [int((in_s & (masked_bins == c)).sum()) for c in classes]
        total += min(counts)
    return int(total)


def within_doc_contrast(man_doc, man_cls, doc_split):
    """How many documents can carry a within-document contrast: they hold
    manifest rows of BOTH the top and the bottom class (the '8 documents'
    census at 400 docs). Reported over all manifest docs and over the
    operative test-doc subset."""
    out = {}
    for tag, keep in (("all", np.ones(len(man_doc), dtype=bool)),
                      ("test", doc_split[man_doc] == 1)):
        d, c = man_doc[keep], man_cls[keep]
        top = set(np.unique(d[c == 2]).tolist())
        bot = set(np.unique(d[c == 0]).tolist())
        both = top & bot
        rows_in_both = int(np.isin(d, list(both)).sum()) if both else 0
        out[tag] = {"docs_with_top_rows": len(top),
                    "docs_with_bottom_rows": len(bot),
                    "docs_with_both": len(both),
                    "manifest_rows_in_those_docs": rows_in_both}
    return out


def build_for_tokenizer(key, model, sent_lists, events, n_reps, out_dir, tag):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    t0 = time.time()

    id_docs, off = [], [0]
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
        ids = np.asarray(enc["input_ids"], dtype=np.int32)
        s_idx, in_sp = lib.sentence_index_per_token(enc["offset_mapping"],
                                                    spans)
        id_docs.append(ids)
        off.append(off[-1] + ids.size)
        sent_idx_all.append(s_idx)
        in_span_all.append(in_sp.astype(np.int8))
        for f in FACES:
            s_lam = pl.sentence_lambda(events[f][d])
            lam[f].append(pl.token_labels_from_sentences(
                s_lam, s_idx).astype(np.float32))
            is_evt[f].append(events[f][d][s_idx])

    ids_flat = np.concatenate(id_docs)
    doc_off = np.array(off, dtype=np.int64)
    print(f"[{key}] tokenized {len(sent_lists)} docs -> {ids_flat.size:,} "
          f"tokens in {time.time() - t0:.0f}s", flush=True)

    # the prefix receipt (see module docstring) — disclosed, never fatal
    ref = np.load(HERE / f"replag_fineweb_{key}.npz")
    ref_ids, ref_off = ref["token_ids"], ref["doc_off"]
    n_ref_docs = len(ref_off) - 1
    tok_prefix = bool(ids_flat.size >= ref_ids.size
                      and np.array_equal(ids_flat[:ref_ids.size], ref_ids))
    off_prefix = bool(len(doc_off) > n_ref_docs
                      and np.array_equal(doc_off[:n_ref_docs + 1], ref_off))
    if tok_prefix and off_prefix:
        print(f"[{key}] replag PREFIX OK — {ref_ids.size:,} of "
              f"{ids_flat.size:,} tokens are already cached", flush=True)
    else:
        print(f"[{key}] replag prefix MISMATCH (tokens={tok_prefix}, "
              f"doc_off={off_prefix}) — cache-reuse claim dropped", flush=True)

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
    strata = pl.pos_strata(pos_of, min_pos=lib.MIN_MANIFEST_POS)
    for f in FACES:
        v = np.concatenate(lam[f])
        evt = np.concatenate(is_evt[f]).astype(np.int8)
        scheme, edges, bins = pl.zero_split_bins(v, train_rows)
        masked_bins = np.where(evt == 1, -1, bins).astype(np.int8)
        d_, p_, c_ = pl.stratified_balanced_manifest(
            masked_bins, strata, doc_of, pos_of, cap=MANIFEST_CAP, seed=SEED)
        arrays[f"lam_{f}"] = v
        arrays[f"{f}_bin"] = bins
        arrays[f"is_{f}"] = evt
        arrays[f"man_{f}_doc"], arrays[f"man_{f}_pos"] = d_, p_
        arrays[f"man_{f}_cls"] = c_

        elig = (masked_bins >= 0) & (pos_of >= lib.MIN_MANIFEST_POS)
        unigram = nl.type_mean_scores(ids_flat, v, train_rows & elig)
        # adopted disclosure statistic: document-mean of the label as the
        # only feature (no frozen kill threshold — reported, not operative)
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
                tb = time.time()
                boot[rname][s] = bo.bootstrap_tercile_auc(
                    sc, masked_bins, rmask, doc_of, n_reps=n_reps, seed=SEED)
                print(f"[{key}/{f}] boot {rname}.{s}: "
                      f"{boot[rname][s]['point']:.4f} "
                      f"[{boot[rname][s]['ci_lo']:.4f}, "
                      f"{boot[rname][s]['ci_hi']:.4f}] "
                      f"({boot[rname][s]['n_rows']:,} rows, "
                      f"{time.time() - tb:.0f}s)", flush=True)

        fin = np.isfinite(v)
        stats_faces[f] = {
            "scheme": scheme, "edges": edges,
            "event_sentence_rate": float(np.mean(np.concatenate(
                [events[f][d] for d in range(n_docs)]))),
            "lam_zero_frac": float((v[fin] == 0).mean()),
            "lam_mean": float(v[fin].mean()), "lam_std": float(v[fin].std()),
            "masked_token_frac": float(evt.mean()),
            "eligible_frac": float(elig.mean()),
            "manifest_cap_per_class": MANIFEST_CAP,
            "manifest_rows_per_class": int(len(d_) // 3),
            "manifest_rows_per_class_supported": supported_rows_per_class(
                masked_bins, strata),
            "within_document_contrast": within_doc_contrast(d_, c_, split),
            "triage_all_eligible_rows": tri["triage_all_eligible_rows"],
            "triage_manifest_rows": tri["triage_manifest_rows"],
            "bootstrap": boot,
        }

    out = out_dir / f"punctint{tag}_fineweb_{key}.npz"
    np.savez_compressed(out, **arrays)
    print(f"[{key}] wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - t0:.0f}s total", flush=True)
    return {"tokenizer": model, "n_docs": n_docs,
            "n_tokens": int(ids_flat.size),
            "token_ids_prefix_matches_replag": tok_prefix,
            "doc_off_prefix_matches_replag": off_prefix,
            "replag_prefix_tokens": int(ref_ids.size),
            "new_tokens_needing_cache": int(ids_flat.size - ref_ids.size)
            if tok_prefix else int(ids_flat.size),
            "faces": stats_faces, "artifact": out.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=None,
                    help="limit docs (smoke runs only)")
    ap.add_argument("--reps", type=int, default=bo.N_REPS)
    ap.add_argument("--tag", default="4k")
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_lists, meta = load_docs(a.n_docs)
    events = doc_events(sent_lists)
    stats = {
        "source": CORPUS.name, "source_meta": meta,
        "n_docs_used": len(sent_lists), "seed": SEED,
        "frozen_logic": "punctint_lib (unchanged) — same kernel, grammar, "
                        "zero_split scheme and position-matched manifests "
                        "as the shipped 400-doc build",
        "kernel": {"half_life_sentences": pl.HALF_LIFE_S,
                   "support_sentences": pl.SUPPORT_S,
                   "mass_within_sentences": {
                       n: pl.kernel_mass_within_sentences(n)
                       for n in (1, 2, 4, 8)}},
        "list_regex": pl.LIST_RE.pattern,
        "min_manifest_pos": lib.MIN_MANIFEST_POS,
        "manifest_cap_per_class": MANIFEST_CAP,
        "bootstrap": {"unit": "document (cluster)", "n_reps": a.reps,
                      "ci_pct": list(bo.CI_PCT), "seed": SEED},
        "per_tokenizer": {},
    }
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, sent_lists, events, a.reps, out_dir, a.tag)
    p = out_dir / f"punctint{a.tag}_stats.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
