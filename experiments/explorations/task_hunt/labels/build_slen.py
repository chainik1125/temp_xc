"""Sentence-length recency-ladder labels (CANDIDATES.md B8 `slen`;
card `../slen/CARD_DRAFT.md`): THREE faces of ONE exact value stream
x_i = ln(word count of sentence i), differing only in temporal
weighting — `lat` (previous sentence's x: a latch, the recency face,
PRIMARY), `lev` (HL-2/support-8 kernel trailing mean — P6 absorbed),
`disp` (kernel trailing std — the program's first second-moment
face). Frozen logic in ``slen_lib.py``; the pre-registered
within-window-shuffle ladder (lat > lev > disp ≈ 0) is in the card.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_slen                       # 4k corpus
    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_slen --n-docs 400 --tag 400  # cache-aligned prefix variant

Machinery follows ``build_punctint4k.py`` unchanged where it applies:
same corpus file, same sentence→token bridge, same replag PREFIX
receipt (the 400-doc variant aligns token-for-token with the existing
GPU caches), same position-matched manifests, doc-level 1,000-rep
bootstrap CIs on every triage AUC, ``doc_mean_only_auc`` disclosure +
within-document-contrast census. Differences, all label-side facts:
no event stream exists, so there is NOTHING to mask beyond the
unified NaN warm-up (sentence idx < 8) — the current sentence never
contributes to any face by construction — and the stats add the
face-correlation matrix plus cross-bundle correlations vs punctint's
``lam_q``/``lam_list`` (the independence receipts) and quote the
unigram estimator's training size (the estimator finding).

Artifacts: ``slen<tag>_fineweb_<tok>.npz`` + ``slen<tag>_stats.json``.
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
from . import punctint_lib as pl
from . import slen_lib as sl
from .build_punctint4k import (load_docs, supported_rows_per_class,
                               within_doc_contrast)

HERE = Path(__file__).resolve().parent
SEED = 0
MANIFEST_CAP = 100_000
FACES = ("lat", "lev", "disp")

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",
}

# independence receipts: correlate faces against the punctint labels
# on the SAME corpus variant (token grids are identical by the prefix
# receipt, asserted before any correlation is computed)
PEER = {"4k": "punctint4k_fineweb_{tok}.npz",
        "400": "punctint_fineweb_{tok}.npz"}


def face_values(sent_lists):
    vals = {f: [] for f in FACES}
    for sents in sent_lists:
        x = sl.sent_log_lengths(sents)
        vals["lat"].append(sl.trailing_latch(x))
        vals["lev"].append(sl.trailing_level(x))
        vals["disp"].append(sl.trailing_disp(x))
    return vals


def build_for_tokenizer(key, model, sent_lists, svals, n_reps, out_dir, tag):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    t0 = time.time()

    id_docs, off, sent_idx_all, in_span_all = [], [0], [], []
    lam = {f: [] for f in FACES}
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
            lam[f].append(pl.token_labels_from_sentences(
                svals[f][d], s_idx).astype(np.float32))

    ids_flat = np.concatenate(id_docs)
    doc_off = np.array(off, dtype=np.int64)
    print(f"[{key}] tokenized {len(sent_lists)} docs -> {ids_flat.size:,} "
          f"tokens in {time.time() - t0:.0f}s", flush=True)

    # the replag prefix receipt (disclosed, never fatal) — for the 400
    # variant this is an exact-identity receipt: zero new caching
    ref = np.load(HERE / f"replag_fineweb_{key}.npz")
    ref_ids, ref_off = ref["token_ids"], ref["doc_off"]
    n_ref_docs = len(ref_off) - 1
    tok_prefix = bool(ids_flat.size >= ref_ids.size
                      and np.array_equal(ids_flat[:ref_ids.size], ref_ids))
    off_prefix = bool(len(doc_off) > n_ref_docs
                      and np.array_equal(doc_off[:n_ref_docs + 1], ref_off))
    if tok_prefix and off_prefix:
        print(f"[{key}] replag PREFIX OK — {ref_ids.size:,} of "
              f"{ids_flat.size:,} tokens already cached", flush=True)
    else:
        print(f"[{key}] replag prefix MISMATCH (tokens={tok_prefix}, "
              f"doc_off={off_prefix}) — cache-reuse claim dropped",
              flush=True)

    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = lib.doc_split(n_docs, seed=SEED)
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1
    n_train_docs = int((split == 0).sum())

    arrays = {"token_ids": ids_flat, "doc_off": doc_off,
              "doc_split": split,
              "sent_idx": np.concatenate(sent_idx_all),
              "in_span": np.concatenate(in_span_all)}
    stats_faces = {}
    strata = pl.pos_strata(pos_of, min_pos=lib.MIN_MANIFEST_POS)
    vflat = {f: np.concatenate(lam[f]) for f in FACES}
    for f in FACES:
        v = vflat[f]
        scheme, edges, bins = pl.zero_split_bins(v, train_rows)
        # no event stream — nothing to mask; -1 marks NaN warm-up only
        d_, p_, c_ = pl.stratified_balanced_manifest(
            bins, strata, doc_of, pos_of, cap=MANIFEST_CAP, seed=SEED)
        arrays[f"val_{f}"] = v
        arrays[f"{f}_bin"] = bins
        arrays[f"man_{f}_doc"], arrays[f"man_{f}_pos"] = d_, p_
        arrays[f"man_{f}_cls"] = c_

        elig = (bins >= 0) & (pos_of >= lib.MIN_MANIFEST_POS)
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
            tri[rname] = {s: nl.tercile_auc(sc, bins, rmask)
                          for s, sc in scores.items()}
            boot[rname] = {}
            for s, sc in scores.items():
                boot[rname][s] = bo.bootstrap_tercile_auc(
                    sc, bins, rmask, doc_of, n_reps=n_reps, seed=SEED)
            print(f"[{key}/{f}] {rname}: "
                  + " ".join(f"{s.split('_auc')[0]}="
                             f"{boot[rname][s]['point']:.4f}"
                             f"[{boot[rname][s]['ci_lo']:.3f},"
                             f"{boot[rname][s]['ci_hi']:.3f}]"
                             for s in scores), flush=True)

        fin = np.isfinite(v)
        stats_faces[f] = {
            "scheme": scheme, "edges": [float(e) for e in edges],
            "val_mean": float(v[fin].mean()),
            "val_std": float(v[fin].std()),
            "eligible_frac": float(elig.mean()),
            "unigram_train_docs": n_train_docs,
            "manifest_cap_per_class": MANIFEST_CAP,
            "manifest_rows_per_class": int(len(d_) // 3),
            "manifest_rows_per_class_supported": supported_rows_per_class(
                bins, strata),
            "within_document_contrast": within_doc_contrast(d_, c_, split),
            "triage_all_eligible_rows": tri["triage_all_eligible_rows"],
            "triage_manifest_rows": tri["triage_manifest_rows"],
            "bootstrap": boot,
        }

    # independence receipts: face-face and face-punctint correlations
    # over rows where everything involved is finite and eligible
    elig_all = (pos_of >= lib.MIN_MANIFEST_POS)
    for f in FACES:
        elig_all &= np.isfinite(vflat[f])
    corr = {}
    for i, a in enumerate(FACES):
        for b in FACES[i + 1:]:
            corr[f"{a}~{b}"] = float(np.corrcoef(
                vflat[a][elig_all], vflat[b][elig_all])[0, 1])
    peer_path = HERE / PEER[tag].format(tok=key) if tag in PEER else None
    if peer_path is not None and peer_path.exists():
        pz = np.load(peer_path)
        if np.array_equal(pz["token_ids"], ids_flat):
            for pf in ("lam_q", "lam_list"):
                pv = pz[pf]
                for f in FACES:
                    m = elig_all & np.isfinite(pv)
                    corr[f"{f}~{pf}"] = float(np.corrcoef(
                        vflat[f][m], pv[m])[0, 1])
        else:
            corr["peer_note"] = "token grid mismatch — peer corr skipped"
    print(f"[{key}] corr: " + " ".join(
        f"{k}={v:+.3f}" for k, v in corr.items()
        if isinstance(v, float)), flush=True)

    out = out_dir / f"slen{tag}_fineweb_{key}.npz"
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
            "correlations": corr,
            "faces": stats_faces, "artifact": out.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=None)
    ap.add_argument("--reps", type=int, default=bo.N_REPS)
    ap.add_argument("--tag", default="4k")
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_lists, meta = load_docs(a.n_docs)
    svals = face_values(sent_lists)
    stats = {
        "source": "fineweb4k_corpus.json.gz", "source_meta": meta,
        "n_docs_used": len(sent_lists), "seed": SEED,
        "frozen_logic": "slen_lib (committed before this run) — "
                        "x=ln(word count), faces lat/lev/disp over "
                        "PREVIOUS sentences, unified warm-up idx<8, "
                        "no masking (no event stream exists)",
        "kernel": {"half_life_sentences": sl.HALF_LIFE_S,
                   "support_sentences": sl.SUPPORT_S,
                   "ess": sl.kernel_ess(),
                   "mass_within_sentences": {
                       n: pl.kernel_mass_within_sentences(
                           n, sl.HALF_LIFE_S, sl.SUPPORT_S)
                       for n in (1, 2, 4, 8)}},
        "min_manifest_pos": lib.MIN_MANIFEST_POS,
        "manifest_cap_per_class": MANIFEST_CAP,
        "bootstrap": {"unit": "document (cluster)", "n_reps": a.reps,
                      "ci_pct": list(bo.CI_PCT), "seed": SEED},
        "per_tokenizer": {},
    }
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, sent_lists, svals, a.reps, out_dir, a.tag)
    p = out_dir / f"slen{a.tag}_stats.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
