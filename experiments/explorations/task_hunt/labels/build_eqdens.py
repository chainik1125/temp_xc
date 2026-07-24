"""Equation-density labels on OpenWebMath (CANDIDATES.md B6) — exact,
zero-API; NEW corpus (round 2, `briefings/candidate-factory-broad-2.md`).

    HF_TOKEN=... .venv/bin/python -m \
        experiments.explorations.task_hunt.labels.build_eqdens

Corpus: ``open-web-math/open-web-math`` (train split, PINNED revision;
ODC-By 1.0 + CommonCrawl ToU). The dataset is ~27M docs — we stream
the FIRST ``N_STREAM`` examples in shard order at the pinned revision
(deterministic; a stated convenience-sample disclosure), filter to
chars in [MIN_CHARS, MAX_CHARS] with >= MIN_SPANS math spans under the
FROZEN grammar (`eqdens_lib.MATH_RE` — the span-count floor kills the
math-doc-vs-prose-doc identity route at pull time), and seeded-
subsample to ``N_DOCS``. **The exact sample ships as
``eqdens_corpus.json.gz``** (texts + meta), so consumers never
re-pull; this builder is also the exact re-pull script. GPU economics:
a NEW token stream (~1M tokens/tokenizer) — one caching pass per
model, minutes on an H100; no existing cache applies.

Per-token arrays per tokenizer (``eqdens_openwebmath_<tok>.npz``):

- ``mrate``    float32 — kernel-smoothed trailing math-token rate over
  the PREVIOUS 64 tokens (half-life 16; current token NEVER in its own
  label; NaN below position 64) — the PRIMARY intensity face;
- ``mrate_bin``  int8 — 3-class via the conditional zero_split/tercile
  scheme, with math-span tokens MASKED to -1 (they read the label
  ambiently — the disclosed anchor face is ``in_math`` itself);
- ``in_math``  int8 — the anchor bit (bracket family, recorded dead;
  secondary/disclosed only, never manifested);
- ``mrate_null`` / ``mrate_null_bin`` — the within-doc-shuffle
  frequency null (same marginal math rate, arrangement destroyed;
  binned with the REAL scheme's edges so classes are comparable) —
  the screen's mechanism-receipt face;
- ``doc_split`` int8 per doc (20 % test, seed 0);
- ``man_mrate_*`` — position-MATCHED balanced manifests (equal class
  counts per log2 position stratum), math tokens masked, pos >= 64
  (the fineweb warm-up floor; strata from 64 up).

Label-side triage (bars FROZEN in ``../eqdens/CARD_DRAFT.md``,
committed before this ran; broad convention pinned there): current-
token type-mean AUC + position AUC, direction-agnostic read,
all-eligible AND manifest rows (manifest operative).
Artifacts land here: ``eqdens_openwebmath_<tok>.npz`` +
``eqdens_stats.json`` + ``eqdens_corpus.json.gz``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import eqdens_lib as el

HERE = Path(__file__).resolve().parent
SEED = 0
DATASET = "open-web-math/open-web-math"
REVISION = "fde8ef8de2300f5e778f56261843dab89f230815"
N_STREAM = 4000
N_DOCS = 600
MIN_CHARS, MAX_CHARS = 1000, 20_000
MIN_SPANS = 3

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}


def _span_form(text: str, a: int) -> str:
    head = text[a: a + 2]
    if head == "$$":
        return "display_$$"
    if head[0] == "$":
        return "inline_$"
    if head == "\\[":
        return "bracket"
    if head == "\\(":
        return "paren"
    return "environment"


def pull_corpus():
    corpus_path = HERE / "eqdens_corpus.json.gz"
    if corpus_path.exists():
        payload = json.loads(gzip.decompress(corpus_path.read_bytes()))
        return payload["docs"], payload["meta"]
    import datasets
    ds = datasets.load_dataset(DATASET, split="train", revision=REVISION,
                               streaming=True)
    keep = []
    for i, ex in enumerate(ds):
        if i >= N_STREAM:
            break
        t = ex["text"]
        if el.doc_passes_filter(t, MIN_CHARS, MAX_CHARS, MIN_SPANS):
            keep.append(t)
    rng = np.random.default_rng(SEED)
    if len(keep) > N_DOCS:
        idx = np.sort(rng.choice(len(keep), size=N_DOCS, replace=False))
        keep = [keep[i] for i in idx]
    form_counts: dict = {}
    for t in keep:
        for a, _ in el.math_spans(t):
            f = _span_form(t, a)
            form_counts[f] = form_counts.get(f, 0) + 1
    meta = {"dataset": DATASET, "revision": REVISION,
            "license": "ODC-By 1.0 (+ CommonCrawl ToU)",
            "split": "train", "stream_prefix": N_STREAM,
            "filter": {"min_chars": MIN_CHARS, "max_chars": MAX_CHARS,
                       "min_spans": MIN_SPANS},
            "seed": SEED, "n_docs": len(keep),
            "span_form_counts": form_counts}
    corpus_path.write_bytes(gzip.compress(json.dumps(
        {"meta": meta, "docs": keep}).encode()))
    return keep, meta


def _bins_with_edges(vals, scheme, edges):
    """Bin `vals` with an already-fit scheme (for the null face)."""
    out = np.full(vals.shape, -1, dtype=np.int8)
    m = np.isfinite(vals)
    if scheme == "zero_split":
        med = edges[1]
        out[m & (vals == 0)] = 0
        out[m & (vals > 0) & (vals <= med)] = 1
        out[m & (vals > med)] = 2
    else:
        out[m] = np.digitize(vals[m], np.asarray(edges)).astype(np.int8)
    return out


def build_for_tokenizer(key, model, docs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    ids_flat, off, bits_all = [], [0], []
    for text in docs:
        cmask = el.char_math_mask(text)
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        bits_all.append(el.token_math_bits(enc["offset_mapping"], cmask))
        ids_flat.extend(enc["input_ids"])
        off.append(len(ids_flat))

    ids_flat = np.array(ids_flat, dtype=np.int32)
    doc_off = np.array(off, dtype=np.int64)
    in_math = np.concatenate(bits_all)

    perm = nl.within_doc_perm(doc_off, seed=SEED)
    bits_null = in_math[perm]
    mrate = np.concatenate([el.trailing_math_rate(b) for b in bits_all])
    mrate_null = np.concatenate(
        [el.trailing_math_rate(bits_null[doc_off[d]: doc_off[d + 1]])
         for d in range(len(doc_off) - 1)])

    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = lib.doc_split(n_docs, seed=SEED)
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1

    scheme, edges, bins = pl.zero_split_bins(mrate, train_rows)
    masked_bins = np.where(in_math == 1, -1, bins).astype(np.int8)
    null_bins = _bins_with_edges(mrate_null, scheme, edges)
    null_masked = np.where(in_math == 1, -1, null_bins).astype(np.int8)
    strata = pl.pos_strata(pos_of, min_pos=el.MIN_POS)
    d_, p_, c_ = pl.stratified_balanced_manifest(
        masked_bins, strata, doc_of, pos_of, seed=SEED)

    elig = (masked_bins >= 0) & (pos_of >= el.MIN_POS)
    unigram = nl.type_mean_scores(ids_flat, mrate, train_rows & elig)
    tri_all = {
        "unigram_auc": nl.tercile_auc(unigram, masked_bins,
                                      test_rows & elig),
        "position_auc": nl.tercile_auc(pos_of.astype(float), masked_bins,
                                       test_rows & elig)}
    man_rows = np.zeros(len(pos_of), dtype=bool)
    man_rows[doc_off[:-1][d_] + p_] = True
    tri_man = {
        "unigram_auc": nl.tercile_auc(unigram, masked_bins,
                                      man_rows & test_rows),
        "position_auc": nl.tercile_auc(pos_of.astype(float), masked_bins,
                                       man_rows & test_rows)}

    out = HERE / f"eqdens_openwebmath_{key}.npz"
    np.savez_compressed(
        out, token_ids=ids_flat, doc_off=doc_off, in_math=in_math,
        mrate=mrate, mrate_bin=masked_bins, mrate_null=mrate_null,
        mrate_null_bin=null_masked, doc_split=split,
        man_mrate_doc=d_, man_mrate_pos=p_, man_mrate_cls=c_)

    fin = np.isfinite(mrate)
    stats = {
        "tokenizer": model, "n_docs": n_docs,
        "n_tokens": int(ids_flat.size),
        "tokens_per_doc_median": float(np.median(np.diff(doc_off))),
        "math_token_frac": float(in_math.mean()),
        "labeled_frac": float(fin.mean()),
        "eligible_frac": float(elig.mean()),
        "train_zero_frac": float(
            (mrate[train_rows & fin] == 0).mean()),
        "scheme": scheme, "edges": edges,
        "mrate_mean": float(mrate[fin].mean()),
        "mrate_std": float(mrate[fin].std()),
        "manifest_rows_per_class": int(len(d_) // 3),
        "triage_all_eligible_rows": tri_all,
        "triage_manifest_rows": tri_man,
        "artifact": out.name,
    }
    print(f"[{key}] {ids_flat.size:,} tok; math_frac="
          f"{in_math.mean():.3f}; scheme={scheme}; "
          f"all={json.dumps(tri_all)}; man={json.dumps(tri_man)}",
          flush=True)
    return stats


def main():
    docs, meta = pull_corpus()
    print(f"corpus: {meta['n_docs']} docs; forms="
          f"{json.dumps(meta['span_form_counts'])}", flush=True)
    stats = {"corpus": meta,
             "kernel": {"half_life": el.HALF_LIFE, "support": el.SUPPORT,
                        "mass_within_T": {T: nl.kernel_mass_within(
                            T, el.HALF_LIFE, el.SUPPORT)
                            for T in (4, 8, 16, 32, 64)}},
             "min_manifest_pos": el.MIN_POS, "per_tokenizer": {}}
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(key, model, docs)
    (HERE / "eqdens_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'eqdens_stats.json'}")


if __name__ == "__main__":
    main()
