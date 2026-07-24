"""Dialogue turn-length LEVEL labels (CANDIDATES.md B5) — exact,
zero-API; NEW corpus (the one non-fineweb bundle of this batch).

    HF_TOKEN=... .venv/bin/python -m \
        experiments.explorations.task_hunt.labels.build_dialevel

Corpus: DailyDialog via the parquet mirror ``OpenRL/daily_dialog``
(train split, PINNED revision — the canonical
``li2017dailydialog/daily_dialog`` uses a legacy loading script that
``datasets`` 4.x refuses; original corpus license CC BY-NC-SA 4.0,
research use). Filter: dialogues with >= 8 turns (48 %), seeded
subsample to N_DIALOGUES. **The exact sampled corpus ships as
``dialevel_corpus.json.gz``** (turn texts + mirror id + revision +
filter + seed), so consumers never re-pull; this builder is also the
exact re-pull script. GPU economics: a NEW token stream — one caching
pass per model (~0.5M tokens, minutes on an H100) — say so wherever
the screen is planned.

Render: turns joined by single newlines (the minimal visible boundary
marker). Per-token arrays per tokenizer
(``dialevel_dailydialog_<tok>.npz``):

- ``tlevel``   float32 — trailing mean turn length in TOKENS over the
  previous 5 turns (current turn NEVER in its own label; NaN while
  fewer than 5 previous turns exist) — the PRIMARY level face;
- ``tlevel_bin`` int8 — 3-class via the conditional zero_split/tercile
  scheme (expected: plain terciles — turn lengths are positive);
- ``tst``      int32 — tokens since turn start (DISCLOSED
  conversion-risky clock face, secondary only);
- ``turn_idx`` int32, ``is_boundary`` int8 (newline-spanning tokens —
  the marker face, MASKED from manifests);
- ``doc_split`` int8 per dialogue (20 % test, seed 0);
- ``man_tlevel_*`` — position-MATCHED balanced manifests
  (equal class counts per log2 position stratum — the guard shipped
  with B3/B4), boundary tokens masked, pos >= 16 (dialogues are ~200
  tokens; the fineweb pos >= 32 floor would gut coverage — a stated
  deviation; T <= 16 windows fit fully, T = 32 windows at pos 16
  truncate and the screen must left-pad or drop, stated here).

Label-side triage (bars FROZEN in ``../dialevel/CARD_DRAFT.md``
committed before this ran): current-token type-mean AUC + position
AUC (direction-agnostic), all-eligible and manifest rows.
Artifacts: ``dialevel_dailydialog_<tok>.npz`` +
``dialevel_stats.json`` + ``dialevel_corpus.json.gz`` here.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import dialevel_lib as dl

HERE = Path(__file__).resolve().parent
SEED = 0
MIRROR = "OpenRL/daily_dialog"
REVISION = "1668faf0c0dc44664f108c489fd0666128db2c48"
MIN_TURNS = 8
N_DIALOGUES = 5000
MIN_POS = 16          # stated deviation from the fineweb pos >= 32 floor

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}


def pull_corpus():
    corpus_path = HERE / "dialevel_corpus.json.gz"
    if corpus_path.exists():
        payload = json.loads(gzip.decompress(corpus_path.read_bytes()))
        return payload["dialogues"], payload["meta"]
    import datasets
    ds = datasets.load_dataset(MIRROR, split="train", revision=REVISION)
    dialogs = [[t.strip() for t in d] for d in ds["dialog"]]
    keep = [d for d in dialogs if len(d) >= MIN_TURNS]
    rng = np.random.default_rng(SEED)
    if len(keep) > N_DIALOGUES:
        idx = np.sort(rng.choice(len(keep), size=N_DIALOGUES, replace=False))
        keep = [keep[i] for i in idx]
    meta = {"mirror": MIRROR, "revision": REVISION,
            "canonical": "li2017dailydialog/daily_dialog",
            "license": "CC BY-NC-SA 4.0 (original DailyDialog)",
            "split": "train", "min_turns": MIN_TURNS, "seed": SEED,
            "n_dialogues": len(keep)}
    corpus_path.write_bytes(gzip.compress(json.dumps(
        {"meta": meta, "dialogues": keep}).encode()))
    return keep, meta


def build_for_tokenizer(key, model, dialogues):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    ids_flat, off = [], [0]
    tlevel_all, turn_all, bound_all = [], [], []
    for turns in dialogues:
        text, spans = dl.render_dialogue(turns)
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        ids = enc["input_ids"]
        t_idx, _ = lib.sentence_index_per_token(enc["offset_mapping"],
                                                spans)
        sizes = np.bincount(t_idx, minlength=len(turns))
        lev = dl.trailing_turn_mean(sizes)
        tlevel_all.append(pl.token_labels_from_sentences(
            lev, t_idx).astype(np.float32))
        turn_all.append(t_idx)
        bound_all.append(dl.boundary_flags(enc["offset_mapping"], text))
        ids_flat.extend(ids)
        off.append(len(ids_flat))

    ids_flat = np.array(ids_flat, dtype=np.int32)
    doc_off = np.array(off, dtype=np.int64)
    tlevel = np.concatenate(tlevel_all)
    turn_idx = np.concatenate(turn_all)
    boundary = np.concatenate(bound_all)
    tst = np.concatenate([dl.tokens_since_turn_start(t) for t in turn_all])

    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = lib.doc_split(n_docs, seed=SEED)
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1

    scheme, edges, bins = pl.zero_split_bins(tlevel, train_rows)
    masked_bins = np.where(boundary == 1, -1, bins).astype(np.int8)
    strata = pl.pos_strata(pos_of, min_pos=MIN_POS)
    d_, p_, c_ = pl.stratified_balanced_manifest(
        masked_bins, strata, doc_of, pos_of, seed=SEED)

    elig = (masked_bins >= 0) & (pos_of >= MIN_POS)
    unigram = nl.type_mean_scores(ids_flat, tlevel, train_rows & elig)
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

    out = HERE / f"dialevel_dailydialog_{key}.npz"
    np.savez_compressed(
        out, token_ids=ids_flat, doc_off=doc_off, tlevel=tlevel,
        tlevel_bin=bins, tst=tst, turn_idx=turn_idx, is_boundary=boundary,
        doc_split=split, man_tlevel_doc=d_, man_tlevel_pos=p_,
        man_tlevel_cls=c_)

    fin = np.isfinite(tlevel)
    stats = {
        "tokenizer": model, "n_dialogues": n_docs,
        "n_tokens": int(ids_flat.size),
        "tokens_per_dialogue_median": float(np.median(np.diff(doc_off))),
        "tokens_per_turn_mean": float(np.diff(doc_off).sum()
                                      / (turn_idx.max() + 1 if n_docs == 1
                                         else sum(t.max() + 1
                                                  for t in turn_all))),
        "labeled_frac": float(fin.mean()),
        "scheme": scheme, "edges": edges,
        "tlevel_mean": float(tlevel[fin].mean()),
        "tlevel_std": float(tlevel[fin].std()),
        "boundary_token_frac": float(boundary.mean()),
        "manifest_rows_per_class": int(len(d_) // 3),
        "triage_all_eligible_rows": tri_all,
        "triage_manifest_rows": tri_man,
        "artifact": out.name,
    }
    print(f"[{key}] {ids_flat.size:,} tok; scheme={scheme}; "
          f"all={json.dumps(tri_all)}; man={json.dumps(tri_man)}")
    return stats


def main():
    dialogues, meta = pull_corpus()
    stats = {"corpus": meta, "support_turns": dl.SUPPORT_TURNS,
             "min_manifest_pos": MIN_POS, "per_tokenizer": {}}
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, dialogues)
    (HERE / "dialevel_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'dialevel_stats.json'}")


if __name__ == "__main__":
    main()
