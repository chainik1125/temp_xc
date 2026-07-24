"""Refusal/deflection-marker intensity labels on WildChat (CANDIDATES.md
B7) — exact, zero-API; NEW corpus (round 2, stretch, unlocked by B6's
honest death; pre-gate PASSED at marker rate 0.147 vs the 0.02 kill
bar — `refmark_pregate.json`).

    HF_TOKEN=... .venv/bin/python -m \
        experiments.explorations.task_hunt.labels.build_refmark

Corpus: ``allenai/WildChat-1M`` (train split, PINNED revision, license
ODC-By 1.0), streamed — first ``N_STREAM`` conversations in shard
order at the pinned revision (the stated convenience-prefix
disclosure), filtered to English conversations with >=
``MIN_A_TURNS`` assistant turns (recurrence) and rendered length in
[MIN_CHARS, MAX_CHARS], seeded-subsampled to ``N_CONVS``. No
toxicity/redaction filtering — refusal-adjacent content IS the event
source (stated). **The exact sample ships as
``refmark_corpus.json.gz``** ((role, content) message lists + meta);
this builder is also the exact re-pull script. GPU economics: a NEW
token stream (~1M tokens/tokenizer) — one caching pass per model,
minutes on an H100; these are BASE models reading chat transcripts
(distribution shift is part of the candidate's framing, stated).

Render: messages (BOTH roles, no speaker tags — tags would be
maskable markers; the register difference is visible anyway) joined
by single newlines (dialevel precedent). Events = assistant messages
matching the FROZEN substring list (`refmark_lib.REFUSAL_SUBSTRINGS`,
the refusal paper's refusal_score set verbatim, committed before any
counting). Per-token arrays per tokenizer
(``refmark_wildchat_<tok>.npz``):

- ``rlam``     float32 — message-level kernel intensity λ̂ over the
  PREVIOUS 8 messages (half-life 2 — punctint geometry at message
  level; current message NEVER in its own label; NaN below message
  index 8), inherited by every token of the message — the PRIMARY;
- ``rlam_bin`` int8 — 3-class via the conditional zero_split/tercile
  scheme, with EVENT-message tokens and newline-boundary tokens
  masked to -1 (the self-stamp discipline);
- ``is_marker`` int8 — the event face (DISCLOSED regime-1 anchor,
  never the primary, never manifested);
- ``is_assistant`` int8, ``turn_idx`` int32, ``is_boundary`` int8;
- ``doc_split`` int8 per conversation (20 % test, seed 0);
- ``man_rlam_*`` — position-MATCHED balanced manifests (equal class
  counts per log2 position stratum, pos >= 32 — the fineweb floor;
  chat conversations run thousands of tokens).

Label-side triage (bars FROZEN in ``../refmark/CARD_DRAFT.md``,
committed before this ran; broad convention pinned there): current-
token type-mean AUC + position AUC (kill authority), PLUS the
newly-adopted ``doc_mean_only_auc`` disclosure statistic
(conversation-mean of λ̂ as the only feature — the document-identity
route the punctint screen surfaced; reported on all-eligible and
manifest rows, no frozen kill threshold yet per the adoption note).
Artifacts land here: ``refmark_wildchat_<tok>.npz`` +
``refmark_stats.json`` + ``refmark_corpus.json.gz``.
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
from . import refmark_lib as rl

HERE = Path(__file__).resolve().parent
SEED = 0
DATASET = "allenai/WildChat-1M"
REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
N_STREAM = 40_000
N_CONVS = 400
MIN_A_TURNS = 8
MIN_CHARS, MAX_CHARS = 2_000, 24_000
MIN_POS = 32
HALF_LIFE_M = 2         # kernel half-life in MESSAGES
SUPPORT_M = 8           # kernel support in messages

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}


def pull_corpus():
    corpus_path = HERE / "refmark_corpus.json.gz"
    if corpus_path.exists():
        payload = json.loads(gzip.decompress(corpus_path.read_bytes()))
        return payload["convs"], payload["meta"]
    import datasets
    ds = datasets.load_dataset(DATASET, split="train", revision=REVISION,
                               streaming=True)
    keep = []
    for i, ex in enumerate(ds):
        if i >= N_STREAM:
            break
        if ex.get("language") != "English":
            continue
        msgs = [(m["role"], m["content"]) for m in ex["conversation"]
                if m.get("content")]
        n_assist = sum(1 for r, _ in msgs if r == "assistant")
        if n_assist < MIN_A_TURNS:
            continue
        text, _ = dl.render_dialogue([c for _, c in msgs])
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        keep.append(msgs)
    rng = np.random.default_rng(SEED)
    if len(keep) > N_CONVS:
        idx = np.sort(rng.choice(len(keep), size=N_CONVS, replace=False))
        keep = [keep[i] for i in idx]
    meta = {"dataset": DATASET, "revision": REVISION,
            "license": "ODC-By 1.0", "split": "train",
            "stream_prefix": N_STREAM,
            "filter": {"language": "English",
                       "min_assistant_turns": MIN_A_TURNS,
                       "min_chars": MIN_CHARS, "max_chars": MAX_CHARS},
            "seed": SEED, "n_convs": len(keep),
            "frozen_list": {"repo": rl.SOURCE_REPO,
                            "commit": rl.SOURCE_COMMIT,
                            "symbol": rl.SOURCE_SYMBOL}}
    corpus_path.write_bytes(gzip.compress(json.dumps(
        {"meta": meta, "convs": keep}).encode()))
    return keep, meta


def build_for_tokenizer(key, model, convs):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    ids_flat, off = [], [0]
    rlam_all, evt_all, ast_all, midx_all, bound_all = [], [], [], [], []
    n_msgs = n_assist = n_marker = 0
    for msgs in convs:
        contents = [c for _, c in msgs]
        text, spans = dl.render_dialogue(contents)
        events = np.array([1 if (r == "assistant" and rl.is_marker_turn(c))
                           else 0 for r, c in msgs], dtype=np.int8)
        assist = np.array([1 if r == "assistant" else 0
                           for r, _ in msgs], dtype=np.int8)
        n_msgs += len(msgs)
        n_assist += int(assist.sum())
        n_marker += int(events.sum())
        lam = pl.sentence_lambda(events, half_life=HALF_LIFE_M,
                                 support=SUPPORT_M)
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        m_idx, _ = lib.sentence_index_per_token(enc["offset_mapping"],
                                                spans)
        rlam_all.append(pl.token_labels_from_sentences(
            lam, m_idx).astype(np.float32))
        evt_all.append(events[m_idx])
        ast_all.append(assist[m_idx])
        midx_all.append(m_idx)
        bound_all.append(dl.boundary_flags(enc["offset_mapping"], text))
        ids_flat.extend(enc["input_ids"])
        off.append(len(ids_flat))

    ids_flat = np.array(ids_flat, dtype=np.int32)
    doc_off = np.array(off, dtype=np.int64)
    rlam = np.concatenate(rlam_all)
    evt_tok = np.concatenate(evt_all)
    ast_tok = np.concatenate(ast_all)
    turn_idx = np.concatenate(midx_all)
    boundary = np.concatenate(bound_all)

    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = lib.doc_split(n_docs, seed=SEED)
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1

    scheme, edges, bins = pl.zero_split_bins(rlam, train_rows)
    masked_bins = np.where((evt_tok == 1) | (boundary == 1), -1,
                           bins).astype(np.int8)
    strata = pl.pos_strata(pos_of, min_pos=MIN_POS)
    d_, p_, c_ = pl.stratified_balanced_manifest(
        masked_bins, strata, doc_of, pos_of, seed=SEED)

    elig = (masked_bins >= 0) & (pos_of >= MIN_POS)
    unigram = nl.type_mean_scores(ids_flat, rlam, train_rows & elig)
    # adopted disclosure statistic: conversation-mean of the label as
    # the only feature (the doc-identity route the punctint screen
    # surfaced; no frozen kill threshold — reported, not operative)
    fin = np.isfinite(rlam)
    docmean = np.full(n_docs, np.nan)
    for d in range(n_docs):
        v = rlam[doc_off[d]: doc_off[d + 1]]
        v = v[np.isfinite(v)]
        if v.size:
            docmean[d] = v.mean()
    docmean_row = docmean[doc_of]

    def triage(mask):
        return {
            "unigram_auc": nl.tercile_auc(unigram, masked_bins, mask),
            "position_auc": nl.tercile_auc(pos_of.astype(float),
                                           masked_bins, mask),
            "doc_mean_only_auc": nl.tercile_auc(docmean_row, masked_bins,
                                                mask)}

    man_rows = np.zeros(len(pos_of), dtype=bool)
    man_rows[doc_off[:-1][d_] + p_] = True
    tri_all = triage(test_rows & elig)
    tri_man = triage(man_rows & test_rows)

    out = HERE / f"refmark_wildchat_{key}.npz"
    np.savez_compressed(
        out, token_ids=ids_flat, doc_off=doc_off, rlam=rlam,
        rlam_bin=masked_bins, is_marker=evt_tok, is_assistant=ast_tok,
        turn_idx=turn_idx, is_boundary=boundary, doc_split=split,
        man_rlam_doc=d_, man_rlam_pos=p_, man_rlam_cls=c_)

    stats = {
        "tokenizer": model, "n_convs": n_docs,
        "n_tokens": int(ids_flat.size),
        "tokens_per_conv_median": float(np.median(np.diff(doc_off))),
        "tokens_per_message_mean": float(ids_flat.size / n_msgs),
        "kernel_support_tokens_mean": float(
            SUPPORT_M * ids_flat.size / n_msgs),
        "marker_rate_assistant_msgs": n_marker / n_assist,
        "marker_rate_all_msgs": n_marker / n_msgs,
        "marker_token_frac": float(evt_tok.mean()),
        "assistant_token_frac": float(ast_tok.mean()),
        "boundary_token_frac": float(boundary.mean()),
        "labeled_frac": float(fin.mean()),
        "eligible_frac": float(elig.mean()),
        "train_zero_frac": float((rlam[train_rows & fin] == 0).mean()),
        "scheme": scheme, "edges": edges,
        "rlam_mean": float(rlam[fin].mean()),
        "rlam_std": float(rlam[fin].std()),
        "manifest_rows_per_class": int(len(d_) // 3),
        "triage_all_eligible_rows": tri_all,
        "triage_manifest_rows": tri_man,
        "artifact": out.name,
    }
    print(f"[{key}] {ids_flat.size:,} tok; marker_rate(assist)="
          f"{n_marker / n_assist:.3f}; scheme={scheme}; "
          f"all={json.dumps(tri_all)}; man={json.dumps(tri_man)}",
          flush=True)
    return stats


def main():
    convs, meta = pull_corpus()
    print(f"corpus: {meta['n_convs']} convs", flush=True)
    stats = {"corpus": meta,
             "kernel": {"half_life_msgs": HALF_LIFE_M,
                        "support_msgs": SUPPORT_M},
             "min_manifest_pos": MIN_POS, "per_tokenizer": {}}
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, convs)
    (HERE / "refmark_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'refmark_stats.json'}")


if __name__ == "__main__":
    main()
