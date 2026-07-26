"""Day-2 dialogue-native faces (W2 bundle): `ttrend` + `dqgap` labels
on the EXISTING dialevel substrate — same committed token stream, same
caches, zero new forward passes.

    HF_TOKEN=... .venv/bin/python -m \
        experiments.explorations.task_hunt.labels.build_diafaces

Everything derives from committed arrays (`dialevel_dailydialog_<tok>
.npz`: token_ids / doc_off / turn_idx / is_boundary / doc_split — the
alignment contract holds by construction because the token stream is
REUSED, never re-tokenized). Question detection is token-level: a
vocab mask of ids whose decoded string contains "?", so the turn flag
is exactly "the turn contains a visible ? token" — the same evidence
the screen's visible floor counts.

Faces (pure logic in ``diafaces_lib.py``):

- ``ttrend`` float32 per token — kernel-weighted trailing slope of
  turn lengths (previous 5 turns, HL 2; current turn never in its own
  label); 3-class via the conditional zero_split/tercile scheme
  (expected: plain terciles — a continuous signed slope).
- ``dqgap`` float32 per token — turns since last PREVIOUS question
  turn (>= 1; NaN before the first question turn); 3-class via
  DETERMINISTIC balanced integer edges (`balanced_int_edges` — plain
  quantile terciles can empty a class on a small-integer face; the
  chosen edges + realized balance ship in the stats and the card).

Manifests: position-matched stratified balanced (`man_tt_*`,
`man_dq_*`), boundary tokens masked, MIN_POS 16 (the dialevel stated
deviation — dialogues are ~150 tokens). Label-side triage at the
SCREEN eligibility (pos >= 64): unigram / position / doc-mean-only
AUC per face, plus the within-dialogue viability count for the
BINDING wd arms. Artifacts: ``diafaces_dailydialog_<tok>.npz`` +
``diafaces_stats.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import diafaces_lib as dfl

HERE = Path(__file__).resolve().parent
SEED = 0
MIN_POS = 16           # manifest floor (dialevel's stated deviation)
TRIAGE_POS = 64        # triage quoted at the screen's eligibility
WD_MIN_DOC_ROWS = 30   # label-side wd viability (screen re-derives)

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",
}


def vocab_q_mask(model: str) -> np.ndarray:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    n = max(tok.vocab_size, max(tok.get_vocab().values()) + 1)
    texts = tok.batch_decode([[i] for i in range(n)],
                             skip_special_tokens=False)
    return np.array(["?" in t for t in texts], dtype=bool)


def _turn_key(doc_of: np.ndarray, turn_idx: np.ndarray) -> np.ndarray:
    """Globally unique (doc, turn) key, contiguous 0..n_turns-1."""
    max_t = int(turn_idx.max()) + 1
    return doc_of.astype(np.int64) * max_t + turn_idx.astype(np.int64)


def build_for_tokenizer(key: str, model: str):
    z = np.load(HERE / f"dialevel_dailydialog_{key}.npz")
    ids, off = z["token_ids"], z["doc_off"]
    turn_idx, boundary, split = z["turn_idx"], z["is_boundary"], z["doc_split"]
    n_docs = len(off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(off)])
    assert np.array_equal(split, lib.doc_split(n_docs, seed=SEED)), \
        "doc_split must be the dialevel split (same corpus, same seed)"

    is_qtok = vocab_q_mask(model)[ids]

    ttrend = np.empty(len(ids), dtype=np.float32)
    dqgap = np.empty(len(ids), dtype=np.float32)
    q_rate_turns, turns_per_doc = [], []
    for d in range(n_docs):
        s, e = off[d], off[d + 1]
        t_idx = turn_idx[s:e]
        n_turns = int(t_idx.max()) + 1
        sizes = np.bincount(t_idx, minlength=n_turns)
        has_q = np.zeros(n_turns, dtype=bool)
        np.logical_or.at(has_q, t_idx, is_qtok[s:e])
        ttrend[s:e] = dfl.trailing_turn_slope(sizes)[t_idx]
        dqgap[s:e] = dfl.turns_since_question(has_q)[t_idx]
        q_rate_turns.append(has_q.mean())
        turns_per_doc.append(n_turns)

    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1

    tt_scheme, tt_edges, tt_bins = pl.zero_split_bins(ttrend, train_rows)
    a, b = dfl.balanced_int_edges(dqgap[train_rows])
    dq_bins = dfl.int_edge_bins(dqgap, a, b)

    strata = pl.pos_strata(pos_of, min_pos=MIN_POS)
    out_npz = {"ttrend": ttrend, "ttrend_bin": tt_bins, "dqgap": dqgap,
               "dqgap_bin": dq_bins, "is_qtok": is_qtok.astype(np.int8),
               "doc_split": split}
    stats = {"tokenizer": model,
             "n_dialogues": n_docs, "n_tokens": int(ids.size),
             "turns_per_dialogue_mean": float(np.mean(turns_per_doc)),
             "q_rate_per_turn": float(np.mean(q_rate_turns)),
             "ttrend": {"scheme": tt_scheme, "edges": tt_edges},
             "dqgap": {"edges": [int(a), int(b)]},
             "faces": {}}

    for face, vals, bins in (("tt", ttrend, tt_bins),
                             ("dq", dqgap, dq_bins)):
        masked = np.where(boundary == 1, -1, bins).astype(np.int8)
        d_, p_, c_ = pl.stratified_balanced_manifest(
            masked, strata, doc_of, pos_of, seed=SEED)
        out_npz[f"man_{face}_doc"] = d_
        out_npz[f"man_{face}_pos"] = p_
        out_npz[f"man_{face}_cls"] = c_

        elig = (masked >= 0) & (pos_of >= TRIAGE_POS)
        unigram = nl.type_mean_scores(ids, vals, train_rows & elig)
        dm = np.zeros(n_docs)
        cnt = np.zeros(n_docs)
        fin = np.isfinite(vals) & elig
        np.add.at(dm, doc_of[fin], vals[fin])
        np.add.at(cnt, doc_of[fin], 1)
        dm_score = np.where(cnt > 0, dm / np.maximum(cnt, 1), np.nan)[doc_of]

        n_wd = 0
        for dd in np.unique(doc_of[elig & test_rows]):
            sel = vals[(doc_of == dd) & elig]
            if len(sel) < WD_MIN_DOC_ROWS:
                continue
            q1, q2 = np.quantile(sel, [1 / 3, 2 / 3])
            n_wd += int(q2 > q1)

        cls_balance = {int(c): int((masked[elig] == c).sum())
                       for c in (0, 1, 2)}
        stats["faces"][face] = {
            "labeled_frac": float(np.isfinite(vals).mean()),
            "manifest_rows_per_class": int(len(d_) // 3),
            "class_balance_eligible": cls_balance,
            "unigram_auc": nl.tercile_auc(unigram, masked,
                                          test_rows & elig),
            "position_auc": nl.tercile_auc(pos_of.astype(float), masked,
                                           test_rows & elig),
            "doc_mean_only_auc": nl.tercile_auc(dm_score, masked,
                                                test_rows & elig),
            "wd_viable_test_docs": n_wd,
        }

    out = HERE / f"diafaces_dailydialog_{key}.npz"
    np.savez_compressed(out, **out_npz)
    stats["artifact"] = out.name
    print(f"[{key}] q_rate/turn={stats['q_rate_per_turn']:.3f} "
          f"dq_edges={stats['dqgap']['edges']} "
          f"tt={tt_scheme}; " + json.dumps(
              {f: {k: round(v, 3) if isinstance(v, float) else v
                   for k, v in s.items() if k.endswith("auc")}
               for f, s in stats["faces"].items()}))
    return stats


def main():
    stats = {"substrate": "dialevel_dailydialog (REUSED token stream + "
                          "caches; builders labels/build_dialevel.py + "
                          "dialevel/cache_acts.py)",
             "support_turns": dfl.SUPPORT_TURNS,
             "kernel_hl": dfl.KERNEL_HL,
             "min_manifest_pos": MIN_POS, "triage_pos": TRIAGE_POS,
             "per_tokenizer": {}}
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(key, model)
    (HERE / "diafaces_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"-> {HERE / 'diafaces_stats.json'}")


if __name__ == "__main__":
    main()
