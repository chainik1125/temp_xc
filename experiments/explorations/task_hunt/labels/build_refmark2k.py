"""Refusal/deflection-marker intensity labels at 5x scale (corpus-scaleup
item 2): the SAME frozen label logic as ``build_refmark.py``, on the
2,000-conversation WildChat pull instead of the shipped 400.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_refmark2k

``build_refmark.py``, ``refmark_lib.py`` (the frozen 12-substring event
list) and the kernel geometry are NOT touched — this builder imports them
unchanged and writes NEW versioned artifacts. What is new, and none of it
is label logic:

1. **``is_user_echo`` ships as an explicit array.** Marker masking covers
   ASSISTANT messages only (they are the events), so a USER message that
   quotes a frozen substring — "why do you keep saying I cannot" — stays
   unmasked and manifest-eligible. mac-local measured that exposure on
   the shipped build at 13/4,713 user messages => 134/59,994 manifest
   rows (0.22 %). Here it is recomputed at scale AND shipped as a
   per-token mask so a screen can drop those rows in one line. It is a
   DISCLOSURE array: it changes no label, no mask, no manifest.
2. **Doc-level bootstrap CIs (>= 1,000 reps)** on every triage AUC and on
   ``doc_mean_only_auc`` (``boot_lib``) — the conversation is the cluster.
3. **Recurrence and within-conversation-contrast census at scale**: the
   fraction of conversations carrying >= 2 marker messages (0.377 on the
   >= 8-turn pre-gate population), and how many conversations hold
   manifest rows of BOTH the top and bottom class — the control the
   shipped card made BINDING when ``doc_mean_only_auc`` came back 0.967.
4. Manifest cap 20k -> 100k rows/class, with the uncapped
   position-matched ceiling reported alongside.

Bars stay frozen (``../refmark/CARD_DRAFT.md``): direction-agnostic
max(AUC, 1-AUC), current-token type-mean AUC >= 0.65 => KILL, position
AUC >= 0.65 => KILL, manifest rows operative, 0.55-0.65 ships with
disclosure. A bar firing at scale is a FINDING that binds the Stage-2
design; it does not retro-kill the shipped 400-conversation bundle.

Artifacts: ``refmark2k_wildchat_<tok>.npz`` + ``refmark2k_stats.json``.
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
from . import dialevel_lib as dl
from . import refmark_lib as rl
from . import pull_refmark2k as pull
# same census definitions as item 1, imported rather than restated
from .build_punctint4k import supported_rows_per_class, within_doc_contrast

HERE = Path(__file__).resolve().parent
SEED = 0
MIN_POS = 32
HALF_LIFE_M = 2         # kernel half-life in MESSAGES (frozen)
SUPPORT_M = 8           # kernel support in messages (frozen)
MANIFEST_CAP = 100_000

TOKENIZERS = {
    "gpt2": "gpt2",
    "gemma2": "google/gemma-2-2b",
    "llama31": "NousResearch/Meta-Llama-3.1-8B",   # = Ward stream tokenizer
}


def corpus_level_stats(convs) -> dict:
    """Message-level receipts that do not depend on a tokenizer:
    marker rate, recurrence, and the user-echo exposure."""
    n_msgs = n_assist = n_marker = n_user = n_user_echo = 0
    per_conv_markers = []
    for msgs in convs:
        k = 0
        for role, content in msgs:
            n_msgs += 1
            if role == "assistant":
                n_assist += 1
                if rl.is_marker_turn(content):
                    n_marker += 1
                    k += 1
            else:
                n_user += 1
                if rl.is_marker_turn(content):
                    n_user_echo += 1
        per_conv_markers.append(k)
    m = np.array(per_conv_markers)
    return {
        "n_convs": len(convs), "n_messages": n_msgs,
        "n_assistant_messages": n_assist, "n_user_messages": n_user,
        "n_marker_messages": n_marker,
        "marker_rate_assistant_msgs": float(n_marker / max(n_assist, 1)),
        "recurrence_frac_convs_ge2_markers": float((m >= 2).mean()),
        "frac_convs_ge1_marker": float((m >= 1).mean()),
        "markers_per_conv_mean": float(m.mean()),
        "markers_per_conv_max": int(m.max()) if m.size else 0,
        "user_echo_messages": n_user_echo,
        "user_echo_frac_of_user_messages": float(
            n_user_echo / max(n_user, 1)),
    }


def build_for_tokenizer(key, model, convs, n_reps, out_dir, tag):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    t0 = time.time()

    id_docs, off = [], [0]
    rlam_all, evt_all, ast_all, midx_all, bound_all, echo_all = (
        [], [], [], [], [], [])
    n_msgs = n_assist = n_marker = 0
    for msgs in convs:
        contents = [c for _, c in msgs]
        text, spans = dl.render_dialogue(contents)
        events = np.array([1 if (r == "assistant" and rl.is_marker_turn(c))
                           else 0 for r, c in msgs], dtype=np.int8)
        assist = np.array([1 if r == "assistant" else 0
                           for r, _ in msgs], dtype=np.int8)
        # DISCLOSURE face only: user messages quoting a frozen substring.
        # They are not events (events are assistant-only) and so are not
        # masked — this array is what lets a screen drop them.
        echo = np.array([1 if (r != "assistant" and rl.is_marker_turn(c))
                         else 0 for r, c in msgs], dtype=np.int8)
        n_msgs += len(msgs)
        n_assist += int(assist.sum())
        n_marker += int(events.sum())
        lam = pl.sentence_lambda(events, half_life=HALF_LIFE_M,
                                 support=SUPPORT_M)
        enc = tok(text, add_special_tokens=False,
                  return_offsets_mapping=True)
        m_idx, _ = lib.sentence_index_per_token(enc["offset_mapping"], spans)
        rlam_all.append(pl.token_labels_from_sentences(
            lam, m_idx).astype(np.float32))
        evt_all.append(events[m_idx])
        ast_all.append(assist[m_idx])
        echo_all.append(echo[m_idx])
        midx_all.append(m_idx)
        bound_all.append(dl.boundary_flags(enc["offset_mapping"], text))
        ids = np.asarray(enc["input_ids"], dtype=np.int32)
        id_docs.append(ids)
        off.append(off[-1] + ids.size)

    ids_flat = np.concatenate(id_docs)
    doc_off = np.array(off, dtype=np.int64)
    rlam = np.concatenate(rlam_all)
    evt_tok = np.concatenate(evt_all)
    ast_tok = np.concatenate(ast_all)
    echo_tok = np.concatenate(echo_all)
    turn_idx = np.concatenate(midx_all)
    boundary = np.concatenate(bound_all)
    print(f"[{key}] tokenized {len(convs)} convs -> {ids_flat.size:,} "
          f"tokens in {time.time() - t0:.0f}s", flush=True)

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
        masked_bins, strata, doc_of, pos_of, cap=MANIFEST_CAP, seed=SEED)

    elig = (masked_bins >= 0) & (pos_of >= MIN_POS)
    unigram = nl.type_mean_scores(ids_flat, rlam, train_rows & elig)
    fin = np.isfinite(rlam)
    docmean = np.full(n_docs, np.nan)
    for d in range(n_docs):
        seg = rlam[doc_off[d]: doc_off[d + 1]]
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
            b = boot[rname][s]
            print(f"[{key}] boot {rname}.{s}: {b['point']:.4f} "
                  f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}] "
                  f"({b['n_rows']:,} rows, {time.time() - tb:.0f}s)",
                  flush=True)

    # user-echo exposure on the rows that actually ship
    echo_stats = {
        "user_echo_token_frac": float(echo_tok.mean()),
        "user_echo_rows_all_eligible": int((echo_tok[test_rows & elig] == 1
                                            ).sum()),
        "all_eligible_rows": int((test_rows & elig).sum()),
        "user_echo_rows_manifest": int((echo_tok[man_rows] == 1).sum()),
        "manifest_rows": int(man_rows.sum()),
    }
    echo_stats["user_echo_frac_manifest_rows"] = float(
        echo_stats["user_echo_rows_manifest"]
        / max(echo_stats["manifest_rows"], 1))

    out = out_dir / f"refmark{tag}_wildchat_{key}.npz"
    np.savez_compressed(
        out, token_ids=ids_flat, doc_off=doc_off, rlam=rlam,
        rlam_bin=masked_bins, is_marker=evt_tok, is_assistant=ast_tok,
        is_user_echo=echo_tok, turn_idx=turn_idx, is_boundary=boundary,
        doc_split=split, man_rlam_doc=d_, man_rlam_pos=p_, man_rlam_cls=c_)

    stats = {
        "tokenizer": model, "n_convs": n_docs,
        "n_tokens": int(ids_flat.size),
        "tokens_per_conv_median": float(np.median(np.diff(doc_off))),
        "tokens_per_message_mean": float(ids_flat.size / n_msgs),
        "kernel_support_tokens_mean": float(
            SUPPORT_M * ids_flat.size / n_msgs),
        "marker_rate_assistant_msgs": n_marker / n_assist,
        "marker_token_frac": float(evt_tok.mean()),
        "assistant_token_frac": float(ast_tok.mean()),
        "boundary_token_frac": float(boundary.mean()),
        "labeled_frac": float(fin.mean()),
        "eligible_frac": float(elig.mean()),
        "train_zero_frac": float((rlam[train_rows & fin] == 0).mean()),
        "scheme": scheme, "edges": edges,
        "rlam_mean": float(rlam[fin].mean()),
        "rlam_std": float(rlam[fin].std()),
        "manifest_cap_per_class": MANIFEST_CAP,
        "manifest_rows_per_class": int(len(d_) // 3),
        "manifest_rows_per_class_supported": supported_rows_per_class(
            masked_bins, strata),
        "within_conversation_contrast": within_doc_contrast(d_, c_, split),
        "user_echo_exposure": echo_stats,
        "triage_all_eligible_rows": tri["triage_all_eligible_rows"],
        "triage_manifest_rows": tri["triage_manifest_rows"],
        "bootstrap": boot,
        "artifact": out.name,
    }
    print(f"[{key}] wrote {out.name} "
          f"({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s",
          flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-convs", type=int, default=None,
                    help="limit conversations (smoke runs only)")
    ap.add_argument("--reps", type=int, default=bo.N_REPS)
    ap.add_argument("--tag", default="2k")
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convs, meta = pull.load()
    if a.n_convs:
        convs = convs[: a.n_convs]
    print(f"corpus: {len(convs)} convs (pool "
          f"{meta.get('pool_size_pre_subsample')})", flush=True)
    stats = {
        "corpus": meta, "n_convs_used": len(convs),
        "frozen_logic": "refmark_lib (frozen 12-substring list) + "
                        "punctint_lib kernel + dialevel_lib render — "
                        "unchanged from the shipped 400-conv build",
        "kernel": {"half_life_msgs": HALF_LIFE_M, "support_msgs": SUPPORT_M},
        "min_manifest_pos": MIN_POS,
        "manifest_cap_per_class": MANIFEST_CAP,
        "corpus_level": corpus_level_stats(convs),
        "bootstrap": {"unit": "conversation (cluster)", "n_reps": a.reps,
                      "ci_pct": list(bo.CI_PCT), "seed": SEED},
        "per_tokenizer": {},
    }
    print(json.dumps(stats["corpus_level"], indent=1), flush=True)
    for key, model in TOKENIZERS.items():
        stats["per_tokenizer"][key] = build_for_tokenizer(
            key, model, convs, a.reps, out_dir, a.tag)
    p = out_dir / f"refmark{a.tag}_stats.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
