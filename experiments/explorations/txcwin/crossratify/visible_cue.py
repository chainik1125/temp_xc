"""GAP-A of crossratify/MINI_CARD.md — the window-surface visible-cue
baseline for the txcwin trailing-novelty claims (mac-b, salvage W2).

Features are computed from `token_ids` only (plus disclosed train-doc
statistics); rows, split, probe and skill are Andrii's own `task_rows`
+ `score_task` VERBATIM, so every number is in the same units as the
dictionary cells in `focus_novresid.json` / `focus_nov_8b.json`.

Arms (see the card §2 GAP-A): V-pos, V-rep, V-uni, V-all.

CPU-only, reads committed label packs. Writes
`crossratify/results/visible_cue_<key>.json`.

Run: .venv/bin/python -m experiments.explorations.txcwin.crossratify.visible_cue
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.txcwin.sweep import (
    LABELS, _npz_key, score_task, task_rows,
)
from experiments.explorations.task_hunt.labels.novelty_lib import (
    N_POS_BINS, kernel_mass_within, kernel_weights, position_bin,
    type_mean_scores,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
STEM = "novelty_fineweb"
MODELS = {"gpt2": "gpt2",
          "llama31": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
TS = (4, 8, 16)
FIELDS = ("nov_resid", "nov_rate")
MAX_ROWS = 8000          # = the focus runs, so rows match cell-for-cell
SPLIT_SEED = 7           # score_task's own; replicated ONLY to build the
                         # token-level train mask for V-uni type means


def window_ids(ids: np.ndarray, starts: np.ndarray, T: int) -> np.ndarray:
    return ids[starts[:, None] + np.arange(T)[None, :]]


def new_in_window(W: np.ndarray) -> np.ndarray:
    """flags[i, j] = 1 iff W[i, j] is the first occurrence of that token
    type within window i (positions 0..j-1 scanned)."""
    n, T = W.shape
    flags = np.ones((n, T), dtype=np.float32)
    for j in range(1, T):
        flags[:, j] = (W[:, :j] != W[:, j:j + 1]).all(axis=1)
    return flags


def v_pos(pos_in_doc: np.ndarray) -> np.ndarray:
    pb = position_bin(pos_in_doc)
    onehot = np.zeros((len(pb), N_POS_BINS + 1), dtype=np.float32)
    onehot[np.arange(len(pb)), pb.astype(int) + 1] = 1.0   # col 0 = below-min
    return np.column_stack([np.log2(1.0 + pos_in_doc).astype(np.float32),
                            onehot])


def v_rep(W: np.ndarray) -> np.ndarray:
    n, T = W.shape
    flags = new_in_window(W)
    w = kernel_weights()                       # lags 1..64, label's own kernel
    lags = np.arange(1, T)                     # reachable lags inside window
    kw = w[lags - 1]
    f_kernel = (flags[:, T - 1 - lags] * kw[None, :]).sum(1) / kw.sum() \
        if T > 1 else np.zeros(n, dtype=np.float32)
    distinct = flags.mean(1)                   # distinct types / T
    last_new = flags[:, -1]
    # per-window repeat structure
    rep_types = np.zeros(n, dtype=np.float32)  # types appearing >= 2, / T
    max_rep = np.zeros(n, dtype=np.float32)    # max count of one type, / T
    log_gap = np.zeros(n, dtype=np.float32)    # mean log(1+gap) at repeats
    for i in range(n):
        vals, cnt = np.unique(W[i], return_counts=True)
        rep_types[i] = (cnt >= 2).sum() / T
        max_rep[i] = cnt.max() / T
        gaps = []
        last_seen: dict = {}
        for j, tok in enumerate(W[i]):
            if tok in last_seen:
                gaps.append(np.log1p(j - last_seen[tok]))
            last_seen[tok] = j
        log_gap[i] = float(np.mean(gaps)) if gaps else 0.0
    return np.column_stack([f_kernel, distinct, last_new,
                            rep_types, max_rep, log_gap]).astype(np.float32)


def v_uni(ids: np.ndarray, y_all: np.ndarray, doc_of_token: np.ndarray,
          row_docs: np.ndarray, W: np.ndarray,
          label_idx: np.ndarray) -> np.ndarray:
    """Train-doc type-mean of the label (their estimator), last token +
    window mean — the token-identity prior in comparable units."""
    uniq = np.unique(row_docs)
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(uniq)
    tr_docs = set(uniq[:max(1, int(0.8 * len(uniq)))].tolist())
    token_train = np.isin(doc_of_token, list(tr_docs))
    tmean = type_mean_scores(ids, y_all, token_train)
    return np.column_stack([tmean[label_idx],
                            tmean[W].mean(1)]).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, model_name in MODELS.items():
        npz = np.load(LABELS / f"{STEM}_{_npz_key(STEM, model_name)}.npz")
        ids = npz["token_ids"].astype(np.int64)
        doc_off = npz["doc_off"].astype(np.int64)
        doc_of_token = np.zeros(len(ids), dtype=np.int32)
        for i in range(len(doc_off) - 1):
            doc_of_token[doc_off[i]:doc_off[i + 1]] = i
        out = {"meta": {"card": "crossratify/MINI_CARD.md GAP-A",
                        "stem": STEM, "npz_key": _npz_key(STEM, model_name),
                        "model_rows_as_in": model_name, "max_rows": MAX_ROWS,
                        "Ts": list(TS), "fields": list(FIELDS)},
               "cells": []}
        for field in FIELDS:
            y_all = npz[field].astype(np.float32)
            for T in TS:
                starts, y, docs = task_rows(STEM, field, model_name, T,
                                            MAX_ROWS)
                label_idx = starts + T - 1
                W = window_ids(ids, starts, T)
                pos_in_doc = label_idx - doc_off[docs]
                arms = {
                    "V-pos": v_pos(pos_in_doc),
                    "V-rep": v_rep(W),
                    "V-uni": v_uni(ids, y_all, doc_of_token, docs, W,
                                   label_idx),
                }
                arms["V-all"] = np.column_stack(list(arms.values()))
                # V-win: window-COMPUTABLE arms jointly (no oracle
                # position) — added post-freeze on mac-local's GAP-A
                # ruling (LOG 56654864d item 3); the operative surface
                # floor for the decomposed surface-quiet reading.
                arms["V-win"] = np.column_stack([arms["V-rep"],
                                                 arms["V-uni"]])
                for arm, X in arms.items():
                    r = score_task(X, y, docs, "reg")
                    out["cells"].append({
                        "arm": arm, "field": field, "T": T,
                        "n_features": int(X.shape[1]),
                        "rows": int(len(starts)),
                        "kernel_mass_within_T": round(
                            kernel_mass_within(T), 4), **r})
                    print(f"[{key}] {field:9s} T={T:<3} {arm:6s} "
                          f"({X.shape[1]:2d}f) skill={r['skill']:+.4f} "
                          f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]",
                          flush=True)
        p = OUT / f"visible_cue_{key}.json"
        p.write_text(json.dumps(out, indent=1))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
