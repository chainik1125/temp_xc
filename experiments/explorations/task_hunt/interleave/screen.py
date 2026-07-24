"""Stage-1 screen — interleaved-document `tss` (executes CARD.md).

Runs the frozen grid on the interleaved corpus cache
(`/workspace/interleave_caches/<model>`) and, as the card's mechanism
receipt, the SAME grid on the shuffled-block **null corpus**
(`<model>_null`, labels `tss_null`) — a stronger receipt than a
within-window shuffle because it destroys document coherence in the
model's input, not merely in the probe's view.

Faces: `tss` (PRIMARY, 3-class terciles) and `source` (DISCLOSED
ANCHOR, binary — a high per-token reading here is expected and does not
count against the candidate).

Row mapping, eligibility, caps, seeds and probe grid are identical to
the novelty/punctint screens so all three fineweb-batch bundles are
directly comparable. Probe stack: frozen `conversion_depth.problib`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.interleave.screen [model ...]
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.interleave.cache_acts import (
    CACHE_ROOT,
    TOK_TAG,
)
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MIN_ROWS,
    MLP_T,
    NULL_SEED,
    NULL_T,
    OFF_MIN,
    POS_MIN,
    SHUF_SEED,
    T_GRID,
    _map_rows,
    _row_lookup,
    _seeded,
    _stack,
)
from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
from experiments.explorations.task_hunt.replag.screen import (
    gather_tok,
    gather_win,
    shuffle_context,
    summarize,
    win_mean,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
RES = HERE / "results"
MEAN_TS = [4, 8, 16, 32, 64]
ANCHOR_T = 16
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")


def build_rows(key: str, null: bool = False):
    """Rows for the real (or null) corpus. The null corpus reuses the
    builder's recomputed labels `tss_null` and its own cache."""
    z = np.load(LABELS / f"interleave_fineweb_{TOK_TAG[key]}.npz")
    cdir = CACHE_ROOT / (f"{key}_null" if null else key)
    c = np.load(cdir / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    off, doc_split = z["doc_off"], z["doc_split"]
    lookup = _row_lookup(doc_idx)

    out, stats = {}, {"n_prefix": n_prefix, "content": content,
                      "null_corpus": null, "n_rows": int(ids.shape[0])}
    n = z["token_ids"].shape[0]
    doc_of = np.searchsorted(off, np.arange(n), side="right") - 1
    pos_of = np.arange(n) - off[doc_of]

    if null:
        # The null stream is the same tokens permuted by null_perm, so a
        # flat index i in the NULL corpus carries label tss_null[i]; the
        # (doc, pos) geometry is unchanged (the permutation is within
        # the corpus, and doc_off still delimits the same rows).
        faces = {"tss": z["tss_null"].astype(np.float64)}
    else:
        faces = {"tss": z["tss"].astype(np.float64),
                 "source": z["source"].astype(np.int64)}

    rows_all, cpos_all = _map_rows(doc_of, pos_of, lookup, content, n_prefix)
    base_elig = ((rows_all >= 0) & (pos_of >= POS_MIN)
                 & (pos_of % content >= OFF_MIN))

    # PRIMARY: use the builder's committed tercile edges on the train split
    edges = np.quantile(faces["tss"][base_elig & (doc_split[doc_of] == 0)
                                     & (faces["tss"] >= 0)], [1 / 3, 2 / 3])
    stats["tss_edges"] = [float(e) for e in edges]
    cls_tss = np.where(faces["tss"] >= 0,
                       np.searchsorted(edges, faces["tss"], side="right"), -1)

    specs = [("tss", cls_tss, 3, base_elig & (faces["tss"] >= 0))]
    if not null:
        specs.append(("source", faces["source"], 2,
                      base_elig & (faces["source"] >= 0)))

    for name, cls, n_cls, elig in specs:
        for split_name, flag in (("train", 0), ("test", 1)):
            m = elig & (doc_split[doc_of] == flag)
            counts = [int((m & (cls == v)).sum()) for v in range(n_cls)]
            n_take = min(min(counts), CAP[split_name])
            per, keep_r, keep_y = {}, [], []
            for v in range(n_cls):
                idx = np.flatnonzero(m & (cls == v))
                rng = _seeded(f"tss/{name}/{key}/{null}/{split_name}/{v}")
                if len(idx) > n_take:
                    idx = rng.choice(idx, n_take, replace=False)
                per[v] = len(idx)
                keep_r.append(_stack(rows_all, cpos_all, pos_of, idx))
                keep_y.append(np.full(len(idx), v, dtype=np.int64))
            stats[f"{name}/{split_name}"] = {
                "rows_per_class": per, "available": counts,
                "ok": bool(min(per.values()) >= MIN_ROWS)}
            out[(name, split_name)] = (np.concatenate(keep_r),
                                       np.concatenate(keep_y))
    return out, stats


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    hs = SCREEN_HS[key]
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs, "card": "CARD.md (frozen)",
                 "t_grid": T_GRID, "mean_ts": MEAN_TS,
                 "block_tokens_q10_50_90": [13, 47, 105]},
        "cells": {}}
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1))

    def run(k, fn):
        if k in cells:
            return
        t0 = time.time()
        cells[k] = fn()
        cells[k]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{key} {k}] " + " ".join(f"{a}={b:.3f}" for a, b in
                                         cells[k].items()
                                         if isinstance(b, float)
                                         and a != "wall_s"), flush=True)
        save()

    for null in (False, True):
        tag = "null/" if null else ""
        man, mstats = build_rows(key, null=null)
        done["meta"][f"rows_{'null' if null else 'real'}"] = mstats
        acts = torch.from_numpy(np.ascontiguousarray(np.load(
            CACHE_ROOT / (f"{key}_null" if null else key) / f"hs{hs}.npy",
            mmap_mode="r")))

        faces = ["tss"] if null else ["source", "tss"]
        for face in faces:
            if not (mstats[f"{face}/train"]["ok"]
                    and mstats[f"{face}/test"]["ok"]):
                print(f"[{key} {tag}{face}] SKIP (rows under floor: "
                      f"{mstats[f'{face}/train']['rows_per_class']})")
                continue
            n_cls = 2 if face == "source" else 3
            cw = n_cls == 2
            rtr, ytr = man[(face, "train")]
            rte, yte = man[(face, "test")]
            ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
            Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
            run(f"{tag}{face}/tok_linear", lambda: summarize(
                fit_probe(Xtr, ytr_t, Xte, yte_t, n_cls, class_weight=cw),
                n_cls))
            if face == "source":                      # anchor: cheap arm
                W1 = gather_win(acts, rtr, ANCHOR_T)
                W2 = gather_win(acts, rte, ANCHOR_T)
                run(f"{tag}{face}/T{ANCHOR_T}/win_mean_linear",
                    lambda: summarize(fit_probe(
                        win_mean(W1), ytr_t, win_mean(W2), yte_t, n_cls,
                        class_weight=cw), n_cls))
                del W1, W2, Xtr, Xte
                continue
            run(f"{tag}{face}/tok_mlp", lambda: summarize(
                fit_probe(Xtr, ytr_t, Xte, yte_t, n_cls, hidden=512), n_cls))
            run(f"{tag}{face}/position_floor", lambda: summarize(
                fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t,
                          n_cls), n_cls))
            for T in MEAN_TS:
                Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
                run(f"{tag}{face}/T{T}/win_mean_linear", lambda: summarize(
                    fit_probe(win_mean(Wtr), ytr_t, win_mean(Wte), yte_t,
                              n_cls), n_cls))
                if T in T_GRID and not null:
                    ftr = Wtr.reshape(len(rtr), -1)
                    fte = Wte.reshape(len(rte), -1)
                    run(f"{tag}{face}/T{T}/win_linear", lambda: summarize(
                        fit_probe(ftr, ytr_t, fte, yte_t, n_cls), n_cls))
                    srng = np.random.default_rng(
                        SHUF_SEED + zlib.crc32(f"{face}/T{T}".encode())
                        % 2 ** 16)
                    Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                    Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                    run(f"{tag}{face}/T{T}/win_shuf_linear",
                        lambda: summarize(fit_probe(Str, ytr_t, Ste, yte_t,
                                                    n_cls), n_cls))
                    if T in MLP_T:
                        run(f"{tag}{face}/T{T}/win_mlp", lambda: summarize(
                            fit_probe(ftr, ytr_t, fte, yte_t, n_cls,
                                      hidden=512), n_cls))
                    if T == NULL_T:
                        nrng = np.random.default_rng(NULL_SEED)
                        yn = torch.from_numpy(nrng.permutation(ytr))
                        run(f"{tag}{face}/T{T}/null_win_linear",
                            lambda: summarize(fit_probe(ftr, yn, fte, yte_t,
                                                        n_cls), n_cls))
                        run(f"{tag}{face}/null_tok_linear",
                            lambda: summarize(fit_probe(Xtr, yn, Xte, yte_t,
                                                        n_cls), n_cls))
                    del ftr, fte, Str, Ste
                del Wtr, Wte
            del Xtr, Xte
        del acts
    save()
    print(f"[{key}] DONE -> {out_path}")


def main():
    for k in (sys.argv[1:] or list(MODELS)):
        screen(k)


if __name__ == "__main__":
    main()
