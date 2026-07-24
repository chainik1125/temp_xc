"""Stage-1 screen — dialogue turn-length LEVEL `dialevel` (executes CARD.md).

Three arms, all on one cache (`/workspace/dialevel_caches/<model>`):

  PRIMARY   within-dialogue binary (min-vs-max `tlevel` inside each
            dialogue, balanced PER DIALOGUE so dialogue identity carries
            exactly zero label information). Scores every KEEP/KILL
            clause. Binary -> rank-AUC.
  REFERENCE the naive global-tercile arm, run and DISCLOSED as
            uninterpretable-as-a-window-claim: `doc_mean_only_auc` is
            0.983-0.986 here, so its gap measures document identity as
            much as trailing structure. It scores nothing; the point is
            the delta against the primary arm.
  ANCHOR    `tst` (tokens since turn start) on the SAME shipped rows,
            above vs below the train median. Same rows, different label,
            so any difference in window advantage is face-specific.

Two per-position routes survive the within-dialogue split (position AUC
0.675-0.697, `tst` AUC 0.651-0.662 -- `design_probe.py`), so the card
promotes them from footnotes to FLOORS: `position_floor`, `tst_floor`
and `postst_floor` are probed on the shipped rows and the window must
clear them, not merely beat the per-token activation probe.

Mapping, eligibility, caps, seeds and probe stack follow the
novelty/punctint/interleave screens so all five bundles are directly
comparable. Probe stack: frozen `conversion_depth.problib`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.dialevel.screen [model ...]
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
from experiments.explorations.task_hunt.dialevel.cache_acts import (
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
REF_MEAN_TS = [4, 8, 16, 32, 64]
REF_FLAT_TS = [16, 32]
ANCHOR_TS = [16, 64]
PER_DOC_CAP = 8            # rows per class per dialogue (card § 4)
MODELS = ("gpt2", "gemma2_2b", "llama31_8b")


def _stack4(rows, cpos, docpos, tst, idx):
    """Row quad (cache_row, cache_pos, doc_pos, tst). `gather_tok` /
    `gather_win` read columns 0/1 only; 2 and 3 carry the two floors
    that survive the within-dialogue control."""
    return np.stack([rows[idx], cpos[idx], docpos[idx], tst[idx]], 1)


def _pos_feats(r):
    cp, dp = r[:, 1].astype(np.float32), r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def _tst_feats(r):
    t = r[:, 3].astype(np.float32)
    f = np.stack([t, np.log2(1.0 + t), t ** 2 / 100.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def _postst_feats(r):
    return torch.cat([_pos_feats(r), _tst_feats(r)], 1)


def build_rows(key: str):
    z = np.load(LABELS / f"dialevel_dailydialog_{TOK_TAG[key]}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    off, doc_split = z["doc_off"], z["doc_split"]
    tlevel, bins = z["tlevel"], z["tlevel_bin"]
    boundary, tst = z["is_boundary"], z["tst"]
    n = z["token_ids"].shape[0]
    doc_of = np.searchsorted(off, np.arange(n), side="right") - 1
    pos_of = np.arange(n) - off[doc_of]
    lookup = _row_lookup(doc_idx)
    rows_all, cpos_all = _map_rows(doc_of, pos_of, lookup, content, n_prefix)

    elig = ((rows_all >= 0) & np.isfinite(tlevel) & (boundary == 0)
            & (pos_of >= POS_MIN) & (pos_of % content >= OFF_MIN))
    out, stats = {}, {"n_prefix": n_prefix, "content": content,
                      "n_rows": int(ids.shape[0]),
                      "eligible": int(elig.sum()),
                      "per_doc_cap": PER_DOC_CAP}

    # ---- PRIMARY: within-dialogue min-vs-max, balanced per dialogue ----
    for split_name, flag in (("train", 0), ("test", 1)):
        m = np.flatnonzero(elig & (doc_split[doc_of] == flag))
        m = m[np.argsort(doc_of[m], kind="stable")]
        d_sorted = doc_of[m]
        bnd = np.flatnonzero(np.r_[True, d_sorted[1:] != d_sorted[:-1], True])
        per_doc = []
        for a, b in zip(bnd[:-1], bnd[1:]):
            sel = m[a:b]
            v = tlevel[sel]
            uq = np.unique(v)
            if len(uq) < 2:
                continue
            lo, hi = sel[v == uq[0]], sel[v == uq[-1]]
            k = min(len(lo), len(hi), PER_DOC_CAP)
            rng = _seeded(f"dialevel/wd/{key}/{split_name}/{int(d_sorted[a])}")
            per_doc.append((rng.choice(lo, k, replace=False),
                            rng.choice(hi, k, replace=False)))
        # draw whole dialogues in seeded order until the global cap is met
        order = _seeded(f"dialevel/wd/order/{key}/{split_name}").permutation(
            len(per_doc))
        lo_keep, hi_keep, deltas, tot = [], [], [], 0
        for i in order:
            lo, hi = per_doc[i]
            take = min(len(lo), CAP[split_name] - tot)
            if take <= 0:
                break
            lo_keep.append(lo[:take])
            hi_keep.append(hi[:take])
            deltas.append(float(tlevel[hi[0]] - tlevel[lo[0]]))
            tot += take
        lo_a, hi_a = np.concatenate(lo_keep), np.concatenate(hi_keep)
        idx = np.concatenate([lo_a, hi_a])
        y = np.concatenate([np.zeros(len(lo_a), np.int64),
                            np.ones(len(hi_a), np.int64)])
        stats[f"wd/{split_name}"] = {
            "n_per_class": int(tot), "n_docs_used": len(deltas),
            "docs_available": int(len(per_doc)),
            "delta_tlevel_median_shipped": float(np.median(deltas)),
            "ok": bool(tot >= MIN_ROWS)}
        out[("wd", split_name)] = (
            _stack4(rows_all, cpos_all, pos_of, tst, idx), y)

    # anchor: tst above/below the TRAIN median of the shipped rows
    thr = float(np.median(out[("wd", "train")][0][:, 3]))
    stats["anchor_tst_median"] = thr
    for split_name in ("train", "test"):
        r = out[("wd", split_name)][0]
        ya = (r[:, 3] > thr).astype(np.int64)
        stats[f"anchor/{split_name}"] = {
            "n_per_class": [int((ya == 0).sum()), int((ya == 1).sum())],
            "ok": bool(min((ya == 0).sum(), (ya == 1).sum()) >= MIN_ROWS)}
        out[("anchor", split_name)] = (r, ya)

    # ---- REFERENCE: naive global terciles (disclosed, scores nothing) --
    for split_name, flag in (("train", 0), ("test", 1)):
        m = elig & (doc_split[doc_of] == flag)
        counts = [int((m & (bins == v)).sum()) for v in range(3)]
        n_take = min(min(counts), CAP[split_name])
        keep_r, keep_y = [], []
        for v in range(3):
            i_v = np.flatnonzero(m & (bins == v))
            rng = _seeded(f"dialevel/ref/{key}/{split_name}/{v}")
            if len(i_v) > n_take:
                i_v = rng.choice(i_v, n_take, replace=False)
            keep_r.append(_stack4(rows_all, cpos_all, pos_of, tst, i_v))
            keep_y.append(np.full(len(i_v), v, dtype=np.int64))
        stats[f"ref/{split_name}"] = {"n_per_class": int(n_take),
                                      "available": counts,
                                      "ok": bool(n_take >= MIN_ROWS)}
        out[("ref", split_name)] = (np.concatenate(keep_r),
                                    np.concatenate(keep_y))
    return out, stats


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    hs = SCREEN_HS[key]
    man, mstats = build_rows(key)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs, "card": "CARD.md (frozen)",
                 "t_grid": T_GRID, "mean_ts": MEAN_TS,
                 "turn_tokens_mean": {"gpt2": 15.54, "gemma2_2b": 15.72,
                                      "llama31_8b": 14.47}[key],
                 "support_tokens": {"gpt2": 77.7, "gemma2_2b": 78.6,
                                    "llama31_8b": 72.3}[key]},
        "cells": {}}
    done["meta"]["rows"] = mstats
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

    acts = torch.from_numpy(np.ascontiguousarray(np.load(
        CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))

    # ---------------- PRIMARY: within-dialogue -------------------------
    if not (mstats["wd/train"]["ok"] and mstats["wd/test"]["ok"]):
        print(f"[{key} wd] SKIP (rows under floor)")
    else:
        rtr, ytr = man[("wd", "train")]
        rte, yte = man[("wd", "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run("wd/tok_linear", lambda: summarize(fit_probe(
            Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
        run("wd/tok_mlp", lambda: summarize(fit_probe(
            Xtr, ytr_t, Xte, yte_t, 2, hidden=512, class_weight=True), 2))
        for nm, fx in (("position_floor", _pos_feats),
                       ("tst_floor", _tst_feats),
                       ("postst_floor", _postst_feats)):
            run(f"wd/{nm}", lambda fx=fx: summarize(fit_probe(
                fx(rtr), ytr_t, fx(rte), yte_t, 2, class_weight=True), 2))
        for T in MEAN_TS:
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            run(f"wd/T{T}/win_mean_linear", lambda: summarize(fit_probe(
                win_mean(Wtr), ytr_t, win_mean(Wte), yte_t, 2,
                class_weight=True), 2))
            if T in T_GRID:
                ftr = Wtr.reshape(len(rtr), -1)
                fte = Wte.reshape(len(rte), -1)
                run(f"wd/T{T}/win_linear", lambda: summarize(fit_probe(
                    ftr, ytr_t, fte, yte_t, 2, class_weight=True), 2))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"wd/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"wd/T{T}/win_shuf_linear", lambda: summarize(fit_probe(
                    Str, ytr_t, Ste, yte_t, 2, class_weight=True), 2))
                if T in MLP_T:
                    run(f"wd/T{T}/win_mlp", lambda: summarize(fit_probe(
                        ftr, ytr_t, fte, yte_t, 2, hidden=512,
                        class_weight=True), 2))
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"wd/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(ftr, yn, fte, yte_t, 2,
                                  class_weight=True), 2))
                    run("wd/null_tok_linear", lambda: summarize(fit_probe(
                        Xtr, yn, Xte, yte_t, 2, class_weight=True), 2))
                del ftr, fte, Str, Ste
            del Wtr, Wte
        del Xtr, Xte

        # ---------- ANCHOR: tst on the SAME rows -----------------------
        if mstats["anchor/train"]["ok"] and mstats["anchor/test"]["ok"]:
            atr, ate = man[("anchor", "train")][1], man[("anchor", "test")][1]
            atr_t, ate_t = torch.from_numpy(atr), torch.from_numpy(ate)
            Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
            run("anchor/tok_linear", lambda: summarize(fit_probe(
                Xtr, atr_t, Xte, ate_t, 2, class_weight=True), 2))
            del Xtr, Xte
            for T in ANCHOR_TS:
                Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
                run(f"anchor/T{T}/win_mean_linear", lambda: summarize(
                    fit_probe(win_mean(Wtr), atr_t, win_mean(Wte), ate_t, 2,
                              class_weight=True), 2))
                del Wtr, Wte

    # ---------------- REFERENCE: naive global arm ----------------------
    if mstats["ref/train"]["ok"] and mstats["ref/test"]["ok"]:
        rtr, ytr = man[("ref", "train")]
        rte, yte = man[("ref", "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run("ref/tok_linear", lambda: summarize(fit_probe(
            Xtr, ytr_t, Xte, yte_t, 3), 3))
        run("ref/position_floor", lambda: summarize(fit_probe(
            _pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, 3), 3))
        run("ref/postst_floor", lambda: summarize(fit_probe(
            _postst_feats(rtr), ytr_t, _postst_feats(rte), yte_t, 3), 3))
        del Xtr, Xte
        for T in REF_MEAN_TS:
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            run(f"ref/T{T}/win_mean_linear", lambda: summarize(fit_probe(
                win_mean(Wtr), ytr_t, win_mean(Wte), yte_t, 3), 3))
            if T in REF_FLAT_TS:
                run(f"ref/T{T}/win_linear", lambda: summarize(fit_probe(
                    Wtr.reshape(len(rtr), -1), ytr_t,
                    Wte.reshape(len(rte), -1), yte_t, 3), 3))
            del Wtr, Wte

    del acts
    save()
    print(f"[{key}] DONE -> {out_path}")


def main():
    for k in (sys.argv[1:] or list(MODELS)):
        screen(k)


if __name__ == "__main__":
    main()
