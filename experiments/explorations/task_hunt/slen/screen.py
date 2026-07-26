"""Stage-1 screen — B8 sentence-length recency ladder (executes CARD.md).

Three faces of ONE value stream (x = ln sentence word count), differing
only in temporal weighting, from `../labels/slen400_fineweb_<tok>.npz`
(the cache-aligned 400-doc variant: token-IDENTICAL to the replag
fineweb caches — zero new caching beyond building those caches). The
pre-registered deliverable is the within-window shuffle ladder
**lat > lev > disp ≈ 0** (card § 6, P3).

Probe grid is the post-withdrawal convention-of-record (LOG 2026-07-24,
runpod-e recommendation, adopted by the R17 re-quote): **fix the probe
class, control width** —

  per-token linear + MLP(512) FIRST (per-token-first triage; the cell
      order is auditable in the incremental results file)
  position-only floor probe on the shipped rows
  `anchor ⊕ context-mean` (actxmean) linear + MLP at T ∈ {4,8,16,32,64},
      each with its width-matched foreign null (true anchor, context
      mean from a DIFFERENT row) printed beside — the order-free arm
  ORDER arms at T ∈ {4,8,16,32}: window-flatten linear, context-SHUFFLED
      linear (anchor slot fixed, seeded), and foreign-context flatten
      (width null for the T·d arms); MLP triple at T ∈ {16,32}
  permutation nulls (NULL_SEED 99) on the tok/flatten pair at T = 16
  WITHIN-DOCUMENT control (binding for `lev`, run for all faces):
      classes = within-doc terciles of the face value, binary rank-AUC;
      tok + actxmean(±foreign) at T ∈ {16,32,64}, flatten/shuffle pair
      at T ∈ {16,32}

`win_mean` is deliberately absent (anchor-dilution artifact, LOG
2026-07-24); NO max-over-arms anywhere — every comparison is
matched-probe-class, every window number beside its width null.

Rows: mapping/eligibility/caps/seeds identical to the novelty/punctint
screens (uniform eligibility, every screened T ≤ 64 reads identical
rows). Metric: faces 3-class acc_test (chance 1/3) + per_class; wd
arms binary rank-AUC (class_weight=True). Incremental/resumable.

Run: .venv/bin/python -m experiments.explorations.task_hunt.slen.screen [model ...]
Writes results/screen_<model>.json next to this file.
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
from experiments.explorations.task_hunt.dialevel.actxmean_null import (
    actxmean_foreign,
)
from experiments.explorations.task_hunt.dialevel.capacity_check import (
    FOREIGN_SEED,
    anchor_ctxmean,
    foreign_context,
)
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MIN_ROWS,
    MODELS,
    NULL_SEED,
    OFF_MIN,
    POS_MIN,
    SHUF_SEED,
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
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
CACHE_ROOT = Path("/workspace/replag_caches")
RES = HERE / "results"

FACES = ("lat", "lev", "disp")
AX_TS = [4, 8, 16, 32, 64]        # actxmean (+ its foreign null): 2d arms
ORD_TS = [4, 8, 16, 32]           # flatten / shuffle / foreign-flat: T·d arms
MLP_T = [16, 32]                  # MLP triple on the order arms
NULL_T = 16
WD_TS = [16, 32, 64]              # within-doc actxmean ladder
WD_ORD_TS = [16, 32]              # within-doc order pair
WD_MIN_DOC_ROWS = 30              # within_doc.py precedent


def build_rows(key: str):
    """{(face|face+'_wd', split): (rows(n,3), y)} + stats."""
    tag = MODELS[key]
    z = np.load(LABELS / f"slen400_fineweb_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    flat, off, doc_split = z["token_ids"], z["doc_off"], z["doc_split"]

    # Re-assert the zero-new-caching identity at run time (the bundle's
    # prefix receipt made the claim; the screen re-checks it).
    lookup = _row_lookup(doc_idx)
    for (d, k), i in list(lookup.items())[:200]:
        s = off[d] + k * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"flat/window mismatch at doc {d} chunk {k}"

    out: dict = {}
    stats = {"n_prefix": n_prefix, "content": content,
             "n_cache_rows": int(ids.shape[0])}

    # ---- PRIMARY: the builder's balanced position-matched manifests ----
    for face in FACES:
        md, mp = z[f"man_{face}_doc"], z[f"man_{face}_pos"]
        mc = z[f"man_{face}_cls"]
        rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
        elig = (rows_all >= 0) & (mp >= POS_MIN) & (mp % content >= OFF_MIN)
        for split_name, flag in (("train", 0), ("test", 1)):
            m = elig & (doc_split[md] == flag)
            per, keep_r, keep_y = {}, [], []
            for cls in (0, 1, 2):
                idx = np.flatnonzero(m & (mc == cls))
                rng = _seeded(f"slen/{face}/{key}/{split_name}/{cls}")
                if len(idx) > CAP[split_name]:
                    idx = rng.choice(idx, CAP[split_name], replace=False)
                per[cls] = len(idx)
                keep_r.append(_stack(rows_all, cpos_all, mp, idx))
                keep_y.append(np.full(len(idx), cls, dtype=np.int64))
            stats[f"{face}/{split_name}"] = {
                "rows_per_class": per,
                "ok": bool(min(per.values()) >= MIN_ROWS)}
            out[(face, split_name)] = (np.concatenate(keep_r),
                                       np.concatenate(keep_y))

    # ---- WITHIN-DOC control rows (within_doc.py recipe, binary) ----
    n_tok = flat.shape[0]
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    rows_all, cpos_all = _map_rows(doc_of, pos_of, lookup, content, n_prefix)
    for face in FACES:
        val = z[f"val_{face}"].astype(np.float64)
        elig = ((rows_all >= 0) & (pos_of >= POS_MIN)
                & (pos_of % content >= OFF_MIN) & np.isfinite(val))
        for split_name, flag in (("train", 0), ("test", 1)):
            m = elig & (doc_split[doc_of] == flag)
            lo_idx, hi_idx, n_docs_used = [], [], 0
            for d in np.unique(doc_of[m]):
                sel = np.flatnonzero(m & (doc_of == d))
                v = val[sel]
                if len(sel) < WD_MIN_DOC_ROWS:
                    continue
                q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
                if not (q2 > q1):              # no within-doc variation
                    continue
                lo_idx.append(sel[v <= q1])
                hi_idx.append(sel[v >= q2])
                n_docs_used += 1
            if not lo_idx:
                stats[f"{face}_wd/{split_name}"] = {"ok": False, "n_docs": 0}
                continue
            lo, hi = np.concatenate(lo_idx), np.concatenate(hi_idx)
            n_take = min(len(lo), len(hi), CAP[split_name] * 2)
            keep_r, keep_y = [], []
            for v_, idx in ((0, lo), (1, hi)):
                rng = _seeded(f"slenwd/{face}/{key}/{split_name}/{v_}")
                if len(idx) > n_take:
                    idx = rng.choice(idx, n_take, replace=False)
                keep_r.append(_stack(rows_all, cpos_all, pos_of, idx))
                keep_y.append(np.full(len(idx), v_, dtype=np.int64))
            stats[f"{face}_wd/{split_name}"] = {
                "ok": bool(n_take >= MIN_ROWS), "n_per_class": int(n_take),
                "n_docs": int(n_docs_used)}
            out[(f"{face}_wd", split_name)] = (np.concatenate(keep_r),
                                              np.concatenate(keep_y))
    return out, stats


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)            # in-chunk position
    dp = r[:, 2].astype(np.float32)            # doc position
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats = build_rows(key)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs, "card": "CARD.md (frozen)",
                 "ax_ts": AX_TS, "ord_ts": ORD_TS, "mlp_t": MLP_T,
                 "null_t": NULL_T, "wd_ts": WD_TS, "wd_ord_ts": WD_ORD_TS,
                 "foreign_seed": FOREIGN_SEED, "rows": mstats,
                 "tokens_per_sentence": "21.1-21.6 (punctint card § 3, "
                                        "same corpus)",
                 "kernel_mass_by_sentence": {"1": 0.312, "2": 0.533,
                                             "4": 0.8, "8": 1.0}},
        "cells": {}}
    cells = done["cells"]

    def save():
        out_path.write_text(json.dumps(done, indent=1))

    def run(cell_key, fn):
        if cell_key in cells:
            return
        t0 = time.time()
        cells[cell_key] = fn()
        cells[cell_key]["wall_s"] = round(time.time() - t0, 1)
        print(f"[{key} {cell_key}] "
              + " ".join(f"{k}={v:.3f}" for k, v in cells[cell_key].items()
                         if isinstance(v, float) and k != "wall_s"),
              flush=True)
        save()

    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))

    for face in FACES:
        if not (mstats[f"{face}/train"]["ok"] and mstats[f"{face}/test"]["ok"]):
            print(f"[{key} {face}] SKIP (insufficient matched rows)")
            continue
        rtr, ytr = manifests[(face, "train")]
        rte, yte = manifests[(face, "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

        # ---- per-token FIRST (triage order auditable on disk) ----
        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{face}/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3), 3))
        run(f"{face}/tok_mlp", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3, hidden=512), 3))
        run(f"{face}/position_floor", lambda: summarize(
            fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, 3), 3))

        for T in sorted(set(AX_TS) | set(ORD_TS)):
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            if T in AX_TS:
                atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
                run(f"{face}/T{T}/actxmean_linear", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 3), 3))
                run(f"{face}/T{T}/actxmean_mlp", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 3, hidden=512), 3))
                fatr = actxmean_foreign(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T))
                fate = actxmean_foreign(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
                run(f"{face}/T{T}/actxmean_foreign_linear", lambda: summarize(
                    fit_probe(fatr, ytr_t, fate, yte_t, 3), 3))
                run(f"{face}/T{T}/actxmean_foreign_mlp", lambda: summarize(
                    fit_probe(fatr, ytr_t, fate, yte_t, 3, hidden=512), 3))
                del atr, ate, fatr, fate
            if T in ORD_TS:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"{face}/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3), 3))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"{face}/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{face}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 3), 3))
                ftr = foreign_context(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T)
                ).reshape(len(rtr), -1)
                fte = foreign_context(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                ).reshape(len(rte), -1)
                run(f"{face}/T{T}/win_foreign_linear", lambda: summarize(
                    fit_probe(ftr, ytr_t, fte, yte_t, 3), 3))
                if T in MLP_T:
                    run(f"{face}/T{T}/win_mlp", lambda: summarize(
                        fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                                  hidden=512), 3))
                    run(f"{face}/T{T}/win_shuf_mlp", lambda: summarize(
                        fit_probe(Str, ytr_t, Ste, yte_t, 3, hidden=512), 3))
                    run(f"{face}/T{T}/win_foreign_mlp", lambda: summarize(
                        fit_probe(ftr, ytr_t, fte, yte_t, 3, hidden=512), 3))
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"{face}/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(flat_tr, yn, flat_te, yte_t, 3), 3))
                    run(f"{face}/null_tok_linear", lambda: summarize(
                        fit_probe(Xtr_tok, yn, Xte_tok, yte_t, 3), 3))
                del flat_tr, flat_te, Str, Ste, ftr, fte
            del Wtr, Wte
        del Xtr_tok, Xte_tok

        # ---- within-document control (binding for lev; run for all) ----
        wd = f"{face}_wd"
        if not (mstats.get(f"{wd}/train", {}).get("ok")
                and mstats.get(f"{wd}/test", {}).get("ok")):
            print(f"[{key} {wd}] SKIP (insufficient within-doc rows; "
                  f"stats recorded)")
            continue
        rtr, ytr = manifests[(wd, "train")]
        rte, yte = manifests[(wd, "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{face}/wd/tok_linear", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
        run(f"{face}/wd/tok_mlp", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, hidden=512,
                      class_weight=True), 2))
        for T in sorted(set(WD_TS) | set(WD_ORD_TS)):
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            if T in WD_TS:
                atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
                run(f"{face}/wd/T{T}/actxmean_linear", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2,
                              class_weight=True), 2))
                run(f"{face}/wd/T{T}/actxmean_mlp", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2, hidden=512,
                              class_weight=True), 2))
                fatr = actxmean_foreign(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T))
                fate = actxmean_foreign(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
                run(f"{face}/wd/T{T}/actxmean_foreign_linear",
                    lambda: summarize(fit_probe(fatr, ytr_t, fate, yte_t, 2,
                                                class_weight=True), 2))
                del atr, ate, fatr, fate
            if T in WD_ORD_TS:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"{face}/wd/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 2,
                              class_weight=True), 2))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"{face}_wd/T{T}".encode())
                    % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{face}/wd/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 2,
                              class_weight=True), 2))
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
        del Xtr, Xte
    del acts
    save()
    print(f"[{key}] DONE -> {out_path}", flush=True)


def main():
    for k in (sys.argv[1:] or ["gpt2", "llama31_8b"]):
        screen(k)


if __name__ == "__main__":
    main()
