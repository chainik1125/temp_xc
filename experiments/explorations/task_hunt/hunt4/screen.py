"""Stage-1 screen — fourth-generation hunt faces on the REUSED
dialevel substrate (executes HUNT4_SCREEN_CARD.md; gen-4 directive
59ad15f38 scaled by c1c5c949e).

FACES (5 survivors of the 7-design slate; xret killed label-side by
the 0.8 anti-dup bar, rdens on its own Ward lane):
  xnov   cross-speaker adoption rate      floor: xfloor_rate + wnov rate
  tret   topic-return intensity           floor: wnov rate (shared, § 3)
  sdom   signed speaker dominance         floor: D/K_cur/K_oth window twin
  xtrend adoption-rate turn-trend         floor: xfloor slope + rate
  tretd  return DEPTH (mean log2 gap)     floor: wnov rate (shared, § 3)

Faces + PRE-COMPUTED per-T visible floors from
`../labels/hunt4_dailydialog_<tok>.npz` (builder labels/build_hunt4.py
— the § 3 evidence lines are these floors' AUCs). Token stream,
caches and screen layers are dialevel's verbatim; probe grid is the
hunt3 convention-of-record clone WITH the c1c5c949e extension:
**shuffle twins on the FULL ladder** — win/win_shuf linear at
T ∈ {4,8,16,32} (foreign kept at {16,32}, hunt3 parity), MLP triple
at T32, nulls at T16, wd arms BINDING with the same extended order
ladder.

Run: .venv/bin/python -m experiments.explorations.task_hunt.hunt4.screen [model ...]
Writes results/screen_<model>.json next to this file (resumable).
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
from experiments.explorations.task_hunt.dialevel.cache_acts import (
    CACHE_ROOT,
    TOK_TAG,
)
from experiments.explorations.task_hunt.dialevel.capacity_check import (
    FOREIGN_SEED,
    anchor_ctxmean,
    foreign_context,
)
from experiments.explorations.task_hunt.novelty.screen import (
    CAP,
    MIN_ROWS,
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
RES = HERE / "results"

FACES = ("xnov", "tret", "sdom", "xtrend", "tretd")
FLOOR_COLS = {
    "xnov": ("xfloor_rate", "floor_rate"),
    "tret": ("floor_rate",),
    "sdom": ("sdom_floor", "sdom_kcur", "sdom_koth"),
    "xtrend": ("xfloor_slope", "xfloor_rate"),
    "tretd": ("floor_rate",),
}
NULL_SEED = 99
AX_TS = [4, 8, 16, 32, 64]
ORD_TS = [4, 8, 16, 32]        # c1c5c949e: shuffle twin on the full ladder
FOREIGN_ORD_TS = [16, 32]      # foreign kept at hunt3 parity
ORD_MLP_T = [32]
NULL_T = 16
WD_TS = [16, 32, 64]
WD_ORD_TS = [4, 8, 16, 32]
WD_FOREIGN_TS = [16, 32]
WD_MIN_DOC_ROWS = 30


def build_rows(key: str):
    tag = TOK_TAG[key]
    zd = np.load(LABELS / f"dialevel_dailydialog_{tag}.npz")
    zf = np.load(LABELS / f"hunt4_dailydialog_{tag}.npz")
    c = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx, n_prefix = c["ids"], c["doc_idx"], int(c["n_prefix"])
    content = ids.shape[1] - n_prefix
    flat, off = zd["token_ids"], zd["doc_off"]
    doc_split, boundary = zd["doc_split"], zd["is_boundary"]

    lookup = _row_lookup(doc_idx)
    for (d, k), i in list(lookup.items())[:200]:
        s = off[d] + k * content
        assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
            f"flat/window mismatch at doc {d} chunk {k}"

    n_tok = flat.shape[0]
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    rows_flat, cpos_flat = _map_rows(doc_of, pos_of, lookup, content,
                                     n_prefix)
    base_elig = ((rows_flat >= 0) & (pos_of >= POS_MIN)
                 & (pos_of % content >= OFF_MIN) & (boundary == 0))

    out: dict = {}
    stats = {"n_prefix": n_prefix, "content": content,
             "n_cache_rows": int(ids.shape[0])}

    for face in FACES:
        md = zf[f"man_{face}_doc"]
        mp = zf[f"man_{face}_pos"]
        mc = zf[f"man_{face}_cls"]
        man_flat = off[md] + mp
        rows_all, cpos_all = _map_rows(md, mp, lookup, content, n_prefix)
        elig = (rows_all >= 0) & (mp >= POS_MIN) & (mp % content >= OFF_MIN)
        for split_name, flag in (("train", 0), ("test", 1)):
            m = elig & (doc_split[md] == flag)
            per, keep_r, keep_y, keep_f = {}, [], [], []
            for cls in (0, 1, 2):
                idx = np.flatnonzero(m & (mc == cls))
                rng = _seeded(f"hunt4/{face}/{key}/{split_name}/{cls}")
                if len(idx) > CAP[split_name]:
                    idx = rng.choice(idx, CAP[split_name], replace=False)
                per[cls] = len(idx)
                keep_r.append(_stack(rows_all, cpos_all, mp, idx))
                keep_y.append(np.full(len(idx), cls, dtype=np.int64))
                keep_f.append(man_flat[idx])
            stats[f"{face}/{split_name}"] = {
                "rows_per_class": per,
                "ok": bool(min(per.values()) >= MIN_ROWS)}
            out[(face, split_name)] = (np.concatenate(keep_r),
                                       np.concatenate(keep_y))
            out[(f"{face}_flat", split_name)] = np.concatenate(keep_f)

        # ---- within-dialogue control (BINDING, ops rule 7) ----
        val = zf[face].astype(np.float64)
        elig_wd = base_elig & np.isfinite(val) & (zf[f"{face}_bin"] >= 0)
        for split_name, flag in (("train", 0), ("test", 1)):
            m = elig_wd & (doc_split[doc_of] == flag)
            lo_idx, hi_idx, n_docs_used = [], [], 0
            for d in np.unique(doc_of[m]):
                sel = np.flatnonzero(m & (doc_of == d))
                v = val[sel]
                if len(sel) < WD_MIN_DOC_ROWS:
                    continue
                q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
                if not (q2 > q1):
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
                rng = _seeded(f"hunt4wd/{face}/{key}/{split_name}/{v_}")
                if len(idx) > n_take:
                    idx = rng.choice(idx, n_take, replace=False)
                keep_r.append(_stack(rows_flat, cpos_flat, pos_of, idx))
                keep_y.append(np.full(len(idx), v_, dtype=np.int64))
            stats[f"{face}_wd/{split_name}"] = {
                "ok": bool(n_take >= MIN_ROWS), "n_per_class": int(n_take),
                "n_docs": int(n_docs_used)}
            out[(f"{face}_wd", split_name)] = (np.concatenate(keep_r),
                                               np.concatenate(keep_y))
    return out, stats, zd, zf


def _pos_feats(r):
    cp = r[:, 1].astype(np.float32)
    dp = r[:, 2].astype(np.float32)
    f = np.stack([cp, cp ** 2 / 128.0, np.log2(1.0 + dp), dp / 1000.0], 1)
    return torch.from_numpy(f).to(torch.float16)


def _floor_feats(zf, flat_pos, T, face):
    """Precomputed window-visible floor features at the manifest rows
    (labels/build_hunt4.py; stored fp16 — the multi-column floors are
    STRICTLY STRONGER than the card's 1-D evidence lines, the
    conservative direction; nan_to_num is the sdom/tretd mass-guard
    belt: a floor that abstains scores no better than 0)."""
    cols = [zf[f"{c}_T{T}"][flat_pos].astype(np.float32)
            for c in FLOOR_COLS[face]]
    f = np.nan_to_num(np.stack(cols, 1), nan=0.0)
    return torch.from_numpy(f).to(torch.float16)


def screen(key: str):
    RES.mkdir(exist_ok=True)
    out_path = RES / f"screen_{key}.json"
    hs = SCREEN_HS[key]
    manifests, mstats, zd, zf = build_rows(key)
    done = json.loads(out_path.read_text()) if out_path.exists() else {
        "meta": {"model": key, "screen_hs": hs,
                 "card": "HUNT4_SCREEN_CARD.md (frozen)",
                 "ax_ts": AX_TS, "ord_ts": ORD_TS,
                 "foreign_ord_ts": FOREIGN_ORD_TS, "null_t": NULL_T,
                 "wd_ts": WD_TS, "wd_ord_ts": WD_ORD_TS,
                 "foreign_seed": FOREIGN_SEED,
                 "rows": mstats,
                 "reach": "turn ≈ 14.5–15.7 tok; all faces support 64 "
                          "tok ≈ 4 turns (xtrend 5-turn ≈ 75 tok); "
                          "tret/tretd events are out-of-window at EVERY "
                          "ladder T by construction (gap > 64)"},
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
        if not (mstats[f"{face}/train"]["ok"]
                and mstats[f"{face}/test"]["ok"]):
            print(f"[{key} {face}] SKIP (insufficient rows)")
            continue
        rtr, ytr = manifests[(face, "train")]
        rte, yte = manifests[(face, "test")]
        ftr = manifests[(f"{face}_flat", "train")]
        fte = manifests[(f"{face}_flat", "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

        Xtr_tok, Xte_tok = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{face}/tok_linear", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3), 3))
        run(f"{face}/tok_mlp", lambda: summarize(
            fit_probe(Xtr_tok, ytr_t, Xte_tok, yte_t, 3, hidden=512), 3))
        run(f"{face}/position_floor", lambda: summarize(
            fit_probe(_pos_feats(rtr), ytr_t, _pos_feats(rte), yte_t, 3),
            3))

        for T in AX_TS:
            run(f"{face}/T{T}/visible_evidence_floor", lambda: summarize(
                fit_probe(_floor_feats(zf, ftr, T, face), ytr_t,
                          _floor_feats(zf, fte, T, face), yte_t, 3), 3))
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
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
                    SHUF_SEED
                    + zlib.crc32(f"{face}/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{face}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 3), 3))
                if T in FOREIGN_ORD_TS:
                    fwtr = foreign_context(
                        Wtr, np.random.default_rng(FOREIGN_SEED + T)
                    ).reshape(len(rtr), -1)
                    fwte = foreign_context(
                        Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                    ).reshape(len(rte), -1)
                    run(f"{face}/T{T}/win_foreign_linear",
                        lambda: summarize(
                            fit_probe(fwtr, ytr_t, fwte, yte_t, 3), 3))
                    if T in ORD_MLP_T:
                        run(f"{face}/T{T}/win_mlp", lambda: summarize(
                            fit_probe(flat_tr, ytr_t, flat_te, yte_t, 3,
                                      hidden=512), 3))
                        run(f"{face}/T{T}/win_shuf_mlp", lambda: summarize(
                            fit_probe(Str, ytr_t, Ste, yte_t, 3,
                                      hidden=512), 3))
                        run(f"{face}/T{T}/win_foreign_mlp",
                            lambda: summarize(
                                fit_probe(fwtr, ytr_t, fwte, yte_t, 3,
                                          hidden=512), 3))
                    del fwtr, fwte
                if T == NULL_T:
                    nrng = np.random.default_rng(NULL_SEED)
                    yn = torch.from_numpy(nrng.permutation(ytr))
                    run(f"{face}/T{T}/null_win_linear", lambda: summarize(
                        fit_probe(flat_tr, yn, flat_te, yte_t, 3), 3))
                    run(f"{face}/null_tok_linear", lambda: summarize(
                        fit_probe(Xtr_tok, yn, Xte_tok, yte_t, 3), 3))
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
        del Xtr_tok, Xte_tok

        wd = f"{face}_wd"
        if not (mstats.get(f"{wd}/train", {}).get("ok")
                and mstats.get(f"{wd}/test", {}).get("ok")):
            print(f"[{key} {wd}] SKIP (insufficient within-dialogue rows "
                  f"— a SKIP here blocks any KEEP, card)")
            continue
        rtr, ytr = manifests[(wd, "train")]
        rte, yte = manifests[(wd, "test")]
        ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
        Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
        run(f"{wd}/tok_linear", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, class_weight=True), 2))
        run(f"{wd}/tok_mlp", lambda: summarize(
            fit_probe(Xtr, ytr_t, Xte, yte_t, 2, hidden=512,
                      class_weight=True), 2))
        for T in sorted(set(WD_TS) | set(WD_ORD_TS)):
            Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
            if T in WD_TS:
                atr, ate = anchor_ctxmean(Wtr), anchor_ctxmean(Wte)
                run(f"{wd}/T{T}/actxmean_linear", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2,
                              class_weight=True), 2))
                run(f"{wd}/T{T}/actxmean_mlp", lambda: summarize(
                    fit_probe(atr, ytr_t, ate, yte_t, 2, hidden=512,
                              class_weight=True), 2))
                fatr = actxmean_foreign(
                    Wtr, np.random.default_rng(FOREIGN_SEED + T))
                fate = actxmean_foreign(
                    Wte, np.random.default_rng(FOREIGN_SEED + T + 1))
                run(f"{wd}/T{T}/actxmean_foreign_linear", lambda: summarize(
                    fit_probe(fatr, ytr_t, fate, yte_t, 2,
                              class_weight=True), 2))
                del atr, ate, fatr, fate
            if T in WD_ORD_TS:
                flat_tr = Wtr.reshape(len(rtr), -1)
                flat_te = Wte.reshape(len(rte), -1)
                run(f"{wd}/T{T}/win_linear", lambda: summarize(
                    fit_probe(flat_tr, ytr_t, flat_te, yte_t, 2,
                              class_weight=True), 2))
                srng = np.random.default_rng(
                    SHUF_SEED + zlib.crc32(f"{wd}/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{wd}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 2,
                              class_weight=True), 2))
                if T in WD_FOREIGN_TS:
                    fwtr = foreign_context(
                        Wtr, np.random.default_rng(FOREIGN_SEED + T)
                    ).reshape(len(rtr), -1)
                    fwte = foreign_context(
                        Wte, np.random.default_rng(FOREIGN_SEED + T + 1)
                    ).reshape(len(rte), -1)
                    run(f"{wd}/T{T}/win_foreign_linear", lambda: summarize(
                        fit_probe(fwtr, ytr_t, fwte, yte_t, 2,
                                  class_weight=True), 2))
                    del fwtr, fwte
                del flat_tr, flat_te, Str, Ste
            del Wtr, Wte
        del Xtr, Xte
    del acts
    save()
    print(f"[{key}] DONE -> {out_path}", flush=True)


def main():
    for k in (sys.argv[1:] or ["gpt2", "gemma2_2b"]):
        screen(k)


if __name__ == "__main__":
    main()
